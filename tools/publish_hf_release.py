#!/usr/bin/env python3
"""Stage verified recovered weights and publish only the explicit allowlist."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi
from verify_weights import verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--token-file", required=True)
    p.add_argument("--proxy", required=True)
    args = p.parse_args()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = args.proxy
    api = HfApi(token=Path(args.token_file).read_text().strip())
    identity = api.whoami()
    if identity["name"] != "tyy1850819840":
        raise ValueError("Unexpected upload account")
    source, stage, code = map(Path, (args.source, args.stage, args.code))
    stage.mkdir(parents=True, exist_ok=True)
    for size in ("4B", "8B"):
        name = "RelMLLM-" + size
        original = source / name
        report_path = stage / (name + "_SOURCE_AUDIT.json")
        report = verify(original)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        target = stage / name
        target.mkdir(exist_ok=True)
        allowed = {"config.json", "generation_config.json", "LICENSE", "NOTICE",
                   "chat_template.jinja", "model.safetensors.index.json",
                   "processor_config.json", "tokenizer_config.json", "tokenizer.json"}
        allowed.update(json.loads((original / "model.safetensors.index.json").read_text())["weight_map"].values())
        for filename in sorted(allowed):
            dest = target / filename
            if not dest.exists():
                dest.symlink_to(original / filename)
        (target / "README.md").write_bytes((code / "model_cards" / (name + ".md")).read_bytes())
        (target / "ORIGINAL_README.md").write_bytes((original / "README.md").read_bytes())
        (target / "RECOVERED_SHA256SUMS").write_bytes((original / "SHA256SUMS").read_bytes())
        (target / "SOURCE_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
        files = sorted(path for path in target.iterdir() if path.name != "SHA256SUMS")
        lines = []
        for path in files:
            if path.name in report["files"] and path.name != "README.md":
                digest = report["files"][path.name]
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(digest + "  " + path.name)
        (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        repo = "tyy1850819840/" + name
        try:
            api.model_info(repo)
        except Exception as error:
            from huggingface_hub.errors import RepositoryNotFoundError
            if not isinstance(error, RepositoryNotFoundError):
                raise
            api.create_repo(repo_id=repo, private=False)
        else:
            raise RuntimeError("Existing model repository: inspect before overwriting " + repo)
        print("Uploading " + repo, flush=True)
        commit = api.upload_folder(repo_id=repo, folder_path=str(target),
                                   commit_message="Release verified recovered merged " + name)
        info = api.model_info(repo, revision=commit.oid, files_metadata=True)
        remote = {item.rfilename: item for item in info.siblings}
        for path in files:
            if path.name.endswith(".safetensors"):
                lfs = remote[path.name].lfs
                if lfs is None or lfs.sha256 != report["files"][path.name]:
                    raise ValueError("Uploaded shard hash mismatch: " + path.name)
        result = {"repo": repo, "revision": commit.oid, "shard_hashes": "passed",
                  "source_audit": report, "remote_files": sorted(remote)}
        (stage / (name + "_PUBLISHED.json")).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"repo": repo, "revision": commit.oid, "verified": True}), flush=True)


if __name__ == "__main__":
    main()
