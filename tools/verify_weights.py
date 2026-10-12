#!/usr/bin/env python3
"""Verify file hashes and safetensors index/header integrity without a GPU."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    hashes = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        name = Path(name).as_posix()
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path outside model root")
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise ValueError("Hash mismatch: " + name)
        hashes[name] = digest
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = {}
    dtypes = {}
    for name in sorted(set(index["weight_map"].values())):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name not in hashes:
            raise ValueError("Unverified shard: " + name)
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 100_000_000:
                raise ValueError("Unexpectedly large safetensors header")
            header = json.loads(handle.read(size))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in tensors or "lora_" in key.lower():
                raise ValueError("Duplicate or adapter tensor: " + key)
            start, end = value["data_offsets"]
            if not 0 <= start <= end <= path.stat().st_size - size - 8:
                raise ValueError("Invalid tensor offsets: " + key)
            tensors[key] = name
            dtype = value["dtype"]
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
    if tensors != index["weight_map"]:
        raise ValueError("Shard keys disagree with index")
    return {"status": "passed", "files": hashes, "tensor_count": len(tensors),
            "tensor_dtypes": dtypes, "merged_no_lora_keys": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = verify(args.model_directory)
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        with Path(args.report).open("x") as handle:
            handle.write(text)
    print(text)
