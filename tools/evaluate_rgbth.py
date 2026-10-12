#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen3_aligned_rope_patch import apply_qwen3_vl_aligned_pair_rope_patch

DEFAULT_MODEL = "tyy1850819840/RelMLLM-8B"


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def find_image(root: Path, item_dir: Path, data_id: str, kind: str) -> Path:
    candidates = [
        item_dir / Path(data_id).name,
        root / data_id,
        item_dir / f"{kind}.jpg",
        item_dir / f"{kind}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {kind} image for {item_dir} data_id={data_id}")


def build_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for qpath in sorted(root.rglob("questions.json")):
        item_dir = qpath.parent
        data = json.loads(qpath.read_text(encoding="utf-8"))["data"]
        for section_idx, section in enumerate(data):
            data_type = section.get("data_type", "")
            data_id = section.get("data_id")
            if data_type == "RGB-Thermal Pair":
                rgb_id, thermal_id = data_id
                rgb_path = find_image(root, item_dir, rgb_id, "rgb")
                thermal_path = find_image(root, item_dir, thermal_id, "thermal")
                image_paths = [rgb_path, thermal_path]
            else:
                rgb_id = data_id if isinstance(data_id, str) else data_id[0]
                image_paths = [find_image(root, item_dir, rgb_id, "rgb")]
            for qa_idx, qa in enumerate(section.get("QAs", [])):
                sample_id = f"{qpath.parent.relative_to(root).as_posix()}__sec{section_idx:02d}__qa{qa_idx:02d}"
                records.append(
                    {
                        "sample_id": sample_id,
                        "question_file": str(qpath),
                        "data_source": section.get("data_source"),
                        "data_type": data_type,
                        "skill_type": section.get("skill_type"),
                        "image_paths": [str(p) for p in image_paths],
                        "question": qa["question"],
                        "answer": str(qa["answer"]).strip(),
                    }
                )
    return records


def build_prompt(record: dict[str, Any], prompt_variant: str) -> list[dict[str, Any]]:
    if record["data_type"] == "RGB-Thermal Pair":
        image_paths = list(record["image_paths"])
        if prompt_variant in {"swapped_correct", "swapped_wrong"}:
            image_paths = [image_paths[1], image_paths[0]]
        if prompt_variant == "no_label":
            image_intro = "You are given two aligned images of the same scene."
        elif prompt_variant == "explicit_label":
            image_intro = "You are given two aligned images of the same scene: Image 1 is RGB/visible, and Image 2 is thermal infrared."
        elif prompt_variant == "swapped_correct":
            image_intro = "You are given two aligned images of the same scene: Image 1 is thermal infrared, and Image 2 is RGB/visible."
        elif prompt_variant == "swapped_wrong":
            image_intro = "You are given two aligned images of the same scene: Image 1 is RGB/visible, and Image 2 is thermal infrared."
        else:
            image_intro = "You are given two aligned images: first RGB, second thermal infrared."
        images = [{"type": "image", "image": load_rgb(path)} for path in image_paths]
    else:
        image_intro = "You are given one image." if prompt_variant == "no_label" else "You are given one RGB image."
        images = [{"type": "image", "image": load_rgb(record["image_paths"][0])}]
    user_text = (
        f"{image_intro}\n"
        "Answer the question with exactly one word: Yes or No. Do not explain.\n"
        f"Question: {record['question']}"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": "Answer visual yes/no questions exactly."}]},
        {"role": "user", "content": images + [{"type": "text", "text": user_text}]},
    ]


def normalize_answer(text: str) -> str:
    text = text.strip()
    match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else "Invalid"


def generate_answer(model, processor, record: dict[str, Any], max_new_tokens: int, max_time: float | None, prompt_variant: str) -> str:
    inputs = processor.apply_chat_template(
        build_prompt(record, prompt_variant),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    if max_time and max_time > 0:
        kwargs["max_time"] = max_time
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    return processor.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows) if rows else 0.0


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for value in sorted({str(r.get(field)) for r in rows}):
        subset = [r for r in rows if str(r.get(field)) == value]
        out[value] = {
            "accuracy": mean(subset, "correct"),
            "valid_rate": mean(subset, "valid"),
            "n": len(subset),
        }
    return out


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    groups = grouped(rows, "data_type")
    group_macro = {}
    for group in ("Single RGB Image", "RGB-Thermal Pair"):
        subset = [row for row in rows if row["data_type"] == group]
        skills = grouped(subset, "skill_type")
        group_macro[group] = sum(float(value["accuracy"]) for value in skills.values()) / len(skills) if skills else None
    summary = {
        "accuracy": mean(rows, "correct"),
        "valid_rate": mean(rows, "valid"),
        "n": len(rows),
        "by_data_type": groups,
        "macro_qacc": sum(group_macro.values()) / 2 if all(value is not None for value in group_macro.values()) else None,
        "macro_qacc_by_data_type": group_macro,
        "macro_qacc_note": "Macro over skill within each data type, then equal average of both types; merge all shards before headline scoring",
        "by_source": grouped(rows, "data_source"),
        "by_skill_type": grouped(rows, "skill_type"),
    }
    text = "# RGB-Th-Bench VQA Eval Report\n\n```json\n"
    text += json.dumps(summary, ensure_ascii=False, indent=2)
    text += "\n```\n"
    path.write_text(text, encoding="utf-8")
    Path(str(path).replace(".md", ".json")).write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL-style MLLM on RGB-Th-Bench yes/no VQA.")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--results_jsonl", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_time", type=float, default=10)
    parser.add_argument("--write_every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--prompt_variant",
        choices=["current", "no_label", "explicit_label", "swapped_correct", "swapped_wrong"],
        default="current",
    )
    return parser.parse_args()


def main() -> None:
    apply_qwen3_vl_aligned_pair_rope_patch()
    args = parse_args()
    records = build_records(Path(args.dataset_root))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    records = records[args.shard_index::args.num_shards]
    if args.max_samples is not None:
        records = records[: args.max_samples]

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    report = Path(args.output_report)
    rows_path = Path(args.results_jsonl) if args.results_jsonl else Path(str(report).replace(".md", ".rows.jsonl"))
    if rows_path.exists() and not args.resume:
        raise FileExistsError("Predictions exist; use --resume or a new output path")
    rows = load_jsonl(rows_path) if args.resume else []
    done = {r.get("sample_id") for r in rows}
    for idx, record in enumerate(records, start=1):
        if args.resume and record["sample_id"] in done:
            print(f"[RGBThVQA] {idx}/{len(records)} skipped {record['sample_id']}", flush=True)
            continue
        response = generate_answer(model, processor, record, args.max_new_tokens, args.max_time, args.prompt_variant)
        pred = normalize_answer(response)
        gold = normalize_answer(record["answer"])
        row = {
            **record,
            "response": response,
            "prediction": pred,
            "gold": gold,
            "valid": float(pred in {"Yes", "No"}),
            "correct": float(pred == gold),
            "prompt_variant": args.prompt_variant,
        }
        rows.append(row)
        append_jsonl(rows_path, row)
        print(
            f"[RGBThVQA] {idx}/{len(records)} {record['sample_id']} pred={pred} gold={gold} correct={row['correct']:.0f}",
            flush=True,
        )
        if args.write_every > 0 and len(rows) % args.write_every == 0:
            write_report(report, rows)
    write_report(report, rows)
    print(f"[RGBThVQA] wrote {report}")


if __name__ == "__main__":
    main()
