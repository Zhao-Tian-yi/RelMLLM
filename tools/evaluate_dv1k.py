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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
ARTICLES = {"a", "an", "the"}


def answer_type(question: str, reference: str | None = None) -> str:
    question = str(question).strip().lower()
    if re.match(r"^(how many|what number|what is the number|number of)\b", question):
        return "count"
    if reference and re.match(r"^\s*(yes|no)\b", str(reference), flags=re.IGNORECASE):
        return "yes_no"
    if re.match(
        r"^(is|are|was|were|do|does|did|can|could|has|have|had|will|would|"
        r"should|may|might)\b",
        question,
    ):
        return "yes_no"
    return "short_answer"


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"```(?:json)?|```", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = [NUMBER_WORDS.get(t, t) for t in text.split() if t not in ARTICLES]
    return " ".join(tokens)


def normalize_answer(text: str, kind: str) -> str | None:
    raw = str(text).strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            raw = str(payload.get("answer", payload.get("final_answer", ""))).strip()
    except (json.JSONDecodeError, TypeError):
        pass

    if kind == "yes_no":
        match = re.search(r"\b(yes|no)\b", raw, flags=re.IGNORECASE)
        return match.group(1).lower() if match else None
    if kind == "count":
        normalized = normalize_text(raw)
        match = re.search(r"\b(\d+)\b", normalized)
        return match.group(1) if match else None
    normalized = normalize_text(raw)
    return normalized or None


def score_answer(prediction: str | None, reference: str | None, kind: str) -> float:
    if prediction is None or reference is None:
        return 0.0
    if kind in {"yes_no", "count"}:
        return float(prediction == reference)
    if prediction == reference:
        return 1.0
    # DualVision short-answer references are often full sentences while models
    # return only the requested entity or attribute.
    if len(prediction) >= 2 and (
        f" {prediction} " in f" {reference} " or f" {reference} " in f" {prediction} "
    ):
        return 1.0
    return 0.0


def build_prompt(record: dict[str, Any], prompt_variant: str) -> list[dict[str, Any]]:
    image_paths = list(record.get("image_paths") or record.get("images"))
    images = [{"type": "image", "image": load_rgb(path)} for path in image_paths]
    if prompt_variant == "no_label":
        image_intro = "You are given two aligned images of the same scene."
    elif prompt_variant == "explicit_label":
        image_intro = "You are given two aligned images of the same scene: Image 1 is RGB/visible, and Image 2 is thermal infrared."
    else:
        image_intro = "You are given two aligned images of the same scene: first RGB/visible, second thermal infrared."
    kind = answer_type(record["question"], record.get("answer"))
    if kind == "yes_no":
        instruction = "Answer with exactly Yes or No."
    elif kind == "count":
        instruction = "Answer with exactly one integer."
    else:
        instruction = "Answer with one concise noun phrase. Do not explain."
    user_text = f"{image_intro}\n{instruction}\nQuestion: {record['question']}"
    return [
        {"role": "system", "content": [{"type": "text", "text": "Answer RGB-T visual questions concisely and exactly."}]},
        {"role": "user", "content": images + [{"type": "text", "text": user_text}]},
    ]


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
    out = {}
    for value in sorted({str(r.get(field)) for r in rows}):
        subset = [r for r in rows if str(r.get(field)) == value]
        out[value] = {"accuracy": mean(subset, "correct"), "valid_rate": mean(subset, "valid"), "n": len(subset)}
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary = {
        "accuracy": mean(rows, "correct"),
        "valid_rate": mean(rows, "valid"),
        "n": len(rows),
        "by_condition": grouped(rows, "condition"),
        "by_category": grouped(rows, "category"),
        "by_answer_type": grouped(rows, "answer_type"),
        "by_expected_modality": grouped(rows, "expected_modality"),
        "input_jsonl": args.input_jsonl,
        "prompt_variant": args.prompt_variant,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# DualVision JSONL VQA Eval Report\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    Path(str(path).replace(".md", ".json")).write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL-style MLLM on mixed-format DualVision JSONL.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--results_jsonl", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_time", type=float, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--prompt_variant", choices=["current", "no_label", "explicit_label"], default="current")
    args = parser.parse_args()

    apply_qwen3_vl_aligned_pair_rope_patch()
    records = load_jsonl(Path(args.input_jsonl))
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
    rows = load_jsonl(rows_path) if args.resume and rows_path.exists() else []
    done = {r.get("sample_id") for r in rows}
    for idx, record in enumerate(records, start=1):
        if args.resume and record["sample_id"] in done:
            continue
        response = generate_answer(model, processor, record, args.max_new_tokens, args.max_time, args.prompt_variant)
        kind = answer_type(record["question"], record.get("answer"))
        pred = normalize_answer(response, kind)
        gold = normalize_answer(record["answer"], kind)
        row = {
            **record,
            "response": response,
            "prediction": pred,
            "gold": gold,
            "answer_type": kind,
            "valid": float(pred is not None),
            "correct": score_answer(pred, gold, kind),
            "prompt_variant": args.prompt_variant,
        }
        rows.append(row)
        append_jsonl(rows_path, row)
        print(f"[VQA] {idx}/{len(records)} {record['sample_id']} pred={pred} gold={gold} correct={row['correct']:.0f}", flush=True)
        if len(rows) % 10 == 0:
            write_report(report, rows, args)
    write_report(report, rows, args)
    print(f"[VQA] wrote {report}")


if __name__ == "__main__":
    main()
