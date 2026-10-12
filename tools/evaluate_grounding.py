#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-code", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-cuda", action="store_true")
    return parser.parse_args()


def valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def normalize_modality(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = value.strip().lower().replace(" ", "")
    mapping = {
        "rgb": "RGB",
        "tir": "TIR",
        "ir": "TIR",
        "both": "Both",
        "rgb|tir|both": "Both",
        "rgb/tir/both": "Both",
        "rgb,tir,both": "Both",
        "none": "None",
    }
    return mapping.get(compact)


def denormalize(bbox: list[float] | None, width: int, height: int) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        x2 * width / 1000.0,
        y2 * height / 1000.0,
    ]


def iou_xyxy(a: list[float] | None, b: list[float]) -> float:
    if a is None:
        return 0.0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    args = parse_args()
    if Path(args.output_jsonl).exists() and not args.resume:
        raise FileExistsError("Predictions exist; use --resume or a new output path")
    sys.path.insert(0, args.release_code)
    from inference import build_messages, extract_json, load_image

    examples = json.load(Path(args.eval_json).open(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0} if args.force_cuda else "auto",
        trust_remote_code=True,
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    device = next(model.parameters()).device

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    completed_ids: set[str] = set()
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(row)
                completed_ids.add(str(row["sample_id"]))
    else:
        output.write_text("", encoding="utf-8")

    pending = [
        example for example in examples
        if str(example["sample_id"]) not in completed_ids
    ]
    for batch_start in range(0, len(pending), args.batch_size):
        batch_examples = pending[batch_start:batch_start + args.batch_size]
        batch_images: list[tuple[Image.Image, Image.Image]] = []
        batch_messages = []
        for example in batch_examples:
            rgb = load_image(example["images"][0])
            tir = load_image(example["images"][1])
            batch_images.append((rgb, tir))
            batch_messages.append(
                build_messages("grounding", "paper", example["query"], rgb, tir)
            )
        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        responses = processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        for example, (rgb, _), response in zip(
            batch_examples, batch_images, responses, strict=True
        ):
            response = response.strip()
            payload = extract_json(response)
            raw_bbox = valid_bbox(payload.get("bbox")) if payload else None
            pred_bbox = denormalize(raw_bbox, rgb.width, rgb.height)
            target_bbox = [float(value) for value in example["target_bbox"]]
            iou = iou_xyxy(pred_bbox, target_bbox)
            pred_modality = None
            if payload:
                pred_modality = normalize_modality(payload.get("modality"))
                if pred_modality is None:
                    pred_modality = normalize_modality(payload.get("dominant_modality"))
            row = {
                "sample_id": example["sample_id"],
                "image_name": example.get("image_name"),
                "query": example.get("query"),
                "target_bbox": target_bbox,
                "expected_modality": example.get("expected_modality"),
                "prompt_variant": "relation_state",
                "response": response,
                "parsed_json": payload,
                "pred_bbox_raw": raw_bbox,
                "pred_bbox": pred_bbox,
                "pred_bbox_format": "qwen_1000" if pred_bbox else None,
                "pred_modality": pred_modality,
                "format_ok": float(payload is not None and pred_bbox is not None and pred_modality is not None),
                "iou": iou,
                "iou@0.25": float(iou >= 0.25),
                "iou@0.5": float(iou >= 0.5),
                "iou@0.75": float(iou >= 0.75),
                "modality_correct": float(pred_modality == example.get("expected_modality")),
            }
            rows.append(row)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done = len(rows)
        recent = rows[-len(batch_examples):]
        print(
            f"[Eval] {done}/{len(examples)} "
            f"batch_acc@0.5={mean(recent, 'iou@0.5'):.4f} "
            f"format={mean(recent, 'format_ok'):.4f}",
            flush=True,
        )

    summary = {
        "num_samples": len(rows),
        "mean_iou": mean(rows, "iou"),
        "acc_iou@0.25": mean(rows, "iou@0.25"),
        "acc_iou@0.5": mean(rows, "iou@0.5"),
        "acc_iou@0.75": mean(rows, "iou@0.75"),
        "format_ok": mean(rows, "format_ok"),
        "modality_acc": mean(rows, "modality_correct"),
        "prompt_variant": "relation_state",
        "merged_model_only": True,
        "batch_size": args.batch_size,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else None
        ),
    }
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
