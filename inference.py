#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from relmllm.aligned_rope import apply_aligned_rgbt_rope_patch
from relmllm.prompts import (
    build_grounding_prompt,
    build_vqa_direct_prompt,
    build_vqa_relation_prompt,
)


def load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    starts = [match.start() for match in re.finditer(r"\{", text)]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : index + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        break
    return None


def normalized_to_pixels(bbox: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [
        x1 * width / 1000.0,
        y1 * height / 1000.0,
        x2 * width / 1000.0,
        y2 * height / 1000.0,
    ]


def build_messages(
    task: str,
    prompt_mode: str,
    query: str,
    rgb: Image.Image,
    tir: Image.Image,
) -> list[dict[str, Any]]:
    if task == "grounding":
        system, user = build_grounding_prompt(query, rgb.width, rgb.height)
    elif prompt_mode == "relation-state":
        system, user = build_vqa_relation_prompt(query)
    else:
        system, user = build_vqa_direct_prompt(query)
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb},
                {"type": "image", "image": tir},
                {"type": "text", "text": user},
            ],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RelMLLM on an aligned RGB-T pair.")
    parser.add_argument("--model", required=True, help="Merged model directory or Hub ID.")
    parser.add_argument("--task", choices=["grounding", "vqa"], default="grounding")
    parser.add_argument(
        "--prompt-mode",
        choices=["paper", "relation-state"],
        default="paper",
        help=(
            "paper: relation-state grounding and direct Yes/No VQA, matching the "
            "reported evaluations; relation-state: structured output for both tasks."
        ),
    )
    parser.add_argument("--rgb", required=True, help="Path to the RGB image.")
    parser.add_argument("--tir", required=True, help="Path to the aligned TIR image.")
    parser.add_argument("--query", required=True, help="Referring expression or yes/no question.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--aligned-rope",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "auto reproduces the paper protocol (off for grounding, on for VQA); "
            "on/off forces one setting."
        ),
    )
    parser.add_argument("--pixel-bbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_aligned_rope = args.aligned_rope == "on" or (
        args.aligned_rope == "auto" and args.task == "vqa"
    )
    if use_aligned_rope:
        apply_aligned_rgbt_rope_patch()

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    rgb = load_image(args.rgb)
    tir = load_image(args.tir)
    messages = build_messages(args.task, args.prompt_mode, args.query, rgb, tir)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
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
    response = processor.decode(
        generated[0][prompt_length:], skip_special_tokens=True
    ).strip()
    payload = extract_json(response)

    normalized_answer = None
    if args.task == "vqa":
        if payload is not None:
            answer = str(payload.get("answer", "")).strip().lower()
            normalized_answer = answer.capitalize() if answer in {"yes", "no"} else None
        else:
            match = re.search(r"\b(yes|no)\b", response, flags=re.IGNORECASE)
            normalized_answer = match.group(1).capitalize() if match else None
    output: dict[str, Any] = {
        "task": args.task,
        "prompt_mode": args.prompt_mode,
        "aligned_rope": use_aligned_rope,
        "response": response,
        "parsed": payload,
    }
    if args.task == "vqa":
        output["answer"] = normalized_answer
    if args.pixel_bbox and args.task == "grounding" and payload is not None:
        output["bbox_pixels"] = normalized_to_pixels(
            payload.get("bbox"), rgb.width, rgb.height
        )
        output["image_size"] = [rgb.width, rgb.height]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
