#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor
from relmllm.evidence_rewards import compute_hgrpo_modality_evidence_reward
from relmllm.relation_rewards import compute_iou_xyxy, parse_bbox, parse_relation_payload, scale_bbox_from_1000
from tools.qwen3_aligned_rope_patch import apply_qwen3_vl_aligned_pair_rope_patch

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
VANILLA_TASK_ONLY = False


def load_json(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def build_prompt(item: dict[str, Any]) -> list[dict[str, Any]]:
    width = item.get("image_width") or item.get("metadata", {}).get("image_width")
    height = item.get("image_height") or item.get("metadata", {}).get("image_height")
    if VANILLA_TASK_ONLY:
        user = (
            "Use both aligned RGB and TIR. Ground the query. Coordinates are original image pixels.\n"
            f"Query: {item['query']}\n"
            f"Image size: {width}x{height}.\n"
            "Return strict JSON only: "
            '{"bbox":[x1,y1,x2,y2],"evidence":"brief visual evidence"}'
        )
    else:
        user = (
        "Use both aligned RGB and TIR. Ground the query. Generate evidence steps and relation state before the final bbox. Coordinates are original image pixels.\n"
        f"Query: {item['query']}\n"
        f"Image size: {width}x{height}.\n"
        "Use numeric bbox values. Keep each sub_answer <= 8 words. Close the JSON object.\n"
        '{"steps":[{"type":"rgb_evidence","modality":"RGB","bbox":[x1,y1,x2,y2] or null,"reliability":"high|medium|low","sub_answer":"short"},'
        '{"type":"tir_evidence","modality":"TIR","bbox":[x1,y1,x2,y2] or null,"reliability":"high|medium|low","sub_answer":"short"},'
        '{"type":"relation_check","modality":"RGB|TIR|Both|None","relation":"consistent|complementary|conflict|uncertain","sub_answer":"short"}],'
        '"rgb_reliability":"high|medium|low","tir_reliability":"high|medium|low","relation":"consistent|complementary|conflict|uncertain","dominant_modality":"RGB|TIR|Both|None","bbox":[x1,y1,x2,y2],"evidence":"short"}'
        )
    return [
        {"role": "system", "content": "Return strict JSON only."},
        {"role": "user", "content": user},
    ]

class BBoxRelativeDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = load_json(path)
        if not self.rows:
            raise ValueError("RL data must not be empty")
        root = Path(path).resolve().parent
        ids = set()
        for row in self.rows:
            sample_id = str(row["sample_id"])
            if sample_id in ids:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            ids.add(sample_id)
            for key, choices in (("relation", RELATION_VALUES),
                                 ("rgb_reliability", RELIABILITY_VALUES),
                                 ("tir_reliability", RELIABILITY_VALUES),
                                 ("dominant_modality", DOMINANT_VALUES)):
                if row.get(key) not in choices:
                    raise ValueError(f"{sample_id}: missing/invalid {key}; relation labels are required")
            for key in ("rgb_path", "tir_path"):
                image_path = Path(row[key])
                if not image_path.is_absolute():
                    image_path = root / image_path
                with Image.open(image_path) as image:
                    size = image.size
                    image.verify()
                if size != (row["image_width"], row["image_height"]):
                    raise ValueError(f"{sample_id}: image dimensions do not match metadata")
                row[key] = str(image_path)
            box = row["bbox"]
            if (len(box) != 4 or not all(math.isfinite(float(v)) for v in box)
                    or not 0 <= box[0] < box[2] <= row["image_width"]
                    or not 0 <= box[1] < box[3] <= row["image_height"]):
                raise ValueError(f"{sample_id}: bbox must be original-image pixel xyxy")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.rows[idx]
        width = item.get("image_width") or item.get("metadata", {}).get("image_width")
        height = item.get("image_height") or item.get("metadata", {}).get("image_height")
        return {
            "prompt": build_prompt(item),
            "images": [item["rgb_path"], item["tir_path"]],
            "num_images": 2,
            "sample_id": item.get("sample_id"),
            "bbox": [float(v) for v in item["bbox"]],
            "image_width": width,
            "image_height": height,
            "rgb_baseline_iou": item.get("rgb_baseline_iou"),
            "gold_relation": item.get("relation"),
            "gold_rgb_reliability": item.get("rgb_reliability"),
            "gold_tir_reliability": item.get("tir_reliability"),
            "gold_dominant_modality": item.get("dominant_modality"),
            "sample_type": item.get("sample_type") or item.get("metadata", {}).get("sample_type") or item.get("metadata", {}).get("source"),
        }


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for message in completion:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(completion)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def final_iou(output: str | dict[str, Any], target: list[float], width: float | None, height: float | None) -> float:
    payload = parse_relation_payload(output)
    bbox = parse_bbox(payload)
    return max(compute_iou_xyxy(bbox, target), compute_iou_xyxy(scale_bbox_from_1000(bbox, width, height), target))



def best_parsed_bbox(output: str | dict[str, Any], target: list[float], width: float | None, height: float | None) -> list[float] | None:
    payload = parse_relation_payload(output)
    bbox = parse_bbox(payload)
    if bbox is None:
        return None
    scaled = scale_bbox_from_1000(bbox, width, height)
    if compute_iou_xyxy(scaled, target) > compute_iou_xyxy(bbox, target):
        return scaled
    return bbox


def valid_box_reward(pred: list[float] | None, width: float | None, height: float | None) -> float:
    if pred is None or len(pred) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in pred]
    if x2 <= x1 or y2 <= y1:
        return 0.0
    if width is not None and height is not None:
        if x2 < 0 or y2 < 0 or x1 > float(width) or y1 > float(height):
            return 0.25
        area = (x2 - x1) * (y2 - y1)
        image_area = max(float(width) * float(height), 1.0)
        if area <= 1.0 or area > 0.95 * image_area:
            return 0.5
    return 1.0


def center_reward(pred: list[float] | None, target: list[float], width: float | None, height: float | None) -> float:
    if pred is None or len(pred) != 4:
        return 0.0
    px = (float(pred[0]) + float(pred[2])) / 2.0
    py = (float(pred[1]) + float(pred[3])) / 2.0
    tx = (float(target[0]) + float(target[2])) / 2.0
    ty = (float(target[1]) + float(target[3])) / 2.0
    if width is None or height is None:
        diag = math.sqrt(max((float(target[2]) - float(target[0])) ** 2 + (float(target[3]) - float(target[1])) ** 2, 1.0))
    else:
        diag = math.sqrt(max(float(width) ** 2 + float(height) ** 2, 1.0))
    dist = math.sqrt((px - tx) ** 2 + (py - ty) ** 2)
    return math.exp(-4.0 * dist / diag)


def area_reward(pred: list[float] | None, target: list[float]) -> float:
    if pred is None or len(pred) != 4:
        return 0.0
    pred_area = max(0.0, float(pred[2]) - float(pred[0])) * max(0.0, float(pred[3]) - float(pred[1]))
    target_area = max(0.0, float(target[2]) - float(target[0])) * max(0.0, float(target[3]) - float(target[1]))
    if pred_area <= 0.0 or target_area <= 0.0:
        return 0.0
    ratio = pred_area / target_area
    return math.exp(-abs(math.log(max(ratio, 1e-6))))


def geometry_reward(output: str | dict[str, Any], target: list[float], width: float | None, height: float | None) -> tuple[float, float]:
    pred = best_parsed_bbox(output, target, width, height)
    iou = max(compute_iou_xyxy(pred, target), 0.0) if pred is not None else 0.0
    acc50 = 1.0 if iou >= 0.50 else 0.0
    acc75 = 1.0 if iou >= 0.75 else 0.0
    valid = valid_box_reward(pred, width, height)
    center = center_reward(pred, target, width, height)
    area = area_reward(pred, target)
    reward = 0.45 * iou + 0.20 * acc50 + 0.10 * acc75 + 0.15 * center + 0.05 * area + 0.05 * valid
    return clip01(reward), iou


RELIABILITY_VALUES = {"high", "medium", "low"}
RELATION_VALUES = {"consistent", "complementary", "conflict", "uncertain"}
DOMINANT_VALUES = {"RGB", "TIR", "Both", "None"}
REQUIRED_SCHEMA_KEYS = {"bbox", "rgb_reliability", "tir_reliability", "relation", "dominant_modality", "evidence"}


def format_reward(output: str | dict[str, Any]) -> float:
    payload = parse_relation_payload(output)
    bbox = parse_bbox(payload)
    if payload is None or bbox is None:
        return 0.0
    if VANILLA_TASK_ONLY:
        evidence = payload.get("evidence", "")
        return 1.0 if isinstance(evidence, str) and 0 < len(evidence.strip()) <= 96 else 0.8
    keys = set(payload.keys())
    score = 0.35
    if REQUIRED_SCHEMA_KEYS.issubset(keys):
        score += 0.20
    if str(payload.get("rgb_reliability", "")).strip() in RELIABILITY_VALUES:
        score += 0.10
    if str(payload.get("tir_reliability", "")).strip() in RELIABILITY_VALUES:
        score += 0.10
    if str(payload.get("relation", "")).strip() in RELATION_VALUES:
        score += 0.10
    if str(payload.get("dominant_modality", "")).strip() in DOMINANT_VALUES:
        score += 0.10
    evidence = payload.get("evidence", "")
    if isinstance(evidence, str) and 0 < len(evidence.strip()) <= 96:
        score += 0.05
    return clip01(score)


def relation_label_reward(
    output: str | dict[str, Any],
    gold_relation: str | None = None,
    gold_rgb_reliability: str | None = None,
    gold_tir_reliability: str | None = None,
    gold_dominant_modality: str | None = None,
    rgb_baseline_iou: float | None = None,
    iou: float = 0.0,
) -> float:
    payload = parse_relation_payload(output)
    if payload is None:
        return 0.0
    relation = str(payload.get("relation", "")).strip()
    rgb_rel = str(payload.get("rgb_reliability", "")).strip()
    tir_rel = str(payload.get("tir_reliability", "")).strip()
    dominant = str(payload.get("dominant_modality", "")).strip()
    score = 0.0
    total = 0.0
    if gold_relation in RELATION_VALUES:
        total += 0.35
        score += 0.35 * float(relation == gold_relation)
    if gold_rgb_reliability in RELIABILITY_VALUES:
        total += 0.20
        score += 0.20 * float(rgb_rel == gold_rgb_reliability)
    if gold_tir_reliability in RELIABILITY_VALUES:
        total += 0.20
        score += 0.20 * float(tir_rel == gold_tir_reliability)
    if gold_dominant_modality in DOMINANT_VALUES:
        total += 0.25
        score += 0.25 * float(dominant == gold_dominant_modality)
    if total > 0:
        return score / total

    # Fallback weak consistency when no relation labels exist.
    weak = 0.0
    checks = 0.0
    if rgb_baseline_iou is not None:
        rgb_iou = float(rgb_baseline_iou)
        if iou >= rgb_iou + 0.04:
            checks += 1.0
            weak += float(relation in {"consistent", "complementary"})
            checks += 1.0
            weak += float(dominant in {"Both", "TIR"})
        elif rgb_iou >= 0.75 and abs(iou - rgb_iou) <= 0.04:
            checks += 1.0
            weak += float(relation in {"consistent", "complementary"})
            checks += 1.0
            weak += float(dominant in {"RGB", "Both"})
        elif iou + 0.08 < rgb_iou:
            checks += 1.0
            weak += float(relation in {"conflict", "uncertain", "complementary"})
            checks += 1.0
            weak += float(dominant in {"RGB", "None"})
    return weak / checks if checks > 0 else 0.5


def relation_reward_weight(sample_type: str | None, has_gold: bool) -> float:
    if not RELATION_REWARD_ENABLED:
        return 0.0
    text = str(sample_type or "").lower()
    if "relation_seed" in text or "manual" in text:
        return 0.12
    if has_gold:
        return RELATION_STATE_WEIGHT
    return 0.03

def relative_reward(iou: float, rgb_baseline_iou: float | None, margin: float = 0.04) -> float:
    if rgb_baseline_iou is None:
        return 0.5
    return clip01((iou - float(rgb_baseline_iou) + margin) / (2.0 * margin))


REWARD_WEIGHTS = {"format": 0.15, "iou": 0.50, "relative": 0.35}
REWARD_MODE = "relative"
RELATIVE_MARGIN = 0.04
RGB_LOW_THRESHOLD = 0.45
RGB_HIGH_THRESHOLD = 0.75
HGRPO_EVIDENCE_WEIGHT = 0.15
RELATION_REWARD_ENABLED = True
RELATION_STATE_WEIGHT = 0.25


def sample_reward_weights(rgb_baseline_iou: float | None) -> dict[str, float]:
    if REWARD_MODE != "gated":
        return REWARD_WEIGHTS
    if rgb_baseline_iou is None:
        return {"format": 0.10, "iou": 0.60, "relative": 0.30}
    rgb_iou = float(rgb_baseline_iou)
    if rgb_iou >= RGB_HIGH_THRESHOLD:
        return {"format": 0.10, "iou": 0.80, "relative": 0.10}
    if rgb_iou >= RGB_LOW_THRESHOLD:
        return {"format": 0.10, "iou": 0.45, "relative": 0.45}
    return {"format": 0.10, "iou": 0.35, "relative": 0.55}


def bbox_relative_reward(
    prompts: list[Any],
    completions: list[Any],
    bbox: list[list[float]],
    image_width: list[float] | None = None,
    image_height: list[float] | None = None,
    rgb_baseline_iou: list[float | None] | None = None,
    gold_relation: list[str | None] | None = None,
    gold_rgb_reliability: list[str | None] | None = None,
    gold_tir_reliability: list[str | None] | None = None,
    gold_dominant_modality: list[str | None] | None = None,
    sample_type: list[str | None] | None = None,
    **_: Any,
) -> list[float]:
    image_width = image_width or [None] * len(completions)
    image_height = image_height or [None] * len(completions)
    rgb_baseline_iou = rgb_baseline_iou or [None] * len(completions)
    gold_relation = gold_relation or [None] * len(completions)
    gold_rgb_reliability = gold_rgb_reliability or [None] * len(completions)
    gold_tir_reliability = gold_tir_reliability or [None] * len(completions)
    gold_dominant_modality = gold_dominant_modality or [None] * len(completions)
    sample_type = sample_type or [None] * len(completions)
    rewards = []
    iterator = zip(completions, bbox, image_width, image_height, rgb_baseline_iou, gold_relation, gold_rgb_reliability, gold_tir_reliability, gold_dominant_modality, sample_type, strict=True)
    for out, target, width, height, rgb_iou, gold_rel, gold_rgb, gold_tir, gold_dom, stype in iterator:
        text = completion_to_text(out)
        geo, iou = geometry_reward(text, target, width, height)
        fmt = format_reward(text)
        rel = relative_reward(iou, rgb_iou, margin=RELATIVE_MARGIN)
        weights = sample_reward_weights(rgb_iou)
        rel_score = relation_label_reward(text, gold_rel, gold_rgb, gold_tir, gold_dom, rgb_iou, iou)
        hgrpo = compute_hgrpo_modality_evidence_reward(
            text,
            target,
            gold_relation=gold_rel,
            gold_rgb_reliability=gold_rgb,
            gold_tir_reliability=gold_tir,
            gold_dominant_modality=gold_dom,
            image_width=width,
            image_height=height,
        )["total"]
        has_gold = any(v in RELATION_VALUES or v in RELIABILITY_VALUES or v in DOMINANT_VALUES for v in [gold_rel, gold_rgb, gold_tir, gold_dom])
        r_weight = relation_reward_weight(stype, has_gold)
        grounding_reward = weights["format"] * fmt + weights["iou"] * geo + weights["relative"] * rel
        # Relation state is optimized, but bbox grounding remains the dominant objective.
        state_reward = (1.0 - r_weight) * grounding_reward + r_weight * rel_score
        rewards.append(float((1.0 - HGRPO_EVIDENCE_WEIGHT) * state_reward + HGRPO_EVIDENCE_WEIGHT * hgrpo))
    return rewards


def dry_run(args: argparse.Namespace) -> None:
    rows = load_json(args.train_json)[:8]
    good = [
        json.dumps(
            {
                "steps": [
                    {"type": "rgb_evidence", "modality": "RGB", "bbox": [float(v) for v in row["bbox"]], "reliability": "high", "sub_answer": "RGB target clear"},
                    {"type": "tir_evidence", "modality": "TIR", "bbox": [float(v) for v in row["bbox"]], "reliability": "medium", "sub_answer": "TIR supports target"},
                    {"type": "relation_check", "modality": "Both", "relation": "complementary", "sub_answer": "modalities agree mostly"},
                ],
                "bbox": [float(v) for v in row["bbox"]],
                "rgb_reliability": "high",
                "tir_reliability": "medium",
                "relation": "complementary",
                "dominant_modality": "Both",
                "evidence": "RGB is clearer while TIR provides auxiliary cues.",
            }
        )
        for row in rows
    ]
    rewards = bbox_relative_reward(
        prompts=[None] * len(rows),
        completions=good,
        bbox=[[float(v) for v in row["bbox"]] for row in rows],
        image_width=[row.get("image_width") or row.get("metadata", {}).get("image_width") for row in rows],
        image_height=[row.get("image_height") or row.get("metadata", {}).get("image_height") for row in rows],
        rgb_baseline_iou=[row.get("rgb_baseline_iou") for row in rows],
        gold_relation=[row.get("relation") for row in rows],
        gold_rgb_reliability=[row.get("rgb_reliability") for row in rows],
        gold_tir_reliability=[row.get("tir_reliability") for row in rows],
        gold_dominant_modality=[row.get("dominant_modality") for row in rows],
        sample_type=[row.get("sample_type") or row.get("metadata", {}).get("source") for row in rows],
    )
    bad = bbox_relative_reward(prompts=[None], completions=["not json"], bbox=[[float(v) for v in rows[0]["bbox"]]], rgb_baseline_iou=[rows[0].get("rgb_baseline_iou")])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"good_rewards": rewards, "mean_good_reward": sum(rewards) / len(rewards), "bad_reward": bad[0]}
    (out / "bbox_relative_grpo_reward_dry_run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper relation-guided RL; defaults to sequence-level GSPO.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter to continue GRPO from.")
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=-1, help="Positive values override the one-epoch schedule")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--image_min_pixels", type=int, default=200704)
    parser.add_argument("--image_max_pixels", type=int, default=200704)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--dry_run_rewards", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--format_weight", type=float, default=0.15)
    parser.add_argument("--iou_weight", type=float, default=0.50)
    parser.add_argument("--relative_weight", type=float, default=0.35)
    parser.add_argument("--reward_mode", choices=["relative", "gated"], default="relative")
    parser.add_argument("--relative_margin", type=float, default=0.04)
    parser.add_argument("--rgb_low_threshold", type=float, default=0.45)
    parser.add_argument("--rgb_high_threshold", type=float, default=0.75)
    parser.add_argument("--hgrpo_evidence_weight", type=float, default=0.15)
    parser.add_argument("--disable_relation_reward", action="store_true")
    parser.add_argument("--vanilla_task_only", action="store_true")
    parser.add_argument("--relation_state_weight", type=float, default=0.25)
    parser.add_argument("--loss_type", choices=["grpo", "dapo"], default="grpo")
    parser.add_argument(
        "--importance_sampling_level",
        choices=["token", "sequence"],
        default="sequence",
        help="Use sequence-level geometric-mean importance ratios for GSPO.",
    )
    parser.add_argument("--scale_rewards", default="group", help='GRPO reward scaling: "group", "batch", "none", or "false".')
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError("Output directory is not empty; use a new directory or explicitly resume")
    for value in (args.hgrpo_evidence_weight, args.relation_state_weight):
        if not 0 <= value <= 1:
            raise ValueError("Reward mixing coefficients must be in [0,1]")
    train_data = BBoxRelativeDataset(args.train_json)
    eval_data = BBoxRelativeDataset(args.eval_json)
    if {r['sample_id'] for r in train_data.rows} & {r['sample_id'] for r in eval_data.rows}:
        raise ValueError("Train and evaluation sample IDs overlap")
    REWARD_WEIGHTS["format"] = float(args.format_weight)
    REWARD_WEIGHTS["iou"] = float(args.iou_weight)
    REWARD_WEIGHTS["relative"] = float(args.relative_weight)
    global REWARD_MODE, RELATIVE_MARGIN, RGB_LOW_THRESHOLD, RGB_HIGH_THRESHOLD, HGRPO_EVIDENCE_WEIGHT
    global RELATION_REWARD_ENABLED, RELATION_STATE_WEIGHT, VANILLA_TASK_ONLY
    REWARD_MODE = str(args.reward_mode)
    RELATIVE_MARGIN = float(args.relative_margin)
    RGB_LOW_THRESHOLD = float(args.rgb_low_threshold)
    RGB_HIGH_THRESHOLD = float(args.rgb_high_threshold)
    HGRPO_EVIDENCE_WEIGHT = float(args.hgrpo_evidence_weight)
    RELATION_REWARD_ENABLED = not bool(args.disable_relation_reward)
    RELATION_STATE_WEIGHT = float(args.relation_state_weight)
    VANILLA_TASK_ONLY = bool(args.vanilla_task_only)
    if VANILLA_TASK_ONLY:
        RELATION_REWARD_ENABLED = False
        HGRPO_EVIDENCE_WEIGHT = 0.0
    print(
        "[BBoxRelativeGRPO] "
        f"reward_mode={REWARD_MODE} reward_weights={REWARD_WEIGHTS} "
        f"relative_margin={RELATIVE_MARGIN} rgb_low={RGB_LOW_THRESHOLD} rgb_high={RGB_HIGH_THRESHOLD} "
        f"hgrpo_evidence_weight={HGRPO_EVIDENCE_WEIGHT} "
        f"relation_reward_enabled={RELATION_REWARD_ENABLED} relation_state_weight={RELATION_STATE_WEIGHT} "
        f"vanilla_task_only={VANILLA_TASK_ONLY}"
    )
    if args.dry_run_rewards:
        dry_run(args)
        return
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    import inspect
    set_seed(args.seed)

    class PairedGRPOTrainer(GRPOTrainer):
        def _get_per_token_logps_and_entropies(self, *positional, **kwargs):
            method = super()._get_per_token_logps_and_entropies
            bound = inspect.signature(method).bind(*positional, **kwargs)
            if all(bound.arguments.get(k) is not None for k in ("pixel_values", "image_grid_thw", "num_images")):
                # Match the historical two-image logprob chunking without editing TRL.
                bound.arguments["batch_size"] = 1
            return method(**bound.arguments)

    apply_qwen3_vl_aligned_pair_rope_patch()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        processor.image_processor.size = {"shortest_edge": int(args.image_min_pixels), "longest_edge": int(args.image_max_pixels)}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if args.adapter_path:
        print(f"[BBoxRelativeGRPO] loading trainable adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", target_modules="all-linear", task_type="CAUSAL_LM")
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        data_seed=args.seed,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type=args.loss_type,
        importance_sampling_level=args.importance_sampling_level,
        scale_rewards=False if str(args.scale_rewards).lower() in {"false", "none", "0"} else args.scale_rewards,
        bf16=args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        save_steps=20,
        logging_steps=1,
        report_to="none",
        dataloader_num_workers=0,
    )
    out.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        (out / "training_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    trainer = PairedGRPOTrainer(
        model=model,
        reward_funcs=bbox_relative_reward,
        args=grpo_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
        peft_config=peft_config,
    )
    resume_from_checkpoint: str | bool | None = args.resume_from_checkpoint
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "true":
        resume_from_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(Path(args.output_dir) / "final_model")
    processor.save_pretrained(Path(args.output_dir) / "final_model")


if __name__ == "__main__":
    main()






