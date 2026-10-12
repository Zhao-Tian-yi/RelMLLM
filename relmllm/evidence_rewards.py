#!/usr/bin/env python3
from __future__ import annotations

import itertools
import re
from typing import Any

from .relation_rewards import (
    VALID_DOMINANT,
    VALID_RELATIONS,
    VALID_RELIABILITY,
    compute_iou_xyxy,
    parse_bbox,
    parse_relation_payload,
    scale_bbox_from_1000,
)


STEP_TYPES = {"rgb_evidence", "tir_evidence", "relation_check"}
MODALITIES = {"RGB", "TIR", "Both", "None"}


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_step_bbox(step: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(step, dict):
        return None
    bbox = step.get("bbox") or step.get("evidence_bbox")
    if bbox is None:
        return None
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        box = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    step_type = str(step.get("type") or step.get("step_type") or "").strip()
    modality = str(step.get("modality") or "").strip()
    relation = str(step.get("relation") or "").strip()
    reliability = str(step.get("reliability") or "").strip()
    text = step.get("sub_answer") or step.get("answer") or step.get("evidence") or ""
    return {
        "type": step_type,
        "modality": modality,
        "relation": relation,
        "reliability": reliability,
        "bbox": parse_step_bbox(step),
        "text": str(text),
    }


def extract_pred_steps(output: str | dict[str, Any]) -> list[dict[str, Any]]:
    payload = parse_relation_payload(output)
    if payload is None:
        return []
    raw_steps = payload.get("steps")
    if isinstance(raw_steps, list):
        steps = [normalize_step(s) for s in raw_steps if isinstance(s, dict)]
        if steps:
            return steps

    # Backward-compatible fallback for the previous flat evidence schema.
    steps: list[dict[str, Any]] = []
    if payload.get("rgb_bbox") is not None or payload.get("rgb_reliability") is not None:
        steps.append(
            normalize_step(
                {
                    "type": "rgb_evidence",
                    "modality": "RGB",
                    "bbox": payload.get("rgb_bbox"),
                    "reliability": payload.get("rgb_reliability"),
                    "evidence": payload.get("evidence", ""),
                }
            )
        )
    if payload.get("tir_bbox") is not None or payload.get("tir_reliability") is not None:
        steps.append(
            normalize_step(
                {
                    "type": "tir_evidence",
                    "modality": "TIR",
                    "bbox": payload.get("tir_bbox"),
                    "reliability": payload.get("tir_reliability"),
                    "evidence": payload.get("evidence", ""),
                }
            )
        )
    if payload.get("relation") is not None or payload.get("dominant_modality") is not None:
        steps.append(
            normalize_step(
                {
                    "type": "relation_check",
                    "modality": payload.get("dominant_modality"),
                    "relation": payload.get("relation"),
                    "evidence": payload.get("evidence", ""),
                }
            )
        )
    return steps


def build_gold_steps(
    target_bbox: list[float],
    gold_relation: str | None = None,
    gold_rgb_reliability: str | None = None,
    gold_tir_reliability: str | None = None,
    gold_dominant_modality: str | None = None,
) -> list[dict[str, Any]]:
    rgb_box = target_bbox if gold_rgb_reliability in {"high", "medium"} else None
    tir_box = target_bbox if gold_tir_reliability in {"high", "medium"} else None
    return [
        normalize_step(
            {
                "type": "rgb_evidence",
                "modality": "RGB",
                "bbox": rgb_box,
                "reliability": gold_rgb_reliability,
                "evidence": "RGB target evidence",
            }
        ),
        normalize_step(
            {
                "type": "tir_evidence",
                "modality": "TIR",
                "bbox": tir_box,
                "reliability": gold_tir_reliability,
                "evidence": "TIR target evidence",
            }
        ),
        normalize_step(
            {
                "type": "relation_check",
                "modality": gold_dominant_modality,
                "relation": gold_relation,
                "evidence": "RGB TIR relation",
            }
        ),
    ]


def token_overlap(a: str, b: str) -> float:
    toks_a = set(re.findall(r"[A-Za-z0-9_]+", a.lower()))
    toks_b = set(re.findall(r"[A-Za-z0-9_]+", b.lower()))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def step_pair_score(
    pred: dict[str, Any],
    gold: dict[str, Any],
    image_width: float | None = None,
    image_height: float | None = None,
) -> float:
    score = 0.0
    score += 0.20 * float(pred.get("type") == gold.get("type") and pred.get("type") in STEP_TYPES)
    score += 0.15 * float(pred.get("modality") == gold.get("modality") and pred.get("modality") in MODALITIES)

    pred_box = pred.get("bbox")
    gold_box = gold.get("bbox")
    if gold_box is None:
        bbox_score = 1.0 if pred_box is None else 0.0
    else:
        bbox_score = max(
            compute_iou_xyxy(pred_box, gold_box),
            compute_iou_xyxy(scale_bbox_from_1000(pred_box, image_width, image_height), gold_box),
        )
    score += 0.30 * bbox_score

    gold_rel = gold.get("reliability")
    pred_rel = pred.get("reliability")
    if gold_rel in VALID_RELIABILITY:
        score += 0.15 * float(pred_rel == gold_rel)
    else:
        score += 0.15 * float(pred_rel in VALID_RELIABILITY or pred_rel == "")

    gold_relation = gold.get("relation")
    pred_relation = pred.get("relation")
    if gold_relation in VALID_RELATIONS:
        score += 0.15 * float(pred_relation == gold_relation)
    else:
        score += 0.15 * float(pred_relation in VALID_RELATIONS or pred_relation == "")

    score += 0.05 * token_overlap(str(pred.get("text", "")), str(gold.get("text", "")))
    return clip01(score)


def hungarian_max_score(
    pred_steps: list[dict[str, Any]],
    gold_steps: list[dict[str, Any]],
    image_width: float | None = None,
    image_height: float | None = None,
) -> float:
    if not pred_steps or not gold_steps:
        return 0.0
    n_pred = min(len(pred_steps), 6)
    n_gold = min(len(gold_steps), 6)
    pred_steps = pred_steps[:n_pred]
    gold_steps = gold_steps[:n_gold]

    if n_pred >= n_gold:
        best = 0.0
        for pred_indices in itertools.permutations(range(n_pred), n_gold):
            total = 0.0
            for gold_idx, pred_idx in enumerate(pred_indices):
                total += step_pair_score(pred_steps[pred_idx], gold_steps[gold_idx], image_width, image_height)
            best = max(best, total / n_gold)
        return best

    best = 0.0
    for gold_indices in itertools.permutations(range(n_gold), n_pred):
        total = 0.0
        for pred_idx, gold_idx in enumerate(gold_indices):
            total += step_pair_score(pred_steps[pred_idx], gold_steps[gold_idx], image_width, image_height)
        best = max(best, total / n_gold)
    return best


def evidence_steps_format_reward(output: str | dict[str, Any]) -> float:
    payload = parse_relation_payload(output)
    if payload is None:
        return 0.0
    steps = extract_pred_steps(payload)
    if not steps:
        return 0.0
    score = 0.25 * float(isinstance(payload.get("steps"), list))
    score += 0.20 * float(parse_bbox(payload) is not None)
    seen = {s.get("type") for s in steps}
    score += 0.30 * (len(seen & STEP_TYPES) / 3.0)
    valid_modalities = sum(1 for s in steps if s.get("modality") in MODALITIES)
    score += 0.15 * (valid_modalities / max(len(steps), 1))
    concise = sum(1 for s in steps if 0 < len(str(s.get("text", "")).strip()) <= 120)
    score += 0.10 * (concise / max(len(steps), 1))
    return clip01(score)


def final_evidence_consistency_reward(
    output: str | dict[str, Any],
    target_bbox: list[float],
    matched_step_score: float,
    image_width: float | None = None,
    image_height: float | None = None,
) -> float:
    payload = parse_relation_payload(output)
    if payload is None:
        return 0.0
    pred = parse_bbox(payload)
    final_iou = max(
        compute_iou_xyxy(pred, target_bbox),
        compute_iou_xyxy(scale_bbox_from_1000(pred, image_width, image_height), target_bbox),
    )
    return clip01(final_iou * (0.50 + 0.50 * matched_step_score))


def compute_hgrpo_modality_evidence_reward(
    output: str | dict[str, Any],
    target_bbox: list[float],
    gold_relation: str | None = None,
    gold_rgb_reliability: str | None = None,
    gold_tir_reliability: str | None = None,
    gold_dominant_modality: str | None = None,
    image_width: float | None = None,
    image_height: float | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = weights or {"format": 0.20, "hungarian": 0.50, "final_consistency": 0.30}
    pred_steps = extract_pred_steps(output)
    gold_steps = build_gold_steps(
        target_bbox,
        gold_relation=gold_relation,
        gold_rgb_reliability=gold_rgb_reliability,
        gold_tir_reliability=gold_tir_reliability,
        gold_dominant_modality=gold_dominant_modality,
    )
    hungarian = hungarian_max_score(pred_steps, gold_steps, image_width=image_width, image_height=image_height)
    parts = {
        "format": evidence_steps_format_reward(output),
        "hungarian": hungarian,
        "final_consistency": final_evidence_consistency_reward(
            output,
            target_bbox,
            matched_step_score=hungarian,
            image_width=image_width,
            image_height=image_height,
        ),
    }
    parts["total"] = float(sum(weights.get(k, 0.0) * v for k, v in parts.items()))
    return parts
