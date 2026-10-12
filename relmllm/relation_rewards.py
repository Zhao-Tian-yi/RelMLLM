#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any


VALID_RELATIONS = {"consistent", "complementary", "conflict", "uncertain"}
VALID_RELIABILITY = {"high", "medium", "low"}
VALID_DOMINANT = {"RGB", "TIR", "Both", "None"}


def find_json_substring(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_relation_payload(text_or_payload: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if text_or_payload is None:
        return None
    if isinstance(text_or_payload, dict):
        return text_or_payload
    blob = find_json_substring(str(text_or_payload))
    if blob is None:
        return None
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_bbox(payload: dict[str, Any] | None) -> list[float] | None:
    if payload is None:
        return None
    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        bbox = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def compute_iou_xyxy(pred: list[float] | None, target: list[float] | None) -> float:
    if pred is None or target is None:
        return 0.0
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = [float(v) for v in target]
    inter_x1 = max(px1, tx1)
    inter_y1 = max(py1, ty1)
    inter_x2 = min(px2, tx2)
    inter_y2 = min(py2, ty2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = pred_area + target_area - inter
    return float(inter / union) if union > 0 else 0.0


def scale_bbox_from_1000(pred: list[float] | None, image_width: float | None, image_height: float | None) -> list[float] | None:
    if pred is None or not image_width or not image_height:
        return None
    return [
        pred[0] * float(image_width) / 1000.0,
        pred[1] * float(image_height) / 1000.0,
        pred[2] * float(image_width) / 1000.0,
        pred[3] * float(image_height) / 1000.0,
    ]


def _valid_evidence(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0 and len(value.strip()) <= 300


def compute_format_reward(output: str | dict[str, Any]) -> float:
    payload = parse_relation_payload(output)
    if payload is None:
        return 0.0
    score = 0.0
    if parse_bbox(payload) is not None:
        score += 0.25
    if payload.get("rgb_reliability") in VALID_RELIABILITY:
        score += 0.15
    if payload.get("tir_reliability") in VALID_RELIABILITY:
        score += 0.15
    if payload.get("relation") in VALID_RELATIONS:
        score += 0.15
    if payload.get("dominant_modality") in VALID_DOMINANT:
        score += 0.15
    if _valid_evidence(payload.get("evidence")):
        score += 0.15
    return min(score, 1.0)


def compute_iou_reward(
    output: str | dict[str, Any],
    target_bbox: list[float],
    image_width: float | None = None,
    image_height: float | None = None,
) -> float:
    payload = parse_relation_payload(output)
    pred = parse_bbox(payload)
    raw_iou = compute_iou_xyxy(pred, target_bbox)
    rescaled_iou = compute_iou_xyxy(scale_bbox_from_1000(pred, image_width, image_height), target_bbox)
    return max(raw_iou, rescaled_iou)


def compute_relation_reward(output: str | dict[str, Any], gold: dict[str, Any] | None = None) -> float:
    payload = parse_relation_payload(output)
    if payload is None or payload.get("relation") not in VALID_RELATIONS:
        return 0.0
    if gold and gold.get("relation") in VALID_RELATIONS:
        return 1.0 if payload.get("relation") == gold.get("relation") else 0.0
    return 0.5


def compute_reliability_reward(output: str | dict[str, Any], gold: dict[str, Any] | None = None) -> float:
    payload = parse_relation_payload(output)
    if payload is None:
        return 0.0
    rgb = payload.get("rgb_reliability")
    tir = payload.get("tir_reliability")
    if rgb not in VALID_RELIABILITY or tir not in VALID_RELIABILITY:
        return 0.0
    if gold and gold.get("rgb_reliability") in VALID_RELIABILITY and gold.get("tir_reliability") in VALID_RELIABILITY:
        return 0.5 * float(rgb == gold.get("rgb_reliability")) + 0.5 * float(tir == gold.get("tir_reliability"))
    return 0.5


def compute_decision_consistency_reward(output: str | dict[str, Any], gold: dict[str, Any] | None = None) -> float:
    payload = parse_relation_payload(output)
    if payload is None:
        return 0.0
    relation = payload.get("relation")
    rgb = payload.get("rgb_reliability")
    tir = payload.get("tir_reliability")
    dominant = payload.get("dominant_modality")
    if relation not in VALID_RELATIONS or rgb not in VALID_RELIABILITY or tir not in VALID_RELIABILITY or dominant not in VALID_DOMINANT:
        return 0.0

    if gold and gold.get("dominant_modality") in VALID_DOMINANT:
        dominant_score = 1.0 if dominant == gold.get("dominant_modality") else 0.0
    else:
        dominant_score = 1.0

    rule_score = 0.0
    if relation == "consistent":
        rule_score = float(rgb == "high" and tir == "high" and dominant == "Both")
    elif relation == "complementary":
        if dominant == "RGB":
            rule_score = float(rgb in {"high", "medium"} and tir in {"medium", "low"} and rgb != "low")
        elif dominant == "TIR":
            rule_score = float(tir in {"high", "medium"} and rgb in {"medium", "low"} and tir != "low")
        elif dominant == "Both":
            rule_score = float(rgb != "low" or tir != "low")
        else:
            rule_score = 0.0
    elif relation == "conflict":
        rule_score = float(dominant in {"RGB", "TIR"})
    elif relation == "uncertain":
        rule_score = float(rgb == "low" and tir == "low" and dominant == "None")
    return 0.7 * rule_score + 0.3 * dominant_score


def compute_total_relation_reward(
    output: str | dict[str, Any],
    target_bbox: list[float],
    gold: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    image_width: float | None = None,
    image_height: float | None = None,
) -> dict[str, float]:
    weights = weights or {
        "format": 0.20,
        "iou": 0.35,
        "relation": 0.15,
        "reliability": 0.15,
        "decision_consistency": 0.15,
    }
    parts = {
        "format": compute_format_reward(output),
        "iou": compute_iou_reward(output, target_bbox, image_width=image_width, image_height=image_height),
        "relation": compute_relation_reward(output, gold),
        "reliability": compute_reliability_reward(output, gold),
        "decision_consistency": compute_decision_consistency_reward(output, gold),
    }
    total = sum(weights.get(k, 0.0) * v for k, v in parts.items())
    parts["total"] = float(total)
    return parts
