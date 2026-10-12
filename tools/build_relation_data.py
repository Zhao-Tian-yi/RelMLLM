#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if sample_id in rows:
                    raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
                rows[sample_id] = row
    return rows


def reliability(iou: float) -> str:
    if iou >= 0.50:
        return "high"
    if iou >= 0.25:
        return "medium"
    return "low"


def derive_label(rgb: dict[str, Any], tir: dict[str, Any], both: dict[str, Any]) -> dict[str, Any]:
    rgb_iou = float(rgb.get("iou", 0.0) or 0.0)
    tir_iou = float(tir.get("iou", 0.0) or 0.0)
    both_iou = float(both.get("iou", 0.0) or 0.0)
    rgb_ok, tir_ok, both_ok = rgb_iou >= 0.50, tir_iou >= 0.50, both_iou >= 0.50
    rgb_confident_wrong = float(rgb.get("format_ok", 0.0) or 0.0) == 1.0 and rgb_iou < 0.10
    tir_confident_wrong = float(tir.get("format_ok", 0.0) or 0.0) == 1.0 and tir_iou < 0.10

    utilization_failure = False
    if rgb_ok and tir_ok:
        if abs(rgb_iou - tir_iou) <= 0.15:
            relation, dominant, rule = "consistent", "Both", "both_accurate_close"
        elif rgb_iou > tir_iou:
            relation, dominant, rule = "complementary", "RGB", "both_accurate_rgb_better"
        else:
            relation, dominant, rule = "complementary", "TIR", "both_accurate_tir_better"
        utilization_failure = not both_ok
    elif rgb_ok and not tir_ok:
        if tir_confident_wrong and not both_ok:
            relation, dominant, rule = "conflict", "RGB", "tir_wrong_drags_paired"
        else:
            relation, dominant, rule = "complementary", "RGB", "rgb_only_reliable"
    elif tir_ok and not rgb_ok:
        if rgb_confident_wrong and not both_ok:
            relation, dominant, rule = "conflict", "TIR", "rgb_wrong_drags_paired"
        else:
            relation, dominant, rule = "complementary", "TIR", "tir_only_reliable"
    elif both_ok:
        relation, dominant, rule = "complementary", "Both", "paired_synergy"
    else:
        relation, dominant, rule = "uncertain", "None", "all_unreliable"

    return {
        "relation": relation,
        "dominant_modality": dominant,
        "rgb_reliability": reliability(rgb_iou),
        "tir_reliability": reliability(tir_iou),
        "annotation_rule": rule,
        "utilization_failure": utilization_failure,
        "rgb_baseline_iou": rgb_iou,
        "tir_baseline_iou": tir_iou,
        "paired_baseline_iou": both_iou,
        "rgb_candidate_format_ok": float(rgb.get("format_ok", 0.0) or 0.0),
        "tir_candidate_format_ok": float(tir.get("format_ok", 0.0) or 0.0),
        "paired_candidate_format_ok": float(both.get("format_ok", 0.0) or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--rgb_jsonl", required=True)
    parser.add_argument("--tir_jsonl", required=True)
    parser.add_argument("--both_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a new output directory; existing relation data is not overwritten")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")

    bases = json.loads(Path(args.base_json).read_text(encoding="utf-8"))
    rgb = load_jsonl(Path(args.rgb_jsonl))
    tir = load_jsonl(Path(args.tir_jsonl))
    both = load_jsonl(Path(args.both_jsonl))
    natural = [
        row
        for row in bases
        if row.get("case_type") == "both_advantage"
        and row.get("metadata", {}).get("source_sample_id")
        and "controlled" not in str(row.get("sample_id", "")).lower()
    ]
    source_ids = [str(row["metadata"]["source_sample_id"]) for row in natural]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("natural candidate pool contains repeated source_sample_id values")
    common = [
        row
        for row in natural
        if str(row["sample_id"]) in rgb and str(row["sample_id"]) in tir and str(row["sample_id"]) in both
    ]
    if len(common) < args.num_samples:
        raise ValueError(f"only {len(common)} aligned natural candidates available")

    rng = random.Random(args.seed)
    rng.shuffle(common)
    selected = common[: args.num_samples]
    output_rows: list[dict[str, Any]] = []
    for base in selected:
        sample_id = str(base["sample_id"])
        label = derive_label(rgb[sample_id], tir[sample_id], both[sample_id])
        width, height = base["image_size"]
        output_rows.append(
            {
                "sample_id": str(base["metadata"]["source_sample_id"]),
                "sample_type": "groundbench_counterfactual_relation_v2",
                "rgb_path": base["rgb_path"],
                "tir_path": base["tir_path"],
                "query": base["query"],
                "bbox": [float(value) for value in base["bbox"]],
                "image_width": int(width),
                "image_height": int(height),
                **label,
                "metadata": {
                    "source": base.get("metadata", {}).get("source"),
                    "source_sample_id": base["metadata"]["source_sample_id"],
                    "candidate_sample_id": sample_id,
                    "relation_source": "qwen3_vl_rgb_tir_paired_counterfactual_iou",
                    "annotation_version": "groundbench_counterfactual_relation_v2_20260721",
                    "thresholds": {
                        "accurate_iou": 0.50,
                        "medium_iou": 0.25,
                        "confident_wrong_iou": 0.10,
                        "close_iou_margin": 0.15,
                    },
                },
            }
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "groundbench_counterfactual_relation_5k.json"
    json_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "groundbench_counterfactual_relation_5k.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "num_samples": len(output_rows),
        "seed": args.seed,
        "natural_candidate_pool": len(natural),
        "aligned_candidate_pool": len(common),
        "unique_source_sample_ids": len({row["sample_id"] for row in output_rows}),
        "relation": Counter(row["relation"] for row in output_rows),
        "dominant_modality": Counter(row["dominant_modality"] for row in output_rows),
        "rgb_reliability": Counter(row["rgb_reliability"] for row in output_rows),
        "tir_reliability": Counter(row["tir_reliability"] for row in output_rows),
        "annotation_rule": Counter(row["annotation_rule"] for row in output_rows),
        "utilization_failure": Counter(str(row["utilization_failure"]) for row in output_rows),
        "source": Counter(str(row["metadata"]["source"]) for row in output_rows),
        "controlled_or_degraded_rows": sum(
            "controlled" in json.dumps(row["metadata"]).lower() for row in output_rows
        ),
    }
    (out / "groundbench_counterfactual_relation_5k_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
