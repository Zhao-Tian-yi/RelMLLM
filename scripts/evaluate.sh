#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: bash scripts/evaluate.sh {groundbench|degraded|rgbth|dv1k} MODEL DATA OUTPUT_DIR" >&2
  exit 2
fi
benchmark=$1
model=$2
data=$3
output=$4
case "$benchmark" in
  groundbench|degraded|dv1k) [[ -f "$data" ]] || { echo "Missing input file: $data" >&2; exit 2; } ;;
  rgbth) [[ -d "$data" ]] || { echo "Missing dataset directory: $data" >&2; exit 2; } ;;
  *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;;
esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data=$(realpath -- "$data")
if [[ -e "$model" ]]; then model=$(realpath -- "$model"); fi
if [[ -e "$output" ]]; then echo "Output already exists; choose a new directory: $output" >&2; exit 2; fi
mkdir -p -- "$output"
output=$(realpath -- "$output")
cd -- "$root"
case "$benchmark" in
  groundbench|degraded)
    python tools/evaluate_grounding.py --model "$model" --eval-json "$data" \
      --batch-size 1 --max-new-tokens 256 --force-cuda \
      --output-jsonl "$output/predictions.jsonl" --summary-json "$output/summary.json" ;;
  rgbth)
    python tools/evaluate_rgbth.py --model_name_or_path "$model" --dataset_root "$data" \
      --bf16 --max_new_tokens 8 --results_jsonl "$output/predictions.jsonl" \
      --output_report "$output/report.md" ;;
  dv1k)
    python tools/evaluate_dv1k.py --model_name_or_path "$model" --input_jsonl "$data" \
      --bf16 --max_new_tokens 32 --results_jsonl "$output/predictions.jsonl" \
      --output_report "$output/report.md" ;;
esac
