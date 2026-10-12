#!/usr/bin/env bash
set -euo pipefail

python inference.py \
  --model "${MODEL_PATH:-tyy1850819840/RelMLLM-8B}" \
  --task vqa \
  --rgb "$1" \
  --tir "$2" \
  --query "$3"

