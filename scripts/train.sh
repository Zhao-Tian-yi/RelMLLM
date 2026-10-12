#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "Usage: bash scripts/train.sh BASE_MODEL SFT_TRAIN_JSONL SFT_VAL_JSONL RL_TRAIN_JSON RL_VAL_JSON OUTPUT_DIR" >&2
  exit 2
fi
model=$1
sft_train=$2
sft_val=$3
rl_train=$4
rl_val=$5
output=$6
for input in "$sft_train" "$sft_val" "$rl_train" "$rl_val"; do
  [[ -f "$input" ]] || { echo "Missing input file: $input" >&2; exit 2; }
done
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sft_train=$(realpath -- "$sft_train")
sft_val=$(realpath -- "$sft_val")
rl_train=$(realpath -- "$rl_train")
rl_val=$(realpath -- "$rl_val")
if [[ -e "$model" ]]; then model=$(realpath -- "$model"); fi
if [[ -e "$output" ]]; then echo "Output already exists; choose a new directory: $output" >&2; exit 2; fi
mkdir -p -- "$output"
output=$(realpath -- "$output")
cd -- "$root"
python train.py --train_json "$rl_train" --eval_json "$rl_val" \
  --output_dir "$output/reward_check" --dry_run_rewards
if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi
torchrun --standalone --nproc_per_node=2 train_sft.py \
  --model_name_or_path "$model" --train_jsonl "$sft_train" --eval_jsonl "$sft_val" \
  --output_dir "$output/sft" --bf16 --gradient_checkpointing \
  --max_steps 600 --learning_rate 3e-6 --gradient_accumulation_steps 4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --seed 42 \
  --save_steps 100 --eval_steps 100
torchrun --standalone --nproc_per_node=2 train.py \
  --model_name_or_path "$model" --adapter_path "$output/sft/final_model" \
  --train_json "$rl_train" --eval_json "$rl_val" --output_dir "$output/gspo" \
  --num_train_epochs 1 --learning_rate 1e-6 --seed 42 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_generations 4 --max_completion_length 1024 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --relation_state_weight 0.25 --hgrpo_evidence_weight 0.15 \
  --loss_type grpo --importance_sampling_level sequence --scale_rewards group --bf16
python tools/merge_adapter.py --model "$model" --adapter "$output/gspo/final_model" \
  --output "$output/merged"
