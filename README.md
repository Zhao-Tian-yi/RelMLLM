# RelMLLM

Code for **Beyond Paired Inputs: Learning to Leverage RGB-T Cross-Modal
Relations in Multimodal Large Language Models**.

A minimal repository for paired RGB/TIR inference, benchmark evaluation, and
relation-guided reinforcement learning (GSPO), with optional SFT warm-up.
Inputs are RGB first, TIR second. Evaluation grounding uses
single-image `[x1,y1,x2,y2]` coordinates normalized to 0..1000.

## Install and Download

Use Python 3.10 and a CUDA-compatible PyTorch installation.

```bash
pip install -r requirements.txt
hf download tyy1850819840/RelMLLM-8B --revision recovered-20261012 --local-dir models/RelMLLM-8B
```

Weights: [RelMLLM-4B](https://huggingface.co/tyy1850819840/RelMLLM-4B) and
[RelMLLM-8B](https://huggingface.co/tyy1850819840/RelMLLM-8B).
Both are merged BF16 checkpoints with the fixed `recovered-20261012` release
tag. For 4B, replace both occurrences of `RelMLLM-8B` in the download command.

## Inference

```bash
python inference.py --model models/RelMLLM-8B --task grounding \
  --rgb /path/to/rgb.jpg --tir /path/to/tir.jpg --query "the person crossing the street"
```

Add `--pixel-bbox` for original-image coordinates. `--task vqa` supports generic
yes/no questions; use the benchmark scripts below for reported VQA scores.

## Datasets

| Dataset | Download | Evaluation input |
|---|---|---|
| RGBT-GroundBench | [Official repository](https://github.com/crazyxiaoxi/RGBT-GroundBench), [images and annotations](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset) | JSON manifest, 10,124 queries |
| RGB-Th-Bench | [Paper](https://arxiv.org/abs/2503.19654), [author-linked archive](https://drive.google.com/file/d/1aNaKYpI0lirhDUm34eq6PChA0pmjoXUD/view?usp=sharing) | Extracted `questions.json` files, 1,856 queries |
| DV-1K-Balanced | [DualVision annotations](https://github.com/abrarmajeedi/DualVision/tree/main/data/v1), [LLVIP](https://bupt-ai-cz.github.io/LLVIP/), [HDRT access](https://www.sciencedirect.com/science/article/abs/pii/S1566253525001824) | Custom mixed-answer JSONL, 1,000 queries |
| Degraded-1K | GroundBench-derived custom cache; no standalone download yet | Grounding JSON manifest, 1,000 queries |

See [DATASETS.md](DATASETS.md) for direct archive links, split definitions and
availability limits. Download third-party data under its original license.
The exact GroundBench converter, DV-1K selector and Degraded-1K corruption/cache
release remain outstanding; arbitrary substitutes do not reproduce the table.

### Grounding Manifest

Prepare a JSON list using your local image paths and pixel-coordinate GT:

```json
[{"sample_id":"example-1","images":["/data/rgb.jpg","/data/tir.jpg"],
  "query":"the person","target_bbox":[10,20,100,200],"expected_modality":"Both"}]
```

### Evaluation Commands

```bash
python tools/evaluate_grounding.py --model models/RelMLLM-8B \
  --eval-json /data/groundbench.json --batch-size 1 --max-new-tokens 256 \
  --output-jsonl outputs/groundbench.jsonl --summary-json outputs/groundbench_summary.json

python tools/evaluate_rgbth.py --model_name_or_path models/RelMLLM-8B \
  --dataset_root /data/RGB-Th-Bench --bf16 --max_new_tokens 8 \
  --results_jsonl outputs/rgbth.jsonl --output_report outputs/rgbth.md

python tools/evaluate_dv1k.py --model_name_or_path models/RelMLLM-8B \
  --input_jsonl /data/dv1k.jsonl --bf16 --max_new_tokens 32 \
  --results_jsonl outputs/dv1k.jsonl --output_report outputs/dv1k.md
```

Use the grounding command with the degraded manifest for Degraded-1K. Existing
prediction files are not overwritten; use a new output path or explicitly resume.

All benchmarks use BF16, batch 1 and greedy decoding. Grounding uses native
positions; both VQA evaluators use paired M-RoPE and a 10-second generation time
limit, which can depend on hardware. Grounding extracts embedded/fenced JSON;
legacy `format_ok` checks box ordering, not strict geometric validity. RGB-Th
QAcc averages skills within each data type, then averages the two data types.
DV-1K includes counts and uses the reference answer to choose the answer-type
prompt; this archived convention is not a GT-blind deployment procedure.

### Released Checkpoint Results

Complete archived results, percentages:

| Model | GroundBench Acc@0.5 | RGB-Th macro QAcc | Degraded-1K Acc@0.5 | DV-1K Acc |
|---|---:|---:|---:|---:|
| 4B recovered | 72.6788 | 75.2481 | 54.20 | 79.30 |
| 8B recovered | 73.1628 | 75.6745 | 59.80 | 78.40 |
| 4B paper | 72.3627 | 75.46 | 48.70 | 79.30 |
| 8B paper | 73.3011 | 75.61 | 61.10 | 78.30 |

These author-approved recovered weights are not an exact reproduction of every
paper number. The Degraded-1K differences are +5.50 / -1.30 points; their cause
has not been established. Custom input availability also limits independent
reproduction of these archived scores.

## Relation-Guided RL

`train.py` is adapted from the AAAI supplementary `train_relation_gspo.py`, not
from later coordinate-binding experiments. The final optimizer is **GSPO**:
TRL's `GRPOTrainer` with `importance_sampling_level=sequence`, `loss_type=grpo`,
and group reward scaling. A class name containing GRPO does not imply token-level
GRPO. [TRL documentation](https://huggingface.co/docs/trl/v0.25.1/en/paper_index)
explains this distinction.

The reward follows the paper's hierarchy:

```text
R = (1 - lambda_e) * ((1 - lambda_r) * R_task + lambda_r * R_rel)
    + lambda_e * R_cons
R_cons = 0.20 * R_step + 0.50 * R_match + 0.30 * R_dec
R_dec = IoU * (0.5 + 0.5 * R_match)
```

The supplied full-method settings use `lambda_r=0.25` and `lambda_e=0.15`.
`R_task` combines format, geometry and RGB-relative task utility at
0.15/0.50/0.35, as in the supplementary implementation. `R_rel` checks the four
state fields; `R_match` matches modality-evidence records. The historical
`R_step` checks schema/type coverage, not strict serialized key/step order.
The training reward accepts the better pixel/normalized box interpretation
against GT; this is a training convention, **not the evaluation parser**.
These details are preserved rather than silently changing the released method.

### Relation Data

Construct outcome-derived labels from RGB-only, TIR-only and paired candidate
predictions. Existing complete predictions can be reused; no additional model
inference is needed to run this builder:

```bash
python tools/build_relation_data.py --base_json /data/candidates.json \
  --rgb_jsonl /data/rgb_predictions.jsonl --tir_jsonl /data/tir_predictions.jsonl \
  --both_jsonl /data/paired_predictions.jsonl --output_dir /data/relation_5k
```

The builder selects 5,000 natural queries with seed 20260721, excluding controlled
diagnostics. Each prediction row needs `sample_id`, `iou`, and `format_ok`.
Base rows need `sample_id`, `case_type=both_advantage`, `rgb_path`, `tir_path`,
`query`, pixel `bbox`, `image_size=[width,height]`, and
`metadata.source_sample_id`. Candidate generation and exact source selection
are not bundled; original serialized July candidate predictions were deleted,
so regenerated annotations are not claimed identical to the originals.

RL accepts a JSON array of records:

```json
[{"sample_id":"train-1","rgb_path":"images/rgb.jpg","tir_path":"images/tir.jpg",
  "query":"the person","bbox":[10,20,100,200],"image_width":640,"image_height":480,
  "rgb_reliability":"high","tir_reliability":"medium",
  "relation":"complementary","dominant_modality":"RGB","rgb_baseline_iou":0.8}]
```

Use original-image **pixel** GT for training. Paths are absolute or relative to
the JSON file. Supply a separate validation JSON array; duplicate/overlapping
sample IDs, missing relation labels and incompatible image dimensions are rejected.
Keep train and validation image content separate as well.

### SFT Warm-Up

`train_sft.py` provides the prerequisite schema-adaptation stage, not the final
method. Supply JSONL conversations with assistant-only supervision:

```json
{"images":["images/rgb.jpg","images/tir.jpg"],"messages":[{"role":"system","content":"Return strict JSON only."},{"role":"user","content":"Locate the person. Return a bbox normalized to 0..1000."},{"role":"assistant","content":"{\"bbox\":[100,200,300,800]}"}]}
```

Replace example messages with relation-supervision conversations and targets;
the example is only a file-format demonstration. Image paths are relative to
the JSONL file or absolute. Supplementary warm-up settings are 600 steps at 3e-6:

```bash
pip install -r requirements-train.txt
torchrun --standalone --nproc_per_node=2 train_sft.py \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --train_jsonl /data/train.jsonl --eval_jsonl /data/val.jsonl \
  --output_dir outputs/sft --bf16 --gradient_checkpointing \
  --max_steps 600 --learning_rate 3e-6 --gradient_accumulation_steps 4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --seed 42 \
  --save_steps 100 --eval_steps 100
```

### GSPO Training and Export

Check data/reward wiring on CPU first, then run the paper-level one-epoch
schedule on two BF16-capable GPUs. No synthetic degradation is used in training.

```bash
python train.py --train_json /data/relation_train.json --eval_json /data/relation_val.json \
  --output_dir outputs/reward_check --dry_run_rewards

torchrun --standalone --nproc_per_node=2 train.py \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --adapter_path outputs/sft/final_model \
  --train_json /data/relation_train.json --eval_json /data/relation_val.json \
  --output_dir outputs/gspo --num_train_epochs 1 --learning_rate 1e-6 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_generations 4 --max_completion_length 1024 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --relation_state_weight 0.25 --hgrpo_evidence_weight 0.15 \
  --loss_type grpo --importance_sampling_level sequence --scale_rewards group --bf16

python tools/merge_adapter.py --model Qwen/Qwen3-VL-8B-Instruct \
  --adapter outputs/gspo/final_model --output outputs/merged
```

For 4B use `Qwen/Qwen3-VL-4B-Instruct` throughout. Evaluate `outputs/merged` with
the benchmark commands above. Nonempty output directories are protected. RL
records its configuration in the output directory. `--max_steps` explicitly
overrides epoch scheduling; the supplementary shell used 1,250 steps, whereas
the paper states one epoch. They must not be assumed equivalent across batch
and generation settings. This release defaults to the paper's one epoch.

The minimal RL implementation preserves the supplementary reward calculations;
paths, validation, defaults and a historical single-sample multimodal logprob
compatibility fix are adapted. The training prompt also removes the contradictory
"BBox first" instruction in the supplement and serializes evidence, relation
state, then final bbox as described in the paper. This prompt correction is not
claimed identical to the historical training prompt. The upstream TRL revision is fixed in
`requirements-train.txt`; the compatibility fix is local, not an upstream edit.
CPU checks do not establish end-to-end retraining equivalence. Exact historical
data, the step/epoch discrepancy and the full training-to-merged-weight chain
still need verification before claiming exact recreation of the hosted weights.

## Tests and License

```bash
python -m unittest discover -s tests -v
```

Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The Qwen3-VL
derivative weights and input datasets retain their applicable upstream terms.
