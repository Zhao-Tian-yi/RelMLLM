# RelMLLM

Code for **Beyond Paired Inputs: Learning to Leverage RGB-T Cross-Modal
Relations in Multimodal Large Language Models**.

A minimal repository for paired RGB/TIR inference, benchmark evaluation, and
relation-supervised LoRA SFT. Inputs are RGB first, TIR second. Grounding uses
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

## Training

`train.py` adapts the historical relation-SFT trainer, with input validation and
assistant-only loss. It uses paired M-RoPE and all-linear LoRA. It is a runnable
SFT component, **not the complete recipe that produced the released checkpoints**.
The historical training-data construction and complete post-training settings
are not yet packaged. The settings below are script defaults, not a verified
paper-training recipe. No later coordinate-binding adapters are used here.

Supply training and held-out validation JSONL files, one conversation per line:

```json
{"images":["images/rgb.jpg","images/tir.jpg"],"messages":[{"role":"system","content":"Return strict JSON only."},{"role":"user","content":"Locate the person. Return a bbox normalized to 0..1000."},{"role":"assistant","content":"{\"bbox\":[100,200,300,800]}"}]}
```

Image paths are absolute or relative to the JSONL file. Replace the example
messages with your actual relation-supervision conversations and targets; the
example defines the file format, not the historical 5K supervision. Keep
validation images separate from training images. Check the inputs without loading
the model, then train from the matching Qwen3-VL base:

```bash
python train.py --train_jsonl /data/train.jsonl --eval_jsonl /data/val.jsonl \
  --output_dir outputs/sft --dry_run

python train.py --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --train_jsonl /data/train.jsonl --eval_jsonl /data/val.jsonl \
  --output_dir outputs/sft --bf16 --gradient_checkpointing \
  --max_steps 120 --learning_rate 1e-5 --gradient_accumulation_steps 4 \
  --lora_r 8 --lora_alpha 16 --seed 42 --merge_output_dir outputs/merged
```

For 4B use `Qwen/Qwen3-VL-4B-Instruct`. The trainer saves a LoRA adapter in
`outputs/sft/final_model`; the optional single-process merge exports a standalone
model to `outputs/merged`. Evaluate it with the commands above, changing the
model path. Nonempty output directories are refused unless explicitly resuming
training. Run `python train.py --help` for all settings.

## Tests and License

```bash
python -m unittest discover -s tests -v
```

Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The Qwen3-VL
derivative weights and input datasets retain their applicable upstream terms.
