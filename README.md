# RelMLLM

Code for **Beyond Paired Inputs: Learning to Leverage RGB-T Cross-Modal Relations in Multimodal Large Language Models**.

Paired RGB/TIR grounding and VQA, with **relation-guided GSPO training** and SFT warm-up.

## Setup

Linux, Python 3.10, and CUDA-compatible PyTorch:

```bash
pip install -r requirements.txt
hf download tyy1850819840/RelMLLM-8B --revision recovered-20261012 --local-dir models/RelMLLM-8B
```

Weights: [4B](https://huggingface.co/tyy1850819840/RelMLLM-4B) / [8B](https://huggingface.co/tyy1850819840/RelMLLM-8B). Replace `RelMLLM-8B` with `RelMLLM-4B` for 4B.

## Evaluate

Our evaluation additions are on Hugging Face:

| Addition | Download | Contents |
|---|---|---|
| Degraded-1K v2 | [HF dataset](https://huggingface.co/datasets/tyy1850819840/RelMLLM-Degraded-1K) | 1,000 annotations and17 query corrections; **no image cache** |
| DV-1K TIR-500 | [HF dataset](https://huggingface.co/datasets/tyy1850819840/RelMLLM-DV-TIR-500) | Exact500-example IR selector; combine with upstream DV500 |

```bash
hf download tyy1850819840/RelMLLM-Degraded-1K --repo-type dataset --revision v2-quality-screening-20261115 --local-dir data/degraded1k
hf download tyy1850819840/RelMLLM-DV-TIR-500 --repo-type dataset --revision v1-selected-20261115 --local-dir data/dv-tir500
```

Prepare local inputs with `tools/build_eval_data.py` using [DATASETS.md](DATASETS.md).
GroundBench, RGB-Th and original DV500 images/annotations are obtained upstream.
Degraded evaluation still requires the original corruption cache; this metadata
release does not regenerate it. DV's IR question category still uses paired RGB/TIR inputs.

```bash
bash scripts/evaluate.sh groundbench models/RelMLLM-8B /data/groundbench.json outputs/groundbench
bash scripts/evaluate.sh rgbth models/RelMLLM-8B /data/RGB-Th-Bench outputs/rgbth
bash scripts/evaluate.sh dv1k models/RelMLLM-8B /data/dv1k.jsonl outputs/dv1k
bash scripts/evaluate.sh degraded models/RelMLLM-8B /data/degraded.json outputs/degraded
```

## Train

Prepare SFT conversations and labeled relation data using [Reproduction Details](docs/REPRODUCIBILITY.md). Use two BF16-capable GPUs:

```bash
pip install -r requirements-train.txt
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh Qwen/Qwen3-VL-8B-Instruct \
  /data/sft_train.jsonl /data/sft_val.jsonl \
  /data/relation_train.json /data/relation_val.json outputs/train-8b
```

Runs reward validation, SFT warm-up (600 steps), GSPO (one epoch), and adapter export. Set `DRY_RUN=1` for CPU reward validation only. For 4B, use `Qwen/Qwen3-VL-4B-Instruct` and a new output directory.

## Agent-Assisted Reproduction

Give your agent this prompt with your paths filled in:

> Read `skills/relmllm-reproduce/SKILL.md` and reproduce RelMLLM using the launch scripts. Mode: evaluation / training / both. Model: [path or model ID]. Data: [local paths]. Output: [new directory]. GPUs: [allowed indices]. Check prerequisites first, run within these resources, and report results and missing inputs. Do not substitute datasets or claim unverified paper-score reproduction.

The [agent skill](skills/relmllm-reproduce/SKILL.md) is a repository-local instruction file; no global installation is required.

## Reproduction Status

Released weights are the author-approved recovered version, **not an exact match to every paper score**. The DV selector and corrected Degraded annotations are public; historical scores are **not v2 measurements**. Image-cache availability, GroundBench conversion and historical training provenance still limit exact reproduction. Default RL batch is1 per device, while the manuscript reports2; epoch/step equivalence depends on batch and generation settings. See [Reproduction Details](docs/REPRODUCIBILITY.md).

Tests: `python -m unittest discover -s tests -v`. Code: [Apache-2.0](LICENSE); upstream model and dataset terms apply.
