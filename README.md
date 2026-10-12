# RelMLLM

## Recovered Checkpoint Release (2026-10-12)

This is an **inference/evaluation release**, not a complete training release.
The author-approved recovered merged BF16 checkpoints have been checked against
the archived release version. They are not later diagnostic adapters.

Weights: [RelMLLM-4B](https://huggingface.co/tyy1850819840/RelMLLM-4B) and
[RelMLLM-8B](https://huggingface.co/tyy1850819840/RelMLLM-8B).
Hosting verification is tracked in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
Weight hosting is in progress until that checklist records successful public
upload verification. Complete archived evaluation summary:
[machine-readable results](reports/recovered_four_dataset_summary.json).

### Complete Archived Results

Percentages below distinguish recovered results from the paper. They are not
a claim of exact reproduction of every paper number.

| Model | GroundBench Acc@0.5 | RGB-Th macro QAcc | Degraded-1K Acc@0.5 | DV-1K Acc |
|---|---:|---:|---:|---:|
| 4B recovered | 72.6788 | 75.2481 | 54.20 | 79.30 |
| 8B recovered | 73.1628 | 75.6745 | 59.80 | 78.40 |
| 4B paper | 72.3627 | 75.46 | 48.70 | 79.30 |
| 8B paper | 73.3011 | 75.61 | 61.10 | 78.30 |

### Dataset Downloads

| Dataset | Official access |
|---|---|
| RGBT-GroundBench (10,124 queries) | [Code](https://github.com/crazyxiaoxi/RGBT-GroundBench), [images/annotations](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset) |
| RGB-Th-Bench (1,856 queries) | [Paper](https://arxiv.org/abs/2503.19654), [author-linked archive](https://drive.google.com/file/d/1aNaKYpI0lirhDUm34eq6PChA0pmjoXUD/view?usp=sharing) |
| DV-1K-Balanced | Derived from [DualVision v1](https://github.com/abrarmajeedi/DualVision/tree/main/data/v1), [LLVIP](https://bupt-ai-cz.github.io/LLVIP/), [HDRT](https://www.sciencedirect.com/science/article/abs/pii/S1566253525001824) |
| Degraded-1K | GroundBench-derived cache; standalone archive not yet supplied |

[DATASETS.md](DATASETS.md) gives downloads and remaining exact custom-selector/cache
availability gaps. No third-party images/annotations are redistributed here.
Independent public reconstruction of custom sets is not yet complete.

### Benchmark Evaluation

```bash
pip install -r requirements-eval.txt
python tools/evaluate_grounding.py --model /path/to/RelMLLM-8B \
  --eval-json /path/to/grounding_manifest.json --batch-size 1 \
  --max-new-tokens 256 --output-jsonl outputs/grounding.jsonl \
  --summary-json outputs/grounding_summary.json
python tools/evaluate_rgbth.py --model_name_or_path /path/to/RelMLLM-8B \
  --dataset_root /path/to/RGB-Th-Bench --bf16 --max_new_tokens 8 \
  --results_jsonl outputs/rgbth.jsonl --output_report outputs/rgbth.md
python tools/evaluate_dv1k.py --model_name_or_path /path/to/RelMLLM-8B \
  --input_jsonl /path/to/dv1k_manifest.jsonl --bf16 --max_new_tokens 32 \
  --results_jsonl outputs/dv1k.jsonl --output_report outputs/dv1k.md
```

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) records benchmark-specific prompts,
scoring and token caps. The generic yes/no VQA example below is not the DV-1K
mixed-answer evaluator. RGB-Th macro QAcc is not micro accuracy.

Official inference release for **Beyond Paired Inputs: Learning to Leverage
RGB-T Cross-Modal Relations in Multimodal Large Language Models**.

RelMLLM takes an aligned RGB image and thermal-infrared (TIR) image and emits
a structured evidence-relation-decision record. The release supports referring
grounding and yes/no visual question answering (VQA).

## Release scope

This repository provides:

- merged RelMLLM-4B and RelMLLM-8B checkpoints;
- a standalone inference entry point;
- the structured prompts and output schemas used by the released models;
- an optional aligned RGB-T M-RoPE patch used during post-training;
- model and license documentation.

Training data, annotations, and training code are not part of this release.
The datasets used in the paper remain subject to their original licenses.

## Installation

```bash
conda create -n relmllm python=3.10 -y
conda activate relmllm
pip install -r requirements.txt
```

The tested environment used PyTorch 2.10.0, Transformers 5.4.0,
Accelerate 1.13.0, Safetensors 0.7.0, and Pillow 12.2.0.
For the recovered evaluation environment, install
`pip install -r requirements-reproduction.txt` instead of the broad dependency
ranges. Select a PyTorch build compatible with your CUDA/driver installation.

## RGB-T grounding

The RGB image must be passed first and the aligned TIR image second.

```bash
python inference.py \
  --model /path/to/RelMLLM-8B \
  --task grounding \
  --rgb examples/rgb.jpg \
  --tir examples/tir.jpg \
  --query "the person crossing the street"
```

The predicted bounding box uses the Qwen-style coordinate frame
`[x1, y1, x2, y2]` normalized to `[0, 1000]`. To also print image-pixel
coordinates, add `--pixel-bbox`.

## RGB-T VQA

```bash
python inference.py \
  --model /path/to/RelMLLM-8B \
  --task vqa \
  --rgb examples/rgb.jpg \
  --tir examples/tir.jpg \
  --query "Is a pedestrian visible in both observations?"
```

The VQA interface is intended for binary questions. Paper-matched mode returns
`Yes` or `No`; relation-state mode returns the answer inside structured JSON.

## Important protocol details

1. Inputs must be aligned observations of the same scene.
2. Input order is fixed: RGB first, TIR second.
3. Deterministic decoding is used by default (`do_sample=False`).
4. Grounding coordinates are normalized to `[0, 1000]` for one image, not a
   horizontally concatenated RGB-T canvas.
5. The default `--prompt-mode paper` selects the generic interface:
   relation-state JSON for grounding and direct Yes/No output for VQA.
6. `--aligned-rope auto` reproduces the reported evaluation code: disabled for
   grounding and enabled for VQA. Use `on` or `off` for controlled studies.

To request a structured relation-state trajectory for VQA, add
`--prompt-mode relation-state`.

## Output semantics

- `rgb_reliability`, `tir_reliability`: task-conditioned evidence utility,
  not physical sensor-health estimates;
- `relation`: one of `consistent`, `complementary`, `conflict`, `uncertain`;
- `dominant_modality`: one of `RGB`, `TIR`, `Both`, `None`;
- `steps`: a machine-checkable evidence-relation-decision record.

See [MODEL_CARD.md](MODEL_CARD.md) for limitations and benchmark results.

## License

The code and released derivative weights are provided under Apache-2.0. The
release derives from Qwen3-VL-Instruct, also licensed under Apache-2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE). Users remain responsible for complying
with the licenses of any input datasets and downstream applications.

## Citation

Please cite the paper if this release is useful. Full bibliographic metadata
will be added after the public paper record is available.
