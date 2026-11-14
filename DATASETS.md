# Dataset Downloads and Reproduction Limits

Download images and source annotations from their original maintainers under
their terms. Our [Degraded-1K annotations and retained cache](https://huggingface.co/datasets/tyy1850819840/RelMLLM-Degraded-1K)
and [DV TIR-500 selector](https://huggingface.co/datasets/tyy1850819840/RelMLLM-DV-TIR-500)
are on HF. Only the selected degraded cache is mirrored, not clean benchmark
images, upstream DV QA text, full GroundBench, RGB-Th or original DV500.
The code licence does not relicense upstream data; see the
[unresolved image-rights notice](https://huggingface.co/datasets/tyy1850819840/RelMLLM-Degraded-1K/blob/main/IMAGE_RIGHTS.md).

## RGBT-GroundBench

[Official code](https://github.com/crazyxiaoxi/RGBT-GroundBench) and
[official data](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset).

| Source | Annotations | Images |
|---|---|---|
| FLIR | [ann_flir.tar](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/ann_flir.tar) | [data_flir.tar](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/data_flir.tar) |
| M3FD | [ann_m3fd.tar](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/ann_m3fd.tar) | [data_m3fd.tar](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/data_m3fd.tar) |
| MFAD | [ann_mfad.tar](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/ann_mfad.tar) | [data_mfad.tar](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/data_mfad.tar) |

Extract archives following the upstream layout. The archived full test contains
2,104 RefFLIR + 3,776 RefM3FD + 4,244 RefMFAD queries = 10,124. TestA/B/C are
overlapping subsets; do not concatenate them as independent full tests.
Current public annotation packaging is not asserted byte-identical to restored
manifests. A version-pinned annotation converter/selector is still outstanding.

`tools/evaluate_grounding.py` accepts a JSON list of records with `sample_id`,
`images` ([RGB_path,TIR_path]), `query`, `target_bbox` (pixel xyxy), and modality fields
documented in its source. Model output is single-image xyxy normalized to 1,000.
Shared RGB/TIR ground truth is not independent TIR annotation under misalignment.

## RGB-Th-Bench

[Official paper](https://arxiv.org/abs/2503.19654) and its
[author-linked archive](https://drive.google.com/file/d/1aNaKYpI0lirhDUm34eq6PChA0pmjoXUD/view?usp=sharing).
Extract locally and pass the directory containing `questions.json` files to
`tools/evaluate_rgbth.py --dataset_root`. Archived input: 1,044 RGB-Txt and 812
RGB-Thermal-Txt queries (1,856 total). Confirm this version/count before comparison.
Headline QAcc first averages skill accuracies within each data type, then
equally averages the two type scores. It is not a sample-weighted micro average.

## DV-1K-Balanced

Obtain [DualVision v1 train.json](https://github.com/abrarmajeedi/DualVision/blob/main/data/v1/train.json)
and [v1 dv500.jsonl](https://github.com/abrarmajeedi/DualVision/blob/main/data/v1/dv500.jsonl).
Images: [LLVIP](https://bupt-ai-cz.github.io/LLVIP/) and
[HDRT original publication/access instructions](https://www.sciencedirect.com/science/article/abs/pii/S1566253525001824).
Use the maintainers' download and licensing instructions, not an unofficial rehost.
Expected image roots are `LLVIP/visible`, `LLVIP/infrared`, `HDRT/visible`,
and `HDRT/infrared` under your local data directory.

The custom 1,000-query set combines 500 DV500 RGB-category examples and 500
held-out IR-category examples from v1 source data (297 LLVIP / 203 HDRT).
Archived seed: 20260718, first QA only. It is not the upstream DV500 alone.

The exact added500-example selector is now [released on HF](https://huggingface.co/datasets/tyy1850819840/RelMLLM-DV-TIR-500).
It records source rows and image/source IDs, not modified QA text. "TIR" refers
to the question category; inference still receives both RGB and TIR images.
The builder combines unchanged upstream DV500 with the selected first IR QA:

```bash
hf download tyy1850819840/RelMLLM-DV-TIR-500 --repo-type dataset --revision v1-selected-20261115 --local-dir data/dv-tir500
python tools/build_eval_data.py dv1k --release-root data/dv-tir500 \
  --image-root /data/DV --train /data/train.json --dv500 /data/dv500.jsonl \
  --output /data/dv1k.jsonl
bash scripts/evaluate.sh dv1k models/RelMLLM-8B /data/dv1k.jsonl outputs/dv1k
```

Use the [pinned train.json](https://github.com/abrarmajeedi/DualVision/blob/37c85d9f52015b7f38c2c6edea0b2cc18c7b3bdf/data/v1/train.json)
and [pinned DV500](https://github.com/abrarmajeedi/DualVision/blob/37c85d9f52015b7f38c2c6edea0b2cc18c7b3bdf/data/v1/dv500.jsonl).
The image root contains `LLVIP/{visible,infrared}` and
`HDRT/{visible,infrared}`. Source versions, paired file existence, IDs and counts
are checked. `--ir-only` exports only the added500-example subset, which must
not be reported as full DV-1K accuracy. Existing outputs are never overwritten.

The mixed-answer evaluator uses question/reference answer to determine answer
type. That archived convention is not a ground-truth-blind deployment protocol.

## Degraded-1K

The [public v2 release](https://huggingface.co/datasets/tyy1850819840/RelMLLM-Degraded-1K)
contains333 RGB-degraded,333 TIR-degraded and334 jointly degraded queries,
source IDs and17 approved query corrections. GT/IDs/image references are
unchanged; the ambiguous legacy `degradation_type` and synthetic state metadata
are omitted. Synthetic relation metadata are not outcome-derived references.

The retained evaluated cache is included as three archives:281 FLIR,454 M3FD
and599 MFAD files (1,334 total;877.3 MiB of image data). Clean counterparts
come from GroundBench. Original corruption parameters and random-state traces
have not been fully recovered; use these retained bytes, not approximate new
corruptions. Upstream image rights remain unresolved, not newly relicensed.

Use this asset layout (local symlinks are supported):

```text
assets/clean/{flir,m3fd,mfad}/{rgb,ir}/...
assets/degraded/{rgb_degraded,tir_degraded,both_degraded}/{rgb,tir}/...
```

```bash
hf download tyy1850819840/RelMLLM-Degraded-1K --repo-type dataset --revision v2-quality-screening-images-20261115 --local-dir data/degraded1k
mkdir -p /data/assets
python -m zipfile -e data/degraded1k/images/degraded-flir.zip /data/assets
python -m zipfile -e data/degraded1k/images/degraded-m3fd.zip /data/assets
python -m zipfile -e data/degraded1k/images/degraded-mfad.zip /data/assets
python tools/build_eval_data.py degraded --release-root data/degraded1k \
  --image-root /data/assets --output /data/degraded-v2.json
bash scripts/evaluate.sh degraded models/RelMLLM-8B /data/degraded-v2.json outputs/degraded-v2
```

Place clean source images under `/data/assets/clean` before running the builder.
The earlier annotation-only tag `v2-quality-screening-20261115` remains unchanged.
The builder checks all1,000 IDs, image files, RGB dimensions, pixel GT and
333/333/334 partition counts. `--check-only` validates without writing outputs.
`--version v1-archived-queries` restores old query strings with known defects
(including target-coordinate-bearing queries), only for disclosed historical
comparisons. The default is corrected v2.

To synchronize the same17 query edits to an existing **normalized** GroundBench
JSON with10,124 original source IDs (not directly to upstream tar archives):

```bash
python tools/build_eval_data.py groundbench-v2 --release-root data/degraded1k \
  --groundbench /data/groundbench-v1.json --output /data/groundbench-v2.json
```

Only query fields are patched; unexpected text, nonempty prompts or missing
source IDs stop the operation. Input files are not overwritten. Full GroundBench
conversion from upstream packaging remains a separate prerequisite.

All1,000 Degraded examples received agent-assisted visual screening, not full
human expert certification; ambiguous cases remain. Historical paper/recovered
scores are not new v2 measurements. Report weight/data versions with every score.
