# Dataset Downloads and Reproduction Limits

Download images and source annotations from their original maintainers under
their terms. Our [Degraded-1K annotations, selected original pairs and retained cache](https://huggingface.co/datasets/tyy1850819840/RelMLLM-Degraded-1K)
and [DV TIR-500 selector](https://huggingface.co/datasets/tyy1850819840/RelMLLM-DV-TIR-500)
are on HF. Only Degraded's selected original/degraded images are mirrored, not
unrelated upstream images, upstream DV QA text, full GroundBench, RGB-Th or original DV500.
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

We randomly selected 500 examples from the DV-204K training set whose original
annotations indicate reliance on TIR evidence, and removed these examples from
the training data used for RelMLLM. We combined them with the original DV-500
evaluation set to form the 1,000-example DV-1K evaluation set.
The selected examples comprise 297 LLVIP and 203 HDRT examples.
Selection seed: 20260718; first QA only, with original QA text unchanged.

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

The [evaluation set](https://huggingface.co/datasets/tyy1850819840/RelMLLM-Degraded-1K)
contains333 RGB-degraded,333 TIR-degraded and334 jointly degraded queries.
`data/degraded.json` contains query text, pixel xyxy GT, dimensions, original
and evaluated RGB/TIR image paths, degradation condition and source sample ID.

The original counterparts are included:414 FLIR,642 M3FD and854 MFAD files
(1,910 files,955 unique RGB/TIR pairs; multiple queries can share one pair).
The retained degraded cache adds281/454/599 files (1,334 total).
Every record maps GT/query to original `clean_images` and evaluated `images`;
these selected1,000 queries need no separate upstream image download.
Original corruption parameters and random-state traces
have not been fully recovered; use these retained bytes, not approximate new
corruptions. Upstream image rights remain unresolved, not newly relicensed.

Use this asset layout (local symlinks are supported):

```text
assets/clean/{flir,m3fd,mfad}/{rgb,ir}/...
assets/degraded/{rgb_degraded,tir_degraded,both_degraded}/{rgb,tir}/...
```

```bash
hf download tyy1850819840/RelMLLM-Degraded-1K --repo-type dataset --revision eval-20261115 --local-dir data/degraded1k
mkdir -p /data/assets
python -m zipfile -e data/degraded1k/images/clean-flir.zip /data/assets
python -m zipfile -e data/degraded1k/images/clean-m3fd.zip /data/assets
python -m zipfile -e data/degraded1k/images/clean-mfad.zip /data/assets
python -m zipfile -e data/degraded1k/images/degraded-flir.zip /data/assets
python -m zipfile -e data/degraded1k/images/degraded-m3fd.zip /data/assets
python -m zipfile -e data/degraded1k/images/degraded-mfad.zip /data/assets
python tools/build_eval_data.py degraded --release-root data/degraded1k \
  --image-root /data/assets --output /data/degraded.json
bash scripts/evaluate.sh degraded models/RelMLLM-8B /data/degraded.json outputs/degraded
```

All six archives extract under `/data/assets`; no historical local path or symlink is required.
The builder checks all1,000 IDs, image files, RGB dimensions, pixel GT and
333/333/334 partition counts. `--check-only` validates without writing outputs.
Input files are not overwritten. Report the actual weights and evaluation inputs
used for each score. Full GroundBench preparation remains a separate prerequisite.
