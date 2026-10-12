# Dataset Downloads and Reproduction Limits

Download images and annotations from their original maintainers under their
terms. No third-party images or annotations are redistributed in this repository.

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
Headline macro QAcc equally averages the two group accuracies, not all samples.

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

Seed alone does not define ordered eligibility or held-out filtering. The exact
sanitized selector is not yet distributed: full independent reconstruction is
an outstanding release item. The mixed-answer evaluator requires the archived
manifest contract; it uses question/reference answer to determine answer type.
That convention is not a ground-truth-blind deployment protocol.

## Degraded-1K

This custom GroundBench-derived cache contains 333 RGB-degraded, 333 TIR-degraded,
and 334 both-degraded queries. There is no verified standalone public download
yet. Exact selection, corruption parameters and cached images remain to
be released. GroundBench alone or an arbitrary corruption does not reconstruct
these scores. Reported values are archived recovered-checkpoint evaluations,
not a claim that the custom dataset is already publicly reproducible.
