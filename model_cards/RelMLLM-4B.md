---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - vision-language
  - rgb-thermal
  - visual-grounding
  - multimodal-reasoning
---

# RelMLLM-4B

RelMLLM-4B is the merged 4B checkpoint accompanying **Beyond Paired Inputs:
Learning to Leverage RGB-T Cross-Modal Relations in Multimodal Large Language
Models**. It is derived from Qwen3-VL-4B-Instruct and accepts aligned RGB and
thermal-infrared observations.

Use the inference code in the companion RelMLLM repository. The RGB image must
be provided first and the TIR image second.

## Selected results

| Benchmark | Metric | Score |
|---|---|---:|
| RGBT-GroundBench | Acc@0.5 | 72.6788 |
| Degraded-1K | Acc@0.5 | 54.20 |
| RGB-Th-Bench | macro QAcc | 75.2481 |
| DV-1K-Balanced | Accuracy | 79.30 |

These are complete archived evaluations of the recovered checkpoint, not exact
paper-table reproduction. Paper GroundBench/Degraded scores were 72.3627/48.70.
Code, prompts, dataset downloads and custom-set reconstruction limitations:
[RelMLLM GitHub](https://github.com/Zhao-Tian-yi/RelMLLM).

The model is intended for research, not safety-critical sensor-health or
surveillance decisions. Reliability fields are task-conditioned evidence
utilities rather than physical sensor diagnoses. See the companion repository
for protocols, limitations, code, license, and attribution.
