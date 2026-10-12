---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3-VL-4B-Instruct
  - Qwen/Qwen3-VL-8B-Instruct
tags:
  - vision-language
  - rgb-thermal
  - visual-grounding
  - multimodal-reasoning
---

# RelMLLM Model Card

## Model description

RelMLLM is an RGB-T multimodal large language model derived from
Qwen3-VL-Instruct. It is post-trained to explicitly assess modality-specific
evidence, predict an RGB-T relation state, and connect that state to a final
grounding or VQA decision.

Two model sizes are released:

| Model | Base model | Intended role |
|---|---|---|
| RelMLLM-4B | Qwen3-VL-4B-Instruct | Lower-cost inference and analysis |
| RelMLLM-8B | Qwen3-VL-8B-Instruct | Recommended model |

## Intended use

- research on aligned RGB-T multimodal understanding;
- RGB-T referring grounding;
- binary RGB-T visual question answering;
- analysis of modality reliability and cross-modal relations.

The model is not intended to provide safety-critical sensor-health judgments.
Its reliability labels describe task-conditioned evidence utility.

## Training summary

The released models were post-trained from Qwen3-VL-Instruct with structured
relation supervision and relation-guided policy optimization. Training used
natural aligned RGB-T pairs. Synthetic degradation examples were not used for
post-training. Training data and annotations are not distributed with this
release because they are subject to separate dataset terms.

## Selected results

| Model | RGBT-GroundBench Acc@0.5 | Degraded-1K Acc@0.5 |
|---|---:|---:|
| RelMLLM-4B recovered | 72.6788 | 54.20 |
| RelMLLM-8B recovered | 73.1628 | 59.80 |

Complete archived recovered-checkpoint evaluations are not exact paper-table
reproduction. See REPRODUCIBILITY.md for paper comparisons and DATASETS.md for
custom subset/cache availability limitations.

GroundBench and Degraded-1K use the structured relation-state grounding
interface. RGB-Th uses direct Yes/No; DV-1K uses answer-type-aware decoding.
Consult the paper for complete protocols, subset breakdowns, and baselines.

## Limitations

- The inputs are expected to be spatially aligned RGB and TIR observations.
- Performance under temporal desynchronization, severe misregistration, and
  real hardware faults has not been comprehensively validated.
- Relation-state labels are outcome-derived and model-conditioned; they are
  not human-certified descriptions of physical sensor quality.
- The 4B model is more likely to produce an invalid or abstaining localization
  under simultaneous severe degradation.
- The model can produce plausible but incorrect evidence statements and
  bounding boxes. Verify predictions before downstream use.
- The model inherits limitations and potential biases from Qwen3-VL and the
  post-training data.

## Privacy and safety

No biometric identification, surveillance, medical, automotive safety, or
other high-stakes deployment has been validated. Users are responsible for
appropriate evaluation, access control, and compliance in their jurisdiction.

## License and attribution

RelMLLM is a derivative of Qwen3-VL-Instruct. The code and released weights are
distributed under Apache-2.0. See `LICENSE` and `NOTICE` for attribution.
