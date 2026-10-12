# Recovered Checkpoint Reproduction

The author-approved release contains recovered merged BF16 RelMLLM-4B / 8B,
not the later coordinate-binding step120 adapters. A fresh full-file SHA256
check passed the recovered manifests (13 entries / 15 entries). Hash agreement
establishes file identity, not exact equality to every paper number.

| Model | GroundBench Acc@0.5 | RGB-Th macro QAcc | Degraded-1K Acc@0.5 | DV-1K Acc |
|---|---:|---:|---:|---:|
| 4B recovered | 72.6788 | 75.2481 | 54.20 | 79.30 |
| 8B recovered | 73.1628 | 75.6745 | 59.80 | 78.40 |
| 4B paper | 72.3627 | 75.46 | 48.70 | 79.30 |
| 8B paper | 73.3011 | 75.61 | 61.10 | 78.30 |

All values are percentages from complete archived evaluation summaries.
Degraded-1K differs by +5.50 points (4B) / -1.30 points (8B); the cause is not
established. We do not label this exact reproduction of the paper table.

## Evaluation Protocol

| Setting | GroundBench / Degraded | RGB-Th-Bench | DV-1K |
|---|---|---|---|
| Precision / batch | BF16 / 1 | BF16 / 1 per shard | BF16 / 1 per shard |
| Max new tokens | 256 | 8 | 32 |
| Positions | Native | Paired M-RoPE | Paired M-RoPE |
| Output | Relation-state JSON | Direct yes/no | Answer-type-aware |

All are greedy, RGB first. VQA retains a 10-second generation time limit,
which can be hardware-sensitive. Grounding extracts JSON including explanatory
or fenced responses and validates box ordering, not all strict finite/range
constraints. `format_ok` is legacy parser validity, not strict geometry validity.
The generic VQA example is not the benchmark-specific prompt. DV-1K includes
counts; an obsolete all-yes/no run is excluded. RGB-Th headline is the equal
average of RGB-Txt and RGB-Thermal-Txt accuracies, not the micro average.

See [DATASETS.md](DATASETS.md) for custom-selector/cache availability gaps.
Training data and training code are not included. This is an inference and
evaluation release, not a complete end-to-end training reproduction.
