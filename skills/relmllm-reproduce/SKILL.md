---
name: relmllm-reproduce
description: Run RelMLLM evaluation or paper-aligned SFT warm-up and relation-guided GSPO training using this repository. Use for agent-assisted reproduction, not coordinate-binding follow-up experiments.
---

# RelMLLM Reproduction

Read README.md, DATASETS.md, and docs/REPRODUCIBILITY.md from the repository root. Use scripts/evaluate.sh and scripts/train.sh rather than inventing training or scoring commands.

## Resolve Inputs

Determine evaluation, training, or both from the request. Obtain missing model/data paths, a new output directory, and authorized GPUs/resource budget before expensive work. Do not overwrite runs, terminate unrelated jobs, publish assets, or install packages globally. Use an isolated environment and pinned requirements. Never request credentials in chat.

Evaluation uses tagged recovered 4B/8B weights, RGB first/TIR second. Record whether evaluating recovered weights or a new training export. Training starts from the corresponding Qwen3-VL-Instruct base, not merged RelMLLM weights.

Degraded-1K requires the user to submit its HF access form (automatic approval) and run `hf auth login` locally with that account before download. Do not request a token in chat or fill in personal details or accept terms on the user's behalf. The DV selector does not require an application.

Inspect real inputs before launch. Grounding needs a JSON array with sample_id, images (RGB then TIR), query and pixel target_bbox. SFT needs JSONL conversations with assistant targets. RL needs pixel GT, dimensions, relation/reliability labels and RGB baseline utility; see the detailed schema. Verify files/dimensions, unique IDs and disjoint train/validation IDs, image content and source groups. No test labels may enter training. Do not fabricate relation labels or infer independent thermal annotations from shared boxes.

The added DV IR500 selector and Degraded-1K evaluation annotations and images are public on HF; follow the pinned downloads, six-archive extraction and `tools/build_eval_data.py` commands in DATASETS.md. DV preparation combines upstream RGB DV500 with selected IR500; both images remain inputs. Degraded includes its1,910 original RGB/TIR files,1,334 degraded files and1,000 annotations in `data/degraded.json`; `clean_images` and `images` define their mapping. Exact corruption generation is unavailable; use the retained cache rather than regenerating substitutes. Image redistribution permissions have not been independently established; do not claim they are cleared. If inputs are missing, report them and stop that stage. Do not silently substitute corruption, source selection or split definitions. Report scores only for the actual model and inputs evaluated.

## Execute

1. Run `python -m unittest discover -s tests -v`. Inspect CUDA/BF16 and authorized resources; training requires two suitable GPUs. Do not alter generation budgets to fit resources without the user's decision.
2. For training, run `DRY_RUN=1 bash scripts/train.sh BASE SFT_TRAIN SFT_VAL RL_TRAIN RL_VAL NEW_CHECK_DIR`. Inspect validation and rewards. Stop on failures; do not weaken validation.
3. With requested inputs/resources ready, run `CUDA_VISIBLE_DEVICES=GPU_IDS bash scripts/train.sh BASE SFT_TRAIN SFT_VAL RL_TRAIN RL_VAL NEW_RUN_DIR`. This runs SFT 600 steps, GSPO one epoch, then export to NEW_RUN_DIR/merged. SFT alone is not the paper method. Keep sequence-level importance sampling and hierarchical rewards unchanged.
4. Run `CUDA_VISIBLE_DEVICES=GPU_IDS bash scripts/evaluate.sh BENCHMARK MODEL DATA NEW_OUTPUT_DIR` for each requested available benchmark: groundbench, rgbth, dv1k or degraded. For newly trained weights use the merged export. Preserve raw predictions and reports.

Use separate directories for preflight and training; scripts refuse existing outputs. On interruption inspect partial results and evaluator resume support before recovery; do not automatically repeat completed inference or training.

## Verify and Report

Check exact requested IDs, duplicates/missing rows, counts (GroundBench 10124, RGB-Th 1856, DV-1K 1000, Degraded 1000 for archived splits), parsed scores and token limits. Preserve failures. RGB-Th is macro QAcc, not pooled accuracy. Preserve DV's reference-selected answer-type prompt and disclose this archived, non-GT-blind convention. Training reward parsing is not evaluation parsing.

Report model/data versions, commands, hardware, completed stages, output paths, metrics and deviations. Distinguish execution success from exact paper reproduction. Recovered checkpoints differ from some paper numbers; historical training provenance is incomplete. One epoch and supplementary 1250 steps are not presumed equivalent. Released prompt corrections and reward conventions are documented in docs/REPRODUCIBILITY.md. CPU tests or completed training do not prove exact recovery of hosted weights. Never claim missing results or identical retraining.
