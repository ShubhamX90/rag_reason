# Method and checkpoint-selection disclosure

## Answer-only checkpoint selection

The reported updated answer-only launcher intentionally preserves its historical selection configuration:

```text
DEV_DOC_VERDICT_WEIGHT=0.0
DEV_FORMAT_WEIGHT=0.0
DEV_ABSTAIN_WEIGHT=1.0
```

In `code/train/train_qlora.py`, the remaining macro-F1 weight is calculated as `max(0, 1 - doc_weight - format_weight - abstain_weight)`. With the settings above, that weight is zero. The adapter stored as `best_dev_f1` is therefore selected by abstention accuracy only, despite its historical directory name.

This is retained to reproduce the historical answer-only recipe; it must not be described as semantics/F1-based model selection. Any revised semantics-first selection policy would create a new experiment and should be reported separately rather than substituted for the stored results.

## Missing current adapters

The latest K/L and updated answer-only adapters were not present in the repository at cleanup time. The release preserves their final evaluated outputs, complete per-example coverage, and launch/evaluation contracts, but cannot support exact adapter-only re-inference until those adapters are separately released.

## Run K training provenance

The exact original Run K Qwen training submission wrapper was not available locally. The K split/message construction and Qwen evaluation matrix scripts remain, but the training command should be described as reconstructed from the retained generic trainer and recipe artifacts, not as an original job script.
