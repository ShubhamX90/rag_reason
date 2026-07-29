# Run J: Benchmark-Augmented Strategy And Results

## Purpose

Run J was the first serious attempt to fix the benchmark over-abstention problem at the training level instead of trying to patch it at inference time.

The core idea was simple:

- add a controlled amount of benchmark-like answerable examples into train and val
- keep a large held-out benchmark untouched
- increase exposure to short answerable cases, especially 5-doc cases
- reduce the relative pull of refusal-heavy training geometry without deleting refusal behavior entirely

## What changed relative to earlier runs

Compared with the earlier F-style recipe, Run J changed three important things at once:

| area | change |
| --- | --- |
| split construction | moved from the old `692/56` setup to benchmark-augmented train/val built by [scripts/prepare_run_j_splits.py](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/scripts/prepare_run_j_splits.py:1) |
| validation set | increased val from `56` to `81` rows so short-context benchmark-like answerables were represented during checkpoint selection |
| mixture weighting | boosted short answerables and lightly downweighted short refusals through [slurm/examples/rebuild_messages_prompt_robust_j_benchmark_aug.sh](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/slurm/examples/rebuild_messages_prompt_robust_j_benchmark_aug.sh:1) and the resulting weight summary |

## Split engineering

Run J selected answerable benchmark rows into train/val while preserving a held-out benchmark large enough to remain meaningful.

### Selection summary

| item | value |
| --- | --- |
| benchmark answerable pool seen by selector | 741 |
| selected from benchmark | 193 |
| moved into train augmentation | 168 |
| moved into val augmentation | 25 |
| final augmented train rows | 862 |
| final combined val rows | 81 |
| held-out answerable benchmark rows | 548 |

### Rows dropped during curation

| reason | count |
| --- | --- |
| blank gold answer | 188 |
| duplicate benchmark query | 2 |
| query overlap with train/val | 2 |

### Selected benchmark rows by conflict type

| conflict type | count |
| --- | --- |
| Complementary information | 55 |
| Conflict due to misinformation | 13 |
| Conflict due to outdated information | 49 |
| Conflicting opinions or research outcomes | 22 |
| No conflict | 54 |

### Selected benchmark rows by doc count

| docs | count |
| --- | --- |
| 4 | 28 |
| 5 | 155 |
| 6 | 1 |
| 8 | 2 |
| 9 | 1 |
| 10 | 6 |

### Selected benchmark rows by evidence bucket

| bucket | count |
| --- | --- |
| support_present | 165 |
| partial_only | 28 |

Source: [data/splits/run_j/run_j_split_summary.json](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/run_j/run_j_split_summary.json:1)

## Training-mixture engineering

Run J used the standard strict + runtime + minimal mixture builder, but it changed the weighting geometry so benchmark-like answerable rows were seen more often during SFT.

### Main weighting knobs

| knob | value |
| --- | --- |
| `answerable_exact_weight` | `1.35` |
| `answerable_short_weight` | `1.4` |
| `decision_answerable_short_extra_weight` | `1.15` |
| `answerable_mid_weight` | `1.15` |
| `answerable_partial_only_weight` | `1.15` |
| `refusal_short_weight` | `0.55` |
| `decision_refusal_short_extra_weight` | `0.75` |
| `refusal_long_weight` | `0.7` |
| `trust_align_refusal_weight` | `0.85` |
| `minimal_e2e_weight` | `4` |
| `minimal_partial_synthesis_weight` | `2` |

### Weighted answerable vs refusal totals

| bucket | raw | weighted |
| --- | --- | --- |
| refusal | 3252 | 1438.95 |
| answerable | 7422 | 10884.6592 |

### Weighted 5-doc geometry

| slice | raw | weighted |
| --- | --- | --- |
| `docs=5, answerable=False` | 3240 | 1430.55 |
| `docs=5, answerable=True` | 1926 | 4179.4987 |

This was the key geometric shift that made Run J much less refusal-biased on benchmark-like prompts.

Source: [data/messages/train_stagewise_prompt_robust_trace_text_j_benchmark_aug_weight_summary.json](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/messages/train_stagewise_prompt_robust_trace_text_j_benchmark_aug_weight_summary.json:1)

## Model/training setup

The 7B and 32B runs were intentionally kept almost identical so the split and weighting changes were the main variable.

| setting | 7B | 32B |
| --- | --- | --- |
| base model | `Qwen2.5-7B-Instruct` | `Qwen2.5-32B-Instruct` |
| epochs | `2` | `2` |
| learning rate | `2e-4` | `2e-4` |
| batch size | `1` | `1` |
| grad accumulation | `8` | `8` |
| max length | `12288` | `12288` |
| LoRA rank | `32` | `32` |
| LoRA alpha | `64` | `64` |
| LoRA dropout | `0.05` | `0.05` |
| NEFTune alpha | `5.0` | `5.0` |
| `CONFLICT_WEIGHT` | `3.55` | `3.55` |
| `CONTRACT_WEIGHT` | `3.0` | `3.0` |
| `ABSTAIN_WEIGHT` | `0.4` | `0.35` |
| `CITATION_WEIGHT` | `1.7` | `1.7` |

Sources:

- [slurm/sharanga/examples/qwen7b_stagewise_ddp_2h100_prompt_robust_j_benchmark_aug.sh](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/slurm/sharanga/examples/qwen7b_stagewise_ddp_2h100_prompt_robust_j_benchmark_aug.sh:1)
- [slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_j_benchmark_aug.sh](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_j_benchmark_aug.sh:1)

## Checkpoint selection

Run J selected checkpoints on the 81-row val set using a composite score that balanced conflict F1, doc-verdict accuracy, format success, abstain accuracy, and explicit penalties for false abstains.

### 7B

| epoch | macro_f1 | doc_acc | format_ok | abstain_acc | false_abstain_partial_only | false_abstain_support | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.5774 | 0.7663 | 0.9877 | 0.9877 | 1 | 0 | 0.7802 |
| 2.0 | 0.5454 | 0.8007 | 1.0000 | 0.9630 | 1 | 2 | 0.7625 |

Selected checkpoint: epoch `1.0`

### 32B

| epoch | macro_f1 | doc_acc | format_ok | abstain_acc | false_abstain_partial_only | false_abstain_support | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.5740 | 0.8170 | 1.0000 | 1.0000 | 0 | 0 | 0.8180 |
| 2.0 | 0.5994 | 0.8333 | 1.0000 | 0.9753 | 0 | 2 | 0.8133 |

Selected checkpoint: epoch `1.0`

Source:

- [outputs/sharanga_sync/run_j_2026_06_25/checkpoints/best_metrics_7b.json](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sharanga_sync/run_j_2026_06_25/checkpoints/best_metrics_7b.json:1)
- [outputs/sharanga_sync/run_j_2026_06_25/checkpoints/best_metrics_32b.json](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sharanga_sync/run_j_2026_06_25/checkpoints/best_metrics_32b.json:1)

## Canonical evaluation sets used in the audit

### Val canon

| rows | answerable | refusal | refusal_rate |
| --- | --- | --- | --- |
| 81 | 59 | 22 | 27.16% |

### Benchmark holdout canon

| rows | answerable | refusal | refusal_rate |
| --- | --- | --- | --- |
| 736 | 608 | 128 | 17.39% |

These figures match the deep audit and are the basis for all result tables below.

## Headline results

| run | contract_ok_pct | citation_pass_pct | conflict_acc_pct | conflict_support | doc_micro_pct | final_abstain_acc_pct | pred_abstain | avg_token_f1 | avg_rougeL_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7B Val | 98.8 | 55.2 | 64.20 | 81 | 76.09 | 98.77 | 23 | 0.4334 | 0.3097 |
| 7B Benchmark | 92.5 | 66.3 | 62.07 | 733 | 77.96 | 94.29 | 164 | 0.3555 | 0.3215 |
| 32B Val | 95.1 | 79.3 | 67.09 | 79 | 83.27 | 100.00 | 22 | 0.4578 | 0.3362 |
| 32B Benchmark | 92.8 | 55.1 | 62.55 | 729 | 82.05 | 94.97 | 149 | 0.3945 | 0.3620 |

## Abstention behavior

### Val abstain confusion

| model | TP | TN | FP | FN | pred_abstain | gold_abstain | accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | 22 | 58 | 1 | 0 | 23 | 22 | 98.77 |
| 32B | 22 | 59 | 0 | 0 | 22 | 22 | 100.00 |

### Benchmark abstain confusion

| model | TP | TN | FP | FN | pred_abstain | gold_abstain | accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7B | 125 | 569 | 39 | 3 | 164 | 128 | 94.29 |
| 32B | 120 | 579 | 29 | 8 | 149 | 128 | 94.97 |

This was the first run where the benchmark no longer looked catastrophically refusal-biased. The remaining weakness was not universal abstention anymore; it was narrower false-abstain behavior on the hardest short-context answerable slices.

## Main failure pattern that remained after Run J

The deep audit showed the remaining pain point was not spread evenly.

### 7B benchmark false-abstain labels

| gold label | count |
| --- | --- |
| Complementary information | 15 |
| Conflict due to misinformation | 10 |
| No conflict | 7 |
| Conflict due to outdated information | 4 |
| Conflicting opinions or research outcomes | 3 |

### 7B val false-abstain label

| gold label | count |
| --- | --- |
| Conflict due to misinformation | 1 |

Interpretation:

- Run J largely fixed broad refusal calibration.
- The hardest remaining cases were short answerable examples, especially `partial_only` and especially misinformation.
- Stage-2 confusion still clustered around `No conflict <-> Complementary information`.

## Why Run J mattered

Run J was the turning point because it proved that the over-abstention problem was not purely an inference problem. A different train/val geometry materially changed benchmark behavior.

In practical terms, Run J established four things:

1. benchmark-like answerable rows needed to be seen during training
2. the validation set also needed benchmark-like answerable rows, otherwise checkpoint selection stayed blind to the true failure mode
3. short answerable rows had to be upweighted relative to short refusals
4. once the worst over-abstention was fixed, the remaining problems became more local and diagnosable

## Stored artifacts

- Strategy + split builder: [slurm/examples/rebuild_messages_prompt_robust_j_benchmark_aug.sh](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/slurm/examples/rebuild_messages_prompt_robust_j_benchmark_aug.sh:1)
- Split summary: [data/splits/run_j/run_j_split_summary.json](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/run_j/run_j_split_summary.json:1)
- Weight summary: [data/messages/train_stagewise_prompt_robust_trace_text_j_benchmark_aug_weight_summary.json](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/messages/train_stagewise_prompt_robust_trace_text_j_benchmark_aug_weight_summary.json:1)
- Deep audit markdown: [archive/run_j_deep_audit_2026_06_25.md](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/docs/experiments/archive/run_j_deep_audit_2026_06_25.md)
- Deep audit json: [archive/run_j_deep_audit_2026_06_25.json](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/docs/experiments/archive/run_j_deep_audit_2026_06_25.json)
- Synced run outputs and reports: [outputs/sharanga_sync/run_j_2026_06_25](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sharanga_sync/run_j_2026_06_25)
