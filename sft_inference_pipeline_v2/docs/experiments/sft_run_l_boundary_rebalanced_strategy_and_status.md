# Run L: Boundary-Rebalanced Strategy And Status

## Purpose

Run L is the follow-up to Run K.

The problem Run K exposed was not that short-context targeting was wrong. The problem was that Run K targeted mostly conflict-bearing short answerables and did not add short `No conflict` answerables. That left a boundary-shape risk:

- the model could learn "short answerable usually means some conflict"
- smaller models could become cleaner on abstention but slightly less balanced on stage-2 labeling

Run L keeps the useful part of Run K and fixes that missing boundary coverage.

## Design goal

Run L was engineered to preserve Run K strengths while repairing its likely source of remaining skew:

- keep strong short-context answerable exposure
- keep refusal weights controlled
- keep the stronger boundary drills
- add short `No conflict` answerable support rows
- explicitly include a small `partial_only` `No conflict` slice
- add mild extra runtime pressure on `No conflict` and misinformation boundaries

In plain language: Run L tries to teach the model that "short and answerable" does not automatically mean "there must be a conflict somewhere."

## Split engineering

Run L starts from Run J, not Run K. It then adds a broader short-context derived set than Run K.

Source builder: [scripts/prepare_run_l_splits.py](../../scripts/prepare_run_l_splits.py)

### Run L split summary

| item | value |
| --- | --- |
| base train rows | 862 |
| base val rows | 81 |
| derived rows added | 48 |
| final train rows | 910 |
| final val rows | 81 |

### Derived rows by conflict type

| conflict type | count |
| --- | --- |
| Complementary information | 10 |
| Conflicting opinions or research outcomes | 10 |
| Conflict due to outdated information | 6 |
| Conflict due to misinformation | 1 |
| No conflict | 21 |

### Derived rows by origin

| origin | count |
| --- | --- |
| `run_l_short5_support` | 27 |
| `run_l_short5_no_conflict_support` | 18 |
| `run_l_short5_no_conflict_partial_only` | 3 |

### Derived rows by parent doc count

| parent docs | count |
| --- | --- |
| 6 | 1 |
| 8 | 4 |
| 9 | 12 |
| 10 | 18 |
| 11 | 4 |
| 12 | 2 |
| 13 | 4 |
| 14 | 2 |
| 15 | 1 |

The key new ingredient versus Run K is the `21` derived `No conflict` rows, including `3` `partial_only` cases.

Source: [data/splits/run_l/run_l_split_summary.json](../../data/splits/run_l/run_l_split_summary.json)

## Message-mixture engineering

Run L rebuild path: [slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh](../../slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh)

### Builder changes relative to Run K

| knob | Run K | Run L |
| --- | --- | --- |
| runtime label `No conflict` | none | `1` |
| runtime label `Misinformation` | none | `2` |
| boundary `No conflict` | 1 | 2 |
| boundary `Complementary information` | 2 | 2 |
| boundary `Conflicting opinions...` | 2 | 2 |
| boundary `Outdated` | 2 | 2 |
| boundary `Misinformation` | 2 | 3 |
| doc-verdict boundary drill | 1 | 1 |
| partial-synthesis drills | same strong setting | same |

### New origin-sensitive sample weights

| origin weight | value |
| --- | --- |
| `run_l_short5_support` | `1.0` |
| `run_l_short5_no_conflict_support` | `1.25` |
| `run_l_short5_no_conflict_partial_only` | `1.35` |

### Weighted geometry

| slice | Run K weighted | Run L weighted |
| --- | --- | --- |
| total refusal | 1662.785 | 1729.76 |
| total answerable | 14204.3825 | 15456.184 |
| 5-doc refusal | 1653.685 | 1719.96 |
| 5-doc answerable | 6235.5271 | 7305.6036 |

Interpretation:

- Run L pushes the short answerable boundary a little further than Run K.
- The extra pressure is now more balanced, because some of that new mass is explicitly short `No conflict`, not only conflict-bearing answerables.

Source: [data/messages/train_stagewise_prompt_robust_trace_text_l_boundary_rebalanced_weight_summary.json](../../data/messages/train_stagewise_prompt_robust_trace_text_l_boundary_rebalanced_weight_summary.json)

## Training recipe

Run L intentionally kept the same core training hyperparameters as Run J so the main variable remained the data/mixture design.

### Qwen launch scripts

| model | script |
| --- | --- |
| Qwen2.5 7B | [slurm/sharanga/examples/qwen7b_stagewise_ddp_2h100_prompt_robust_l_boundary_rebalanced.sh](../../slurm/sharanga/examples/qwen7b_stagewise_ddp_2h100_prompt_robust_l_boundary_rebalanced.sh) |
| Qwen2.5 32B | [slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_l_boundary_rebalanced.sh](../../slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_l_boundary_rebalanced.sh) |

### Additional small-model launches

| model | script |
| --- | --- |
| Llama 3.1 8B | [slurm/sharanga/examples/llama8b_stagewise_1gpu_prompt_robust_l_boundary_rebalanced.sh](../../slurm/sharanga/examples/llama8b_stagewise_1gpu_prompt_robust_l_boundary_rebalanced.sh) |
| Mistral 7B | [slurm/sharanga/examples/mistral7b_stagewise_1gpu_prompt_robust_l_boundary_rebalanced.sh](../../slurm/sharanga/examples/mistral7b_stagewise_1gpu_prompt_robust_l_boundary_rebalanced.sh) |

### Core hyperparameters

| setting | 7B | 32B |
| --- | --- | --- |
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

## Expected win condition

Run L is only worth keeping if it does all of the following at once:

1. keep Run K-like benchmark abstention strength
2. avoid reintroducing high false-abstain counts
3. preserve or improve conflict-type accuracy, especially `No conflict` vs `Complementary information`
4. preserve or improve doc-verdict accuracy
5. avoid sacrificing lexical overlap and citation discipline

If it improves abstention but hurts the rest, it is not a true upgrade.

## Current artifact status

This repository contains the complete Run L recipe and the completed Llama 3.1 8B and Mistral 7B benchmark matrices.

### Present locally

- split builder
- split summary
- weight summary
- rebuild script
- Qwen launch scripts
- 1-GPU Llama/Mistral launch scripts

### Completed stored result artifacts

- `final_model_outputs/llama8b/` contains the full 24-row baseline/SFT Run L benchmark matrix and per-example evaluations.
- `final_model_outputs/mistral7b/` contains the full 24-row baseline/SFT Run L benchmark matrix and per-example evaluations.
- The matrix-level analyses are `final_model_outputs/llama8b_benchmark_matrix_analysis.md` and `final_model_outputs/mistral7b_benchmark_matrix_analysis.md`.

The latest stored Qwen matrix remains Run K; Run L retains the Qwen training launchers and message recipe, but no completed Qwen Run L result matrix was present locally at cleanup time.

## Stored artifacts

- Split builder: [scripts/prepare_run_l_splits.py](../../scripts/prepare_run_l_splits.py)
- Split summary: [data/splits/run_l/run_l_split_summary.json](../../data/splits/run_l/run_l_split_summary.json)
- Weight summary: [data/messages/train_stagewise_prompt_robust_trace_text_l_boundary_rebalanced_weight_summary.json](../../data/messages/train_stagewise_prompt_robust_trace_text_l_boundary_rebalanced_weight_summary.json)
- Rebuild script: [slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh](../../slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh)
- Qwen launch wrapper: [slurm/sharanga/examples/launch_qwen_run_l_boundary_rebalanced.sh](../../slurm/sharanga/examples/launch_qwen_run_l_boundary_rebalanced.sh)
