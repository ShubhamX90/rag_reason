# Reviewer reproduction guide

## Scope and honest artifact status

This repository contains the current code, prompts, train/validation/benchmark data, generated SFT messages, evaluation code, SLURM launchers, and all locally stored final outputs. It does **not** contain base-model weights or the latest LoRA adapters used to generate the reported matrices.

| Result family | Recipe/data available | Exact stored-adapter inference available | Stored evaluated outputs |
| --- | --- | --- | --- |
| Run K Qwen 7B/32B | Yes | No; provide `SFT_LORA_DIR` | Yes, 24 rows/model |
| Run L Llama 8B/Mistral 7B | Yes | No; train or provide adapter | Yes, 24 rows/model |
| Updated answer-only Qwen 7B/Llama 8B/Mistral 7B | Yes | No; train or provide adapter | Yes, final-only + minimal/model |

The included outputs are complete evaluation artifacts, not substitutes for an adapter release. Do not use archived historical adapters in `legacies/` to claim exact reproduction of the current results.

## Standalone answer-only release

[`answer_only_sft_export/`](../answer_only_sft_export/) is an active, self-contained release surface rather than a legacy snapshot. Its own README and reproduction guide document both the established updated answer-only workflow and the clean basic 862-example answer-only baseline. Use it when the answer-only pipeline is being reviewed or shared independently; its portable archive is [`answer_only_sft_export.zip`](../answer_only_sft_export.zip).

## 1. Prerequisites

The retained launchers target a SLURM GPU cluster with Conda, CUDA 12.4-compatible PyTorch wheels, and Hugging Face access for the base models. Put the required base models under a directory of your choice:

```bash
export PROJECT_ROOT="$(pwd)"
export WORK_ROOT="/path/to/writable/workspace"
export MODEL_ROOT="/path/to/models"
export SCRATCH="/path/to/scratch"             # Required by the Sharanga helper.
```

Expected model directory names are:

- `Qwen2.5-7B-Instruct`
- `Qwen2.5-32B-Instruct`
- `Llama-3.1-8B-Instruct`
- `Mistral-7B-Instruct-v0.3`

Create the pinned environment:

```bash
INSTALL_MODE=frozen bash scripts/bootstrap_sharanga_env.sh
```

The default frozen install uses `env/sharanga-working-freeze.txt`. `INSTALL_MODE=compatible` retains the previous version-range installation behavior and is not the exact-environment path.

## 2. Verify the data and message contracts

```bash
python3 scripts/check_trace_text_messages.py \
  data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_messages.jsonl \
  data/messages/train_stagewise_prompt_robust_trace_text_l_boundary_rebalanced_messages.jsonl

python3 scripts/check_trace_text_messages.py --forbid_think --forbid_task_prefix \
  data/messages/train_stagewise_answer_only_matched_f_messages.jsonl \
  data/messages/val_stagewise_answer_only_final_only_messages.jsonl \
  data/messages/val_stagewise_answer_only_minimal_messages.jsonl
```

Expected row counts are 12,659 for Run K, 13,349 for Run L, and 10,344 for the weighted answer-only mixture. The canonical held-out benchmark is `data/splits/benchmark_final_v2_holdout_clean_736.jsonl`.

## 3. Rebuild the training messages

These commands regenerate the retained data/message products in place. Run them from a clean checkout or a dedicated working copy if you want to compare generated files against the stored artifacts.

```bash
bash slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh
bash slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh
bash slurm/examples/rebuild_messages_answer_only_matched_f.sh
```

The K/L builders temporarily restore the 862/81 compatibility split files from `data/splits/run_j/` when they exit. This is intentional and preserves the current base split.

## 4. Train and evaluate

Run L has retained dedicated training launchers for Qwen, Llama, and Mistral. The updated answer-only Qwen 7B launcher is `slurm/sharanga/examples/launch_answer_only_updated_split_qwen7b.sh`. For updated answer-only Llama/Mistral, use the retained small-model launcher with an explicit current result-family name, for example:

```bash
MODELS=llama31 \
RUN_NAME=main_answer_only_updated_split_llama8b_20260701_r1 \
bash slurm/sharanga/examples/launch_answer_only_matched_f_small_models.sh
```

The filename `matched_f` is historical; the launcher rebuilds and trains on the current 862/81 updated answer-only mixture. It is retained for compatibility and must be invoked with an explicit `RUN_NAME` when reproducing the updated small-model family.

Run K's retained Qwen scripts reproduce its benchmark matrix from a supplied adapter:

```bash
SFT_LORA_DIR=/path/to/qwen7b-run-k/best_dev_f1 \
  bash slurm/sharanga/examples/launch_qwen7b_run_k_benchmark_matrix.sh
```

The original dedicated Run K Qwen training wrapper was not present in the working repository. The K split builder, message builder, generic DDP trainer, and exact evaluation launcher are retained, but this release does not assert that a newly reconstructed training submission is byte-for-byte identical to the original remote job.

## 5. Compare with supplied artifacts

Each full matrix contains 24 baseline/SFT, prompt-mode/profile combinations and per-example files covering all 736 benchmark IDs:

- `final_model_outputs/qwen7b/`
- `final_model_outputs/qwen32b/`
- `final_model_outputs/llama8b/`
- `final_model_outputs/mistral7b/`

Updated answer-only outputs are in `final_model_outputs/answer_only_sft/` and contain final-only and minimal evaluation reports for Qwen 7B, Llama 8B, and Mistral 7B.

## Determinism note

The repository makes data and prompt regeneration deterministic at the file level, but complete training reproduction still depends on external base-model revisions, CUDA/driver versions, distributed training behavior, and the unavailable current adapters. Reported outputs should therefore be treated as supplied evaluation artifacts unless the authors separately release the corresponding adapters and model revision hashes.
