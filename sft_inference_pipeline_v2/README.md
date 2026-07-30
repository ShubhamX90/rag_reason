# RAG Reasoning SFT + Inference Pipeline v2

This is the cleaned, current-facing reproduction repository. The root contains only the latest retained workflows, their direct inputs, and their final result artifacts. Historical runs, superseded scripts, stale checkpoints, old exports, logs, and duplicate snapshots are preserved under [`legacies/`](legacies/README.md), which is not needed to inspect or run the current workflows.

Start with [the reviewer reproduction guide](docs/REPRODUCIBILITY.md). It identifies what can be regenerated from this repository, what requires an externally supplied model/adapter, and the exact environment and path configuration expected by the cluster launchers.

## Current workflow map

| Workflow | Current recipe | Stored final results |
| --- | --- | --- |
| Trace-text SFT — Run K | Qwen short-context targeted recipe | `final_model_outputs/qwen7b/`, `final_model_outputs/qwen32b/` |
| Trace-text SFT — Run L | boundary-rebalanced recipe | `final_model_outputs/llama8b/`, `final_model_outputs/mistral7b/` |
| Answer-only SFT | updated 862/81 split, final-only + minimal mixture | `final_model_outputs/answer_only_sft/` |

The self-contained, active answer-only release is [`answer_only_sft_export/`](answer_only_sft_export/). It contains the documented basic 862-example answer-only baseline, its standalone runners, validation, and retained artifacts. A portable copy is available as [`answer_only_sft_export.zip`](answer_only_sft_export.zip).

The current held-out benchmark is `data/splits/benchmark_final_v2_holdout_clean_736.jsonl` (736 examples). Current train/validation backbone is 862/81; Run K adds 27 derived rows (889 train) and Run L adds 48 derived rows (910 train).

## Reproduce the current recipes

Run commands from the repository root. The launchers now discover this checkout automatically; an explicit `PROJECT_ROOT` always takes precedence. Before submitting a cluster job, configure its storage locations and install the pinned environment:

```bash
export PROJECT_ROOT="$(pwd)"
export WORK_ROOT="/path/to/workspace"
export MODEL_ROOT="/path/to/downloaded-models"
INSTALL_MODE=frozen bash scripts/bootstrap_sharanga_env.sh
```

`INSTALL_MODE=frozen` is the default and installs `env/sharanga-working-freeze.txt`. Use `INSTALL_MODE=compatible` only for an existing environment where the broader version ranges are intentionally desired.

### Run L trace-text recipe

```bash
bash slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh
bash slurm/sharanga/examples/launch_qwen_run_l_boundary_rebalanced.sh

# Small-model training and their completed benchmark matrices
bash slurm/sharanga/examples/llama8b_stagewise_1gpu_prompt_robust_l_boundary_rebalanced.sh
bash slurm/sharanga/examples/mistral7b_stagewise_1gpu_prompt_robust_l_boundary_rebalanced.sh
bash slurm/sharanga/examples/launch_llama_run_l_benchmark_matrix.sh
bash slurm/sharanga/examples/launch_mistral_run_l_benchmark_matrix.sh
```

### Run K Qwen benchmark recipe

```bash
bash slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh
SFT_LORA_DIR=/path/to/run-k-adapter/best_dev_f1 \
  bash slurm/sharanga/examples/launch_qwen7b_run_k_benchmark_matrix.sh
```

The 32B matrix uses its own `SFT_LORA_DIR` and launcher. The retained launchers reproduce evaluation from an adapter; the original dedicated Run K Qwen training wrapper was not available locally and is not represented as an exact runnable command here.

### Answer-only SFT recipe

```bash
bash slurm/examples/rebuild_messages_answer_only_matched_f.sh
bash slurm/sharanga/examples/launch_answer_only_updated_split_qwen7b.sh
```

The updated answer-only message mixture is `data/messages/train_stagewise_answer_only_matched_f_messages.jsonl` (10,344 weighted rows) with 81-example final-only and minimal validation sets. The legacy filename is retained because the active launcher and prior result naming use it; its contents are the updated 862/81-split recipe.

## Verification and results

- Current data audits: `docs/dataset_audits/`
- Current K/L experiment notes: `docs/experiments/`
- Final matrices and per-example artifacts: `final_model_outputs/`
- Message-contract checker: `scripts/check_trace_text_messages.py`
- Reproduction and external-artifact requirements: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Method and checkpoint-selection disclosure: [docs/METHOD_LIMITATIONS.md](docs/METHOD_LIMITATIONS.md)
- Public-release checklist: [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md)

For a current training run, pass the resulting adapter path explicitly to the relevant benchmark launcher. The completed K/L and answer-only result artifacts are stored here, but the corresponding latest remote/scratch adapters were not present locally at cleanup time. The old local adapters are historical and therefore archived; they must not be substituted for the current reported runs.

## Legacy archive

`legacies/` preserves every relocated file with its SHA-256 in `legacies/MANIFEST_2026-07-29.jsonl`. Removing that one directory yields the clean submission/review surface without deleting historical provenance from this working copy.
