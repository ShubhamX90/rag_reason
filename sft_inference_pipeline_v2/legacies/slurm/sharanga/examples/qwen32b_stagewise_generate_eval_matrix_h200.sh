#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_ROOT"

export MODEL_NAME="${MODEL_NAME:-qwen25_32b}"
export BASE_MODEL="${BASE_MODEL:-/scratch/$USER/rag-reason/models/Qwen2.5-32B-Instruct}"
export TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
export DATASET_LABEL="${DATASET_LABEL:-val_stagewise}"
export PROMPT_MODE="${PROMPT_MODE:-e2e}"
export TRACE_CONTRACT_MODE="${TRACE_CONTRACT_MODE:-trace}"
export MINIMAL_CONTRACT_MODE="${MINIMAL_CONTRACT_MODE:-none}"
export DTYPE="${DTYPE:-bf16}"
export LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export RESUME="${RESUME:-0}"

if [ "${REBUILD_MINIMAL:-1}" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_minimal_inference.sh
  "$PYTHON_BIN" scripts/check_trace_text_messages.py --require_think data/messages/val_stagewise_e2e_minimal_messages.jsonl
fi

submit_pair() {
  local label="$1"
  local variant="$2"
  local run_name="$3"
  local prompt_profile="$4"
  local lora_dir="${5:-}"
  local contract_mode="$TRACE_CONTRACT_MODE"
  local retry_attempts="${RETRY_ATTEMPTS:-1}"
  local max_new_base="${MAX_NEW_TOKENS_BASE:-1200}"
  local max_new_cap="${MAX_NEW_TOKENS_CAP:-2200}"

  if [ "$prompt_profile" = "minimal" ]; then
    contract_mode="$MINIMAL_CONTRACT_MODE"
    retry_attempts="${MINIMAL_RETRY_ATTEMPTS:-0}"
    max_new_base="${MINIMAL_MAX_NEW_TOKENS_BASE:-700}"
    max_new_cap="${MINIMAL_MAX_NEW_TOKENS_CAP:-1400}"
  fi

  echo "===== submit $label ====="
  local gen_job
  gen_job="$(
    MODEL_VARIANT="$variant" \
    RUN_NAME="$run_name" \
    PROMPT_PROFILE="$prompt_profile" \
    CONTRACT_MODE="$contract_mode" \
    RETRY_ATTEMPTS="$retry_attempts" \
    MAX_NEW_TOKENS_BASE="$max_new_base" \
    MAX_NEW_TOKENS_CAP="$max_new_cap" \
    LORA_DIR="$lora_dir" \
    sbatch \
      --partition="${PARTITION:-gpu_h200_8}" \
      --nodelist="${NODELIST:-gpunode7}" \
      slurm/sharanga/generate_experiment.sh |
    awk '{print $4}'
  )"
  echo "$label generate job: $gen_job"

  MODEL_VARIANT="$variant" \
  RUN_NAME="$run_name" \
  PROMPT_PROFILE="$prompt_profile" \
  sbatch --dependency="afterok:$gen_job" slurm/sharanga/evaluate_experiment.sh
}

# Needed now:
# - baseline under runtime-style trace_text and true minimal prompts
# - SFT B/C under true minimal prompts
# SFT B/C trace_text runtime outputs already exist from the previous runs.
submit_pair "baseline-trace_text" "baseline" "base" "trace_text"
submit_pair "baseline-minimal" "baseline" "base" "minimal"
submit_pair "sft-b-minimal" "sft" "main_trace_text_b" "minimal" \
  "/scratch/$USER/rag-reason/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_b/best_dev_f1"
submit_pair "sft-c-minimal" "sft" "main_trace_text_c_conflict_focus" "minimal" \
  "/scratch/$USER/rag-reason/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_c_conflict_focus/best_dev_f1"
