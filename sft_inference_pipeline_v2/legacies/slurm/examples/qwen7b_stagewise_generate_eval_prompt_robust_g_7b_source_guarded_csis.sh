#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_ROOT/slurm/common_env.sh"
cd "$PROJECT_ROOT"

export MODEL_NAME="${MODEL_NAME:-qwen25}"
export BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT:-/nfs_home/users/vsshekhawat/projects/rag-reason/models}/Qwen2.5-7B-Instruct}"
export TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
export DATASET_LABEL="${DATASET_LABEL:-val_stagewise}"
export PROMPT_MODE="${PROMPT_MODE:-e2e}"
export MODEL_VARIANT="${MODEL_VARIANT:-sft}"
export RUN_NAME="${RUN_NAME:-main_trace_text_g_7b_source_guarded_csis}"
export LORA_DIR="${LORA_DIR:-checkpoints/qwen25_stagewise_e2e_main_trace_text_g_7b_source_guarded_csis/best_dev_f1}"
export DTYPE="${DTYPE:-bf16}"
export LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export RESUME="${RESUME:-0}"

if [ "${REBUILD_MESSAGES:-1}" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_prompt_robust_g_7b_source_guarded.sh
fi

submit_eval() {
  local label="$1"
  local prompt_profile="$2"
  local message_tag="$3"
  local contract_mode="$4"
  local retry_attempts="$5"
  local max_new_base="$6"
  local max_new_cap="$7"

  echo "===== submit $label ====="
  local gen_job
  gen_job="$(
    PROMPT_PROFILE="$prompt_profile" \
    MESSAGE_TAG="$message_tag" \
    CONTRACT_MODE="$contract_mode" \
    RETRY_ATTEMPTS="$retry_attempts" \
    MAX_NEW_TOKENS_BASE="$max_new_base" \
    MAX_NEW_TOKENS_CAP="$max_new_cap" \
    sbatch slurm/generate_experiment.sh |
    awk '{print $4}'
  )"
  echo "$label generate job: $gen_job"

  PROMPT_PROFILE="$prompt_profile" \
  MESSAGE_TAG="$message_tag" \
  sbatch --dependency="afterok:$gen_job" slurm/evaluate_experiment.sh
}

submit_eval "sft-g7-strict" "default" "strict" "trace" "${STRICT_RETRY_ATTEMPTS:-1}" "${STRICT_MAX_NEW_TOKENS_BASE:-1400}" "${STRICT_MAX_NEW_TOKENS_CAP:-3200}"
submit_eval "sft-g7-runtime" "runtime" "trace_text" "trace" "${TRACE_RETRY_ATTEMPTS:-1}" "${TRACE_MAX_NEW_TOKENS_BASE:-1200}" "${TRACE_MAX_NEW_TOKENS_CAP:-2200}"
submit_eval "sft-g7-minimal" "minimal" "" "none" "${MINIMAL_RETRY_ATTEMPTS:-0}" "${MINIMAL_MAX_NEW_TOKENS_BASE:-900}" "${MINIMAL_MAX_NEW_TOKENS_CAP:-1800}"
