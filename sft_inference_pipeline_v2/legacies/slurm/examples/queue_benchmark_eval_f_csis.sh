#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_ROOT/slurm/common_env.sh"
cd "$PROJECT_ROOT"

DATASET_LABEL="${DATASET_LABEL:-benchmark_final}"
MODELS="${MODELS:-qwen25 llama31 mistral7b}"
MODEL_VARIANTS="${MODEL_VARIANTS:-sft baseline}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
GEN_JOB_IDS="${GEN_JOB_IDS:-}"
PROMPT_MODES="${PROMPT_MODES:-e2e}"
PROMPT_PROFILES="${PROMPT_PROFILES:-default runtime minimal}"
MESSAGE_TAGS="${MESSAGE_TAGS:-strict trace_text minimal}"

if [ -z "$GEN_JOB_IDS" ]; then
  echo "GEN_JOB_IDS is required. Pass generation job ids in the exact launcher submission order." >&2
  exit 1
fi

read -r -a job_ids <<< "$GEN_JOB_IDS"
read -r -a prompt_modes <<< "$PROMPT_MODES"
read -r -a profiles <<< "$PROMPT_PROFILES"
read -r -a message_tags <<< "$MESSAGE_TAGS"

if [ "${#profiles[@]}" -ne "${#message_tags[@]}" ]; then
  echo "PROMPT_PROFILES and MESSAGE_TAGS must have the same number of entries." >&2
  exit 1
fi

expected_count=0
for _model_name in $MODELS; do
  for _variant in $MODEL_VARIANTS; do
    expected_count=$((expected_count + ${#prompt_modes[@]} * ${#profiles[@]}))
  done
done

if [ "${#job_ids[@]}" -ne "$expected_count" ]; then
  echo "Expected $expected_count generation job ids, got ${#job_ids[@]}." >&2
  exit 1
fi

submit_eval() {
  local gen_job_id="$1"
  local model_variant="$2"
  local model_name="$3"
  local run_name="$4"
  local prompt_mode="$5"
  local profile="$6"
  local message_tag="$7"

  echo "===== queue eval afterok:${gen_job_id} :: ${model_variant} ${model_name} ${prompt_mode} ${profile} ====="
  env \
    MODEL_NAME="$model_name" \
    RUN_NAME="$run_name" \
    TRAIN_STRATEGY="$TRAIN_STRATEGY" \
    DATASET_LABEL="$DATASET_LABEL" \
    PROMPT_MODE="$prompt_mode" \
    PROMPT_PROFILE="$profile" \
    MESSAGE_TAG="$message_tag" \
    MODEL_VARIANT="$model_variant" \
    sbatch --dependency="afterok:${gen_job_id}" slurm/evaluate_experiment.sh
}

idx=0
for model_name in $MODELS; do
  for model_variant in $MODEL_VARIANTS; do
    run_name="base"
    if [ "$model_variant" = "sft" ]; then
      run_name="main_trace_text_f_boundary_guarded_csis"
    fi

    for prompt_mode in "${prompt_modes[@]}"; do
      for i in "${!profiles[@]}"; do
        submit_eval "${job_ids[$idx]}" "$model_variant" "$model_name" "$run_name" "$prompt_mode" "${profiles[$i]}" "${message_tags[$i]}"
        idx=$((idx + 1))
      done
    done
  done
done
