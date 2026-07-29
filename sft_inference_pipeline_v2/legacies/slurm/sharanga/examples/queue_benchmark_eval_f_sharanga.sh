#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"
cd "$PROJECT_ROOT"

DATASET_LABEL="${DATASET_LABEL:-benchmark_final}"
TRACE_MODELS="${TRACE_MODELS:-qwen25_32b qwen25 llama31 mistral7b}"
TRACE_MODEL_VARIANTS="${TRACE_MODEL_VARIANTS:-baseline sft}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
GEN_JOB_IDS="${GEN_JOB_IDS:-}"
TRACE_PROMPT_MODES="${TRACE_PROMPT_MODES:-e2e oracle_conflict oracle_notes oracle_both}"
TRACE_PROMPT_PROFILE="${TRACE_PROMPT_PROFILE:-runtime}"
TRACE_MESSAGE_TAG="${TRACE_MESSAGE_TAG:-trace_text}"
ANSWER_ONLY_MODELS="${ANSWER_ONLY_MODELS:-qwen25_32b qwen25}"
ANSWER_ONLY_PROMPT_MODE="${ANSWER_ONLY_PROMPT_MODE:-e2e}"
ANSWER_ONLY_PROMPT_PROFILE="${ANSWER_ONLY_PROMPT_PROFILE:-final_only}"
ANSWER_ONLY_MESSAGE_TAG="${ANSWER_ONLY_MESSAGE_TAG:-final_only}"

if [ -z "$GEN_JOB_IDS" ]; then
  echo "GEN_JOB_IDS is required. Pass generation job ids in the exact launcher submission order." >&2
  exit 1
fi

read -r -a job_ids <<< "$GEN_JOB_IDS"
read -r -a trace_models <<< "$TRACE_MODELS"
read -r -a trace_model_variants <<< "$TRACE_MODEL_VARIANTS"
read -r -a trace_prompt_modes <<< "$TRACE_PROMPT_MODES"
read -r -a answer_only_models <<< "$ANSWER_ONLY_MODELS"

expected_count=0
for _model_name in "${trace_models[@]}"; do
  for _variant in "${trace_model_variants[@]}"; do
    expected_count=$((expected_count + ${#trace_prompt_modes[@]}))
  done
done
expected_count=$((expected_count + ${#answer_only_models[@]}))

if [ "${#job_ids[@]}" -ne "$expected_count" ]; then
  echo "Expected $expected_count generation job ids, got ${#job_ids[@]}." >&2
  exit 1
fi

trace_run_name_for() {
  case "$1" in
    qwen25_32b) echo "main_trace_text_f_boundary_guarded" ;;
    qwen25|llama31|mistral7b) echo "main_trace_text_f_boundary_guarded_csis" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

answer_only_run_name_for() {
  case "$1" in
    qwen25_32b|qwen25) echo "main_answer_only_matched_f" ;;
    *) echo "Unknown answer-only model alias: $1" >&2; return 1 ;;
  esac
}

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
    sbatch --dependency="afterok:${gen_job_id}" slurm/sharanga/evaluate_experiment.sh
}

idx=0
for model_name in "${trace_models[@]}"; do
  for model_variant in "${trace_model_variants[@]}"; do
    run_name="base"
    if [ "$model_variant" = "sft" ]; then
      run_name="$(trace_run_name_for "$model_name")"
    fi

    for prompt_mode in "${trace_prompt_modes[@]}"; do
      submit_eval "${job_ids[$idx]}" "$model_variant" "$model_name" "$run_name" "$prompt_mode" "$TRACE_PROMPT_PROFILE" "$TRACE_MESSAGE_TAG"
      idx=$((idx + 1))
    done
  done
done

for model_name in "${answer_only_models[@]}"; do
  submit_eval \
    "${job_ids[$idx]}" \
    sft \
    "$model_name" \
    "$(answer_only_run_name_for "$model_name")" \
    "$ANSWER_ONLY_PROMPT_MODE" \
    "$ANSWER_ONLY_PROMPT_PROFILE" \
    "$ANSWER_ONLY_MESSAGE_TAG"
  idx=$((idx + 1))
done
