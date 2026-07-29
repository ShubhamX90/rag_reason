#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_ROOT/slurm/common_env.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_LABEL="${DATASET_LABEL:-benchmark_final}"
BENCHMARK_INPUT="${BENCHMARK_INPUT:-data/Benchmark Dataset/benchmark_final.jsonl}"
MODELS="${MODELS:-qwen25 llama31 mistral7b}"
MODEL_VARIANTS="${MODEL_VARIANTS:-sft baseline}"
PROMPT_MODES="${PROMPT_MODES:-e2e}"
PROMPT_PROFILES="${PROMPT_PROFILES:-default runtime minimal}"
MESSAGE_TAGS="${MESSAGE_TAGS:-strict trace_text minimal}"
CONTRACT_MODES="${CONTRACT_MODES:-trace trace none}"
RETRY_ATTEMPTS_LIST="${RETRY_ATTEMPTS_LIST:-1 1 0}"
MAX_NEW_BASE_LIST="${MAX_NEW_BASE_LIST:-1400 1200 900}"
MAX_NEW_CAP_LIST="${MAX_NEW_CAP_LIST:-3200 2200 1800}"
DTYPE="${DTYPE:-bf16}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
RESUME="${RESUME:-0}"

echo "===== Preflight: benchmark message build ====="
if [ "$DATASET_LABEL" = "benchmark_final" ]; then
  PYTHON_BIN="$PYTHON_BIN" \
  BENCHMARK_INPUT="$BENCHMARK_INPUT" \
  DATASET_LABEL="$DATASET_LABEL" \
  bash slurm/examples/rebuild_benchmark_messages_f_boundary_guarded.sh
elif [ "$DATASET_LABEL" = "val_stagewise" ]; then
  PYTHON_BIN="$PYTHON_BIN" \
  DATASET_LABEL="$DATASET_LABEL" \
  bash slurm/examples/rebuild_val_messages_f_boundary_guarded.sh
else
  echo "Unsupported DATASET_LABEL for preflight build: $DATASET_LABEL" >&2
  exit 1
fi

submit_generation() {
  local model_variant="$1"
  local model_name="$2"
  local base_model="$3"
  local lora_dir="$4"
  local run_name="$5"
  local prompt_mode="$6"
  local profile="$7"
  local message_tag="$8"
  local contract_mode="$9"
  local retry_attempts="${10}"
  local max_new_base="${11}"
  local max_new_cap="${12}"

  echo "===== submit ${model_variant} ${model_name} ${prompt_mode} ${profile} ====="
  local env_args=(
    MODEL_NAME="$model_name"
    BASE_MODEL="$base_model"
    TRAIN_STRATEGY=stagewise
    DATASET_LABEL="$DATASET_LABEL"
    PROMPT_MODE="$prompt_mode"
    PROMPT_PROFILE="$profile"
    MESSAGE_TAG="$message_tag"
    MODEL_VARIANT="$model_variant"
    RUN_NAME="$run_name"
    DTYPE="$DTYPE"
    LOAD_IN_4BIT="$LOAD_IN_4BIT"
    RESUME="$RESUME"
    CONTRACT_MODE="$contract_mode"
    RETRY_ATTEMPTS="$retry_attempts"
    MAX_NEW_TOKENS_BASE="$max_new_base"
    MAX_NEW_TOKENS_CAP="$max_new_cap"
  )
  if [ "$model_variant" = "sft" ]; then
    env_args+=(LORA_DIR="$lora_dir")
  fi
  env "${env_args[@]}" sbatch slurm/generate_experiment.sh
}

read -r -a prompt_modes <<< "$PROMPT_MODES"
read -r -a prompt_profiles <<< "$PROMPT_PROFILES"
read -r -a message_tags <<< "$MESSAGE_TAGS"
read -r -a contract_modes <<< "$CONTRACT_MODES"
read -r -a retry_attempts_list <<< "$RETRY_ATTEMPTS_LIST"
read -r -a max_new_base_list <<< "$MAX_NEW_BASE_LIST"
read -r -a max_new_cap_list <<< "$MAX_NEW_CAP_LIST"

num_profiles="${#prompt_profiles[@]}"
if [ "${#message_tags[@]}" -ne "$num_profiles" ] || \
   [ "${#contract_modes[@]}" -ne "$num_profiles" ] || \
   [ "${#retry_attempts_list[@]}" -ne "$num_profiles" ] || \
   [ "${#max_new_base_list[@]}" -ne "$num_profiles" ] || \
   [ "${#max_new_cap_list[@]}" -ne "$num_profiles" ]; then
  echo "PROMPT_PROFILES, MESSAGE_TAGS, CONTRACT_MODES, RETRY_ATTEMPTS_LIST, MAX_NEW_BASE_LIST, and MAX_NEW_CAP_LIST must have the same number of entries." >&2
  exit 1
fi

for model_name in $MODELS; do
  case "$model_name" in
    qwen25)
      base_model="${MODEL_ROOT}/Qwen2.5-7B-Instruct"
      lora_dir="checkpoints/qwen25_stagewise_e2e_main_trace_text_f_boundary_guarded_csis/best_dev_f1"
      ;;
    llama31)
      base_model="${MODEL_ROOT}/Llama-3.1-8B-Instruct"
      lora_dir="checkpoints/llama31_stagewise_e2e_main_trace_text_f_boundary_guarded_csis/best_dev_f1"
      ;;
    mistral7b)
      base_model="${MODEL_ROOT}/Mistral-7B-Instruct-v0.3"
      lora_dir="checkpoints/mistral7b_stagewise_e2e_main_trace_text_f_boundary_guarded_csis/best_dev_f1"
      ;;
    *)
      echo "Unknown model name: $model_name" >&2
      exit 1
      ;;
  esac

  test -d "$base_model"
  for model_variant in $MODEL_VARIANTS; do
    run_name="base"
    if [ "$model_variant" = "sft" ]; then
      test -d "$lora_dir"
      run_name="main_trace_text_f_boundary_guarded_csis"
    fi

    for prompt_mode in "${prompt_modes[@]}"; do
      for i in "${!prompt_profiles[@]}"; do
        submit_generation \
          "$model_variant" \
          "$model_name" \
          "$base_model" \
          "$lora_dir" \
          "$run_name" \
          "$prompt_mode" \
          "${prompt_profiles[$i]}" \
          "${message_tags[$i]}" \
          "${contract_modes[$i]}" \
          "${retry_attempts_list[$i]}" \
          "${max_new_base_list[$i]}" \
          "${max_new_cap_list[$i]}"
      done
    done
  done
done
