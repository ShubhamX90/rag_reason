#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"
cd "$PROJECT_ROOT"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-$SHARANGA_JOB_PREFIX}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_LABEL="${DATASET_LABEL:-benchmark_final}"
BENCHMARK_INPUT="${BENCHMARK_INPUT:-data/Benchmark Dataset/benchmark_final_sanitized.jsonl}"
REBUILD_MESSAGES="${REBUILD_MESSAGES:-1}"

TRACE_MODELS="${TRACE_MODELS:-qwen25_32b qwen25 llama31 mistral7b}"
TRACE_MODEL_VARIANTS="${TRACE_MODEL_VARIANTS:-baseline sft}"
TRACE_PROMPT_MODES="${TRACE_PROMPT_MODES:-e2e oracle_conflict oracle_notes oracle_both}"
TRACE_PROMPT_PROFILE="${TRACE_PROMPT_PROFILE:-runtime}"
TRACE_MESSAGE_TAG="${TRACE_MESSAGE_TAG:-trace_text}"
TRACE_CONTRACT_MODE="${TRACE_CONTRACT_MODE:-trace}"
TRACE_RETRY_ATTEMPTS="${TRACE_RETRY_ATTEMPTS:-1}"
TRACE_MAX_NEW_BASE="${TRACE_MAX_NEW_BASE:-1200}"
TRACE_MAX_NEW_CAP="${TRACE_MAX_NEW_CAP:-2200}"
TRACE_PROFILE_SPECS="${TRACE_PROFILE_SPECS:-}"

ANSWER_ONLY_MODELS="${ANSWER_ONLY_MODELS:-qwen25_32b qwen25}"
ANSWER_ONLY_PROMPT_MODE="${ANSWER_ONLY_PROMPT_MODE:-e2e}"
ANSWER_ONLY_PROMPT_PROFILE="${ANSWER_ONLY_PROMPT_PROFILE:-final_only}"
ANSWER_ONLY_MESSAGE_TAG="${ANSWER_ONLY_MESSAGE_TAG:-final_only}"
ANSWER_ONLY_CONTRACT_MODE="${ANSWER_ONLY_CONTRACT_MODE:-final}"
ANSWER_ONLY_RETRY_ATTEMPTS="${ANSWER_ONLY_RETRY_ATTEMPTS:-1}"
ANSWER_ONLY_MAX_NEW_BASE="${ANSWER_ONLY_MAX_NEW_BASE:-500}"
ANSWER_ONLY_MAX_NEW_CAP="${ANSWER_ONLY_MAX_NEW_CAP:-1200}"
RUN_TRACE="${RUN_TRACE:-1}"
RUN_ANSWER_ONLY="${RUN_ANSWER_ONLY:-1}"

DTYPE="${DTYPE:-bf16}"
RESUME="${RESUME:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

LARGE_PARTITION="${LARGE_PARTITION:-gpu_h200_8}"
LARGE_NODELIST="${LARGE_NODELIST:-}"
LARGE_TIME="${LARGE_TIME:-23:30:00}"
SMALL_H100_PARTITION="${SMALL_H100_PARTITION:-gpu_h100_4}"
SMALL_H100_NODELIST="${SMALL_H100_NODELIST:-}"
SMALL_H100_TIME="${SMALL_H100_TIME:-20:00:00}"
SMALL_H100_CPUS="${SMALL_H100_CPUS:-2}"
SMALL_A100_PARTITION="${SMALL_A100_PARTITION:-gpu_a100_8}"
SMALL_A100_NODELIST="${SMALL_A100_NODELIST:-}"
SMALL_A100_TIME="${SMALL_A100_TIME:-24:00:00}"
SMALL_A100_CPUS="${SMALL_A100_CPUS:-4}"
H100_MODELS="${H100_MODELS:-}"
V100_MODELS="${V100_MODELS:-}"
V100_PARTITION="${V100_PARTITION:-gpu_v100_2}"
V100_NODELIST="${V100_NODELIST:-}"
V100_TIME="${V100_TIME:-20:00:00}"
V100_CPUS="${V100_CPUS:-4}"
V100_DTYPE="${V100_DTYPE:-fp16}"
LARGE_CPUS="${LARGE_CPUS:-4}"
EVAL_PARTITION="${EVAL_PARTITION:-compute}"
EVAL_TIME="${EVAL_TIME:-02:00:00}"

SUBMISSION_LOG="${SUBMISSION_LOG:-logs/benchmark_sharanga_submissions.tsv}"

mkdir -p "$(dirname "$SUBMISSION_LOG")"
if [ ! -f "$SUBMISSION_LOG" ]; then
  printf "kind\tmodel\tvariant\tprompt_mode\tprompt_profile\tmessage_tag\tgen_job_id\teval_job_id\tout_sanitized\n" > "$SUBMISSION_LOG"
fi

echo "===== Preflight: benchmark message build ====="
if [ "$REBUILD_MESSAGES" = "1" ] && [ "$DATASET_LABEL" = "benchmark_final" ]; then
  PYTHON_BIN="$PYTHON_BIN" \
  BENCHMARK_INPUT="$BENCHMARK_INPUT" \
  DATASET_LABEL="$DATASET_LABEL" \
  bash slurm/examples/rebuild_benchmark_messages_f_boundary_guarded.sh
elif [ "$DATASET_LABEL" != "benchmark_final" ]; then
  echo "Unsupported DATASET_LABEL for preflight build: $DATASET_LABEL" >&2
  exit 1
fi

trace_run_name_for() {
  case "$1" in
    qwen25_32b) echo "main_trace_text_f_boundary_guarded" ;;
    qwen25|llama31|mistral7b) echo "main_trace_text_f_boundary_guarded_csis" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

trace_lora_dir_for() {
  local model_name="$1"
  echo "$WORK_ROOT/checkpoints/${model_name}_stagewise_e2e_$(trace_run_name_for "$model_name")/best_dev_f1"
}

answer_only_run_name_for() {
  case "$1" in
    qwen25_32b|qwen25) echo "main_answer_only_matched_f" ;;
    *) echo "Unknown answer-only model alias: $1" >&2; return 1 ;;
  esac
}

answer_only_lora_dir_for() {
  local model_name="$1"
  echo "$WORK_ROOT/checkpoints/${model_name}_stagewise_e2e_$(answer_only_run_name_for "$model_name")/best_dev_f1"
}

model_in_space_list() {
  local model_name="$1"
  shift
  local item
  for item in "$@"; do
    if [ "$item" = "$model_name" ]; then
      return 0
    fi
  done
  return 1
}

base_model_for() {
  case "$1" in
    qwen25_32b) echo "$MODEL_ROOT/Qwen2.5-32B-Instruct" ;;
    qwen25) echo "$MODEL_ROOT/Qwen2.5-7B-Instruct" ;;
    llama31) echo "$MODEL_ROOT/Llama-3.1-8B-Instruct" ;;
    mistral7b) echo "$MODEL_ROOT/Mistral-7B-Instruct-v0.3" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

load_in_4bit_for() {
  case "$1" in
    qwen25_32b) echo "1" ;;
    qwen25|llama31|mistral7b) echo "0" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

attn_impl_for() {
  case "$1" in
    qwen25_32b|qwen25|llama31|mistral7b) echo "sdpa" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

partition_for() {
  local model_name="$1"
  if [ -n "$H100_MODELS" ]; then
    read -r -a h100_models <<< "$H100_MODELS"
    if model_in_space_list "$model_name" "${h100_models[@]}"; then
      echo "$SMALL_H100_PARTITION"
      return 0
    fi
  fi
  if [ -n "$V100_MODELS" ]; then
    read -r -a v100_models <<< "$V100_MODELS"
    if model_in_space_list "$model_name" "${v100_models[@]}"; then
      echo "$V100_PARTITION"
      return 0
    fi
  fi
  case "$1" in
    qwen25_32b|qwen25) echo "$LARGE_PARTITION" ;;
    llama31) echo "$SMALL_H100_PARTITION" ;;
    mistral7b) echo "$SMALL_A100_PARTITION" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

nodelist_for() {
  local model_name="$1"
  if [ -n "$H100_MODELS" ]; then
    read -r -a h100_models <<< "$H100_MODELS"
    if model_in_space_list "$model_name" "${h100_models[@]}"; then
      echo "$SMALL_H100_NODELIST"
      return 0
    fi
  fi
  if [ -n "$V100_MODELS" ]; then
    read -r -a v100_models <<< "$V100_MODELS"
    if model_in_space_list "$model_name" "${v100_models[@]}"; then
      echo "$V100_NODELIST"
      return 0
    fi
  fi
  case "$1" in
    qwen25_32b|qwen25) echo "$LARGE_NODELIST" ;;
    llama31) echo "$SMALL_H100_NODELIST" ;;
    mistral7b) echo "$SMALL_A100_NODELIST" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

time_for() {
  local model_name="$1"
  if [ -n "$H100_MODELS" ]; then
    read -r -a h100_models <<< "$H100_MODELS"
    if model_in_space_list "$model_name" "${h100_models[@]}"; then
      echo "$SMALL_H100_TIME"
      return 0
    fi
  fi
  if [ -n "$V100_MODELS" ]; then
    read -r -a v100_models <<< "$V100_MODELS"
    if model_in_space_list "$model_name" "${v100_models[@]}"; then
      echo "$V100_TIME"
      return 0
    fi
  fi
  case "$1" in
    qwen25_32b|qwen25) echo "$LARGE_TIME" ;;
    llama31) echo "$SMALL_H100_TIME" ;;
    mistral7b) echo "$SMALL_A100_TIME" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

dtype_for() {
  local model_name="$1"
  if [ -n "$V100_MODELS" ]; then
    read -r -a v100_models <<< "$V100_MODELS"
    if model_in_space_list "$model_name" "${v100_models[@]}"; then
      echo "$V100_DTYPE"
      return 0
    fi
  fi
  echo "$DTYPE"
}

cpus_for() {
  local model_name="$1"
  if [ -n "$H100_MODELS" ]; then
    read -r -a h100_models <<< "$H100_MODELS"
    if model_in_space_list "$model_name" "${h100_models[@]}"; then
      echo "$SMALL_H100_CPUS"
      return 0
    fi
  fi
  if [ -n "$V100_MODELS" ]; then
    read -r -a v100_models <<< "$V100_MODELS"
    if model_in_space_list "$model_name" "${v100_models[@]}"; then
      echo "$V100_CPUS"
      return 0
    fi
  fi
  case "$1" in
    qwen25_32b|qwen25) echo "$LARGE_CPUS" ;;
    llama31) echo "$SMALL_H100_CPUS" ;;
    mistral7b) echo "$SMALL_A100_CPUS" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

prompt_suffix_for() {
  local message_tag="$1"
  if [ -n "$message_tag" ]; then
    echo "_$message_tag"
  else
    echo ""
  fi
}

tag_base_for() {
  local model_name="$1"
  local model_variant="$2"
  local run_name="$3"
  echo "${model_variant}_${model_name}_stagewise_${run_name}"
}

sanitized_output_for() {
  local model_name="$1"
  local model_variant="$2"
  local run_name="$3"
  local prompt_mode="$4"
  local message_tag="$5"
  local prompt_label="${prompt_mode}$(prompt_suffix_for "$message_tag")"
  echo "outputs/$(tag_base_for "$model_name" "$model_variant" "$run_name")_${prompt_label}_${DATASET_LABEL}.sanitized.jsonl"
}

report_dir_for() {
  local model_name="$1"
  local model_variant="$2"
  local run_name="$3"
  local prompt_mode="$4"
  local message_tag="$5"
  local prompt_label="${prompt_mode}$(prompt_suffix_for "$message_tag")"
  echo "outputs/reports/$(tag_base_for "$model_name" "$model_variant" "$run_name")_${prompt_label}_${DATASET_LABEL}"
}

submit_generate_eval() {
  local kind="$1"
  local model_name="$2"
  local model_variant="$3"
  local run_name="$4"
  local prompt_mode="$5"
  local prompt_profile="$6"
  local message_tag="$7"
  local contract_mode="$8"
  local retry_attempts="$9"
  local max_new_base="${10}"
  local max_new_cap="${11}"
  local lora_dir="${12}"

  local base_model partition nodelist walltime run_dtype load_in_4bit attn_impl out_san report_dir cpu_per_task
  local name_suffix
  base_model="$(base_model_for "$model_name")"
  partition="$(partition_for "$model_name")"
  nodelist="$(nodelist_for "$model_name")"
  walltime="$(time_for "$model_name")"
  cpu_per_task="$(cpus_for "$model_name")"
  run_dtype="$(dtype_for "$model_name")"
  load_in_4bit="$(load_in_4bit_for "$model_name")"
  attn_impl="$(attn_impl_for "$model_name")"
  out_san="$(sanitized_output_for "$model_name" "$model_variant" "$run_name" "$prompt_mode" "$message_tag")"
  report_dir="$(report_dir_for "$model_name" "$model_variant" "$run_name" "$prompt_mode" "$message_tag")"
  name_suffix="${message_tag:-default}"

  echo "===== queue ${kind} :: ${model_name} ${model_variant} ${prompt_mode} ${prompt_profile} ====="
  echo "base_model=$base_model"
  echo "partition=$partition time=$walltime cpus_per_task=$cpu_per_task nodelist=${nodelist:-<auto>}"
  echo "dtype=$run_dtype"
  echo "out_sanitized=$out_san"

  test -d "$base_model"
  if [ "$model_variant" = "sft" ]; then
    test -d "$lora_dir"
  fi

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_san" ] && [ -f "$report_dir/final_answer.json" ]; then
    echo "[Skip] Existing sanitized output and eval report detected."
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$kind" "$model_name" "$model_variant" "$prompt_mode" "$prompt_profile" "$message_tag" "SKIPPED" "SKIPPED" "$out_san" \
      >> "$SUBMISSION_LOG"
    return 0
  fi

  local env_args=(
    MODEL_NAME="$model_name"
    BASE_MODEL="$base_model"
    TRAIN_STRATEGY=stagewise
    DATASET_LABEL="$DATASET_LABEL"
    PROMPT_MODE="$prompt_mode"
    PROMPT_PROFILE="$prompt_profile"
    MESSAGE_TAG="$message_tag"
    MODEL_VARIANT="$model_variant"
    RUN_NAME="$run_name"
    DTYPE="$run_dtype"
    LOAD_IN_4BIT="$load_in_4bit"
    RESUME="$RESUME"
    CONTRACT_MODE="$contract_mode"
    RETRY_ATTEMPTS="$retry_attempts"
    MAX_NEW_TOKENS_BASE="$max_new_base"
    MAX_NEW_TOKENS_CAP="$max_new_cap"
    ATTN_IMPL="$attn_impl"
  )
  if [ "$model_variant" = "sft" ]; then
    env_args+=(LORA_DIR="$lora_dir")
  fi

  local sbatch_args=(
    --parsable
    --partition="$partition"
    --time="$walltime"
    --cpus-per-task="$cpu_per_task"
    --job-name="${JOB_NAME_PREFIX}-bench-${kind}-${model_name}-${model_variant}-${prompt_mode}-${name_suffix}"
  )
  if [ -n "$nodelist" ]; then
    sbatch_args+=(--nodelist="$nodelist")
  fi

  local gen_job_id
  gen_job_id="$(
    env "${env_args[@]}" \
      sbatch \
        "${sbatch_args[@]}" \
        slurm/sharanga/generate_experiment.sh
  )"
  gen_job_id="${gen_job_id%%;*}"
  echo "generation job id: $gen_job_id"

  local eval_job_id
  eval_job_id="$(
    env \
      MODEL_NAME="$model_name" \
      RUN_NAME="$run_name" \
      TRAIN_STRATEGY=stagewise \
      DATASET_LABEL="$DATASET_LABEL" \
      PROMPT_MODE="$prompt_mode" \
      PROMPT_PROFILE="$prompt_profile" \
      MESSAGE_TAG="$message_tag" \
      MODEL_VARIANT="$model_variant" \
      sbatch \
        --parsable \
        --dependency="afterok:${gen_job_id}" \
        --partition="$EVAL_PARTITION" \
        --time="$EVAL_TIME" \
        --job-name="${JOB_NAME_PREFIX}-bench-eval-${kind}-${model_name}-${prompt_mode}-${name_suffix}" \
        slurm/sharanga/evaluate_experiment.sh
  )"
  eval_job_id="${eval_job_id%%;*}"
  echo "eval job id: $eval_job_id"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$kind" "$model_name" "$model_variant" "$prompt_mode" "$prompt_profile" "$message_tag" "$gen_job_id" "$eval_job_id" "$out_san" \
    >> "$SUBMISSION_LOG"
}

read -r -a trace_models <<< "$TRACE_MODELS"
read -r -a trace_model_variants <<< "$TRACE_MODEL_VARIANTS"
read -r -a trace_prompt_modes <<< "$TRACE_PROMPT_MODES"
read -r -a answer_only_models <<< "$ANSWER_ONLY_MODELS"

if [ -z "$TRACE_PROFILE_SPECS" ]; then
  TRACE_PROFILE_SPECS="${TRACE_PROMPT_PROFILE}:${TRACE_MESSAGE_TAG}:${TRACE_CONTRACT_MODE}:${TRACE_RETRY_ATTEMPTS}:${TRACE_MAX_NEW_BASE}:${TRACE_MAX_NEW_CAP}"
fi

if [ "$RUN_TRACE" = "1" ]; then
  IFS=';' read -r -a trace_profile_specs <<< "$TRACE_PROFILE_SPECS"
  for model_name in "${trace_models[@]}"; do
    test -d "$(base_model_for "$model_name")"
    test -d "$(trace_lora_dir_for "$model_name")"
    for model_variant in "${trace_model_variants[@]}"; do
      run_name="base"
      lora_dir=""
      if [ "$model_variant" = "sft" ]; then
        run_name="$(trace_run_name_for "$model_name")"
        lora_dir="$(trace_lora_dir_for "$model_name")"
      fi
      for prompt_mode in "${trace_prompt_modes[@]}"; do
        for spec in "${trace_profile_specs[@]}"; do
          [ -n "$spec" ] || continue
          IFS=':' read -r prompt_profile message_tag contract_mode retry_attempts max_new_base max_new_cap <<< "$spec"
          submit_generate_eval \
            trace \
            "$model_name" \
            "$model_variant" \
            "$run_name" \
            "$prompt_mode" \
            "$prompt_profile" \
            "$message_tag" \
            "$contract_mode" \
            "$retry_attempts" \
            "$max_new_base" \
            "$max_new_cap" \
            "$lora_dir"
        done
      done
    done
  done
fi

if [ "$RUN_ANSWER_ONLY" = "1" ]; then
  for model_name in "${answer_only_models[@]}"; do
    [ -n "$model_name" ] || continue
    test -d "$(base_model_for "$model_name")"
    test -d "$(answer_only_lora_dir_for "$model_name")"
    submit_generate_eval \
      answer_only \
      "$model_name" \
      sft \
      "$(answer_only_run_name_for "$model_name")" \
      "$ANSWER_ONLY_PROMPT_MODE" \
      "$ANSWER_ONLY_PROMPT_PROFILE" \
      "$ANSWER_ONLY_MESSAGE_TAG" \
      "$ANSWER_ONLY_CONTRACT_MODE" \
      "$ANSWER_ONLY_RETRY_ATTEMPTS" \
      "$ANSWER_ONLY_MAX_NEW_BASE" \
      "$ANSWER_ONLY_MAX_NEW_CAP" \
      "$(answer_only_lora_dir_for "$model_name")"
  done
fi

echo "===== submission log ====="
echo "$SUBMISSION_LOG"
