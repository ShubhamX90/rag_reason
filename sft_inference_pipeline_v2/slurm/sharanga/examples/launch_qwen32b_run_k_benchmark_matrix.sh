#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_LABEL="${DATASET_LABEL:-benchmark_final_v2_holdout_clean_736}"
BENCHMARK_INPUT="${BENCHMARK_INPUT:-data/splits/benchmark_final_v2_holdout_clean_736.jsonl}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"

MODEL_NAME="${MODEL_NAME:-qwen25_32b}"
BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/Qwen2.5-32B-Instruct}"
SFT_RUN_NAME="${SFT_RUN_NAME:-main_trace_text_k_short_context_targeted_retry1}"
SFT_LORA_DIR="${SFT_LORA_DIR:-$WORK_ROOT/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_k_short_context_targeted_retry1/best_dev_f1}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_SFT="${RUN_SFT:-1}"
PROMPT_MODES="${PROMPT_MODES:-e2e oracle_conflict oracle_notes oracle_both}"
PROFILE_SPECS="${PROFILE_SPECS:-default:strict:trace:1:1400:3200;runtime:trace_text:trace:1:1200:2200;minimal:minimal:none:0:900:1800}"

PREPARE_BENCHMARK_MESSAGES="${PREPARE_BENCHMARK_MESSAGES:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

GEN_PARTITION="${GEN_PARTITION:-gpu_h200_8}"
GEN_NODELIST="${GEN_NODELIST:-}"
GEN_TIME="${GEN_TIME:-18:00:00}"
GEN_CPUS="${GEN_CPUS:-4}"
GEN_MEM="${GEN_MEM:-96G}"

EVAL_PARTITION="${EVAL_PARTITION:-compute}"
EVAL_TIME="${EVAL_TIME:-02:00:00}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-${SHARANGA_JOB_PREFIX:-atlas}}"

SUBMISSION_LOG="${SUBMISSION_LOG:-$WORK_ROOT/logs/qwen32b_run_k_benchmark_matrix_submissions.tsv}"

mkdir -p "$(dirname "$SUBMISSION_LOG")"
if [ ! -f "$SUBMISSION_LOG" ]; then
  printf "variant\tprompt_mode\tprompt_profile\tmessage_tag\tgen_job_id\teval_job_id\tout_sanitized\n" > "$SUBMISSION_LOG"
fi

if [ "$PREPARE_BENCHMARK_MESSAGES" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" \
  BENCHMARK_INPUT="$BENCHMARK_INPUT" \
  DATASET_LABEL="$DATASET_LABEL" \
  bash slurm/examples/rebuild_benchmark_messages_holdout_736_matrix.sh
fi

test -d "$BASE_MODEL"
test -f "data/splits/${DATASET_LABEL}.jsonl"
if [ "$RUN_SFT" = "1" ]; then
  test -d "$SFT_LORA_DIR"
fi

prompt_suffix_for() {
  local message_tag="$1"
  if [ -n "$message_tag" ]; then
    echo "_$message_tag"
  else
    echo ""
  fi
}

submit_pair() {
  local model_variant="$1"
  local run_name="$2"
  local prompt_mode="$3"
  local prompt_profile="$4"
  local message_tag="$5"
  local contract_mode="$6"
  local retry_attempts="$7"
  local max_new_base="$8"
  local max_new_cap="$9"
  local lora_dir="${10}"

  local prompt_suffix prompt_label tag_base out_san report_dir input_jsonl
  prompt_suffix="$(prompt_suffix_for "$message_tag")"
  prompt_label="${prompt_mode}${prompt_suffix}"
  tag_base="${model_variant}_${MODEL_NAME}_${TRAIN_STRATEGY}_${run_name}"
  out_san="outputs/${tag_base}_${prompt_label}_${DATASET_LABEL}.sanitized.jsonl"
  report_dir="outputs/reports/${tag_base}_${prompt_label}_${DATASET_LABEL}"
  input_jsonl="data/messages/${DATASET_LABEL}_${prompt_mode}${prompt_suffix}_messages.jsonl"

  test -f "$input_jsonl"
  if [ "$model_variant" = "sft" ]; then
    test -d "$lora_dir"
  fi

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out_san" ] && [ -f "$report_dir/final_answer.json" ]; then
    echo "[Skip] ${model_variant} ${prompt_mode} ${message_tag:-default} already complete"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$model_variant" "$prompt_mode" "$prompt_profile" "$message_tag" "SKIPPED" "SKIPPED" "$out_san" \
      >> "$SUBMISSION_LOG"
    return 0
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "[Dry run] ${model_variant} ${prompt_mode} ${message_tag:-default}"
    echo "  input_jsonl=$input_jsonl"
    echo "  out_sanitized=$out_san"
    if [ "$model_variant" = "sft" ]; then
      echo "  lora_dir=$lora_dir"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$model_variant" "$prompt_mode" "$prompt_profile" "$message_tag" "DRY_RUN" "DRY_RUN" "$out_san" \
      >> "$SUBMISSION_LOG"
    return 0
  fi

  local gen_job_id sbatch_args
  sbatch_args=(
    --parsable
    --partition="$GEN_PARTITION"
    --time="$GEN_TIME"
    --cpus-per-task="$GEN_CPUS"
    --mem="$GEN_MEM"
    --job-name="${JOB_NAME_PREFIX}-q32b-${model_variant}-${prompt_mode}-${message_tag:-default}"
  )
  if [ -n "$GEN_NODELIST" ]; then
    sbatch_args+=(--nodelist="$GEN_NODELIST")
  fi

  local env_args=(
    MODEL_NAME="$MODEL_NAME"
    BASE_MODEL="$BASE_MODEL"
    TRAIN_STRATEGY="$TRAIN_STRATEGY"
    DATASET_LABEL="$DATASET_LABEL"
    PROMPT_MODE="$prompt_mode"
    PROMPT_PROFILE="$prompt_profile"
    MESSAGE_TAG="$message_tag"
    MODEL_VARIANT="$model_variant"
    RUN_NAME="$run_name"
    DTYPE="${DTYPE:-bf16}"
    LOAD_IN_4BIT=1
    RESUME="$RESUME"
    CONTRACT_MODE="$contract_mode"
    RETRY_ATTEMPTS="$retry_attempts"
    MAX_NEW_TOKENS_BASE="$max_new_base"
    MAX_NEW_TOKENS_CAP="$max_new_cap"
    ATTN_IMPL="${ATTN_IMPL:-sdpa}"
  )
  if [ "$model_variant" = "sft" ]; then
    env_args+=(LORA_DIR="$lora_dir")
  fi

  gen_job_id="$(
    env "${env_args[@]}" \
      sbatch "${sbatch_args[@]}" slurm/sharanga/generate_experiment.sh
  )"
  gen_job_id="${gen_job_id%%;*}"

  local eval_job_id
  eval_job_id="$(
    env \
      MODEL_NAME="$MODEL_NAME" \
      TRAIN_STRATEGY="$TRAIN_STRATEGY" \
      DATASET_LABEL="$DATASET_LABEL" \
      PROMPT_MODE="$prompt_mode" \
      PROMPT_PROFILE="$prompt_profile" \
      MESSAGE_TAG="$message_tag" \
      MODEL_VARIANT="$model_variant" \
      RUN_NAME="$run_name" \
      sbatch \
        --parsable \
        --dependency="afterok:${gen_job_id}" \
        --partition="$EVAL_PARTITION" \
        --time="$EVAL_TIME" \
        --job-name="${JOB_NAME_PREFIX}-q32b-eval-${model_variant}-${prompt_mode}-${message_tag:-default}" \
        slurm/sharanga/evaluate_experiment.sh
  )"
  eval_job_id="${eval_job_id%%;*}"

  echo "Queued ${model_variant} ${prompt_mode} ${message_tag:-default}: gen=${gen_job_id} eval=${eval_job_id}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$model_variant" "$prompt_mode" "$prompt_profile" "$message_tag" "$gen_job_id" "$eval_job_id" "$out_san" \
    >> "$SUBMISSION_LOG"
}

echo "===== Qwen32B Run K benchmark matrix launch ====="
echo "DATASET_LABEL=$DATASET_LABEL"
echo "BENCHMARK_INPUT=$BENCHMARK_INPUT"
echo "BASE_MODEL=$BASE_MODEL"
echo "SFT_LORA_DIR=$SFT_LORA_DIR"
echo "GEN_PARTITION=$GEN_PARTITION"
echo "GEN_NODELIST=${GEN_NODELIST:-<auto>}"
echo "JOB_NAME_PREFIX=$JOB_NAME_PREFIX"
echo "RUN_BASELINE=$RUN_BASELINE RUN_SFT=$RUN_SFT"
echo "PROMPT_MODES=$PROMPT_MODES"
echo "PROFILE_SPECS=$PROFILE_SPECS"
echo "DRY_RUN=$DRY_RUN"

read -r -a prompt_modes <<< "$PROMPT_MODES"
IFS=';' read -r -a profile_specs <<< "$PROFILE_SPECS"

for prompt_mode in "${prompt_modes[@]}"; do
  [ -n "$prompt_mode" ] || continue
  for spec in "${profile_specs[@]}"; do
    [ -n "$spec" ] || continue
    IFS=':' read -r prompt_profile message_tag contract_mode retry_attempts max_new_base max_new_cap <<< "$spec"
    if [ "$RUN_BASELINE" = "1" ]; then
      submit_pair \
        baseline \
        base \
        "$prompt_mode" \
        "$prompt_profile" \
        "$message_tag" \
        "$contract_mode" \
        "$retry_attempts" \
        "$max_new_base" \
        "$max_new_cap" \
        ""
    fi
    if [ "$RUN_SFT" = "1" ]; then
      submit_pair \
        sft \
        "$SFT_RUN_NAME" \
        "$prompt_mode" \
        "$prompt_profile" \
        "$message_tag" \
        "$contract_mode" \
        "$retry_attempts" \
        "$max_new_base" \
        "$max_new_cap" \
        "$SFT_LORA_DIR"
    fi
  done
done

echo "===== Submission log ====="
echo "$SUBMISSION_LOG"
