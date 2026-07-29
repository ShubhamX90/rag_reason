#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"
cd "$PROJECT_ROOT"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-$SHARANGA_JOB_PREFIX}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-main_trace_text_l_boundary_rebalanced}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
VAL_DATASET_LABEL="${VAL_DATASET_LABEL:-val_stagewise}"
BENCHMARK_INPUT="${BENCHMARK_INPUT:-data/splits/benchmark_final_v2_holdout_clean_736.jsonl}"
BENCHMARK_DATASET_LABEL="${BENCHMARK_DATASET_LABEL:-benchmark_final_v2_holdout_clean_736}"
QWEN7B_BASE_MODEL="${QWEN7B_BASE_MODEL:-$MODEL_ROOT/Qwen2.5-7B-Instruct}"
QWEN32B_BASE_MODEL="${QWEN32B_BASE_MODEL:-$MODEL_ROOT/Qwen2.5-32B-Instruct}"

TRAIN_PARTITION_7B="${TRAIN_PARTITION_7B:-gpu_h100_4}"
TRAIN_PARTITION_32B="${TRAIN_PARTITION_32B:-gpu_h200_8}"
EVAL_PARTITION_7B="${EVAL_PARTITION_7B:-gpu_h100_4}"
EVAL_PARTITION_32B="${EVAL_PARTITION_32B:-gpu_h200_8}"

if [ "${REBUILD_MESSAGES:-1}" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh
fi

if [ "${PREPARE_BENCHMARK_MINIMAL:-1}" = "1" ]; then
  "$PYTHON_BIN" scripts/prepare_benchmark_inference.py \
    --input_jsonl "$BENCHMARK_INPUT" \
    --out_dir data \
    --prompts_dir prompts \
    --dataset_label "$BENCHMARK_DATASET_LABEL" \
    --prompt_profile minimal \
    --message_tag minimal \
    --modes e2e
fi

submit_train() {
  local label="$1"
  local script_path="$2"
  local partition="$3"
  local nodelist="${4:-}"
  local base_model="$5"
  local job_id
  if [ -n "$nodelist" ]; then
    job_id="$(
      REBUILD_MESSAGES=0 \
      RUN_NAME="$RUN_NAME" \
      BASE_MODEL="$base_model" \
      PARTITION="$partition" \
      NODELIST="$nodelist" \
      bash "$script_path" | awk '{print $4}'
    )"
  else
    job_id="$(
      REBUILD_MESSAGES=0 \
      RUN_NAME="$RUN_NAME" \
      BASE_MODEL="$base_model" \
      PARTITION="$partition" \
      bash "$script_path" | awk '{print $4}'
    )"
  fi
  echo "$label train job: $job_id" >&2
  printf '%s' "$job_id"
}

submit_minimal_chain() {
  local label="$1"
  local model_name="$2"
  local base_model="$3"
  local run_name="$4"
  local train_job="$5"
  local dataset_label="$6"
  local gen_partition="$7"
  local load_in_4bit="$8"
  local gen_nodelist="${9:-}"
  local lora_dir="$WORK_ROOT/checkpoints/${model_name}_${TRAIN_STRATEGY}_e2e_${run_name}/best_dev_f1"

  local gen_job
  if [ -n "$gen_nodelist" ]; then
    gen_job="$(
      MODEL_NAME="$model_name" \
      BASE_MODEL="$base_model" \
      TRAIN_STRATEGY="$TRAIN_STRATEGY" \
      DATASET_LABEL="$dataset_label" \
      PROMPT_MODE=e2e \
      PROMPT_PROFILE=minimal \
      MODEL_VARIANT=sft \
      RUN_NAME="$run_name" \
      LORA_DIR="$lora_dir" \
      DTYPE="${DTYPE:-bf16}" \
      LOAD_IN_4BIT="$load_in_4bit" \
      RESUME=0 \
      MAX_NEW_TOKENS_BASE="${MINIMAL_MAX_NEW_TOKENS_BASE:-900}" \
      MAX_NEW_TOKENS_CAP="${MINIMAL_MAX_NEW_TOKENS_CAP:-1800}" \
      sbatch \
        --dependency="afterok:${train_job}" \
        --partition="$gen_partition" \
        --nodelist="$gen_nodelist" \
        --job-name="${JOB_NAME_PREFIX}-${label}-gen" \
        slurm/sharanga/generate_experiment.sh | awk '{print $4}'
    )"
  else
    gen_job="$(
      MODEL_NAME="$model_name" \
      BASE_MODEL="$base_model" \
      TRAIN_STRATEGY="$TRAIN_STRATEGY" \
      DATASET_LABEL="$dataset_label" \
      PROMPT_MODE=e2e \
      PROMPT_PROFILE=minimal \
      MODEL_VARIANT=sft \
      RUN_NAME="$run_name" \
      LORA_DIR="$lora_dir" \
      DTYPE="${DTYPE:-bf16}" \
      LOAD_IN_4BIT="$load_in_4bit" \
      RESUME=0 \
      MAX_NEW_TOKENS_BASE="${MINIMAL_MAX_NEW_TOKENS_BASE:-900}" \
      MAX_NEW_TOKENS_CAP="${MINIMAL_MAX_NEW_TOKENS_CAP:-1800}" \
      sbatch \
        --dependency="afterok:${train_job}" \
        --partition="$gen_partition" \
        --job-name="${JOB_NAME_PREFIX}-${label}-gen" \
        slurm/sharanga/generate_experiment.sh | awk '{print $4}'
    )"
  fi
  echo "$label generate job: $gen_job"

  local eval_job
  eval_job="$(
    MODEL_NAME="$model_name" \
    TRAIN_STRATEGY="$TRAIN_STRATEGY" \
    DATASET_LABEL="$dataset_label" \
    PROMPT_MODE=e2e \
    PROMPT_PROFILE=minimal \
    MODEL_VARIANT=sft \
    RUN_NAME="$run_name" \
    sbatch --dependency="afterok:${gen_job}" --job-name="${JOB_NAME_PREFIX}-${label}-eval" slurm/sharanga/evaluate_experiment.sh | awk '{print $4}'
  )"
  echo "$label eval job: $eval_job"
}

TRAIN_JOB_7B="$(
  submit_train \
    "runl-7b" \
    "slurm/sharanga/examples/qwen7b_stagewise_ddp_2h100_prompt_robust_l_boundary_rebalanced.sh" \
    "$TRAIN_PARTITION_7B" \
    "${TRAIN_NODELIST_7B:-}" \
    "$QWEN7B_BASE_MODEL"
)"

TRAIN_JOB_32B="$(
  submit_train \
    "runl-32b" \
    "slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_l_boundary_rebalanced.sh" \
    "$TRAIN_PARTITION_32B" \
    "${TRAIN_NODELIST_32B:-}" \
    "$QWEN32B_BASE_MODEL"
)"

submit_minimal_chain \
  "runl-7b-val" \
  "qwen25" \
  "$QWEN7B_BASE_MODEL" \
  "$RUN_NAME" \
  "$TRAIN_JOB_7B" \
  "$VAL_DATASET_LABEL" \
  "$EVAL_PARTITION_7B" \
  "0" \
  "${EVAL_NODELIST_7B:-}"

submit_minimal_chain \
  "runl-7b-benchmark" \
  "qwen25" \
  "$QWEN7B_BASE_MODEL" \
  "$RUN_NAME" \
  "$TRAIN_JOB_7B" \
  "$BENCHMARK_DATASET_LABEL" \
  "$EVAL_PARTITION_7B" \
  "0" \
  "${EVAL_NODELIST_7B:-}"

submit_minimal_chain \
  "runl-32b-val" \
  "qwen25_32b" \
  "$QWEN32B_BASE_MODEL" \
  "$RUN_NAME" \
  "$TRAIN_JOB_32B" \
  "$VAL_DATASET_LABEL" \
  "$EVAL_PARTITION_32B" \
  "1" \
  "${EVAL_NODELIST_32B:-}"

submit_minimal_chain \
  "runl-32b-benchmark" \
  "qwen25_32b" \
  "$QWEN32B_BASE_MODEL" \
  "$RUN_NAME" \
  "$TRAIN_JOB_32B" \
  "$BENCHMARK_DATASET_LABEL" \
  "$EVAL_PARTITION_32B" \
  "1" \
  "${EVAL_NODELIST_32B:-}"
