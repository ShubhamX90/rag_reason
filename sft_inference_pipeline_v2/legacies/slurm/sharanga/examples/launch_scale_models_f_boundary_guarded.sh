#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"
cd "$PROJECT_ROOT"

JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-$SHARANGA_JOB_PREFIX}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_JSONL="${TRAIN_JSONL:-data/messages/train_stagewise_prompt_robust_trace_text_f_boundary_guarded_messages.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/messages/val_stagewise_e2e_minimal_messages.jsonl}"
MODELS="${MODELS:-qwen3_32b mistral24 gemma3_27b}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SEQUENTIAL_TRAIN="${SEQUENTIAL_TRAIN:-1}"
PARTITION="${PARTITION:-gpu_h200_8}"
NODELIST="${NODELIST:-gpunode7}"

model_base() {
  case "$1" in
    qwen3_32b) echo "/scratch/$USER/rag-reason/models/Qwen3-32B" ;;
    mistral24) echo "/scratch/$USER/rag-reason/models/Mistral-Small-3.2-24B-Instruct-2506" ;;
    gemma3_27b) echo "/scratch/$USER/rag-reason/models/gemma-3-27b-it" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

model_name() {
  case "$1" in
    qwen3_32b) echo "qwen3_32b" ;;
    mistral24) echo "mistral24" ;;
    gemma3_27b) echo "gemma3_27b" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

model_gen_script() {
  case "$1" in
    qwen3_32b) echo "slurm/sharanga/examples/qwen3_32b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_h200.sh" ;;
    mistral24) echo "slurm/sharanga/examples/mistral24_stagewise_generate_eval_prompt_robust_f_boundary_guarded_h200.sh" ;;
    gemma3_27b) echo "slurm/sharanga/examples/gemma27_stagewise_generate_eval_prompt_robust_f_boundary_guarded_h200.sh" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

model_attn_impl() {
  case "$1" in
    gemma3_27b) echo "eager" ;;
    *) echo "sdpa" ;;
  esac
}

echo "===== Preflight: rebuild F messages ====="
PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_prompt_robust_f_boundary_guarded.sh

echo "===== Preflight: files ====="
test -f "$TRAIN_JSONL"
test -f "$VAL_JSONL"
test -f scripts/preflight_model_compat.py
test -f code/train/train_qlora.py
test -f code/eval/generate.py

preflight_specs=()
for alias in $MODELS; do
  base="$(model_base "$alias")"
  name="$(model_name "$alias")"
  out_dir="/scratch/$USER/rag-reason/checkpoints/${name}_stagewise_e2e_main_trace_text_f_boundary_guarded"
  echo "model=$alias base=$base out_dir=$out_dir"
  test -d "$base"
  if [ -d "$out_dir" ] && [ "$(find "$out_dir" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ] && [ "$ALLOW_EXISTING" != "1" ]; then
    echo "Refusing to submit because output dir is non-empty: $out_dir" >&2
    echo "Set ALLOW_EXISTING=1 only if you intentionally want the train job to decide how to handle it." >&2
    exit 1
  fi
  preflight_specs+=("${alias}=${base}")
done

echo "===== Preflight: tokenizer/config/model-class compatibility ====="
"$PYTHON_BIN" scripts/preflight_model_compat.py "${preflight_specs[@]}"

submit_train_and_dependent_eval() {
  local alias="$1"
  local upstream_train="${2:-}"
  local name base gen_script attn_impl smoke_job dep train_job submit_prefix
  name="$(model_name "$alias")"
  base="$(model_base "$alias")"
  gen_script="$(model_gen_script "$alias")"
  attn_impl="$(model_attn_impl "$alias")"
  submit_prefix="${name}_f"

  dep=""
  if [ "$RUN_SMOKE" = "1" ]; then
    echo "===== submit smoke: $alias ====="
    smoke_job="$(
      MODEL_PATH="$base" \
      MODEL_ALIAS="$alias" \
      LOAD_IN_4BIT=1 \
      ATTN_IMPL="$attn_impl" \
      sbatch \
        --partition="$PARTITION" \
        --nodelist="$NODELIST" \
        slurm/sharanga/smoke_model_load_2gpu.sh |
      awk '{print $4}'
    )"
    echo "$alias smoke job: $smoke_job"
    dep="afterok:$smoke_job"
  fi

  if [ "$SEQUENTIAL_TRAIN" = "1" ] && [ -n "$upstream_train" ]; then
    if [ -n "$dep" ]; then
      dep="${dep}:$upstream_train"
    else
      dep="afterok:$upstream_train"
    fi
  fi

  echo "===== submit train: $alias ====="
  local dep_args=()
  if [ -n "$dep" ]; then
    dep_args+=(--dependency="$dep")
    echo "$alias train dependency: $dep"
  fi

  train_job="$(
    MODEL_NAME="$name" \
    BASE_MODEL="$base" \
    TRAIN_STRATEGY=stagewise \
    VAL_STRATEGY=stagewise \
    RUN_NAME=main_trace_text_f_boundary_guarded \
    TRAIN_JSONL="$TRAIN_JSONL" \
    VAL_JSONL="$VAL_JSONL" \
    EPOCHS="${EPOCHS:-2}" \
    LR="${LR:-2e-4}" \
    BSZ="${BSZ:-1}" \
    GRAD_ACCUM="${GRAD_ACCUM:-8}" \
    MAX_LEN="${MAX_LEN:-12288}" \
    LORA_R="${LORA_R:-32}" \
    LORA_ALPHA="${LORA_ALPHA:-64}" \
    LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
    NEFTUNE_ALPHA="${NEFTUNE_ALPHA:-5.0}" \
    CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-3.6}" \
    CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-3.0}" \
    ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.25}" \
    CITATION_WEIGHT="${CITATION_WEIGHT:-1.7}" \
    CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.55}" \
    PATIENCE="${PATIENCE:-3}" \
    DEV_SUBSET="${DEV_SUBSET:-49}" \
    DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-900}" \
    DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1800}" \
    DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.20}" \
    DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.35}" \
    DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-0.15}" \
    DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-0}" \
    DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}" \
    DEV_RETRY_CAP="${DEV_RETRY_CAP:-2600}" \
    DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-10800}" \
    ATTN_IMPL="$attn_impl" \
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}" \
    sbatch "${dep_args[@]}" \
      --partition="$PARTITION" \
      --nodelist="$NODELIST" \
      slurm/sharanga/train_experiment_ddp_2gpu.sh |
    awk '{print $4}'
  )"
  echo "$alias train job: $train_job"
  LAST_TRAIN_JOB="$train_job"

  echo "===== queue generate/eval submitter: $alias ====="
  sbatch \
    --job-name="${JOB_NAME_PREFIX}-${submit_prefix}-gen" \
    --partition=compute \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=8G \
    --time=00:30:00 \
    --dependency="afterok:${train_job}" \
    --output="logs/${submit_prefix}_submit_generate_%j.out" \
    --error="logs/${submit_prefix}_submit_generate_%j.err" \
    --wrap="cd '$PROJECT_ROOT' && REBUILD_MESSAGES=0 MODEL_NAME='$name' BASE_MODEL='$base' RUN_NAME='main_trace_text_f_boundary_guarded' ATTN_IMPL='$attn_impl' bash '$gen_script'"
}

last_train=""
for alias in $MODELS; do
  submit_train_and_dependent_eval "$alias" "$last_train"
  last_train="$LAST_TRAIN_JOB"
done
