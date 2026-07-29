#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-qwen25_32b qwen3_32b}"
PARTITION="${PARTITION:-gpu_h200_8}"
NODELIST="${NODELIST:-gpunode7}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
GEN_TIME="${GEN_TIME:-06:00:00}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
REBUILD_MESSAGES="${REBUILD_MESSAGES:-1}"
RUN_MODEL_PREFLIGHT="${RUN_MODEL_PREFLIGHT:-0}"

TRAIN_JSONL="${TRAIN_JSONL:-data/messages/train_stagewise_answer_only_matched_f_messages.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/messages/val_stagewise_answer_only_minimal_messages.jsonl}"
VAL_FINAL_ONLY_JSONL="${VAL_FINAL_ONLY_JSONL:-data/messages/val_stagewise_answer_only_final_only_messages.jsonl}"
RUN_NAME="${RUN_NAME:-main_answer_only_matched_f}"

model_base() {
  case "$1" in
    qwen25_32b) echo "$MODEL_ROOT/Qwen2.5-32B-Instruct" ;;
    qwen3_32b) echo "$MODEL_ROOT/Qwen3-32B" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

model_attn_impl() {
  case "$1" in
    qwen25_32b|qwen3_32b) echo "sdpa" ;;
    *) echo "Unknown model alias: $1" >&2; return 1 ;;
  esac
}

out_dir_for() {
  local name="$1"
  echo "$WORK_ROOT/checkpoints/${name}_stagewise_e2e_${RUN_NAME}"
}

echo "===== Preflight: answer-only matched-F messages ====="
if [ "$REBUILD_MESSAGES" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_answer_only_matched_f.sh
fi

test -f "$TRAIN_JSONL"
test -f "$VAL_JSONL"
test -f "$VAL_FINAL_ONLY_JSONL"
"$PYTHON_BIN" scripts/check_trace_text_messages.py \
  --forbid_think \
  --forbid_task_prefix \
  "$TRAIN_JSONL" \
  "$VAL_JSONL" \
  "$VAL_FINAL_ONLY_JSONL"

echo "===== Preflight: model paths and output dirs ====="
preflight_specs=()
for name in $MODELS; do
  base="$(model_base "$name")"
  out_dir="$(out_dir_for "$name")"
  echo "model=$name base=$base out_dir=$out_dir"
  test -d "$base"
  if [ -d "$out_dir" ] && [ "$(find "$out_dir" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ] && [ "$ALLOW_EXISTING" != "1" ]; then
    echo "Refusing to submit because output dir is non-empty: $out_dir" >&2
    echo "Set ALLOW_EXISTING=1 only if you intentionally want the train job to decide how to handle it." >&2
    exit 1
  fi
  preflight_specs+=("${name}=${base}")
done

if [ "$RUN_MODEL_PREFLIGHT" = "1" ]; then
  test -f scripts/preflight_model_compat.py
  "$PYTHON_BIN" scripts/preflight_model_compat.py "${preflight_specs[@]}"
fi

submit_train() {
  local name="$1"
  local upstream_train="${2:-}"
  local base out_dir attn_impl train_job
  base="$(model_base "$name")"
  out_dir="$(out_dir_for "$name")"
  attn_impl="$(model_attn_impl "$name")"

  local dep_args=()
  if [ -n "$upstream_train" ]; then
    dep_args+=(--dependency="afterok:${upstream_train}")
    echo "$name train dependency: afterok:${upstream_train}"
  fi

  echo "===== submit answer-only train: $name ====="
  train_job="$(
    MODEL_NAME="$name" \
    BASE_MODEL="$base" \
    TRAIN_STRATEGY=stagewise \
    VAL_STRATEGY=stagewise \
    RUN_NAME="$RUN_NAME" \
    OUT_DIR="$out_dir" \
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
    CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-1.0}" \
    CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-3.0}" \
    ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.0}" \
    CITATION_WEIGHT="${CITATION_WEIGHT:-1.7}" \
    CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.0}" \
    PATIENCE="${PATIENCE:-3}" \
    DEV_SUBSET="${DEV_SUBSET:-49}" \
    DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-500}" \
    DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1200}" \
    DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.0}" \
    DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.0}" \
    DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-1.0}" \
    DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-0}" \
    DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}" \
    DEV_RETRY_CAP="${DEV_RETRY_CAP:-1600}" \
    DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-10800}" \
    ATTN_IMPL="$attn_impl" \
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}" \
    sbatch "${dep_args[@]}" \
      --partition="$PARTITION" \
      --nodelist="$NODELIST" \
      --time="$TRAIN_TIME" \
      slurm/sharanga/train_experiment_ddp_2gpu.sh |
    awk '{print $4}'
  )"
  echo "$name train job: $train_job"
  LAST_TRAIN_JOB="$train_job"
}

submit_eval_track() {
  local name="$1"
  local train_gate="$2"
  local label="$3"
  local prompt_profile="$4"
  local message_tag="$5"
  local retry_attempts="$6"
  local max_new_base="$7"
  local max_new_cap="$8"
  local base lora_dir attn_impl gen_job

  base="$(model_base "$name")"
  attn_impl="$(model_attn_impl "$name")"
  lora_dir="$(out_dir_for "$name")/best_dev_f1"

  echo "===== queue answer-only val generation: $label ====="
  gen_job="$(
    MODEL_NAME="$name" \
    BASE_MODEL="$base" \
    TRAIN_STRATEGY=stagewise \
    DATASET_LABEL=val_stagewise \
    PROMPT_MODE=e2e \
    PROMPT_PROFILE="$prompt_profile" \
    MESSAGE_TAG="$message_tag" \
    MODEL_VARIANT=sft \
    RUN_NAME="$RUN_NAME" \
    LORA_DIR="$lora_dir" \
    DTYPE="${DTYPE:-bf16}" \
    LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}" \
    CONTRACT_MODE=final \
    RETRY_ATTEMPTS="$retry_attempts" \
    MAX_NEW_TOKENS_BASE="$max_new_base" \
    MAX_NEW_TOKENS_CAP="$max_new_cap" \
    RETRY_MAX_NEW_CAP="${RETRY_MAX_NEW_CAP:-1600}" \
    RESUME="${RESUME:-0}" \
    ATTN_IMPL="$attn_impl" \
    sbatch \
      --dependency="afterok:${train_gate}" \
      --partition="$PARTITION" \
      --nodelist="$NODELIST" \
      --time="$GEN_TIME" \
      slurm/sharanga/generate_experiment.sh |
    awk '{print $4}'
  )"
  echo "$label generate job: $gen_job"

  echo "===== queue answer-only val eval: $label ====="
  MODEL_NAME="$name" \
  TRAIN_STRATEGY=stagewise \
  DATASET_LABEL=val_stagewise \
  PROMPT_MODE=e2e \
  PROMPT_PROFILE="$prompt_profile" \
  MESSAGE_TAG="$message_tag" \
  MODEL_VARIANT=sft \
  RUN_NAME="$RUN_NAME" \
  sbatch \
    --dependency="afterok:${gen_job}" \
    slurm/sharanga/evaluate_experiment.sh
}

last_train=""
for name in $MODELS; do
  submit_train "$name" "$last_train"
  last_train="$LAST_TRAIN_JOB"
done

echo "===== queue val generation/eval after final train: $last_train ====="
for name in $MODELS; do
  submit_eval_track \
    "$name" \
    "$last_train" \
    "${name}-answer-only-final" \
    final_only \
    final_only \
    "${FINAL_ONLY_RETRY_ATTEMPTS:-1}" \
    "${FINAL_ONLY_MAX_NEW_TOKENS_BASE:-500}" \
    "${FINAL_ONLY_MAX_NEW_TOKENS_CAP:-1200}"
  submit_eval_track \
    "$name" \
    "$last_train" \
    "${name}-answer-only-minimal" \
    minimal \
    minimal \
    "${MINIMAL_RETRY_ATTEMPTS:-0}" \
    "${MINIMAL_MAX_NEW_TOKENS_BASE:-500}" \
    "${MINIMAL_MAX_NEW_TOKENS_CAP:-1200}"
done
