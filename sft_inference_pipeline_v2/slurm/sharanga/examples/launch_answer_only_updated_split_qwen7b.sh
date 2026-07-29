#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_PARTITION="${TRAIN_PARTITION:-gpu_a100_8}"
TRAIN_NODELIST="${TRAIN_NODELIST:-gpunode4}"
TRAIN_TIME="${TRAIN_TIME:-24:00:00}"
GEN_PARTITION="${GEN_PARTITION:-gpu_a100_8}"
GEN_NODELIST="${GEN_NODELIST:-gpunode4}"
GEN_TIME="${GEN_TIME:-06:00:00}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
RUN_MODEL_PREFLIGHT="${RUN_MODEL_PREFLIGHT:-0}"

STAGEWISE_TRAIN_JSONL="${STAGEWISE_TRAIN_JSONL:-data/splits/train.jsonl}"
STAGEWISE_VAL_JSONL="${STAGEWISE_VAL_JSONL:-data/splits/val.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-data/messages/train_stagewise_answer_only_matched_f_messages.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/messages/val_stagewise_answer_only_minimal_messages.jsonl}"
VAL_FINAL_ONLY_JSONL="${VAL_FINAL_ONLY_JSONL:-data/messages/val_stagewise_answer_only_final_only_messages.jsonl}"
RUN_NAME="${RUN_NAME:-main_answer_only_updated_split_qwen7b}"

BASE_MODEL="${BASE_MODEL:-/scratch/$USER/rag-reason/models/Qwen2.5-7B-Instruct}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/rag-reason/checkpoints/qwen25_stagewise_e2e_${RUN_NAME}}"

echo "===== Preflight: rebuild updated answer-only messages ====="
STAGEWISE_TRAIN_JSONL="$STAGEWISE_TRAIN_JSONL" \
STAGEWISE_VAL_JSONL="$STAGEWISE_VAL_JSONL" \
OUT_JSONL="$TRAIN_JSONL" \
VAL_MINIMAL_JSONL="$VAL_JSONL" \
VAL_FINAL_ONLY_JSONL="$VAL_FINAL_ONLY_JSONL" \
PYTHON_BIN="$PYTHON_BIN" \
bash slurm/examples/rebuild_messages_answer_only_matched_f.sh

test -f "$TRAIN_JSONL"
test -f "$VAL_JSONL"
test -f "$VAL_FINAL_ONLY_JSONL"
test -d "$BASE_MODEL"

"$PYTHON_BIN" scripts/check_trace_text_messages.py \
  --forbid_think \
  --forbid_task_prefix \
  "$TRAIN_JSONL" \
  "$VAL_JSONL" \
  "$VAL_FINAL_ONLY_JSONL"

if [ -d "$OUT_DIR" ] && [ "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ] && [ "$ALLOW_EXISTING" != "1" ]; then
  echo "Refusing to submit because output dir is non-empty: $OUT_DIR" >&2
  exit 1
fi

if [ "$RUN_MODEL_PREFLIGHT" = "1" ]; then
  "$PYTHON_BIN" scripts/preflight_model_compat.py "qwen25=$BASE_MODEL"
fi

echo "===== submit updated answer-only qwen25 train ====="
TRAIN_JOB="$(
  MODEL_NAME=qwen25 \
  BASE_MODEL="$BASE_MODEL" \
  TRAIN_STRATEGY=stagewise \
  VAL_STRATEGY=stagewise \
  RUN_NAME="$RUN_NAME" \
  OUT_DIR="$OUT_DIR" \
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
  DEV_SUBSET="${DEV_SUBSET:-0}" \
  DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-500}" \
  DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1200}" \
  DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.0}" \
  DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.0}" \
  DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-1.0}" \
  DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-0}" \
  DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}" \
  DEV_RETRY_CAP="${DEV_RETRY_CAP:-1600}" \
  DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-10800}" \
  ATTN_IMPL="${ATTN_IMPL:-sdpa}" \
  OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}" \
  sbatch \
    --job-name="${TRAIN_JOB_NAME:-ao7-train}" \
    --partition="$TRAIN_PARTITION" \
    --nodelist="$TRAIN_NODELIST" \
    --time="$TRAIN_TIME" \
    slurm/sharanga/train_experiment_ddp_2gpu.sh |
  awk '{print $4}'
)"
echo "train job: $TRAIN_JOB"

echo "===== queue updated answer-only val generation: final_only ====="
FINAL_GEN_JOB="$(
  MODEL_NAME=qwen25 \
  BASE_MODEL="$BASE_MODEL" \
  TRAIN_STRATEGY=stagewise \
  DATASET_LABEL=val_stagewise \
  PROMPT_MODE=e2e \
  PROMPT_PROFILE=final_only \
  MESSAGE_TAG=final_only \
  MODEL_VARIANT=sft \
  RUN_NAME="$RUN_NAME" \
  LORA_DIR="$OUT_DIR/best_dev_f1" \
  DTYPE="${DTYPE:-bf16}" \
  LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}" \
  CONTRACT_MODE=final \
  RETRY_ATTEMPTS="${FINAL_ONLY_RETRY_ATTEMPTS:-1}" \
  MAX_NEW_TOKENS_BASE="${FINAL_ONLY_MAX_NEW_TOKENS_BASE:-500}" \
  MAX_NEW_TOKENS_CAP="${FINAL_ONLY_MAX_NEW_TOKENS_CAP:-1200}" \
  RETRY_MAX_NEW_CAP="${RETRY_MAX_NEW_CAP:-1600}" \
  RESUME="${RESUME:-0}" \
  ATTN_IMPL="${ATTN_IMPL:-sdpa}" \
  sbatch \
    --dependency="afterok:${TRAIN_JOB}" \
    --job-name="${FINAL_GEN_JOB_NAME:-ao7-genf}" \
    --partition="$GEN_PARTITION" \
    --nodelist="$GEN_NODELIST" \
    --time="$GEN_TIME" \
    slurm/sharanga/generate_experiment.sh |
  awk '{print $4}'
)"
echo "final_only gen job: $FINAL_GEN_JOB"

MODEL_NAME=qwen25 \
TRAIN_STRATEGY=stagewise \
DATASET_LABEL=val_stagewise \
PROMPT_MODE=e2e \
PROMPT_PROFILE=final_only \
MESSAGE_TAG=final_only \
MODEL_VARIANT=sft \
RUN_NAME="$RUN_NAME" \
sbatch \
  --dependency="afterok:${FINAL_GEN_JOB}" \
  --job-name="${FINAL_EVAL_JOB_NAME:-ao7-evlf}" \
  slurm/sharanga/evaluate_experiment.sh

echo "===== queue updated answer-only val generation: minimal ====="
MINIMAL_GEN_JOB="$(
  MODEL_NAME=qwen25 \
  BASE_MODEL="$BASE_MODEL" \
  TRAIN_STRATEGY=stagewise \
  DATASET_LABEL=val_stagewise \
  PROMPT_MODE=e2e \
  PROMPT_PROFILE=minimal \
  MESSAGE_TAG=minimal \
  MODEL_VARIANT=sft \
  RUN_NAME="$RUN_NAME" \
  LORA_DIR="$OUT_DIR/best_dev_f1" \
  DTYPE="${DTYPE:-bf16}" \
  LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}" \
  CONTRACT_MODE=final \
  RETRY_ATTEMPTS="${MINIMAL_RETRY_ATTEMPTS:-0}" \
  MAX_NEW_TOKENS_BASE="${MINIMAL_MAX_NEW_TOKENS_BASE:-500}" \
  MAX_NEW_TOKENS_CAP="${MINIMAL_MAX_NEW_TOKENS_CAP:-1200}" \
  RETRY_MAX_NEW_CAP="${RETRY_MAX_NEW_CAP:-1600}" \
  RESUME="${RESUME:-0}" \
  ATTN_IMPL="${ATTN_IMPL:-sdpa}" \
  sbatch \
    --dependency="afterok:${TRAIN_JOB}" \
    --job-name="${MINIMAL_GEN_JOB_NAME:-ao7-genm}" \
    --partition="$GEN_PARTITION" \
    --nodelist="$GEN_NODELIST" \
    --time="$GEN_TIME" \
    slurm/sharanga/generate_experiment.sh |
  awk '{print $4}'
)"
echo "minimal gen job: $MINIMAL_GEN_JOB"

MODEL_NAME=qwen25 \
TRAIN_STRATEGY=stagewise \
DATASET_LABEL=val_stagewise \
PROMPT_MODE=e2e \
PROMPT_PROFILE=minimal \
MESSAGE_TAG=minimal \
MODEL_VARIANT=sft \
RUN_NAME="$RUN_NAME" \
sbatch \
  --dependency="afterok:${MINIMAL_GEN_JOB}" \
  --job-name="${MINIMAL_EVAL_JOB_NAME:-ao7-evlm}" \
  slurm/sharanga/evaluate_experiment.sh
