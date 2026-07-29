#!/bin/bash
#SBATCH --job-name=rag-train
#SBATCH --partition=gpu-1day
#SBATCH --qos=gpu-1day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=20:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
source "$PROJECT_ROOT/slurm/common_env.sh"

MODEL_NAME="${MODEL_NAME:-qwen25}"
BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/Qwen2.5-7B-Instruct}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
VAL_STRATEGY="${VAL_STRATEGY:-$TRAIN_STRATEGY}"
RUN_NAME="${RUN_NAME:-pilot1}"
OUT_DIR="${OUT_DIR:-checkpoints/${MODEL_NAME}_${TRAIN_STRATEGY}_e2e_${RUN_NAME}}"

EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-4}"
BSZ="${BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LEN="${MAX_LEN:-8192}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NEFTUNE_ALPHA="${NEFTUNE_ALPHA:-5.0}"
CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-3.0}"
CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-2.5}"
ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.35}"
CITATION_WEIGHT="${CITATION_WEIGHT:-1.75}"
CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.5}"
PATIENCE="${PATIENCE:-3}"
DEV_SUBSET="${DEV_SUBSET:-32}"
DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-1200}"
DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-2200}"
DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.35}"
DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-0.1}"
DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-1}"
DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}"
DEV_RETRY_CAP="${DEV_RETRY_CAP:-2400}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
RESUME_FROM="${RESUME_FROM:-}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"

TRAIN_JSONL="data/messages/train_${TRAIN_STRATEGY}_e2e_messages.jsonl"
VAL_JSONL="data/messages/val_${VAL_STRATEGY}_e2e_messages.jsonl"

if [ ! -d "$BASE_MODEL" ]; then
  echo "Base model directory not found: $BASE_MODEL" >&2
  exit 1
fi
if [ ! -f "$TRAIN_JSONL" ]; then
  echo "Training message file not found: $TRAIN_JSONL" >&2
  exit 1
fi
if [ ! -f "$VAL_JSONL" ]; then
  echo "Validation message file not found: $VAL_JSONL" >&2
  exit 1
fi

CMD=(
  python code/train/train_qlora.py
  --base_model "$BASE_MODEL"
  --train_jsonl "$TRAIN_JSONL"
  --val_jsonl "$VAL_JSONL"
  --out_dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --lr "$LR"
  --bsz "$BSZ"
  --grad_accum "$GRAD_ACCUM"
  --max_len "$MAX_LEN"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --neftune_alpha "$NEFTUNE_ALPHA"
  --conflict_weight "$CONFLICT_WEIGHT"
  --contract_weight "$CONTRACT_WEIGHT"
  --array_weight "$ARRAY_WEIGHT"
  --citation_weight "$CITATION_WEIGHT"
  --class_balance_power "$CLASS_BALANCE_POWER"
  --patience "$PATIENCE"
  --dev_subset "$DEV_SUBSET"
  --dev_max_new_base "$DEV_MAX_NEW_BASE"
  --dev_max_new_cap "$DEV_MAX_NEW_CAP"
  --dev_format_weight "$DEV_FORMAT_WEIGHT"
  --dev_abstain_weight "$DEV_ABSTAIN_WEIGHT"
  --dev_retry_attempts "$DEV_RETRY_ATTEMPTS"
  --dev_retry_scale "$DEV_RETRY_SCALE"
  --dev_retry_cap "$DEV_RETRY_CAP"
  --attn_impl "$ATTN_IMPL"
)

if [ -n "$RESUME_FROM" ]; then
  CMD+=(--resume_from "$RESUME_FROM")
fi
if [ "$OVERWRITE_OUTPUT_DIR" = "1" ]; then
  CMD+=(--overwrite_output_dir)
fi

echo "===== TRAIN EXPERIMENT ====="
echo "MODEL_NAME=$MODEL_NAME"
echo "BASE_MODEL=$BASE_MODEL"
echo "TRAIN_STRATEGY=$TRAIN_STRATEGY"
echo "VAL_STRATEGY=$VAL_STRATEGY"
echo "OUT_DIR=$OUT_DIR"
echo "EPOCHS=$EPOCHS LR=$LR BSZ=$BSZ GRAD_ACCUM=$GRAD_ACCUM MAX_LEN=$MAX_LEN"
echo "LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA LORA_DROPOUT=$LORA_DROPOUT"
echo "NEFTUNE_ALPHA=$NEFTUNE_ALPHA CONFLICT_WEIGHT=$CONFLICT_WEIGHT CONTRACT_WEIGHT=$CONTRACT_WEIGHT ARRAY_WEIGHT=$ARRAY_WEIGHT CITATION_WEIGHT=$CITATION_WEIGHT CLASS_BALANCE_POWER=$CLASS_BALANCE_POWER"
echo "PATIENCE=$PATIENCE DEV_SUBSET=$DEV_SUBSET DEV_MAX_NEW_BASE=$DEV_MAX_NEW_BASE DEV_MAX_NEW_CAP=$DEV_MAX_NEW_CAP DEV_FORMAT_WEIGHT=$DEV_FORMAT_WEIGHT DEV_ABSTAIN_WEIGHT=$DEV_ABSTAIN_WEIGHT DEV_RETRY_ATTEMPTS=$DEV_RETRY_ATTEMPTS DEV_RETRY_SCALE=$DEV_RETRY_SCALE DEV_RETRY_CAP=$DEV_RETRY_CAP"
echo "ATTN_IMPL=$ATTN_IMPL"
echo "============================"

"${CMD[@]}"
