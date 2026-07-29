#!/bin/bash
#SBATCH --job-name=rag-train-sh1
#SBATCH --partition=gpu_h100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/sharanga_train_1gpu_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/sharanga_train_1gpu_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

MODEL_NAME="${MODEL_NAME:-llama31}"
BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/Llama-3.1-8B-Instruct}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
VAL_STRATEGY="${VAL_STRATEGY:-$TRAIN_STRATEGY}"
RUN_NAME="${RUN_NAME:-pilot_sharanga_1gpu}"
OUT_DIR="${OUT_DIR:-$WORK_ROOT/checkpoints/${MODEL_NAME}_${TRAIN_STRATEGY}_e2e_${RUN_NAME}}"

EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-4}"
BSZ="${BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LEN="${MAX_LEN:-12288}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
NEFTUNE_ALPHA="${NEFTUNE_ALPHA:-5.0}"
CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-3.55}"
CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-3.0}"
ABSTAIN_WEIGHT="${ABSTAIN_WEIGHT:-0.4}"
ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.25}"
CITATION_WEIGHT="${CITATION_WEIGHT:-1.7}"
CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.4}"
PATIENCE="${PATIENCE:-3}"
DEV_SUBSET="${DEV_SUBSET:-0}"
DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-900}"
DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1800}"
DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.18}"
DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.25}"
DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-0.22}"
DEV_FALSE_ABSTAIN_PARTIAL_PENALTY="${DEV_FALSE_ABSTAIN_PARTIAL_PENALTY:-0.12}"
DEV_FALSE_ABSTAIN_SUPPORT_PENALTY="${DEV_FALSE_ABSTAIN_SUPPORT_PENALTY:-0.28}"
DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-0}"
DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}"
DEV_RETRY_CAP="${DEV_RETRY_CAP:-2600}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
RESUME_FROM="${RESUME_FROM:-}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"

TRAIN_JSONL="${TRAIN_JSONL:-data/messages/train_stagewise_prompt_robust_trace_text_l_boundary_rebalanced_messages.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/messages/val_stagewise_e2e_minimal_messages.jsonl}"

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
  --abstain_weight "$ABSTAIN_WEIGHT"
  --array_weight "$ARRAY_WEIGHT"
  --citation_weight "$CITATION_WEIGHT"
  --class_balance_power "$CLASS_BALANCE_POWER"
  --patience "$PATIENCE"
  --dev_subset "$DEV_SUBSET"
  --dev_max_new_base "$DEV_MAX_NEW_BASE"
  --dev_max_new_cap "$DEV_MAX_NEW_CAP"
  --dev_doc_verdict_weight "$DEV_DOC_VERDICT_WEIGHT"
  --dev_format_weight "$DEV_FORMAT_WEIGHT"
  --dev_abstain_weight "$DEV_ABSTAIN_WEIGHT"
  --dev_false_abstain_partial_penalty "$DEV_FALSE_ABSTAIN_PARTIAL_PENALTY"
  --dev_false_abstain_support_penalty "$DEV_FALSE_ABSTAIN_SUPPORT_PENALTY"
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

echo "===== SHARANGA TRAIN 1-GPU EXPERIMENT ====="
echo "HOSTNAME=$(hostname)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "WORK_ROOT=$WORK_ROOT"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "MODEL_NAME=$MODEL_NAME"
echo "BASE_MODEL=$BASE_MODEL"
echo "TRAIN_STRATEGY=$TRAIN_STRATEGY"
echo "VAL_STRATEGY=$VAL_STRATEGY"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "VAL_JSONL=$VAL_JSONL"
echo "OUT_DIR=$OUT_DIR"
echo "EPOCHS=$EPOCHS LR=$LR BSZ=$BSZ GRAD_ACCUM=$GRAD_ACCUM MAX_LEN=$MAX_LEN"
echo "LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA LORA_DROPOUT=$LORA_DROPOUT"
echo "NEFTUNE_ALPHA=$NEFTUNE_ALPHA CONFLICT_WEIGHT=$CONFLICT_WEIGHT CONTRACT_WEIGHT=$CONTRACT_WEIGHT ABSTAIN_WEIGHT=$ABSTAIN_WEIGHT ARRAY_WEIGHT=$ARRAY_WEIGHT CITATION_WEIGHT=$CITATION_WEIGHT CLASS_BALANCE_POWER=$CLASS_BALANCE_POWER"
echo "PATIENCE=$PATIENCE DEV_SUBSET=$DEV_SUBSET DEV_MAX_NEW_BASE=$DEV_MAX_NEW_BASE DEV_MAX_NEW_CAP=$DEV_MAX_NEW_CAP DEV_DOC_VERDICT_WEIGHT=$DEV_DOC_VERDICT_WEIGHT DEV_FORMAT_WEIGHT=$DEV_FORMAT_WEIGHT DEV_ABSTAIN_WEIGHT=$DEV_ABSTAIN_WEIGHT DEV_FALSE_ABSTAIN_PARTIAL_PENALTY=$DEV_FALSE_ABSTAIN_PARTIAL_PENALTY DEV_FALSE_ABSTAIN_SUPPORT_PENALTY=$DEV_FALSE_ABSTAIN_SUPPORT_PENALTY DEV_RETRY_ATTEMPTS=$DEV_RETRY_ATTEMPTS DEV_RETRY_SCALE=$DEV_RETRY_SCALE DEV_RETRY_CAP=$DEV_RETRY_CAP"
echo "ATTN_IMPL=$ATTN_IMPL"
echo "==========================================="

"${CMD[@]}"
