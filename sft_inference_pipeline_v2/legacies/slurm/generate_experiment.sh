#!/bin/bash
#SBATCH --job-name=rag-generate
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

set -euo pipefail

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
source "$PROJECT_ROOT/slurm/common_env.sh"

MODEL_NAME="${MODEL_NAME:-qwen25}"
BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/Qwen2.5-7B-Instruct}"
RUN_NAME="${RUN_NAME:-pilot1}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
DATASET_LABEL="${DATASET_LABEL:-val_stagewise}"
PROMPT_MODE="${PROMPT_MODE:-e2e}"   # e2e | oracle_conflict | oracle_notes | oracle_both
PROMPT_PROFILE="${PROMPT_PROFILE:-default}" # default | minimal | legacy_text_contract | runtime | final_only
MESSAGE_TAG="${MESSAGE_TAG:-}"
MODEL_VARIANT="${MODEL_VARIANT:-sft}" # sft | baseline
CONTRACT_MODE_WAS_SET="${CONTRACT_MODE+x}"
RETRY_ATTEMPTS_WAS_SET="${RETRY_ATTEMPTS+x}"
CONTRACT_MODE="${CONTRACT_MODE:-trace}" # trace | final | none
TEMPERATURE="${TEMPERATURE:-0.0}"
DTYPE="${DTYPE:-bf16}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
AUTO_LENGTH="${AUTO_LENGTH:-1}"
MAX_NEW_TOKENS_BASE="${MAX_NEW_TOKENS_BASE:-1200}"
MAX_NEW_TOKENS_CAP="${MAX_NEW_TOKENS_CAP:-2200}"
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-1}"
RETRY_SCALE="${RETRY_SCALE:-1.6}"
RETRY_MAX_NEW_CAP="${RETRY_MAX_NEW_CAP:-3200}"
LIMIT="${LIMIT:-0}"
RESUME="${RESUME:-1}"
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-}"

if [ -z "$MESSAGE_TAG" ] && [ "$PROMPT_PROFILE" != "default" ]; then
  MESSAGE_TAG="$PROMPT_PROFILE"
fi
if [ "$PROMPT_PROFILE" = "final_only" ] && [ "${CONTRACT_MODE:-trace}" = "trace" ]; then
  CONTRACT_MODE="final"
fi
if [ "$PROMPT_PROFILE" = "minimal" ] && [ -z "$CONTRACT_MODE_WAS_SET" ]; then
  CONTRACT_MODE="none"
fi
if [ "$PROMPT_PROFILE" = "minimal" ] && [ -z "$RETRY_ATTEMPTS_WAS_SET" ]; then
  RETRY_ATTEMPTS="0"
fi
PROMPT_SUFFIX=""
if [ -n "$MESSAGE_TAG" ]; then
  PROMPT_SUFFIX="_$MESSAGE_TAG"
fi
PROMPT_LABEL="${PROMPT_MODE}${PROMPT_SUFFIX}"

INPUT_JSONL="${INPUT_JSONL:-data/messages/${DATASET_LABEL}_${PROMPT_MODE}${PROMPT_SUFFIX}_messages.jsonl}"
TAG_BASE="${TAG_BASE:-${MODEL_VARIANT}_${MODEL_NAME}_${TRAIN_STRATEGY}_${RUN_NAME}}"
OUT_RAW="${OUT_RAW:-outputs/${TAG_BASE}_${PROMPT_LABEL}_${DATASET_LABEL}.raw.jsonl}"
OUT_SAN="${OUT_SAN:-outputs/${TAG_BASE}_${PROMPT_LABEL}_${DATASET_LABEL}.sanitized.jsonl}"
CANON_JSONL="${CANON_JSONL:-data/splits/${DATASET_LABEL}.jsonl}"
LORA_DIR="${LORA_DIR:-checkpoints/${MODEL_NAME}_${TRAIN_STRATEGY}_e2e_${RUN_NAME}/best_dev_f1}"

if [ ! -d "$BASE_MODEL" ]; then
  echo "Base model directory not found: $BASE_MODEL" >&2
  exit 1
fi
if [ ! -f "$INPUT_JSONL" ]; then
  echo "Input message file not found: $INPUT_JSONL" >&2
  exit 1
fi

GEN_CMD=(
  python code/eval/generate.py
  --base_model "$BASE_MODEL"
  --input_jsonl "$INPUT_JSONL"
  --out_jsonl "$OUT_RAW"
  --max_new_tokens_base "$MAX_NEW_TOKENS_BASE"
  --max_new_tokens_cap "$MAX_NEW_TOKENS_CAP"
  --retry_attempts "$RETRY_ATTEMPTS"
  --retry_scale "$RETRY_SCALE"
  --retry_max_new_cap "$RETRY_MAX_NEW_CAP"
  --contract_mode "$CONTRACT_MODE"
  --temperature "$TEMPERATURE"
  --dtype "$DTYPE"
)

if [ "$MODEL_VARIANT" = "sft" ]; then
  if [ ! -d "$LORA_DIR" ]; then
    echo "LoRA adapter directory not found: $LORA_DIR" >&2
    exit 1
  fi
  GEN_CMD+=(--lora_dir "$LORA_DIR")
fi
if [ "$AUTO_LENGTH" = "1" ]; then
  GEN_CMD+=(--auto_length)
fi
if [ "$LOAD_IN_4BIT" = "1" ]; then
  GEN_CMD+=(--load_in_4bit)
fi
if [ "$LIMIT" != "0" ]; then
  GEN_CMD+=(--limit "$LIMIT")
fi
if [ "$RESUME" = "1" ]; then
  GEN_CMD+=(--resume)
fi
if [ -n "$SYSTEM_PROMPT_PATH" ]; then
  GEN_CMD+=(--system_prompt_path "$SYSTEM_PROMPT_PATH")
fi

SAN_CMD=(
  python code/eval/sanitize.py
  --in_jsonl "$OUT_RAW"
  --out_jsonl "$OUT_SAN"
)
if [ -f "$CANON_JSONL" ]; then
  SAN_CMD+=(--canon_jsonl "$CANON_JSONL")
fi

echo "===== GENERATE EXPERIMENT ====="
echo "MODEL_VARIANT=$MODEL_VARIANT"
echo "MODEL_NAME=$MODEL_NAME"
echo "BASE_MODEL=$BASE_MODEL"
echo "PROMPT_MODE=$PROMPT_MODE"
echo "PROMPT_PROFILE=$PROMPT_PROFILE"
echo "CONTRACT_MODE=$CONTRACT_MODE"
echo "DATASET_LABEL=$DATASET_LABEL"
echo "INPUT_JSONL=$INPUT_JSONL"
echo "OUT_RAW=$OUT_RAW"
echo "OUT_SAN=$OUT_SAN"
if [ "$MODEL_VARIANT" = "sft" ]; then
  echo "LORA_DIR=$LORA_DIR"
fi
echo "==============================="

"${GEN_CMD[@]}"
"${SAN_CMD[@]}"
