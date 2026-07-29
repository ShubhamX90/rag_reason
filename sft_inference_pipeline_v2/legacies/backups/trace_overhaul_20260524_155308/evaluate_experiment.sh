#!/bin/bash
#SBATCH --job-name=rag-eval
#SBATCH --partition=cpu-short
#SBATCH --qos=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=csis.mn1
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
source "$PROJECT_ROOT/slurm/common_env.sh"

MODEL_NAME="${MODEL_NAME:-qwen25}"
RUN_NAME="${RUN_NAME:-pilot1}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
DATASET_LABEL="${DATASET_LABEL:-val_stagewise}"
PROMPT_MODE="${PROMPT_MODE:-e2e}"
PROMPT_PROFILE="${PROMPT_PROFILE:-default}"
MESSAGE_TAG="${MESSAGE_TAG:-}"
MODEL_VARIANT="${MODEL_VARIANT:-sft}" # sft | baseline
TAG_BASE="${TAG_BASE:-${MODEL_VARIANT}_${MODEL_NAME}_${TRAIN_STRATEGY}_${RUN_NAME}}"
if [ -z "$MESSAGE_TAG" ] && [ "$PROMPT_PROFILE" != "default" ]; then
  MESSAGE_TAG="$PROMPT_PROFILE"
fi
PROMPT_SUFFIX=""
if [ -n "$MESSAGE_TAG" ]; then
  PROMPT_SUFFIX="_$MESSAGE_TAG"
fi
PROMPT_LABEL="${PROMPT_MODE}${PROMPT_SUFFIX}"
GENS_JSONL="${GENS_JSONL:-outputs/${TAG_BASE}_${PROMPT_LABEL}_${DATASET_LABEL}.sanitized.jsonl}"
CANON_JSONL="${CANON_JSONL:-data/splits/${DATASET_LABEL}.jsonl}"
REPORTS_DIR="${REPORTS_DIR:-outputs/reports/${TAG_BASE}_${PROMPT_LABEL}_${DATASET_LABEL}}"

if [ ! -f "$GENS_JSONL" ]; then
  echo "Sanitized generations not found: $GENS_JSONL" >&2
  exit 1
fi
if [ ! -f "$CANON_JSONL" ]; then
  echo "Canon file not found: $CANON_JSONL" >&2
  exit 1
fi

mkdir -p "$REPORTS_DIR"

python code/eval/eval_contract.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "$GENS_JSONL" \
  --report_json "$REPORTS_DIR/contract.json"

python code/eval/eval_doc_verdicts.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "$GENS_JSONL" \
  --report_json "$REPORTS_DIR/doc_verdicts.json"

python code/eval/eval_conflict_type.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "$GENS_JSONL" \
  --report_json "$REPORTS_DIR/conflict_type.json"

echo "===== EVAL REPORTS ====="
echo "$REPORTS_DIR/contract.json"
echo "$REPORTS_DIR/doc_verdicts.json"
echo "$REPORTS_DIR/conflict_type.json"
