#!/bin/bash
#SBATCH --job-name=rag-eval-sh
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/sharanga_eval_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/sharanga_eval_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

MODEL_NAME="${MODEL_NAME:-qwen25}"
RUN_NAME="${RUN_NAME:-pilot_sharanga}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
DATASET_LABEL="${DATASET_LABEL:-val_stagewise}"
PROMPT_MODE="${PROMPT_MODE:-e2e}"
PROMPT_PROFILE="${PROMPT_PROFILE:-default}" # default | minimal | legacy_text_contract | runtime | final_only | trace_text
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

GENS_JSONL="${GENS_JSONL:-$OUTPUT_ROOT/${TAG_BASE}_${PROMPT_LABEL}_${DATASET_LABEL}.sanitized.jsonl}"
CANON_JSONL="${CANON_JSONL:-data/splits/${DATASET_LABEL}.jsonl}"
REPORTS_DIR="${REPORTS_DIR:-$OUTPUT_ROOT/reports/${TAG_BASE}_${PROMPT_LABEL}_${DATASET_LABEL}}"

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

python code/eval/eval_final_answer.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "$GENS_JSONL" \
  --report_json "$REPORTS_DIR/final_answer.json" \
  --per_id_jsonl "$REPORTS_DIR/final_answer_per_id.jsonl"

echo "===== SHARANGA EVAL REPORTS ====="
echo "$REPORTS_DIR/contract.json"
echo "$REPORTS_DIR/doc_verdicts.json"
echo "$REPORTS_DIR/conflict_type.json"
echo "$REPORTS_DIR/final_answer.json"
