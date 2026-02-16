#!/bin/bash
# Simple RAG Baseline Execution Script - FIXED
# =============================================
# Basic RAG without conflict awareness
#
# Usage: bash run_simple_rag.sh <split> <model_path> [--tmux|--nohup]

set -e

SPLIT=$1
MODEL_PATH=$2
RUN_MODE=$3

if [ -z "$SPLIT" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <split> <model_path> [--tmux|--nohup]"
    echo ""
    echo "Example: $0 test /path/to/model --tmux"
    exit 1
fi

SESSION_NAME="simple_rag_${SPLIT}"

# Run mode handling
if [ "$RUN_MODE" == "--tmux" ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Error: tmux session exists! Attach: tmux attach -t $SESSION_NAME"
        exit 1
    fi
    tmux new-session -d -s "$SESSION_NAME" "bash $0 $SPLIT $MODEL_PATH"
    echo "✓ Launched: $SESSION_NAME"
    exit 0
elif [ "$RUN_MODE" == "--nohup" ]; then
    LOG_FILE="logs/simple_rag_${SPLIT}.log"
    mkdir -p logs
    nohup bash $0 $SPLIT $MODEL_PATH > "$LOG_FILE" 2>&1 &
    echo "✓ Started! Monitor: tail -f $LOG_FILE"
    exit 0
fi

echo "=========================================="
echo "SIMPLE RAG BASELINE - $SPLIT"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo ""

CANON_JSONL="data/splits/${SPLIT}_v5.jsonl"
SYSTEM_PROMPT="prompts/baselines/system_simplerag.txt"
USER_PROMPT="prompts/baselines/user_simplerag.txt"
OUTPUT_DIR="outputs/baselines"
REPORTS_DIR="${OUTPUT_DIR}/reports"

mkdir -p "$OUTPUT_DIR" "$REPORTS_DIR"

# Verify inputs
if [ ! -f "$CANON_JSONL" ]; then
    echo "Error: Data file not found: $CANON_JSONL"
    exit 1
fi

if [ ! -f "$SYSTEM_PROMPT" ]; then
    echo "Error: System prompt not found: $SYSTEM_PROMPT"
    exit 1
fi

if [ ! -f "$USER_PROMPT" ]; then
    echo "Error: User prompt not found: $USER_PROMPT"
    exit 1
fi

echo "[1/3] Generating..."

# FIXED: Changed --system_prompt_path to --system_prompt and added --user_prompt
python code/eval/generate_simple_rag.py \
  --base_model "$MODEL_PATH" \
  --in_jsonl "$CANON_JSONL" \
  --system_prompt "$SYSTEM_PROMPT" \
  --user_prompt "$USER_PROMPT" \
  --output_jsonl "${OUTPUT_DIR}/simple_rag_${SPLIT}.raw.jsonl" \
  --load_in_4bit \
  --auto_length \
  --temperature 0.0 \
  --save_every 25 \
  --resume

echo ""
echo "[2/3] Sanitizing..."

python code/eval/sanitize_textmode_v5.py \
  --in_jsonl "${OUTPUT_DIR}/simple_rag_${SPLIT}.raw.jsonl" \
  --out_jsonl "${OUTPUT_DIR}/simple_rag_${SPLIT}.sanitized.jsonl"

echo ""
echo "[3/3] Evaluating..."

python code/eval/eval_text_contract_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/simple_rag_${SPLIT}.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/contract.json" > /dev/null

echo ""
echo "=========================================="
echo "COMPLETE! $(date)"
echo "=========================================="

if command -v jq &> /dev/null; then
    echo ""
    jq -r '"Contract: " + (.ok_rate_pct|tostring) + "%"' "${REPORTS_DIR}/contract.json" 2>/dev/null || true
fi

echo ""
echo "Outputs: $OUTPUT_DIR"
echo "=========================================="