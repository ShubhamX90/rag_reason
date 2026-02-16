#!/bin/bash
# Monolithic Baseline Execution Script - Complete & Production-Ready
# ===================================================================
# Single-shot generation using pre-formatted messages
#
# Usage: bash run_monolithic.sh <oracle_level> <split> <model_path> [--tmux|--nohup]

set -e

ORACLE_LEVEL=$1
SPLIT=$2
MODEL_PATH=$3
RUN_MODE=$4

if [ -z "$ORACLE_LEVEL" ] || [ -z "$SPLIT" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <oracle_level> <split> <model_path> [--tmux|--nohup]"
    echo ""
    echo "Oracle Levels:"
    echo "  e2e      - Generate everything in one shot"
    echo "  oracle1  - Given conflict_type"
    echo "  oracle2  - Given per_doc_notes"
    echo "  oracle3  - Given per_doc_notes + conflict"
    echo ""
    echo "Example: $0 e2e test /path/to/model --tmux"
    exit 1
fi

if [[ ! "$ORACLE_LEVEL" =~ ^(e2e|oracle1|oracle2|oracle3)$ ]]; then
    echo "Error: Oracle level must be: e2e, oracle1, oracle2, or oracle3"
    exit 1
fi

SESSION_NAME="monolithic_${ORACLE_LEVEL}_${SPLIT}"

# Run mode handling
if [ "$RUN_MODE" == "--tmux" ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Error: tmux session '$SESSION_NAME' already exists!"
        echo "Attach: tmux attach -t $SESSION_NAME"
        exit 1
    fi
    tmux new-session -d -s "$SESSION_NAME" "bash $0 $ORACLE_LEVEL $SPLIT $MODEL_PATH"
    echo "✓ Launched in tmux: $SESSION_NAME"
    echo "  Attach: tmux attach -t $SESSION_NAME"
    exit 0
elif [ "$RUN_MODE" == "--nohup" ]; then
    LOG_FILE="logs/monolithic_${ORACLE_LEVEL}_${SPLIT}.log"
    mkdir -p logs
    nohup bash $0 $ORACLE_LEVEL $SPLIT $MODEL_PATH > "$LOG_FILE" 2>&1 &
    echo "✓ Started! PID: $! | Monitor: tail -f $LOG_FILE"
    exit 0
fi

# Main pipeline
echo "=========================================="
echo "MONOLITHIC: $ORACLE_LEVEL - $SPLIT"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo ""

IN_MESSAGES="data/splits/${SPLIT}_v5_monolithic_${ORACLE_LEVEL}_messages.jsonl"
SYSTEM_PROMPT="prompts/monolithic/system_monolithic_${ORACLE_LEVEL}.txt"
CANON_JSONL="data/splits/${SPLIT}_v5.jsonl"
OUTPUT_DIR="outputs/monolithic/${ORACLE_LEVEL}/${SPLIT}"
REPORTS_DIR="${OUTPUT_DIR}/reports"

mkdir -p "$OUTPUT_DIR" "$REPORTS_DIR"

# Verify inputs
if [ ! -f "$IN_MESSAGES" ]; then
    echo "Error: Message file not found: $IN_MESSAGES"
    exit 1
fi

if [ ! -f "$SYSTEM_PROMPT" ]; then
    echo "Error: System prompt not found: $SYSTEM_PROMPT"
    exit 1
fi

echo "[1/4] Generating monolithic outputs..."

python code/eval/generate_textmode_baseline_v6.py \
  --base_model "$MODEL_PATH" \
  --input_jsonl "$IN_MESSAGES" \
  --system_prompt_path "$SYSTEM_PROMPT" \
  --out_jsonl "${OUTPUT_DIR}/baseline_${ORACLE_LEVEL}_${SPLIT}.raw.jsonl" \
  --auto_length \
  --load_in_4bit \
  --max_new_tokens_base 1200 \
  --max_new_tokens_cap 2200 \
  --temperature 0.0 \
  --top_p 1.0

echo ""
echo "[2/4] Sanitizing..."

python code/eval/sanitize_textmode_v5.py \
  --in_jsonl "${OUTPUT_DIR}/baseline_${ORACLE_LEVEL}_${SPLIT}.raw.jsonl" \
  --out_jsonl "${OUTPUT_DIR}/baseline_${ORACLE_LEVEL}_${SPLIT}.sanitized.jsonl"

echo ""
echo "[3/4] Evaluating..."

python code/eval/eval_text_contract_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/baseline_${ORACLE_LEVEL}_${SPLIT}.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/contract.json" > /dev/null

python code/eval/eval_doc_verdicts_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/baseline_${ORACLE_LEVEL}_${SPLIT}.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null

python code/eval/eval_conflict_type_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/baseline_${ORACLE_LEVEL}_${SPLIT}.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null

echo ""
echo "=========================================="
echo "COMPLETE! $(date)"
echo "=========================================="

if command -v jq &> /dev/null; then
    echo ""
    jq -r '"Contract: " + (.ok_rate_pct|tostring) + "%"' "${REPORTS_DIR}/contract.json" 2>/dev/null || true
    jq -r '"DocVerdict: " + (.totals.micro_accuracy_doc_level|tostring) + "%"' "${REPORTS_DIR}/doc_verdicts.json" 2>/dev/null || true
    jq -r '"Conflict: " + (.overall.accuracy|tostring) + "%"' "${REPORTS_DIR}/conflict_type.json" 2>/dev/null || true
fi

echo ""
echo "Outputs: $OUTPUT_DIR"
echo "=========================================="