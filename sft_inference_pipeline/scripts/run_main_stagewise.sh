#!/bin/bash
# Main Stagewise Execution Script - Complete & Production-Ready
# ==============================================================
# Handles all 4 oracle levels: e2e, oracle1, oracle2, oracle3
#
# Usage: bash run_main_stagewise.sh <oracle_level> <split> <model_path> [--tmux|--nohup]
#
# Oracle levels:
#   e2e:     3 calls (adjudication → conflict → answer)
#   oracle1: 3 calls (adjudication → conflict[given type] → answer)
#   oracle2: 2 calls (conflict[given notes] → answer)
#   oracle3: 2 calls (copy given conflict → answer)

set -e

ORACLE_LEVEL=$1
SPLIT=$2
MODEL_PATH=$3
RUN_MODE=$4

# ============================================
# Validation
# ============================================

if [ -z "$ORACLE_LEVEL" ] || [ -z "$SPLIT" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <oracle_level> <split> <model_path> [--tmux|--nohup]"
    echo ""
    echo "Oracle Levels:"
    echo "  e2e      - End-to-end: generate all (adjudications + conflict + answer)"
    echo "  oracle1  - Given conflict_type: generate adjudications + conflict reason + answer"
    echo "  oracle2  - Given per_doc_notes: generate conflict + answer"
    echo "  oracle3  - Given per_doc_notes + conflict: generate answer only"
    echo ""
    echo "Example:"
    echo "  $0 e2e test /path/to/Llama-3.1-8B-Instruct"
    echo "  $0 oracle1 test /path/to/model --tmux"
    exit 1
fi

if [[ ! "$ORACLE_LEVEL" =~ ^(e2e|oracle1|oracle2|oracle3)$ ]]; then
    echo "Error: Oracle level must be one of: e2e, oracle1, oracle2, oracle3"
    exit 1
fi

SESSION_NAME="main_stagewise_${ORACLE_LEVEL}_${SPLIT}"

# ============================================
# Run Mode Handling
# ============================================

if [ "$RUN_MODE" == "--tmux" ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Error: tmux session '$SESSION_NAME' already exists!"
        echo "Attach with: tmux attach -t $SESSION_NAME"
        exit 1
    fi
    
    echo "Launching in tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" "bash $0 $ORACLE_LEVEL $SPLIT $MODEL_PATH"
    echo "✓ Session created!"
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Detach: Ctrl+B, then D"
    exit 0

elif [ "$RUN_MODE" == "--nohup" ]; then
    LOG_FILE="logs/main_stagewise_${ORACLE_LEVEL}_${SPLIT}.log"
    mkdir -p logs
    nohup bash $0 $ORACLE_LEVEL $SPLIT $MODEL_PATH > "$LOG_FILE" 2>&1 &
    echo "✓ Started in background! PID: $!"
    echo "  Monitor: tail -f $LOG_FILE"
    exit 0
fi

# ============================================
# Main Pipeline
# ============================================

echo "=========================================="
echo "MAIN STAGEWISE: $ORACLE_LEVEL - $SPLIT"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo ""

# Paths
CANON_JSONL="data/splits/${SPLIT}_v5.jsonl"
PROMPTS_DIR="prompts/main_stagewise"
OUTPUT_DIR="outputs/main_stagewise/${ORACLE_LEVEL}/${SPLIT}"
REPORTS_DIR="${OUTPUT_DIR}/reports"

mkdir -p "$OUTPUT_DIR" "$REPORTS_DIR"

# Verify inputs
if [ ! -f "$CANON_JSONL" ]; then
    echo "Error: Canonical data not found: $CANON_JSONL"
    exit 1
fi

if [ ! -d "$PROMPTS_DIR" ]; then
    echo "Error: Prompts directory not found: $PROMPTS_DIR"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Verify prompts exist
echo "Verifying prompts..."
REQUIRED_PROMPTS=(
    "system_call1_stagewise_${ORACLE_LEVEL}.txt"
    "user_call1_stagewise_${ORACLE_LEVEL}.txt"
    "system_call2_stagewise_${ORACLE_LEVEL}.txt"
    "user_call2_stagewise_${ORACLE_LEVEL}.txt"
)

# E2E and Oracle1 need call3 prompts
if [[ "$ORACLE_LEVEL" == "e2e" || "$ORACLE_LEVEL" == "oracle1" ]]; then
    REQUIRED_PROMPTS+=(
        "system_call3_stagewise_${ORACLE_LEVEL}.txt"
        "user_call3_stagewise_${ORACLE_LEVEL}.txt"
    )
fi

for prompt in "${REQUIRED_PROMPTS[@]}"; do
    if [ ! -f "$PROMPTS_DIR/$prompt" ]; then
        echo "Error: Missing prompt file: $PROMPTS_DIR/$prompt"
        exit 1
    fi
done
echo "✓ All prompts verified"
echo ""

# ============================================
# Step 1: Generation
# ============================================

echo "[1/4] Generating stagewise outputs..."
echo "This may take 5-10 hours depending on oracle level..."
echo ""

python code/eval/generate_main_stagewise.py \
  --base_model "$MODEL_PATH" \
  --in_jsonl "$CANON_JSONL" \
  --oracle_level "$ORACLE_LEVEL" \
  --prompts_dir "$PROMPTS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --auto_length \
  --load_in_4bit \
  --max_new_tokens_base 1200 \
  --max_new_tokens_cap 2500 \
  --save_every 25 \
  --resume \
  --temperature 0.0 \
  --top_p 1.0

if [ $? -ne 0 ]; then
    echo "Error: Generation failed!"
    exit 1
fi

echo ""
echo "✓ Generation complete"
echo ""

# ============================================
# Step 2: Conversion to Monolithic Format
# ============================================

echo "[2/4] Converting to monolithic format..."

# FIXED: Use correct arguments
python code/eval/convert_main_stagewise.py \
  --stagewise_dir "$OUTPUT_DIR" \
  --canon_jsonl "$CANON_JSONL" \
  --oracle_level "$ORACLE_LEVEL" \
  --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"

if [ $? -ne 0 ]; then
    echo "Error: Conversion failed!"
    exit 1
fi

echo "✓ Conversion complete"
echo ""

# ============================================
# Step 3: Sanitization
# ============================================

echo "[3/4] Sanitizing outputs..."

python code/eval/sanitize_textmode_v5.py \
  --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
  --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"

if [ $? -ne 0 ]; then
    echo "Error: Sanitization failed!"
    exit 1
fi

echo "✓ Sanitization complete"
echo ""

# ============================================
# Step 4: Evaluation
# ============================================

echo "[4/4] Evaluating..."

# Contract compliance
python code/eval/eval_text_contract_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/contract.json" > /dev/null

# Document verdicts
python code/eval/eval_doc_verdicts_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null

# Conflict type
python code/eval/eval_conflict_type_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null

echo "✓ Evaluation complete"
echo ""

# ============================================
# Summary
# ============================================

echo "=========================================="
echo "COMPLETE! $(date)"
echo "=========================================="
echo ""

# Display results
if command -v jq &> /dev/null; then
    echo "Results:"
    CONTRACT_OK=$(jq -r '.ok_rate_pct' "${REPORTS_DIR}/contract.json" 2>/dev/null || echo "N/A")
    DOC_VERDICT_ACC=$(jq -r '.totals.micro_accuracy_doc_level' "${REPORTS_DIR}/doc_verdicts.json" 2>/dev/null || echo "N/A")
    CONFLICT_ACC=$(jq -r '.overall.accuracy' "${REPORTS_DIR}/conflict_type.json" 2>/dev/null || echo "N/A")
    
    echo "  Contract OK: ${CONTRACT_OK}%"
    echo "  DocVerdict Accuracy: ${DOC_VERDICT_ACC}%"
    echo "  Conflict Accuracy: ${CONFLICT_ACC}%"
else
    echo "Install jq to see results summary"
    echo "Reports available at: $REPORTS_DIR"
fi

echo ""
echo "Outputs: $OUTPUT_DIR"
echo "  - call1_outputs.jsonl"
echo "  - call2_outputs.jsonl"
if [[ "$ORACLE_LEVEL" == "e2e" || "$ORACLE_LEVEL" == "oracle1" ]]; then
    echo "  - call3_outputs.jsonl"
fi
echo "  - combined.raw.jsonl"
echo "  - combined.sanitized.jsonl"
echo "  - reports/"
echo ""
echo "=========================================="