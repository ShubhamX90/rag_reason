#!/bin/bash
# Ablation Type1 Execution Script - FIXED
# ========================================
# Type 1: Direct conflict analysis (skips document adjudication)
#
# Usage: bash run_ablation_type1.sh <oracle_level> <split> <model_path> [--tmux|--nohup]

set -e

ORACLE_LEVEL=$1
SPLIT=$2
MODEL_PATH=$3
RUN_MODE=$4

if [ -z "$ORACLE_LEVEL" ] || [ -z "$SPLIT" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <oracle_level> <split> <model_path> [--tmux|--nohup]"
    echo ""
    echo "Oracle Levels:"
    echo "  e2e    - 2 calls: direct conflict analysis → answer"
    echo "  oracle - 2 calls: conflict reasoning [given type] → answer"
    echo ""
    echo "Example: $0 e2e test /path/to/model --tmux"
    exit 1
fi

if [[ ! "$ORACLE_LEVEL" =~ ^(e2e|oracle)$ ]]; then
    echo "Error: Oracle level must be: e2e or oracle"
    exit 1
fi

SESSION_NAME="ablation_type1_${ORACLE_LEVEL}_${SPLIT}"

# Run mode handling
if [ "$RUN_MODE" == "--tmux" ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Error: tmux session exists! Attach: tmux attach -t $SESSION_NAME"
        exit 1
    fi
    tmux new-session -d -s "$SESSION_NAME" "bash $0 $ORACLE_LEVEL $SPLIT $MODEL_PATH"
    echo "✓ Launched: $SESSION_NAME"
    exit 0
elif [ "$RUN_MODE" == "--nohup" ]; then
    LOG_FILE="logs/ablation_type1_${ORACLE_LEVEL}_${SPLIT}.log"
    mkdir -p logs
    nohup bash $0 $ORACLE_LEVEL $SPLIT $MODEL_PATH > "$LOG_FILE" 2>&1 &
    echo "✓ Started! Monitor: tail -f $LOG_FILE"
    exit 0
fi

echo "=========================================="
echo "ABLATION TYPE1: $ORACLE_LEVEL - $SPLIT"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Start: $(date)"
echo ""

CANON_JSONL="data/splits/${SPLIT}_v5.jsonl"
PROMPTS_DIR="prompts/ablations"
OUTPUT_DIR="outputs/ablations/type1_${ORACLE_LEVEL}/${SPLIT}"
REPORTS_DIR="${OUTPUT_DIR}/reports"

mkdir -p "$OUTPUT_DIR" "$REPORTS_DIR"

# Verify inputs
if [ ! -f "$CANON_JSONL" ]; then
    echo "Error: Data file not found: $CANON_JSONL"
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
    "system_call1_type1_${ORACLE_LEVEL}.txt"
    "user_call1_type1_${ORACLE_LEVEL}.txt"
    "system_call2_type1.txt"
    "user_call2_type1.txt"
)

for prompt in "${REQUIRED_PROMPTS[@]}"; do
    if [ ! -f "$PROMPTS_DIR/$prompt" ]; then
        echo "Error: Missing prompt file: $PROMPTS_DIR/$prompt"
        exit 1
    fi
done
echo "✓ All prompts verified"
echo ""

echo "[1/6] Generating Type1 outputs..."

python code/eval/generate_ablation_type1.py \
  --base_model "$MODEL_PATH" \
  --in_jsonl "$CANON_JSONL" \
  --oracle_level "$ORACLE_LEVEL" \
  --prompts_dir "$PROMPTS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --auto_length \
  --load_in_4bit \
  --save_every 25 \
  --resume

if [ $? -ne 0 ]; then
    echo "Error: Generation failed!"
    exit 1
fi

echo ""
echo "[2/6] Converting to monolithic format..."

# FIXED: Use correct arguments
python code/eval/convert_ablation_type1.py \
  --ablation_dir "$OUTPUT_DIR" \
  --canon_jsonl "$CANON_JSONL" \
  --oracle_level "$ORACLE_LEVEL" \
  --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"

if [ $? -ne 0 ]; then
    echo "Error: Conversion failed!"
    exit 1
fi

echo ""
echo "[3/6] Sanitizing..."

python code/eval/sanitize_textmode_v5.py \
  --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
  --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"

if [ $? -ne 0 ]; then
    echo "Error: Sanitization failed!"
    exit 1
fi

echo ""
echo "[4/6] Evaluating..."

# Contract compliance
python code/eval/eval_text_contract_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/contract.json" > /dev/null

# Document verdicts - ADDED THIS MISSING EVALUATION
python code/eval/eval_doc_verdicts_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
  --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null

# Conflict type
python code/eval/eval_conflict_type_v5.py \
  --canon_jsonl "$CANON_JSONL" \
  --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
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
echo "  - call1_outputs.jsonl"
echo "  - call2_outputs.jsonl"
echo "  - combined.raw.jsonl"
echo "  - combined.sanitized.jsonl"
echo "  - reports/"
echo "=========================================="