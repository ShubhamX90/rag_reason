#!/bin/bash
# ═══════════════════════════════════════════════════
# evaluate.sh  –  Run All Evaluation Metrics
# ═══════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       Evaluation Suite (v2)        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════╝${NC}"
echo ""

# Auto mode: evaluate all sanitized files found
if [ "$1" == "--auto" ]; then
    echo -e "${YELLOW}Auto mode: evaluating all sanitized output files...${NC}"
    SANITIZED_FILES=(outputs/*.sanitized.jsonl)
    if [ ${#SANITIZED_FILES[@]} -eq 0 ] || [ ! -f "${SANITIZED_FILES[0]}" ]; then
        echo -e "${RED}No sanitized files found in outputs/${NC}"
        exit 1
    fi
else
    # Interactive selection
    echo "Available sanitized output files:"
    echo ""
    SANITIZED_FILES=()
    i=1
    for f in outputs/*.sanitized.jsonl; do
        if [ -f "$f" ]; then
            echo "  $i) $(basename $f)"
            SANITIZED_FILES+=("$f")
            i=$((i + 1))
        fi
    done

    if [ ${#SANITIZED_FILES[@]} -eq 0 ]; then
        echo -e "${RED}No sanitized output files found.${NC}"
        echo "Run: bash scripts/generate.sh first"
        exit 1
    fi

    echo ""
    echo "  a) Evaluate ALL files"
    echo ""
    read -p "Choice: " EVAL_CHOICE

    if [ "$EVAL_CHOICE" != "a" ]; then
        IDX=$((EVAL_CHOICE - 1))
        if [ $IDX -lt 0 ] || [ $IDX -ge ${#SANITIZED_FILES[@]} ]; then
            echo "Invalid."
            exit 1
        fi
        SANITIZED_FILES=("${SANITIZED_FILES[$IDX]}")
    fi
fi

# Determine split for canon data
# Convention: filename is {tag}_{mode}_{split}.sanitized.jsonl
get_split() {
    local fname=$(basename "$1" .sanitized.jsonl)
    # Extract the last part (test or val)
    echo "$fname" | grep -oP '(test|val)(?=$)' || echo "test"
}

echo ""

for GENS_FILE in "${SANITIZED_FILES[@]}"; do
    BASENAME=$(basename "$GENS_FILE" .sanitized.jsonl)
    SPLIT=$(get_split "$GENS_FILE")
    CANON="data/splits/${SPLIT}.jsonl"
    REPORTS_DIR="outputs/reports/${BASENAME}"

    if [ ! -f "$CANON" ]; then
        echo -e "${RED}Canon file not found: $CANON${NC}"
        continue
    fi

    mkdir -p "$REPORTS_DIR"

    echo -e "${CYAN}══════════════════════════════════════${NC}"
    echo -e "${BOLD}  Evaluating: ${BASENAME}${NC}"
    echo -e "${CYAN}══════════════════════════════════════${NC}"
    echo "  Generations: $GENS_FILE"
    echo "  Canon:       $CANON"
    echo "  Reports:     $REPORTS_DIR/"
    echo ""

    # 1. Contract compliance
    echo -e "  ${GREEN}[1/3]${NC} Contract compliance..."
    python code/eval/eval_contract.py \
        --canon_jsonl "$CANON" \
        --gens_jsonl "$GENS_FILE" \
        --report_json "$REPORTS_DIR/contract.json" > /dev/null 2>&1

    # 2. Doc verdicts
    echo -e "  ${GREEN}[2/3]${NC} Doc verdict accuracy..."
    python code/eval/eval_doc_verdicts.py \
        --canon_jsonl "$CANON" \
        --gens_jsonl "$GENS_FILE" \
        --report_json "$REPORTS_DIR/doc_verdicts.json" > /dev/null 2>&1

    # 3. Conflict type
    echo -e "  ${GREEN}[3/3]${NC} Conflict type classification..."
    python code/eval/eval_conflict_type.py \
        --canon_jsonl "$CANON" \
        --gens_jsonl "$GENS_FILE" \
        --report_json "$REPORTS_DIR/conflict_type.json" > /dev/null 2>&1

    # Print summary
    echo ""
    echo -e "  ${BOLD}Results:${NC}"

    if command -v python3 &> /dev/null; then
        python3 -c "
import json, sys

def safe_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

contract = safe_load('$REPORTS_DIR/contract.json')
verdicts = safe_load('$REPORTS_DIR/doc_verdicts.json')
conflict = safe_load('$REPORTS_DIR/conflict_type.json')

print(f'    Contract compliance:  {contract.get(\"ok_rate_pct\", \"?\"):>6}%')
print(f'    Doc verdict accuracy: {verdicts.get(\"totals\", {}).get(\"micro_accuracy_doc_level\", \"?\"):>6}%')
print(f'    Doc verdict macro-F1: {verdicts.get(\"overall\", {}).get(\"macro_f1\", \"?\"):>6}')
print(f'    Conflict type acc:    {conflict.get(\"overall\", {}).get(\"accuracy\", \"?\"):>6}%')

# Per-class F1 for conflict type
pclass = conflict.get('overall', {}).get('per_class', {})
if pclass:
    print()
    print('    Conflict type per-class F1:')
    for label, metrics in pclass.items():
        f1 = metrics.get('f1', 0)
        sup = metrics.get('support', 0)
        print(f'      {label:<45s} F1={f1:.3f}  (n={sup})')
"
    fi

    echo ""
done

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ Evaluation complete!                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Full reports saved to: outputs/reports/"
