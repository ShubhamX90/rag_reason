#!/bin/bash
# quick_fix_reports.sh - Generate missing reports from existing call outputs
# This script processes existing call outputs to generate missing reports
# without re-running the expensive model inference steps.

set -e

CANON_JSONL="data/splits/test_v5.jsonl"

echo "=========================================="
echo "Quick Fix - Generate Missing Reports"
echo "=========================================="
echo "This will generate reports from existing call outputs."
echo "No model inference will be performed."
echo ""

# Verify canonical data exists
if [ ! -f "$CANON_JSONL" ]; then
    echo "Error: Canonical data not found: $CANON_JSONL"
    echo "Please run this script from your project root directory."
    exit 1
fi

# ============================================
# Fix Ablation Type1 Oracle
# ============================================

echo "[1/6] Processing ablation type1 oracle..."
OUTPUT_DIR="outputs/ablations/type1_oracle/test"
REPORTS_DIR="${OUTPUT_DIR}/reports"

if [ -f "${OUTPUT_DIR}/call1_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call2_outputs.jsonl" ]; then
    mkdir -p "$REPORTS_DIR"
    
    # Convert
    echo "  Converting..."
    python code/eval/convert_ablation_type1.py \
      --ablation_dir "$OUTPUT_DIR" \
      --canon_jsonl "$CANON_JSONL" \
      --oracle_level "oracle" \
      --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"
    
    # Sanitize
    echo "  Sanitizing..."
    python code/eval/sanitize_textmode_v5.py \
      --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
      --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"
    
    # Evaluate
    echo "  Evaluating..."
    python code/eval/eval_text_contract_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/contract.json" > /dev/null
    
    python code/eval/eval_doc_verdicts_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null
    
    python code/eval/eval_conflict_type_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null
    
    echo "  ✓ Complete"
else
    echo "  ⚠ Skipping - call outputs not found"
fi

# ============================================
# Fix Ablation Type1 E2E
# ============================================

echo ""
echo "[2/6] Processing ablation type1 e2e..."
OUTPUT_DIR="outputs/ablations/type1_e2e/test"
REPORTS_DIR="${OUTPUT_DIR}/reports"

if [ -f "${OUTPUT_DIR}/call1_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call2_outputs.jsonl" ]; then
    mkdir -p "$REPORTS_DIR"
    
    # Convert
    echo "  Converting..."
    python code/eval/convert_ablation_type1.py \
      --ablation_dir "$OUTPUT_DIR" \
      --canon_jsonl "$CANON_JSONL" \
      --oracle_level "e2e" \
      --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"
    
    # Sanitize
    echo "  Sanitizing..."
    python code/eval/sanitize_textmode_v5.py \
      --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
      --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"
    
    # Evaluate
    echo "  Evaluating..."
    python code/eval/eval_text_contract_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/contract.json" > /dev/null
    
    python code/eval/eval_doc_verdicts_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null
    
    python code/eval/eval_conflict_type_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null
    
    echo "  ✓ Complete"
else
    echo "  ⚠ Skipping - call outputs not found"
fi

# ============================================
# Fix Main Stagewise E2E
# ============================================

echo ""
echo "[3/6] Processing main stagewise e2e..."
OUTPUT_DIR="outputs/main_stagewise/e2e/test"
REPORTS_DIR="${OUTPUT_DIR}/reports"

if [ -f "${OUTPUT_DIR}/call1_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call2_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call3_outputs.jsonl" ]; then
    mkdir -p "$REPORTS_DIR"
    
    # Convert
    echo "  Converting..."
    python code/eval/convert_main_stagewise.py \
      --stagewise_dir "$OUTPUT_DIR" \
      --canon_jsonl "$CANON_JSONL" \
      --oracle_level "e2e" \
      --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"
    
    # Sanitize
    echo "  Sanitizing..."
    python code/eval/sanitize_textmode_v5.py \
      --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
      --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"
    
    # Evaluate
    echo "  Evaluating..."
    python code/eval/eval_text_contract_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/contract.json" > /dev/null
    
    python code/eval/eval_doc_verdicts_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null
    
    python code/eval/eval_conflict_type_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null
    
    echo "  ✓ Complete"
else
    echo "  ⚠ Skipping - call outputs not found"
fi

# ============================================
# Fix Main Stagewise Oracle1
# ============================================

echo ""
echo "[4/6] Processing main stagewise oracle1..."
OUTPUT_DIR="outputs/main_stagewise/oracle1/test"
REPORTS_DIR="${OUTPUT_DIR}/reports"

if [ -f "${OUTPUT_DIR}/call1_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call2_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call3_outputs.jsonl" ]; then
    mkdir -p "$REPORTS_DIR"
    
    # Convert
    echo "  Converting..."
    python code/eval/convert_main_stagewise.py \
      --stagewise_dir "$OUTPUT_DIR" \
      --canon_jsonl "$CANON_JSONL" \
      --oracle_level "oracle1" \
      --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"
    
    # Sanitize
    echo "  Sanitizing..."
    python code/eval/sanitize_textmode_v5.py \
      --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
      --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"
    
    # Evaluate
    echo "  Evaluating..."
    python code/eval/eval_text_contract_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/contract.json" > /dev/null
    
    python code/eval/eval_doc_verdicts_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null
    
    python code/eval/eval_conflict_type_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null
    
    echo "  ✓ Complete"
else
    echo "  ⚠ Skipping - call outputs not found"
fi

# ============================================
# Fix Main Stagewise Oracle2
# ============================================

echo ""
echo "[5/6] Processing main stagewise oracle2..."
OUTPUT_DIR="outputs/main_stagewise/oracle2/test"
REPORTS_DIR="${OUTPUT_DIR}/reports"

if [ -f "${OUTPUT_DIR}/call1_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call2_outputs.jsonl" ]; then
    mkdir -p "$REPORTS_DIR"
    
    # Convert
    echo "  Converting..."
    python code/eval/convert_main_stagewise.py \
      --stagewise_dir "$OUTPUT_DIR" \
      --canon_jsonl "$CANON_JSONL" \
      --oracle_level "oracle2" \
      --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"
    
    # Sanitize
    echo "  Sanitizing..."
    python code/eval/sanitize_textmode_v5.py \
      --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
      --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"
    
    # Evaluate
    echo "  Evaluating..."
    python code/eval/eval_text_contract_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/contract.json" > /dev/null
    
    python code/eval/eval_doc_verdicts_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null
    
    python code/eval/eval_conflict_type_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null
    
    echo "  ✓ Complete"
else
    echo "  ⚠ Skipping - call outputs not found"
fi

# ============================================
# Fix Main Stagewise Oracle3
# ============================================

echo ""
echo "[6/6] Processing main stagewise oracle3..."
OUTPUT_DIR="outputs/main_stagewise/oracle3/test"
REPORTS_DIR="${OUTPUT_DIR}/reports"

if [ -f "${OUTPUT_DIR}/call1_outputs.jsonl" ] && [ -f "${OUTPUT_DIR}/call2_outputs.jsonl" ]; then
    mkdir -p "$REPORTS_DIR"
    
    # Convert
    echo "  Converting..."
    python code/eval/convert_main_stagewise.py \
      --stagewise_dir "$OUTPUT_DIR" \
      --canon_jsonl "$CANON_JSONL" \
      --oracle_level "oracle3" \
      --out_jsonl "${OUTPUT_DIR}/combined.raw.jsonl"
    
    # Sanitize
    echo "  Sanitizing..."
    python code/eval/sanitize_textmode_v5.py \
      --in_jsonl "${OUTPUT_DIR}/combined.raw.jsonl" \
      --out_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl"
    
    # Evaluate
    echo "  Evaluating..."
    python code/eval/eval_text_contract_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/contract.json" > /dev/null
    
    python code/eval/eval_doc_verdicts_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/doc_verdicts.json" > /dev/null
    
    python code/eval/eval_conflict_type_v5.py \
      --canon_jsonl "$CANON_JSONL" \
      --gens_jsonl "${OUTPUT_DIR}/combined.sanitized.jsonl" \
      --report_json "${REPORTS_DIR}/conflict_type.json" > /dev/null
    
    echo "  ✓ Complete"
else
    echo "  ⚠ Skipping - call outputs not found"
fi

# ============================================
# Summary
# ============================================

echo ""
echo "=========================================="
echo "✓ Report Generation Complete!"
echo "=========================================="
echo ""
echo "Verify reports exist:"
echo "  ls outputs/ablations/type1_oracle/test/reports/"
echo "  ls outputs/ablations/type1_e2e/test/reports/"
echo "  ls outputs/main_stagewise/*/test/reports/"
echo ""
echo "Each should contain:"
echo "  - contract.json"
echo "  - doc_verdicts.json"
echo "  - conflict_type.json"
echo ""
echo "=========================================="