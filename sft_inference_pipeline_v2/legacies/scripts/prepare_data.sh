#!/bin/bash
# ================================================
# prepare_data.sh - Prepare existing train/val data
# ================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}   Split-Aware Data Preparation (v2)    ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

STAGEWISE_TRAIN_JSONL="data/splits/stagewise_multi/train/stage3_final.jsonl"
STAGEWISE_VAL_JSONL="data/splits/stagewise_multi/val/stage3_final.jsonl"
MONOLITHIC_TRAIN_JSONL="data/splits/monolithic_multi/train/monolithic_final.jsonl"
MONOLITHIC_VAL_JSONL="data/splits/monolithic_multi/val/monolithic_final.jsonl"
OUT_DIR="data"
PROMPTS_DIR="prompts"

if [ ! -f "$STAGEWISE_TRAIN_JSONL" ]; then
    echo -e "${RED}Error: stagewise train split not found at $STAGEWISE_TRAIN_JSONL${NC}"
    exit 1
fi

if [ ! -f "$STAGEWISE_VAL_JSONL" ]; then
    echo -e "${RED}Error: stagewise val split not found at $STAGEWISE_VAL_JSONL${NC}"
    exit 1
fi

echo "Stagewise train:   $STAGEWISE_TRAIN_JSONL"
echo "Stagewise val:     $STAGEWISE_VAL_JSONL"
if [ -f "$MONOLITHIC_TRAIN_JSONL" ] && [ -f "$MONOLITHIC_VAL_JSONL" ]; then
    echo "Monolithic train:  $MONOLITHIC_TRAIN_JSONL"
    echo "Monolithic val:    $MONOLITHIC_VAL_JSONL"
    HAS_MONOLITHIC="y"
else
    echo "Monolithic train:  (not found)"
    echo "Monolithic val:    (not found)"
    HAS_MONOLITHIC="n"
fi

MONO_ARGS=()
if [ "$HAS_MONOLITHIC" == "y" ]; then
    MONO_ARGS+=(--monolithic_train_jsonl "$MONOLITHIC_TRAIN_JSONL")
    MONO_ARGS+=(--monolithic_val_jsonl "$MONOLITHIC_VAL_JSONL")
fi
echo "Output dir:   $OUT_DIR"
echo "Prompts dir:  $PROMPTS_DIR"
echo ""
echo "This will:"
echo "  - audit and normalize the existing split files"
echo "  - write canonical stagewise and monolithic JSONL files"
echo "  - build stagewise e2e/oracle messages"
if [ "$HAS_MONOLITHIC" == "y" ]; then
    echo "  - build monolithic e2e/oracle messages"
fi
echo ""
read -p "Proceed? [y/n]: " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Running split-aware data preparation...${NC}"
echo ""

"$PYTHON_BIN" code/data/prepare_data.py \
    --stagewise_train_jsonl "$STAGEWISE_TRAIN_JSONL" \
    --stagewise_val_jsonl "$STAGEWISE_VAL_JSONL" \
    "${MONO_ARGS[@]}" \
    --out_dir "$OUT_DIR" \
    --prompts_dir "$PROMPTS_DIR"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Data preparation complete             ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Canonical splits:"
echo "  data/splits/train.jsonl"
echo "  data/splits/val.jsonl"
echo "  data/splits/train_stagewise.jsonl"
echo "  data/splits/val_stagewise.jsonl"
if [ "$HAS_MONOLITHIC" == "y" ]; then
    echo "  data/splits/train_monolithic.jsonl"
    echo "  data/splits/val_monolithic.jsonl"
fi
echo ""
echo "Message files:"
echo "  data/messages/train_stagewise_e2e_messages.jsonl"
if [ "$HAS_MONOLITHIC" == "y" ]; then
    echo "  data/messages/train_monolithic_e2e_messages.jsonl"
fi
echo "  data/messages/val_stagewise_e2e_messages.jsonl"
echo "  data/messages/val_stagewise_oracle_conflict_messages.jsonl"
echo "  data/messages/val_stagewise_oracle_notes_messages.jsonl"
echo "  data/messages/val_stagewise_oracle_both_messages.jsonl"
if [ "$HAS_MONOLITHIC" == "y" ]; then
    echo "  data/messages/val_monolithic_e2e_messages.jsonl"
    echo "  data/messages/val_monolithic_oracle_conflict_messages.jsonl"
    echo "  data/messages/val_monolithic_oracle_notes_messages.jsonl"
    echo "  data/messages/val_monolithic_oracle_both_messages.jsonl"
fi
echo ""
echo "Next step: bash scripts/train.sh"
