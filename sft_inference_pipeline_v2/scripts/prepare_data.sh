#!/bin/bash
# ═══════════════════════════════════════════════════
# prepare_data.sh  –  Split raw data + build messages
# ═══════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Data Preparation (v2)      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════╝${NC}"
echo ""

# Defaults
RAW_JSONL="data/raw/stage3_final.jsonl"
OUT_DIR="data"
PROMPTS_DIR="prompts"
TRAIN_R=0.8
VAL_R=0.1
TEST_R=0.1
SEED=42

# Check raw data exists
if [ ! -f "$RAW_JSONL" ]; then
    echo -e "${RED}Error: Raw data not found at $RAW_JSONL${NC}"
    echo "Place your stage3_final.jsonl in data/raw/"
    exit 1
fi

echo "Raw data:    $RAW_JSONL"
echo "Output dir:  $OUT_DIR"
echo "Split ratio: train=$TRAIN_R / val=$VAL_R / test=$TEST_R"
echo "Seed:        $SEED"
echo ""

# Check if splits already exist
if [ -f "$OUT_DIR/splits/train.jsonl" ]; then
    echo -e "${YELLOW}Existing splits found. What would you like to do?${NC}"
    echo "  1) Re-split from raw data (overwrites existing splits)"
    echo "  2) Skip splitting, just rebuild message files"
    echo "  3) Cancel"
    read -p "Choice [1-3]: " SPLIT_CHOICE
    case $SPLIT_CHOICE in
        1) SKIP_SPLIT="" ;;
        2) SKIP_SPLIT="--skip_split" ;;
        3) echo "Cancelled."; exit 0 ;;
        *) echo "Invalid."; exit 1 ;;
    esac
else
    SKIP_SPLIT=""
fi

echo -e "\n${GREEN}Running data preparation...${NC}\n"

python code/data/prepare_data.py \
    --raw_jsonl "$RAW_JSONL" \
    --out_dir "$OUT_DIR" \
    --prompts_dir "$PROMPTS_DIR" \
    --train_ratio $TRAIN_R \
    --val_ratio $VAL_R \
    --test_ratio $TEST_R \
    --seed $SEED \
    $SKIP_SPLIT

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ Data preparation complete!          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Files created:"
echo "  Splits:   data/splits/{train,val,test}.jsonl"
echo "  Messages: data/messages/{train,val,test}_{e2e,oracle}_messages.jsonl"
echo ""
echo "Next step: bash scripts/train.sh"
