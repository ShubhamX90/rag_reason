#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run.sh  –  SFT Inference Pipeline v2  –  Main Entry Point
# ═══════════════════════════════════════════════════════════════════
# Interactive launcher for the complete pipeline:
#   1. Prepare data   (split + build messages)
#   2. Train           (QLoRA fine-tuning)
#   3. Generate        (SFT or baseline inference)
#   4. Evaluate        (contract + doc verdicts + conflict type)
#
# Usage:
#   bash run.sh
# ═══════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  SFT Inference Pipeline v2  –  Conflict-Aware RAG ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo ""
}

print_header

echo -e "What would you like to do?\n"
echo "  1)  Prepare data      (split raw data + build message files)"
echo "  2)  Train             (QLoRA fine-tuning)"
echo "  3)  Generate          (run inference: SFT or baseline)"
echo "  4)  Evaluate          (run all evaluation metrics)"
echo "  5)  Full pipeline     (prepare → train → generate → evaluate)"
echo ""
echo "  6)  Quick baseline    (generate + evaluate with untuned model)"
echo "  7)  Quick SFT eval    (generate + evaluate with SFT model)"
echo ""
echo "  0)  Exit"
echo ""
read -p "Enter choice [0-7]: " CHOICE

case $CHOICE in
    1) bash scripts/prepare_data.sh ;;
    2) bash scripts/train.sh ;;
    3) bash scripts/generate.sh ;;
    4) bash scripts/evaluate.sh ;;
    5)
        echo -e "\n${YELLOW}Running full pipeline...${NC}\n"
        bash scripts/prepare_data.sh
        bash scripts/train.sh
        bash scripts/generate.sh
        bash scripts/evaluate.sh
        ;;
    6)
        echo -e "\n${YELLOW}Quick baseline evaluation...${NC}\n"
        bash scripts/generate.sh --quick-baseline
        bash scripts/evaluate.sh --auto
        ;;
    7)
        echo -e "\n${YELLOW}Quick SFT evaluation...${NC}\n"
        bash scripts/generate.sh --quick-sft
        bash scripts/evaluate.sh --auto
        ;;
    0)
        echo "Bye!"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac
