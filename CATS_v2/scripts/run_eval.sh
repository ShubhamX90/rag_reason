#!/bin/bash
# scripts/run_eval.sh
# Interactive evaluation runner for CATS v2.0

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================"
echo "CATS v2.0 - Interactive Evaluation"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run: ./scripts/setup.sh first"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please run: ./scripts/setup.sh first and configure API keys"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check for input file
echo -e "${BLUE}Select input file:${NC}"
echo ""

# List available JSONL files
JSONL_FILES=($(find data -name "*.jsonl" 2>/dev/null))

if [ ${#JSONL_FILES[@]} -eq 0 ]; then
    echo -e "${RED}No .jsonl files found in data/ directory${NC}"
    echo ""
    read -p "Enter path to input file: " INPUT_FILE
else
    echo "Available files:"
    for i in "${!JSONL_FILES[@]}"; do
        echo "  $((i+1)). ${JSONL_FILES[$i]}"
    done
    echo "  0. Enter custom path"
    echo ""
    
    read -p "Select file (1-${#JSONL_FILES[@]}, or 0): " FILE_CHOICE
    
    if [ "$FILE_CHOICE" == "0" ]; then
        read -p "Enter path to input file: " INPUT_FILE
    else
        FILE_INDEX=$((FILE_CHOICE-1))
        INPUT_FILE="${JSONL_FILES[$FILE_INDEX]}"
    fi
fi

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: File not found: $INPUT_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Input file: $INPUT_FILE${NC}"
echo ""

# Select output directory
echo -e "${BLUE}Select output directory:${NC}"
echo "  1. outputs (default)"
echo "  2. outputs/run_1"
echo "  3. outputs/run_2"
echo "  4. outputs/run_3"
echo "  5. outputs/run_4"
echo "  6. outputs/run_5"
echo "  0. Enter custom path"
echo ""

read -p "Select output directory (1-6, or 0): " OUTPUT_CHOICE

case $OUTPUT_CHOICE in
    1)
        OUTPUT_DIR="outputs"
        ;;
    2)
        OUTPUT_DIR="outputs/run_1"
        ;;
    3)
        OUTPUT_DIR="outputs/run_2"
        ;;
    4)
        OUTPUT_DIR="outputs/run_3"
        ;;
    5)
        OUTPUT_DIR="outputs/run_4"
        ;;
    6)
        OUTPUT_DIR="outputs/run_5"
        ;;
    0)
        read -p "Enter output directory path: " OUTPUT_DIR
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Using default outputs directory.${NC}"
        OUTPUT_DIR="outputs"
        ;;
esac

echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}"
echo ""

# Select judge committee
echo -e "${BLUE}Select judge committee:${NC}"
echo "  1. Default (Haiku + DeepSeek + Qwen) - Balanced cost/quality"
echo "  2. Conservative (Haiku + Qwen + Mistral-Free) - Lowest cost"
echo "  3. None (Single judge) - Fastest, no voting"
echo ""

read -p "Select committee (1-3): " COMMITTEE_CHOICE

case $COMMITTEE_CHOICE in
    1)
        COMMITTEE="default"
        echo -e "${GREEN}✓ Using default committee${NC}"
        ;;
    2)
        COMMITTEE="conservative"
        echo -e "${GREEN}✓ Using conservative committee${NC}"
        ;;
    3)
        COMMITTEE="none"
        echo -e "${YELLOW}⚠  Using single judge (no committee voting)${NC}"
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Using default.${NC}"
        COMMITTEE="default"
        ;;
esac
echo ""

# Ask about sample limit
echo -e "${BLUE}Limit number of samples? (for testing)${NC}"
read -p "Enter max samples (or press Enter for all): " MAX_SAMPLES

EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max-samples $MAX_SAMPLES"
    echo -e "${YELLOW}⚠  Will evaluate first $MAX_SAMPLES samples only${NC}"
fi
echo ""

# Ask about verbosity
read -p "Enable verbose logging? (y/N): " VERBOSE
if [[ $VERBOSE =~ ^[Yy]$ ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --verbose"
fi
echo ""

# Confirm and run
echo "========================================"
echo "Configuration Summary:"
echo "========================================"
echo "Input: $INPUT_FILE"
echo "Committee: $COMMITTEE"
echo "Output: $OUTPUT_DIR"
[ -n "$MAX_SAMPLES" ] && echo "Max Samples: $MAX_SAMPLES"
echo ""

read -p "Start evaluation? (Y/n): " CONFIRM
if [[ $CONFIRM =~ ^[Nn]$ ]]; then
    echo "Evaluation cancelled."
    exit 0
fi

echo ""
echo "========================================"
echo "Starting Evaluation..."
echo "========================================"
echo ""

# Run evaluation
python run_evaluation.py \
    --input "$INPUT_FILE" \
    --committee "$COMMITTEE" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo ""
echo "========================================"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo "========================================"
echo ""
echo "Results available in:"
echo "  📄 $OUTPUT_DIR/eval_report.md"
echo "  📊 $OUTPUT_DIR/detailed_results.json"
echo "  📝 logs/cats_eval.log"
echo ""
