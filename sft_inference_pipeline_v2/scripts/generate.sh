#!/bin/bash
# ═══════════════════════════════════════════════════
# generate.sh  –  Interactive Generation Launcher
# ═══════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# Quick modes from run.sh
if [ "$1" == "--quick-baseline" ] || [ "$1" == "--quick-sft" ]; then
    # These get handled by the evaluate step; just prompt for needed info
    :
fi

echo -e "${CYAN}╔════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    Model Generation / Inference     ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Base model ──
read -p "Base model path: " MODEL_PATH
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found: $MODEL_PATH${NC}"
    exit 1
fi

# ── Step 2: Model type ──
echo ""
echo "Model type:"
echo "  1) SFT (fine-tuned with LoRA adapter)"
echo "  2) Baseline (untuned model)"
read -p "Choice [1-2]: " MODEL_TYPE

LORA_ARG=""
TAG=""
if [ "$MODEL_TYPE" == "1" ]; then
    read -p "LoRA adapter directory: " LORA_DIR
    if [ ! -d "$LORA_DIR" ]; then
        echo -e "${RED}Error: LoRA dir not found: $LORA_DIR${NC}"
        exit 1
    fi
    LORA_ARG="--lora_dir $LORA_DIR"
    TAG="sft"
else
    TAG="baseline"
fi

# ── Step 3: Prompt mode ──
echo ""
echo "Prompt mode:"
echo "  1) E2E     – model predicts everything"
echo "  2) Oracle  – gold conflict type provided"
echo "  3) Both    – run E2E and Oracle sequentially"
read -p "Choice [1-3]: " PROMPT_MODE

# ── Step 4: Data split ──
echo ""
echo "Data split:"
echo "  1) test"
echo "  2) val"
read -p "Choice [1-2]: " SPLIT_CHOICE
case $SPLIT_CHOICE in
    1) SPLIT="test" ;;
    2) SPLIT="val" ;;
    *) echo "Invalid."; exit 1 ;;
esac

# ── Step 5: Build job list ──
JOBS=()
case $PROMPT_MODE in
    1) JOBS+=("e2e") ;;
    2) JOBS+=("oracle") ;;
    3) JOBS+=("e2e" "oracle") ;;
    *) echo "Invalid."; exit 1 ;;
esac

# ── Step 6: Confirm and run ──
echo ""
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo "  Model:    $MODEL_PATH"
echo "  Type:     $TAG"
echo "  Modes:    ${JOBS[*]}"
echo "  Split:    $SPLIT"
if [ -n "$LORA_ARG" ]; then
    echo "  LoRA:     $LORA_DIR"
fi
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""
read -p "Proceed? [y/n]: " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

mkdir -p outputs

for MODE in "${JOBS[@]}"; do
    INPUT="data/messages/${SPLIT}_${MODE}_messages.jsonl"
    OUT_RAW="outputs/${TAG}_${MODE}_${SPLIT}.raw.jsonl"
    OUT_SAN="outputs/${TAG}_${MODE}_${SPLIT}.sanitized.jsonl"

    if [ ! -f "$INPUT" ]; then
        echo -e "${RED}Error: Input not found: $INPUT${NC}"
        echo "Run: bash scripts/prepare_data.sh first"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}[1/2] Generating: ${TAG} / ${MODE} / ${SPLIT}${NC}"
    echo ""

    python code/eval/generate.py \
        --base_model "$MODEL_PATH" \
        $LORA_ARG \
        --input_jsonl "$INPUT" \
        --out_jsonl "$OUT_RAW" \
        --auto_length \
        --max_new_tokens_base 1200 \
        --max_new_tokens_cap 2200 \
        --load_in_4bit \
        --temperature 0.0 \
        --resume

    echo ""
    echo -e "${GREEN}[2/2] Sanitizing...${NC}"

    python code/eval/sanitize.py \
        --in_jsonl "$OUT_RAW" \
        --out_jsonl "$OUT_SAN"

    echo ""
    echo -e "${GREEN}✓ Done: $OUT_SAN${NC}"
done

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ Generation complete!                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Outputs in: outputs/"
ls -la outputs/*.sanitized.jsonl 2>/dev/null || echo "(no sanitized files yet)"
echo ""
echo "Next step: bash scripts/evaluate.sh"
