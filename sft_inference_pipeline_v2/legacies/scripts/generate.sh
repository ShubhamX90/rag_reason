#!/bin/bash
# ================================================
# generate.sh - Interactive Generation Launcher
# ================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
PYTHON_BIN="${PYTHON_BIN:-python3}"

QUICK_MODE="$1"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}       Model Generation / Inference     ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

read -p "Base model path: " MODEL_PATH
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found: $MODEL_PATH${NC}"
    exit 1
fi

MODEL_SLUG="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"

echo ""
echo "Model type:"
echo "  1) SFT (fine-tuned with LoRA adapter)"
echo "  2) Baseline (untuned model)"
if [ "$QUICK_MODE" == "--quick-baseline" ]; then
    MODEL_TYPE="2"
elif [ "$QUICK_MODE" == "--quick-sft" ]; then
    MODEL_TYPE="1"
else
    read -p "Choice [1-2]: " MODEL_TYPE
fi

LORA_ARG=""
TAG=""
if [ "$MODEL_TYPE" == "1" ]; then
    read -p "LoRA adapter directory: " LORA_DIR
    if [ ! -d "$LORA_DIR" ]; then
        echo -e "${RED}Error: LoRA dir not found: $LORA_DIR${NC}"
        exit 1
    fi
    LORA_ARG="--lora_dir $LORA_DIR"
    ADAPTER_SLUG="$(basename "$LORA_DIR" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"
    TAG="sft_${MODEL_SLUG}_${ADAPTER_SLUG}"
else
    TAG="baseline_${MODEL_SLUG}"
fi

echo ""
echo "Prompt mode:"
echo "  1) e2e"
echo "  2) oracle_conflict"
echo "  3) oracle_notes"
echo "  4) oracle_both"
echo "  5) all_oracles"
echo "  6) all_modes"
if [ -n "$QUICK_MODE" ]; then
    PROMPT_MODE="1"
else
    read -p "Choice [1-6]: " PROMPT_MODE
fi

JOBS=()
case $PROMPT_MODE in
    1) JOBS+=("e2e") ;;
    2) JOBS+=("oracle_conflict") ;;
    3) JOBS+=("oracle_notes") ;;
    4) JOBS+=("oracle_both") ;;
    5) JOBS+=("oracle_conflict" "oracle_notes" "oracle_both") ;;
    6) JOBS+=("e2e" "oracle_conflict" "oracle_notes" "oracle_both") ;;
    *) echo "Invalid."; exit 1 ;;
esac

echo ""
echo "Dataset preset:"
echo "  1) val_stagewise"
echo "  2) val_monolithic"
echo "  3) custom label / custom paths"
read -p "Choice [1-3]: " DATASET_CHOICE
DATASET_CHOICE="${DATASET_CHOICE:-1}"
case $DATASET_CHOICE in
    1) DATASET_LABEL="val_stagewise" ;;
    2) DATASET_LABEL="val_monolithic" ;;
    3)
        read -p "Dataset label for filenames: " DATASET_LABEL
        if [ -z "$DATASET_LABEL" ]; then
            echo "Dataset label is required."
            exit 1
        fi
        ;;
    *)
        echo "Invalid."
        exit 1
        ;;
esac

echo ""
echo "Prompt profile:"
echo "  1) default (current strict / upper-bound prompts)"
echo "  2) minimal (bare prompt; tests SFT-internalized behavior)"
echo "  3) runtime (recommended lightweight 3-stage trace prompts)"
echo "  4) final_only (baseline final-answer-only prompts)"
echo "  5) legacy_text_contract (old strict JSON/text contract)"
read -p "Choice [1-5]: " PROMPT_PROFILE_CHOICE
PROMPT_PROFILE_CHOICE="${PROMPT_PROFILE_CHOICE:-1}"
CONTRACT_MODE="trace"
case $PROMPT_PROFILE_CHOICE in
    1)
        PROMPT_PROFILE="default"
        MESSAGE_SUFFIX=""
        ;;
    2)
        PROMPT_PROFILE="minimal"
        MESSAGE_SUFFIX="_minimal"
        ;;
    3)
        PROMPT_PROFILE="runtime"
        MESSAGE_SUFFIX="_runtime"
        ;;
    4)
        PROMPT_PROFILE="final_only"
        MESSAGE_SUFFIX="_final_only"
        CONTRACT_MODE="final"
        ;;
    5)
        PROMPT_PROFILE="legacy_text_contract"
        MESSAGE_SUFFIX="_legacy_text_contract"
        ;;
    *)
        echo "Invalid."
        exit 1
        ;;
esac

echo ""
echo "Inference precision:"
echo "  1) bf16 full model (recommended if memory allows)"
echo "  2) 4-bit quantized"
echo "  3) fp16 full model"
read -p "Choice [1-3]: " PREC_CHOICE
PREC_CHOICE="${PREC_CHOICE:-1}"
PREC_ARGS=()
case $PREC_CHOICE in
    1) PREC_ARGS+=(--dtype bf16) ;;
    2) PREC_ARGS+=(--load_in_4bit) ;;
    3) PREC_ARGS+=(--dtype fp16) ;;
    *) echo "Invalid."; exit 1 ;;
esac

echo "Expected default message files:"
for MODE in "${JOBS[@]}"; do
    echo "  data/messages/${DATASET_LABEL}_${MODE}${MESSAGE_SUFFIX}_messages.jsonl"
done
echo ""
read -p "Use those default paths? [y/n]: " USE_DEFAULT_PATHS
USE_DEFAULT_PATHS="${USE_DEFAULT_PATHS:-y}"

mkdir -p outputs

declare -A INPUTS
for MODE in "${JOBS[@]}"; do
    DEFAULT_INPUT="data/messages/${DATASET_LABEL}_${MODE}${MESSAGE_SUFFIX}_messages.jsonl"
    if [ "$USE_DEFAULT_PATHS" == "y" ]; then
        INPUTS[$MODE]="$DEFAULT_INPUT"
    else
        read -p "Input JSONL for ${MODE}: " INPUTS[$MODE]
    fi
    if [ ! -f "${INPUTS[$MODE]}" ]; then
        echo -e "${RED}Error: Input not found: ${INPUTS[$MODE]}${NC}"
        exit 1
    fi
done

echo ""
echo -e "${CYAN}========================================${NC}"
echo "  Model:         $MODEL_PATH"
echo "  Type:          $TAG"
echo "  Dataset label: $DATASET_LABEL"
echo "  Prompt profile:$PROMPT_PROFILE"
echo "  Contract mode: $CONTRACT_MODE"
echo "  Modes:         ${JOBS[*]}"
echo "  Precision:     ${PREC_ARGS[*]}"
if [ -n "$LORA_ARG" ]; then
    echo "  LoRA:          $LORA_DIR"
fi
echo -e "${CYAN}========================================${NC}"
echo ""
read -p "Proceed? [y/n]: " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

for MODE in "${JOBS[@]}"; do
    INPUT="${INPUTS[$MODE]}"
    MODE_LABEL="${MODE}${MESSAGE_SUFFIX}"
    OUT_RAW="outputs/${TAG}_${MODE_LABEL}_${DATASET_LABEL}.raw.jsonl"
    OUT_SAN="outputs/${TAG}_${MODE_LABEL}_${DATASET_LABEL}.sanitized.jsonl"

    echo ""
    echo -e "${GREEN}[1/2] Generating: ${TAG} / ${MODE} / ${DATASET_LABEL}${NC}"

    GEN_CMD=(
        "$PYTHON_BIN" code/eval/generate.py
        --base_model "$MODEL_PATH"
        --input_jsonl "$INPUT"
        --out_jsonl "$OUT_RAW"
        --auto_length
        --max_new_tokens_base 1200
        --max_new_tokens_cap 2200
        --temperature 0.0
        --contract_mode "$CONTRACT_MODE"
        --resume
    )
    if [ -n "$LORA_ARG" ]; then
        GEN_CMD+=(--lora_dir "$LORA_DIR")
    fi
    GEN_CMD+=("${PREC_ARGS[@]}")
    "${GEN_CMD[@]}"

    echo ""
    echo -e "${GREEN}[2/2] Sanitizing...${NC}"

    CANON_ARG=()
    if [ -f "data/splits/${DATASET_LABEL}.jsonl" ]; then
        CANON_ARG=(--canon_jsonl "data/splits/${DATASET_LABEL}.jsonl")
    fi

    SAN_CMD=(
        "$PYTHON_BIN" code/eval/sanitize.py
        --in_jsonl "$OUT_RAW"
        --out_jsonl "$OUT_SAN"
    )
    SAN_CMD+=("${CANON_ARG[@]}")
    "${SAN_CMD[@]}"

    echo ""
    echo -e "${GREEN}Done: $OUT_SAN${NC}"
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Generation complete                   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Outputs in: outputs/"
ls -la outputs/*.sanitized.jsonl 2>/dev/null || echo "(no sanitized files yet)"
