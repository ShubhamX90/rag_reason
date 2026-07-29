#!/bin/bash
# ================================================
# train.sh - Interactive QLoRA Fine-Tuning Launcher
# ================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}      QLoRA Fine-Tuning (E2E only)      ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

TRAIN_STAGEWISE="data/messages/train_stagewise_e2e_messages.jsonl"
TRAIN_MONOLITHIC="data/messages/train_monolithic_e2e_messages.jsonl"
VAL_STAGEWISE="data/messages/val_stagewise_e2e_messages.jsonl"
VAL_MONOLITHIC="data/messages/val_monolithic_e2e_messages.jsonl"

if [ ! -f "$TRAIN_STAGEWISE" ] || [ ! -f "$VAL_STAGEWISE" ]; then
    echo -e "${RED}Missing prepared stagewise e2e message files.${NC}"
    echo "Run: bash scripts/prepare_data.sh"
    exit 1
fi

echo ""
echo "Training dataset:"
echo "  1) stagewise (recommended)"
if [ -f "$TRAIN_MONOLITHIC" ]; then
    echo "  2) monolithic"
fi
read -p "Choice [1]: " TRAIN_CHOICE
TRAIN_CHOICE="${TRAIN_CHOICE:-1}"
case $TRAIN_CHOICE in
    1)
        TRAIN_FILES="$TRAIN_STAGEWISE"
        TRAIN_DESC="stagewise"
        DEFAULT_VAL_FILE="$VAL_STAGEWISE"
        DEFAULT_VAL_DESC="stagewise"
        ;;
    2)
        if [ -f "$TRAIN_MONOLITHIC" ] && [ -f "$VAL_MONOLITHIC" ]; then
            TRAIN_FILES="$TRAIN_MONOLITHIC"
            TRAIN_DESC="monolithic"
            DEFAULT_VAL_FILE="$VAL_MONOLITHIC"
            DEFAULT_VAL_DESC="monolithic"
        else
            echo "Invalid."
            exit 1
        fi
        ;;
    *)
        echo "Invalid."
        exit 1
        ;;
esac

echo ""
echo "Validation set:"
echo "  1) Match training dataset (${DEFAULT_VAL_DESC}, recommended)"
if [ "$TRAIN_DESC" == "stagewise" ] && [ -f "$VAL_MONOLITHIC" ]; then
    echo "  2) monolithic"
elif [ "$TRAIN_DESC" == "monolithic" ] && [ -f "$VAL_STAGEWISE" ]; then
    echo "  2) stagewise"
fi
read -p "Choice [1]: " VAL_CHOICE
VAL_CHOICE="${VAL_CHOICE:-1}"
case $VAL_CHOICE in
    1)
        VAL_FILE="$DEFAULT_VAL_FILE"
        VAL_DESC="$DEFAULT_VAL_DESC"
        ;;
    2)
        if [ "$TRAIN_DESC" == "stagewise" ] && [ -f "$VAL_MONOLITHIC" ]; then
            VAL_FILE="$VAL_MONOLITHIC"
            VAL_DESC="monolithic"
        elif [ "$TRAIN_DESC" == "monolithic" ] && [ -f "$VAL_STAGEWISE" ]; then
            VAL_FILE="$VAL_STAGEWISE"
            VAL_DESC="stagewise"
        else
            echo "Invalid."
            exit 1
        fi
        ;;
    *)
        echo "Invalid."
        exit 1
        ;;
esac

if [ -n "$1" ]; then
    MODEL_PATH="$1"
else
    read -p "Base model path: " MODEL_PATH
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_PATH${NC}"
    exit 1
fi

MODEL_SLUG="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"

read -p "Run name (e.g., run1): " RUN_NAME
RUN_NAME="${RUN_NAME:-run1}"
DEFAULT_OUT_DIR="checkpoints/${MODEL_SLUG}_${TRAIN_DESC}_e2e_${RUN_NAME}"
read -p "Output dir [$DEFAULT_OUT_DIR]: " OUT_DIR
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"

echo ""
echo -e "${YELLOW}Hyperparameters (press Enter for defaults):${NC}"
read -p "  Epochs         [6]: " EPOCHS;        EPOCHS="${EPOCHS:-6}"
read -p "  Learning rate  [2e-4]: " LR;         LR="${LR:-2e-4}"
read -p "  Batch size     [1]: " BSZ;          BSZ="${BSZ:-1}"
read -p "  Grad accum     [16]: " GRAD_ACCUM;  GRAD_ACCUM="${GRAD_ACCUM:-16}"
read -p "  Max seq len    [8192]: " MAX_LEN;    MAX_LEN="${MAX_LEN:-8192}"
read -p "  LoRA rank      [32]: " LORA_R;      LORA_R="${LORA_R:-32}"
read -p "  LoRA alpha     [64]: " LORA_ALPHA;  LORA_ALPHA="${LORA_ALPHA:-64}"
read -p "  NEFTune alpha  [5.0]: " NEFT_ALPHA; NEFT_ALPHA="${NEFT_ALPHA:-5.0}"
read -p "  Conflict wt    [3.0]: " CONF_WT;    CONF_WT="${CONF_WT:-3.0}"
read -p "  Early stop pat [4]: " PATIENCE;    PATIENCE="${PATIENCE:-4}"

echo ""
echo -e "${CYAN}========================================${NC}"
echo "  Mode:           e2e only"
echo "  Base model:     $MODEL_PATH"
echo "  Train data:     $TRAIN_DESC"
echo "  Train files:    $TRAIN_FILES"
echo "  Val file:       $VAL_FILE ($VAL_DESC)"
echo "  Output:         $OUT_DIR"
echo "  Epochs:         $EPOCHS"
echo "  LR:             $LR"
echo "  Effective BSZ:  $((BSZ * GRAD_ACCUM))"
echo "  Max seq len:    $MAX_LEN"
echo "  LoRA r/alpha:   $LORA_R / $LORA_ALPHA"
echo "  NEFTune alpha:  $NEFT_ALPHA"
echo "  Conflict wt:    $CONF_WT"
echo -e "${CYAN}========================================${NC}"
echo ""
read -p "Proceed? [y/n]: " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "How to run?"
echo "  1) Foreground"
echo "  2) Tmux"
echo "  3) Nohup"
read -p "Choice [1-3]: " LAUNCH

TRAIN_CMD=(
  "$PYTHON_BIN" code/train/train_qlora.py
  --base_model "$MODEL_PATH"
  --train_jsonl "$TRAIN_FILES"
  --val_jsonl "$VAL_FILE"
  --out_dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --lr "$LR"
  --bsz "$BSZ"
  --grad_accum "$GRAD_ACCUM"
  --max_len "$MAX_LEN"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --neftune_alpha "$NEFT_ALPHA"
  --conflict_weight "$CONF_WT"
  --patience "$PATIENCE"
)
printf -v TRAIN_CMD_STR '%q ' "${TRAIN_CMD[@]}"
printf -v SCRIPT_DIR_Q '%q' "$SCRIPT_DIR"

case $LAUNCH in
    1)
        echo -e "\n${GREEN}Starting training...${NC}\n"
        "${TRAIN_CMD[@]}"
        ;;
    2)
        SESSION="train_e2e_${RUN_NAME}"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo -e "${RED}Tmux session '$SESSION' already exists.${NC}"
            echo "Attach: tmux attach -t $SESSION"
            exit 1
        fi
        tmux new-session -d -s "$SESSION" "cd $SCRIPT_DIR_Q && $TRAIN_CMD_STR"
        echo -e "${GREEN}Training launched in tmux session: $SESSION${NC}"
        echo "Attach: tmux attach -t $SESSION"
        ;;
    3)
        mkdir -p logs
        LOG="logs/train_e2e_${RUN_NAME}.log"
        nohup bash -lc "cd $SCRIPT_DIR_Q && $TRAIN_CMD_STR" > "$LOG" 2>&1 &
        echo -e "${GREEN}Training launched in background (PID: $!)${NC}"
        echo "Monitor: tail -f $LOG"
        ;;
    *)
        echo "Invalid."
        exit 1
        ;;
esac

echo ""
echo "Best checkpoint will be saved under:"
echo "  $OUT_DIR/best_dev_f1/"
echo ""
echo "Next step: bash scripts/generate.sh"
