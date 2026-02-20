#!/bin/bash
# ═══════════════════════════════════════════════════
# train.sh  –  Interactive QLoRA Fine-Tuning Launcher
# ═══════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════╗${NC}"
echo -e "${CYAN}║    QLoRA Fine-Tuning (v2)      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Model path ──
if [ -n "$1" ]; then
    MODEL_PATH="$1"
else
    read -p "Base model path: " MODEL_PATH
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_PATH${NC}"
    exit 1
fi

# ── Step 2: Training mode ──
echo ""
echo "Select training mode:"
echo "  1) E2E          – Model learns to predict everything"
echo "  2) Oracle        – Model gets gold conflict type, predicts rest"
echo "  3) Mixed         – Train on both E2E + Oracle data (recommended)"
echo ""
read -p "Choice [1-3]: " MODE_CHOICE

case $MODE_CHOICE in
    1)
        MODE="e2e"
        TRAIN_FILES="data/messages/train_e2e_messages.jsonl"
        VAL_FILE="data/messages/val_e2e_messages.jsonl"
        ;;
    2)
        MODE="oracle"
        TRAIN_FILES="data/messages/train_oracle_messages.jsonl"
        VAL_FILE="data/messages/val_oracle_messages.jsonl"
        ;;
    3)
        MODE="mixed"
        TRAIN_FILES="data/messages/train_e2e_messages.jsonl data/messages/train_oracle_messages.jsonl"
        VAL_FILE="data/messages/val_e2e_messages.jsonl"
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac

# ── Step 3: Run name ──
read -p "Run name (e.g., run1): " RUN_NAME
RUN_NAME="${RUN_NAME:-run1}"
OUT_DIR="checkpoints/sft_${MODE}_${RUN_NAME}"

# ── Step 4: Hyperparameters ──
echo ""
echo -e "${YELLOW}Hyperparameters (press Enter for defaults):${NC}"

read -p "  Epochs        [6]:  " EPOCHS;       EPOCHS="${EPOCHS:-6}"
read -p "  Learning rate [2e-4]: " LR;         LR="${LR:-2e-4}"
read -p "  Batch size    [1]:  " BSZ;          BSZ="${BSZ:-1}"
read -p "  Grad accum    [16]: " GRAD_ACCUM;   GRAD_ACCUM="${GRAD_ACCUM:-16}"
read -p "  Max seq len   [8192]: " MAX_LEN;    MAX_LEN="${MAX_LEN:-8192}"
read -p "  LoRA rank     [32]: " LORA_R;       LORA_R="${LORA_R:-32}"
read -p "  LoRA alpha    [64]: " LORA_ALPHA;   LORA_ALPHA="${LORA_ALPHA:-64}"
read -p "  NEFTune alpha [5.0]: " NEFT_ALPHA;  NEFT_ALPHA="${NEFT_ALPHA:-5.0}"
read -p "  Conflict wt   [3.0]: " CONF_WT;     CONF_WT="${CONF_WT:-3.0}"
read -p "  Early stop pat [4]:  " PATIENCE;     PATIENCE="${PATIENCE:-4}"

# ── Step 5: Confirm ──
echo ""
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo "  Mode:           $MODE"
echo "  Base model:     $MODEL_PATH"
echo "  Train files:    $TRAIN_FILES"
echo "  Val file:       $VAL_FILE"
echo "  Output:         $OUT_DIR"
echo "  Epochs:         $EPOCHS"
echo "  LR:             $LR"
echo "  Effective BSZ:  $((BSZ * GRAD_ACCUM))"
echo "  Max seq len:    $MAX_LEN"
echo "  LoRA r/alpha:   $LORA_R / $LORA_ALPHA"
echo "  NEFTune alpha:  $NEFT_ALPHA"
echo "  Conflict wt:    $CONF_WT"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""
read -p "Proceed? [y/n]: " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

# ── Step 6: Launch mode ──
echo ""
echo "How to run?"
echo "  1) Foreground  (see output directly)"
echo "  2) Tmux        (detachable session)"
echo "  3) Nohup       (background with log file)"
read -p "Choice [1-3]: " LAUNCH

TRAIN_CMD="python code/train/train_qlora.py \
  --base_model $MODEL_PATH \
  --train_jsonl $TRAIN_FILES \
  --val_jsonl $VAL_FILE \
  --out_dir $OUT_DIR \
  --epochs $EPOCHS \
  --lr $LR \
  --bsz $BSZ \
  --grad_accum $GRAD_ACCUM \
  --max_len $MAX_LEN \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --neftune_alpha $NEFT_ALPHA \
  --conflict_weight $CONF_WT \
  --patience $PATIENCE"

case $LAUNCH in
    1)
        echo -e "\n${GREEN}Starting training...${NC}\n"
        eval $TRAIN_CMD
        ;;
    2)
        SESSION="train_${MODE}_${RUN_NAME}"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo -e "${RED}Tmux session '$SESSION' already exists!${NC}"
            echo "  Attach: tmux attach -t $SESSION"
            exit 1
        fi
        tmux new-session -d -s "$SESSION" "cd $SCRIPT_DIR && $TRAIN_CMD"
        echo -e "${GREEN}✓ Training launched in tmux session: $SESSION${NC}"
        echo "  Attach:  tmux attach -t $SESSION"
        echo "  Detach:  Ctrl+B then D"
        ;;
    3)
        mkdir -p logs
        LOG="logs/train_${MODE}_${RUN_NAME}.log"
        nohup bash -c "cd $SCRIPT_DIR && $TRAIN_CMD" > "$LOG" 2>&1 &
        echo -e "${GREEN}✓ Training launched in background (PID: $!)${NC}"
        echo "  Monitor: tail -f $LOG"
        ;;
    *)
        echo "Invalid."
        exit 1
        ;;
esac

echo ""
echo "After training, best checkpoint will be at:"
echo "  $OUT_DIR/best_dev_f1/"
echo ""
echo "Next step: bash scripts/generate.sh"
