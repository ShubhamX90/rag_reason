#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_LABEL="${DATASET_LABEL:-val_stagewise}"
STAGEWISE_TRAIN_JSONL="${STAGEWISE_TRAIN_JSONL:-data/splits/stagewise_multi/train/stage3_final.jsonl}"
STAGEWISE_VAL_JSONL="${STAGEWISE_VAL_JSONL:-data/splits/stagewise_multi/val/stage3_final.jsonl}"
VAL_MODES="${VAL_MODES:-e2e oracle_conflict oracle_notes oracle_both}"

if [ "$DATASET_LABEL" != "val_stagewise" ]; then
  echo "rebuild_val_messages_f_boundary_guarded.sh only supports DATASET_LABEL=val_stagewise." >&2
  exit 1
fi

if [ ! -f "$STAGEWISE_TRAIN_JSONL" ]; then
  echo "Stagewise train split not found: $STAGEWISE_TRAIN_JSONL" >&2
  exit 1
fi

if [ ! -f "$STAGEWISE_VAL_JSONL" ]; then
  echo "Stagewise val split not found: $STAGEWISE_VAL_JSONL" >&2
  exit 1
fi

"$PYTHON_BIN" code/data/prepare_data.py \
  --stagewise_train_jsonl "$STAGEWISE_TRAIN_JSONL" \
  --stagewise_val_jsonl "$STAGEWISE_VAL_JSONL" \
  --out_dir data \
  --prompts_dir prompts \
  --prompt_profile default \
  --message_tag strict \
  --val_modes $VAL_MODES \
  --train_modes e2e \
  --train_tasks e2e_trace \
  --val_tasks e2e_trace

"$PYTHON_BIN" code/data/prepare_data.py \
  --stagewise_train_jsonl "$STAGEWISE_TRAIN_JSONL" \
  --stagewise_val_jsonl "$STAGEWISE_VAL_JSONL" \
  --out_dir data \
  --prompts_dir prompts \
  --prompt_profile runtime \
  --message_tag trace_text \
  --val_modes $VAL_MODES \
  --train_modes e2e \
  --train_tasks e2e_trace \
  --val_tasks e2e_trace

"$PYTHON_BIN" code/data/prepare_data.py \
  --stagewise_train_jsonl "$STAGEWISE_TRAIN_JSONL" \
  --stagewise_val_jsonl "$STAGEWISE_VAL_JSONL" \
  --out_dir data \
  --prompts_dir prompts \
  --prompt_profile minimal \
  --message_tag minimal \
  --val_modes $VAL_MODES \
  --train_modes e2e \
  --train_tasks e2e_trace \
  --val_tasks e2e_trace
