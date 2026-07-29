#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ASSISTANT_TARGET_STYLE="${ASSISTANT_TARGET_STYLE:-trace_text}"
PROMPT_PROFILE="${PROMPT_PROFILE:-minimal}"
MESSAGE_TAG="${MESSAGE_TAG:-minimal}"
STAGEWISE_TRAIN_JSONL="${STAGEWISE_TRAIN_JSONL:-data/splits/stagewise_multi/train/stage3_final.jsonl}"
STAGEWISE_VAL_JSONL="${STAGEWISE_VAL_JSONL:-data/splits/stagewise_multi/val/stage3_final.jsonl}"
MONOLITHIC_TRAIN_JSONL="${MONOLITHIC_TRAIN_JSONL:-data/splits/monolithic_multi/train/monolithic_final.jsonl}"
MONOLITHIC_VAL_JSONL="${MONOLITHIC_VAL_JSONL:-data/splits/monolithic_multi/val/monolithic_final.jsonl}"

"$PYTHON_BIN" code/data/prepare_data.py \
  --stagewise_train_jsonl "$STAGEWISE_TRAIN_JSONL" \
  --stagewise_val_jsonl "$STAGEWISE_VAL_JSONL" \
  --monolithic_train_jsonl "$MONOLITHIC_TRAIN_JSONL" \
  --monolithic_val_jsonl "$MONOLITHIC_VAL_JSONL" \
  --out_dir data \
  --prompts_dir prompts \
  --prompt_profile "$PROMPT_PROFILE" \
  --message_tag "$MESSAGE_TAG" \
  --assistant_target_style "$ASSISTANT_TARGET_STYLE" \
  --train_tasks e2e_trace \
  --val_tasks e2e_trace
