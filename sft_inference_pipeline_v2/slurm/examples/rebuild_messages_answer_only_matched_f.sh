#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_JSONL="${OUT_JSONL:-data/messages/train_stagewise_answer_only_matched_f_messages.jsonl}"
VAL_MINIMAL_JSONL="${VAL_MINIMAL_JSONL:-data/messages/val_stagewise_answer_only_minimal_messages.jsonl}"
VAL_FINAL_ONLY_JSONL="${VAL_FINAL_ONLY_JSONL:-data/messages/val_stagewise_answer_only_final_only_messages.jsonl}"
STAGEWISE_TRAIN_JSONL="${STAGEWISE_TRAIN_JSONL:-data/splits/train.jsonl}"
STAGEWISE_VAL_JSONL="${STAGEWISE_VAL_JSONL:-data/splits/val.jsonl}"
MONOLITHIC_TRAIN_JSONL="${MONOLITHIC_TRAIN_JSONL:-}"
MONOLITHIC_VAL_JSONL="${MONOLITHIC_VAL_JSONL:-}"

prepare_args=(
  --stagewise_train_jsonl "$STAGEWISE_TRAIN_JSONL"
  --stagewise_val_jsonl "$STAGEWISE_VAL_JSONL"
  --out_dir data
  --prompts_dir prompts
)

if [ -n "$MONOLITHIC_TRAIN_JSONL" ] && [ -n "$MONOLITHIC_VAL_JSONL" ]; then
  prepare_args+=(
    --monolithic_train_jsonl "$MONOLITHIC_TRAIN_JSONL"
    --monolithic_val_jsonl "$MONOLITHIC_VAL_JSONL"
  )
fi

# Build final-answer-only targets under the explicit final-only prompt.
"$PYTHON_BIN" code/data/prepare_data.py \
  "${prepare_args[@]}" \
  --prompt_profile final_only \
  --message_tag final_only \
  --assistant_target_style trace_text \
  --train_tasks e2e_trace answer_only \
  --val_tasks e2e_trace answer_only

# Build final-answer-only targets under the true minimal prompt.
"$PYTHON_BIN" code/data/prepare_data.py \
  "${prepare_args[@]}" \
  --prompt_profile minimal \
  --message_tag minimal \
  --assistant_target_style trace_text \
  --train_tasks e2e_trace answer_only \
  --val_tasks e2e_trace answer_only

# Keep the legacy answer-only mixture shape:
#   final-only answer rows: N * FINAL_ONLY_WEIGHT
#   minimal answer rows:    N * MINIMAL_WEIGHT
"$PYTHON_BIN" scripts/build_answer_only_sft_messages.py \
  --final-only-input data/messages/train_stagewise_e2e_answer_only_final_only_messages.jsonl \
  --minimal-input data/messages/train_stagewise_e2e_answer_only_minimal_messages.jsonl \
  --output "$OUT_JSONL" \
  --final-only-weight "${FINAL_ONLY_WEIGHT:-8}" \
  --minimal-weight "${MINIMAL_WEIGHT:-4}"

"$PYTHON_BIN" scripts/build_answer_only_sft_messages.py \
  --final-only-input data/messages/val_stagewise_e2e_answer_only_final_only_messages.jsonl \
  --minimal-input data/messages/val_stagewise_e2e_answer_only_minimal_messages.jsonl \
  --output "$VAL_MINIMAL_JSONL" \
  --final-only-weight 0 \
  --minimal-weight 1

"$PYTHON_BIN" scripts/build_answer_only_sft_messages.py \
  --final-only-input data/messages/val_stagewise_e2e_answer_only_final_only_messages.jsonl \
  --minimal-input data/messages/val_stagewise_e2e_answer_only_minimal_messages.jsonl \
  --output "$VAL_FINAL_ONLY_JSONL" \
  --final-only-weight 1 \
  --minimal-weight 0

"$PYTHON_BIN" scripts/check_trace_text_messages.py \
  --forbid_think \
  --forbid_task_prefix \
  "$OUT_JSONL" \
  "$VAL_MINIMAL_JSONL" \
  "$VAL_FINAL_ONLY_JSONL"
