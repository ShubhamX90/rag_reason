#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_JSONL="${OUT_JSONL:-data/messages/train_stagewise_prompt_robust_trace_text_d_messages.jsonl}"

PROMPT_PROFILE=default \
MESSAGE_TAG=strict \
ASSISTANT_TARGET_STYLE=trace_text \
TRAIN_TASKS=e2e_trace \
VAL_TASKS=e2e_trace \
"$PYTHON_BIN" code/data/prepare_data.py \
  --stagewise_train_jsonl data/splits/stagewise_multi/train/stage3_final.jsonl \
  --stagewise_val_jsonl data/splits/stagewise_multi/val/stage3_final.jsonl \
  --monolithic_train_jsonl data/splits/monolithic_multi/train/monolithic_final.jsonl \
  --monolithic_val_jsonl data/splits/monolithic_multi/val/monolithic_final.jsonl \
  --out_dir data \
  --prompts_dir prompts \
  --prompt_profile default \
  --message_tag strict \
  --assistant_target_style trace_text \
  --train_tasks e2e_trace \
  --val_tasks e2e_trace

PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_trace_text_multitask.sh
PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_minimal_inference.sh

"$PYTHON_BIN" scripts/build_prompt_robust_messages.py \
  --strict-input data/messages/train_stagewise_e2e_strict_messages.jsonl \
  --runtime-input data/messages/train_stagewise_multitask_trace_text_messages.jsonl \
  --minimal-input data/messages/train_stagewise_e2e_minimal_messages.jsonl \
  --output "$OUT_JSONL" \
  --strict-e2e-weight "${STRICT_E2E_WEIGHT:-2}" \
  --runtime-task-weight e2e_trace=1 \
  --runtime-task-weight doc_verdict=1 \
  --runtime-task-weight conflict_type=2 \
  --runtime-task-weight answer_only=1 \
  --minimal-e2e-weight "${MINIMAL_E2E_WEIGHT:-4}"

"$PYTHON_BIN" scripts/check_trace_text_messages.py "$OUT_JSONL"
"$PYTHON_BIN" scripts/check_trace_text_messages.py --require_think data/messages/train_stagewise_e2e_strict_messages.jsonl
"$PYTHON_BIN" scripts/check_trace_text_messages.py --require_think data/messages/val_stagewise_e2e_trace_text_messages.jsonl
"$PYTHON_BIN" scripts/check_trace_text_messages.py --require_think data/messages/val_stagewise_e2e_minimal_messages.jsonl
