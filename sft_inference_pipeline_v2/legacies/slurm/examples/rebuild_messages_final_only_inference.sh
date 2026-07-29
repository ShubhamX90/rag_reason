#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

ASSISTANT_TARGET_STYLE="${ASSISTANT_TARGET_STYLE:-trace_text}"
PROMPT_PROFILE="${PROMPT_PROFILE:-final_only}"
MESSAGE_TAG="${MESSAGE_TAG:-final_only}"

python code/data/prepare_data.py \
  --stagewise_train_jsonl data/splits/stagewise_multi/train/stage3_final.jsonl \
  --stagewise_val_jsonl data/splits/stagewise_multi/val/stage3_final.jsonl \
  --monolithic_train_jsonl data/splits/monolithic_multi/train/monolithic_final.jsonl \
  --monolithic_val_jsonl data/splits/monolithic_multi/val/monolithic_final.jsonl \
  --out_dir data \
  --prompts_dir prompts \
  --prompt_profile "$PROMPT_PROFILE" \
  --message_tag "$MESSAGE_TAG" \
  --assistant_target_style "$ASSISTANT_TARGET_STYLE" \
  --train_tasks e2e_trace \
  --val_tasks e2e_trace
