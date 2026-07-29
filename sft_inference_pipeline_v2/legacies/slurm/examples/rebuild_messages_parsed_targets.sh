#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

ASSISTANT_TARGET_STYLE="${ASSISTANT_TARGET_STYLE:-parsed}"
PROMPT_PROFILE="${PROMPT_PROFILE:-default}"
MESSAGE_TAG="${MESSAGE_TAG:-}"

python code/data/prepare_data.py \
  --stagewise_train_jsonl data/splits/stagewise_multi/train/stage3_final.jsonl \
  --stagewise_val_jsonl data/splits/stagewise_multi/val/stage3_final.jsonl \
  --monolithic_train_jsonl data/splits/monolithic_multi/train/monolithic_final.jsonl \
  --monolithic_val_jsonl data/splits/monolithic_multi/val/monolithic_final.jsonl \
  --out_dir data \
  --prompts_dir prompts \
  --prompt_profile "$PROMPT_PROFILE" \
  --message_tag "$MESSAGE_TAG" \
  --assistant_target_style "$ASSISTANT_TARGET_STYLE"
