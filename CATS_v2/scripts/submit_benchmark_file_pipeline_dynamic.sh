#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$ROOT_DIR/inputs/prepped_model_eval_inputs/benchmark_set_all_modes}"
USER_NAME="${USER:-$(whoami)}"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /abs/path/to/input.jsonl" >&2
  exit 1
fi

INPUT_FILE="$1"

: "${QWEN_BASE_URL:?Set QWEN_BASE_URL to the live Qwen endpoint, e.g. http://gpunodeX:8001/v1}"
: "${MISTRAL_BASE_URL:?Set MISTRAL_BASE_URL to the live Mistral endpoint, e.g. http://gpunodeY:8004/v1}"
: "${DEEPSEEK_BASE_URL:?Set DEEPSEEK_BASE_URL to the live DeepSeek endpoint, e.g. http://gpunodeZ:8002/v1}"

GOLD_FILE="${GOLD_FILE:-$ROOT_DIR/data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-736}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER_NAME}/rag-reason/cats_outputs/benchmark_local_committee_3judge}"
RUN_LABEL="${RUN_LABEL:-}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_RETRIES="${MAX_RETRIES:-2}"

exec python3 "$ROOT_DIR/scripts/watch_benchmark_file_pipeline.py" \
  "$INPUT_FILE" \
  --input-root "$INPUT_ROOT" \
  --gold-file "$GOLD_FILE" \
  --expected-rows "$EXPECTED_ROWS" \
  --output-root "$OUTPUT_ROOT" \
  --run-label "$RUN_LABEL" \
  --poll-interval "$POLL_INTERVAL" \
  --max-retries "$MAX_RETRIES" \
  --qwen-base-url "$QWEN_BASE_URL" \
  --mistral-base-url "$MISTRAL_BASE_URL" \
  --deepseek-base-url "$DEEPSEEK_BASE_URL"
