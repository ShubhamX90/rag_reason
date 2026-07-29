#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK_INPUT="${BENCHMARK_INPUT:-data/Benchmark Dataset/benchmark_final_sanitized.jsonl}"
DATASET_LABEL="${DATASET_LABEL:-benchmark_final}"
BENCHMARK_TRACE_MODES="${BENCHMARK_TRACE_MODES:-e2e oracle_conflict oracle_notes oracle_both}"
BUILD_FINAL_ONLY="${BUILD_FINAL_ONLY:-1}"
FINAL_ONLY_MODES="${FINAL_ONLY_MODES:-e2e}"

"$PYTHON_BIN" scripts/prepare_benchmark_inference.py \
  --input_jsonl "$BENCHMARK_INPUT" \
  --out_dir data \
  --prompts_dir prompts \
  --dataset_label "$DATASET_LABEL" \
  --prompt_profile default \
  --message_tag strict \
  --modes $BENCHMARK_TRACE_MODES

"$PYTHON_BIN" scripts/prepare_benchmark_inference.py \
  --input_jsonl "$BENCHMARK_INPUT" \
  --out_dir data \
  --prompts_dir prompts \
  --dataset_label "$DATASET_LABEL" \
  --prompt_profile runtime \
  --message_tag trace_text \
  --modes $BENCHMARK_TRACE_MODES

"$PYTHON_BIN" scripts/prepare_benchmark_inference.py \
  --input_jsonl "$BENCHMARK_INPUT" \
  --out_dir data \
  --prompts_dir prompts \
  --dataset_label "$DATASET_LABEL" \
  --prompt_profile minimal \
  --message_tag minimal \
  --modes $BENCHMARK_TRACE_MODES

if [ "$BUILD_FINAL_ONLY" = "1" ]; then
  "$PYTHON_BIN" scripts/prepare_benchmark_inference.py \
    --input_jsonl "$BENCHMARK_INPUT" \
    --out_dir data \
    --prompts_dir prompts \
    --dataset_label "$DATASET_LABEL" \
    --prompt_profile final_only \
    --message_tag final_only \
    --modes $FINAL_ONLY_MODES
fi
