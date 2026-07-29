#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK_INPUT="${BENCHMARK_INPUT:-data/splits/benchmark_final_v2_holdout_clean_736.jsonl}"
DATASET_LABEL="${DATASET_LABEL:-benchmark_final_v2_holdout_clean_736}"
BENCHMARK_TRACE_MODES="${BENCHMARK_TRACE_MODES:-e2e oracle_conflict oracle_notes oracle_both}"

echo "===== Rebuild benchmark messages: heldout 736 matrix ====="
echo "BENCHMARK_INPUT=$BENCHMARK_INPUT"
echo "DATASET_LABEL=$DATASET_LABEL"
echo "BENCHMARK_TRACE_MODES=$BENCHMARK_TRACE_MODES"

test -f "$BENCHMARK_INPUT"

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

echo "===== Benchmark message counts ====="
for mode in e2e oracle_conflict oracle_notes oracle_both; do
  for tag in strict trace_text minimal; do
    path="data/messages/${DATASET_LABEL}_${mode}_${tag}_messages.jsonl"
    if [ -f "$path" ]; then
      wc -l "$path"
    fi
  done
done
