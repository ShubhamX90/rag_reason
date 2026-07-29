#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_K_SPLIT_DIR="${RUN_K_SPLIT_DIR:-data/splits/run_k}"
RUN_K_TRAIN_SPLIT="${RUN_K_TRAIN_SPLIT:-$RUN_K_SPLIT_DIR/stagewise_train_augmented.jsonl}"
RUN_K_VAL_SPLIT="${RUN_K_VAL_SPLIT:-$RUN_K_SPLIT_DIR/stagewise_val_combined.jsonl}"
RAW_OUT_JSONL="${RAW_OUT_JSONL:-data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_raw_messages.jsonl}"
OUT_JSONL="${OUT_JSONL:-data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_messages.jsonl}"
WEIGHT_SUMMARY_JSON="${WEIGHT_SUMMARY_JSON:-data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_weight_summary.json}"
BASE_TRAIN_JSONL_SOURCE="${BASE_TRAIN_JSONL_SOURCE:-data/splits/run_j/stagewise_train_augmented.jsonl}"
BASE_VAL_JSONL_SOURCE="${BASE_VAL_JSONL_SOURCE:-data/splits/run_j/stagewise_val_combined.jsonl}"

restore_split_compat() {
  cp "$BASE_TRAIN_JSONL_SOURCE" data/splits/train.jsonl
  cp "$BASE_VAL_JSONL_SOURCE" data/splits/val.jsonl
  cp "$BASE_TRAIN_JSONL_SOURCE" data/splits/train_stagewise.jsonl
  cp "$BASE_VAL_JSONL_SOURCE" data/splits/val_stagewise.jsonl
}

trap restore_split_compat EXIT

"$PYTHON_BIN" scripts/prepare_run_k_splits.py \
  --base_train_jsonl "$BASE_TRAIN_JSONL_SOURCE" \
  --base_val_jsonl "$BASE_VAL_JSONL_SOURCE" \
  --out_dir "$RUN_K_SPLIT_DIR"

export STAGEWISE_TRAIN_JSONL="$RUN_K_TRAIN_SPLIT"
export STAGEWISE_VAL_JSONL="$RUN_K_VAL_SPLIT"
export MONOLITHIC_TRAIN_JSONL="${MONOLITHIC_TRAIN_JSONL:-data/splits/monolithic_multi/train/monolithic_final.jsonl}"
export MONOLITHIC_VAL_JSONL="${MONOLITHIC_VAL_JSONL:-data/splits/monolithic_multi/val/monolithic_final.jsonl}"

PROMPT_PROFILE=default \
MESSAGE_TAG=strict \
ASSISTANT_TARGET_STYLE=trace_text \
TRAIN_TASKS=e2e_trace \
VAL_TASKS=e2e_trace \
"$PYTHON_BIN" code/data/prepare_data.py \
  --stagewise_train_jsonl "$STAGEWISE_TRAIN_JSONL" \
  --stagewise_val_jsonl "$STAGEWISE_VAL_JSONL" \
  --monolithic_train_jsonl "$MONOLITHIC_TRAIN_JSONL" \
  --monolithic_val_jsonl "$MONOLITHIC_VAL_JSONL" \
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
  --output "$RAW_OUT_JSONL" \
  --strict-e2e-weight "${STRICT_E2E_WEIGHT:-2}" \
  --runtime-task-weight e2e_trace="${RUNTIME_E2E_WEIGHT:-1}" \
  --runtime-task-weight doc_verdict="${RUNTIME_DOC_VERDICT_WEIGHT:-1}" \
  --runtime-task-weight conflict_type="${RUNTIME_CONFLICT_TYPE_WEIGHT:-2}" \
  --runtime-task-weight answer_only="${RUNTIME_ANSWER_ONLY_WEIGHT:-1}" \
  --boundary-conflict-label-weight "No conflict=${BOUNDARY_WEIGHT_NO_CONFLICT:-1}" \
  --boundary-conflict-label-weight "Complementary information=${BOUNDARY_WEIGHT_COMPLEMENTARY:-2}" \
  --boundary-conflict-label-weight "Conflicting opinions or research outcomes=${BOUNDARY_WEIGHT_CONFLICTING:-2}" \
  --boundary-conflict-label-weight "Conflict due to outdated information=${BOUNDARY_WEIGHT_OUTDATED:-2}" \
  --boundary-conflict-label-weight "Conflict due to misinformation=${BOUNDARY_WEIGHT_MISINFORMATION:-2}" \
  --doc-verdict-boundary-weight "${DOC_VERDICT_BOUNDARY_WEIGHT:-1}" \
  --strict-partial-synthesis-weight "${STRICT_PARTIAL_SYNTHESIS_WEIGHT:-1}" \
  --runtime-partial-synthesis-e2e-weight "${RUNTIME_PARTIAL_SYNTHESIS_E2E_WEIGHT:-2}" \
  --runtime-partial-synthesis-answer-only-weight "${RUNTIME_PARTIAL_SYNTHESIS_ANSWER_ONLY_WEIGHT:-2}" \
  --minimal-partial-synthesis-weight "${MINIMAL_PARTIAL_SYNTHESIS_WEIGHT:-3}" \
  --minimal-e2e-weight "${MINIMAL_E2E_WEIGHT:-4}"

"$PYTHON_BIN" scripts/annotate_sample_weights.py \
  --input "$RAW_OUT_JSONL" \
  --output "$OUT_JSONL" \
  --metadata_jsonl "$RUN_K_TRAIN_SPLIT" \
  --summary_json "$WEIGHT_SUMMARY_JSON" \
  --answerable_exact_docs "${ANSWERABLE_EXACT_DOCS:-5}" \
  --answerable_exact_weight "${ANSWERABLE_EXACT_WEIGHT:-1.35}" \
  --answerable_short_max_docs "${ANSWERABLE_SHORT_MAX_DOCS:-7}" \
  --answerable_short_weight "${ANSWERABLE_SHORT_WEIGHT:-1.4}" \
  --decision_answerable_short_extra_weight "${DECISION_ANSWERABLE_SHORT_EXTRA_WEIGHT:-1.15}" \
  --answerable_mid_max_docs "${ANSWERABLE_MID_MAX_DOCS:-10}" \
  --answerable_mid_weight "${ANSWERABLE_MID_WEIGHT:-1.15}" \
  --answerable_partial_only_weight "${ANSWERABLE_PARTIAL_ONLY_WEIGHT:-1.2}" \
  --benchmark_like_aug_weight "${BENCHMARK_LIKE_AUG_WEIGHT:-1.05}" \
  --answerable-origin-weight "run_k_short5_support=${RUN_K_SHORT5_SUPPORT_WEIGHT:-1.0}" \
  --answerable-origin-weight "run_k_short5_partial_only=${RUN_K_SHORT5_PARTIAL_ONLY_WEIGHT:-1.5}" \
  --answerable-partial-only-conflict-label-weight "Complementary information=${PARTIAL_ONLY_COMPLEMENTARY_WEIGHT:-1.15}" \
  --answerable-partial-only-conflict-label-weight "Conflicting opinions or research outcomes=${PARTIAL_ONLY_CONFLICTING_WEIGHT:-1.15}" \
  --answerable-partial-only-conflict-label-weight "Conflict due to misinformation=${PARTIAL_ONLY_MISINFORMATION_WEIGHT:-1.6}" \
  --refusal_short_max_docs "${REFUSAL_SHORT_MAX_DOCS:-5}" \
  --refusal_short_weight "${REFUSAL_SHORT_WEIGHT:-0.55}" \
  --decision_refusal_short_extra_weight "${DECISION_REFUSAL_SHORT_EXTRA_WEIGHT:-0.75}" \
  --refusal_long_weight "${REFUSAL_LONG_WEIGHT:-0.7}" \
  --trust_align_refusal_weight "${TRUST_ALIGN_REFUSAL_WEIGHT:-0.85}"

"$PYTHON_BIN" scripts/check_trace_text_messages.py "$OUT_JSONL"
"$PYTHON_BIN" scripts/check_trace_text_messages.py --require_think data/messages/train_stagewise_e2e_strict_messages.jsonl
"$PYTHON_BIN" scripts/check_trace_text_messages.py --require_think data/messages/val_stagewise_e2e_trace_text_messages.jsonl
"$PYTHON_BIN" scripts/check_trace_text_messages.py --require_think data/messages/val_stagewise_e2e_minimal_messages.jsonl
