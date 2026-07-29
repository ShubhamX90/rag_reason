#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_ROOT/slurm/common_env.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_JSONL="${TRAIN_JSONL:-data/messages/train_stagewise_prompt_robust_trace_text_f_boundary_guarded_messages.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/messages/val_stagewise_e2e_minimal_messages.jsonl}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

LLAMA_BASE="${LLAMA_BASE:-$MODEL_ROOT/Llama-3.1-8B-Instruct}"
MISTRAL_BASE="${MISTRAL_BASE:-$MODEL_ROOT/Mistral-7B-Instruct-v0.3}"

echo "===== Preflight: rebuild F messages ====="
PYTHON_BIN="$PYTHON_BIN" bash slurm/examples/rebuild_messages_prompt_robust_f_boundary_guarded.sh

echo "===== Preflight: files and model dirs ====="
test -f "$TRAIN_JSONL"
test -f "$VAL_JSONL"
test -d "$LLAMA_BASE"
test -d "$MISTRAL_BASE"

check_out_dir() {
  local model_name="$1"
  local out_dir="checkpoints/${model_name}_stagewise_e2e_main_trace_text_f_boundary_guarded_csis"
  if [ -d "$out_dir" ] && [ "$(find "$out_dir" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ] && [ "$ALLOW_EXISTING" != "1" ]; then
    echo "Refusing to submit because output dir is non-empty: $out_dir" >&2
    echo "Set ALLOW_EXISTING=1 only if you intentionally want the train job to decide how to handle it." >&2
    exit 1
  fi
}

check_out_dir "llama31"
check_out_dir "mistral7b"

echo "===== Preflight: tokenizer chat templates ====="
"$PYTHON_BIN" - "$LLAMA_BASE" "$MISTRAL_BASE" <<'PY'
import sys
from transformers import AutoTokenizer

for model_path in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    msgs = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User prompt."},
    ]
    try:
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        mode = "system+user"
    except Exception as exc:
        merged = [{"role": "user", "content": "System prompt.\n\nUser prompt."}]
        rendered = tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
        mode = f"merged-system-user fallback ({exc})"
    print(f"tokenizer_ok={model_path} mode={mode} rendered_chars={len(rendered)}")
PY

submit_train_and_dependent_eval() {
  local model_name="$1"
  local base_model="$2"
  local gen_script="$3"
  local submit_log_prefix="$4"
  local train_dependency="${5:-}"

  echo "===== submit train: $model_name ====="
  local dependency_args=()
  if [ -n "$train_dependency" ]; then
    dependency_args+=(--dependency="afterok:${train_dependency}")
    echo "$model_name train dependency: afterok:$train_dependency"
  fi

  local train_job
  train_job="$(
    MODEL_NAME="$model_name" \
    BASE_MODEL="$base_model" \
    TRAIN_STRATEGY=stagewise \
    VAL_STRATEGY=stagewise \
    RUN_NAME=main_trace_text_f_boundary_guarded_csis \
    TRAIN_JSONL="$TRAIN_JSONL" \
    VAL_JSONL="$VAL_JSONL" \
    EPOCHS="${EPOCHS:-2}" \
    LR="${LR:-2e-4}" \
    BSZ="${BSZ:-1}" \
    GRAD_ACCUM="${GRAD_ACCUM:-8}" \
    MAX_LEN="${MAX_LEN:-12288}" \
    LORA_R="${LORA_R:-32}" \
    LORA_ALPHA="${LORA_ALPHA:-64}" \
    LORA_DROPOUT="${LORA_DROPOUT:-0.05}" \
    NEFTUNE_ALPHA="${NEFTUNE_ALPHA:-5.0}" \
    CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-3.6}" \
    CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-3.0}" \
    ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.25}" \
    CITATION_WEIGHT="${CITATION_WEIGHT:-1.7}" \
    CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.55}" \
    PATIENCE="${PATIENCE:-3}" \
    DEV_SUBSET="${DEV_SUBSET:-49}" \
    DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-900}" \
    DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1800}" \
    DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.20}" \
    DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.35}" \
    DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-0.15}" \
    DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-0}" \
    DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}" \
    DEV_RETRY_CAP="${DEV_RETRY_CAP:-2600}" \
    DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-10800}" \
    OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}" \
    sbatch "${dependency_args[@]}" slurm/train_experiment_ddp_2gpu.sh | awk '{print $4}'
  )"
  echo "$model_name train job: $train_job"
  LAST_TRAIN_JOB="$train_job"

  echo "===== queue generate/eval submitter for $model_name ====="
  sbatch \
    --job-name="${submit_log_prefix}-gen" \
    --partition=cpu-short \
    --dependency="afterok:${train_job}" \
    --output="logs/${submit_log_prefix}_submit_generate_%j.out" \
    --error="logs/${submit_log_prefix}_submit_generate_%j.err" \
    --wrap="cd '$PROJECT_ROOT' && REBUILD_MESSAGES=0 bash '$gen_script'"
}

submit_train_and_dependent_eval \
  "llama31" \
  "$LLAMA_BASE" \
  "slurm/examples/llama8b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_csis.sh" \
  "llama31_f"
llama_job="$LAST_TRAIN_JOB"

submit_train_and_dependent_eval \
  "mistral7b" \
  "$MISTRAL_BASE" \
  "slurm/examples/mistral7b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_csis.sh" \
  "mistral7b_f" \
  "$llama_job"
