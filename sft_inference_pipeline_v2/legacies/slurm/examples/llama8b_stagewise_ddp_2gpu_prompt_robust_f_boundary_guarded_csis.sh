#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_ROOT/slurm/common_env.sh"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}" bash slurm/examples/rebuild_messages_prompt_robust_f_boundary_guarded.sh

export GPU_COUNT=2
export MODEL_NAME="${MODEL_NAME:-llama31}"
export BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT:-/nfs_home/users/vsshekhawat/projects/rag-reason/models}/Llama-3.1-8B-Instruct}"
export TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
export VAL_STRATEGY="${VAL_STRATEGY:-stagewise}"
export RUN_NAME="${RUN_NAME:-main_trace_text_f_boundary_guarded_csis}"

export TRAIN_JSONL="${TRAIN_JSONL:-data/messages/train_stagewise_prompt_robust_trace_text_f_boundary_guarded_messages.jsonl}"
export VAL_JSONL="${VAL_JSONL:-data/messages/val_stagewise_e2e_minimal_messages.jsonl}"

export EPOCHS="${EPOCHS:-2}"
export LR="${LR:-2e-4}"
export BSZ="${BSZ:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export MAX_LEN="${MAX_LEN:-12288}"
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export NEFTUNE_ALPHA="${NEFTUNE_ALPHA:-5.0}"

export CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-3.6}"
export CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-3.0}"
export ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.25}"
export CITATION_WEIGHT="${CITATION_WEIGHT:-1.7}"
export CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.55}"

export PATIENCE="${PATIENCE:-3}"
export DEV_SUBSET="${DEV_SUBSET:-49}"
export DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-900}"
export DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1800}"
export DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.20}"
export DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.35}"
export DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-0.15}"
export DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-0}"
export DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}"
export DEV_RETRY_CAP="${DEV_RETRY_CAP:-2600}"
export DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-10800}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"

exec sbatch "$PROJECT_ROOT/slurm/train_experiment_ddp_2gpu.sh"
