#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

WEIGHTED_TRAIN_JSONL="${WEIGHTED_TRAIN_JSONL:-data/messages/train_stagewise_multitask_trace_text_conflict3_answer2_messages.jsonl}"

python scripts/make_weighted_multitask_messages.py \
  --input data/messages/train_stagewise_multitask_trace_text_messages.jsonl \
  --output "$WEIGHTED_TRAIN_JSONL" \
  --task-weight e2e_trace=1 \
  --task-weight doc_verdict=1 \
  --task-weight conflict_type=3 \
  --task-weight answer_only=2

python scripts/check_trace_text_messages.py "$WEIGHTED_TRAIN_JSONL"
python scripts/check_trace_text_messages.py --require_think data/messages/val_stagewise_e2e_trace_text_messages.jsonl

export MODEL_NAME="${MODEL_NAME:-qwen25_32b}"
export BASE_MODEL="${BASE_MODEL:-/scratch/$USER/rag-reason/models/Qwen2.5-32B-Instruct}"
export TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
export VAL_STRATEGY="${VAL_STRATEGY:-stagewise}"
export RUN_NAME="${RUN_NAME:-main_trace_text_c_conflict_focus}"
export OUT_DIR="${OUT_DIR:-/scratch/$USER/rag-reason/checkpoints/${MODEL_NAME}_${TRAIN_STRATEGY}_e2e_${RUN_NAME}}"
export TRAIN_JSONL="$WEIGHTED_TRAIN_JSONL"
export VAL_JSONL="${VAL_JSONL:-data/messages/val_stagewise_e2e_trace_text_messages.jsonl}"

export EPOCHS="${EPOCHS:-2}"
export LR="${LR:-2e-4}"
export BSZ="${BSZ:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export MAX_LEN="${MAX_LEN:-12288}"
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export NEFTUNE_ALPHA="${NEFTUNE_ALPHA:-5.0}"

export CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-5.0}"
export CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-1.6}"
export ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.25}"
export CITATION_WEIGHT="${CITATION_WEIGHT:-2.25}"
export CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.55}"

export PATIENCE="${PATIENCE:-3}"
export DEV_SUBSET="${DEV_SUBSET:-49}"
export DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-900}"
export DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1800}"
export DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.10}"
export DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.05}"
export DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-0.05}"
export DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-1}"
export DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}"
export DEV_RETRY_CAP="${DEV_RETRY_CAP:-2600}"
export DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-10800}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"

sbatch \
  --partition="${PARTITION:-gpu_h200_8}" \
  --nodelist="${NODELIST:-gpunode7}" \
  slurm/sharanga/train_experiment_ddp_2gpu.sh
