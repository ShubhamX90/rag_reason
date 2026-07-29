#!/bin/bash

export GPU_COUNT=2
export MODEL_NAME=qwen25
export BASE_MODEL=/nfs_home/users/vsshekhawat/projects/rag-reason/models/Qwen2.5-7B-Instruct
export TRAIN_STRATEGY=stagewise
export VAL_STRATEGY=stagewise
export RUN_NAME="${RUN_NAME:-main_ddp2e}"

# Preserve the strongest known core recipe from the main_ddp2c run.
export EPOCHS="${EPOCHS:-2}"
export LR="${LR:-2e-4}"
export BSZ="${BSZ:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export MAX_LEN="${MAX_LEN:-12288}"
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export NEFTUNE_ALPHA="${NEFTUNE_ALPHA:-5.0}"
export CONFLICT_WEIGHT="${CONFLICT_WEIGHT:-3.0}"
export CONTRACT_WEIGHT="${CONTRACT_WEIGHT:-1.5}"
export ARRAY_WEIGHT="${ARRAY_WEIGHT:-1.15}"
export CITATION_WEIGHT="${CITATION_WEIGHT:-1.4}"
export CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.65}"
export PATIENCE="${PATIENCE:-3}"

# Make checkpoint selection more semantics-first:
# larger stratified dev slice, direct doc-verdict scoring, only light format pressure.
export DEV_SUBSET="${DEV_SUBSET:-24}"
export DEV_MAX_NEW_BASE="${DEV_MAX_NEW_BASE:-900}"
export DEV_MAX_NEW_CAP="${DEV_MAX_NEW_CAP:-1800}"
export DEV_DOC_VERDICT_WEIGHT="${DEV_DOC_VERDICT_WEIGHT:-0.35}"
export DEV_FORMAT_WEIGHT="${DEV_FORMAT_WEIGHT:-0.05}"
export DEV_ABSTAIN_WEIGHT="${DEV_ABSTAIN_WEIGHT:-0.10}"
export DEV_RETRY_ATTEMPTS="${DEV_RETRY_ATTEMPTS:-1}"
export DEV_RETRY_SCALE="${DEV_RETRY_SCALE:-1.6}"
export DEV_RETRY_CAP="${DEV_RETRY_CAP:-2600}"
export DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-10800}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-0}"

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
exec sbatch "$PROJECT_ROOT/slurm/train_experiment_ddp_2gpu.sh"
