#!/bin/bash

export GPU_COUNT=2
export MODEL_NAME=qwen25
export BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT:-/nfs_home/users/vsshekhawat/projects/rag-reason/models}/Qwen2.5-7B-Instruct}"
export TRAIN_STRATEGY=stagewise
export VAL_STRATEGY=stagewise
export RUN_NAME=pilot_ddp2

export EPOCHS=1
export LR=2e-4
export BSZ=1
export GRAD_ACCUM=8
export MAX_LEN=8192
export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05
export NEFTUNE_ALPHA=0.0
export CONFLICT_WEIGHT=3.0
export CONTRACT_WEIGHT=2.5
export ARRAY_WEIGHT=1.35
export CITATION_WEIGHT=1.75
export PATIENCE=2
export DEV_SUBSET=8
export DEV_MAX_NEW_BASE=512
export DEV_MAX_NEW_CAP=1024
export DEV_FORMAT_WEIGHT=0.35
export OVERWRITE_OUTPUT_DIR=0

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec sbatch "$PROJECT_ROOT/slurm/train_experiment_ddp_2gpu.sh"
