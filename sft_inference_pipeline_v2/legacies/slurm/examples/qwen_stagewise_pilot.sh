#!/bin/bash

# Usage:
#   sbatch slurm/examples/qwen_stagewise_pilot.sh

# This file is intentionally a thin wrapper around the real launcher so
# the first full pilot is easy to reproduce and modify.

export MODEL_NAME=qwen25
export BASE_MODEL="${BASE_MODEL:-${MODEL_ROOT:-/nfs_home/users/vsshekhawat/projects/rag-reason/models}/Qwen2.5-7B-Instruct}"
export TRAIN_STRATEGY=stagewise
export VAL_STRATEGY=stagewise
export RUN_NAME=pilot1

export EPOCHS=2
export LR=2e-4
export BSZ=1
export GRAD_ACCUM=16
export MAX_LEN=8192
export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05
export NEFTUNE_ALPHA=5.0
export CONFLICT_WEIGHT=3.0
export PATIENCE=3
export DEV_SUBSET=32
export DEV_MAX_NEW_BASE=1200
export DEV_MAX_NEW_CAP=2200
export OVERWRITE_OUTPUT_DIR=0

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec sbatch "$PROJECT_ROOT/slurm/train_experiment.sh"
