#!/bin/bash

set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
  export SCRATCH="/scratch/$USER"
fi

if [ ! -d "$SCRATCH" ]; then
  echo "Sharanga scratch directory not found: $SCRATCH" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

ENV_PREFIX="${ENV_PREFIX:-$SCRATCH/rag-reason/envs/rag-reason}"
conda activate "$ENV_PREFIX"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$COMMON_ENV_DIR/../.." && pwd)}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/rag-reason}"
export MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/outputs}"
export LOG_ROOT="${LOG_ROOT:-$WORK_ROOT/logs}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/cache/pip}"

case "${USER:-}" in
  pabitra)
    export SHARANGA_ACCOUNT_TAG="${SHARANGA_ACCOUNT_TAG:-pb}"
    export SHARANGA_JOB_PREFIX="${SHARANGA_JOB_PREFIX:-atlas}"
    ;;
  kudhru)
    export SHARANGA_ACCOUNT_TAG="${SHARANGA_ACCOUNT_TAG:-kd}"
    export SHARANGA_JOB_PREFIX="${SHARANGA_JOB_PREFIX:-nova}"
    ;;
  *)
    export SHARANGA_ACCOUNT_TAG="${SHARANGA_ACCOUNT_TAG:-u}"
    export SHARANGA_JOB_PREFIX="${SHARANGA_JOB_PREFIX:-orbit}"
    ;;
esac

mkdir -p "$LOG_ROOT" "$WORK_ROOT/checkpoints" "$OUTPUT_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$PIP_CACHE_DIR"

cd "$PROJECT_ROOT"
