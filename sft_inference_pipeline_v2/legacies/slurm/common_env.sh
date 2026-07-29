#!/bin/bash

set -eo pipefail

module load miniconda
module load cuda/12.8

# CSIS conda hooks can reference optional CONDA_* variables while activating or
# deactivating environments. Keep nounset off only for conda, then restore it.
set +u
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate rag-reason
set -u

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
export MODEL_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/models

mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/outputs"
mkdir -p "$PROJECT_ROOT/checkpoints"

cd "$PROJECT_ROOT"
