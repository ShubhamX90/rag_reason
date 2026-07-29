#!/bin/bash
#SBATCH --job-name=rag-h100-smoke
#SBATCH --partition=gpu_h100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/sharanga_h100_smoke_%j.out
#SBATCH --error=logs/sharanga_h100_smoke_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

echo "===== SHARANGA H100 SMOKE ====="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "WORK_ROOT=$WORK_ROOT"
echo "MODEL_ROOT=$MODEL_ROOT"
echo "ENV_PREFIX=${ENV_PREFIX:-$SCRATCH/rag-reason/envs/rag-reason}"
echo "==============================="

which python
python --version
which nvidia-smi
nvidia-smi

python - <<'PY'
import torch
import transformers
import peft
import bitsandbytes as bnb

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bnb.__version__)
PY
