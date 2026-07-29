#!/bin/bash
#SBATCH --job-name=rag-a100-4g-smoke
#SBATCH --partition=gpu_a100_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=00:20:00
#SBATCH --output=logs/sharanga_a100_4gpu_smoke_%j.out
#SBATCH --error=logs/sharanga_a100_4gpu_smoke_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

echo "===== SHARANGA A100 4-GPU SMOKE ====="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "====================================="

nvidia-smi -L

TMP_PY="${SLURM_TMPDIR:-/tmp}/sharanga_smoke_ddp.py"
cat > "$TMP_PY" <<'PY'
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
name = torch.cuda.get_device_name(local_rank)
xbf16 = torch.cuda.is_bf16_supported()
t = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(t)
print({
    "rank": rank,
    "world_size": world,
    "local_rank": local_rank,
    "gpu_name": name,
    "bf16": xbf16,
    "all_reduce_sum": float(t.item()),
}, flush=True)
dist.barrier()
dist.destroy_process_group()
PY

torchrun --standalone --nnodes=1 --nproc_per_node=4 "$TMP_PY"
