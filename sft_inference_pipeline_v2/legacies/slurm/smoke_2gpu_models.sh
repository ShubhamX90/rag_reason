#!/bin/bash
#SBATCH --job-name=rag-smoke-2gpu
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100-80gb:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --exclude=csis.mn1
#SBATCH --output=logs/smoke_2gpu_%j.out
#SBATCH --error=logs/smoke_2gpu_%j.err

set -euo pipefail

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
source "$PROJECT_ROOT/slurm/common_env.sh"

export NCCL_DEBUG=INFO
export MASTER_PORT=29500

echo "===== BASIC ENV ====="
hostname || true
echo "USER=${USER:-unknown}"
id || true
id -u || true
id -un 2>/dev/null || echo "username lookup unavailable"
pwd || true
date || true
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_NNODES=${SLURM_NNODES:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_PROCID=${SLURM_PROCID:-unset}"
echo "SLURM_LOCALID=${SLURM_LOCALID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo
echo "===== NVIDIA SMI ====="
nvidia-smi || true

echo
echo "===== MASTER ADDR SETUP ====="
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

echo
echo "===== PYTHON PACKAGE CHECK ====="
python - <<'PY'
import torch, transformers, accelerate, peft, bitsandbytes, datasets, safetensors
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("datasets:", datasets.__version__)
print("safetensors:", safetensors.__version__)
for i in range(torch.cuda.device_count()):
    print(f"gpu[{i}]:", torch.cuda.get_device_name(i))
PY

echo
echo "===== TOKENIZER CHECK ====="
python - <<'PY'
from transformers import AutoTokenizer

paths = [
    "/nfs_home/users/vsshekhawat/projects/rag-reason/models/Llama-3.1-8B-Instruct",
    "/nfs_home/users/vsshekhawat/projects/rag-reason/models/Mistral-7B-Instruct-v0.3",
    "/nfs_home/users/vsshekhawat/projects/rag-reason/models/Qwen2.5-7B-Instruct",
]
for p in paths:
    print("CHECK:", p)
    tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
    print("  tokenizer ok")
PY

echo
echo "===== 2-GPU NCCL SMOKE ====="
srun --ntasks=2 --ntasks-per-node=2 python - <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["SLURM_PROCID"])
local_rank = int(os.environ["SLURM_LOCALID"])
world_size = int(os.environ["SLURM_NTASKS"])

torch.cuda.set_device(local_rank)

dist.init_process_group(
    backend="nccl",
    init_method="env://",
    rank=rank,
    world_size=world_size,
)

x = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)

print(
    f"rank={rank} local_rank={local_rank} "
    f"device={torch.cuda.get_device_name(local_rank)} "
    f"all_reduce_result={x.item()}"
)

dist.barrier()
dist.destroy_process_group()
PY

echo
echo "===== SINGLE-MODEL GPU LOAD + TINY GENERATION ====="
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/nfs_home/users/vsshekhawat/projects/rag-reason/models/Qwen2.5-7B-Instruct"

tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "Answer in one short sentence: What is 2 plus 2?"
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

text = tok.decode(out[0], skip_special_tokens=True)
print("GENERATION:", text)
PY

echo
echo "===== DONE ====="
date || true
