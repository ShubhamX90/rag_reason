#!/bin/bash
#SBATCH --job-name=rag-model-2g
#SBATCH --partition=gpu_h100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sharanga_model_load_2gpu_%j.out
#SBATCH --error=logs/sharanga_model_load_2gpu_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen2.5-7B-Instruct}"
MODEL_ALIAS="${MODEL_ALIAS:-$(basename "$MODEL_PATH")}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
export MODEL_PATH MODEL_ALIAS LOAD_IN_4BIT MAX_NEW_TOKENS ATTN_IMPL

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model directory not found: $MODEL_PATH" >&2
  exit 1
fi

echo "===== SHARANGA MODEL LOAD 2-GPU SMOKE ====="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL_ALIAS=$MODEL_ALIAS"
echo "MODEL_PATH=$MODEL_PATH"
echo "LOAD_IN_4BIT=$LOAD_IN_4BIT"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "ATTN_IMPL=$ATTN_IMPL"
echo "==========================================="
nvidia-smi -L

TMP_PY="${SLURM_TMPDIR:-/tmp}/sharanga_model_load_2gpu_smoke.py"
cat > "$TMP_PY" <<'PY'
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


def main():
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    model_path = os.environ["MODEL_PATH"]
    model_alias = os.environ.get("MODEL_ALIAS", os.path.basename(model_path))
    load_in_4bit = os.environ.get("LOAD_IN_4BIT", "0") == "1"
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "12"))
    attn_impl = os.environ.get("ATTN_IMPL", "sdpa")

    print(
        {
            "rank": rank,
            "world_size": world,
            "local_rank": local_rank,
            "gpu_name": torch.cuda.get_device_name(local_rank),
            "bf16": torch.cuda.is_bf16_supported(),
            "model_alias": model_alias,
            "load_in_4bit": load_in_4bit,
            "attn_impl": attn_impl,
        },
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        loader_name = "AutoModelForCausalLM"
    except Exception as causal_exc:
        if AutoModelForImageTextToText is None:
            raise
        print({"rank": rank, "causal_lm_failed": repr(causal_exc), "fallback": "AutoModelForImageTextToText"}, flush=True)
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        loader_name = "AutoModelForImageTextToText"

    prompt = "Answer in one short sentence: what is 2 + 2?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True).replace("\n", " ")
    t = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(t)

    print(
        {
            "rank": rank,
            "model_alias": model_alias,
            "loader_name": loader_name,
            "generated": decoded[:300],
            "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated(local_rank) / 1024**3, 3),
            "all_reduce_sum": float(t.item()),
            "smoke_ok": True,
        },
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

torchrun --standalone --nnodes=1 --nproc_per_node=2 "$TMP_PY"
