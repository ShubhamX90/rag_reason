#!/bin/bash
#SBATCH --job-name=rag-model-smoke
#SBATCH --partition=gpu_h100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sharanga_model_load_smoke_%j.out
#SBATCH --error=logs/sharanga_model_load_smoke_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen2.5-7B-Instruct}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export MODEL_PATH LOAD_IN_4BIT MAX_NEW_TOKENS

echo "===== SHARANGA MODEL LOAD SMOKE ====="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL_PATH=$MODEL_PATH"
echo "LOAD_IN_4BIT=$LOAD_IN_4BIT"
echo "====================================="
nvidia-smi -L

python - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = os.environ["MODEL_PATH"]
load_in_4bit = os.environ.get("LOAD_IN_4BIT", "0") == "1"
max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "16"))

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)

quantization_config = None
if load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="sdpa",
)

prompt = "Answer in one short sentence: what is 2 + 2?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

decoded = tokenizer.decode(out[0], skip_special_tokens=True)
print("generated:", decoded.replace("\n", " ")[:500])
print("max_memory_allocated_gb", round(torch.cuda.max_memory_allocated() / 1024**3, 3))
print("smoke_ok", True)
PY
