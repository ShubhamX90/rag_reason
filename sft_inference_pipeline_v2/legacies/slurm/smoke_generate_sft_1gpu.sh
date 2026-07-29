#!/bin/bash
#SBATCH --job-name=rag-gen-sft
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --exclude=csis.mn1
#SBATCH --output=logs/smoke_generate_sft_%j.out
#SBATCH --error=logs/smoke_generate_sft_%j.err

set -euo pipefail

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
source "$PROJECT_ROOT/slurm/common_env.sh"

bash "$PROJECT_ROOT/slurm/prepare_smoke_subsets.sh"

python code/eval/generate.py \
  --base_model "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --lora_dir checkpoints/smoke_qwen_stagewise_1gpu/best_dev_f1 \
  --input_jsonl data/messages/smoke_gen_val_stagewise_e2e.jsonl \
  --out_jsonl outputs/smoke_sft_qwen_val_stagewise.raw.jsonl \
  --auto_length \
  --max_new_tokens_base 256 \
  --max_new_tokens_cap 512 \
  --dtype bf16

python code/eval/sanitize.py \
  --in_jsonl outputs/smoke_sft_qwen_val_stagewise.raw.jsonl \
  --out_jsonl outputs/smoke_sft_qwen_val_stagewise.sanitized.jsonl \
  --canon_jsonl data/splits/val_stagewise.jsonl
