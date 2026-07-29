#!/bin/bash
#SBATCH --job-name=rag-train-smoke
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --exclude=csis.mn1
#SBATCH --output=logs/smoke_train_1gpu_%j.out
#SBATCH --error=logs/smoke_train_1gpu_%j.err

set -euo pipefail

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
source "$PROJECT_ROOT/slurm/common_env.sh"

bash "$PROJECT_ROOT/slurm/prepare_smoke_subsets.sh"

python code/train/train_qlora.py \
  --base_model "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --train_jsonl data/messages/smoke_train_stagewise_e2e.jsonl \
  --val_jsonl data/messages/smoke_val_stagewise_e2e.jsonl \
  --out_dir checkpoints/smoke_qwen_stagewise_1gpu \
  --epochs 1 \
  --lr 2e-4 \
  --bsz 1 \
  --grad_accum 1 \
  --max_len 4096 \
  --lora_r 16 \
  --lora_alpha 32 \
  --neftune_alpha 0 \
  --conflict_weight 3.0 \
  --patience 1 \
  --dev_subset 2 \
  --dev_max_new_base 256 \
  --dev_max_new_cap 512 \
  --overwrite_output_dir
