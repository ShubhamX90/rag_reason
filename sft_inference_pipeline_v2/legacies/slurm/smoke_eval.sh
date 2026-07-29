#!/bin/bash
#SBATCH --job-name=rag-eval-smoke
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --exclude=csis.mn1
#SBATCH --output=logs/smoke_eval_%j.out
#SBATCH --error=logs/smoke_eval_%j.err

set -euo pipefail

PROJECT_ROOT=/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
source "$PROJECT_ROOT/slurm/common_env.sh"

mkdir -p outputs/reports/smoke_sft_qwen_val_stagewise

python code/eval/eval_contract.py \
  --canon_jsonl data/splits/val_stagewise.jsonl \
  --gens_jsonl outputs/smoke_sft_qwen_val_stagewise.sanitized.jsonl \
  --report_json outputs/reports/smoke_sft_qwen_val_stagewise/contract.json

python code/eval/eval_doc_verdicts.py \
  --canon_jsonl data/splits/val_stagewise.jsonl \
  --gens_jsonl outputs/smoke_sft_qwen_val_stagewise.sanitized.jsonl \
  --report_json outputs/reports/smoke_sft_qwen_val_stagewise/doc_verdicts.json

python code/eval/eval_conflict_type.py \
  --canon_jsonl data/splits/val_stagewise.jsonl \
  --gens_jsonl outputs/smoke_sft_qwen_val_stagewise.sanitized.jsonl \
  --report_json outputs/reports/smoke_sft_qwen_val_stagewise/conflict_type.json
