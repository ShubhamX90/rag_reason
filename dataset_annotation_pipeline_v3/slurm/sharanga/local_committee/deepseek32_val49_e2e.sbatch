#!/usr/bin/env bash
#SBATCH --job-name=ragann_deepseek32_val49_e2e
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/pabitra/rag-reason/logs/ragann_deepseek32_val49_e2e_%j.out
#SBATCH --error=/scratch/pabitra/rag-reason/logs/ragann_deepseek32_val49_e2e_%j.err

export PROJECT_DIR="${PROJECT_DIR:-$HOME/rag-reason/dataset_annotation_pipeline_v3}"
export JUDGE_SHORT=deepseek
source "$PROJECT_DIR/slurm/sharanga/local_committee/val49_e2e_common.sh"
