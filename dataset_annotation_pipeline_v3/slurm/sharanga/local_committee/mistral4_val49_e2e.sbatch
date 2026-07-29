#!/usr/bin/env bash
#SBATCH --job-name=ragann_mistral4_val49_e2e
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pabitra/rag-reason/logs/ragann_mistral4_val49_e2e_%j.out
#SBATCH --error=/scratch/pabitra/rag-reason/logs/ragann_mistral4_val49_e2e_%j.err

export PROJECT_DIR="${PROJECT_DIR:-$HOME/rag-reason/dataset_annotation_pipeline_v3}"
export JUDGE_SHORT=mistral
source "$PROJECT_DIR/slurm/sharanga/local_committee/val49_e2e_common.sh"
