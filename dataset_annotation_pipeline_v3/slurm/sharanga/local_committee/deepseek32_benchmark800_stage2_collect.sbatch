#!/usr/bin/env bash
#SBATCH -J ragann_b800_ds_s2
#SBATCH -p gpu_h200_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/pabitra/rag-reason/logs/%x_%j.out
#SBATCH -e /scratch/pabitra/rag-reason/logs/%x_%j.err
export JUDGE_SHORT=deepseek
export PIPELINE_STAGE=stage2_collect
source "$HOME/rag-reason/dataset_annotation_pipeline_v3/slurm/sharanga/local_committee/benchmark800_collect_common.sh"
