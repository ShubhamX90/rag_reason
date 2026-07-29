#!/usr/bin/env bash
#SBATCH -J ragann_b800_qwen_s1
#SBATCH -p gpu_h200_8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH -t 18:00:00
#SBATCH -o /scratch/pabitra/rag-reason/logs/%x_%j.out
#SBATCH -e /scratch/pabitra/rag-reason/logs/%x_%j.err
export JUDGE_SHORT=qwen
export PIPELINE_STAGE=stage1_collect
source "$HOME/rag-reason/dataset_annotation_pipeline_v3/slurm/sharanga/local_committee/benchmark800_collect_common.sh"
