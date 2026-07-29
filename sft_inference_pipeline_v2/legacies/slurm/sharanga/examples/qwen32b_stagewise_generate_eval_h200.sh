#!/bin/bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-qwen25_32b}"
export BASE_MODEL="${BASE_MODEL:-/scratch/$USER/rag-reason/models/Qwen2.5-32B-Instruct}"
export TRAIN_STRATEGY="${TRAIN_STRATEGY:-stagewise}"
export RUN_NAME="${RUN_NAME:-main_trace_text_b}"
export DATASET_LABEL="${DATASET_LABEL:-val_stagewise}"
export PROMPT_MODE="${PROMPT_MODE:-e2e}"
export PROMPT_PROFILE="${PROMPT_PROFILE:-trace_text}"
export CONTRACT_MODE="${CONTRACT_MODE:-trace}"
export MODEL_VARIANT="${MODEL_VARIANT:-sft}"
export LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
export LORA_DIR="${LORA_DIR:-/scratch/$USER/rag-reason/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_b/best_dev_f1}"

GEN_JOB_ID="$(
  sbatch \
    --partition="${PARTITION:-gpu_h200_8}" \
    --nodelist="${NODELIST:-gpunode7}" \
    slurm/sharanga/generate_experiment.sh |
  awk '{print $4}'
)"

echo "generate job: $GEN_JOB_ID"

sbatch \
  --dependency="afterok:$GEN_JOB_ID" \
  slurm/sharanga/evaluate_experiment.sh
