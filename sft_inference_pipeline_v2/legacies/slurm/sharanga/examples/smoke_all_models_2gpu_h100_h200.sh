#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/rag-reason/sft_inference_pipeline_v2}"
source "$PROJECT_ROOT/slurm/sharanga/common_env.sh"

submit_smoke() {
  local label="$1"
  local partition="$2"
  local nodelist="$3"
  local alias="$4"
  local path="$5"
  local load_in_4bit="$6"
  local safe_alias="${alias//[^A-Za-z0-9_]/_}"

  if [ ! -d "$path" ]; then
    echo "[Skip] Missing model path for $alias: $path" >&2
    return 1
  fi

  sbatch \
    --partition="$partition" \
    --nodelist="$nodelist" \
    --job-name="rag-${label}-${safe_alias}" \
    --output="logs/sharanga_${label}_${safe_alias}_2gpu_model_smoke_%j.out" \
    --error="logs/sharanga_${label}_${safe_alias}_2gpu_model_smoke_%j.err" \
    --export=ALL,MODEL_ALIAS="$alias",MODEL_PATH="$path",LOAD_IN_4BIT="$load_in_4bit" \
    slurm/sharanga/smoke_model_load_2gpu.sh
}

models=(
  "qwen25|$MODEL_ROOT/Qwen2.5-7B-Instruct|0"
  "mistral7b|$MODEL_ROOT/Mistral-7B-Instruct-v0.3|0"
  "llama31|$MODEL_ROOT/Llama-3.1-8B-Instruct|0"
  "qwen25_32b|$MODEL_ROOT/Qwen2.5-32B-Instruct|1"
)

for spec in "${models[@]}"; do
  IFS='|' read -r alias path load_in_4bit <<< "$spec"
  submit_smoke "h100" "gpu_h100_4" "${H100_NODELIST:-gpunode6}" "$alias" "$path" "$load_in_4bit"
  submit_smoke "h200" "gpu_h200_8" "${H200_NODELIST:-gpunode7}" "$alias" "$path" "$load_in_4bit"
done
