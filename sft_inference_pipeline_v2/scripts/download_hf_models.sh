#!/bin/bash
set -euo pipefail

if [ -z "${SCRATCH:-}" ]; then
  export SCRATCH="/scratch/$USER"
fi

WORK_ROOT="${WORK_ROOT:-$SCRATCH/rag-reason}"
MODEL_ROOT="${MODEL_ROOT:-$WORK_ROOT/models}"
HF_HOME="${HF_HOME:-$WORK_ROOT/cache/hf}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
MODELS="${MODELS:-${1:-qwen}}"

mkdir -p "$MODEL_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

download_model() {
  local alias="$1"
  local repo_id="$2"
  local local_name="$3"
  local target="$MODEL_ROOT/$local_name"

  echo "===== Downloading $alias ====="
  echo "repo_id=$repo_id"
  echo "target=$target"

  python - "$repo_id" "$target" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target = sys.argv[2]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

snapshot_download(
    repo_id=repo_id,
    local_dir=target,
    token=token,
)

print(f"downloaded={target}")
PY
}

for model in $(echo "$MODELS" | tr ',' ' '); do
  case "$model" in
    qwen|qwen25|Qwen2.5-7B-Instruct)
      download_model "qwen25" "Qwen/Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct"
      ;;
    qwen32|qwen25-32b|Qwen2.5-32B-Instruct)
      download_model "qwen25-32b" "Qwen/Qwen2.5-32B-Instruct" "Qwen2.5-32B-Instruct"
      ;;
    qwen3-32b|qwen3_32b|Qwen3-32B)
      download_model "qwen3-32b" "Qwen/Qwen3-32B" "Qwen3-32B"
      ;;
    mistral|mistral7b|Mistral-7B-Instruct-v0.3)
      download_model "mistral7b" "mistralai/Mistral-7B-Instruct-v0.3" "Mistral-7B-Instruct-v0.3"
      ;;
    mistral24|mistral-small-24b|Mistral-Small-3.2-24B-Instruct-2506)
      download_model "mistral-small-24b" "mistralai/Mistral-Small-3.2-24B-Instruct-2506" "Mistral-Small-3.2-24B-Instruct-2506"
      ;;
    gemma27|gemma3-27b|Gemma-3-27B|gemma-3-27b-it)
      download_model "gemma3-27b-it" "google/gemma-3-27b-it" "gemma-3-27b-it"
      ;;
    llama|llama31|Llama-3.1-8B-Instruct)
      download_model "llama31" "meta-llama/Llama-3.1-8B-Instruct" "Llama-3.1-8B-Instruct"
      ;;
    all)
      MODELS="qwen mistral llama" "$0"
      exit 0
      ;;
    all-plus-qwen32)
      MODELS="qwen qwen32 mistral llama" "$0"
      exit 0
      ;;
    scale-candidates)
      MODELS="mistral24 qwen3-32b gemma27" "$0"
      exit 0
      ;;
    *)
      echo "Unknown model alias: $model" >&2
      echo "Use one of: qwen, qwen32, qwen3-32b, mistral, mistral24, gemma27, llama, all, all-plus-qwen32, scale-candidates" >&2
      exit 2
      ;;
  esac
done

echo
echo "Model directories now under: $MODEL_ROOT"
find "$MODEL_ROOT" -maxdepth 2 -type f \( -name 'config.json' -o -name 'tokenizer_config.json' -o -name '*.safetensors' -o -name '*.index.json' \) | sort
