#!/bin/bash
#SBATCH --job-name=cats_dsv4_ct
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/pabitra/rag-reason/logs/deepseek_v4_flash_container_h200_tp2_%j.log

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/scratch/pabitra/rag-reason/models/DeepSeek-V4-Flash-W4A16-FP8-MTP-vllmfix}"
CONTAINER_DIR="${CONTAINER_DIR:-/scratch/pabitra/rag-reason/containers/vllm_openai_latest.sandbox2}"
PORT="${PORT:-8009}"
HOST="${HOST:-0.0.0.0}"
STAGE_MODEL="${STAGE_MODEL:-1}"
LOCAL_MODEL_DIR=""
LOCAL_PYCACHE_DIR="/tmp/${USER}/cats_pycache_${SLURM_JOB_ID}"

cleanup() {
  if [[ -n "${LOCAL_MODEL_DIR}" && -d "${LOCAL_MODEL_DIR}" ]]; then
    echo "cleaning_up=${LOCAL_MODEL_DIR}"
    rm -rf "${LOCAL_MODEL_DIR}"
  fi
  if [[ -d "${LOCAL_PYCACHE_DIR}" ]]; then
    echo "cleaning_pycache=${LOCAL_PYCACHE_DIR}"
    rm -rf "${LOCAL_PYCACHE_DIR}"
  fi
}
trap cleanup EXIT TERM INT

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/pabitra/rag-reason/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/pabitra/rag-reason/apptainer-tmp}"
export HF_HOME="${HF_HOME:-/scratch/pabitra/rag-reason/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$LOCAL_PYCACHE_DIR}"
export CATS_DEEPSEEK_V4_DISABLE_CUTEDSL_INDEXER="${CATS_DEEPSEEK_V4_DISABLE_CUTEDSL_INDEXER:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME" "$PYTHONPYCACHEPREFIX"

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "model=${MODEL_DIR}"
echo "container=${CONTAINER_DIR}"
echo "port=${PORT}"
date
nvidia-smi
df -h /tmp || true

SERVE_MODEL_DIR="$MODEL_DIR"
if [[ "$STAGE_MODEL" == "1" ]]; then
  LOCAL_MODEL_DIR="/tmp/${USER}/cats_deepseek_v4_${SLURM_JOB_ID}"
  echo "staging_model_to=${LOCAL_MODEL_DIR}"
  mkdir -p "$LOCAL_MODEL_DIR"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$MODEL_DIR"/ "$LOCAL_MODEL_DIR"/
  else
    cp -a "$MODEL_DIR"/. "$LOCAL_MODEL_DIR"/
  fi
  SERVE_MODEL_DIR="$LOCAL_MODEL_DIR"
  echo "staged_model_size=$(du -sh "$SERVE_MODEL_DIR" | awk '{print $1}')"
fi

echo "container_source_preflight=begin"
apptainer exec --nv \
  --bind /scratch/pabitra/rag-reason:/scratch/pabitra/rag-reason \
  --bind /tmp:/tmp \
  "$CONTAINER_DIR" \
  python3 -u - <<'PY'
from pathlib import Path

for path in [
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/attention.py",
]:
    p = Path(path)
    data = p.read_bytes()
    print(f"preflight_file={path} len={len(data)} nul={data.count(b'\\x00')}", flush=True)
    compile(data.decode("utf-8"), path, "exec")

print("container_source_preflight=ok", flush=True)
PY

apptainer exec --nv \
  --bind /scratch/pabitra/rag-reason:/scratch/pabitra/rag-reason \
  --bind /tmp:/tmp \
  "$CONTAINER_DIR" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$SERVE_MODEL_DIR" \
    --served-model-name deepseek-v4-flash-local \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --trust-remote-code \
    --host "$HOST" \
    --port "$PORT"
