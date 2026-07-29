#!/usr/bin/env bash
#SBATCH --job-name=ragann_qwen397_nvfp4
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pabitra/rag-reason/logs/ragann_qwen397_nvfp4_%j.out
#SBATCH --error=/scratch/pabitra/rag-reason/logs/ragann_qwen397_nvfp4_%j.err

set -euo pipefail

ENV_DIR="${ENV_DIR:-/scratch/pabitra/rag-reason/envs/local-judge-serving}"
MODEL_DIR="${MODEL_DIR:-/scratch/pabitra/rag-reason/models/Qwen3.5-397B-A17B-NVFP4}"
GCC_MODULE="${GCC_MODULE:-gcc-11.2.0-gcc-8.5.0-ov3qrz6}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.1.0-gcc-11.2.0-s5o57xp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local/qwen3.5-397b-a17b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
STAGE_MODEL_LOCAL="${STAGE_MODEL_LOCAL:-1}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/tmp/${USER:-pabitra}/ragann_models/Qwen3.5-397B-A17B-NVFP4}"

export HF_HOME="${HF_HOME:-/scratch/pabitra/rag-reason/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/pabitra/rag-reason/cache/hf/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/pabitra/rag-reason/cache/vllm}"
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch/pabitra/rag-reason/logs "$VLLM_CACHE_ROOT"

set +eu
unset -f module ml clearMT clearLmod 2>/dev/null || true
unset LMOD_CMD LMOD_DIR LMOD_SETTARG_CMD
source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module purge >/dev/null 2>&1 || true
module load "$GCC_MODULE" "$CUDA_MODULE"
MODULE_STATUS=$?
set -eu
if [ "$MODULE_STATUS" -ne 0 ]; then
  echo "warning=module_load_failed; continuing with inherited/system compiler paths"
fi

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
GCC_LIBSTDCPP_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
FALLBACK_GCC_LIBSTDCPP_DIR="/apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-11.2.0-ov3qrz6n2r7ysjhyvps5rlyu2qskp5b2/lib64"
if [[ "$GCC_LIBSTDCPP_DIR" == *redhat* && -d "$FALLBACK_GCC_LIBSTDCPP_DIR" ]]; then
  GCC_LIBSTDCPP_DIR="$FALLBACK_GCC_LIBSTDCPP_DIR"
fi
export LD_LIBRARY_PATH="${GCC_LIBSTDCPP_DIR}:${LD_LIBRARY_PATH:-}"

source "$ENV_DIR/bin/activate"

SERVE_MODEL_DIR="$MODEL_DIR"
if [ "$STAGE_MODEL_LOCAL" = "1" ]; then
  mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
  MODEL_SIZE_KB="$(du -sk "$MODEL_DIR" | awk '{print $1}')"
  LOCAL_AVAIL_KB="$(df -Pk "$(dirname "$LOCAL_MODEL_DIR")" | awk 'NR == 2 {print $4}')"
  if [ "$LOCAL_AVAIL_KB" -gt "$((MODEL_SIZE_KB + 10485760))" ]; then
    echo "stage_model_local=1"
    echo "stage_source=$MODEL_DIR"
    echo "stage_target=$LOCAL_MODEL_DIR"
    echo "stage_started_at=$(date)"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete --info=progress2 "$MODEL_DIR"/ "$LOCAL_MODEL_DIR"/
    else
      cp -a "$MODEL_DIR"/. "$LOCAL_MODEL_DIR"/
    fi
    echo "stage_finished_at=$(date)"
    SERVE_MODEL_DIR="$LOCAL_MODEL_DIR"
  else
    echo "stage_model_local=0; reason=insufficient_local_space; required_kb=$((MODEL_SIZE_KB + 10485760)); available_kb=$LOCAL_AVAIL_KB"
  fi
fi

echo "started_at=$(date)"
echo "hostname=$(hostname -f)"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "env_dir=$ENV_DIR"
echo "model_dir=$MODEL_DIR"
echo "serve_model_dir=$SERVE_MODEL_DIR"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "endpoint=http://$(hostname -f):${PORT}/v1"
echo "max_model_len=$MAX_MODEL_LEN"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "gcc_module=$GCC_MODULE"
echo "cuda_module=$CUDA_MODULE"
echo "gcc=$(command -v gcc || echo missing)"
echo "g++=$(command -v g++ || echo missing)"
echo "nvcc=$(command -v nvcc || echo missing)"
echo "libstdcxx_dir=$GCC_LIBSTDCPP_DIR"

nvidia-smi

echo "launching_vllm_at=$(date)"
exec vllm serve "$SERVE_MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --language-model-only \
  --reasoning-parser qwen3 \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --quantization modelopt_fp4 \
  --gdn-prefill-backend triton \
  --enable-prefix-caching \
  --enforce-eager \
  --uvicorn-log-level info \
  --host "$HOST" \
  --port "$PORT"
