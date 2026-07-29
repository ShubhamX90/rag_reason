#!/usr/bin/env bash
#SBATCH --job-name=cats_deepseek32_server
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/cats_deepseek32_server_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/cats_deepseek32_server_%j.err

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER_NAME}/rag-reason}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/cache}"
ENV_DIR="${ENV_DIR:-${SCRATCH_ROOT}/envs/local-judge-serving}"
MODEL_DIR="${MODEL_DIR:-${SCRATCH_ROOT}/models/DeepSeek-R1-Distill-Qwen-32B}"
GCC_MODULE="${GCC_MODULE:-gcc-11.2.0-gcc-8.5.0-ov3qrz6}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.1.0-gcc-11.2.0-s5o57xp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local/deepseek-r1-distill-32b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/hf/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_USE_DEEP_GEMM=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_DIR" "$VLLM_CACHE_ROOT"

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

echo "started_at=$(date)"
echo "hostname=$(hostname -f)"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "env_dir=$ENV_DIR"
echo "model_dir=$MODEL_DIR"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "endpoint=http://$(hostname -f):${PORT}/v1"
echo "max_model_len=$MAX_MODEL_LEN"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "gcc=$(command -v gcc || echo missing)"
echo "nvcc=$(command -v nvcc || echo missing)"
echo "libstdcxx_dir=$GCC_LIBSTDCPP_DIR"

nvidia-smi

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --uvicorn-log-level info \
  --host "$HOST" \
  --port "$PORT"
