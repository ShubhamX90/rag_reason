#!/usr/bin/env bash
#SBATCH --job-name=cats_mistral4_a100
#SBATCH --partition=gpu_a100_8
#SBATCH --qos=qos_gpu_a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/cats_mistral4_a100_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/cats_mistral4_a100_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USER_NAME="${USER:-$(whoami)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER_NAME}/rag-reason}"
HOME_ROOT="${HOME_ROOT:-/home/${USER_NAME}/rag-reason}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/cache}"
REPO_ROOT="${REPO_ROOT:-${HOME_ROOT}/CATS_v2}"
ENV_DIR="${ENV_DIR:-${SCRATCH_ROOT}/envs/local-judge-serving}"
MODEL_DIR="${MODEL_DIR:-${SCRATCH_ROOT}/models/Mistral-Small-4-119B-2603}"
GCC_MODULE="${GCC_MODULE:-gcc-11.2.0-gcc-8.5.0-ov3qrz6}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.1.0-gcc-11.2.0-s5o57xp}"
GCC_BIN_DIR="${GCC_BIN_DIR:-/apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-11.2.0-ov3qrz6n2r7ysjhyvps5rlyu2qskp5b2/bin}"
CUDA_BIN_DIR="${CUDA_BIN_DIR:-/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/cuda-12.1.0-s5o57xpc7sqqhfd6sxfealzbqxtmvh7p/bin}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/cuda-12.1.0-s5o57xpc7sqqhfd6sxfealzbqxtmvh7p/lib64}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8004}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local/mistral-small-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/mistral_chat_template.jinja}"
if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  CHAT_TEMPLATE="$REPO_ROOT/slurm/sharanga/local_committee/mistral_chat_template.jinja"
fi

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/hf/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP8=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_DIR" "$VLLM_CACHE_ROOT"

if [[ "${ALLOW_UNVERIFIED_A100_MISTRAL:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to launch Mistral Small 4 on A100 by default.

Reason:
- The current Sharanga/vLLM path for this model falls into an FP8/Marlin kernel
  route that has already failed on A100 with:
    RuntimeError: unsupported `a` scalar_type

Use one of these instead:
- slurm/sharanga/local_committee/mistral_small4_h100_tp2_server.sbatch
- slurm/sharanga/local_committee/mistral_small4_h200_tp2_server.sbatch

If you intentionally want to retry the A100 path after changing the serving
stack, resubmit with:
  --export=ALL,ALLOW_UNVERIFIED_A100_MISTRAL=1
EOF
  exit 2
fi

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

export PATH="${CUDA_BIN_DIR}:${GCC_BIN_DIR}:${PATH}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
export CUDACXX="${CUDA_BIN_DIR}/nvcc"
GCC_LIBSTDCPP_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
FALLBACK_GCC_LIBSTDCPP_DIR="/apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-11.2.0-ov3qrz6n2r7ysjhyvps5rlyu2qskp5b2/lib64"
if [[ "$GCC_LIBSTDCPP_DIR" == *redhat* && -d "$FALLBACK_GCC_LIBSTDCPP_DIR" ]]; then
  GCC_LIBSTDCPP_DIR="$FALLBACK_GCC_LIBSTDCPP_DIR"
fi
export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${GCC_LIBSTDCPP_DIR}:${LD_LIBRARY_PATH:-}"

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
echo "chat_template=$CHAT_TEMPLATE"
echo "gcc=$(command -v gcc || echo missing)"
echo "nvcc=$(command -v nvcc || echo missing)"
echo "libstdcxx_dir=$GCC_LIBSTDCPP_DIR"

nvidia-smi

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --language-model-only \
  --skip-mm-profiling \
  --chat-template "$CHAT_TEMPLATE" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --uvicorn-log-level info \
  --host "$HOST" \
  --port "$PORT"
