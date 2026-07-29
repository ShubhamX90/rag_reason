#!/usr/bin/env bash
# Build the Canada Quant / jasl DeepSeek-V4 H200 vLLM stack against CUDA 12.8.
# Sharanga's torch 2.11 cu13 stack hangs at import time; torch 2.11 cu128 works.

#SBATCH --job-name=cats_vllm_cu128
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pabitra/rag-reason/logs/cats_vllm_cu128_%j.out
#SBATCH --error=/scratch/pabitra/rag-reason/logs/cats_vllm_cu128_%j.err

set -euo pipefail

ENV_DIR="${ENV_DIR:-/scratch/pabitra/rag-reason/envs/deepseek-v4-serving-cu128}"
WORK_ROOT="${WORK_ROOT:-/scratch/pabitra/rag-reason/projects}"
VLLM_SRC="${VLLM_SRC:-${WORK_ROOT}/vllm-jasl-3424fba5-cu128}"
PATCH_REPO="${PATCH_REPO:-/scratch/pabitra/rag-reason/projects/dsv4-flash-w4a16-fp8-mtp}"
VLLM_SHA="${VLLM_SHA:-3424fba51301504262c3d8355e2560469f18c9c4}"
GCC_MODULE="${GCC_MODULE:-gcc-11.2.0-gcc-8.5.0-ov3qrz6}"

mkdir -p /scratch/pabitra/rag-reason/logs "$WORK_ROOT"

set +eu
unset -f module ml clearMT clearLmod 2>/dev/null || true
unset LMOD_CMD LMOD_DIR LMOD_SETTARG_CMD
source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module purge >/dev/null 2>&1 || true
module load "$GCC_MODULE"
MODULE_STATUS=$?
set -eu
if [ "$MODULE_STATUS" -ne 0 ]; then
  echo "warning=module_load_failed; continuing with inherited/system compiler paths"
fi
unset PYTHONHOME PYTHONPATH

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

echo "started_at=$(date)"
echo "hostname=$(hostname -f)"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "env_dir=$ENV_DIR"
echo "vllm_src=$VLLM_SRC"
echo "patch_repo=$PATCH_REPO"
echo "cuda_home=$CUDA_HOME"
echo "nvcc=$($CUDA_HOME/bin/nvcc --version | grep release | head -1)"
echo "gcc=$($CC --version | head -1)"
echo "g++=$($CXX --version | head -1)"

source "$ENV_DIR/bin/activate"
python - <<'PY'
import torch
print("torch", torch.__version__)
PY

pip install --upgrade "setuptools<78" wheel
pip install ninja cmake "numpy<3" pybind11 packaging setuptools-scm

if [ ! -d "$VLLM_SRC/.git" ]; then
  git clone https://github.com/jasl/vllm.git "$VLLM_SRC"
fi
cd "$VLLM_SRC"
git fetch --depth=1 origin "$VLLM_SHA"
git checkout "$VLLM_SHA"

echo "build_started_at=$(date)"
pip install --no-build-isolation -v --no-deps -e "$VLLM_SRC"
echo "build_finished_at=$(date)"

# Install runtime Python deps after the extension build, without letting pip
# replace the working torch cu128 stack.
pip install -r "$VLLM_SRC/requirements/common.txt" --no-deps || true

python "$PATCH_REPO/scripts/patch_v4_forcausal_packed_mapping.py" "$VLLM_SRC/vllm"
python "$PATCH_REPO/scripts/patch_mtp_packed_mapping.py" "$VLLM_SRC/vllm"

python - <<'PY'
import torch
import vllm
print("torch", torch.__version__)
print("vllm", getattr(vllm, "__version__", "unknown"))
print("vllm_path", vllm.__path__[0])
PY

echo "done_at=$(date)"
