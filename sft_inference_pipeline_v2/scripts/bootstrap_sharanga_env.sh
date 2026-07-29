#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -z "${SCRATCH:-}" ]; then
  export SCRATCH="/scratch/$USER"
fi

if [ ! -d "$SCRATCH" ]; then
  echo "Sharanga scratch directory not found: $SCRATCH" >&2
  exit 1
fi

ENV_PREFIX="${ENV_PREFIX:-$SCRATCH/rag-reason/envs/rag-reason}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PROJECT_CACHE_ROOT="${PROJECT_CACHE_ROOT:-$SCRATCH/rag-reason/cache}"
INSTALL_MODE="${INSTALL_MODE:-frozen}"
FREEZE_FILE="${FREEZE_FILE:-env/sharanga-working-freeze.txt}"

mkdir -p "$(dirname "$ENV_PREFIX")" "$PROJECT_CACHE_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if [ -d "$ENV_PREFIX" ]; then
  echo "[Sharanga] Reusing existing env: $ENV_PREFIX"
else
  echo "[Sharanga] Creating env at: $ENV_PREFIX"
  conda create -y -p "$ENV_PREFIX" python="$PYTHON_VERSION" pip
fi

conda activate "$ENV_PREFIX"

export HF_HOME="${HF_HOME:-$PROJECT_CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_CACHE_ROOT/pip}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$PIP_CACHE_DIR"

python -m pip install --upgrade pip setuptools wheel
case "$INSTALL_MODE" in
  frozen)
    test -f "$FREEZE_FILE"
    echo "[Sharanga] Installing pinned environment from: $FREEZE_FILE"
    python -m pip install --extra-index-url "$TORCH_INDEX_URL" -r "$FREEZE_FILE"
    ;;
  compatible)
    echo "[Sharanga] Installing compatibility ranges from env/common-requirements.txt"
    python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
    python -m pip install -r env/common-requirements.txt
    ;;
  *)
    echo "INSTALL_MODE must be 'frozen' or 'compatible', got: $INSTALL_MODE" >&2
    exit 2
    ;;
esac

python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
try:
    import bitsandbytes as bnb
    print("bitsandbytes", bnb.__version__)
except Exception as exc:
    print("bitsandbytes_import_error", repr(exc))
    raise
PY
