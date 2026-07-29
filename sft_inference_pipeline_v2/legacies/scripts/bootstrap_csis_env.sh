#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-rag-reason}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[CSIS] Reusing existing env: $ENV_NAME"
else
  echo "[CSIS] Creating env from env/csis-conda.yml"
  conda env create -f env/csis-conda.yml
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -r env/common-requirements.txt

python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
PY
