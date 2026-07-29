#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export SHARANGA_HOST="${SHARANGA_HOST:-sharanga1}"
export SHARANGA_DEST="${SHARANGA_DEST:-/home/kudhru/rag-reason/sft_inference_pipeline_v2}"

exec bash "$PROJECT_ROOT/scripts/sync_sharanga.sh"
