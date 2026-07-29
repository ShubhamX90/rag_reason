#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

SHARANGA_HOST="${SHARANGA_HOST:-pabitra@login.hpc.bits-hyderabad.ac.in}"
SHARANGA_DEST="${SHARANGA_DEST:-/home/pabitra/rag-reason/sft_inference_pipeline_v2}"
RSYNC_FLAGS="${RSYNC_FLAGS:--avzP}"
DRY_RUN="${DRY_RUN:-0}"
RSYNC_REMOTE="mkdir -p '$SHARANGA_DEST' && rsync"
SYNC_ITEMS=(
  README.md
  run.sh
  code
  prompts
  scripts
  slurm
  env
  docs
  data
)
EXISTING_SYNC_ITEMS=()

if [ "$DRY_RUN" = "1" ]; then
  RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"
fi

for item in "${SYNC_ITEMS[@]}"; do
  if [ -e "$item" ]; then
    EXISTING_SYNC_ITEMS+=("$item")
  fi
done

rsync $RSYNC_FLAGS \
  --rsync-path="$RSYNC_REMOTE" \
  --exclude '.git/' \
  --exclude 'backups/' \
  --exclude 'checkpoints/' \
  --exclude 'outputs/' \
  --exclude 'models/' \
  --exclude 'data/messages/' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  "${EXISTING_SYNC_ITEMS[@]}" \
  "$SHARANGA_HOST:$SHARANGA_DEST/"
