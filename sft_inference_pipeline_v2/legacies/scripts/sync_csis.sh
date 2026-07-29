#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

CSIS_HOST="${CSIS_HOST:-vsshekhawat@172.24.16.132}"
CSIS_DEST="${CSIS_DEST:-/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2}"
RSYNC_FLAGS="${RSYNC_FLAGS:--avzP}"
DRY_RUN="${DRY_RUN:-0}"
RSYNC_REMOTE="mkdir -p '$CSIS_DEST' && rsync"

if [ "$DRY_RUN" = "1" ]; then
  RSYNC_FLAGS="$RSYNC_FLAGS --dry-run"
fi

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
  README.md \
  run.sh \
  code \
  prompts \
  scripts \
  slurm \
  env \
  docs \
  data \
  "$CSIS_HOST:$CSIS_DEST/"
