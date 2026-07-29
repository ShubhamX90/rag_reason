#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /abs/path/to/input.jsonl" >&2
  exit 1
fi

echo "Delegating to dynamic watcher-based pipeline launcher to avoid duplicate collect jobs." >&2
exec "$ROOT_DIR/scripts/submit_benchmark_file_pipeline_dynamic.sh" "$1"
