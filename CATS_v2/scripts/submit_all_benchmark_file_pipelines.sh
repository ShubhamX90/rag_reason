#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$ROOT_DIR/inputs/prepped_model_eval_inputs/benchmark_set_all_modes}"

tmp_manifest="$(mktemp)"
trap 'rm -f "$tmp_manifest"' EXIT

find "$INPUT_ROOT" -type f -name 'input.jsonl' | sort > "$tmp_manifest"

if [[ ! -s "$tmp_manifest" ]]; then
  echo "No benchmark input.jsonl files found under: $INPUT_ROOT" >&2
  exit 1
fi

exec "$ROOT_DIR/scripts/submit_manifest_benchmark_watchers.sh" "$tmp_manifest"
