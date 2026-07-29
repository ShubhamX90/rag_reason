#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 scripts/benchmark_human_preselection_cli.py "$@"
