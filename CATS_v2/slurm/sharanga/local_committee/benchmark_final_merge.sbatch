#!/usr/bin/env bash
#SBATCH --job-name=cats_bench_merge
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/cats_bench_merge_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/cats_bench_merge_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
INPUT_FILE="${INPUT_FILE:?INPUT_FILE is required}"
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"

USER_NAME="${USER:-$(whoami)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER_NAME}/rag-reason}"
ENV_DIR="${ENV_DIR:-${SCRATCH_ROOT}/envs/local-judge-serving}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/cats_outputs/benchmark_local_committee_3judge}"
RUN_LABEL="${RUN_LABEL:-}"
BATCH_SIZE="${BATCH_SIZE:-12}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-12}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-30}"
QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8001/v1}"
MISTRAL_BASE_URL="${MISTRAL_BASE_URL:-http://127.0.0.1:8004/v1}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8002/v1}"

source "$ENV_DIR/bin/activate"

abs_input="$(python - <<'PY' "$INPUT_FILE"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
abs_root="$(python - <<'PY' "$INPUT_ROOT"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"

rel_dir="$(python - <<'PY' "$abs_root" "$abs_input"
from pathlib import Path
import sys
root = Path(sys.argv[1])
inp = Path(sys.argv[2])
print(inp.parent.relative_to(root))
PY
)"

if [[ -n "$RUN_LABEL" ]]; then
  file_root="$OUTPUT_ROOT/$RUN_LABEL/$rel_dir"
else
  file_root="$OUTPUT_ROOT/$rel_dir"
fi
cache_dir="$file_root/response_cache"
run_dir="$file_root/final"
config_path="$run_dir/run_config.yaml"

mkdir -p "$run_dir"

python "$REPO_ROOT/scripts/render_benchmark_local_committee_run_config.py" \
  --mode final \
  --run-dir "$run_dir" \
  --cache-dir "$cache_dir" \
  --batch-size "$BATCH_SIZE" \
  --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --qwen-base-url "$QWEN_BASE_URL" \
  --mistral-base-url "$MISTRAL_BASE_URL" \
  --deepseek-base-url "$DEEPSEEK_BASE_URL" \
  --output-config "$config_path"

echo "input_file=$INPUT_FILE"
echo "rel_dir=$rel_dir"
echo "run_dir=$run_dir"
echo "cache_dir=$cache_dir"

python "$REPO_ROOT/run_evaluation.py" \
  --input "$INPUT_FILE" \
  --config "$config_path" \
  --committee local
