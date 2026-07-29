#!/usr/bin/env bash
#SBATCH --job-name=cats_bench_collect
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/cats_bench_collect_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/cats_bench_collect_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
INPUT_FILE="${INPUT_FILE:?INPUT_FILE is required}"
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
JUDGE_NAME="${JUDGE_NAME:?JUDGE_NAME is required}"
BASE_URL="${BASE_URL:-}"
SERVER_JOB_ID="${SERVER_JOB_ID:-}"
SERVER_PORT="${SERVER_PORT:-}"

USER_NAME="${USER:-$(whoami)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER_NAME}/rag-reason}"
ENV_DIR="${ENV_DIR:-${SCRATCH_ROOT}/envs/local-judge-serving}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/cats_outputs/benchmark_local_committee_3judge}"
RUN_LABEL="${RUN_LABEL:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
GOLD_FILE="${GOLD_FILE:-$REPO_ROOT/data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-736}"

source "$ENV_DIR/bin/activate"

if [[ -z "$BASE_URL" ]]; then
  if [[ -z "$SERVER_JOB_ID" || -z "$SERVER_PORT" ]]; then
    echo "BASE_URL is required unless both SERVER_JOB_ID and SERVER_PORT are set" >&2
    exit 1
  fi
  BASE_URL="$(bash "$REPO_ROOT/scripts/resolve_slurm_job_endpoint.sh" "$SERVER_JOB_ID" "$SERVER_PORT")"
fi

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
run_dir="$file_root/staged/${JUDGE_NAME}_collect"
config_path="$run_dir/run_config.yaml"

mkdir -p "$run_dir" "$cache_dir"

python "$REPO_ROOT/scripts/validate_eval_input_jsonl.py" \
  --input "$INPUT_FILE" \
  --mode benchmark_prepped \
  --gold "$GOLD_FILE" \
  --expected-rows "$EXPECTED_ROWS"

python "$REPO_ROOT/scripts/render_benchmark_local_committee_run_config.py" \
  --mode collect \
  --judge "$JUDGE_NAME" \
  --run-dir "$run_dir" \
  --cache-dir "$cache_dir" \
  --base-url "$BASE_URL" \
  --batch-size "$BATCH_SIZE" \
  --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --output-config "$config_path"

echo "input_file=$INPUT_FILE"
echo "judge_name=$JUDGE_NAME"
echo "base_url=$BASE_URL"
echo "rel_dir=$rel_dir"
echo "run_dir=$run_dir"
echo "cache_dir=$cache_dir"

python "$REPO_ROOT/run_evaluation.py" \
  --input "$INPUT_FILE" \
  --config "$config_path" \
  --committee local
