#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$ROOT_DIR/inputs/prepped_model_eval_inputs/other_techniques}"
USER_NAME="${USER:-$(whoami)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER_NAME}/rag-reason/cats_outputs/benchmark_local_committee_3judge}"
GOLD_FILE="${GOLD_FILE:-$ROOT_DIR/data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-736}"
RUN_LABEL="${RUN_LABEL:-}"

QWEN_BASE_URL="${QWEN_BASE_URL:-}"
QWEN_SERVER_JOB_ID="${QWEN_SERVER_JOB_ID:-}"
QWEN_PORT="${QWEN_PORT:-8001}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-local/qwen3.5-397b-a17b}"

MISTRAL_BASE_URL="${MISTRAL_BASE_URL:-}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-}"

START_AFTER_JOB_ID="${START_AFTER_JOB_ID:-}"

HEALTH_POLL_INTERVAL="${HEALTH_POLL_INTERVAL:-30}"
HEALTH_PROBE_TIMEOUT="${HEALTH_PROBE_TIMEOUT:-60}"
HEALTH_PROBE_RETRIES="${HEALTH_PROBE_RETRIES:-1}"
HEALTH_MAX_WAIT_SECONDS="${HEALTH_MAX_WAIT_SECONDS:-86400}"

if [[ $# -ne 1 ]]; then
  echo "Usage: RUN_LABEL=... QWEN_SERVER_JOB_ID=... MISTRAL_BASE_URL=... DEEPSEEK_BASE_URL=... [START_AFTER_JOB_ID=...] $0 /abs/path/to/manifest.txt" >&2
  exit 1
fi

MANIFEST_PATH="$1"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

if [[ -z "$RUN_LABEL" ]]; then
  echo "RUN_LABEL is required" >&2
  exit 1
fi

if [[ -z "$QWEN_BASE_URL" && -z "$QWEN_SERVER_JOB_ID" ]]; then
  echo "Either QWEN_BASE_URL or QWEN_SERVER_JOB_ID is required" >&2
  exit 1
fi

if [[ -z "$MISTRAL_BASE_URL" ]]; then
  echo "MISTRAL_BASE_URL is required" >&2
  exit 1
fi

if [[ -z "$DEEPSEEK_BASE_URL" ]]; then
  echo "DEEPSEEK_BASE_URL is required" >&2
  exit 1
fi

select_controller_args() {
  local cpus="$1"
  local partition=""
  local qos=""
  local request_cpus=""
  local reason=""

  while IFS='=' read -r key value; do
    case "$key" in
      partition) partition="$value" ;;
      qos) qos="$value" ;;
      request_cpus) request_cpus="$value" ;;
      reason) reason="$value" ;;
    esac
  done < <(bash "$ROOT_DIR/scripts/select_controller_partition.sh" "$cpus")

  echo "controller_partition=$partition controller_qos=${qos:-none} controller_cpus=${request_cpus:-$cpus} controller_reason=$reason" >&2

  CONTROLLER_SBATCH_ARGS=(--partition "$partition")
  if [[ -n "$qos" ]]; then
    CONTROLLER_SBATCH_ARGS+=(--qos "$qos")
  fi
  if [[ -n "$request_cpus" ]]; then
    CONTROLLER_SBATCH_ARGS+=(--cpus-per-task "$request_cpus")
  fi
}

gate_dep_args=()
if [[ -n "$START_AFTER_JOB_ID" ]]; then
  gate_dep_args+=(--dependency "afterok:${START_AFTER_JOB_ID}")
fi

select_controller_args 4
gate_job="$(sbatch --parsable \
  "${gate_dep_args[@]}" \
  "${CONTROLLER_SBATCH_ARGS[@]}" \
  --job-name "cats_gate_qwen" \
  --export=REPO_ROOT="$ROOT_DIR",QWEN_BASE_URL="$QWEN_BASE_URL",QWEN_SERVER_JOB_ID="${QWEN_SERVER_JOB_ID:-}",QWEN_MODEL_ID="$QWEN_MODEL_ID",QWEN_PORT="$QWEN_PORT",HEALTH_POLL_INTERVAL="$HEALTH_POLL_INTERVAL",HEALTH_PROBE_TIMEOUT="$HEALTH_PROBE_TIMEOUT",HEALTH_PROBE_RETRIES="$HEALTH_PROBE_RETRIES",HEALTH_MAX_WAIT_SECONDS="$HEALTH_MAX_WAIT_SECONDS" \
  "$ROOT_DIR/slurm/sharanga/local_committee/benchmark_endpoint_health_gate.sbatch")"
echo "health_gate_job=$gate_job" >&2

prev_collect_job="$gate_job"
order=0

while IFS= read -r rel_input || [[ -n "$rel_input" ]]; do
  [[ -z "$rel_input" ]] && continue
  order=$((order + 1))

  if [[ "$rel_input" = /* ]]; then
    input_file="$rel_input"
    rel_label="$(python3 - <<'PY' "$INPUT_ROOT" "$input_file"
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
inp = Path(sys.argv[2]).resolve()
print(inp.relative_to(root))
PY
)"
  elif [[ -f "$ROOT_DIR/$rel_input" ]]; then
    input_file="$ROOT_DIR/$rel_input"
    rel_label="$(python3 - <<'PY' "$ROOT_DIR" "$input_file"
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
inp = Path(sys.argv[2]).resolve()
print(inp.relative_to(root))
PY
)"
  else
    input_file="$INPUT_ROOT/$rel_input"
    rel_label="$rel_input"
  fi

  if [[ ! -f "$input_file" ]]; then
    echo "Input file missing: $input_file" >&2
    exit 1
  fi

  select_controller_args 4
  collect_job="$(sbatch --parsable \
    --dependency "afterok:${prev_collect_job}" \
    "${CONTROLLER_SBATCH_ARGS[@]}" \
    --job-name "cats_otq$(printf '%02d' "$order")" \
    --export=ALL,REPO_ROOT="$ROOT_DIR",INPUT_ROOT="$INPUT_ROOT",INPUT_FILE="$input_file",JUDGE_NAME="qwen397",BASE_URL="$QWEN_BASE_URL",SERVER_JOB_ID="${QWEN_SERVER_JOB_ID:-}",SERVER_PORT="$QWEN_PORT",GOLD_FILE="$GOLD_FILE",EXPECTED_ROWS="$EXPECTED_ROWS",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_LABEL="$RUN_LABEL" \
    "$ROOT_DIR/slurm/sharanga/local_committee/benchmark_collect_eval.sbatch")"

  select_controller_args 4
  merge_job="$(sbatch --parsable \
    --dependency "afterok:${collect_job}" \
    "${CONTROLLER_SBATCH_ARGS[@]}" \
    --job-name "cats_otg$(printf '%02d' "$order")" \
    --export=ALL,REPO_ROOT="$ROOT_DIR",INPUT_ROOT="$INPUT_ROOT",INPUT_FILE="$input_file",QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:${QWEN_PORT}/v1}",MISTRAL_BASE_URL="$MISTRAL_BASE_URL",DEEPSEEK_BASE_URL="$DEEPSEEK_BASE_URL",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_LABEL="$RUN_LABEL" \
    "$ROOT_DIR/slurm/sharanga/local_committee/benchmark_final_merge.sbatch")"

  printf "%03d\t%s\t%s\t%s\n" "$order" "$collect_job" "$merge_job" "$rel_label"
  prev_collect_job="$collect_job"
done < "$MANIFEST_PATH"

if (( order == 0 )); then
  echo "Manifest is empty: $MANIFEST_PATH" >&2
  exit 1
fi
