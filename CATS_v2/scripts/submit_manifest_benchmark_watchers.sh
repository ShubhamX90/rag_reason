#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$ROOT_DIR/inputs/prepped_model_eval_inputs/benchmark_set_all_modes}"
USER_NAME="${USER:-$(whoami)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER_NAME}/rag-reason/cats_outputs/benchmark_local_committee_3judge}"
RUN_LABEL="${RUN_LABEL:-}"
GOLD_FILE="${GOLD_FILE:-$ROOT_DIR/data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-736}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_RETRIES="${MAX_RETRIES:-2}"
HEALTH_POLL_INTERVAL="${HEALTH_POLL_INTERVAL:-30}"
HEALTH_PROBE_TIMEOUT="${HEALTH_PROBE_TIMEOUT:-60}"
HEALTH_PROBE_RETRIES="${HEALTH_PROBE_RETRIES:-1}"
HEALTH_MAX_WAIT_SECONDS="${HEALTH_MAX_WAIT_SECONDS:-86400}"

QWEN_SERVER_JOB_ID="${QWEN_SERVER_JOB_ID:-}"
MISTRAL_SERVER_JOB_ID="${MISTRAL_SERVER_JOB_ID:-}"
DEEPSEEK_SERVER_JOB_ID="${DEEPSEEK_SERVER_JOB_ID:-}"

QWEN_PORT="${QWEN_PORT:-8001}"
MISTRAL_PORT="${MISTRAL_PORT:-8004}"
DEEPSEEK_PORT="${DEEPSEEK_PORT:-8002}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-local/qwen3.5-397b-a17b}"
MISTRAL_MODEL_ID="${MISTRAL_MODEL_ID:-local/mistral-small-4}"
DEEPSEEK_MODEL_ID="${DEEPSEEK_MODEL_ID:-local/deepseek-r1-distill-32b}"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /abs/path/to/manifest.txt" >&2
  exit 1
fi

MANIFEST_PATH="$1"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
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

prev_job=""
order=0

if [[ -n "$QWEN_SERVER_JOB_ID" || -n "$MISTRAL_SERVER_JOB_ID" || -n "$DEEPSEEK_SERVER_JOB_ID" ]]; then
  if [[ -z "$QWEN_SERVER_JOB_ID" || -z "$MISTRAL_SERVER_JOB_ID" || -z "$DEEPSEEK_SERVER_JOB_ID" ]]; then
    echo "If using health gating with server jobs, set all of QWEN_SERVER_JOB_ID, MISTRAL_SERVER_JOB_ID, and DEEPSEEK_SERVER_JOB_ID." >&2
    exit 1
  fi
else
  : "${QWEN_BASE_URL:?Set QWEN_BASE_URL to the live Qwen endpoint}"
  : "${MISTRAL_BASE_URL:?Set MISTRAL_BASE_URL to the live Mistral endpoint}"
  : "${DEEPSEEK_BASE_URL:?Set DEEPSEEK_BASE_URL to the live DeepSeek endpoint}"
fi

if [[ -n "$QWEN_SERVER_JOB_ID" || -n "$MISTRAL_SERVER_JOB_ID" || -n "$DEEPSEEK_SERVER_JOB_ID" ]]; then
  select_controller_args 4
  gate_job="$(sbatch --parsable \
    --dependency "after:${QWEN_SERVER_JOB_ID}:${MISTRAL_SERVER_JOB_ID}:${DEEPSEEK_SERVER_JOB_ID}" \
    "${CONTROLLER_SBATCH_ARGS[@]}" \
    --job-name "cats_gate" \
    --export=REPO_ROOT="$ROOT_DIR",QWEN_BASE_URL="${QWEN_BASE_URL:-}",MISTRAL_BASE_URL="${MISTRAL_BASE_URL:-}",DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-}",QWEN_SERVER_JOB_ID="$QWEN_SERVER_JOB_ID",MISTRAL_SERVER_JOB_ID="$MISTRAL_SERVER_JOB_ID",DEEPSEEK_SERVER_JOB_ID="$DEEPSEEK_SERVER_JOB_ID",QWEN_MODEL_ID="$QWEN_MODEL_ID",MISTRAL_MODEL_ID="$MISTRAL_MODEL_ID",DEEPSEEK_MODEL_ID="$DEEPSEEK_MODEL_ID",QWEN_PORT="$QWEN_PORT",MISTRAL_PORT="$MISTRAL_PORT",DEEPSEEK_PORT="$DEEPSEEK_PORT",HEALTH_POLL_INTERVAL="$HEALTH_POLL_INTERVAL",HEALTH_PROBE_TIMEOUT="$HEALTH_PROBE_TIMEOUT",HEALTH_PROBE_RETRIES="$HEALTH_PROBE_RETRIES",HEALTH_MAX_WAIT_SECONDS="$HEALTH_MAX_WAIT_SECONDS" \
    "$ROOT_DIR/slurm/sharanga/local_committee/benchmark_endpoint_health_gate.sbatch")"
  echo "health_gate_job=$gate_job" >&2
  prev_job="$gate_job"
fi

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

  dep_args=()
  if [[ -n "$prev_job" ]]; then
    dep_args=(--dependency "afterok:${prev_job}")
  fi

  select_controller_args 4
  watch_job="$(sbatch --parsable "${dep_args[@]}" "${CONTROLLER_SBATCH_ARGS[@]}" \
    --job-name "cats_sft$(printf '%02d' "$order")" \
    --export=ALL,REPO_ROOT="$ROOT_DIR",INPUT_ROOT="$INPUT_ROOT",INPUT_FILE="$input_file",QWEN_BASE_URL="${QWEN_BASE_URL:-}",MISTRAL_BASE_URL="${MISTRAL_BASE_URL:-}",DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-}",QWEN_SERVER_JOB_ID="${QWEN_SERVER_JOB_ID:-}",MISTRAL_SERVER_JOB_ID="${MISTRAL_SERVER_JOB_ID:-}",DEEPSEEK_SERVER_JOB_ID="${DEEPSEEK_SERVER_JOB_ID:-}",QWEN_PORT="$QWEN_PORT",MISTRAL_PORT="$MISTRAL_PORT",DEEPSEEK_PORT="$DEEPSEEK_PORT",GOLD_FILE="$GOLD_FILE",EXPECTED_ROWS="$EXPECTED_ROWS",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_LABEL="$RUN_LABEL",POLL_INTERVAL="$POLL_INTERVAL",MAX_RETRIES="$MAX_RETRIES" \
    "$ROOT_DIR/slurm/sharanga/local_committee/benchmark_watch_pipeline.sbatch")"

  printf "%03d\t%s\t%s\n" "$order" "$watch_job" "$rel_label"
  prev_job="$watch_job"
done < "$MANIFEST_PATH"

if (( order == 0 )); then
  echo "Manifest is empty: $MANIFEST_PATH" >&2
  exit 1
fi
