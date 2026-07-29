#!/usr/bin/env bash
#SBATCH --job-name=cats_bench_watch
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/cats_bench_watch_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/cats_bench_watch_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
INPUT_FILE="${INPUT_FILE:?INPUT_FILE is required}"
INPUT_ROOT="${INPUT_ROOT:?INPUT_ROOT is required}"
QWEN_BASE_URL="${QWEN_BASE_URL:-}"
MISTRAL_BASE_URL="${MISTRAL_BASE_URL:-}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-}"
QWEN_SERVER_JOB_ID="${QWEN_SERVER_JOB_ID:-}"
MISTRAL_SERVER_JOB_ID="${MISTRAL_SERVER_JOB_ID:-}"
DEEPSEEK_SERVER_JOB_ID="${DEEPSEEK_SERVER_JOB_ID:-}"
QWEN_PORT="${QWEN_PORT:-8001}"
MISTRAL_PORT="${MISTRAL_PORT:-8004}"
DEEPSEEK_PORT="${DEEPSEEK_PORT:-8002}"

GOLD_FILE="${GOLD_FILE:-$REPO_ROOT/data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-736}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/rag-reason/cats_outputs/benchmark_local_committee_3judge}"
RUN_LABEL="${RUN_LABEL:-}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_RETRIES="${MAX_RETRIES:-2}"

resolve_endpoint() {
  local current_url="$1"
  local job_id="$2"
  local port="$3"
  if [[ -n "$current_url" ]]; then
    printf '%s\n' "$current_url"
    return 0
  fi
  if [[ -z "$job_id" ]]; then
    return 1
  fi
  bash "$REPO_ROOT/scripts/resolve_slurm_job_endpoint.sh" "$job_id" "$port"
}

QWEN_BASE_URL="$(resolve_endpoint "$QWEN_BASE_URL" "$QWEN_SERVER_JOB_ID" "$QWEN_PORT" || true)"
MISTRAL_BASE_URL="$(resolve_endpoint "$MISTRAL_BASE_URL" "$MISTRAL_SERVER_JOB_ID" "$MISTRAL_PORT" || true)"
DEEPSEEK_BASE_URL="$(resolve_endpoint "$DEEPSEEK_BASE_URL" "$DEEPSEEK_SERVER_JOB_ID" "$DEEPSEEK_PORT" || true)"

: "${QWEN_BASE_URL:?QWEN_BASE_URL could not be resolved}"
: "${MISTRAL_BASE_URL:?MISTRAL_BASE_URL could not be resolved}"
: "${DEEPSEEK_BASE_URL:?DEEPSEEK_BASE_URL could not be resolved}"

exec bash "$REPO_ROOT/scripts/submit_benchmark_file_pipeline_dynamic.sh" \
  "$INPUT_FILE"
