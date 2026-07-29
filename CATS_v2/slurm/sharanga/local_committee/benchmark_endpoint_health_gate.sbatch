#!/usr/bin/env bash
#SBATCH --job-name=cats_endpoint_gate
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/%u/rag-reason/logs/cats_endpoint_gate_%j.out
#SBATCH --error=/scratch/%u/rag-reason/logs/cats_endpoint_gate_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?REPO_ROOT is required}"
QWEN_BASE_URL="${QWEN_BASE_URL:-}"
MISTRAL_BASE_URL="${MISTRAL_BASE_URL:-}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-}"

QWEN_SERVER_JOB_ID="${QWEN_SERVER_JOB_ID:-}"
MISTRAL_SERVER_JOB_ID="${MISTRAL_SERVER_JOB_ID:-}"
DEEPSEEK_SERVER_JOB_ID="${DEEPSEEK_SERVER_JOB_ID:-}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-local/qwen3.5-397b-a17b}"
MISTRAL_MODEL_ID="${MISTRAL_MODEL_ID:-local/mistral-small-4}"
DEEPSEEK_MODEL_ID="${DEEPSEEK_MODEL_ID:-local/deepseek-r1-distill-32b}"

QWEN_PORT="${QWEN_PORT:-8001}"
MISTRAL_PORT="${MISTRAL_PORT:-8004}"
DEEPSEEK_PORT="${DEEPSEEK_PORT:-8002}"

HEALTH_POLL_INTERVAL="${HEALTH_POLL_INTERVAL:-30}"
HEALTH_PROBE_TIMEOUT="${HEALTH_PROBE_TIMEOUT:-60}"
HEALTH_PROBE_RETRIES="${HEALTH_PROBE_RETRIES:-1}"
HEALTH_MAX_WAIT_SECONDS="${HEALTH_MAX_WAIT_SECONDS:-86400}"

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

echo "resolved_qwen_base_url=${QWEN_BASE_URL:-unset}"
echo "resolved_mistral_base_url=${MISTRAL_BASE_URL:-unset}"
echo "resolved_deepseek_base_url=${DEEPSEEK_BASE_URL:-unset}"

cmd=(
  python3 "$REPO_ROOT/scripts/wait_for_local_committee_health.py"
  --poll-interval "$HEALTH_POLL_INTERVAL"
  --probe-timeout "$HEALTH_PROBE_TIMEOUT"
  --probe-retries "$HEALTH_PROBE_RETRIES"
  --max-wait-seconds "$HEALTH_MAX_WAIT_SECONDS"
)

if [[ -n "$QWEN_BASE_URL" ]]; then
  cmd+=(--qwen-base-url "$QWEN_BASE_URL" --qwen-model "$QWEN_MODEL_ID")
fi
if [[ -n "$QWEN_SERVER_JOB_ID" ]]; then
  cmd+=(--qwen-job-id "$QWEN_SERVER_JOB_ID")
fi

if [[ -n "$MISTRAL_BASE_URL" ]]; then
  cmd+=(--mistral-base-url "$MISTRAL_BASE_URL" --mistral-model "$MISTRAL_MODEL_ID")
fi
if [[ -n "$MISTRAL_SERVER_JOB_ID" ]]; then
  cmd+=(--mistral-job-id "$MISTRAL_SERVER_JOB_ID")
fi

if [[ -n "$DEEPSEEK_BASE_URL" ]]; then
  cmd+=(--deepseek-base-url "$DEEPSEEK_BASE_URL" --deepseek-model "$DEEPSEEK_MODEL_ID")
fi
if [[ -n "$DEEPSEEK_SERVER_JOB_ID" ]]; then
  cmd+=(--deepseek-job-id "$DEEPSEEK_SERVER_JOB_ID")
fi

exec "${cmd[@]}"
