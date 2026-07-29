#!/usr/bin/env bash
#SBATCH --job-name=ragann_qwen397_val49_e2e
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/pabitra/rag-reason/logs/ragann_qwen397_val49_e2e_%j.out
#SBATCH --error=/scratch/pabitra/rag-reason/logs/ragann_qwen397_val49_e2e_%j.err

set -euo pipefail

ENV_DIR="${ENV_DIR:-/scratch/pabitra/rag-reason/envs/local-judge-serving}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/rag-reason/dataset_annotation_pipeline_v3}"
MODEL_DIR="${MODEL_DIR:-/scratch/pabitra/rag-reason/models/Qwen3.5-397B-A17B-NVFP4}"
GCC_MODULE="${GCC_MODULE:-gcc-11.2.0-gcc-8.5.0-ov3qrz6}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.1.0-gcc-11.2.0-s5o57xp}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local/qwen3.5-397b-a17b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
STAGE_MODEL_LOCAL="${STAGE_MODEL_LOCAL:-1}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/tmp/${USER:-pabitra}/ragann_models/Qwen3.5-397B-A17B-NVFP4}"

INPUT_ALL="${INPUT_ALL:-outputs/local_committee_val49/inputs/val49_all_input.jsonl}"
OUT_ROOT="${OUT_ROOT:-outputs/local_committee_val49/qwen_e2e}"
STAGE1_OUT="${STAGE1_OUT:-outputs/local_committee_val49/collect/stage1_qwen_collect.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-$OUT_ROOT/stage1_split}"
STAGE2_CONFLICTS_OUT="${STAGE2_CONFLICTS_OUT:-$OUT_ROOT/stage2_conflicts_qwen.jsonl}"
STAGE2_REFUSALS_OUT="${STAGE2_REFUSALS_OUT:-$OUT_ROOT/stage2_refusals_qwen.jsonl}"
STAGE3_CONFLICTS_OUT="${STAGE3_CONFLICTS_OUT:-$OUT_ROOT/stage3_conflicts_qwen.jsonl}"
STAGE3_REFUSALS_OUT="${STAGE3_REFUSALS_OUT:-$OUT_ROOT/stage3_refusals_qwen.jsonl}"
FINAL_OUT="${FINAL_OUT:-$OUT_ROOT/stage3_qwen_combined.jsonl}"
CACHE_DIR="${CACHE_DIR:-data/.llm_cache/local_committee_val49}"
CONFIG="${CONFIG:-configs/local_committee/benchmark_stage_qwen397_collect.json}"
CONFIG_STAGE3="${CONFIG_STAGE3:-configs/local_committee/benchmark_stage_qwen397_stage3_collect.json}"
CONCURRENCY="${CONCURRENCY:-1}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$OUT_ROOT/vllm_${SLURM_JOB_ID:-manual}.log}"
ENDPOINT_READY_ATTEMPTS="${ENDPOINT_READY_ATTEMPTS:-480}"
ENDPOINT_READY_SLEEP_S="${ENDPOINT_READY_SLEEP_S:-10}"

export HF_HOME="${HF_HOME:-/scratch/pabitra/rag-reason/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/pabitra/rag-reason/cache/hf/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/pabitra/rag-reason/cache/vllm}"
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /scratch/pabitra/rag-reason/logs "$VLLM_CACHE_ROOT"

set +eu
unset -f module ml clearMT clearLmod 2>/dev/null || true
unset LMOD_CMD LMOD_DIR LMOD_SETTARG_CMD
source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module purge >/dev/null 2>&1 || true
module load "$GCC_MODULE" "$CUDA_MODULE"
MODULE_STATUS=$?
set -eu
if [ "$MODULE_STATUS" -ne 0 ]; then
  echo "warning=module_load_failed; continuing with inherited/system compiler paths"
fi

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
GCC_LIBSTDCPP_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
FALLBACK_GCC_LIBSTDCPP_DIR="/apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-11.2.0-ov3qrz6n2r7ysjhyvps5rlyu2qskp5b2/lib64"
if [[ "$GCC_LIBSTDCPP_DIR" == *redhat* && -d "$FALLBACK_GCC_LIBSTDCPP_DIR" ]]; then
  GCC_LIBSTDCPP_DIR="$FALLBACK_GCC_LIBSTDCPP_DIR"
fi
export LD_LIBRARY_PATH="${GCC_LIBSTDCPP_DIR}:${LD_LIBRARY_PATH:-}"

source "$ENV_DIR/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT" "$(dirname "$STAGE1_OUT")"

SERVE_MODEL_DIR="$MODEL_DIR"
if [ "$STAGE_MODEL_LOCAL" = "1" ]; then
  mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
  MODEL_SIZE_KB="$(du -sk "$MODEL_DIR" | awk '{print $1}')"
  LOCAL_SIZE_KB="0"
  if [ -d "$LOCAL_MODEL_DIR" ]; then
    LOCAL_SIZE_KB="$(du -sk "$LOCAL_MODEL_DIR" | awk '{print $1}')"
  fi
  if [ -f "$LOCAL_MODEL_DIR/config.json" ] && [ "$LOCAL_SIZE_KB" -gt "$((MODEL_SIZE_KB * 90 / 100))" ]; then
    echo "stage_model_local=1"
    echo "stage_reuse_existing=1"
    echo "stage_target=$LOCAL_MODEL_DIR"
    echo "stage_target_kb=$LOCAL_SIZE_KB"
    SERVE_MODEL_DIR="$LOCAL_MODEL_DIR"
  else
    LOCAL_AVAIL_KB="$(df -Pk "$(dirname "$LOCAL_MODEL_DIR")" | awk 'NR == 2 {print $4}')"
    if [ "$LOCAL_AVAIL_KB" -gt "$((MODEL_SIZE_KB + 10485760))" ]; then
    echo "stage_model_local=1"
    echo "stage_source=$MODEL_DIR"
    echo "stage_target=$LOCAL_MODEL_DIR"
    echo "stage_started_at=$(date)"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete --info=progress2 "$MODEL_DIR"/ "$LOCAL_MODEL_DIR"/
    else
      cp -a "$MODEL_DIR"/. "$LOCAL_MODEL_DIR"/
    fi
    echo "stage_finished_at=$(date)"
    SERVE_MODEL_DIR="$LOCAL_MODEL_DIR"
    else
      echo "stage_model_local=0; reason=insufficient_local_space; required_kb=$((MODEL_SIZE_KB + 10485760)); available_kb=$LOCAL_AVAIL_KB; existing_target_kb=$LOCAL_SIZE_KB"
    fi
  fi
fi

BASE_URL="http://$(hostname -f):${PORT}/v1"
export LOCAL_QWEN_BASE_URL="$BASE_URL"

echo "started_at=$(date)"
echo "hostname=$(hostname -f)"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project_dir=$PROJECT_DIR"
echo "env_dir=$ENV_DIR"
echo "model_dir=$MODEL_DIR"
echo "serve_model_dir=$SERVE_MODEL_DIR"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "endpoint=$BASE_URL"
echo "max_model_len=$MAX_MODEL_LEN"
echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "reasoning_parser=$REASONING_PARSER"
echo "stage1_out=$STAGE1_OUT"
echo "out_root=$OUT_ROOT"
echo "cache_dir=$CACHE_DIR"
echo "config=$CONFIG"
echo "config_stage3=$CONFIG_STAGE3"
echo "vllm_log_file=$VLLM_LOG_FILE"
echo "vllm_engine_ready_timeout_s=$VLLM_ENGINE_READY_TIMEOUT_S"
echo "endpoint_ready_attempts=$ENDPOINT_READY_ATTEMPTS"
echo "endpoint_ready_sleep_s=$ENDPOINT_READY_SLEEP_S"

nvidia-smi

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "stopping_vllm_pid=$VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "launching_vllm_at=$(date)"
VLLM_REASONING_ARGS=()
if [ "$REASONING_PARSER" != "none" ] && [ -n "$REASONING_PARSER" ]; then
  VLLM_REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER")
fi

stdbuf -oL -eL vllm serve "$SERVE_MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --language-model-only \
  "${VLLM_REASONING_ARGS[@]}" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --quantization modelopt_fp4 \
  --gdn-prefill-backend triton \
  --enable-prefix-caching \
  --enforce-eager \
  --uvicorn-log-level info \
  --host "$HOST" \
  --port "$PORT" > "$VLLM_LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "vllm_pid=$VLLM_PID"
echo "waiting_for_endpoint_at=$(date)"
for attempt in $(seq 1 "$ENDPOINT_READY_ATTEMPTS"); do
  if python - "$BASE_URL" <<'PY'
import sys
import httpx

base_url = sys.argv[1]
try:
    r = httpx.get(f"{base_url}/models", timeout=5.0)
    raise SystemExit(0 if r.status_code == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    echo "endpoint_ready_at=$(date)"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vllm_exited_before_ready"
    tail -120 "$VLLM_LOG_FILE" || true
    wait "$VLLM_PID"
  fi
  if [ "$attempt" -eq "$ENDPOINT_READY_ATTEMPTS" ]; then
    echo "endpoint_not_ready_after_attempts=$attempt"
    tail -200 "$VLLM_LOG_FILE" || true
    exit 1
  fi
  sleep "$ENDPOINT_READY_SLEEP_S"
done

tail -80 "$VLLM_LOG_FILE" || true

python scripts/probe_local_openai_endpoint.py \
  --base-url "$BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --timeout 240 \
  --retries 1 \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'

run_python() {
  echo
  echo ">>> $*"
  "$@"
}

run_python python scripts/run_stage1_multi_async.py \
  --input "$INPUT_ALL" \
  --output "$STAGE1_OUT" \
  --committee-backend local_openai \
  --committee-config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --cache-mode read_write \
  --concurrency "$CONCURRENCY"

run_python python scripts/split_val49_by_origin.py \
  --input "$STAGE1_OUT" \
  --output-dir "$SPLIT_DIR"

run_python python scripts/run_stage2_multi_async.py \
  --input "$SPLIT_DIR/val49_conflicts.jsonl" \
  --output "$STAGE2_CONFLICTS_OUT" \
  --committee-backend local_openai \
  --committee-config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --cache-mode read_write \
  --concurrency "$CONCURRENCY"

run_python python scripts/run_stage2_multi_async.py \
  --input "$SPLIT_DIR/val49_refusals.jsonl" \
  --output "$STAGE2_REFUSALS_OUT" \
  --refusal-mode \
  --committee-backend local_openai \
  --committee-config "$CONFIG" \
  --cache-dir "$CACHE_DIR" \
  --cache-mode read_write \
  --concurrency "$CONCURRENCY"

run_python python scripts/run_stage3_multi_async.py \
  --input "$STAGE2_CONFLICTS_OUT" \
  --output "$STAGE3_CONFLICTS_OUT" \
  --committee-backend local_openai \
  --committee-config "$CONFIG_STAGE3" \
  --cache-dir "$CACHE_DIR" \
  --cache-mode read_write \
  --concurrency "$CONCURRENCY"

run_python python scripts/run_stage3_multi_async.py \
  --input "$STAGE2_REFUSALS_OUT" \
  --output "$STAGE3_REFUSALS_OUT" \
  --refusal-mode \
  --committee-backend local_openai \
  --committee-config "$CONFIG_STAGE3" \
  --cache-dir "$CACHE_DIR" \
  --cache-mode read_write \
  --concurrency "$CONCURRENCY"

run_python python scripts/merge_val49_outputs.py \
  --conflicts "$STAGE3_CONFLICTS_OUT" \
  --refusals "$STAGE3_REFUSALS_OUT" \
  --output "$FINAL_OUT"

echo
echo "summary_at=$(date)"
wc -l "$STAGE1_OUT" "$STAGE2_CONFLICTS_OUT" "$STAGE2_REFUSALS_OUT" "$STAGE3_CONFLICTS_OUT" "$STAGE3_REFUSALS_OUT" "$FINAL_OUT"
python - "$FINAL_OUT" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
print("final_rows", len(rows))
print("origin_counts", Counter(row.get("_val49_origin") for row in rows))
print("conflict_type_counts", Counter(row.get("conflict_type") for row in rows))
print("answerable_counts", Counter(row.get("answerable_under_evidence") for row in rows))
PY

echo "finished_at=$(date)"
