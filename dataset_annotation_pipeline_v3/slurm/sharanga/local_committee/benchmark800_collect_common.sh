#!/usr/bin/env bash

set -euo pipefail

if [ -z "${JUDGE_SHORT:-}" ]; then
  echo "JUDGE_SHORT is required: deepseek | mistral | qwen" >&2
  exit 2
fi
if [ -z "${PIPELINE_STAGE:-}" ]; then
  echo "PIPELINE_STAGE is required: stage1_collect | stage2_collect" >&2
  exit 2
fi

case "$JUDGE_SHORT" in
  deepseek)
    MODEL_DIR="${MODEL_DIR:-/scratch/pabitra/rag-reason/models/DeepSeek-R1-Distill-Qwen-32B}"
    PORT="${PORT:-8002}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local/deepseek-r1-distill-32b}"
    CONFIG="${CONFIG:-configs/local_committee/benchmark3_stage_deepseek32_collect.json}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    STAGE_MODEL_LOCAL="${STAGE_MODEL_LOCAL:-0}"
    LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/tmp/${USER:-pabitra}/ragann_models/DeepSeek-R1-Distill-Qwen-32B}"
    BASE_URL_ENV_NAME="LOCAL_DEEPSEEK_BASE_URL"
    ;;
  mistral)
    MODEL_DIR="${MODEL_DIR:-/scratch/pabitra/rag-reason/models/Mistral-Small-4-119B-2603}"
    PORT="${PORT:-8004}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local/mistral-small-4}"
    CONFIG="${CONFIG:-configs/local_committee/benchmark3_stage_mistral4_collect.json}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    STAGE_MODEL_LOCAL="${STAGE_MODEL_LOCAL:-1}"
    LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/tmp/${USER:-pabitra}/ragann_models/Mistral-Small-4-119B-2603}"
    BASE_URL_ENV_NAME="LOCAL_MISTRAL_BASE_URL"
    ;;
  qwen)
    MODEL_DIR="${MODEL_DIR:-/scratch/pabitra/rag-reason/models/Qwen3.5-397B-A17B-NVFP4}"
    PORT="${PORT:-8001}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local/qwen3.5-397b-a17b}"
    CONFIG="${CONFIG:-configs/local_committee/benchmark3_stage_qwen397_collect.json}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
    STAGE_MODEL_LOCAL="${STAGE_MODEL_LOCAL:-1}"
    LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/tmp/${USER:-pabitra}/ragann_models/Qwen3.5-397B-A17B-NVFP4}"
    REASONING_PARSER="${REASONING_PARSER:-none}"
    BASE_URL_ENV_NAME="LOCAL_QWEN_BASE_URL"
    ;;
  *)
    echo "Unknown JUDGE_SHORT=$JUDGE_SHORT" >&2
    exit 2
    ;;
esac

ENV_DIR="${ENV_DIR:-/scratch/pabitra/rag-reason/envs/local-judge-serving}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/rag-reason/dataset_annotation_pipeline_v3}"
GCC_MODULE="${GCC_MODULE:-gcc-11.2.0-gcc-8.5.0-ov3qrz6}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.1.0-gcc-11.2.0-s5o57xp}"
HOST="${HOST:-0.0.0.0}"
SELECTION_ROOT="${SELECTION_ROOT:-human_reviews/benchmark/first_pass/benchmark_selection_final}"
RAW_INPUT="${RAW_INPUT:-$SELECTION_ROOT/benchmark_non_refusal_selected_800_raw.jsonl}"
PREPARED_INPUT="${PREPARED_INPUT:-$SELECTION_ROOT/benchmark_non_refusal_selected_800_prepared.jsonl}"
OUT_ROOT="${OUT_ROOT:-outputs/local_committee_benchmark800_3model}"
COLLECT_ROOT="${COLLECT_ROOT:-$OUT_ROOT/collect}"
FINAL_ROOT="${FINAL_ROOT:-$OUT_ROOT/final}"
STAGE1_OUT="${STAGE1_OUT:-$COLLECT_ROOT/stage1_${JUDGE_SHORT}_collect.jsonl}"
STAGE2_OUT="${STAGE2_OUT:-$COLLECT_ROOT/stage2_${JUDGE_SHORT}_collect.jsonl}"
CACHE_DIR="${CACHE_DIR:-data/.llm_cache/local_committee_benchmark800_3model}"
CONCURRENCY="${CONCURRENCY:-1}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$OUT_ROOT/vllm_${PIPELINE_STAGE}_${JUDGE_SHORT}_${SLURM_JOB_ID:-manual}.log}"
ENDPOINT_READY_ATTEMPTS="${ENDPOINT_READY_ATTEMPTS:-480}"
ENDPOINT_READY_SLEEP_S="${ENDPOINT_READY_SLEEP_S:-10}"

export HF_HOME="${HF_HOME:-/scratch/pabitra/rag-reason/cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/pabitra/rag-reason/cache/hf/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/pabitra/rag-reason/cache/vllm}"
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-2400}"
export VLLM_USE_DEEP_GEMM=0
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
GCC_LIBSTDCPP_PATH="$(readlink -f "$(gcc -print-file-name=libstdc++.so.6)")"
GCC_LIBSTDCPP_DIR="$(dirname "$GCC_LIBSTDCPP_PATH")"
FALLBACK_GCC_LIBSTDCPP_DIR="/apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/gcc-11.2.0-ov3qrz6n2r7ysjhyvps5rlyu2qskp5b2/lib64"
if [[ "$GCC_LIBSTDCPP_DIR" == *redhat* && -d "$FALLBACK_GCC_LIBSTDCPP_DIR" ]]; then
  GCC_LIBSTDCPP_DIR="$FALLBACK_GCC_LIBSTDCPP_DIR"
  GCC_LIBSTDCPP_PATH="$GCC_LIBSTDCPP_DIR/libstdc++.so.6"
fi
export LD_LIBRARY_PATH="${GCC_LIBSTDCPP_DIR}:${LD_LIBRARY_PATH:-}"
if [ ! -f "$GCC_LIBSTDCPP_PATH" ]; then
  echo "fatal=libstdcpp_missing path=$GCC_LIBSTDCPP_PATH" >&2
  exit 3
fi
export LD_PRELOAD="${GCC_LIBSTDCPP_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"

source "$ENV_DIR/bin/activate"
cd "$PROJECT_DIR"
mkdir -p "$OUT_ROOT" "$COLLECT_ROOT" "$FINAL_ROOT" "$(dirname "$STAGE1_OUT")" "$(dirname "$STAGE2_OUT")"

SERVE_MODEL_DIR="$MODEL_DIR"
if [ "$STAGE_MODEL_LOCAL" = "1" ]; then
  mkdir -p "$(dirname "$LOCAL_MODEL_DIR")"
  MODEL_SIZE_KB="$(du -sk "$MODEL_DIR" | awk '{print $1}')"
  LOCAL_SIZE_KB="0"
  if [ -d "$LOCAL_MODEL_DIR" ]; then
    LOCAL_SIZE_KB="$(du -sk "$LOCAL_MODEL_DIR" | awk '{print $1}')"
  fi
  echo "local_stage_enabled=1"
  echo "model_size_kb=$MODEL_SIZE_KB"
  echo "local_model_dir=$LOCAL_MODEL_DIR"
  echo "local_model_size_kb_before=$LOCAL_SIZE_KB"
  if [ -f "$LOCAL_MODEL_DIR/config.json" ] && [ "$LOCAL_SIZE_KB" -gt "$((MODEL_SIZE_KB * 90 / 100))" ]; then
    echo "local_stage_reuse=1"
    SERVE_MODEL_DIR="$LOCAL_MODEL_DIR"
  else
    LOCAL_AVAIL_KB="$(df -Pk "$(dirname "$LOCAL_MODEL_DIR")" | awk 'NR == 2 {print $4}')"
    echo "local_avail_kb=$LOCAL_AVAIL_KB"
    if [ "$LOCAL_AVAIL_KB" -gt "$((MODEL_SIZE_KB + 10485760))" ]; then
      echo "local_stage_copy=1"
      echo "local_stage_from=$MODEL_DIR"
      echo "local_stage_to=$LOCAL_MODEL_DIR"
      if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --info=progress2 "$MODEL_DIR"/ "$LOCAL_MODEL_DIR"/
      else
        echo "local_stage_fallback=cp"
        cp -a "$MODEL_DIR"/. "$LOCAL_MODEL_DIR"/
      fi
      LOCAL_SIZE_KB_AFTER="$(du -sk "$LOCAL_MODEL_DIR" | awk '{print $1}')"
      echo "local_model_size_kb_after=$LOCAL_SIZE_KB_AFTER"
      SERVE_MODEL_DIR="$LOCAL_MODEL_DIR"
    else
      echo "local_stage_copy=0"
      echo "local_stage_reason=insufficient_local_space"
    fi
  fi
fi

BASE_URL="http://$(hostname -f):${PORT}/v1"
export "$BASE_URL_ENV_NAME=$BASE_URL"

echo "started_at=$(date)"
echo "hostname=$(hostname -f)"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "judge_short=$JUDGE_SHORT"
echo "pipeline_stage=$PIPELINE_STAGE"
echo "project_dir=$PROJECT_DIR"
echo "serve_model_dir=$SERVE_MODEL_DIR"
echo "served_model_name=$SERVED_MODEL_NAME"
echo "endpoint=$BASE_URL"
echo "prepared_input=$PREPARED_INPUT"
echo "stage1_out=$STAGE1_OUT"
echo "stage2_out=$STAGE2_OUT"
echo "cache_dir=$CACHE_DIR"
echo "config=$CONFIG"
echo "cc=$CC"
echo "cxx=$CXX"
echo "gcc_libstdcpp_path=$GCC_LIBSTDCPP_PATH"
echo "gcc_libstdcpp_dir=$GCC_LIBSTDCPP_DIR"
echo "ld_library_path=$LD_LIBRARY_PATH"
echo "ld_preload=$LD_PRELOAD"

nvidia-smi

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

case "$JUDGE_SHORT" in
  deepseek)
    stdbuf -oL -eL vllm serve "$SERVE_MODEL_DIR" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size 1 \
      --trust-remote-code \
      --dtype auto \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --enforce-eager \
      --uvicorn-log-level info \
      --host "$HOST" \
      --port "$PORT" > "$VLLM_LOG_FILE" 2>&1 &
    ;;
  mistral)
    export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"
    stdbuf -oL -eL vllm serve "$SERVE_MODEL_DIR" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size 2 \
      --distributed-executor-backend mp \
      --trust-remote-code \
      --language-model-only \
      --skip-mm-profiling \
      --dtype auto \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --moe-backend triton \
      --linear-backend cutlass \
      --enforce-eager \
      --uvicorn-log-level info \
      --host "$HOST" \
      --port "$PORT" > "$VLLM_LOG_FILE" 2>&1 &
    ;;
  qwen)
    VLLM_REASONING_ARGS=()
    if [ "${REASONING_PARSER:-none}" != "none" ] && [ -n "${REASONING_PARSER:-}" ]; then
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
    ;;
esac
VLLM_PID=$!

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
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    tail -120 "$VLLM_LOG_FILE" || true
    wait "$VLLM_PID"
  fi
  if [ "$attempt" -eq "$ENDPOINT_READY_ATTEMPTS" ]; then
    tail -200 "$VLLM_LOG_FILE" || true
    exit 1
  fi
  sleep "$ENDPOINT_READY_SLEEP_S"
done

PROBE_EXTRA_BODY="${PROBE_EXTRA_BODY:-}"
if [ "$JUDGE_SHORT" = "qwen" ]; then
  PROBE_EXTRA_BODY="${PROBE_EXTRA_BODY:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
fi
PROBE_ARGS=()
if [ -n "$PROBE_EXTRA_BODY" ]; then
  PROBE_ARGS=(--extra-body-json "$PROBE_EXTRA_BODY")
fi
python scripts/probe_local_openai_endpoint.py \
  --base-url "$BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --timeout 240 \
  --retries 1 \
  "${PROBE_ARGS[@]}"

if [ "$PIPELINE_STAGE" = "stage1_collect" ]; then
  python scripts/run_stage1_multi_async.py \
    --input "$PREPARED_INPUT" \
    --output "$STAGE1_OUT" \
    --committee-backend local_openai \
    --committee-config "$CONFIG" \
    --cache-dir "$CACHE_DIR" \
    --cache-mode read_write \
    --concurrency "$CONCURRENCY" \
    --system-prompt prompts/system_stage1_benchmark.txt \
    --user-prompt prompts/user_stage1_benchmark.txt
  wc -l "$STAGE1_OUT"
  exit 0
fi

if [ "$PIPELINE_STAGE" = "stage2_collect" ]; then
  python scripts/run_stage2_multi_async.py \
    --input "$FINAL_ROOT/stage1_final_readonly.jsonl" \
    --output "$STAGE2_OUT" \
    --committee-backend local_openai \
    --committee-config "$CONFIG" \
    --cache-dir "$CACHE_DIR" \
    --cache-mode read_write \
    --concurrency "$CONCURRENCY"
  wc -l "$STAGE2_OUT"
  exit 0
fi

echo "Unknown PIPELINE_STAGE=$PIPELINE_STAGE" >&2
exit 2
