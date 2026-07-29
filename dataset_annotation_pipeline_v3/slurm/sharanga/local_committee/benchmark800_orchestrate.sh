#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/rag-reason/dataset_annotation_pipeline_v3}"
ENV_DIR="${ENV_DIR:-/scratch/pabitra/rag-reason/envs/local-judge-serving}"
SELECTION_ROOT="${SELECTION_ROOT:-$PROJECT_DIR/human_reviews/benchmark/first_pass/benchmark_selection_final}"
PREPARED_INPUT="${PREPARED_INPUT:-$SELECTION_ROOT/benchmark_non_refusal_selected_800_prepared.jsonl}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/outputs/local_committee_benchmark800_3model}"
COLLECT_ROOT="${COLLECT_ROOT:-$OUT_ROOT/collect}"
FINAL_ROOT="${FINAL_ROOT:-$OUT_ROOT/final}"
CACHE_DIR="${CACHE_DIR:-data/.llm_cache/local_committee_benchmark800_3model}"
FINAL_CFG="${FINAL_CFG:-configs/local_committee/benchmark3_stage_final_readonly.json}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
LOG_ROOT="${LOG_ROOT:-/scratch/pabitra/rag-reason/logs}"

mkdir -p "$COLLECT_ROOT" "$FINAL_ROOT" "$LOG_ROOT"
cd "$PROJECT_DIR"

wait_for_jobs() {
  local stage_label="$1"
  shift
  local jobs=("$@")
  echo "waiting_for_${stage_label}_jobs=${jobs[*]}"
  while true; do
    local active
    active="$(squeue -h -j "$(IFS=,; echo "${jobs[*]}")" -o '%A %T' || true)"
    if [ -z "$active" ]; then
      break
    fi
    echo "[$(date)] ${stage_label}_active_jobs"
    echo "$active"
    sleep 60
  done

  local failed=0
  for job in "${jobs[@]}"; do
    local state
    state="$(sacct -j "$job" --format=State -n | head -n 1 | xargs)"
    echo "${stage_label}_job_${job}_state=${state}"
    case "$state" in
      COMPLETED|COMPLETING) ;;
      *)
        failed=1
        ;;
    esac
  done
  if [ "$failed" -ne 0 ]; then
    echo "One or more ${stage_label} jobs failed." >&2
    exit 1
  fi
}

echo "orchestrator_started_at=$(date)"
echo "project_dir=$PROJECT_DIR"
echo "prepared_input=$PREPARED_INPUT"
echo "out_root=$OUT_ROOT"

if [ ! -f "$PREPARED_INPUT" ]; then
  echo "Prepared input not found: $PREPARED_INPUT" >&2
  exit 1
fi

echo "submitting_stage1_collect_jobs_at=$(date)"
JOB_QWEN_S1="$(sbatch --parsable slurm/sharanga/local_committee/qwen397_benchmark800_stage1_collect.sbatch)"
JOB_DS_S1="$(sbatch --parsable slurm/sharanga/local_committee/deepseek32_benchmark800_stage1_collect.sbatch)"
JOB_MIS_S1="$(sbatch --parsable slurm/sharanga/local_committee/mistral4_benchmark800_stage1_collect.sbatch)"
echo "stage1_jobs qwen=$JOB_QWEN_S1 deepseek=$JOB_DS_S1 mistral=$JOB_MIS_S1"
wait_for_jobs "stage1" "$JOB_QWEN_S1" "$JOB_DS_S1" "$JOB_MIS_S1"

echo "running_stage1_final_readonly_at=$(date)"
"$PYTHON_BIN" scripts/run_stage1_multi_async.py \
  --input "$PREPARED_INPUT" \
  --output "$FINAL_ROOT/stage1_final_readonly.jsonl" \
  --committee-backend local_openai \
  --committee-config "$FINAL_CFG" \
  --cache-dir "$CACHE_DIR" \
  --cache-mode read_only \
  --concurrency 8 \
  --system-prompt prompts/system_stage1_benchmark.txt \
  --user-prompt prompts/user_stage1_benchmark.txt

"$PYTHON_BIN" scripts/validate_benchmark_gold.py \
  --input "$FINAL_ROOT/stage1_final_readonly.jsonl" \
  --stage stage1

echo "submitting_stage2_collect_jobs_at=$(date)"
JOB_QWEN_S2="$(sbatch --parsable slurm/sharanga/local_committee/qwen397_benchmark800_stage2_collect.sbatch)"
JOB_DS_S2="$(sbatch --parsable slurm/sharanga/local_committee/deepseek32_benchmark800_stage2_collect.sbatch)"
JOB_MIS_S2="$(sbatch --parsable slurm/sharanga/local_committee/mistral4_benchmark800_stage2_collect.sbatch)"
echo "stage2_jobs qwen=$JOB_QWEN_S2 deepseek=$JOB_DS_S2 mistral=$JOB_MIS_S2"
wait_for_jobs "stage2" "$JOB_QWEN_S2" "$JOB_DS_S2" "$JOB_MIS_S2"

echo "running_stage2_final_readonly_at=$(date)"
"$PYTHON_BIN" scripts/run_stage2_multi_async.py \
  --input "$FINAL_ROOT/stage1_final_readonly.jsonl" \
  --output "$FINAL_ROOT/stage2_final_readonly.jsonl" \
  --committee-backend local_openai \
  --committee-config "$FINAL_CFG" \
  --cache-dir "$CACHE_DIR" \
  --cache-mode read_only \
  --concurrency 8

"$PYTHON_BIN" scripts/validate_benchmark_gold.py \
  --input "$FINAL_ROOT/stage2_final_readonly.jsonl" \
  --stage stage2

echo "orchestrator_finished_at=$(date)"
wc -l "$FINAL_ROOT/stage1_final_readonly.jsonl" "$FINAL_ROOT/stage2_final_readonly.jsonl"
