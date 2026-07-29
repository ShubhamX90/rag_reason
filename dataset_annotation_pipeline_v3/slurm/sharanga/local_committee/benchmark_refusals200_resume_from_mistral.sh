#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/rag-reason/dataset_annotation_pipeline_v3}"
ENV_DIR="${ENV_DIR:-/scratch/pabitra/rag-reason/envs/local-judge-serving}"
SELECTION_ROOT="${SELECTION_ROOT:-$PROJECT_DIR/data/benchmarks/fresh_refusals_selection_2026-06-21}"
RAW_INPUT="${RAW_INPUT:-$SELECTION_ROOT/refusals_200_fresh_high_quality_strict.jsonl}"
PREPARED_INPUT="${PREPARED_INPUT:-$SELECTION_ROOT/refusals_200_fresh_high_quality_strict_prepared.jsonl}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/outputs/local_committee_refusals200_3model}"
COLLECT_ROOT="${COLLECT_ROOT:-$OUT_ROOT/collect}"
FINAL_ROOT="${FINAL_ROOT:-$OUT_ROOT/final}"
CACHE_DIR="${CACHE_DIR:-data/.llm_cache/local_committee_refusals200_3model}"
FINAL_CFG="${FINAL_CFG:-configs/local_committee/benchmark3_stage_final_readonly.json}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_DIR/bin/python}"
LOG_ROOT="${LOG_ROOT:-/scratch/pabitra/rag-reason/logs}"

mkdir -p "$COLLECT_ROOT" "$FINAL_ROOT" "$LOG_ROOT"
cd "$PROJECT_DIR"

wait_for_job() {
  local label="$1"
  local job="$2"
  echo "waiting_for_${label}_job=$job"
  while true; do
    local active
    active="$(squeue -h -j "$job" -o '%A %T' || true)"
    if [ -z "$active" ]; then
      break
    fi
    echo "[$(date)] ${label}_active_job"
    echo "$active"
    sleep 60
  done

  local state
  state="$(sacct -j "$job" --format=State -n | head -n 1 | xargs)"
  echo "${label}_job_${job}_state=${state}"
  case "$state" in
    COMPLETED|COMPLETING) ;;
    *)
      echo "${label} failed for job $job" >&2
      exit 1
      ;;
  esac
}

submit_one() {
  local label="$1"
  local script="$2"
  local job
  echo "submitting_${label}_at=$(date)"
  job="$(sbatch --parsable \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_DIR="$ENV_DIR",SELECTION_ROOT="$SELECTION_ROOT",RAW_INPUT="$RAW_INPUT",PREPARED_INPUT="$PREPARED_INPUT",OUT_ROOT="$OUT_ROOT",COLLECT_ROOT="$COLLECT_ROOT",FINAL_ROOT="$FINAL_ROOT",CACHE_DIR="$CACHE_DIR",FINAL_CFG="$FINAL_CFG",PYTHON_BIN="$PYTHON_BIN",LOG_ROOT="$LOG_ROOT" \
    "$script")"
  echo "${label}_job_id=$job script=$script"
  wait_for_job "$label" "$job"
}

echo "resume_started_at=$(date)"
echo "project_dir=$PROJECT_DIR"
echo "prepared_input=$PREPARED_INPUT"
echo "out_root=$OUT_ROOT"
echo "cache_dir=$CACHE_DIR"

if [ ! -f "$PREPARED_INPUT" ]; then
  echo "Prepared input not found: $PREPARED_INPUT" >&2
  exit 1
fi

for required in \
  "$COLLECT_ROOT/stage1_qwen_collect.jsonl" \
  "$COLLECT_ROOT/stage1_deepseek_collect.jsonl"
do
  if [ ! -f "$required" ]; then
    echo "Required collect file missing: $required" >&2
    exit 1
  fi
done

rm -f \
  "$FINAL_ROOT/stage1_final_readonly.jsonl" \
  "$FINAL_ROOT/stage1_final_readonly_cost_report.json" \
  "$FINAL_ROOT/stage1_final_readonly_cost_ledger.jsonl" \
  "$FINAL_ROOT/stage1_final_readonly_cost_cumulative.json" \
  "$FINAL_ROOT/stage2_final_readonly.jsonl" \
  "$FINAL_ROOT/stage2_final_readonly_cost_report.json" \
  "$FINAL_ROOT/stage2_final_readonly_cost_ledger.jsonl" \
  "$FINAL_ROOT/stage2_final_readonly_cost_cumulative.json"

submit_one "stage1_mistral" "slurm/sharanga/local_committee/mistral4_refusals200_stage1_collect.sbatch"

echo "running_stage1_final_readonly_at=$(date)"
"$PYTHON_BIN" scripts/merge_stage1_committee_collect.py \
  --committee-config "$FINAL_CFG" \
  --member "local/qwen3.5-397b-a17b=$COLLECT_ROOT/stage1_qwen_collect.jsonl" \
  --member "local/deepseek-r1-distill-32b=$COLLECT_ROOT/stage1_deepseek_collect.jsonl" \
  --member "local/mistral-small-4=$COLLECT_ROOT/stage1_mistral_collect.jsonl" \
  --output "$FINAL_ROOT/stage1_final_readonly.jsonl"

"$PYTHON_BIN" scripts/validate_benchmark_gold.py \
  --input "$FINAL_ROOT/stage1_final_readonly.jsonl" \
  --stage stage1

submit_one "stage2_qwen" "slurm/sharanga/local_committee/qwen397_refusals200_stage2_collect.sbatch"
submit_one "stage2_deepseek" "slurm/sharanga/local_committee/deepseek32_refusals200_stage2_collect.sbatch"
submit_one "stage2_mistral" "slurm/sharanga/local_committee/mistral4_refusals200_stage2_collect.sbatch"

echo "running_stage2_final_readonly_at=$(date)"
"$PYTHON_BIN" scripts/merge_stage2_committee_collect.py \
  --committee-config "$FINAL_CFG" \
  --mode refusal \
  --member "local/qwen3.5-397b-a17b=$COLLECT_ROOT/stage2_qwen_collect.jsonl" \
  --member "local/deepseek-r1-distill-32b=$COLLECT_ROOT/stage2_deepseek_collect.jsonl" \
  --member "local/mistral-small-4=$COLLECT_ROOT/stage2_mistral_collect.jsonl" \
  --output "$FINAL_ROOT/stage2_final_readonly.jsonl"

"$PYTHON_BIN" scripts/validate_benchmark_gold.py \
  --input "$FINAL_ROOT/stage2_final_readonly.jsonl" \
  --stage stage2

echo "resume_finished_at=$(date)"
wc -l "$FINAL_ROOT/stage1_final_readonly.jsonl" "$FINAL_ROOT/stage2_final_readonly.jsonl"
