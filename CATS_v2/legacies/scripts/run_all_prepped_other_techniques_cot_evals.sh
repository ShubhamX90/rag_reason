#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$ROOT_DIR/outputs/prepped_model_eval_inputs/other_techniques_cot"
RUN_ROOT="$ROOT_DIR/outputs/model_output_eval_runs/other_techniques_cot"

if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Prepared input directory not found: $INPUT_ROOT" >&2
  exit 1
fi

INPUT_FILES=()
while IFS= read -r file; do
  INPUT_FILES+=("$file")
done < <(find "$INPUT_ROOT" -type f -name '*.jsonl' | sort)

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "No prepared JSONL files found under: $INPUT_ROOT" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"

echo "Found ${#INPUT_FILES[@]} prepared files."
echo "Results will be written under: $RUN_ROOT"
echo

for input_file in "${INPUT_FILES[@]}"; do
  rel_path="${input_file#"$INPUT_ROOT"/}"
  stem="$(basename "$input_file" .jsonl)"
  run_dir="$RUN_ROOT/$stem"
  run_config="$run_dir/run_config.yaml"

  mkdir -p "$run_dir"

  cat > "$run_config" <<EOF
outputs_dir: "$run_dir"
report_md: "$run_dir/eval_report.md"
detailed_results_json: "$run_dir/detailed_results.json"

pipeline:
  batch_size: 6
  verbose: true

conflict_eval:
  enable: true
  use_judge_committee: true
  correct_refusal_full_credit: true
  require_cross_doc_verification: false
  max_claims_per_answer: 8
  allow_paraphrases: true

  committee:
    type: "mixed"
    codex_model: "gpt-5.4"
    codex_priority: 3
    deepseek_model: "deepseek-v4-flash"
    deepseek_priority: 2
    max_concurrent_requests: 3
    voting_strategy: "weighted_majority"
EOF

  echo "============================================================"
  echo "Evaluating: $rel_path"
  echo "Run dir:    $run_dir"
  echo "============================================================"

  python3 "$ROOT_DIR/run_evaluation.py" \
    --input "$input_file" \
    --config "$run_config" \
    --committee cli

  echo
done

echo "All evaluations completed."
echo "Outputs are under: $RUN_ROOT"
