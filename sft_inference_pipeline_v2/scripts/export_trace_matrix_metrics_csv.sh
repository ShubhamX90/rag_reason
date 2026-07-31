#!/usr/bin/env bash
# Consolidate the 96 trace-text benchmark conditions into one CSV.
#
# The export intentionally omits refusal/abstention-specific metrics, including
# abstention accuracy, refusal P/R/F1, prediction counts, and confusion counts.
# It retains the evaluation denominators needed to interpret the remaining
# structural, citation, document-verdict, conflict-type, and answer-overlap
# metrics.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results_root="$repo_root/final_model_outputs"
output_file="${1:-$results_root/trace_matrix_metrics_96.csv}"
temporary_file="$(mktemp)"

cleanup() {
  rm -f "$temporary_file"
}
trap cleanup EXIT

printf '%s\n' 'model,training_recipe,variant,information_mode,prompt_profile,benchmark_rows,contract_ok_pct,citation_pass_rate_pct,citation_avg_sentence_coverage,doc_examples_with_any_eval,doc_pairs_evaluated,doc_micro_accuracy_pct,doc_macro_f1,conflict_evaluated_ids,conflict_support,conflict_accuracy_pct,final_answer_scored_pairs,token_f1,rouge_l_f1,final_avg_citation_count,final_avg_unique_citations,final_citation_sentence_coverage,rows_with_invalid_citations,raw_output_path' > "$temporary_file"

while IFS= read -r raw_output; do
  relative_path="${raw_output#"$results_root"/}"
  IFS=/ read -r model information_mode prompt_profile variant _ <<< "$relative_path"
  report_prefix="${raw_output%.raw.jsonl}"

  case "$raw_output" in
    *'_trace_text_k_'*) training_recipe='Run K' ;;
    *'_trace_text_l_'*) training_recipe='Run L' ;;
    *) training_recipe='Base model' ;;
  esac

  jq -nr \
    --arg model "$model" \
    --arg recipe "$training_recipe" \
    --arg variant "$variant" \
    --arg mode "$information_mode" \
    --arg profile "$prompt_profile" \
    --arg raw_path "final_model_outputs/$relative_path" \
    --slurpfile contract "$report_prefix.contract.json" \
    --slurpfile doc "$report_prefix.doc_verdicts.json" \
    --slurpfile conflict "$report_prefix.conflict_type.json" \
    --slurpfile final "$report_prefix.final_answer.json" \
    '[
      $model,
      $recipe,
      $variant,
      $mode,
      $profile,
      $contract[0].total,
      $contract[0].ok_rate_pct,
      $contract[0].citation_coverage.pass_rate_pct,
      $contract[0].citation_coverage.avg_sentence_coverage,
      $doc[0].totals.examples_with_any_eval,
      $doc[0].totals.total_doc_pairs_evaluated,
      $doc[0].totals.micro_accuracy_doc_level,
      $doc[0].overall.macro_f1,
      $conflict[0].totals.evaluated_ids,
      $conflict[0].overall.support,
      $conflict[0].overall.accuracy,
      $final[0].lexical_overlap_non_abstain.scored_pairs,
      $final[0].lexical_overlap_non_abstain.avg_token_f1,
      $final[0].lexical_overlap_non_abstain.avg_rouge_l_f1,
      $final[0].citations.avg_citation_count,
      $final[0].citations.avg_unique_citations,
      $final[0].citations.avg_sentence_coverage,
      $final[0].citations.rows_with_invalid_citations,
      $raw_path
    ] | @csv' >> "$temporary_file"
done < <(
  find \
    "$results_root/qwen7b" \
    "$results_root/qwen32b" \
    "$results_root/llama8b" \
    "$results_root/mistral7b" \
    -type f -name '*.raw.jsonl' | sort
)

row_count="$(( $(wc -l < "$temporary_file") - 1 ))"
if [[ "$row_count" -ne 96 ]]; then
  printf 'Expected 96 trace-matrix rows; produced %s. Refusing to write %s.\n' "$row_count" "$output_file" >&2
  exit 1
fi

mv "$temporary_file" "$output_file"
trap - EXIT
printf 'Wrote %s rows to %s\n' "$row_count" "$output_file"
