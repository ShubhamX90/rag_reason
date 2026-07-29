# Validation Set Metrics Compilation

Generated on: 2026-05-31 14:20:08

Scope: all validation-set generation files under `model_output_exports/val set`.

- Total exports covered: 78
- Metrics sources per export: `final_answer.json`, `contract.json`, `conflict_type.json`, `doc_verdicts.json`
- Export-to-report mapping method: exact file hash match against `outputs/*.jsonl`, then report folder lookup in `outputs/reports/`
- List-valued diagnostic fields are recorded as `.count`; raw example IDs and samples remain in the underlying JSON reports.

## `e2e` / `baselines`

### `model_output_exports/val set/e2e/baselines/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `71.43` |
| `abstain.false_negative_ids.count` | `14` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `1` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2165` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3054` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/llama31_8b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `71.43` |
| `abstain.false_negative_ids.count` | `14` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `1` |
| `citations.avg_citation_count` | `0.3878` |
| `citations.avg_sentence_coverage` | `0.165` |
| `citations.avg_unique_citations` | `0.3673` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.146` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.1926` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `71.4` |
| `abstain_gold.correct` | `35` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `14` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.1684` |
| `citation_coverage.below_threshold_examples.count` | `42` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `48` |
| `citation_coverage.pass_count` | `6` |
| `citation_coverage.pass_rate_pct` | `12.5` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `14` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `5` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `17` |
| `label_f1.macro_f1` | `0.2095` |
| `label_f1.pairs_evaluated` | `46` |
| `label_f1.per_class.Complementary information.f1` | `0.0` |
| `label_f1.per_class.Complementary information.precision` | `0.0` |
| `label_f1.per_class.Complementary information.recall` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.4615` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.3333` |
| `label_f1.per_class.No conflict.f1` | `0.5862` |
| `label_f1.per_class.No conflict.precision` | `0.4146` |
| `label_f1.per_class.No conflict.recall` | `1.0` |
| `ok_all_checks` | `32` |
| `ok_ignoring_abstain_evidence_violation` | `33` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `67.3` |
| `ok_ignoring_abstain_support_violation` | `33` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `67.3` |
| `ok_rate_pct` | `65.3` |
| `problems.count` | `17` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `43.48` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `14` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `5` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `17` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_actual.No conflict` | `17` |
| `overall.distribution_pred.Complementary information` | `1` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `4` |
| `overall.distribution_pred.No conflict` | `41` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.462` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.333` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `9` |
| `overall.per_class.No conflict.f1` | `0.586` |
| `overall.per_class.No conflict.precision` | `0.415` |
| `overall.per_class.No conflict.recall` | `1.0` |
| `overall.per_class.No conflict.support` | `17` |
| `overall.support` | `46` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `8` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `25` |
| `overall.confusion_matrix.irrelevant.partially supports` | `5` |
| `overall.confusion_matrix.irrelevant.supports` | `25` |
| `overall.confusion_matrix.partially supports.irrelevant` | `20` |
| `overall.confusion_matrix.partially supports.partially supports` | `14` |
| `overall.confusion_matrix.partially supports.supports` | `119` |
| `overall.confusion_matrix.supports.irrelevant` | `6` |
| `overall.confusion_matrix.supports.partially supports` | `3` |
| `overall.confusion_matrix.supports.supports` | `166` |
| `overall.macro_f1` | `0.4387` |
| `overall.per_class.irrelevant.f1` | `0.4717` |
| `overall.per_class.irrelevant.precision` | `0.4902` |
| `overall.per_class.irrelevant.recall` | `0.4545` |
| `overall.per_class.partially supports.f1` | `0.16` |
| `overall.per_class.partially supports.precision` | `0.6364` |
| `overall.per_class.partially supports.recall` | `0.0915` |
| `overall.per_class.supports.f1` | `0.6845` |
| `overall.per_class.supports.precision` | `0.5355` |
| `overall.per_class.supports.recall` | `0.9486` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `205` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `53.52` |
| `totals.total_doc_pairs_evaluated` | `383` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `57.14` |
| `abstain.false_negative_ids.count` | `11` |
| `abstain.false_positive_ids.count` | `10` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `14` |
| `citations.avg_citation_count` | `2.2449` |
| `citations.avg_sentence_coverage` | `0.3472` |
| `citations.avg_unique_citations` | `2.0204` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.1899` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2899` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `24` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `48` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `12` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `10` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `10` |
| `abstain_gold.accuracy_pct` | `57.1` |
| `abstain_gold.correct` | `28` |
| `abstain_gold.false_abstain_ids.count` | `9` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `12` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.4726` |
| `citation_coverage.below_threshold_examples.count` | `25` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `36` |
| `citation_coverage.pass_count` | `11` |
| `citation_coverage.pass_rate_pct` | `30.6` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `13` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `8` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `16` |
| `label_f1.macro_f1` | `0.16` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.0` |
| `label_f1.per_class.Complementary information.precision` | `0.0` |
| `label_f1.per_class.Complementary information.recall` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.2667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.4` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.2` |
| `label_f1.per_class.No conflict.f1` | `0.5333` |
| `label_f1.per_class.No conflict.precision` | `0.381` |
| `label_f1.per_class.No conflict.recall` | `0.8889` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `37.5` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `13` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `8` |
| `overall.confusion_matrix.No conflict.Complementary information` | `1` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `16` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `1` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `5` |
| `overall.distribution_pred.No conflict` | `42` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.267` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.4` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.2` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.533` |
| `overall.per_class.No conflict.precision` | `0.381` |
| `overall.per_class.No conflict.recall` | `0.889` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.no_json_array` | `48` |
| `error_counts.think_block_missing_or_misaligned` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2351` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3511` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/mistral7b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `65.31` |
| `abstain.false_negative_ids.count` | `14` |
| `abstain.false_positive_ids.count` | `3` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `4` |
| `citations.avg_citation_count` | `0.1633` |
| `citations.avg_sentence_coverage` | `0.0162` |
| `citations.avg_unique_citations` | `0.1633` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.166` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2445` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `31` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `36` |
| `trace_presence.think_count` | `3` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `3` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `1` |
| `label_f1.macro_f1` | `0.3333` |
| `label_f1.pairs_evaluated` | `3` |
| `label_f1.per_class.Complementary information.f1` | `0.6667` |
| `label_f1.per_class.Complementary information.precision` | `0.5` |
| `label_f1.per_class.Complementary information.recall` | `1.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `label_f1.per_class.No conflict.f1` | `1.0` |
| `label_f1.per_class.No conflict.precision` | `1.0` |
| `label_f1.per_class.No conflict.recall` | `1.0` |
| `ok_all_checks` | `2` |
| `ok_ignoring_abstain_evidence_violation` | `2` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `4.1` |
| `ok_ignoring_abstain_support_violation` | `2` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `4.1` |
| `ok_rate_pct` | `4.1` |
| `problems.count` | `47` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `66.67` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `1` |
| `overall.distribution_actual.Complementary information` | `1` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `1` |
| `overall.distribution_actual.No conflict` | `1` |
| `overall.distribution_pred.Complementary information` | `2` |
| `overall.distribution_pred.No conflict` | `1` |
| `overall.per_class.Complementary information.f1` | `0.667` |
| `overall.per_class.Complementary information.precision` | `0.5` |
| `overall.per_class.Complementary information.recall` | `1.0` |
| `overall.per_class.Complementary information.support` | `1` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `1` |
| `overall.per_class.No conflict.f1` | `1.0` |
| `overall.per_class.No conflict.precision` | `1.0` |
| `overall.per_class.No conflict.recall` | `1.0` |
| `overall.per_class.No conflict.support` | `1` |
| `overall.support` | `3` |
| `top_confusions.count` | `5` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `1` |
| `error_counts.think_block_missing_or_misaligned` | `46` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `1` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `13` |
| `overall.confusion_matrix.supports.irrelevant` | `1` |
| `overall.confusion_matrix.supports.partially supports` | `2` |
| `overall.confusion_matrix.supports.supports` | `15` |
| `overall.macro_f1` | `0.2128` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.6383` |
| `overall.per_class.supports.precision` | `0.5172` |
| `overall.per_class.supports.recall` | `0.8333` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `15` |
| `totals.examples_with_any_eval` | `3` |
| `totals.micro_accuracy_doc_level` | `46.88` |
| `totals.total_doc_pairs_evaluated` | `32` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `73.47` |
| `abstain.false_negative_ids.count` | `13` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `2` |
| `citations.avg_citation_count` | `2.4286` |
| `citations.avg_sentence_coverage` | `0.264` |
| `citations.avg_unique_citations` | `2.3673` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.1831` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.269` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `73.5` |
| `abstain_gold.correct` | `36` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `13` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.2752` |
| `citation_coverage.below_threshold_examples.count` | `44` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `47` |
| `citation_coverage.pass_count` | `3` |
| `citation_coverage.pass_rate_pct` | `6.4` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `5` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `10` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `7` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `6` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `13` |
| `label_f1.macro_f1` | `0.1648` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.3333` |
| `label_f1.per_class.Complementary information.precision` | `0.3333` |
| `label_f1.per_class.Complementary information.recall` | `0.3333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `label_f1.per_class.No conflict.f1` | `0.4906` |
| `label_f1.per_class.No conflict.precision` | `0.3824` |
| `label_f1.per_class.No conflict.recall` | `0.6842` |
| `ok_all_checks` | `33` |
| `ok_ignoring_abstain_evidence_violation` | `33` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `67.3` |
| `ok_ignoring_abstain_support_violation` | `33` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `67.3` |
| `ok_rate_pct` | `67.3` |
| `problems.count` | `16` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `36.73` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `5` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `10` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `4` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `7` |
| `overall.confusion_matrix.No conflict.Complementary information` | `6` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `13` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.No conflict` | `34` |
| `overall.per_class.Complementary information.f1` | `0.333` |
| `overall.per_class.Complementary information.precision` | `0.333` |
| `overall.per_class.Complementary information.recall` | `0.333` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.491` |
| `overall.per_class.No conflict.precision` | `0.382` |
| `overall.per_class.No conflict.recall` | `0.684` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `6` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `19` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `33` |
| `overall.confusion_matrix.irrelevant.partially supports` | `15` |
| `overall.confusion_matrix.irrelevant.supports` | `8` |
| `overall.confusion_matrix.partially supports.irrelevant` | `34` |
| `overall.confusion_matrix.partially supports.partially supports` | `55` |
| `overall.confusion_matrix.partially supports.supports` | `62` |
| `overall.confusion_matrix.supports.irrelevant` | `24` |
| `overall.confusion_matrix.supports.partially supports` | `38` |
| `overall.confusion_matrix.supports.supports` | `103` |
| `overall.macro_f1` | `0.4944` |
| `overall.per_class.irrelevant.f1` | `0.449` |
| `overall.per_class.irrelevant.precision` | `0.3626` |
| `overall.per_class.irrelevant.recall` | `0.5893` |
| `overall.per_class.partially supports.f1` | `0.4247` |
| `overall.per_class.partially supports.precision` | `0.5093` |
| `overall.per_class.partially supports.recall` | `0.3642` |
| `overall.per_class.supports.f1` | `0.6095` |
| `overall.per_class.supports.precision` | `0.5954` |
| `overall.per_class.supports.recall` | `0.6242` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `191` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `51.34` |
| `totals.total_doc_pairs_evaluated` | `372` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2802` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4414` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen25_32b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `83.67` |
| `abstain.false_negative_ids.count` | `8` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `7` |
| `citations.avg_citation_count` | `4.2041` |
| `citations.avg_sentence_coverage` | `0.6755` |
| `citations.avg_unique_citations` | `3.7143` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2053` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2844` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `4` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `77.6` |
| `abstain_gold.correct` | `38` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `11` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.7356` |
| `citation_coverage.below_threshold_examples.count` | `16` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `45` |
| `citation_coverage.pass_count` | `29` |
| `citation_coverage.pass_rate_pct` | `64.4` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `7` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `8` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `15` |
| `label_f1.macro_f1` | `0.2439` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.4375` |
| `label_f1.per_class.Complementary information.precision` | `0.4118` |
| `label_f1.per_class.Complementary information.recall` | `0.4667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.1818` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.1` |
| `label_f1.per_class.No conflict.f1` | `0.6` |
| `label_f1.per_class.No conflict.precision` | `0.4839` |
| `label_f1.per_class.No conflict.recall` | `0.7895` |
| `ok_all_checks` | `37` |
| `ok_ignoring_abstain_evidence_violation` | `37` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `75.5` |
| `ok_ignoring_abstain_support_violation` | `37` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `75.5` |
| `ok_rate_pct` | `75.5` |
| `problems.count` | `12` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `46.94` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `7` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `8` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `15` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `17` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `1` |
| `overall.distribution_pred.No conflict` | `31` |
| `overall.per_class.Complementary information.f1` | `0.437` |
| `overall.per_class.Complementary information.precision` | `0.412` |
| `overall.per_class.Complementary information.recall` | `0.467` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.182` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.1` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.6` |
| `overall.per_class.No conflict.precision` | `0.484` |
| `overall.per_class.No conflict.recall` | `0.789` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `5` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `4` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `50` |
| `overall.confusion_matrix.irrelevant.partially supports` | `5` |
| `overall.confusion_matrix.irrelevant.supports` | `1` |
| `overall.confusion_matrix.partially supports.irrelevant` | `49` |
| `overall.confusion_matrix.partially supports.partially supports` | `58` |
| `overall.confusion_matrix.partially supports.supports` | `48` |
| `overall.confusion_matrix.supports.irrelevant` | `13` |
| `overall.confusion_matrix.supports.partially supports` | `23` |
| `overall.confusion_matrix.supports.supports` | `140` |
| `overall.macro_f1` | `0.6146` |
| `overall.per_class.irrelevant.f1` | `0.5952` |
| `overall.per_class.irrelevant.precision` | `0.4464` |
| `overall.per_class.irrelevant.recall` | `0.8929` |
| `overall.per_class.partially supports.f1` | `0.4813` |
| `overall.per_class.partially supports.precision` | `0.6744` |
| `overall.per_class.partially supports.recall` | `0.3742` |
| `overall.per_class.supports.f1` | `0.7671` |
| `overall.per_class.supports.precision` | `0.7407` |
| `overall.per_class.supports.recall` | `0.7955` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `248` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `64.08` |
| `totals.total_doc_pairs_evaluated` | `387` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `71.43` |
| `abstain.false_negative_ids.count` | `12` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `5` |
| `citations.avg_citation_count` | `5.8163` |
| `citations.avg_sentence_coverage` | `0.7364` |
| `citations.avg_unique_citations` | `4.3061` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2179` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3097` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `73.5` |
| `abstain_gold.correct` | `36` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `13` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.7606` |
| `citation_coverage.below_threshold_examples.count` | `19` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `47` |
| `citation_coverage.pass_count` | `28` |
| `citation_coverage.pass_rate_pct` | `59.6` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `8` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `6` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `4` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `6` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.381` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.5` |
| `label_f1.per_class.Complementary information.precision` | `0.4706` |
| `label_f1.per_class.Complementary information.recall` | `0.5333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.3333` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `1.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.2` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.5` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.6667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.4` |
| `label_f1.per_class.No conflict.f1` | `0.5714` |
| `label_f1.per_class.No conflict.precision` | `0.5` |
| `label_f1.per_class.No conflict.recall` | `0.6667` |
| `ok_all_checks` | `35` |
| `ok_ignoring_abstain_evidence_violation` | `35` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `71.4` |
| `ok_ignoring_abstain_support_violation` | `35` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `71.4` |
| `ok_rate_pct` | `71.4` |
| `problems.count` | `14` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `24` |
| `overall.accuracy` | `52.08` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `8` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `6` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `4` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `overall.confusion_matrix.No conflict.Complementary information` | `6` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `17` |
| `overall.distribution_pred.Conflict due to outdated information` | `1` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `6` |
| `overall.distribution_pred.No conflict` | `24` |
| `overall.per_class.Complementary information.f1` | `0.5` |
| `overall.per_class.Complementary information.precision` | `0.471` |
| `overall.per_class.Complementary information.recall` | `0.533` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.333` |
| `overall.per_class.Conflict due to outdated information.precision` | `1.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.2` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.5` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.667` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.4` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.571` |
| `overall.per_class.No conflict.precision` | `0.5` |
| `overall.per_class.No conflict.recall` | `0.667` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `38` |
| `overall.confusion_matrix.irrelevant.partially supports` | `18` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `31` |
| `overall.confusion_matrix.partially supports.partially supports` | `112` |
| `overall.confusion_matrix.partially supports.supports` | `15` |
| `overall.confusion_matrix.supports.irrelevant` | `6` |
| `overall.confusion_matrix.supports.partially supports` | `73` |
| `overall.confusion_matrix.supports.supports` | `98` |
| `overall.macro_f1` | `0.6255` |
| `overall.per_class.irrelevant.f1` | `0.5802` |
| `overall.per_class.irrelevant.precision` | `0.5067` |
| `overall.per_class.irrelevant.recall` | `0.6786` |
| `overall.per_class.partially supports.f1` | `0.6205` |
| `overall.per_class.partially supports.precision` | `0.5517` |
| `overall.per_class.partially supports.recall` | `0.7089` |
| `overall.per_class.supports.f1` | `0.6759` |
| `overall.per_class.supports.precision` | `0.8673` |
| `overall.per_class.supports.recall` | `0.5537` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `248` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `63.43` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise.raw.jsonl`, `outputs/baseline_qwen25_stagewise_base_refresh_e2e_minimal_val_stagewise.raw.jsonl`, `outputs/baseline_qwen25_stagewise_base_refresh_e2e_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `71.43` |
| `abstain.false_negative_ids.count` | `14` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `1` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2689` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.402` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen25_7b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen25_stagewise_base_refresh_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `63.27` |
| `abstain.false_negative_ids.count` | `9` |
| `abstain.false_positive_ids.count` | `9` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `3.2449` |
| `citations.avg_sentence_coverage` | `0.2735` |
| `citations.avg_unique_citations` | `3.1633` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.1894` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2558` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `25` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `10` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `5` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `5` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `5` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `5` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `5` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `10` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.3222` |
| `citation_coverage.below_threshold_examples.count` | `37` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `39` |
| `citation_coverage.pass_count` | `2` |
| `citation_coverage.pass_rate_pct` | `5.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `3` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `3` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `9` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `4` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `17` |
| `label_f1.macro_f1` | `0.2748` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.2727` |
| `label_f1.per_class.Complementary information.precision` | `0.4286` |
| `label_f1.per_class.Complementary information.recall` | `0.2` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.4211` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.4444` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.4` |
| `label_f1.per_class.No conflict.f1` | `0.68` |
| `label_f1.per_class.No conflict.precision` | `0.5312` |
| `label_f1.per_class.No conflict.recall` | `0.9444` |
| `ok_all_checks` | `15` |
| `ok_ignoring_abstain_evidence_violation` | `16` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `32.7` |
| `ok_ignoring_abstain_support_violation` | `16` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `32.7` |
| `ok_rate_pct` | `30.6` |
| `problems.count` | `34` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `50.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `3` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `3` |
| `overall.confusion_matrix.Complementary information.No conflict` | `9` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `4` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `17` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `7` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `32` |
| `overall.per_class.Complementary information.f1` | `0.273` |
| `overall.per_class.Complementary information.precision` | `0.429` |
| `overall.per_class.Complementary information.recall` | `0.2` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.421` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.444` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.4` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.68` |
| `overall.per_class.No conflict.precision` | `0.531` |
| `overall.per_class.No conflict.recall` | `0.944` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `9` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `59` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `27` |
| `overall.confusion_matrix.irrelevant.partially supports` | `4` |
| `overall.confusion_matrix.irrelevant.supports` | `9` |
| `overall.confusion_matrix.partially supports.irrelevant` | `23` |
| `overall.confusion_matrix.partially supports.partially supports` | `20` |
| `overall.confusion_matrix.partially supports.supports` | `88` |
| `overall.confusion_matrix.supports.irrelevant` | `12` |
| `overall.confusion_matrix.supports.partially supports` | `6` |
| `overall.confusion_matrix.supports.supports` | `143` |
| `overall.macro_f1` | `0.497` |
| `overall.per_class.irrelevant.f1` | `0.5294` |
| `overall.per_class.irrelevant.precision` | `0.4355` |
| `overall.per_class.irrelevant.recall` | `0.675` |
| `overall.per_class.partially supports.f1` | `0.2484` |
| `overall.per_class.partially supports.precision` | `0.6667` |
| `overall.per_class.partially supports.recall` | `0.1527` |
| `overall.per_class.supports.f1` | `0.7132` |
| `overall.per_class.supports.precision` | `0.5958` |
| `overall.per_class.supports.recall` | `0.8882` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `190` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `57.23` |
| `totals.total_doc_pairs_evaluated` | `332` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_refresh_e2e_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_refresh_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_refresh_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `63.27` |
| `abstain.false_negative_ids.count` | `3` |
| `abstain.false_positive_ids.count` | `15` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `27` |
| `citations.avg_citation_count` | `1.2245` |
| `citations.avg_sentence_coverage` | `0.199` |
| `citations.avg_unique_citations` | `1.2245` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.1884` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2485` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `19` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_refresh_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `27` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `11` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `11` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `15` |
| `abstain_gold.accuracy_pct` | `63.3` |
| `abstain_gold.correct` | `31` |
| `abstain_gold.false_abstain_ids.count` | `15` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `3` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.4432` |
| `citation_coverage.below_threshold_examples.count` | `17` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `22` |
| `citation_coverage.pass_count` | `5` |
| `citation_coverage.pass_rate_pct` | `22.7` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `3` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `11` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `19` |
| `label_f1.macro_f1` | `0.1819` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.2308` |
| `label_f1.per_class.Complementary information.precision` | `0.2727` |
| `label_f1.per_class.Complementary information.recall` | `0.2` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `label_f1.per_class.No conflict.f1` | `0.6786` |
| `label_f1.per_class.No conflict.precision` | `0.5135` |
| `label_f1.per_class.No conflict.recall` | `1.0` |
| `ok_all_checks` | `29` |
| `ok_ignoring_abstain_evidence_violation` | `31` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `63.3` |
| `ok_ignoring_abstain_support_violation` | `31` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `63.3` |
| `ok_rate_pct` | `59.2` |
| `problems.count` | `20` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_refresh_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `44.9` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `3` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `11` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `4` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `3` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `19` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `11` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `1` |
| `overall.distribution_pred.No conflict` | `37` |
| `overall.per_class.Complementary information.f1` | `0.231` |
| `overall.per_class.Complementary information.precision` | `0.273` |
| `overall.per_class.Complementary information.recall` | `0.2` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.679` |
| `overall.per_class.No conflict.precision` | `0.514` |
| `overall.per_class.No conflict.recall` | `1.0` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `6` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_refresh_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_refresh_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `52` |
| `overall.confusion_matrix.irrelevant.partially supports` | `1` |
| `overall.confusion_matrix.irrelevant.supports` | `3` |
| `overall.confusion_matrix.partially supports.irrelevant` | `84` |
| `overall.confusion_matrix.partially supports.partially supports` | `22` |
| `overall.confusion_matrix.partially supports.supports` | `52` |
| `overall.confusion_matrix.supports.irrelevant` | `40` |
| `overall.confusion_matrix.supports.partially supports` | `30` |
| `overall.confusion_matrix.supports.supports` | `107` |
| `overall.macro_f1` | `0.4294` |
| `overall.per_class.irrelevant.f1` | `0.4483` |
| `overall.per_class.irrelevant.precision` | `0.2955` |
| `overall.per_class.irrelevant.recall` | `0.9286` |
| `overall.per_class.partially supports.f1` | `0.2085` |
| `overall.per_class.partially supports.precision` | `0.4151` |
| `overall.per_class.partially supports.recall` | `0.1392` |
| `overall.per_class.supports.f1` | `0.6313` |
| `overall.per_class.supports.precision` | `0.6605` |
| `overall.per_class.supports.recall` | `0.6045` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `181` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `46.29` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen3_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen3_32b_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2915` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4698` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `48` |
| `trace_presence.think_count` | `48` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `48` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `48` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `5` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.no_json_array` | `48` |
| `error_counts.think_block_missing_or_misaligned` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen3_32b_stagewise_base_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen3_32b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen3_32b_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `67.35` |
| `abstain.false_negative_ids.count` | `12` |
| `abstain.false_positive_ids.count` | `4` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `7` |
| `citations.avg_citation_count` | `3.6939` |
| `citations.avg_sentence_coverage` | `0.1418` |
| `citations.avg_unique_citations` | `3.1224` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.1806` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.337` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `30` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `5` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `4` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `4` |
| `abstain_gold.accuracy_pct` | `63.3` |
| `abstain_gold.correct` | `31` |
| `abstain_gold.false_abstain_ids.count` | `4` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `14` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.1455` |
| `citation_coverage.below_threshold_examples.count` | `43` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `44` |
| `citation_coverage.pass_count` | `1` |
| `citation_coverage.pass_rate_pct` | `2.3` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.no_json_array` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen3_32b_stagewise_base_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/baselines/qwen3_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `57.14` |
| `abstain.false_negative_ids.count` | `7` |
| `abstain.false_positive_ids.count` | `14` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `22` |
| `citations.avg_citation_count` | `4.6122` |
| `citations.avg_sentence_coverage` | `0.0721` |
| `citations.avg_unique_citations` | `3.2245` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.1474` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2214` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `20` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `45` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `19` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `6` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `6` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `12` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `12` |
| `abstain_gold.accuracy_pct` | `63.3` |
| `abstain_gold.correct` | `31` |
| `abstain_gold.false_abstain_ids.count` | `11` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `7` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.1026` |
| `citation_coverage.below_threshold_examples.count` | `30` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `30` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen3_32b_stagewise_base_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

## `e2e` / `sft`

### `model_output_exports/val set/e2e/sft/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `95.92` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `17` |
| `citations.avg_citation_count` | `6.0204` |
| `citations.avg_sentence_coverage` | `0.5871` |
| `citations.avg_unique_citations` | `4.551` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3461` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5493` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `17` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `3` |
| `abstain_gold.accuracy_pct` | `95.9` |
| `abstain_gold.correct` | `47` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8906` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `32` |
| `citation_coverage.pass_count` | `29` |
| `citation_coverage.pass_rate_pct` | `90.6` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `10` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `8` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `9` |
| `label_f1.macro_f1` | `0.4707` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.5714` |
| `label_f1.per_class.Complementary information.precision` | `0.4762` |
| `label_f1.per_class.Complementary information.recall` | `0.7143` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.4444` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.4` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8235` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `label_f1.per_class.No conflict.f1` | `0.5143` |
| `label_f1.per_class.No conflict.precision` | `0.5625` |
| `label_f1.per_class.No conflict.recall` | `0.4737` |
| `ok_all_checks` | `45` |
| `ok_ignoring_abstain_evidence_violation` | `46` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `93.9` |
| `ok_ignoring_abstain_support_violation` | `46` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `93.9` |
| `ok_rate_pct` | `91.8` |
| `problems.count` | `4` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `21` |
| `overall.accuracy` | `58.33` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `10` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `4` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `8` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `9` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `21` |
| `overall.distribution_pred.Conflict due to outdated information` | `4` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `7` |
| `overall.distribution_pred.No conflict` | `16` |
| `overall.per_class.Complementary information.f1` | `0.571` |
| `overall.per_class.Complementary information.precision` | `0.476` |
| `overall.per_class.Complementary information.recall` | `0.714` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.444` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.5` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.4` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.824` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.514` |
| `overall.per_class.No conflict.precision` | `0.562` |
| `overall.per_class.No conflict.recall` | `0.474` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `48` |
| `top_confusions.count` | `6` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `34` |
| `overall.confusion_matrix.irrelevant.partially supports` | `17` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `6` |
| `overall.confusion_matrix.partially supports.partially supports` | `131` |
| `overall.confusion_matrix.partially supports.supports` | `12` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `39` |
| `overall.confusion_matrix.supports.supports` | `138` |
| `overall.macro_f1` | `0.7731` |
| `overall.per_class.irrelevant.f1` | `0.7083` |
| `overall.per_class.irrelevant.precision` | `0.85` |
| `overall.per_class.irrelevant.recall` | `0.6071` |
| `overall.per_class.partially supports.f1` | `0.7798` |
| `overall.per_class.partially supports.precision` | `0.7005` |
| `overall.per_class.partially supports.recall` | `0.8792` |
| `overall.per_class.supports.f1` | `0.8313` |
| `overall.per_class.supports.precision` | `0.8903` |
| `overall.per_class.supports.recall` | `0.7797` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `303` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `79.32` |
| `totals.total_doc_pairs_evaluated` | `382` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/llama31_8b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `95.92` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `17` |
| `citations.avg_citation_count` | `6.0408` |
| `citations.avg_sentence_coverage` | `0.5789` |
| `citations.avg_unique_citations` | `4.6122` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3354` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5399` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `17` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `3` |
| `abstain_gold.accuracy_pct` | `95.9` |
| `abstain_gold.correct` | `47` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8865` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `32` |
| `citation_coverage.pass_count` | `30` |
| `citation_coverage.pass_rate_pct` | `93.8` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `9` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `8` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `9` |
| `label_f1.macro_f1` | `0.4684` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5143` |
| `label_f1.per_class.Complementary information.precision` | `0.45` |
| `label_f1.per_class.Complementary information.recall` | `0.6` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.5455` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7368` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.7778` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `label_f1.per_class.No conflict.f1` | `0.5455` |
| `label_f1.per_class.No conflict.precision` | `0.6429` |
| `label_f1.per_class.No conflict.recall` | `0.4737` |
| `ok_all_checks` | `46` |
| `ok_ignoring_abstain_evidence_violation` | `47` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `95.9` |
| `ok_ignoring_abstain_support_violation` | `47` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `95.9` |
| `ok_rate_pct` | `93.9` |
| `problems.count` | `3` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `21` |
| `overall.accuracy` | `57.14` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `9` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `3` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `8` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `9` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `20` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `14` |
| `overall.per_class.Complementary information.f1` | `0.514` |
| `overall.per_class.Complementary information.precision` | `0.45` |
| `overall.per_class.Complementary information.recall` | `0.6` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.545` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.5` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.737` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.778` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.545` |
| `overall.per_class.No conflict.precision` | `0.643` |
| `overall.per_class.No conflict.recall` | `0.474` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `34` |
| `overall.confusion_matrix.irrelevant.partially supports` | `17` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `7` |
| `overall.confusion_matrix.partially supports.partially supports` | `136` |
| `overall.confusion_matrix.partially supports.supports` | `15` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `33` |
| `overall.confusion_matrix.supports.supports` | `144` |
| `overall.macro_f1` | `0.7788` |
| `overall.per_class.irrelevant.f1` | `0.701` |
| `overall.per_class.irrelevant.precision` | `0.8293` |
| `overall.per_class.irrelevant.recall` | `0.6071` |
| `overall.per_class.partially supports.f1` | `0.7907` |
| `overall.per_class.partially supports.precision` | `0.7312` |
| `overall.per_class.partially supports.recall` | `0.8608` |
| `overall.per_class.supports.f1` | `0.8446` |
| `overall.per_class.supports.precision` | `0.878` |
| `overall.per_class.supports.recall` | `0.8136` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `314` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `80.31` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `6.0408` |
| `citations.avg_sentence_coverage` | `0.584` |
| `citations.avg_unique_citations` | `4.6531` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3393` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5356` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8672` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `30` |
| `citation_coverage.pass_rate_pct` | `90.9` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `10` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `6` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `10` |
| `label_f1.macro_f1` | `0.4597` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.6061` |
| `label_f1.per_class.Complementary information.precision` | `0.5556` |
| `label_f1.per_class.Complementary information.recall` | `0.6667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.4` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.4` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.4` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7368` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.7778` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `label_f1.per_class.No conflict.f1` | `0.5556` |
| `label_f1.per_class.No conflict.precision` | `0.5882` |
| `label_f1.per_class.No conflict.recall` | `0.5263` |
| `ok_all_checks` | `47` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `95.9` |
| `problems.count` | `2` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `20` |
| `overall.accuracy` | `59.18` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `10` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `3` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `6` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `10` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `0.606` |
| `overall.per_class.Complementary information.precision` | `0.556` |
| `overall.per_class.Complementary information.recall` | `0.667` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.4` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.4` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.4` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.737` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.778` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.556` |
| `overall.per_class.No conflict.precision` | `0.588` |
| `overall.per_class.No conflict.recall` | `0.526` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `9` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `34` |
| `overall.confusion_matrix.irrelevant.partially supports` | `17` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `6` |
| `overall.confusion_matrix.partially supports.partially supports` | `143` |
| `overall.confusion_matrix.partially supports.supports` | `9` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `43` |
| `overall.confusion_matrix.supports.supports` | `134` |
| `overall.macro_f1` | `0.7751` |
| `overall.per_class.irrelevant.f1` | `0.7083` |
| `overall.per_class.irrelevant.precision` | `0.85` |
| `overall.per_class.irrelevant.recall` | `0.6071` |
| `overall.per_class.partially supports.f1` | `0.7922` |
| `overall.per_class.partially supports.precision` | `0.7044` |
| `overall.per_class.partially supports.recall` | `0.9051` |
| `overall.per_class.supports.f1` | `0.8246` |
| `overall.per_class.supports.precision` | `0.9054` |
| `overall.per_class.supports.recall` | `0.7571` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `311` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.54` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.6531` |
| `citations.avg_sentence_coverage` | `0.4796` |
| `citations.avg_unique_citations` | `4.4286` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2946` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4918` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.6912` |
| `citation_coverage.below_threshold_examples.count` | `15` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `19` |
| `citation_coverage.pass_rate_pct` | `55.9` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `7` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.4964` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5` |
| `label_f1.per_class.Complementary information.precision` | `0.5385` |
| `label_f1.per_class.Complementary information.recall` | `0.4667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.6667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `label_f1.per_class.No conflict.f1` | `0.6486` |
| `label_f1.per_class.No conflict.precision` | `0.6667` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `19` |
| `overall.accuracy` | `61.22` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `7` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `4` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `13` |
| `overall.distribution_pred.Conflict due to outdated information` | `10` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `0.5` |
| `overall.per_class.Complementary information.precision` | `0.538` |
| `overall.per_class.Complementary information.recall` | `0.467` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.667` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.5` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.667` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.649` |
| `overall.per_class.No conflict.precision` | `0.667` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `36` |
| `overall.confusion_matrix.irrelevant.partially supports` | `11` |
| `overall.confusion_matrix.irrelevant.supports` | `9` |
| `overall.confusion_matrix.partially supports.irrelevant` | `13` |
| `overall.confusion_matrix.partially supports.partially supports` | `122` |
| `overall.confusion_matrix.partially supports.supports` | `23` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `23` |
| `overall.confusion_matrix.supports.supports` | `154` |
| `overall.macro_f1` | `0.7704` |
| `overall.per_class.irrelevant.f1` | `0.6857` |
| `overall.per_class.irrelevant.precision` | `0.7347` |
| `overall.per_class.irrelevant.recall` | `0.6429` |
| `overall.per_class.partially supports.f1` | `0.7771` |
| `overall.per_class.partially supports.precision` | `0.7821` |
| `overall.per_class.partially supports.recall` | `0.7722` |
| `overall.per_class.supports.f1` | `0.8485` |
| `overall.per_class.supports.precision` | `0.828` |
| `overall.per_class.supports.recall` | `0.8701` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `312` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.8` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/mistral7b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `6.0` |
| `citations.avg_sentence_coverage` | `0.5148` |
| `citations.avg_unique_citations` | `4.3673` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2844` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4847` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.7419` |
| `citation_coverage.below_threshold_examples.count` | `14` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `20` |
| `citation_coverage.pass_rate_pct` | `58.8` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `7` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.5025` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.4828` |
| `label_f1.per_class.Complementary information.precision` | `0.5` |
| `label_f1.per_class.Complementary information.recall` | `0.4667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.6667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `label_f1.per_class.No conflict.f1` | `0.6486` |
| `label_f1.per_class.No conflict.precision` | `0.6667` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `19` |
| `overall.accuracy` | `61.22` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `7` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `5` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `14` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `0.483` |
| `overall.per_class.Complementary information.precision` | `0.5` |
| `overall.per_class.Complementary information.recall` | `0.467` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.667` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.649` |
| `overall.per_class.No conflict.precision` | `0.667` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `38` |
| `overall.confusion_matrix.irrelevant.partially supports` | `10` |
| `overall.confusion_matrix.irrelevant.supports` | `8` |
| `overall.confusion_matrix.partially supports.irrelevant` | `12` |
| `overall.confusion_matrix.partially supports.partially supports` | `123` |
| `overall.confusion_matrix.partially supports.supports` | `23` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `27` |
| `overall.confusion_matrix.supports.supports` | `150` |
| `overall.macro_f1` | `0.7762` |
| `overall.per_class.irrelevant.f1` | `0.717` |
| `overall.per_class.irrelevant.precision` | `0.76` |
| `overall.per_class.irrelevant.recall` | `0.6786` |
| `overall.per_class.partially supports.f1` | `0.7736` |
| `overall.per_class.partially supports.precision` | `0.7688` |
| `overall.per_class.partially supports.recall` | `0.7785` |
| `overall.per_class.supports.f1` | `0.838` |
| `overall.per_class.supports.precision` | `0.8287` |
| `overall.per_class.supports.recall` | `0.8475` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `311` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.54` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `6.4694` |
| `citations.avg_sentence_coverage` | `0.5043` |
| `citations.avg_unique_citations` | `5.0204` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3018` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4936` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.7268` |
| `citation_coverage.below_threshold_examples.count` | `13` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `21` |
| `citation_coverage.pass_rate_pct` | `61.8` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `8` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `11` |
| `label_f1.macro_f1` | `0.5016` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5333` |
| `label_f1.per_class.Complementary information.precision` | `0.5333` |
| `label_f1.per_class.Complementary information.recall` | `0.5333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.6316` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.6667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `label_f1.per_class.No conflict.f1` | `0.6286` |
| `label_f1.per_class.No conflict.precision` | `0.6875` |
| `label_f1.per_class.No conflict.recall` | `0.5789` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `19` |
| `overall.accuracy` | `61.22` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `8` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `4` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `11` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `16` |
| `overall.per_class.Complementary information.f1` | `0.533` |
| `overall.per_class.Complementary information.precision` | `0.533` |
| `overall.per_class.Complementary information.recall` | `0.533` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.632` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.667` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.629` |
| `overall.per_class.No conflict.precision` | `0.688` |
| `overall.per_class.No conflict.recall` | `0.579` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `35` |
| `overall.confusion_matrix.irrelevant.partially supports` | `13` |
| `overall.confusion_matrix.irrelevant.supports` | `8` |
| `overall.confusion_matrix.partially supports.irrelevant` | `11` |
| `overall.confusion_matrix.partially supports.partially supports` | `123` |
| `overall.confusion_matrix.partially supports.supports` | `24` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `27` |
| `overall.confusion_matrix.supports.supports` | `150` |
| `overall.macro_f1` | `0.7628` |
| `overall.per_class.irrelevant.f1` | `0.6863` |
| `overall.per_class.irrelevant.precision` | `0.7609` |
| `overall.per_class.irrelevant.recall` | `0.625` |
| `overall.per_class.partially supports.f1` | `0.7664` |
| `overall.per_class.partially supports.precision` | `0.7546` |
| `overall.per_class.partially supports.recall` | `0.7785` |
| `overall.per_class.supports.f1` | `0.8357` |
| `overall.per_class.supports.precision` | `0.8242` |
| `overall.per_class.supports.recall` | `0.8475` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `308` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `78.77` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.1633` |
| `citations.avg_sentence_coverage` | `0.6286` |
| `citations.avg_unique_citations` | `4.1429` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3465` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5344` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9059` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `13` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `6` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `11` |
| `label_f1.macro_f1` | `0.601` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.7222` |
| `label_f1.per_class.Complementary information.precision` | `0.619` |
| `label_f1.per_class.Complementary information.recall` | `0.8667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7273` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.6667` |
| `label_f1.per_class.No conflict.precision` | `0.7857` |
| `label_f1.per_class.No conflict.recall` | `0.5789` |
| `ok_all_checks` | `49` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `100.0` |
| `problems.count` | `0` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `13` |
| `overall.accuracy` | `73.47` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `13` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `2` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `6` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `11` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `21` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `14` |
| `overall.per_class.Complementary information.f1` | `0.722` |
| `overall.per_class.Complementary information.precision` | `0.619` |
| `overall.per_class.Complementary information.recall` | `0.867` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.727` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.667` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.8` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.667` |
| `overall.per_class.No conflict.precision` | `0.786` |
| `overall.per_class.No conflict.recall` | `0.579` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `5` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `39` |
| `overall.confusion_matrix.irrelevant.partially supports` | `17` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `4` |
| `overall.confusion_matrix.partially supports.partially supports` | `138` |
| `overall.confusion_matrix.partially supports.supports` | `16` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `37` |
| `overall.confusion_matrix.supports.supports` | `140` |
| `overall.macro_f1` | `0.8058` |
| `overall.per_class.irrelevant.f1` | `0.7879` |
| `overall.per_class.irrelevant.precision` | `0.907` |
| `overall.per_class.irrelevant.recall` | `0.6964` |
| `overall.per_class.partially supports.f1` | `0.7886` |
| `overall.per_class.partially supports.precision` | `0.7188` |
| `overall.per_class.partially supports.recall` | `0.8734` |
| `overall.per_class.supports.f1` | `0.8408` |
| `overall.per_class.supports.precision` | `0.8974` |
| `overall.per_class.supports.recall` | `0.791` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `317` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `81.07` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen25_32b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.2449` |
| `citations.avg_sentence_coverage` | `0.6252` |
| `citations.avg_unique_citations` | `4.0408` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3568` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5437` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.901` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `12` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `13` |
| `label_f1.macro_f1` | `0.587` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.75` |
| `label_f1.per_class.Complementary information.precision` | `0.7059` |
| `label_f1.per_class.Complementary information.recall` | `0.8` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8421` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7429` |
| `label_f1.per_class.No conflict.precision` | `0.7647` |
| `label_f1.per_class.No conflict.recall` | `0.7222` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `13` |
| `overall.accuracy` | `75.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `12` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `2` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `3` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `13` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `17` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `0.75` |
| `overall.per_class.Complementary information.precision` | `0.706` |
| `overall.per_class.Complementary information.recall` | `0.8` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.842` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.743` |
| `overall.per_class.No conflict.precision` | `0.765` |
| `overall.per_class.No conflict.recall` | `0.722` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `37` |
| `overall.confusion_matrix.irrelevant.partially supports` | `18` |
| `overall.confusion_matrix.irrelevant.supports` | `1` |
| `overall.confusion_matrix.partially supports.irrelevant` | `5` |
| `overall.confusion_matrix.partially supports.partially supports` | `139` |
| `overall.confusion_matrix.partially supports.supports` | `14` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `42` |
| `overall.confusion_matrix.supports.supports` | `135` |
| `overall.macro_f1` | `0.7865` |
| `overall.per_class.irrelevant.f1` | `0.7551` |
| `overall.per_class.irrelevant.precision` | `0.881` |
| `overall.per_class.irrelevant.recall` | `0.6607` |
| `overall.per_class.partially supports.f1` | `0.7787` |
| `overall.per_class.partially supports.precision` | `0.6985` |
| `overall.per_class.partially supports.recall` | `0.8797` |
| `overall.per_class.supports.f1` | `0.8257` |
| `overall.per_class.supports.precision` | `0.9` |
| `overall.per_class.supports.recall` | `0.7627` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `311` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.54` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.1633` |
| `citations.avg_sentence_coverage` | `0.6401` |
| `citations.avg_unique_citations` | `4.102` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3591` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5561` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9225` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `13` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `14` |
| `label_f1.macro_f1` | `0.6016` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.7879` |
| `label_f1.per_class.Complementary information.precision` | `0.7222` |
| `label_f1.per_class.Complementary information.recall` | `0.8667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8421` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7778` |
| `label_f1.per_class.No conflict.precision` | `0.8235` |
| `label_f1.per_class.No conflict.recall` | `0.7368` |
| `ok_all_checks` | `49` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `100.0` |
| `problems.count` | `0` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `11` |
| `overall.accuracy` | `77.55` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `13` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `1` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `3` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `14` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `0.788` |
| `overall.per_class.Complementary information.precision` | `0.722` |
| `overall.per_class.Complementary information.recall` | `0.867` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.842` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.778` |
| `overall.per_class.No conflict.precision` | `0.824` |
| `overall.per_class.No conflict.recall` | `0.737` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `6` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `35` |
| `overall.confusion_matrix.irrelevant.partially supports` | `14` |
| `overall.confusion_matrix.irrelevant.supports` | `7` |
| `overall.confusion_matrix.partially supports.irrelevant` | `3` |
| `overall.confusion_matrix.partially supports.partially supports` | `139` |
| `overall.confusion_matrix.partially supports.supports` | `16` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `36` |
| `overall.confusion_matrix.supports.supports` | `141` |
| `overall.macro_f1` | `0.7909` |
| `overall.per_class.irrelevant.f1` | `0.7447` |
| `overall.per_class.irrelevant.precision` | `0.9211` |
| `overall.per_class.irrelevant.recall` | `0.625` |
| `overall.per_class.partially supports.f1` | `0.8012` |
| `overall.per_class.partially supports.precision` | `0.7354` |
| `overall.per_class.partially supports.recall` | `0.8797` |
| `overall.per_class.supports.f1` | `0.827` |
| `overall.per_class.supports.precision` | `0.8598` |
| `overall.per_class.supports.recall` | `0.7966` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `315` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `80.56` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `1` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `14` |
| `citations.avg_citation_count` | `5.3469` |
| `citations.avg_sentence_coverage` | `0.6255` |
| `citations.avg_unique_citations` | `4.2857` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3249` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5209` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `48` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `14` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `12` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `12` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `1` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9015` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `7` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `7` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `16` |
| `label_f1.macro_f1` | `0.5887` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.5833` |
| `label_f1.per_class.Complementary information.precision` | `0.7778` |
| `label_f1.per_class.Complementary information.recall` | `0.4667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7273` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7442` |
| `label_f1.per_class.No conflict.precision` | `0.64` |
| `label_f1.per_class.No conflict.recall` | `0.8889` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `14` |
| `overall.accuracy` | `72.92` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `7` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `7` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `1` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `16` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `9` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `25` |
| `overall.per_class.Complementary information.f1` | `0.583` |
| `overall.per_class.Complementary information.precision` | `0.778` |
| `overall.per_class.Complementary information.recall` | `0.467` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.727` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.667` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.8` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.744` |
| `overall.per_class.No conflict.precision` | `0.64` |
| `overall.per_class.No conflict.recall` | `0.889` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `37` |
| `overall.confusion_matrix.irrelevant.partially supports` | `8` |
| `overall.confusion_matrix.irrelevant.supports` | `7` |
| `overall.confusion_matrix.partially supports.irrelevant` | `13` |
| `overall.confusion_matrix.partially supports.partially supports` | `122` |
| `overall.confusion_matrix.partially supports.supports` | `22` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `39` |
| `overall.confusion_matrix.supports.supports` | `138` |
| `overall.macro_f1` | `0.7588` |
| `overall.per_class.irrelevant.f1` | `0.7255` |
| `overall.per_class.irrelevant.precision` | `0.74` |
| `overall.per_class.irrelevant.recall` | `0.7115` |
| `overall.per_class.partially supports.f1` | `0.7485` |
| `overall.per_class.partially supports.precision` | `0.7219` |
| `overall.per_class.partially supports.recall` | `0.7771` |
| `overall.per_class.supports.f1` | `0.8023` |
| `overall.per_class.supports.precision` | `0.8263` |
| `overall.per_class.supports.recall` | `0.7797` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `297` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `76.94` |
| `totals.total_doc_pairs_evaluated` | `386` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen25_7b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.3673` |
| `citations.avg_sentence_coverage` | `0.5927` |
| `citations.avg_unique_citations` | `4.3061` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3099` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.512` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8542` |
| `citation_coverage.below_threshold_examples.count` | `5` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `29` |
| `citation_coverage.pass_rate_pct` | `85.3` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `7` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `6` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `16` |
| `label_f1.macro_f1` | `0.5719` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.56` |
| `label_f1.per_class.Complementary information.precision` | `0.7` |
| `label_f1.per_class.Complementary information.recall` | `0.4667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5714` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7442` |
| `label_f1.per_class.No conflict.precision` | `0.6667` |
| `label_f1.per_class.No conflict.recall` | `0.8421` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `14` |
| `overall.accuracy` | `71.43` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `7` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `6` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `2` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `16` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `10` |
| `overall.distribution_pred.Conflict due to outdated information` | `7` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `24` |
| `overall.per_class.Complementary information.f1` | `0.56` |
| `overall.per_class.Complementary information.precision` | `0.7` |
| `overall.per_class.Complementary information.recall` | `0.467` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.667` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.571` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.8` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.744` |
| `overall.per_class.No conflict.precision` | `0.667` |
| `overall.per_class.No conflict.recall` | `0.842` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `2` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `39` |
| `overall.confusion_matrix.irrelevant.partially supports` | `9` |
| `overall.confusion_matrix.irrelevant.supports` | `7` |
| `overall.confusion_matrix.partially supports.irrelevant` | `14` |
| `overall.confusion_matrix.partially supports.partially supports` | `124` |
| `overall.confusion_matrix.partially supports.supports` | `19` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `40` |
| `overall.confusion_matrix.supports.supports` | `137` |
| `overall.macro_f1` | `0.7599` |
| `overall.per_class.irrelevant.f1` | `0.7222` |
| `overall.per_class.irrelevant.precision` | `0.7358` |
| `overall.per_class.irrelevant.recall` | `0.7091` |
| `overall.per_class.partially supports.f1` | `0.7515` |
| `overall.per_class.partially supports.precision` | `0.7168` |
| `overall.per_class.partially supports.recall` | `0.7898` |
| `overall.per_class.supports.f1` | `0.8059` |
| `overall.per_class.supports.precision` | `0.8405` |
| `overall.per_class.supports.recall` | `0.774` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `300` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `77.12` |
| `totals.total_doc_pairs_evaluated` | `389` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.2857` |
| `citations.avg_sentence_coverage` | `0.6119` |
| `citations.avg_unique_citations` | `4.3469` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3099` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5101` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8819` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `32` |
| `citation_coverage.pass_rate_pct` | `94.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `8` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `17` |
| `label_f1.macro_f1` | `0.6215` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.64` |
| `label_f1.per_class.Complementary information.precision` | `0.8` |
| `label_f1.per_class.Complementary information.recall` | `0.5333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7692` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.625` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.8095` |
| `label_f1.per_class.No conflict.precision` | `0.7391` |
| `label_f1.per_class.No conflict.recall` | `0.8947` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `11` |
| `overall.accuracy` | `77.55` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `8` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `5` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `1` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `17` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `10` |
| `overall.distribution_pred.Conflict due to outdated information` | `8` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `23` |
| `overall.per_class.Complementary information.f1` | `0.64` |
| `overall.per_class.Complementary information.precision` | `0.8` |
| `overall.per_class.Complementary information.recall` | `0.533` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.769` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.625` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.81` |
| `overall.per_class.No conflict.precision` | `0.739` |
| `overall.per_class.No conflict.recall` | `0.895` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `6` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `2` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `41` |
| `overall.confusion_matrix.irrelevant.partially supports` | `9` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `11` |
| `overall.confusion_matrix.partially supports.partially supports` | `127` |
| `overall.confusion_matrix.partially supports.supports` | `19` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `36` |
| `overall.confusion_matrix.supports.supports` | `141` |
| `overall.macro_f1` | `0.7877` |
| `overall.per_class.irrelevant.f1` | `0.7664` |
| `overall.per_class.irrelevant.precision` | `0.7885` |
| `overall.per_class.irrelevant.recall` | `0.7455` |
| `overall.per_class.partially supports.f1` | `0.772` |
| `overall.per_class.partially supports.precision` | `0.7384` |
| `overall.per_class.partially supports.recall` | `0.8089` |
| `overall.per_class.supports.f1` | `0.8246` |
| `overall.per_class.supports.precision` | `0.8545` |
| `overall.per_class.supports.recall` | `0.7966` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `309` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.43` |
| `totals.total_doc_pairs_evaluated` | `389` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen3_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.9796` |
| `citations.avg_sentence_coverage` | `0.633` |
| `citations.avg_unique_citations` | `4.6531` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3597` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5593` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9123` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `32` |
| `citation_coverage.pass_rate_pct` | `94.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `10` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `3` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `13` |
| `label_f1.macro_f1` | `0.5298` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.6061` |
| `label_f1.per_class.Complementary information.precision` | `0.5556` |
| `label_f1.per_class.Complementary information.recall` | `0.6667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.7` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `label_f1.per_class.No conflict.f1` | `0.7429` |
| `label_f1.per_class.No conflict.precision` | `0.8125` |
| `label_f1.per_class.No conflict.recall` | `0.6842` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `16` |
| `overall.accuracy` | `67.35` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `10` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `3` |
| `overall.confusion_matrix.Complementary information.No conflict` | `2` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `13` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `16` |
| `overall.per_class.Complementary information.f1` | `0.606` |
| `overall.per_class.Complementary information.precision` | `0.556` |
| `overall.per_class.Complementary information.recall` | `0.667` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.743` |
| `overall.per_class.No conflict.precision` | `0.812` |
| `overall.per_class.No conflict.recall` | `0.684` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `43` |
| `overall.confusion_matrix.irrelevant.partially supports` | `13` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `6` |
| `overall.confusion_matrix.partially supports.partially supports` | `131` |
| `overall.confusion_matrix.partially supports.supports` | `21` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `29` |
| `overall.confusion_matrix.supports.supports` | `148` |
| `overall.macro_f1` | `0.822` |
| `overall.per_class.irrelevant.f1` | `0.819` |
| `overall.per_class.irrelevant.precision` | `0.8776` |
| `overall.per_class.irrelevant.recall` | `0.7679` |
| `overall.per_class.partially supports.f1` | `0.7915` |
| `overall.per_class.partially supports.precision` | `0.7572` |
| `overall.per_class.partially supports.recall` | `0.8291` |
| `overall.per_class.supports.f1` | `0.8555` |
| `overall.per_class.supports.precision` | `0.8757` |
| `overall.per_class.supports.recall` | `0.8362` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `322` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `82.35` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen3_32b/runtime_helper_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `5.5102` |
| `citations.avg_sentence_coverage` | `0.6095` |
| `citations.avg_unique_citations` | `4.3673` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3643` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5627` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9051` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `93.9` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `11` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `5` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `11` |
| `label_f1.macro_f1` | `0.5645` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.6667` |
| `label_f1.per_class.Complementary information.precision` | `0.6111` |
| `label_f1.per_class.Complementary information.recall` | `0.7333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7273` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7619` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.7273` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.6667` |
| `label_f1.per_class.No conflict.precision` | `0.7857` |
| `label_f1.per_class.No conflict.recall` | `0.5789` |
| `ok_all_checks` | `47` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `95.9` |
| `problems.count` | `2` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `15` |
| `overall.accuracy` | `69.39` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `11` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `2` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `5` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `11` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `11` |
| `overall.distribution_pred.No conflict` | `14` |
| `overall.per_class.Complementary information.f1` | `0.667` |
| `overall.per_class.Complementary information.precision` | `0.611` |
| `overall.per_class.Complementary information.recall` | `0.733` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.727` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.667` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.8` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.762` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.727` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.667` |
| `overall.per_class.No conflict.precision` | `0.786` |
| `overall.per_class.No conflict.recall` | `0.579` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `37` |
| `overall.confusion_matrix.irrelevant.partially supports` | `14` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `4` |
| `overall.confusion_matrix.partially supports.partially supports` | `134` |
| `overall.confusion_matrix.partially supports.supports` | `20` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `32` |
| `overall.confusion_matrix.supports.supports` | `145` |
| `overall.macro_f1` | `0.7972` |
| `overall.per_class.irrelevant.f1` | `0.7629` |
| `overall.per_class.irrelevant.precision` | `0.9024` |
| `overall.per_class.irrelevant.recall` | `0.6607` |
| `overall.per_class.partially supports.f1` | `0.7929` |
| `overall.per_class.partially supports.precision` | `0.7444` |
| `overall.per_class.partially supports.recall` | `0.8481` |
| `overall.per_class.supports.f1` | `0.8357` |
| `overall.per_class.supports.precision` | `0.8529` |
| `overall.per_class.supports.recall` | `0.8192` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `316` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `80.82` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/e2e/sft/qwen3_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `6.2041` |
| `citations.avg_sentence_coverage` | `0.6476` |
| `citations.avg_unique_citations` | `4.6327` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3635` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5728` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9333` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `32` |
| `citation_coverage.pass_rate_pct` | `94.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `9` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `3` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `7` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `10` |
| `label_f1.macro_f1` | `0.4995` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5294` |
| `label_f1.per_class.Complementary information.precision` | `0.4737` |
| `label_f1.per_class.Complementary information.recall` | `0.6` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7619` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.7273` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.6061` |
| `label_f1.per_class.No conflict.precision` | `0.7143` |
| `label_f1.per_class.No conflict.recall` | `0.5263` |
| `ok_all_checks` | `49` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `100.0` |
| `problems.count` | `0` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `19` |
| `overall.accuracy` | `61.22` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `9` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `3` |
| `overall.confusion_matrix.Complementary information.No conflict` | `3` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `7` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `10` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `19` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `11` |
| `overall.distribution_pred.No conflict` | `14` |
| `overall.per_class.Complementary information.f1` | `0.529` |
| `overall.per_class.Complementary information.precision` | `0.474` |
| `overall.per_class.Complementary information.recall` | `0.6` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.762` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.727` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.606` |
| `overall.per_class.No conflict.precision` | `0.714` |
| `overall.per_class.No conflict.recall` | `0.526` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen3_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `39` |
| `overall.confusion_matrix.irrelevant.partially supports` | `12` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `6` |
| `overall.confusion_matrix.partially supports.partially supports` | `134` |
| `overall.confusion_matrix.partially supports.supports` | `18` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `32` |
| `overall.confusion_matrix.supports.supports` | `145` |
| `overall.macro_f1` | `0.8035` |
| `overall.per_class.irrelevant.f1` | `0.7723` |
| `overall.per_class.irrelevant.precision` | `0.8667` |
| `overall.per_class.irrelevant.recall` | `0.6964` |
| `overall.per_class.partially supports.f1` | `0.7976` |
| `overall.per_class.partially supports.precision` | `0.7528` |
| `overall.per_class.partially supports.recall` | `0.8481` |
| `overall.per_class.supports.f1` | `0.8406` |
| `overall.per_class.supports.precision` | `0.8631` |
| `overall.per_class.supports.recall` | `0.8192` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `318` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `81.33` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

## `oracle_both` / `baselines`

### `model_output_exports/val set/oracle_both/baselines/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2023` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3266` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/baselines/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `57.14` |
| `abstain.false_negative_ids.count` | `8` |
| `abstain.false_positive_ids.count` | `13` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `20` |
| `citations.avg_citation_count` | `13.8571` |
| `citations.avg_sentence_coverage` | `0.1986` |
| `citations.avg_unique_citations` | `1.2245` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2279` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3117` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `21` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `6` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `6` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `9` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `9` |
| `abstain_gold.accuracy_pct` | `61.2` |
| `abstain_gold.correct` | `30` |
| `abstain_gold.false_abstain_ids.count` | `10` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `9` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.2949` |
| `citation_coverage.below_threshold_examples.count` | `30` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `3` |
| `citation_coverage.pass_rate_pct` | `9.1` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.no_json_array` | `48` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `1` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `4` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.6667` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `1.0` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `5` |
| `totals.examples_with_any_eval` | `1` |
| `totals.micro_accuracy_doc_level` | `100.0` |
| `totals.total_doc_pairs_evaluated` | `5` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/baselines/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `73.47` |
| `abstain.false_negative_ids.count` | `13` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `2` |
| `citations.avg_citation_count` | `0.1224` |
| `citations.avg_sentence_coverage` | `0.0175` |
| `citations.avg_unique_citations` | `0.102` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2392` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.368` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/baselines/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `67.35` |
| `abstain.false_negative_ids.count` | `13` |
| `abstain.false_positive_ids.count` | `3` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `5` |
| `citations.avg_citation_count` | `2.3878` |
| `citations.avg_sentence_coverage` | `0.3863` |
| `citations.avg_unique_citations` | `2.3469` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2269` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3384` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `31` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `4` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `3` |
| `abstain_gold.accuracy_pct` | `65.3` |
| `abstain_gold.correct` | `32` |
| `abstain_gold.false_abstain_ids.count` | `3` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `14` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.3686` |
| `citation_coverage.below_threshold_examples.count` | `34` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `45` |
| `citation_coverage.pass_count` | `11` |
| `citation_coverage.pass_rate_pct` | `24.4` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `6` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `7` |
| `label_f1.macro_f1` | `0.8` |
| `label_f1.pairs_evaluated` | `24` |
| `label_f1.per_class.Complementary information.f1` | `1.0` |
| `label_f1.per_class.Complementary information.precision` | `1.0` |
| `label_f1.per_class.Complementary information.recall` | `1.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `1.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `1.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `1.0` |
| `label_f1.per_class.No conflict.precision` | `1.0` |
| `label_f1.per_class.No conflict.recall` | `1.0` |
| `ok_all_checks` | `16` |
| `ok_ignoring_abstain_evidence_violation` | `16` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `32.7` |
| `ok_ignoring_abstain_support_violation` | `16` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `32.7` |
| `ok_rate_pct` | `32.7` |
| `problems.count` | `33` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `100.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `6` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `7` |
| `overall.distribution_actual.Complementary information` | `6` |
| `overall.distribution_actual.Conflict due to outdated information` | `3` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_actual.No conflict` | `7` |
| `overall.distribution_pred.Complementary information` | `6` |
| `overall.distribution_pred.Conflict due to outdated information` | `3` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `7` |
| `overall.per_class.Complementary information.f1` | `1.0` |
| `overall.per_class.Complementary information.precision` | `1.0` |
| `overall.per_class.Complementary information.recall` | `1.0` |
| `overall.per_class.Complementary information.support` | `6` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `1.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `1.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `3` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `8` |
| `overall.per_class.No conflict.f1` | `1.0` |
| `overall.per_class.No conflict.precision` | `1.0` |
| `overall.per_class.No conflict.recall` | `1.0` |
| `overall.per_class.No conflict.support` | `7` |
| `overall.support` | `24` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `33` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `31` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `1` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `147` |
| `overall.confusion_matrix.partially supports.supports` | `3` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `175` |
| `overall.macro_f1` | `0.9855` |
| `overall.per_class.irrelevant.f1` | `0.9841` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `0.9688` |
| `overall.per_class.partially supports.f1` | `0.9866` |
| `overall.per_class.partially supports.precision` | `0.9932` |
| `overall.per_class.partially supports.recall` | `0.98` |
| `overall.per_class.supports.f1` | `0.9859` |
| `overall.per_class.supports.precision` | `0.9777` |
| `overall.per_class.supports.recall` | `0.9943` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `353` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `98.6` |
| `totals.total_doc_pairs_evaluated` | `358` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/baselines/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2739` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4267` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/baselines/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `75.51` |
| `abstain.false_negative_ids.count` | `10` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `7` |
| `citations.avg_citation_count` | `6.0204` |
| `citations.avg_sentence_coverage` | `0.7121` |
| `citations.avg_unique_citations` | `4.4898` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2721` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3763` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `6` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `73.5` |
| `abstain_gold.correct` | `36` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `11` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8115` |
| `citation_coverage.below_threshold_examples.count` | `17` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `43` |
| `citation_coverage.pass_count` | `26` |
| `citation_coverage.pass_rate_pct` | `60.5` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `0` |
| `label_f1.macro_f1` | `0.2` |
| `label_f1.pairs_evaluated` | `1` |
| `label_f1.per_class.Complementary information.f1` | `0.0` |
| `label_f1.per_class.Complementary information.precision` | `0.0` |
| `label_f1.per_class.Complementary information.recall` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.0` |
| `label_f1.per_class.No conflict.precision` | `0.0` |
| `label_f1.per_class.No conflict.recall` | `0.0` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `100.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `1` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `1` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `1` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `1` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `157` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `177` |
| `overall.macro_f1` | `1.0` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `1.0` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `1.0` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `1.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `390` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `100.0` |
| `totals.total_doc_pairs_evaluated` | `390` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/baselines/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `71.43` |
| `abstain.false_negative_ids.count` | `14` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `1` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2621` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4253` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/baselines/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `67.35` |
| `abstain.false_negative_ids.count` | `1` |
| `abstain.false_positive_ids.count` | `15` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `29` |
| `citations.avg_citation_count` | `1.6735` |
| `citations.avg_sentence_coverage` | `0.2092` |
| `citations.avg_unique_citations` | `1.5918` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2104` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2857` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `19` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `29` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `15` |
| `abstain_gold.accuracy_pct` | `67.3` |
| `abstain_gold.correct` | `33` |
| `abstain_gold.false_abstain_ids.count` | `15` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `1` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.5125` |
| `citation_coverage.below_threshold_examples.count` | `16` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `20` |
| `citation_coverage.pass_count` | `4` |
| `citation_coverage.pass_rate_pct` | `20.0` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `0` |
| `label_f1.macro_f1` | `0.6` |
| `label_f1.pairs_evaluated` | `3` |
| `label_f1.per_class.Complementary information.f1` | `1.0` |
| `label_f1.per_class.Complementary information.precision` | `1.0` |
| `label_f1.per_class.Complementary information.recall` | `1.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `1.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `1.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.0` |
| `label_f1.per_class.No conflict.precision` | `0.0` |
| `label_f1.per_class.No conflict.recall` | `0.0` |
| `ok_all_checks` | `1` |
| `ok_ignoring_abstain_evidence_violation` | `1` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `2.0` |
| `ok_ignoring_abstain_support_violation` | `1` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `2.0` |
| `ok_rate_pct` | `2.0` |
| `problems.count` | `48` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `100.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.distribution_actual.Complementary information` | `1` |
| `overall.distribution_actual.Conflict due to outdated information` | `1` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `1` |
| `overall.distribution_pred.Complementary information` | `1` |
| `overall.distribution_pred.Conflict due to outdated information` | `1` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `1` |
| `overall.per_class.Complementary information.f1` | `1.0` |
| `overall.per_class.Complementary information.precision` | `1.0` |
| `overall.per_class.Complementary information.recall` | `1.0` |
| `overall.per_class.Complementary information.support` | `1` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `1.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `1.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `1` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `1` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `3` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `176` |
| `overall.macro_f1` | `0.998` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9968` |
| `overall.per_class.partially supports.precision` | `0.9937` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9972` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9944` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `390` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `99.74` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

## `oracle_both` / `sft`

### `model_output_exports/val set/oracle_both/sft/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `6.2857` |
| `citations.avg_sentence_coverage` | `0.5966` |
| `citations.avg_unique_citations` | `4.5102` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3411` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5578` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.848` |
| `citation_coverage.below_threshold_examples.count` | `5` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `29` |
| `citation_coverage.pass_rate_pct` | `85.3` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `12` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `5` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `11` |
| `label_f1.macro_f1` | `0.5804` |
| `label_f1.pairs_evaluated` | `46` |
| `label_f1.per_class.Complementary information.f1` | `0.75` |
| `label_f1.per_class.Complementary information.precision` | `0.6667` |
| `label_f1.per_class.Complementary information.recall` | `0.8571` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8421` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8889` |
| `label_f1.per_class.No conflict.f1` | `0.7097` |
| `label_f1.per_class.No conflict.precision` | `0.8462` |
| `label_f1.per_class.No conflict.recall` | `0.6111` |
| `ok_all_checks` | `45` |
| `ok_ignoring_abstain_evidence_violation` | `46` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `93.9` |
| `ok_ignoring_abstain_support_violation` | `46` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `93.9` |
| `ok_rate_pct` | `91.8` |
| `problems.count` | `4` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `15` |
| `overall.accuracy` | `73.91` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `12` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `5` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `11` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `13` |
| `overall.per_class.Complementary information.f1` | `0.75` |
| `overall.per_class.Complementary information.precision` | `0.667` |
| `overall.per_class.Complementary information.recall` | `0.857` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.842` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `9` |
| `overall.per_class.No conflict.f1` | `0.71` |
| `overall.per_class.No conflict.precision` | `0.846` |
| `overall.per_class.No conflict.recall` | `0.611` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `46` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.json_array_parse_error: Expecting value: line 1 column 2 (char 1)` | `1` |
| `error_counts.think_block_not_unique` | `2` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `51` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `154` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `4` |
| `overall.confusion_matrix.supports.supports` | `159` |
| `overall.macro_f1` | `0.9916` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9872` |
| `overall.per_class.partially supports.precision` | `0.9747` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9876` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9755` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `364` |
| `totals.examples_with_any_eval` | `46` |
| `totals.micro_accuracy_doc_level` | `98.91` |
| `totals.total_doc_pairs_evaluated` | `368` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/sft/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `6.0612` |
| `citations.avg_sentence_coverage` | `0.5993` |
| `citations.avg_unique_citations` | `4.6939` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3441` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5592` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8508` |
| `citation_coverage.below_threshold_examples.count` | `4` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `29` |
| `citation_coverage.pass_rate_pct` | `87.9` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `12` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `9` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `5` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `10` |
| `label_f1.macro_f1` | `0.5794` |
| `label_f1.pairs_evaluated` | `46` |
| `label_f1.per_class.Complementary information.f1` | `0.75` |
| `label_f1.per_class.Complementary information.precision` | `0.6667` |
| `label_f1.per_class.Complementary information.recall` | `0.8571` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8571` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8182` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.9` |
| `label_f1.per_class.No conflict.f1` | `0.6897` |
| `label_f1.per_class.No conflict.precision` | `0.8333` |
| `label_f1.per_class.No conflict.recall` | `0.5882` |
| `ok_all_checks` | `44` |
| `ok_ignoring_abstain_evidence_violation` | `45` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `91.8` |
| `ok_ignoring_abstain_support_violation` | `45` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `91.8` |
| `ok_rate_pct` | `89.8` |
| `problems.count` | `5` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `15` |
| `overall.accuracy` | `73.91` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `12` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `9` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `5` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `10` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `17` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `11` |
| `overall.distribution_pred.No conflict` | `12` |
| `overall.per_class.Complementary information.f1` | `0.75` |
| `overall.per_class.Complementary information.precision` | `0.667` |
| `overall.per_class.Complementary information.recall` | `0.857` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.857` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.818` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.9` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.69` |
| `overall.per_class.No conflict.precision` | `0.833` |
| `overall.per_class.No conflict.recall` | `0.588` |
| `overall.per_class.No conflict.support` | `17` |
| `overall.support` | `46` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `2` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `153` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `2` |
| `overall.confusion_matrix.supports.supports` | `161` |
| `overall.macro_f1` | `0.9958` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9935` |
| `overall.per_class.partially supports.precision` | `0.9871` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9938` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9877` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `370` |
| `totals.examples_with_any_eval` | `47` |
| `totals.micro_accuracy_doc_level` | `99.46` |
| `totals.total_doc_pairs_evaluated` | `372` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/sft/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `95.92` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `17` |
| `citations.avg_citation_count` | `5.9388` |
| `citations.avg_sentence_coverage` | `0.4544` |
| `citations.avg_unique_citations` | `4.4898` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3164` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.513` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `17` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `3` |
| `abstain_gold.accuracy_pct` | `95.9` |
| `abstain_gold.correct` | `47` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.6785` |
| `citation_coverage.below_threshold_examples.count` | `17` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `32` |
| `citation_coverage.pass_count` | `15` |
| `citation_coverage.pass_rate_pct` | `46.9` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `8` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.528` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.5714` |
| `label_f1.per_class.Complementary information.precision` | `0.5714` |
| `label_f1.per_class.Complementary information.recall` | `0.5714` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7059` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8571` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `label_f1.per_class.No conflict.f1` | `0.6486` |
| `label_f1.per_class.No conflict.precision` | `0.6667` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `45` |
| `ok_ignoring_abstain_evidence_violation` | `46` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `93.9` |
| `ok_ignoring_abstain_support_violation` | `46` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `93.9` |
| `ok_rate_pct` | `91.8` |
| `problems.count` | `4` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `18` |
| `overall.accuracy` | `64.58` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `8` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `4` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `14` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `7` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `0.571` |
| `overall.per_class.Complementary information.precision` | `0.571` |
| `overall.per_class.Complementary information.recall` | `0.571` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.706` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.857` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.649` |
| `overall.per_class.No conflict.precision` | `0.667` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `48` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `53` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `154` |
| `overall.confusion_matrix.partially supports.supports` | `2` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `172` |
| `overall.macro_f1` | `0.9939` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9904` |
| `overall.per_class.partially supports.precision` | `0.9935` |
| `overall.per_class.partially supports.recall` | `0.9872` |
| `overall.per_class.supports.f1` | `0.9914` |
| `overall.per_class.supports.precision` | `0.9885` |
| `overall.per_class.supports.recall` | `0.9942` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `379` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `99.21` |
| `totals.total_doc_pairs_evaluated` | `382` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/sft/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `5.6939` |
| `citations.avg_sentence_coverage` | `0.4755` |
| `citations.avg_unique_citations` | `4.3673` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3161` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5082` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.6979` |
| `citation_coverage.below_threshold_examples.count` | `15` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `18` |
| `citation_coverage.pass_rate_pct` | `54.5` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `9` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.5533` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.6207` |
| `label_f1.per_class.Complementary information.precision` | `0.6429` |
| `label_f1.per_class.Complementary information.recall` | `0.6` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6667` |
| `label_f1.per_class.No conflict.f1` | `0.6316` |
| `label_f1.per_class.No conflict.precision` | `0.6316` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `46` |
| `ok_ignoring_abstain_evidence_violation` | `47` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `95.9` |
| `ok_ignoring_abstain_support_violation` | `47` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `95.9` |
| `ok_rate_pct` | `93.9` |
| `problems.count` | `3` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `17` |
| `overall.accuracy` | `66.67` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `9` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `5` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `14` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `6` |
| `overall.distribution_pred.No conflict` | `19` |
| `overall.per_class.Complementary information.f1` | `0.621` |
| `overall.per_class.Complementary information.precision` | `0.643` |
| `overall.per_class.Complementary information.recall` | `0.6` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.667` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `9` |
| `overall.per_class.No conflict.f1` | `0.632` |
| `overall.per_class.No conflict.precision` | `0.632` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `48` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `55` |
| `overall.confusion_matrix.irrelevant.partially supports` | `1` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `146` |
| `overall.confusion_matrix.partially supports.supports` | `2` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `174` |
| `overall.macro_f1` | `0.9896` |
| `overall.per_class.irrelevant.f1` | `0.991` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `0.9821` |
| `overall.per_class.partially supports.f1` | `0.9865` |
| `overall.per_class.partially supports.precision` | `0.9865` |
| `overall.per_class.partially supports.recall` | `0.9865` |
| `overall.per_class.supports.f1` | `0.9915` |
| `overall.per_class.supports.precision` | `0.9886` |
| `overall.per_class.supports.recall` | `0.9943` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `375` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `98.94` |
| `totals.total_doc_pairs_evaluated` | `379` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/sft/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.5102` |
| `citations.avg_sentence_coverage` | `0.636` |
| `citations.avg_unique_citations` | `4.2449` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3675` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5622` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9166` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `14` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `16` |
| `label_f1.macro_f1` | `0.7528` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.875` |
| `label_f1.per_class.Complementary information.precision` | `0.8235` |
| `label_f1.per_class.Complementary information.recall` | `0.9333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `1.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `1.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.8889` |
| `label_f1.per_class.No conflict.precision` | `0.9412` |
| `label_f1.per_class.No conflict.recall` | `0.8421` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `4` |
| `overall.accuracy` | `91.84` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `14` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `1` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `3` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `16` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `17` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `0.875` |
| `overall.per_class.Complementary information.precision` | `0.824` |
| `overall.per_class.Complementary information.recall` | `0.933` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `1.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `1.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.889` |
| `overall.per_class.No conflict.precision` | `0.941` |
| `overall.per_class.No conflict.recall` | `0.842` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `2` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `177` |
| `overall.macro_f1` | `1.0` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `1.0` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `1.0` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `1.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `391` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `100.0` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/sft/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.4694` |
| `citations.avg_sentence_coverage` | `0.637` |
| `citations.avg_unique_citations` | `4.3878` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3516` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5483` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9181` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `32` |
| `citation_coverage.pass_rate_pct` | `94.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `13` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `15` |
| `label_f1.macro_f1` | `0.7162` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.8387` |
| `label_f1.per_class.Complementary information.precision` | `0.8125` |
| `label_f1.per_class.Complementary information.recall` | `0.8667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.9091` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.8333` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.8333` |
| `label_f1.per_class.No conflict.precision` | `0.8824` |
| `label_f1.per_class.No conflict.recall` | `0.7895` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `6` |
| `overall.accuracy` | `87.76` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `13` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `2` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `3` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `15` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `16` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `0.839` |
| `overall.per_class.Complementary information.precision` | `0.812` |
| `overall.per_class.Complementary information.recall` | `0.867` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.909` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.833` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.833` |
| `overall.per_class.No conflict.precision` | `0.882` |
| `overall.per_class.No conflict.recall` | `0.789` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `3` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `177` |
| `overall.macro_f1` | `1.0` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `1.0` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `1.0` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `1.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `391` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `100.0` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/sft/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `95.92` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `17` |
| `citations.avg_citation_count` | `5.3469` |
| `citations.avg_sentence_coverage` | `0.5644` |
| `citations.avg_unique_citations` | `4.1837` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3391` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5404` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `48` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `17` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `3` |
| `abstain_gold.accuracy_pct` | `95.9` |
| `abstain_gold.correct` | `47` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8892` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `31` |
| `citation_coverage.pass_count` | `29` |
| `citation_coverage.pass_rate_pct` | `93.5` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `15` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `17` |
| `label_f1.macro_f1` | `0.7761` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `1.0` |
| `label_f1.per_class.Complementary information.precision` | `1.0` |
| `label_f1.per_class.Complementary information.recall` | `1.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.9091` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.8333` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.9714` |
| `label_f1.per_class.No conflict.precision` | `1.0` |
| `label_f1.per_class.No conflict.recall` | `0.9444` |
| `ok_all_checks` | `45` |
| `ok_ignoring_abstain_evidence_violation` | `46` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `93.9` |
| `ok_ignoring_abstain_support_violation` | `46` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `93.9` |
| `ok_rate_pct` | `91.8` |
| `problems.count` | `4` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `2` |
| `overall.accuracy` | `97.92` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `15` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `17` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `1.0` |
| `overall.per_class.Complementary information.precision` | `1.0` |
| `overall.per_class.Complementary information.recall` | `1.0` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.909` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.833` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.971` |
| `overall.per_class.No conflict.precision` | `1.0` |
| `overall.per_class.No conflict.recall` | `0.944` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `2` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `1` |
| `overall.confusion_matrix.partially supports.partially supports` | `156` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `3` |
| `overall.confusion_matrix.supports.supports` | `164` |
| `overall.macro_f1` | `0.9898` |
| `overall.per_class.irrelevant.f1` | `0.9912` |
| `overall.per_class.irrelevant.precision` | `0.9825` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9873` |
| `overall.per_class.partially supports.precision` | `0.9811` |
| `overall.per_class.partially supports.recall` | `0.9936` |
| `overall.per_class.supports.f1` | `0.9909` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.982` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `376` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `98.95` |
| `totals.total_doc_pairs_evaluated` | `380` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_both/sft/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `5.3469` |
| `citations.avg_sentence_coverage` | `0.599` |
| `citations.avg_unique_citations` | `4.2857` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.346` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5429` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8894` |
| `citation_coverage.below_threshold_examples.count` | `1` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `32` |
| `citation_coverage.pass_rate_pct` | `97.0` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `15` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `18` |
| `label_f1.macro_f1` | `0.7764` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `1.0` |
| `label_f1.per_class.Complementary information.precision` | `1.0` |
| `label_f1.per_class.Complementary information.recall` | `1.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.9091` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.8333` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.973` |
| `label_f1.per_class.No conflict.precision` | `1.0` |
| `label_f1.per_class.No conflict.recall` | `0.9474` |
| `ok_all_checks` | `47` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `95.9` |
| `problems.count` | `2` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `1` |
| `overall.accuracy` | `97.96` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `15` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `18` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `1.0` |
| `overall.per_class.Complementary information.precision` | `1.0` |
| `overall.per_class.Complementary information.recall` | `1.0` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.909` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.833` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.973` |
| `overall.per_class.No conflict.precision` | `1.0` |
| `overall.per_class.No conflict.recall` | `0.947` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `1` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_both_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `3` |
| `overall.confusion_matrix.supports.supports` | `174` |
| `overall.macro_f1` | `0.994` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9906` |
| `overall.per_class.partially supports.precision` | `0.9814` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9915` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9831` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `388` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `99.23` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

## `oracle_conflict_type` / `baselines`

### `model_output_exports/val set/oracle_conflict_type/baselines/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2093` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3099` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/baselines/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `73.47` |
| `abstain.false_negative_ids.count` | `12` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `4` |
| `citations.avg_citation_count` | `0.3265` |
| `citations.avg_sentence_coverage` | `0.0358` |
| `citations.avg_unique_citations` | `0.3061` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.163` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2516` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `47` |
| `trace_presence.think_count` | `28` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0618` |
| `citation_coverage.below_threshold_examples.count` | `28` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `28` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.extra_docs_in_pred` | `6` |
| `error_counts.think_block_missing_or_misaligned` | `21` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `15` |
| `overall.confusion_matrix.irrelevant.partially supports` | `5` |
| `overall.confusion_matrix.irrelevant.supports` | `4` |
| `overall.confusion_matrix.partially supports.irrelevant` | `24` |
| `overall.confusion_matrix.partially supports.partially supports` | `48` |
| `overall.confusion_matrix.partially supports.supports` | `10` |
| `overall.confusion_matrix.supports.irrelevant` | `14` |
| `overall.confusion_matrix.supports.partially supports` | `37` |
| `overall.confusion_matrix.supports.supports` | `72` |
| `overall.macro_f1` | `0.5456` |
| `overall.per_class.irrelevant.f1` | `0.3896` |
| `overall.per_class.irrelevant.precision` | `0.283` |
| `overall.per_class.irrelevant.recall` | `0.625` |
| `overall.per_class.partially supports.f1` | `0.5581` |
| `overall.per_class.partially supports.precision` | `0.5333` |
| `overall.per_class.partially supports.recall` | `0.5854` |
| `overall.per_class.supports.f1` | `0.689` |
| `overall.per_class.supports.precision` | `0.8372` |
| `overall.per_class.supports.recall` | `0.5854` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `135` |
| `totals.examples_with_any_eval` | `28` |
| `totals.micro_accuracy_doc_level` | `58.95` |
| `totals.total_doc_pairs_evaluated` | `229` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/baselines/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2479` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3917` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/baselines/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `75.51` |
| `abstain.false_negative_ids.count` | `9` |
| `abstain.false_positive_ids.count` | `3` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `9` |
| `citations.avg_citation_count` | `0.1633` |
| `citations.avg_sentence_coverage` | `0.0025` |
| `citations.avg_unique_citations` | `0.1633` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.1626` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2231` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `31` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `21` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `21` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `21` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `28` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `11` |
| `overall.confusion_matrix.irrelevant.partially supports` | `11` |
| `overall.confusion_matrix.irrelevant.supports` | `6` |
| `overall.confusion_matrix.partially supports.irrelevant` | `20` |
| `overall.confusion_matrix.partially supports.partially supports` | `19` |
| `overall.confusion_matrix.partially supports.supports` | `37` |
| `overall.confusion_matrix.supports.irrelevant` | `7` |
| `overall.confusion_matrix.supports.partially supports` | `15` |
| `overall.confusion_matrix.supports.supports` | `44` |
| `overall.macro_f1` | `0.4075` |
| `overall.per_class.irrelevant.f1` | `0.3333` |
| `overall.per_class.irrelevant.precision` | `0.2895` |
| `overall.per_class.irrelevant.recall` | `0.3929` |
| `overall.per_class.partially supports.f1` | `0.314` |
| `overall.per_class.partially supports.precision` | `0.4222` |
| `overall.per_class.partially supports.recall` | `0.25` |
| `overall.per_class.supports.f1` | `0.5752` |
| `overall.per_class.supports.precision` | `0.5057` |
| `overall.per_class.supports.recall` | `0.6667` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `74` |
| `totals.examples_with_any_eval` | `21` |
| `totals.micro_accuracy_doc_level` | `43.53` |
| `totals.total_doc_pairs_evaluated` | `170` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/baselines/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2698` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4328` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/baselines/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `46.94` |
| `abstain.false_negative_ids.count` | `2` |
| `abstain.false_positive_ids.count` | `24` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `37` |
| `citations.avg_citation_count` | `5.8367` |
| `citations.avg_sentence_coverage` | `0.7062` |
| `citations.avg_unique_citations` | `4.6735` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2198` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3698` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `10` |
| `lexical_overlap_non_abstain.scored_pairs` | `10` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `73.5` |
| `abstain_gold.correct` | `36` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `13` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.7363` |
| `citation_coverage.below_threshold_examples.count` | `22` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `47` |
| `citation_coverage.pass_count` | `25` |
| `citation_coverage.pass_rate_pct` | `53.2` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `35` |
| `overall.confusion_matrix.irrelevant.partially supports` | `21` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `21` |
| `overall.confusion_matrix.partially supports.partially supports` | `126` |
| `overall.confusion_matrix.partially supports.supports` | `11` |
| `overall.confusion_matrix.supports.irrelevant` | `7` |
| `overall.confusion_matrix.supports.partially supports` | `98` |
| `overall.confusion_matrix.supports.supports` | `72` |
| `overall.macro_f1` | `0.5891` |
| `overall.per_class.irrelevant.f1` | `0.5882` |
| `overall.per_class.irrelevant.precision` | `0.5556` |
| `overall.per_class.irrelevant.recall` | `0.625` |
| `overall.per_class.partially supports.f1` | `0.6253` |
| `overall.per_class.partially supports.precision` | `0.5143` |
| `overall.per_class.partially supports.recall` | `0.7975` |
| `overall.per_class.supports.f1` | `0.5538` |
| `overall.per_class.supports.precision` | `0.8675` |
| `overall.per_class.supports.recall` | `0.4068` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `233` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `59.59` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/baselines/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2651` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4195` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/baselines/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `53.06` |
| `abstain.false_negative_ids.count` | `6` |
| `abstain.false_positive_ids.count` | `17` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `26` |
| `citations.avg_citation_count` | `0.8571` |
| `citations.avg_sentence_coverage` | `0.2143` |
| `citations.avg_unique_citations` | `0.8571` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.197` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.2638` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `17` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `7` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `7` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `8` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `8` |
| `abstain_gold.accuracy_pct` | `73.5` |
| `abstain_gold.correct` | `36` |
| `abstain_gold.false_abstain_ids.count` | `7` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `6` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.3182` |
| `citation_coverage.below_threshold_examples.count` | `23` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `10` |
| `citation_coverage.pass_rate_pct` | `30.3` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `44` |
| `overall.confusion_matrix.irrelevant.partially supports` | `2` |
| `overall.confusion_matrix.irrelevant.supports` | `10` |
| `overall.confusion_matrix.partially supports.irrelevant` | `74` |
| `overall.confusion_matrix.partially supports.partially supports` | `30` |
| `overall.confusion_matrix.partially supports.supports` | `54` |
| `overall.confusion_matrix.supports.irrelevant` | `49` |
| `overall.confusion_matrix.supports.partially supports` | `18` |
| `overall.confusion_matrix.supports.supports` | `110` |
| `overall.macro_f1` | `0.4366` |
| `overall.per_class.irrelevant.f1` | `0.3946` |
| `overall.per_class.irrelevant.precision` | `0.2635` |
| `overall.per_class.irrelevant.recall` | `0.7857` |
| `overall.per_class.partially supports.f1` | `0.2885` |
| `overall.per_class.partially supports.precision` | `0.6` |
| `overall.per_class.partially supports.recall` | `0.1899` |
| `overall.per_class.supports.f1` | `0.6268` |
| `overall.per_class.supports.precision` | `0.6322` |
| `overall.per_class.supports.recall` | `0.6215` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `184` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `47.06` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

## `oracle_conflict_type` / `sft`

### `model_output_exports/val set/oracle_conflict_type/sft/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `5.8163` |
| `citations.avg_sentence_coverage` | `0.5633` |
| `citations.avg_unique_citations` | `4.3673` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3296` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5287` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8295` |
| `citation_coverage.below_threshold_examples.count` | `6` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `27` |
| `citation_coverage.pass_rate_pct` | `81.8` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `13` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `5` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `11` |
| `label_f1.macro_f1` | `0.5705` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.7429` |
| `label_f1.per_class.Complementary information.precision` | `0.65` |
| `label_f1.per_class.Complementary information.recall` | `0.8667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7097` |
| `label_f1.per_class.No conflict.precision` | `0.8462` |
| `label_f1.per_class.No conflict.recall` | `0.6111` |
| `ok_all_checks` | `46` |
| `ok_ignoring_abstain_evidence_violation` | `47` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `95.9` |
| `ok_ignoring_abstain_support_violation` | `47` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `95.9` |
| `ok_rate_pct` | `93.9` |
| `problems.count` | `3` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `14` |
| `overall.accuracy` | `72.92` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `13` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `5` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `11` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `20` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `13` |
| `overall.per_class.Complementary information.f1` | `0.743` |
| `overall.per_class.Complementary information.precision` | `0.65` |
| `overall.per_class.Complementary information.recall` | `0.867` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.71` |
| `overall.per_class.No conflict.precision` | `0.846` |
| `overall.per_class.No conflict.recall` | `0.611` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `6` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `34` |
| `overall.confusion_matrix.irrelevant.partially supports` | `17` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `7` |
| `overall.confusion_matrix.partially supports.partially supports` | `133` |
| `overall.confusion_matrix.partially supports.supports` | `13` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `35` |
| `overall.confusion_matrix.supports.supports` | `138` |
| `overall.macro_f1` | `0.7756` |
| `overall.per_class.irrelevant.f1` | `0.701` |
| `overall.per_class.irrelevant.precision` | `0.8293` |
| `overall.per_class.irrelevant.recall` | `0.6071` |
| `overall.per_class.partially supports.f1` | `0.787` |
| `overall.per_class.partially supports.precision` | `0.7189` |
| `overall.per_class.partially supports.recall` | `0.8693` |
| `overall.per_class.supports.f1` | `0.8389` |
| `overall.per_class.supports.precision` | `0.8846` |
| `overall.per_class.supports.recall` | `0.7977` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `305` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `79.84` |
| `totals.total_doc_pairs_evaluated` | `382` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/sft/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `95.92` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `17` |
| `citations.avg_citation_count` | `5.8776` |
| `citations.avg_sentence_coverage` | `0.5963` |
| `citations.avg_unique_citations` | `4.551` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3469` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.543` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `17` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `3` |
| `abstain_gold.accuracy_pct` | `95.9` |
| `abstain_gold.correct` | `47` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8888` |
| `citation_coverage.below_threshold_examples.count` | `4` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `32` |
| `citation_coverage.pass_count` | `28` |
| `citation_coverage.pass_rate_pct` | `87.5` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `12` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `5` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `11` |
| `label_f1.macro_f1` | `0.5675` |
| `label_f1.pairs_evaluated` | `47` |
| `label_f1.per_class.Complementary information.f1` | `0.75` |
| `label_f1.per_class.Complementary information.precision` | `0.6316` |
| `label_f1.per_class.Complementary information.recall` | `0.9231` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.6875` |
| `label_f1.per_class.No conflict.precision` | `0.8462` |
| `label_f1.per_class.No conflict.recall` | `0.5789` |
| `ok_all_checks` | `45` |
| `ok_ignoring_abstain_evidence_violation` | `45` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `91.8` |
| `ok_ignoring_abstain_support_violation` | `45` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `91.8` |
| `ok_rate_pct` | `91.8` |
| `problems.count` | `4` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `15` |
| `overall.accuracy` | `72.34` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `12` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `5` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `11` |
| `overall.distribution_actual.Complementary information` | `13` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `19` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `13` |
| `overall.per_class.Complementary information.f1` | `0.75` |
| `overall.per_class.Complementary information.precision` | `0.632` |
| `overall.per_class.Complementary information.recall` | `0.923` |
| `overall.per_class.Complementary information.support` | `13` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.688` |
| `overall.per_class.No conflict.precision` | `0.846` |
| `overall.per_class.No conflict.recall` | `0.579` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `47` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `2` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `34` |
| `overall.confusion_matrix.irrelevant.partially supports` | `17` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `7` |
| `overall.confusion_matrix.partially supports.partially supports` | `132` |
| `overall.confusion_matrix.partially supports.supports` | `8` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `32` |
| `overall.confusion_matrix.supports.supports` | `137` |
| `overall.macro_f1` | `0.7883` |
| `overall.per_class.irrelevant.f1` | `0.701` |
| `overall.per_class.irrelevant.precision` | `0.8293` |
| `overall.per_class.irrelevant.recall` | `0.6071` |
| `overall.per_class.partially supports.f1` | `0.8049` |
| `overall.per_class.partially supports.precision` | `0.7293` |
| `overall.per_class.partially supports.recall` | `0.898` |
| `overall.per_class.supports.f1` | `0.8589` |
| `overall.per_class.supports.precision` | `0.9133` |
| `overall.per_class.supports.recall` | `0.8107` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `303` |
| `totals.examples_with_any_eval` | `47` |
| `totals.micro_accuracy_doc_level` | `81.45` |
| `totals.total_doc_pairs_evaluated` | `372` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/sft/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.3878` |
| `citations.avg_sentence_coverage` | `0.4653` |
| `citations.avg_unique_citations` | `4.2245` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3006` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4851` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.6706` |
| `citation_coverage.below_threshold_examples.count` | `13` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `21` |
| `citation_coverage.pass_rate_pct` | `61.8` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `8` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.5204` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5333` |
| `label_f1.per_class.Complementary information.precision` | `0.5333` |
| `label_f1.per_class.Complementary information.recall` | `0.5333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7059` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8571` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `label_f1.per_class.No conflict.f1` | `0.6486` |
| `label_f1.per_class.No conflict.precision` | `0.6667` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `18` |
| `overall.accuracy` | `63.27` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `8` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `5` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `7` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `0.533` |
| `overall.per_class.Complementary information.precision` | `0.533` |
| `overall.per_class.Complementary information.recall` | `0.533` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.706` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.857` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.649` |
| `overall.per_class.No conflict.precision` | `0.667` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `34` |
| `overall.confusion_matrix.irrelevant.partially supports` | `13` |
| `overall.confusion_matrix.irrelevant.supports` | `9` |
| `overall.confusion_matrix.partially supports.irrelevant` | `10` |
| `overall.confusion_matrix.partially supports.partially supports` | `126` |
| `overall.confusion_matrix.partially supports.supports` | `22` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `27` |
| `overall.confusion_matrix.supports.supports` | `150` |
| `overall.macro_f1` | `0.7653` |
| `overall.per_class.irrelevant.f1` | `0.68` |
| `overall.per_class.irrelevant.precision` | `0.7727` |
| `overall.per_class.irrelevant.recall` | `0.6071` |
| `overall.per_class.partially supports.f1` | `0.7778` |
| `overall.per_class.partially supports.precision` | `0.759` |
| `overall.per_class.partially supports.recall` | `0.7975` |
| `overall.per_class.supports.f1` | `0.838` |
| `overall.per_class.supports.precision` | `0.8287` |
| `overall.per_class.supports.recall` | `0.8475` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `310` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.28` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/sft/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.8776` |
| `citations.avg_sentence_coverage` | `0.4711` |
| `citations.avg_unique_citations` | `4.4286` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3077` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5039` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.6789` |
| `citation_coverage.below_threshold_examples.count` | `15` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `19` |
| `citation_coverage.pass_rate_pct` | `55.9` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `8` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.5162` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5333` |
| `label_f1.per_class.Complementary information.precision` | `0.5333` |
| `label_f1.per_class.Complementary information.recall` | `0.5333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.6667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `label_f1.per_class.No conflict.f1` | `0.6667` |
| `label_f1.per_class.No conflict.precision` | `0.7059` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `18` |
| `overall.accuracy` | `63.27` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `8` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `4` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `0.533` |
| `overall.per_class.Complementary information.precision` | `0.533` |
| `overall.per_class.Complementary information.recall` | `0.533` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.667` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.75` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.667` |
| `overall.per_class.No conflict.precision` | `0.706` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `35` |
| `overall.confusion_matrix.irrelevant.partially supports` | `13` |
| `overall.confusion_matrix.irrelevant.supports` | `8` |
| `overall.confusion_matrix.partially supports.irrelevant` | `9` |
| `overall.confusion_matrix.partially supports.partially supports` | `125` |
| `overall.confusion_matrix.partially supports.supports` | `24` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `28` |
| `overall.confusion_matrix.supports.supports` | `149` |
| `overall.macro_f1` | `0.768` |
| `overall.per_class.irrelevant.f1` | `0.7` |
| `overall.per_class.irrelevant.precision` | `0.7955` |
| `overall.per_class.irrelevant.recall` | `0.625` |
| `overall.per_class.partially supports.f1` | `0.7716` |
| `overall.per_class.partially supports.precision` | `0.753` |
| `overall.per_class.partially supports.recall` | `0.7911` |
| `overall.per_class.supports.f1` | `0.8324` |
| `overall.per_class.supports.precision` | `0.8232` |
| `overall.per_class.supports.recall` | `0.8418` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `309` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.03` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/sft/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.2653` |
| `citations.avg_sentence_coverage` | `0.6415` |
| `citations.avg_unique_citations` | `4.2449` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3405` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5336` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9245` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `13` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `9` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `13` |
| `label_f1.macro_f1` | `0.6411` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.7879` |
| `label_f1.per_class.Complementary information.precision` | `0.7222` |
| `label_f1.per_class.Complementary information.recall` | `0.8667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7273` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.9474` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.9` |
| `label_f1.per_class.No conflict.f1` | `0.7429` |
| `label_f1.per_class.No conflict.precision` | `0.8125` |
| `label_f1.per_class.No conflict.recall` | `0.6842` |
| `ok_all_checks` | `49` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `100.0` |
| `problems.count` | `0` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `10` |
| `overall.accuracy` | `79.59` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `13` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `2` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `9` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `13` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `16` |
| `overall.per_class.Complementary information.f1` | `0.788` |
| `overall.per_class.Complementary information.precision` | `0.722` |
| `overall.per_class.Complementary information.recall` | `0.867` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.727` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.667` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.8` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.947` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.9` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.743` |
| `overall.per_class.No conflict.precision` | `0.812` |
| `overall.per_class.No conflict.recall` | `0.684` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `5` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `38` |
| `overall.confusion_matrix.irrelevant.partially supports` | `17` |
| `overall.confusion_matrix.irrelevant.supports` | `1` |
| `overall.confusion_matrix.partially supports.irrelevant` | `3` |
| `overall.confusion_matrix.partially supports.partially supports` | `138` |
| `overall.confusion_matrix.partially supports.supports` | `17` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `43` |
| `overall.confusion_matrix.supports.supports` | `134` |
| `overall.macro_f1` | `0.7911` |
| `overall.per_class.irrelevant.f1` | `0.7835` |
| `overall.per_class.irrelevant.precision` | `0.9268` |
| `overall.per_class.irrelevant.recall` | `0.6786` |
| `overall.per_class.partially supports.f1` | `0.7753` |
| `overall.per_class.partially supports.precision` | `0.697` |
| `overall.per_class.partially supports.recall` | `0.8734` |
| `overall.per_class.supports.f1` | `0.8146` |
| `overall.per_class.supports.precision` | `0.8816` |
| `overall.per_class.supports.recall` | `0.7571` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `310` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `79.28` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/sft/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.5714` |
| `citations.avg_sentence_coverage` | `0.652` |
| `citations.avg_unique_citations` | `4.3061` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3531` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5408` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9397` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `32` |
| `citation_coverage.pass_rate_pct` | `94.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `14` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `9` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `16` |
| `label_f1.macro_f1` | `0.7241` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.875` |
| `label_f1.per_class.Complementary information.precision` | `0.8235` |
| `label_f1.per_class.Complementary information.recall` | `0.9333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.9091` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.8333` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.9474` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.9` |
| `label_f1.per_class.No conflict.f1` | `0.8889` |
| `label_f1.per_class.No conflict.precision` | `0.9412` |
| `label_f1.per_class.No conflict.recall` | `0.8421` |
| `ok_all_checks` | `49` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `100.0` |
| `problems.count` | `0` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `5` |
| `overall.accuracy` | `89.8` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `14` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `1` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `9` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `2` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `16` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `17` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `17` |
| `overall.per_class.Complementary information.f1` | `0.875` |
| `overall.per_class.Complementary information.precision` | `0.824` |
| `overall.per_class.Complementary information.recall` | `0.933` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.909` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.833` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.947` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.9` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.889` |
| `overall.per_class.No conflict.precision` | `0.941` |
| `overall.per_class.No conflict.recall` | `0.842` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `33` |
| `overall.confusion_matrix.irrelevant.partially supports` | `16` |
| `overall.confusion_matrix.irrelevant.supports` | `7` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `141` |
| `overall.confusion_matrix.partially supports.supports` | `17` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `38` |
| `overall.confusion_matrix.supports.supports` | `139` |
| `overall.macro_f1` | `0.786` |
| `overall.per_class.irrelevant.f1` | `0.7416` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `0.5893` |
| `overall.per_class.partially supports.f1` | `0.7989` |
| `overall.per_class.partially supports.precision` | `0.7231` |
| `overall.per_class.partially supports.recall` | `0.8924` |
| `overall.per_class.supports.f1` | `0.8176` |
| `overall.per_class.supports.precision` | `0.8528` |
| `overall.per_class.supports.recall` | `0.7853` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `313` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `80.05` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/sft/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.102` |
| `citations.avg_sentence_coverage` | `0.5898` |
| `citations.avg_unique_citations` | `4.2041` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3255` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5224` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.85` |
| `citation_coverage.below_threshold_examples.count` | `6` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `28` |
| `citation_coverage.pass_rate_pct` | `82.4` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `15` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `18` |
| `label_f1.macro_f1` | `0.7764` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `1.0` |
| `label_f1.per_class.Complementary information.precision` | `1.0` |
| `label_f1.per_class.Complementary information.recall` | `1.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.9091` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.8333` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.973` |
| `label_f1.per_class.No conflict.precision` | `1.0` |
| `label_f1.per_class.No conflict.recall` | `0.9474` |
| `ok_all_checks` | `49` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `100.0` |
| `problems.count` | `0` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `1` |
| `overall.accuracy` | `97.96` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `15` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `18` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `1.0` |
| `overall.per_class.Complementary information.precision` | `1.0` |
| `overall.per_class.Complementary information.recall` | `1.0` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.909` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.833` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.973` |
| `overall.per_class.No conflict.precision` | `1.0` |
| `overall.per_class.No conflict.recall` | `0.947` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `1` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `43` |
| `overall.confusion_matrix.irrelevant.partially supports` | `8` |
| `overall.confusion_matrix.irrelevant.supports` | `5` |
| `overall.confusion_matrix.partially supports.irrelevant` | `13` |
| `overall.confusion_matrix.partially supports.partially supports` | `121` |
| `overall.confusion_matrix.partially supports.supports` | `24` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `38` |
| `overall.confusion_matrix.supports.supports` | `139` |
| `overall.macro_f1` | `0.7728` |
| `overall.per_class.irrelevant.f1` | `0.7679` |
| `overall.per_class.irrelevant.precision` | `0.7679` |
| `overall.per_class.irrelevant.recall` | `0.7679` |
| `overall.per_class.partially supports.f1` | `0.7446` |
| `overall.per_class.partially supports.precision` | `0.7246` |
| `overall.per_class.partially supports.recall` | `0.7658` |
| `overall.per_class.supports.f1` | `0.8058` |
| `overall.per_class.supports.precision` | `0.8274` |
| `overall.per_class.supports.recall` | `0.7853` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `303` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `77.49` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_conflict_type/sft/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.6122` |
| `citations.avg_sentence_coverage` | `0.6102` |
| `citations.avg_unique_citations` | `4.449` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3126` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5202` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8793` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `15` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `18` |
| `label_f1.macro_f1` | `0.7764` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `1.0` |
| `label_f1.per_class.Complementary information.precision` | `1.0` |
| `label_f1.per_class.Complementary information.recall` | `1.0` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.9091` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.8333` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `label_f1.per_class.No conflict.f1` | `0.973` |
| `label_f1.per_class.No conflict.precision` | `1.0` |
| `label_f1.per_class.No conflict.recall` | `0.9474` |
| `ok_all_checks` | `49` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `100.0` |
| `problems.count` | `0` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `1` |
| `overall.accuracy` | `97.96` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `15` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `10` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `18` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `1.0` |
| `overall.per_class.Complementary information.precision` | `1.0` |
| `overall.per_class.Complementary information.recall` | `1.0` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.909` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.833` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.973` |
| `overall.per_class.No conflict.precision` | `1.0` |
| `overall.per_class.No conflict.recall` | `0.947` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `1` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_conflict_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `40` |
| `overall.confusion_matrix.irrelevant.partially supports` | `9` |
| `overall.confusion_matrix.irrelevant.supports` | `7` |
| `overall.confusion_matrix.partially supports.irrelevant` | `12` |
| `overall.confusion_matrix.partially supports.partially supports` | `128` |
| `overall.confusion_matrix.partially supports.supports` | `18` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `42` |
| `overall.confusion_matrix.supports.supports` | `135` |
| `overall.macro_f1` | `0.7672` |
| `overall.per_class.irrelevant.f1` | `0.7407` |
| `overall.per_class.irrelevant.precision` | `0.7692` |
| `overall.per_class.irrelevant.recall` | `0.7143` |
| `overall.per_class.partially supports.f1` | `0.7596` |
| `overall.per_class.partially supports.precision` | `0.7151` |
| `overall.per_class.partially supports.recall` | `0.8101` |
| `overall.per_class.supports.f1` | `0.8012` |
| `overall.per_class.supports.precision` | `0.8438` |
| `overall.per_class.supports.recall` | `0.7627` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `303` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `77.49` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

## `oracle_per_doc_notes` / `baselines`

### `model_output_exports/val set/oracle_per_doc_notes/baselines/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.203` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3136` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/baselines/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_llama31_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `75.51` |
| `abstain.false_negative_ids.count` | `2` |
| `abstain.false_positive_ids.count` | `10` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `23` |
| `citations.avg_citation_count` | `1.3061` |
| `citations.avg_sentence_coverage` | `0.166` |
| `citations.avg_unique_citations` | `1.1633` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2242` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3146` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `24` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `22` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `8` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `8` |
| `abstain_gold.accuracy_pct` | `77.6` |
| `abstain_gold.correct` | `38` |
| `abstain_gold.false_abstain_ids.count` | `9` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `2` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.3012` |
| `citation_coverage.below_threshold_examples.count` | `22` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `27` |
| `citation_coverage.pass_count` | `5` |
| `citation_coverage.pass_rate_pct` | `18.5` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `2` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `11` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `4` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `5` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `18` |
| `label_f1.macro_f1` | `0.2177` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.1905` |
| `label_f1.per_class.Complementary information.precision` | `0.3333` |
| `label_f1.per_class.Complementary information.recall` | `0.1333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.2667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.4` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.2` |
| `label_f1.per_class.No conflict.f1` | `0.6316` |
| `label_f1.per_class.No conflict.precision` | `0.4737` |
| `label_f1.per_class.No conflict.recall` | `0.9474` |
| `ok_all_checks` | `17` |
| `ok_ignoring_abstain_evidence_violation` | `17` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `34.7` |
| `ok_ignoring_abstain_support_violation` | `17` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `34.7` |
| `ok_rate_pct` | `34.7` |
| `problems.count` | `32` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `44.9` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `2` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `11` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `4` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `5` |
| `overall.confusion_matrix.No conflict.Complementary information` | `1` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `18` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `5` |
| `overall.distribution_pred.No conflict` | `38` |
| `overall.per_class.Complementary information.f1` | `0.19` |
| `overall.per_class.Complementary information.precision` | `0.333` |
| `overall.per_class.Complementary information.recall` | `0.133` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.267` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.4` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.2` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.632` |
| `overall.per_class.No conflict.precision` | `0.474` |
| `overall.per_class.No conflict.recall` | `0.947` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_llama31_stagewise_base_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.no_json_array` | `28` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_llama31_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `41` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `55` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `1` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `33` |
| `overall.macro_f1` | `0.9832` |
| `overall.per_class.irrelevant.f1` | `0.988` |
| `overall.per_class.irrelevant.precision` | `0.9762` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.991` |
| `overall.per_class.partially supports.precision` | `0.9821` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9706` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9429` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `129` |
| `totals.examples_with_any_eval` | `21` |
| `totals.micro_accuracy_doc_level` | `98.47` |
| `totals.total_doc_pairs_evaluated` | `131` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/baselines/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `73.47` |
| `abstain.false_negative_ids.count` | `13` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `2` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2397` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.371` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/baselines/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_mistral7b_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `67.35` |
| `abstain.false_negative_ids.count` | `14` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `3` |
| `citations.avg_citation_count` | `2.5306` |
| `citations.avg_sentence_coverage` | `0.2613` |
| `citations.avg_unique_citations` | `2.5306` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2154` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3167` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `3` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `67.3` |
| `abstain_gold.correct` | `33` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `14` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.2784` |
| `citation_coverage.below_threshold_examples.count` | `42` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `46` |
| `citation_coverage.pass_count` | `4` |
| `citation_coverage.pass_rate_pct` | `8.7` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `5` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `7` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `5` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `14` |
| `label_f1.macro_f1` | `0.2802` |
| `label_f1.pairs_evaluated` | `46` |
| `label_f1.per_class.Complementary information.f1` | `0.4545` |
| `label_f1.per_class.Complementary information.precision` | `0.625` |
| `label_f1.per_class.Complementary information.recall` | `0.3571` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.375` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.4286` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.3333` |
| `label_f1.per_class.No conflict.f1` | `0.5714` |
| `label_f1.per_class.No conflict.precision` | `0.4516` |
| `label_f1.per_class.No conflict.recall` | `0.7778` |
| `ok_all_checks` | `26` |
| `ok_ignoring_abstain_evidence_violation` | `26` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `53.1` |
| `ok_ignoring_abstain_support_violation` | `26` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `53.1` |
| `ok_rate_pct` | `53.1` |
| `problems.count` | `23` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `47.83` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `5` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `7` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `5` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `5` |
| `overall.confusion_matrix.No conflict.Complementary information` | `2` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.No conflict.No conflict` | `14` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `8` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `7` |
| `overall.distribution_pred.No conflict` | `31` |
| `overall.per_class.Complementary information.f1` | `0.455` |
| `overall.per_class.Complementary information.precision` | `0.625` |
| `overall.per_class.Complementary information.recall` | `0.357` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.375` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.429` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.333` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `9` |
| `overall.per_class.No conflict.f1` | `0.571` |
| `overall.per_class.No conflict.precision` | `0.452` |
| `overall.per_class.No conflict.recall` | `0.778` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `46` |
| `top_confusions.count` | `10` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_mistral7b_stagewise_base_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.docs_missing_in_pred` | `50` |
| `error_counts.think_block_not_unique` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_mistral7b_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `16` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `1` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `140` |
| `overall.confusion_matrix.partially supports.supports` | `3` |
| `overall.confusion_matrix.supports.irrelevant` | `1` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `174` |
| `overall.macro_f1` | `0.97` |
| `overall.per_class.irrelevant.f1` | `0.9412` |
| `overall.per_class.irrelevant.precision` | `0.9412` |
| `overall.per_class.irrelevant.recall` | `0.9412` |
| `overall.per_class.partially supports.f1` | `0.9859` |
| `overall.per_class.partially supports.precision` | `0.9929` |
| `overall.per_class.partially supports.recall` | `0.979` |
| `overall.per_class.supports.f1` | `0.9831` |
| `overall.per_class.supports.precision` | `0.9775` |
| `overall.per_class.supports.recall` | `0.9886` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `330` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `98.21` |
| `totals.total_doc_pairs_evaluated` | `336` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/baselines/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `71.43` |
| `abstain.false_negative_ids.count` | `14` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `1` |
| `citations.avg_citation_count` | `0.0816` |
| `citations.avg_sentence_coverage` | `0.0136` |
| `citations.avg_unique_citations` | `0.0816` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2697` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4298` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/baselines/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_32b_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `77.55` |
| `abstain.false_negative_ids.count` | `9` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `8` |
| `citations.avg_citation_count` | `6.1224` |
| `citations.avg_sentence_coverage` | `0.7588` |
| `citations.avg_unique_citations` | `4.2041` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.273` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3838` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `6` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `4` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `4` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `77.6` |
| `abstain_gold.correct` | `38` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `10` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8337` |
| `citation_coverage.below_threshold_examples.count` | `12` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `43` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `72.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `8` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `6` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `14` |
| `label_f1.macro_f1` | `0.371` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5517` |
| `label_f1.per_class.Complementary information.precision` | `0.5714` |
| `label_f1.per_class.Complementary information.recall` | `0.5333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.6364` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.5833` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `label_f1.per_class.No conflict.f1` | `0.6667` |
| `label_f1.per_class.No conflict.precision` | `0.6087` |
| `label_f1.per_class.No conflict.recall` | `0.7368` |
| `ok_all_checks` | `38` |
| `ok_ignoring_abstain_evidence_violation` | `38` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `77.6` |
| `ok_ignoring_abstain_support_violation` | `38` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `77.6` |
| `ok_rate_pct` | `77.6` |
| `problems.count` | `11` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `20` |
| `overall.accuracy` | `59.18` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `8` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `6` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `3` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.No conflict.No conflict` | `14` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `14` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `12` |
| `overall.distribution_pred.No conflict` | `23` |
| `overall.per_class.Complementary information.f1` | `0.552` |
| `overall.per_class.Complementary information.precision` | `0.571` |
| `overall.per_class.Complementary information.recall` | `0.533` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.636` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.583` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.667` |
| `overall.per_class.No conflict.precision` | `0.609` |
| `overall.per_class.No conflict.recall` | `0.737` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_32b_stagewise_base_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_32b_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `177` |
| `overall.macro_f1` | `1.0` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `1.0` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `1.0` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `1.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `391` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `100.0` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/baselines/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `69.39` |
| `abstain.false_negative_ids.count` | `15` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `0` |
| `citations.avg_citation_count` | `0.0` |
| `citations.avg_sentence_coverage` | `0.0` |
| `citations.avg_unique_citations` | `0.0` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2588` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.4063` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `0` |
| `trace_presence.think_count` | `0` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `0` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `0` |
| `abstain_gold.accuracy_pct` | `69.4` |
| `abstain_gold.correct` | `34` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `15` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.0` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `0` |
| `citation_coverage.pass_count` | `0` |
| `citation_coverage.pass_rate_pct` | `0.0` |
| `citation_coverage.threshold` | `0.75` |
| `ok_all_checks` | `0` |
| `ok_ignoring_abstain_evidence_violation` | `0` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `0.0` |
| `ok_ignoring_abstain_support_violation` | `0` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `0.0` |
| `ok_rate_pct` | `0.0` |
| `problems.count` | `49` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `25` |
| `overall.accuracy` | `0.0` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `0` |
| `overall.per_class.Complementary information.f1` | `0.0` |
| `overall.per_class.Complementary information.precision` | `0.0` |
| `overall.per_class.Complementary information.recall` | `0.0` |
| `overall.per_class.Complementary information.support` | `0` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `0` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.0` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `0` |
| `overall.per_class.No conflict.f1` | `0.0` |
| `overall.per_class.No conflict.precision` | `0.0` |
| `overall.per_class.No conflict.recall` | `0.0` |
| `overall.per_class.No conflict.support` | `0` |
| `overall.support` | `0` |
| `top_confusions.count` | `4` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `49` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `0` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `0` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `0` |
| `overall.macro_f1` | `0.0` |
| `overall.per_class.irrelevant.f1` | `0.0` |
| `overall.per_class.irrelevant.precision` | `0.0` |
| `overall.per_class.irrelevant.recall` | `0.0` |
| `overall.per_class.partially supports.f1` | `0.0` |
| `overall.per_class.partially supports.precision` | `0.0` |
| `overall.per_class.partially supports.recall` | `0.0` |
| `overall.per_class.supports.f1` | `0.0` |
| `overall.per_class.supports.precision` | `0.0` |
| `overall.per_class.supports.recall` | `0.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `0` |
| `totals.examples_with_any_eval` | `0` |
| `totals.micro_accuracy_doc_level` | `0.0` |
| `totals.total_doc_pairs_evaluated` | `0` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/baselines/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/baseline_qwen25_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `73.47` |
| `abstain.false_negative_ids.count` | `4` |
| `abstain.false_positive_ids.count` | `9` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `20` |
| `citations.avg_citation_count` | `2.0816` |
| `citations.avg_sentence_coverage` | `0.2687` |
| `citations.avg_unique_citations` | `2.0816` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.2306` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.3075` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `25` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `19` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `10` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `10` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `8` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `8` |
| `abstain_gold.accuracy_pct` | `75.5` |
| `abstain_gold.correct` | `37` |
| `abstain_gold.false_abstain_ids.count` | `8` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `4` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.4389` |
| `citation_coverage.below_threshold_examples.count` | `26` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `30` |
| `citation_coverage.pass_count` | `4` |
| `citation_coverage.pass_rate_pct` | `13.3` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `9` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `6` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `15` |
| `label_f1.macro_f1` | `0.267` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.4865` |
| `label_f1.per_class.Complementary information.precision` | `0.4091` |
| `label_f1.per_class.Complementary information.recall` | `0.6` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.1667` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.5` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.1` |
| `label_f1.per_class.No conflict.f1` | `0.6818` |
| `label_f1.per_class.No conflict.precision` | `0.6` |
| `label_f1.per_class.No conflict.recall` | `0.7895` |
| `ok_all_checks` | `37` |
| `ok_ignoring_abstain_evidence_violation` | `37` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `75.5` |
| `ok_ignoring_abstain_support_violation` | `37` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `75.5` |
| `ok_rate_pct` | `75.5` |
| `problems.count` | `12` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `24` |
| `overall.accuracy` | `51.02` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `9` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `6` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `overall.confusion_matrix.No conflict.Complementary information` | `3` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `15` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `22` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `2` |
| `overall.distribution_pred.No conflict` | `25` |
| `overall.per_class.Complementary information.f1` | `0.486` |
| `overall.per_class.Complementary information.precision` | `0.409` |
| `overall.per_class.Complementary information.recall` | `0.6` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.0` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.0` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.167` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.5` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.1` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.682` |
| `overall.per_class.No conflict.precision` | `0.6` |
| `overall.per_class.No conflict.recall` | `0.789` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/baseline_qwen25_stagewise_base_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/baseline_qwen25_stagewise_base_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `176` |
| `overall.macro_f1` | `0.998` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9968` |
| `overall.per_class.partially supports.precision` | `0.9937` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9972` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9944` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `390` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `99.74` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

## `oracle_per_doc_notes` / `sft`

### `model_output_exports/val set/oracle_per_doc_notes/sft/llama31_8b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `93.88` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `3` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `18` |
| `citations.avg_citation_count` | `5.7143` |
| `citations.avg_sentence_coverage` | `0.5486` |
| `citations.avg_unique_citations` | `4.2041` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3512` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5607` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `31` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8146` |
| `citation_coverage.below_threshold_examples.count` | `5` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `28` |
| `citation_coverage.pass_rate_pct` | `84.8` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `11` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `7` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `9` |
| `label_f1.macro_f1` | `0.525` |
| `label_f1.pairs_evaluated` | `46` |
| `label_f1.per_class.Complementary information.f1` | `0.6471` |
| `label_f1.per_class.Complementary information.precision` | `0.55` |
| `label_f1.per_class.Complementary information.recall` | `0.7857` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7778` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.7778` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7778` |
| `label_f1.per_class.No conflict.f1` | `0.6` |
| `label_f1.per_class.No conflict.precision` | `0.75` |
| `label_f1.per_class.No conflict.recall` | `0.5` |
| `ok_all_checks` | `44` |
| `ok_ignoring_abstain_evidence_violation` | `45` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `91.8` |
| `ok_ignoring_abstain_support_violation` | `45` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `91.8` |
| `ok_rate_pct` | `89.8` |
| `problems.count` | `5` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `19` |
| `overall.accuracy` | `65.22` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `11` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `1` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `7` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `9` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `20` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `12` |
| `overall.per_class.Complementary information.f1` | `0.647` |
| `overall.per_class.Complementary information.precision` | `0.55` |
| `overall.per_class.Complementary information.recall` | `0.786` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.778` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.778` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.778` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `9` |
| `overall.per_class.No conflict.f1` | `0.6` |
| `overall.per_class.No conflict.precision` | `0.75` |
| `overall.per_class.No conflict.recall` | `0.5` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `46` |
| `top_confusions.count` | `9` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.json_array_parse_error: Expecting value: line 1 column 2 (char 1)` | `2` |
| `error_counts.think_block_not_unique` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `51` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `154` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `2` |
| `overall.confusion_matrix.supports.supports` | `161` |
| `overall.macro_f1` | `0.9958` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9935` |
| `overall.per_class.partially supports.precision` | `0.9872` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9938` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9877` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `366` |
| `totals.examples_with_any_eval` | `46` |
| `totals.micro_accuracy_doc_level` | `99.46` |
| `totals.total_doc_pairs_evaluated` | `368` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/sft/llama31_8b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `6.0816` |
| `citations.avg_sentence_coverage` | `0.6061` |
| `citations.avg_unique_citations` | `4.6122` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3552` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5599` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8486` |
| `citation_coverage.below_threshold_examples.count` | `5` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `28` |
| `citation_coverage.pass_rate_pct` | `84.8` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `10` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `5` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `11` |
| `label_f1.macro_f1` | `0.5432` |
| `label_f1.pairs_evaluated` | `46` |
| `label_f1.per_class.Complementary information.f1` | `0.625` |
| `label_f1.per_class.Complementary information.precision` | `0.5556` |
| `label_f1.per_class.Complementary information.recall` | `0.7143` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.75` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7368` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.7778` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `label_f1.per_class.No conflict.f1` | `0.6875` |
| `label_f1.per_class.No conflict.precision` | `0.7857` |
| `label_f1.per_class.No conflict.recall` | `0.6111` |
| `ok_all_checks` | `44` |
| `ok_ignoring_abstain_evidence_violation` | `45` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `91.8` |
| `ok_ignoring_abstain_support_violation` | `45` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `91.8` |
| `ok_rate_pct` | `89.8` |
| `problems.count` | `5` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `18` |
| `overall.accuracy` | `67.39` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `10` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `2` |
| `overall.confusion_matrix.Complementary information.No conflict` | `2` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `3` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `5` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `11` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `4` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `18` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `14` |
| `overall.per_class.Complementary information.f1` | `0.625` |
| `overall.per_class.Complementary information.precision` | `0.556` |
| `overall.per_class.Complementary information.recall` | `0.714` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.667` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.75` |
| `overall.per_class.Conflict due to outdated information.support` | `4` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.737` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.778` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.688` |
| `overall.per_class.No conflict.precision` | `0.786` |
| `overall.per_class.No conflict.recall` | `0.611` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `46` |
| `top_confusions.count` | `9` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `3` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_llama31_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `143` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `2` |
| `overall.confusion_matrix.supports.supports` | `161` |
| `overall.macro_f1` | `0.9956` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9931` |
| `overall.per_class.partially supports.precision` | `0.9862` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9938` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.9877` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `360` |
| `totals.examples_with_any_eval` | `46` |
| `totals.micro_accuracy_doc_level` | `99.45` |
| `totals.total_doc_pairs_evaluated` | `362` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/sft/mistral7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `97.96` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `1` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `16` |
| `citations.avg_citation_count` | `5.7143` |
| `citations.avg_sentence_coverage` | `0.484` |
| `citations.avg_unique_citations` | `4.2653` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3203` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5194` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `33` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `16` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `98.0` |
| `abstain_gold.correct` | `48` |
| `abstain_gold.false_abstain_ids.count` | `1` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.7031` |
| `citation_coverage.below_threshold_examples.count` | `13` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `20` |
| `citation_coverage.pass_rate_pct` | `60.6` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `7` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.5141` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.5185` |
| `label_f1.per_class.Complementary information.precision` | `0.5385` |
| `label_f1.per_class.Complementary information.recall` | `0.5` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7059` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8571` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `label_f1.per_class.No conflict.f1` | `0.6316` |
| `label_f1.per_class.No conflict.precision` | `0.6316` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `46` |
| `ok_ignoring_abstain_evidence_violation` | `47` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `95.9` |
| `ok_ignoring_abstain_support_violation` | `47` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `95.9` |
| `ok_rate_pct` | `93.9` |
| `problems.count` | `3` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `19` |
| `overall.accuracy` | `62.5` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `7` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `5` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `6` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `2` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `14` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `13` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `7` |
| `overall.distribution_pred.No conflict` | `19` |
| `overall.per_class.Complementary information.f1` | `0.519` |
| `overall.per_class.Complementary information.precision` | `0.538` |
| `overall.per_class.Complementary information.recall` | `0.5` |
| `overall.per_class.Complementary information.support` | `14` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.706` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.857` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.6` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.632` |
| `overall.per_class.No conflict.precision` | `0.632` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `48` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `52` |
| `overall.confusion_matrix.irrelevant.partially supports` | `1` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `154` |
| `overall.confusion_matrix.partially supports.supports` | `2` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `172` |
| `overall.macro_f1` | `0.9897` |
| `overall.per_class.irrelevant.f1` | `0.9905` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `0.9811` |
| `overall.per_class.partially supports.f1` | `0.9872` |
| `overall.per_class.partially supports.precision` | `0.9872` |
| `overall.per_class.partially supports.recall` | `0.9872` |
| `overall.per_class.supports.f1` | `0.9914` |
| `overall.per_class.supports.precision` | `0.9885` |
| `overall.per_class.supports.recall` | `0.9942` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `378` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `98.95` |
| `totals.total_doc_pairs_evaluated` | `382` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/sft/mistral7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `95.92` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `2` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `17` |
| `citations.avg_citation_count` | `5.6531` |
| `citations.avg_sentence_coverage` | `0.4684` |
| `citations.avg_unique_citations` | `4.2245` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3077` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.504` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `32` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `17` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `14` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `14` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `2` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `2` |
| `abstain_gold.accuracy_pct` | `95.9` |
| `abstain_gold.correct` | `47` |
| `abstain_gold.false_abstain_ids.count` | `2` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.6971` |
| `citation_coverage.below_threshold_examples.count` | `13` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `32` |
| `citation_coverage.pass_count` | `19` |
| `citation_coverage.pass_rate_pct` | `59.4` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `9` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `5` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `5` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `12` |
| `label_f1.macro_f1` | `0.5464` |
| `label_f1.pairs_evaluated` | `47` |
| `label_f1.per_class.Complementary information.f1` | `0.6` |
| `label_f1.per_class.Complementary information.precision` | `0.6` |
| `label_f1.per_class.Complementary information.recall` | `0.6` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7143` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.5556` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7692` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.625` |
| `label_f1.per_class.No conflict.f1` | `0.6486` |
| `label_f1.per_class.No conflict.precision` | `0.6667` |
| `label_f1.per_class.No conflict.recall` | `0.6316` |
| `ok_all_checks` | `44` |
| `ok_ignoring_abstain_evidence_violation` | `45` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `91.8` |
| `ok_ignoring_abstain_support_violation` | `45` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `91.8` |
| `ok_rate_pct` | `89.8` |
| `problems.count` | `5` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `18` |
| `overall.accuracy` | `65.96` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `9` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Complementary information.No conflict` | `5` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `5` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `5` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `12` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `15` |
| `overall.distribution_pred.Conflict due to outdated information` | `9` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `5` |
| `overall.distribution_pred.No conflict` | `18` |
| `overall.per_class.Complementary information.f1` | `0.6` |
| `overall.per_class.Complementary information.precision` | `0.6` |
| `overall.per_class.Complementary information.recall` | `0.6` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.714` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.556` |
| `overall.per_class.Conflict due to outdated information.recall` | `1.0` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.769` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `1.0` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.625` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `8` |
| `overall.per_class.No conflict.f1` | `0.649` |
| `overall.per_class.No conflict.precision` | `0.667` |
| `overall.per_class.No conflict.recall` | `0.632` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `47` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_not_unique` | `2` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_mistral7b_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `55` |
| `overall.confusion_matrix.irrelevant.partially supports` | `1` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `146` |
| `overall.confusion_matrix.partially supports.supports` | `2` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `1` |
| `overall.confusion_matrix.supports.supports` | `160` |
| `overall.macro_f1` | `0.9894` |
| `overall.per_class.irrelevant.f1` | `0.991` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `0.9821` |
| `overall.per_class.partially supports.f1` | `0.9865` |
| `overall.per_class.partially supports.precision` | `0.9865` |
| `overall.per_class.partially supports.recall` | `0.9865` |
| `overall.per_class.supports.f1` | `0.9907` |
| `overall.per_class.supports.precision` | `0.9877` |
| `overall.per_class.supports.recall` | `0.9938` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `361` |
| `totals.examples_with_any_eval` | `47` |
| `totals.micro_accuracy_doc_level` | `98.9` |
| `totals.total_doc_pairs_evaluated` | `365` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/sft/qwen25_32b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.3469` |
| `citations.avg_sentence_coverage` | `0.6312` |
| `citations.avg_unique_citations` | `4.102` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3638` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5604` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9097` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `11` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `4` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `14` |
| `label_f1.macro_f1` | `0.57` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.6875` |
| `label_f1.per_class.Complementary information.precision` | `0.6471` |
| `label_f1.per_class.Complementary information.recall` | `0.7333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.75` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.7778` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.875` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `label_f1.per_class.No conflict.f1` | `0.7179` |
| `label_f1.per_class.No conflict.precision` | `0.7` |
| `label_f1.per_class.No conflict.recall` | `0.7368` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `14` |
| `overall.accuracy` | `71.43` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `11` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `3` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `7` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `4` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `14` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `17` |
| `overall.distribution_pred.Conflict due to outdated information` | `4` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `8` |
| `overall.distribution_pred.No conflict` | `20` |
| `overall.per_class.Complementary information.f1` | `0.688` |
| `overall.per_class.Complementary information.precision` | `0.647` |
| `overall.per_class.Complementary information.recall` | `0.733` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.667` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.75` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.778` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.875` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.7` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.718` |
| `overall.per_class.No conflict.precision` | `0.7` |
| `overall.per_class.No conflict.recall` | `0.737` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `7` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `177` |
| `overall.macro_f1` | `1.0` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `1.0` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `1.0` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `1.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `391` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `100.0` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/sft/qwen25_32b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.5306` |
| `citations.avg_sentence_coverage` | `0.6265` |
| `citations.avg_unique_citations` | `4.3673` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.356` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5489` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.9029` |
| `citation_coverage.below_threshold_examples.count` | `2` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `32` |
| `citation_coverage.pass_rate_pct` | `94.1` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `11` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `3` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `3` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `14` |
| `label_f1.macro_f1` | `0.5777` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.7097` |
| `label_f1.per_class.Complementary information.precision` | `0.6875` |
| `label_f1.per_class.Complementary information.recall` | `0.7333` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.6` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8421` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7368` |
| `label_f1.per_class.No conflict.precision` | `0.7368` |
| `label_f1.per_class.No conflict.recall` | `0.7368` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `13` |
| `overall.accuracy` | `73.47` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `11` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `3` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `3` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `2` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `0` |
| `overall.confusion_matrix.No conflict.Complementary information` | `3` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `2` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `14` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `16` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `19` |
| `overall.per_class.Complementary information.f1` | `0.71` |
| `overall.per_class.Complementary information.precision` | `0.688` |
| `overall.per_class.Complementary information.recall` | `0.733` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.6` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.6` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.6` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.842` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.737` |
| `overall.per_class.No conflict.precision` | `0.737` |
| `overall.per_class.No conflict.recall` | `0.737` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `6` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `158` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `177` |
| `overall.macro_f1` | `1.0` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `1.0` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `1.0` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `1.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `391` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `100.0` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/sft/qwen25_7b/minimal_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.sanitized.jsonl`
- Additional identical output files: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.raw.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.4694` |
| `citations.avg_sentence_coverage` | `0.6197` |
| `citations.avg_unique_citations` | `4.4082` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3398` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5434` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `49` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.8931` |
| `citation_coverage.below_threshold_examples.count` | `3` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `34` |
| `citation_coverage.pass_count` | `31` |
| `citation_coverage.pass_rate_pct` | `91.2` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `7` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `7` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `16` |
| `label_f1.macro_f1` | `0.5821` |
| `label_f1.pairs_evaluated` | `49` |
| `label_f1.per_class.Complementary information.f1` | `0.5833` |
| `label_f1.per_class.Complementary information.precision` | `0.7778` |
| `label_f1.per_class.Complementary information.recall` | `0.4667` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.8` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.8` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7273` |
| `label_f1.per_class.No conflict.precision` | `0.64` |
| `label_f1.per_class.No conflict.recall` | `0.8421` |
| `ok_all_checks` | `48` |
| `ok_ignoring_abstain_evidence_violation` | `49` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `100.0` |
| `ok_ignoring_abstain_support_violation` | `49` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `100.0` |
| `ok_rate_pct` | `98.0` |
| `problems.count` | `1` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `14` |
| `overall.accuracy` | `71.43` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `7` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `7` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `1` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.No conflict.No conflict` | `16` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `19` |
| `overall.distribution_pred.Complementary information` | `9` |
| `overall.distribution_pred.Conflict due to outdated information` | `5` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_pred.No conflict` | `25` |
| `overall.per_class.Complementary information.f1` | `0.583` |
| `overall.per_class.Complementary information.precision` | `0.778` |
| `overall.per_class.Complementary information.recall` | `0.467` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.8` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.8` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.8` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.727` |
| `overall.per_class.No conflict.precision` | `0.64` |
| `overall.per_class.No conflict.recall` | `0.842` |
| `overall.per_class.No conflict.support` | `19` |
| `overall.support` | `49` |
| `top_confusions.count` | `8` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_minimal_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `1` |
| `overall.confusion_matrix.partially supports.partially supports` | `157` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `0` |
| `overall.confusion_matrix.supports.supports` | `177` |
| `overall.macro_f1` | `0.996` |
| `overall.per_class.irrelevant.f1` | `0.9912` |
| `overall.per_class.irrelevant.precision` | `0.9825` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9968` |
| `overall.per_class.partially supports.precision` | `1.0` |
| `overall.per_class.partially supports.recall` | `0.9937` |
| `overall.per_class.supports.f1` | `1.0` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `1.0` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `390` |
| `totals.examples_with_any_eval` | `49` |
| `totals.micro_accuracy_doc_level` | `99.74` |
| `totals.total_doc_pairs_evaluated` | `391` |
| `totals.total_examples_in_gens` | `49` |

### `model_output_exports/val set/oracle_per_doc_notes/sft/qwen25_7b/strict_prompt_outputs.jsonl`

- Matched output: `outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise.sanitized.jsonl`
- Report directory: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise`

#### `final_answer.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/final_answer.json`

| Metric | Value |
| --- | --- |
| `abstain.accuracy_pct` | `100.0` |
| `abstain.false_negative_ids.count` | `0` |
| `abstain.false_positive_ids.count` | `0` |
| `abstain.gold_abstain_count` | `15` |
| `abstain.pred_abstain_count` | `15` |
| `citations.avg_citation_count` | `5.4694` |
| `citations.avg_sentence_coverage` | `0.6059` |
| `citations.avg_unique_citations` | `4.2857` |
| `citations.rows_with_invalid_citations` | `0` |
| `lexical_overlap_non_abstain.avg_rouge_l_f1` | `0.3398` |
| `lexical_overlap_non_abstain.avg_token_f1` | `0.5343` |
| `lexical_overlap_non_abstain.gold_non_abstain_count` | `34` |
| `lexical_overlap_non_abstain.gold_non_abstain_usable_answer_count` | `34` |
| `lexical_overlap_non_abstain.low_overlap_sample.count` | `12` |
| `lexical_overlap_non_abstain.scored_pairs` | `34` |
| `notes.description` | `Final-answer proxy metrics for prompt settings where traces may not be required.` |
| `notes.gold_abstain_source` | `expected_response.abstain when present, otherwise answerable_under_evidence` |
| `notes.gold_answer_filter` | `Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.` |
| `notes.warning` | `Lexical overlap is not a semantic quality judge. Use it for triage only.` |
| `total` | `49` |
| `trace_presence.sentinel_count` | `49` |
| `trace_presence.think_count` | `48` |

#### `contract.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/contract.json`

| Metric | Value |
| --- | --- |
| `abstain_count` | `15` |
| `abstain_diagnostics.pred_abstain_with_partial_only_count` | `13` |
| `abstain_diagnostics.pred_abstain_with_partial_only_ids.count` | `13` |
| `abstain_diagnostics.pred_abstain_with_support_count` | `1` |
| `abstain_diagnostics.pred_abstain_with_support_ids.count` | `1` |
| `abstain_gold.accuracy_pct` | `100.0` |
| `abstain_gold.correct` | `49` |
| `abstain_gold.false_abstain_ids.count` | `0` |
| `abstain_gold.gold_abstain_with_partial_doc_count` | `14` |
| `abstain_gold.gold_abstain_with_partial_doc_ids.count` | `14` |
| `abstain_gold.gold_abstain_with_supporting_doc_count` | `1` |
| `abstain_gold.gold_abstain_with_supporting_doc_ids.count` | `1` |
| `abstain_gold.missed_abstain_ids.count` | `0` |
| `abstain_gold.total_with_gold` | `49` |
| `citation_coverage.avg_sentence_coverage` | `0.897` |
| `citation_coverage.below_threshold_examples.count` | `0` |
| `citation_coverage.definition` | `Separate citation-discipline metric over non-abstaining final answers. Sentence coverage is the fraction of final-answer sentences that contain at least one in-range citation. This does not affect contract_ok.` |
| `citation_coverage.evaluated_non_abstain_count` | `33` |
| `citation_coverage.pass_count` | `33` |
| `citation_coverage.pass_rate_pct` | `100.0` |
| `citation_coverage.threshold` | `0.75` |
| `label_f1.confusion_matrix.Complementary information.Complementary information` | `6` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `label_f1.confusion_matrix.Complementary information.No conflict` | `7` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `label_f1.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `label_f1.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `label_f1.confusion_matrix.No conflict.Complementary information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `label_f1.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `label_f1.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `label_f1.confusion_matrix.No conflict.No conflict` | `16` |
| `label_f1.macro_f1` | `0.5671` |
| `label_f1.pairs_evaluated` | `48` |
| `label_f1.per_class.Complementary information.f1` | `0.5217` |
| `label_f1.per_class.Complementary information.precision` | `0.75` |
| `label_f1.per_class.Complementary information.recall` | `0.4` |
| `label_f1.per_class.Conflict due to misinformation.f1` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.precision` | `0.0` |
| `label_f1.per_class.Conflict due to misinformation.recall` | `0.0` |
| `label_f1.per_class.Conflict due to outdated information.f1` | `0.7273` |
| `label_f1.per_class.Conflict due to outdated information.precision` | `0.6667` |
| `label_f1.per_class.Conflict due to outdated information.recall` | `0.8` |
| `label_f1.per_class.Conflicting opinions or research outcomes.f1` | `0.8421` |
| `label_f1.per_class.Conflicting opinions or research outcomes.precision` | `0.8889` |
| `label_f1.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `label_f1.per_class.No conflict.f1` | `0.7442` |
| `label_f1.per_class.No conflict.precision` | `0.64` |
| `label_f1.per_class.No conflict.recall` | `0.8889` |
| `ok_all_checks` | `47` |
| `ok_ignoring_abstain_evidence_violation` | `48` |
| `ok_ignoring_abstain_evidence_violation_rate_pct` | `98.0` |
| `ok_ignoring_abstain_support_violation` | `48` |
| `ok_ignoring_abstain_support_violation_rate_pct` | `98.0` |
| `ok_rate_pct` | `95.9` |
| `problems.count` | `2` |
| `total` | `49` |

#### `conflict_type.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/conflict_type.json`

| Metric | Value |
| --- | --- |
| `mismatches_sample.count` | `15` |
| `overall.accuracy` | `70.83` |
| `overall.confusion_matrix.Complementary information.Complementary information` | `6` |
| `overall.confusion_matrix.Complementary information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Complementary information.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.Complementary information.Conflicting opinions or research outcomes` | `1` |
| `overall.confusion_matrix.Complementary information.No conflict` | `7` |
| `overall.confusion_matrix.Conflict due to misinformation.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to misinformation.No conflict` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Complementary information` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflict due to outdated information` | `4` |
| `overall.confusion_matrix.Conflict due to outdated information.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.Conflict due to outdated information.No conflict` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Complementary information` | `1` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflict due to outdated information` | `0` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.Conflicting opinions or research outcomes` | `8` |
| `overall.confusion_matrix.Conflicting opinions or research outcomes.No conflict` | `1` |
| `overall.confusion_matrix.No conflict.Complementary information` | `1` |
| `overall.confusion_matrix.No conflict.Conflict due to misinformation` | `0` |
| `overall.confusion_matrix.No conflict.Conflict due to outdated information` | `1` |
| `overall.confusion_matrix.No conflict.Conflicting opinions or research outcomes` | `0` |
| `overall.confusion_matrix.No conflict.No conflict` | `16` |
| `overall.distribution_actual.Complementary information` | `15` |
| `overall.distribution_actual.Conflict due to outdated information` | `5` |
| `overall.distribution_actual.Conflicting opinions or research outcomes` | `10` |
| `overall.distribution_actual.No conflict` | `18` |
| `overall.distribution_pred.Complementary information` | `8` |
| `overall.distribution_pred.Conflict due to outdated information` | `6` |
| `overall.distribution_pred.Conflicting opinions or research outcomes` | `9` |
| `overall.distribution_pred.No conflict` | `25` |
| `overall.per_class.Complementary information.f1` | `0.522` |
| `overall.per_class.Complementary information.precision` | `0.75` |
| `overall.per_class.Complementary information.recall` | `0.4` |
| `overall.per_class.Complementary information.support` | `15` |
| `overall.per_class.Conflict due to misinformation.f1` | `0.0` |
| `overall.per_class.Conflict due to misinformation.precision` | `0.0` |
| `overall.per_class.Conflict due to misinformation.recall` | `0.0` |
| `overall.per_class.Conflict due to misinformation.support` | `0` |
| `overall.per_class.Conflict due to outdated information.f1` | `0.727` |
| `overall.per_class.Conflict due to outdated information.precision` | `0.667` |
| `overall.per_class.Conflict due to outdated information.recall` | `0.8` |
| `overall.per_class.Conflict due to outdated information.support` | `5` |
| `overall.per_class.Conflicting opinions or research outcomes.f1` | `0.842` |
| `overall.per_class.Conflicting opinions or research outcomes.precision` | `0.889` |
| `overall.per_class.Conflicting opinions or research outcomes.recall` | `0.8` |
| `overall.per_class.Conflicting opinions or research outcomes.support` | `10` |
| `overall.per_class.No conflict.f1` | `0.744` |
| `overall.per_class.No conflict.precision` | `0.64` |
| `overall.per_class.No conflict.recall` | `0.889` |
| `overall.per_class.No conflict.support` | `18` |
| `overall.support` | `48` |
| `top_confusions.count` | `9` |
| `totals.errors_count` | `0` |
| `totals.evaluated_ids` | `49` |
| `totals.unique_ids_in_outputs` | `49` |

#### `doc_verdicts.json`

Source: `outputs/reports/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise/doc_verdicts.json`

| Metric | Value |
| --- | --- |
| `error_counts.think_block_missing_or_misaligned` | `1` |
| `notes.canon_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/data/splits/val_stagewise.jsonl` |
| `notes.description` | `Doc-level verdict comparison between Stage-3 per_doc_notes and text-mode v5 outputs.` |
| `notes.gens_source` | `/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_oracle_notes_strict_val_stagewise.sanitized.jsonl` |
| `overall.confusion_matrix.irrelevant.irrelevant` | `56` |
| `overall.confusion_matrix.irrelevant.partially supports` | `0` |
| `overall.confusion_matrix.irrelevant.supports` | `0` |
| `overall.confusion_matrix.partially supports.irrelevant` | `0` |
| `overall.confusion_matrix.partially supports.partially supports` | `157` |
| `overall.confusion_matrix.partially supports.supports` | `0` |
| `overall.confusion_matrix.supports.irrelevant` | `0` |
| `overall.confusion_matrix.supports.partially supports` | `3` |
| `overall.confusion_matrix.supports.supports` | `164` |
| `overall.macro_f1` | `0.9938` |
| `overall.per_class.irrelevant.f1` | `1.0` |
| `overall.per_class.irrelevant.precision` | `1.0` |
| `overall.per_class.irrelevant.recall` | `1.0` |
| `overall.per_class.partially supports.f1` | `0.9905` |
| `overall.per_class.partially supports.precision` | `0.9812` |
| `overall.per_class.partially supports.recall` | `1.0` |
| `overall.per_class.supports.f1` | `0.9909` |
| `overall.per_class.supports.precision` | `1.0` |
| `overall.per_class.supports.recall` | `0.982` |
| `overall.verdict_labels.count` | `3` |
| `totals.correct_doc_pairs` | `377` |
| `totals.examples_with_any_eval` | `48` |
| `totals.micro_accuracy_doc_level` | `99.21` |
| `totals.total_doc_pairs_evaluated` | `380` |
| `totals.total_examples_in_gens` | `49` |

