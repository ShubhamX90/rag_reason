# Benchmark Human-Review Agreement Statistics

## Consolidated population

The consolidated benchmark-preselection review dataset contains **1,454 unique records**. Every record is represented using the same first/second-review schema, enabling one consistent agreement analysis across the complete population while retaining review provenance for reproducibility.

The final release holdout contains 736 examples. Of these, 503 have a benchmark preselection record; the remaining 105 answerable and 128 refusal-quality examples are included through their corresponding consensus-completion strata.

## Agreement metrics and their meaning

### What is the IAA score in this study?

Inter-annotator agreement (IAA) is the reliability of the labels assigned to the same item by different reviewers; it is a property of a particular task and label field rather than one universal scalar. For this benchmark, the **primary IAA result** is Cohen's κ for the nominal five-way `preliminary_conflict_type` label across all 1,454 review pairs: **κ = 0.9217**, with **94.77% raw agreement**. This is the principal number to report for taxonomy reliability. The companion retention-decision IAA is **κ = 0.9228** with **98.21% raw agreement**.

This follows the standard two-coder nominal-label setting introduced by Cohen: report both observed agreement and a chance-corrected coefficient. In computational-linguistics annotation work, agreement coefficients must be interpreted in light of the task, label inventory, and marginal label frequencies rather than used as a context-free quality score. See [Cohen (1960)](https://journals.sagepub.com/doi/abs/10.1177/001316446002000104) and [Artstein and Poesio (2008)](https://aclanthology.org/J08-4004/).

**Raw agreement** is the proportion of records on which the two review labels are identical. It communicates the directly observable consistency of the review decisions.

**Cohen's kappa (κ)** measures agreement after accounting for agreement expected from the reviewers' marginal label frequencies. It is therefore the primary chance-corrected reliability statistic for these categorical two-reviewer decisions. Values near 1 indicate highly stable judgments; values near 0 indicate no more agreement than expected from the label distributions alone.

For a field with categories `c`, let `P_o` be raw agreement and let `p_{1,c}` and `p_{2,c}` be the proportions assigned to category `c` by the first and second reviewer. Expected agreement is `P_e = Σ_c p_{1,c} p_{2,c}`, and the reported score is `κ = (P_o − P_e) / (1 − P_e)`. The implementation computes these quantities directly from the 1,454 paired labels; no ordinal weighting is used because the primary conflict taxonomy is nominal.

### Calculation

For `N` review pairs, with first and second labels `y_i^(1)` and `y_i^(2)`, the reported quantities are:

$$ P_o = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i^{(1)} = y_i^{(2)}] $$

$$ P_e = \sum_{c \in C} p_{1,c} p_{2,c} $$

$$ \kappa = \frac{P_o - P_e}{1 - P_e} $$

Here, `C` is the field's category set, `p_{1,c}` and `p_{2,c}` are the reviewers' marginal proportions for category `c`, and `𝟙[·]` equals 1 when the labels match and 0 otherwise. Raw agreement is `P_o`; Cohen's κ is the chance-corrected agreement score. The implementation applies these formulas separately to each reported field.

Kappa is reported only for non-degenerate label distributions. When every refusal-quality item receives the same label from both sides, raw agreement is fully informative and κ is mathematically undefined because there is no category variation from which to estimate chance agreement.

### Annotation dimensions and use

| Dimension | Type | What it establishes |
|---|---|---|
| `preliminary_conflict_type` | Five-way nominal taxonomy | Reliability of the benchmark's conflict categorization; primary IAA. |
| `human_preselect_decision` | Four-way nominal decision | Stability of item-retention judgments. |
| Confidence, retrieval quality, evidence sufficiency, clarity, specificity, reliability, relevant-document bin | Ordered categorical diagnostics | Stability of the evidence-quality criteria used to screen benchmark items. |
| `gold_answer_possible` | Binary | Agreement on whether the retrieved evidence permits a supported gold answer. |
| Refusal-quality fields | Binary/nominal refusal diagnostics | Agreement that abstention is required and that the refusal target and rationale are evidence-grounded. |

## Overall benchmark-review reliability (n = 1,454)

Field | n | Raw agreement | Cohen's kappa
---|---:|---:|---:
human_preselect_decision | 1454 | 98.21% | 0.9228
preliminary_conflict_type | 1454 | 94.77% | 0.9217
preselection_confidence | 1454 | 99.38% | 0.9767
retrieval_quality | 1454 | 98.76% | 0.9550
evidence_sufficiency | 1454 | 98.76% | 0.9600
conflict_clarity | 1454 | 99.11% | 0.9660
query_specificity | 1454 | 99.66% | 0.9875
source_reliability | 1454 | 99.45% | 0.9881
relevant_doc_count_bin | 1454 | 98.07% | 0.9527
gold_answer_possible | 1454 | 99.04% | 0.9669

### Interpretation

The central benchmark-selection outcome is the conflict-type decision: 94.77% raw agreement with κ = 0.9217. The decision to retain an item is similarly stable (98.21%, κ = 0.9228). The supporting evidence-quality dimensions are even more consistent: all have raw agreement above 98% and κ from 0.9527 to 0.9881. Together, these results indicate that the review protocol yields stable retained-item, conflict-taxonomy, and evidence-assessment judgments across the complete consolidated population.

For paper presentation, lead with the conflict-type IAA because it evaluates the benchmark's central taxonomy. Report the retention-decision score beside it, then summarize the evidence-quality fields as supporting reliability checks. This avoids conflating the different annotation questions while making clear that the benchmark was screened for both semantic conflict structure and evidence adequacy.

## Refusal-quality agreement (n = 128)

Refusal examples are assessed with the same common review-pair structure, supplemented by refusal-specific quality checks. A valid refusal requires that the provided documents do not support a defensible answer, that the abstention ground truth matches this evidence condition, and that the stated rationale identifies the evidence gap.

Field | Definition | n | Raw agreement | Cohen's kappa
---|---|---:|---:|---:
refusal_required | Whether abstention is required under the retrieved evidence. | 128 | 100.00% | N/A (single-category)
refusal_ground_truth_valid | Whether the benchmark refusal target is evidence-grounded. | 128 | 100.00% | N/A (single-category)
refusal_rationale_quality | Quality of the stated evidence-gap rationale. | 128 | 100.00% | N/A (single-category)
refusal_quality_label | Overall validity label for the refusal item. | 128 | 100.00% | N/A (single-category)

All 128 refusal-quality checks have exact agreement. Their common `valid_refusal` label produces a single-category distribution, so chance-corrected κ is not estimable; the appropriate reported result is 100.00% exact agreement for each refusal-quality criterion.

## Why the final 736-example holdout was retained

The final holdout is a quality-controlled benchmark subset with 736 examples: 503 items with benchmark-preselection records, 105 answerable consensus-completion items, and 128 refusal-quality items. This composition keeps ordinary answerable evaluation and evidence-sensitive refusal evaluation in the same final benchmark while preserving a distinct refusal-quality assessment.

Among the 503 holdout items with preselection records, 386 (76.7%) were accepted and 117 (23.3%) were borderline accepted. The corresponding evidence-quality profile is strong: 383 (76.1%) high-confidence assessments, 404 (80.3%) good-retrieval assessments, 431 (85.7%) sufficient-evidence assessments, and 391 (77.7%) clear conflict judgments.

The preselected holdout component also retains all five conflict categories: {'Complementary information': 157, 'Conflict due to misinformation': 36, 'Conflict due to outdated information': 67, 'Conflicting opinions or research outcomes': 92, 'No conflict': 151}. It contains 151 items satisfying the strict top-tier no-conflict profile used during benchmark selection (accepted, high confidence, good retrieval, sufficient evidence, clear conflict status, specific query, strong sources, 4–6 relevant documents, and answerable evidence). The refusal-quality component adds 128 explicitly evidence-insufficient cases, enabling evaluation of both answer generation and calibrated abstention.

## Review-process summary

| Action | Count |
|---|---:|
| accept_first_review | 673 |
| edited_fields | 127 |

## Files

- `benchmark_preselection_consensus_1454.jsonl`: one consolidated first/second review pair per record.
- `benchmark_preselection_agreement_metrics.csv`: field-level agreement and kappa table.
- `benchmark_preselection_agreement_metrics.json`: machine-readable summary and label distributions.

## References

- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37–46. https://doi.org/10.1177/001316446002000104
- Artstein, R., & Poesio, M. (2008). *Inter-Coder Agreement for Computational Linguistics*. Computational Linguistics, 34(4), 555–596. https://doi.org/10.1162/coli.07-034-R2
