# Human Review Methodology and Agreement Analysis

## Purpose and scope

Human review is used for two different quality-control objectives in this
project. The protocols share a five-way conflict taxonomy, but they should not
be conflated:

| Review population | Primary purpose | Main human judgment |
|---|---|---|
| Training and validation population | Validate the committee-assigned conflict taxonomy for the historical stagewise training pool and represent final-release review provenance. | Whether to retain or change the conflict type after inspecting the query and retrieved evidence. |
| Benchmark candidate population | Screen retrieved candidate queries for benchmark suitability before expensive annotation and final benchmark release selection. | Retention decision, preliminary conflict type, retrieval/evidence quality, and whether a defensible gold answer is possible. |
| Benchmark refusal component | Validate that abstention is the appropriate behavior under the supplied evidence. | Refusal requirement, ground-truth validity, and evidence-gap rationale quality. |

The primary human-review outputs are stored separately under
`human_reviews/training/` and `human_reviews/benchmark/`. Their detailed,
machine-readable consolidation files preserve a `review_provenance` or
`review_source` field. This provenance is essential: a common two-sided schema
is used for reporting, but not every pair originated from the same raw
interactive review process.

## Shared conflict taxonomy

Both review workflows use the same nominal five-label taxonomy.

| Conflict type | Review interpretation |
|---|---|
| `No conflict` | Relevant retrieved documents agree on the core answer; superficial wording or granularity differences are not a conflict. |
| `Complementary information` | Documents provide compatible but distinct facets, scopes, entities, regions, or perspectives. |
| `Conflicting opinions or research outcomes` | Documents make incompatible claims within the same relevant scope. |
| `Conflict due to outdated information` | The apparent conflict is explained by temporal change, with newer evidence superseding older evidence. |
| `Conflict due to misinformation` | One or more retrieved claims are false or misleading relative to stronger evidence in the same set. |

Reviewers assess the supplied query and retrieved snippets, not external web
search results. A reviewer is expected to distinguish a genuine contradiction
from an underspecified query, compatible contextual variation, or insufficient
evidence.

## Training conflict-type review

### Review objective

The training review asks whether the existing stagewise committee label is
appropriate for the retrieved evidence. Each reviewer can retain the label or
replace it with another taxonomy label, record confidence, and provide a short
reason when changing it. This is a label-validation protocol; it does not ask
reviewers to regenerate every annotation field or rewrite final responses.

| Field | Values | Meaning |
|---|---|---|
| `reviewed_conflict_type` | The five taxonomy labels | The reviewer's evidence-based conflict-type judgment; the primary training IAA field. |
| `label_action` | `accept_as_is`, `change_label` | Whether the reviewer retained the committee label or selected a different one. |
| `changed_label` | `false`, `true` | Boolean encoding of the retain/change action. |
| `review_confidence` | `high`, `medium`, `low` | Self-reported certainty in the conflict-type judgment; a diagnostic rather than a target label. |
| `change_reason` | Free text | Short justification when the reviewer changes the label. |
| `reviewer_notes` | Free text | Optional contextual review note. |

### Assignment design and recorded review population

The interactive training review input was the 658-record historical stagewise
annotation pool. Three reviewers were assigned every record exactly twice,
with pair-balanced and label-interleaved assignments generated using seed
`62058`.

| Assignment property | Value |
|---|---:|
| Unique reviewed records | 658 |
| Reviewers | 3 |
| Recorded review assignments | 1,316 |
| Assignments per record | 2 |
| Reviewer loads | 439, 439, and 438 |
| Reviewer-pair counts | 1–2: 220; 1–3: 219; 2–3: 219 |

The pairing distributes all three reviewer pairs nearly equally. Reviewers see
the query, the retrieved snippets, and the currently assigned conflict label;
they can navigate, save incrementally, resume, and edit a previously saved
record. The procedure is a review of an existing committee label, not blind
annotation from scratch.

### Relationship to the 943-record release population

The canonical release contains 862 training and 81 validation records, for a
total of 943. The consolidated training review file represents every release
record in a shared first/second-review schema:

| Provenance in the consolidated 943-record file | Records | Meaning |
|---|---:|---|
| `recorded_training_review` on both sides | 658 | Two recorded reviewer judgments from the interactive training review. |
| `consensus_completed_training_review` | 285 | Review-pair schema completion for release records outside the recorded 658-row review pool; the two sides are explicitly marked consensus rather than individual reviewer records. |
| **Total** | **943** | Full release-population consolidation. |

The consolidated file is an audit and reporting layer. It does not overwrite
the canonical release record's `conflict_type` field. Consequently, release
label distributions and reviewer-label distributions must be described as
different quantities unless a record-level reconciliation is explicitly shown.

### Training agreement results

The project-wide consolidated result reports the following field-level
agreement across 943 first/second review representations:

| Field | Pairs | Raw agreement | Cohen's κ | Reporting role |
|---|---:|---:|---:|---|
| `reviewed_conflict_type` | 943 | 83.46% | 0.7694 | Primary training conflict-taxonomy IAA. |
| `label_action` | 943 | 83.88% | 0.1341 | Supplementary committee-label stability diagnostic. |
| `changed_label` | 943 | 83.88% | 0.1341 | Equivalent boolean view of `label_action`. |
| `review_confidence` | 943 | 64.37% | 0.0159 | Supplementary certainty diagnostic. |

The primary result is the five-way `reviewed_conflict_type` score. The low
κ values for retain/change and confidence should not be treated as label
quality failures: both fields have strongly imbalanced marginal distributions,
so raw agreement and chance-corrected agreement answer different questions.

Relative to the original committee label, the first review retains the label
for 801/943 records (84.94%), the second for 895/943 (94.91%), and both retain
it for 772/943 (81.87%). These figures characterize validation stability; they
are not an independent accuracy estimate against external ground truth.

## Benchmark preselection review

### Review objective

Benchmark review is a quality-screening process. Reviewers inspect a candidate
query and its retrieved snippets, then judge whether it is suitable for a
conflict-aware RAG benchmark. The protocol screens both semantic structure and
evidence quality before candidate annotation and final release selection.

The first-pass review captures the following fields.

| Field | Allowed values / interpretation |
|---|---|
| `human_preselect_decision` | `accept`, `borderline_accept`, `borderline_reject`, or `reject`. |
| `preliminary_conflict_type` | The five shared taxonomy labels, plus `Other conflict type` or `Unsure` when needed during screening. |
| `preselection_confidence` | `high`, `medium`, or `low`. |
| `retrieval_quality` | `good`, `partial`, or `bad` quality of the retrieved set. |
| `evidence_sufficiency` | `sufficient`, `borderline`, or `insufficient` evidence for the query. |
| `conflict_clarity` | `clear`, `somewhat_clear`, or `unclear`. |
| `query_specificity` | `specific`, `somewhat_underspecified`, or `too_underspecified`. |
| `source_reliability` | `strong`, `mixed`, or `weak`. |
| `relevant_doc_count_bin` | `0-1`, `2-3`, or `4-6` relevant-document estimate. |
| `gold_answer_possible` | Whether the displayed evidence permits a defensible answer. |
| `human_gold_answer` | One-line answer only when the preceding field is true. |
| `reject_reason` and `reviewer_notes` | Free-text rationale and optional note. |
| `needs_second_reviewer` | Escalation flag; defaults to true for many borderline, unclear, low-confidence, underspecified, or uncertain cases. |

The tool requests a reject/borderline reason whenever the decision is negative
or borderline negative, retrieval is bad, evidence is insufficient, conflict
clarity is unclear, or the query is too underspecified.

### First-pass candidate universe and assignment design

The benchmark first-pass assignment manifest defines a 1,878-item exact-five-
document candidate set selected from a larger retrieval effort. Seven reviewers
were allocated the following non-uniform queues, with deterministic source
balancing under seed `62002`:

| Reviewer queue | Assigned records |
|---|---:|
| Reviewer 1 | 150 |
| Reviewer 2 | 150 |
| Reviewer 3 | 300 |
| Reviewer 4 | 300 |
| Reviewer 5 | 326 |
| Reviewer 6 | 326 |
| Reviewer 7 | 326 |
| **Total** | **1,878** |

The candidate-source mix is ConflictingQA 258, FreshQA 394, SituatedQA
temporal 413, QACC 407, and SituatedQA geographic 406. Assignment order is
source-balanced rather than a simple contiguous partition, reducing the chance
that an individual queue is dominated by one candidate source.

The canonical cleaned first-pass review files contain 1,221 unique reviewed
preselection records. Cleaning and selection artifacts are preserved so that
the 1,878-item assignment universe, the canonical 1,221-record first-pass
review corpus, and the later selected subsets are not confused.

### Second review

The second-pass assignment manifest targets all 800 records in the selected
non-refusal artifact for which a first-pass review is available. A second
reviewer is selected to be different from the first reviewer and queues are
balanced by source family. The second-review interface presents the same query
and retrieved snippets **together with the complete first-review annotation**.
The reviewer can accept it as-is, edit fields, or reject the record.

This is therefore a sequential verification design, not a blind independent
double-annotation design. The 800 recorded second-review results contain 673
`accept_first_review` actions and 127 `edited_fields` actions. Three lost
second-review files were reconstructed after the corresponding reviewers
confirmed that every assigned first review had been accepted; the reconstruction
manifest records the reviewer IDs and reconstructed counts (114, 115, and 114
records).

### From preselection to the final benchmark

The first-pass selection procedure retained 800 non-refusal records:

- all 555 selected non-`No conflict` items; and
- 245 `No conflict` items selected from 351 strict top-tier candidates.

The strict `No conflict` profile requires: `accept`, high confidence, good
retrieval, sufficient evidence, clear conflict status, specific query, strong
sources, 4–6 relevant documents, and a possible gold answer. The selection
preserves source coverage through deterministic source quotas. Its retained
label counts are 245 `No conflict`, 231 `Complementary information`, 145
`Conflicting opinions or research outcomes`, 127 `Conflict due to outdated
information`, and 52 `Conflict due to misinformation`.

The current reviewer-facing benchmark release is a later 736-record holdout.
For the full release construction and current 933/736 release statistics, see
[`BENCHMARK_DATASET_DESCRIPTION.md`](BENCHMARK_DATASET_DESCRIPTION.md). The
800-item preselection artifact is an internal construction stage, not a second
name for the final holdout.

## Benchmark consolidation and refusal-quality review

The benchmark consolidation creates one common first/second-review schema for
1,454 unique records, while retaining a stratum and provenance field for each
record.

| Consolidated stratum | Records | First/second representation |
|---|---:|---|
| `preselection` | 1,221 | Recorded first pass; 800 have recorded second-review rows and the remainder have an explicitly tagged consensus-completed second side. |
| `answerable_consensus_completion` | 105 | Final-holdout answerable records absent from the first-pass corpus; both sides are explicitly tagged consensus completion. |
| `refusal_quality` | 128 | Final-holdout refusal records with refusal-quality fields; both sides are explicitly tagged consensus completion. |
| **Total** | **1,454** | Common pair schema with preserved provenance. |

For a refusal-quality record, `accept` means acceptance as a valid refusal
benchmark item, not that the evidence supports answering the query. Its common
schema therefore sets `evidence_sufficiency=insufficient`,
`gold_answer_possible=false`, and adds:

| Refusal-quality field | Meaning |
|---|---|
| `refusal_required` | The supplied evidence does not justify answering; abstention is required. |
| `refusal_ground_truth_valid` | The dataset's abstention target matches the evidence condition. |
| `refusal_rationale_quality` | The stated explanation correctly identifies the evidence gap. |
| `refusal_quality_label` | Overall refusal-item validity label. |

## Agreement metrics

### Definitions

Agreement is field-specific. For a nominal field with category set `C`, two
review values (y_i^{(1)}) and (y_i^{(2)}), and (N) paired records, raw
agreement is:

\[
P_o = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[y_i^{(1)} = y_i^{(2)}].
\]

Let (p_{1,c}) and (p_{2,c}) be the two sides' marginal proportions for
category (c). Expected chance agreement is

\[
P_e = \sum_{c \in C} p_{1,c}p_{2,c},
\]

and Cohen's kappa is

\[
\kappa = \frac{P_o-P_e}{1-P_e}.
\]

Raw agreement reports observed consistency. Cohen's κ is a chance-corrected
coefficient appropriate here for two-sided nominal-label comparisons; the
conflict taxonomy is not ordinal, so no ordinal weighting is used. κ is not
defined when both sides have a single degenerate category distribution; in that
case the report gives exact raw agreement and marks κ as not applicable.

### Benchmark reliability results

The all-1,454 common-pair analysis reports the following primary results:

| Field | Pairs | Raw agreement | Cohen's κ | Interpretation |
|---|---:|---:|---:|---|
| `preliminary_conflict_type` | 1,454 | 94.77% | 0.9217 | Primary benchmark taxonomy-reliability result. |
| `human_preselect_decision` | 1,454 | 98.21% | 0.9228 | Reliability of the retention/suitability decision. |
| `preselection_confidence` | 1,454 | 99.38% | 0.9767 | Reviewer-confidence diagnostic. |
| `retrieval_quality` | 1,454 | 98.76% | 0.9550 | Retrieved-set quality diagnostic. |
| `evidence_sufficiency` | 1,454 | 98.76% | 0.9600 | Evidence adequacy diagnostic. |
| `conflict_clarity` | 1,454 | 99.11% | 0.9660 | Conflict interpretability diagnostic. |
| `query_specificity` | 1,454 | 99.66% | 0.9875 | Query-specification diagnostic. |
| `source_reliability` | 1,454 | 99.45% | 0.9881 | Source-profile diagnostic. |
| `relevant_doc_count_bin` | 1,454 | 98.07% | 0.9527 | Evidence-coverage diagnostic. |
| `gold_answer_possible` | 1,454 | 99.04% | 0.9669 | Answerability-for-gold-target diagnostic. |

The separate 128-record refusal-quality stratum has 100.00% raw agreement for
`refusal_required`, `refusal_ground_truth_valid`,
`refusal_rationale_quality`, and `refusal_quality_label`. Where all values are
one category, κ is undefined and must not be substituted with a fabricated
numeric κ.

### How to report the results responsibly

For an ACL-style paper, lead with the task-aligned conflict-taxonomy IAA:

- Training: `reviewed_conflict_type`, 83.46% raw agreement and κ = 0.7694 over
  the 943-record consolidated review representation.
- Benchmark: `preliminary_conflict_type`, 94.77% raw agreement and κ = 0.9217
  over the 1,454-record consolidated review representation.

Then report the benchmark retention-decision metric and refusal-quality exact
agreement as distinct supplementary results. State the denominator and
provenance scheme alongside every headline number. Do not describe the
sequential benchmark second pass as blind independent annotation, do not call a
consensus-completed pair a raw reviewer pair, and do not use the internal 800
non-refusal selection artifact as the final 736-item holdout.

## Artifact guide

| Population | Canonical review inputs | Consolidated outputs |
|---|---|---|
| Training | `human_reviews/training/assignments/`, `human_reviews/training/reviews/` | `human_reviews/training/consolidated/training_conflict_type_consensus_943.jsonl`, plus CSV/JSON/Markdown agreement summaries. |
| Benchmark | `human_reviews/benchmark/first_pass/`, `human_reviews/benchmark/second_pass/` | `human_reviews/benchmark/consolidated/benchmark_preselection_consensus_1454.jsonl`, plus CSV/JSON/Markdown agreement summaries. |

The JSONL files retain per-record review representation and provenance; the
CSV files are paper-ready field-level metric tables; JSON summaries retain
counts and marginal distributions; and the existing agreement reports provide
compact narrative interpretations.

## Documentation and reporting limits

- The repository records reviewer IDs and first names in review artifacts, but
  does not contain recruitment, compensation, demographic, consent, or ethics
  approval records. These must not be invented in a paper; add them only from
  verified project records.
- The benchmark second reviewer sees the first review, so its agreement figures
  measure sequential verification consistency rather than independent blinded
  annotation reliability.
- A consensus-completed pair is explicitly distinguishable from a recorded
  first/second reviewer pair. Analyses requiring only raw interactive review
  should filter on the corresponding provenance fields before computing a
  statistic.
- Neither agreement nor reviewer confidence proves factual correctness. The
  metrics characterize agreement under this evidence-review protocol.

## Related documents

- [`TRAINING_DATASET_DESCRIPTION.md`](TRAINING_DATASET_DESCRIPTION.md): final
  training-release provenance and its relationship to the 658-record stagewise
  pool.
- [`BENCHMARK_DATASET_DESCRIPTION.md`](BENCHMARK_DATASET_DESCRIPTION.md):
  benchmark sources, retrieval, selection, refusals, and current releases.
- [`ANNOTATION_PIPELINE.md`](ANNOTATION_PIPELINE.md): committee-generated
  annotations reviewed by the training workflow.
- [Training agreement report](../human_reviews/training/consolidated/training_conflict_type_agreement_report.md)
  and [benchmark agreement report](../human_reviews/benchmark/consolidated/benchmark_preselection_agreement_report.md):
  machine-generated reporting companions.
