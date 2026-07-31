# Data Format and Schema Reference

## Purpose and scope

This is the repository-wide field reference for reading the released JSONL datasets, retained stagewise outputs, and human-review consolidation files. It complements the dataset cards: those documents explain provenance and statistics, while this document defines how to safely load, join, and interpret the stored records.

The reference covers the current artifacts only:

| Artifact family | Primary files | Role |
|---|---|---|
| Training release | data/releases/training_dataset_v2/train.jsonl and val.jsonl | Canonical 862/81 supervised splits |
| Benchmark release | data/releases/benchmark_dataset_v2/benchmark_final_v2_holdout_clean_736.jsonl | Canonical 736-record held-out evaluation set |
| Broader benchmark pack | data/releases/benchmark_dataset_v2/benchmark_final_v2.jsonl | Current 933-record release pack; the holdout is its exact ID subset |
| Historical stagewise output | data/final_annotations/stagewise_multi/ | Retained 658-record Stage 1, 2, and 3 pipeline artifact |
| Training review consolidation | human_reviews/training/consolidated/training_conflict_type_consensus_943.jsonl | Review/reporting layer for the 943-record release population |
| Benchmark review consolidation | human_reviews/benchmark/consolidated/benchmark_preselection_consensus_1454.jsonl | Preselection/review layer for the 1,454-record population |

This is not a claim that every field is present in every file. In particular, the benchmark holdout retains the evidence and Stage-2-style decision fields but does not carry the training release's Stage-3 expected-response object. Consumers must select a schema appropriate to the artifact they are loading.

## JSONL conventions

Every current JSONL file is UTF-8 and consists of one JSON object per physical line. A consumer should stream records rather than assume that a specific record occupies a specific line number; record identity is carried by id.

The recommended join key is the exact id string. Do not coerce IDs to integers, strip prefixes, or infer source membership from their visual form. IDs are stable within the documented release lineage, while record order is not a semantic contract.

Unless a file-specific document states otherwise:

- JSON objects may contain additional underscore-prefixed provenance or audit fields.
- A missing optional field and an empty string are distinct states.
- Empty gold_answer, answer, evidence, quote, or abstain_reason fields should be interpreted in the context of answerability and the stage that emitted the record, not converted automatically to null.
- A list may be empty only when that is valid for its workflow; do not invent documents, notes, or evidence entries during loading.

## Canonical conflict taxonomy

The string in conflict_type is the canonical task label. It is the field to use for training, evaluation, filtering, and paper statistics.

| Canonical string label | Meaning |
|---|---|
| No conflict | Retrieved evidence supports a compatible core answer. |
| Complementary information | Multiple documents contribute compatible, non-identical facts needed for the response. |
| Conflicting opinions or research outcomes | Credible sources or interpretations make substantively incompatible claims. |
| Conflict due to outdated information | The apparent disagreement is explained by temporal change, updating, or supersession. |
| Conflict due to misinformation | A misleading or unreliable claim is present and should not determine the answer. |

A numeric conflict_category_id occurs in the benchmark release and in some origin-specific training records. The active selection and review mapping is:

| Numeric value | Canonical conflict type |
|---:|---|
| 1 | No conflict |
| 2 | Complementary information |
| 3 | Conflicting opinions or research outcomes |
| 4 | Conflict due to outdated information |
| 5 | Conflict due to misinformation |
| -1 | Refusal-track marker, not a sixth conflict label |

The numeric value is auxiliary lineage metadata, not the primary supervision target. A refusal record has answerable_under_evidence=false and conflict_category_id=-1 while retaining a conflict_type that describes the evidence configuration. Some retained records also preserve source-stage numeric metadata after later label processing. Therefore, use conflict_type as authoritative whenever a numeric value and string label are not perfectly aligned.

## Shared dataset-record schema

The following fields occur in both canonical training/validation release files and the 736-record benchmark holdout.

| Field | Type | Required | Meaning and safe use |
|---|---|---:|---|
| id | string | Yes | Stable record identifier and join key. |
| query | string | Yes | User question shown to the evidence-grounded system. |
| retrieved_docs | array of document objects | Yes | Evidence supplied to the model. Do not supplement it with outside retrieval when evaluating evidence-grounded behavior. |
| conflict_type | string | Yes | One of the five canonical labels above; primary conflict target. |
| conflict_reason | string | Yes | Set-level explanation of why the conflict label applies. |
| gold_answer | string | Yes | Evidence-grounded target answer when applicable. It is intentionally empty for many refusal/insufficient-evidence records. |
| per_doc_notes | array of note objects | Yes | Document-level evidence annotations, matched to retrieved_docs by doc_id. |
| answerable_under_evidence | boolean | Yes | Whether the supplied retrieved_docs justify a grounded answer. It assesses evidence sufficiency, not general world knowledge. |

The two arrays are aligned by document identifier, not by a promise that a particular array position is semantically meaningful. In the canonical releases, every document note has a matching retrieved document and document IDs are normalized within a record as d1 through dN. A validator should check this identifier relation and should not rely solely on parallel array indices.

### Retrieved-document object

Each retrieved_docs item uses this normalized shape:

| Field | Type | Meaning |
|---|---|---|
| doc_id | string | Within-record document identifier, normally d1 through dN. |
| snippet | string | Retrieved text exposed to the annotation process. It is the evidence to evaluate, not a guarantee of a complete source document. |
| source_url | string | Source URL retained from retrieval or source preparation. |
| timestamp | string | Retrieval/source time metadata when available; it may be empty or non-comparable across source families. |

A document's rank is not represented as a universal field in the canonical release schema. The order in retrieved_docs reflects the retained evidence packet, but retrieval-rank research should use the retrieval methodology and retained retrieval artifacts rather than infer rank from a later release file.

### Per-document note object

Each per_doc_notes item is associated with a retrieved document through the same doc_id.

| Field | Type | Meaning |
|---|---|---|
| doc_id | string | Must match one retrieved_docs identifier in the same record. |
| verdict | string | One of supports, partially supports, or irrelevant. |
| key_fact | string | Concise statement of the evidence contribution identified by the annotation process. |
| quote | string | Short snippet-grounded quote supporting the note. |
| verdict_reason | string | Rationale for the verdict. |
| source_quality | string | Coarse evidence-quality signal used in the annotation process. |
| _vote_tally | object | Per-verdict weighted committee tally retained for audit. |
| _winner_model | string | Highest-weight model on the selected verdict side. |
| _all_verdicts | object or array-like provenance object | Individual model verdict information retained by the run. |

A small number of training notes include _parse_error or _validation_errors in addition to the normal fields. These are retained diagnostics. They should not be silently removed or treated as a sixth verdict value.

## Answerability and final-response fields

Answerability and refusal behavior are separate concepts.

| Field | Interpretation |
|---|---|
| answerable_under_evidence | Whether the displayed evidence set contains enough material to support an answer. |
| expected_response.abstain | Whether the final response selected by Stage 3 abstains. |
| gold_answer | Reference answer when an answer is appropriate; often empty when a refusal is required. |

The training release includes the Stage-3 output object expected_response and a reasoning field think. The 736-record benchmark holdout does not include these fields, so an evaluation pipeline must not assume their existence for benchmark records.

### Training-release Stage-3 object

Every current training/validation release record has:

| expected_response field | Type | Meaning |
|---|---|---|
| abstain | boolean | Selected answer-versus-abstain decision. |
| abstain_reason | string | Explanation for abstaining when abstain is true; may be empty otherwise. |
| answer | string | Grounded response text when abstain is false; may be empty for a refusal. |
| evidence | array | Evidence references or evidence text retained with the answer bundle. |

think is the retained Stage-3 reasoning text. It is part of the released training record contract but should be handled according to the research or model-training policy governing its use. It is not present in the benchmark holdout schema.

## Committee audit and lineage fields

Underscore-prefixed fields preserve run-specific provenance. They are useful for auditing a retained artifact but should not be assumed to form a stable, universal public schema.

| Field pattern | Where it appears | Meaning |
|---|---|---|
| _vote_tally and _winner_model | Per-document notes | Stage-1 weighted verdict decision and winning-side model. |
| _all_verdicts | Per-document notes | Individual committee outputs or their retained representation. |
| _ans_vote_tally and _ans_winner_model | Training and Stage-2/3 artifacts | Weighted answerability vote and winning-side model. |
| _abstain_vote_tally and _abstain_winner_model | Training and Stage-3 artifacts | Weighted abstention vote and winning-side model. |
| _ct_vote_tally and _ct_winner_model | Some benchmark/refusal-derived training records and Stage-2 artifacts | Weighted conflict-type vote and winner. |
| _gold_conflict_type | Benchmark/refusal stage inputs or derivatives | Input conflict type retained before a Stage-2 conflict-type reclassification. |
| _run_j_origin | Some training records | Historical release-assembly provenance. |
| _parse_error and _validation_errors | Small number of retained notes | Parsing/validation diagnostics from the original run. |

For a minimal supervised-data loader, retain the semantic fields and either preserve unknown underscore fields losslessly or drop them deliberately after recording that decision. Do not reconstruct a committee vote from the winner field alone, and do not treat a winner model as an accuracy label.

## File-specific schema differences

### Training and validation release

The canonical training files are train.jsonl and val.jsonl. They share the common record schema plus expected_response, think, answerability audit fields, abstention audit fields, and origin-specific optional metadata.

Not every training row has the same optional key set because the final 943-record release combines several documented source families and retained pipeline lineages. Code that requires identical keys across all rows should use a schema projection rather than rejecting valid records. The stable semantic projection is:

~~~text
id, query, retrieved_docs, per_doc_notes,
conflict_type, conflict_reason, answerable_under_evidence,
gold_answer, expected_response, think
~~~

The final two fields are stable in the current training/validation release but are not shared by the benchmark holdout.

### Benchmark release

The canonical benchmark holdout and broader 933-record release pack share this stable projection:

~~~text
id, query, retrieved_docs, per_doc_notes,
conflict_type, conflict_category_id, conflict_reason,
answerable_under_evidence, gold_answer
~~~

The release benchmark is an evidence-structure and answerability resource; it does not expose the current training release's Stage-3 response bundle. The 736-record holdout contains five retrieved documents per record in the retained release, whereas the training release has variable evidence-set size. That count is a property of these current artifacts, not a general schema guarantee.

### Historical stagewise artifact

The historical 658-record artifact has one file per completed stage:

| File | Added content relative to its input |
|---|---|
| data/final_annotations/stagewise_multi/stage1.jsonl | per_doc_notes and document-level committee audit data |
| data/final_annotations/stagewise_multi/stage2.jsonl | answerable_under_evidence, conflict_reason, and answerability audit data |
| data/final_annotations/stagewise_multi/stage3_final.jsonl | expected_response, think, and abstention audit data |

It is useful for following the retained pipeline state transitions, but it is not the complete 943-record training release.

## Human-review consolidation schema

The human-review consolidation files are reporting and audit artifacts, not replacements for the canonical dataset records. They join to release or preselection populations through id and contain a reduced query field, reviewer-side objects, and explicit provenance.

### Training-review consolidation

Each row in training_conflict_type_consensus_943.jsonl contains:

| Field | Meaning |
|---|---|
| id and query | Join key and review-facing question text. |
| split | Release split membership, train or val. |
| original_conflict_type | Committee/release label presented for review context. |
| first_review and second_review | Parallel review-side objects. |
| review_provenance | How the pair representation was recorded or completed. |

Each review-side object has reviewed_conflict_type as its primary label, plus label_action, changed_label, review_confidence, change_reason, reviewer_notes, original-label fields, reviewer metadata, paired_reviewer_id, and review_source.

The two review-provenance values are recorded_two_reviewer_training_review for the 658 records with two recorded interactive reviews, and consensus_completed_training_review for the remaining 285 records represented in the full release-population consolidation. Keep that provenance in any analysis and do not overwrite the canonical dataset conflict_type from a consolidation file without an explicit reconciliation rule.

### Benchmark-review consolidation

Each row in benchmark_preselection_consensus_1454.jsonl contains:

| Field | Meaning |
|---|---|
| id and query | Join key and review-facing question text. |
| review_stratum | One of preselection, answerable_consensus_completion, or refusal_quality. |
| first_review and second_review | Common preselection-review objects. |
| review_provenance | Recorded versus consensus-completed review-side lineage. |
| final_dataset_membership | final_holdout or preselection_pool_only. |

Each review-side object records the preselection decision, preliminary conflict type, confidence, retrieval quality, evidence sufficiency, conflict clarity, query specificity, source reliability, relevant-document count bin, gold-answer feasibility, optional human answer, rejection rationale, reviewer metadata, and review-source metadata. The second-review object also includes second_review_action.

The exact raw/consensus provenance labels are part of the evidence needed to interpret the agreement reports. Consult the human-review methodology before treating a pair as an independent blind annotation pair.

## Safe loading and validation checklist

Before using a file in training, analysis, or evaluation:

1. Select the canonical path for the intended task; do not substitute the historical 658-row or internal 1,000-row artifacts for a release dataset.
2. Parse every non-empty physical line as one JSON object.
3. Verify unique id values within the selected file.
4. Validate conflict_type against the five canonical strings.
5. Validate answerable_under_evidence as a boolean, not the strings True or False.
6. For every record, compare the set of retrieved_docs.doc_id values with per_doc_notes.doc_id values.
7. If consuming the training release, validate the four expected_response keys; if consuming the benchmark holdout, do not require expected_response or think.
8. Preserve review_provenance, review_stratum, and final_dataset_membership when analyzing consolidated review data.
9. Treat underscore-prefixed keys as optional provenance; retain them where auditability matters.
10. Report which exact file and record count were used.

## Related documentation

- [Repository overview and release boundaries](../README.md)
- [Released-data entry point](../data/releases/README.md)
- [Training dataset description](TRAINING_DATASET_DESCRIPTION.md)
- [Benchmark dataset description](BENCHMARK_DATASET_DESCRIPTION.md)
- [Annotation pipeline](ANNOTATION_PIPELINE.md)
- [Human-review methodology](HUMAN_REVIEW_METHODOLOGY.md)

