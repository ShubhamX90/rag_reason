# Training Dataset Description

## Purpose and canonical release

This document is the dataset card and provenance record for the repository's
current supervised-training release. The canonical files are:

| File | Records | Role |
|---|---:|---|
| `data/releases/training_dataset_v2/train.jsonl` | 862 | Canonical training split |
| `data/releases/training_dataset_v2/val.jsonl` | 81 | Canonical validation split |
| `data/releases/training_dataset_v2/train_stagewise.jsonl` | 862 | Stagewise-named companion to the training split |
| `data/releases/training_dataset_v2/val_stagewise.jsonl` | 81 | Stagewise-named companion to the validation split |

The release population is therefore **943 unique records**. It was promoted
from the sibling `sft_inference_pipeline_v2` release pack on 2026-07-29; see
`data/releases/release_manifest.json`. Earlier 609/49 internal split artifacts
are retained for reproducing the original 658-record committee run, but they
are not the canonical 862/81 release and must not be substituted for it.

The two validation files are byte-identical. The two training files contain
the same 862 IDs and all record contents match except for one Unicode line-
separator difference in one retrieved snippet (`conflictingqa_44c804eae12e`,
document `d1`). For practical use they are the same release split; use
`train.jsonl` and `val.jsonl` as the canonical public entry points.

## Task and record contract

Each JSONL record trains or evaluates evidence-grounded RAG behavior. It
contains the query, retrieved documents, evidence-level notes, a conflict
taxonomy label, a concise conflict rationale, an answerability decision, and a
grounded expected response with document citations. The current top-level
schema is:

```text
id
query
retrieved_docs
conflict_type
conflict_reason
gold_answer
per_doc_notes
answerable_under_evidence
expected_response
think
_ans_vote_tally / _ans_winner_model
_abstain_vote_tally / _abstain_winner_model
```

`retrieved_docs` uses `doc_id`, `snippet`, `source_url`, and `timestamp`.
Each `per_doc_notes` entry aligns to a document by `doc_id` and contains an
evidence verdict (`supports`, `partially supports`, or `irrelevant`), a
grounded key fact and quote where relevant, a verdict rationale, source-quality
label, and committee audit fields. `expected_response` has `answer`,
`evidence`, `abstain`, and `abstain_reason`; `think` is the retained internal
reasoning trace. Consumers should treat these records as data artifacts and
handle the `think` field under their own release/privacy policy.

The five nominal conflict labels are:

| Label | Meaning for answer generation |
|---|---|
| `No conflict` | Evidence supports one compatible conclusion. |
| `Complementary information` | Compatible evidence supplies different pieces needed for a coherent answer. |
| `Conflicting opinions or research outcomes` | Evidence expresses materially incompatible claims or findings. |
| `Conflict due to outdated information` | Differences arise because a newer state of affairs supersedes older evidence. |
| `Conflict due to misinformation` | An unreliable or misleading claim conflicts with stronger evidence in the retrieved set. |

`answerable_under_evidence` is independent of the five-way taxonomy. A record
can retain a conflict-type label while its supplied evidence is inadequate for a
responsible answer; then `expected_response.abstain=true` and its answer is
exactly `CANNOT ANSWER, INSUFFICIENT EVIDENCE`.

## Release-level statistics

### Split sizes, answerability, and labels

| Split | Records | Answerable | Abstention target |
|---|---:|---:|---:|
| Train | 862 | 590 (68.45%) | 272 (31.55%) |
| Validation | 81 | 59 (72.84%) | 22 (27.16%) |
| Total | 943 | 649 (68.82%) | 294 (31.18%) |

| Conflict type | Train | Validation | Total | Total share |
|---|---:|---:|---:|---:|
| No conflict | 315 | 28 | 343 | 36.37% |
| Complementary information | 262 | 24 | 286 | 30.33% |
| Conflicting opinions or research outcomes | 144 | 14 | 158 | 16.76% |
| Conflict due to outdated information | 123 | 13 | 136 | 14.42% |
| Conflict due to misinformation | 18 | 2 | 20 | 2.12% |

There are 493 empty `gold_answer` fields: all 294 abstention-target records and
199 answerable records. `gold_answer` is inherited source metadata and is not
the only supervision target: the generated, cited `expected_response` is
present for every release record. An empty `gold_answer` must never be treated
as permission to invent an answer.

### Evidence volume and notes

The full release contains 6,642 documents (mean 7.043 and median 5 documents
per record), with document-count distribution: 4 docs: 28 records; 5: 461; 6:
18; 7: 31; 8: 56; 9: 190; 10: 110; 11: 15; 12: 9; 13: 8; 14: 6; 15: 6; 16: 2;
19: 1; and 20: 2. Every release record has one note per retrieved document.

Across all notes, 2,623 are `supports`, 3,263 `partially supports`, and 756
`irrelevant`; 2,270 are marked high source quality and 4,372 low source
quality. These source-quality values are annotation metadata following the
pipeline prompt's URL-based policy, not an independently audited ranking of
every publisher.

## Provenance: exact record-level composition

The final 943 records are a union of three source families. This table is based
on direct ID joins to the retained source and output artifacts; all four rows
are disjoint and sum exactly to 943.

| Source family and retained join | Train | Validation | Total | Typical final behavior |
|---|---:|---:|---:|---|
| CONFLICTS normalized pool: `data/normalized/conflicts_normalized.jsonl` | 424 | 34 | 458 | 456 answerable; 2 evidence-conditioned abstentions |
| TRUST-ALIGN initial refusal pool: `data/normalized/refusals_normalized.jsonl` | 185 | 15 | 200 | Refusal target |
| TRUST-ALIGN later refusal addition: `outputs/local_committee_refusals200_3model/final/stage2_final_readonly.jsonl` | 66 | 6 | 72 | Refusal target |
| Unused benchmark-source material: current benchmark-v2 plus retained selected-800 outputs | 187 | 26 | 213 | 193 answerable; 20 evidence-conditioned abstentions |
| **Total** | **862** | **81** | **943** | **649 answerable; 294 abstention targets** |

Thus 458 records (48.57%) are sourced from CONFLICTS, 272 (28.84%) from the
TRUST-ALIGN refusal source, and 213 (22.59%) are drawn from benchmark-source
material that was not admitted to the final 736-example benchmark holdout.
This is source provenance, not the final conflict-label distribution.

The following matrix gives the final release `conflict_type` distribution within
each direct-provenance component. It is useful for detecting source/label
coupling when reporting model performance.

| Component | No conflict | Complementary | Conflicting opinions/research | Outdated | Misinformation | Total |
|---|---:|---:|---:|---:|---:|---:|
| CONFLICTS normalized pool | 161 | 115 | 115 | 62 | 5 | 458 |
| TRUST-ALIGN initial 200 | 90 | 89 | 13 | 7 | 1 | 200 |
| TRUST-ALIGN later 72 | 32 | 25 | 8 | 7 | 0 | 72 |
| Benchmark-v2 additions | 54 | 55 | 22 | 49 | 13 | 193 |
| Selected-800-only benchmark additions | 6 | 2 | 0 | 11 | 1 | 20 |
| **Full release** | **343** | **286** | **158** | **136** | **20** | **943** |

### CONFLICTS source family: 458 records

The retained raw input is `data/raw/conflicts.jsonl`, a local copy of the
CONFLICTS data from *[DRAGged into Conflicts: Detecting and Addressing
Conflicting Sources in Search-Augmented LLMs](https://arxiv.org/abs/2506.08500)*.
The pinned external-source record is `data/external_sources/source_manifest.json`
at repository commit `81ba921dd684a93db41a7e9dda6b6a7c67348a88`. CONFLICTS
already contains questions, search results, an expert conflict label, and a
correct answer field; it is not retrieved anew by the Tavily benchmark route.

The 458 raw records are composed of 162 ConflictingQA, 105 SituatedQA
geographic, 95 FreshQA, 55 QACC, and 41 SituatedQA temporal source examples.
Their supplied conflict labels are No conflict 161, Complementary 115,
Conflicting opinions **and** research outcomes 115, Outdated 62, and
Misinformation 5. The pipeline canonicalizes the one `and`/`or` wording
variant to `Conflicting opinions or research outcomes` for split construction
and final reporting.

`scripts/normalize_raw_dataset.py` performs the retained normalization:

1. assigns a record ID, extracts `question` into `query`, copies the source
   label, and maps `correct_answer` to `gold_answer`;
2. converts each source `search_results` item to a document with `d1..dN`, URL,
   merged cleaned title/snippet/short-text evidence, and normalized date;
3. removes duplicate documents by URL/text signature and rejects an input row
   that has no valid document after cleaning.

The normalized source file preserves 458 records. One record has the literal
opaque ID `rgba(1, 1, 2, 0.4)` rather than the usual numeric `#....` form; it
is preserved unchanged in the release. Code joining records must treat IDs as
opaque strings and must not assume an ID format.

### TRUST-ALIGN refusal source family: 272 records

The refusal family originates from the held-out refusal pool associated with
*[Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded
Attributions and Learning to Refuse](https://arxiv.org/abs/2409.11242)*
(TRUST-ALIGN; ICLR 2025). These examples teach/select abstention conditioned on
the supplied evidence; they are not Tavily-retrieved benchmark candidates.

The retained normalized refusal corpus has 506 five-document, empty-gold-answer
records in `data/normalized/refusals_normalized_all.jsonl`. The historical
initial training input is its 200-record normalized subset,
`data/normalized/refusals_normalized.jsonl`; every one of those 200 is in the
final release. A further 72 records, identifiable by `trust_align_` IDs, are
joined from the retained local refusal-committee output. Together they make 272
TRUST-ALIGN-source release rows (251 train, 21 validation), all with a final
abstention target.

The initial normalized 200 are five-document records and have source-level
labels No conflict 120, Complementary 71, Outdated 6, Misinformation 1, and
Conflicting opinions/research outcomes 2. These source labels should not be
confused with final committee labels: refusal-mode Stage 2 independently
reasons over the evidence and every refusal-mode Stage 3 response is forced to
abstain.

### Benchmark-source additions: 213 records, all outside the 736 holdout

The release also reuses high-quality material from the benchmark construction
stream without contaminating the final benchmark evaluation set. ID joins show
that 193 training/validation records occur in the broader 933-record
`benchmark_final_v2.jsonl` but **zero** occur in
`benchmark_final_v2_holdout_clean_736.jsonl`. The remaining 20 occur in the
retained selected-800 non-refusal stagewise output but were not carried into
the current benchmark-v2 release. All 213 are therefore outside the final
736-record holdout; there is also zero train/validation ID overlap.

Of these 213 benchmark-source records, 193 are answerable and 20 are abstention
targets after evidence assessment. The latter began in the non-refusal
benchmark-selection stream, so their source family should not be mistaken for
TRUST-ALIGN merely because their final expected response abstains. Their
source-prefix mix is ConflictingQA 36, FreshQA 44, QACC 51, SituatedQA 48
(40 temporal, 8 geographic), WikiRevision 28, and HotpotQA 6. The 20 retained
selected-800-only additions are FreshQA 14, SituatedQA temporal 4, and QACC 2.

This partition demonstrates a key leakage safeguard: a query need not be
discarded merely because it was considered during broader benchmark building,
but no record in the released train/validation set shares an ID with the final
736-record benchmark holdout.

### Selection, quality gates, and what human review did—and did not—select

There is no single post-review selector in the retained repository that takes
all 943 rows and emits the canonical 862/81 release. The release manifest
records promotion from the sibling release pack, while the ID joins above
reconstruct its membership. This means the documentation can state exact
membership and the quality gates whose artifacts remain, but must not invent a
universal threshold or claim that every released row passed the same interactive
selection procedure.

The retained, source-specific selection evidence is as follows.

1. **CONFLICTS base (458) and initial TRUST-ALIGN base (200):** these form the
   original 658-row stagewise annotation pool. Their retained training human
   review is a conflict-label validation pass, not a keep/drop preselection
   workflow. Every one of the 658 was assigned to two reviewers; neither the
   review CLI nor the consolidation script contains a rule that removes an
   item from the source 658 pool.
2. **Benchmark-source additions that were reviewed in the selected-800 flow:**
   175 of the 213 benchmark-source additions occur in the retained
   `benchmark_non_refusal_selected_800.jsonl` artifact—155 later answerable
   benchmark-v2 records and 20 selected-800-only records that ultimately have
   an abstention target. That selected-800 flow retains every non-`No conflict`
   item and chooses `No conflict` records only from a strict top-tier predicate:
   accepted; high review confidence; good retrieval quality; sufficient
   evidence; clear conflict assessment; specific query; strong source
   reliability; 4–6 relevant documents; and a possible gold answer. The
   selected `No conflict` quota is deterministically balanced across source
   availability. See the benchmark dataset card for the full 800-record
   selection derivation: [`BENCHMARK_DATASET_DESCRIPTION.md`](BENCHMARK_DATASET_DESCRIPTION.md).
3. **Other benchmark-v2 additions (38):** these occur in the broader current
   benchmark-v2 release but not in the retained selected-800 artifact. The
   current repository has no per-record final-selection rubric or assembly
   script that explains their later promotion to the training release. They are
   included here because the release file contains them; their provenance is
   explicit, while an unrecorded quality threshold is not asserted.
4. **Later TRUST-ALIGN additions (72):** these are retained in the local
   200-record refusal-committee output and have a refusal target. The current
   repository preserves their output lineage but not a separate release
   selection script that reduces that output to these 72 training IDs.

This is the appropriate scientific interpretation of “selection after human
review” for the training release: human label review directly covers the 658
base rows; a documented human quality-selection predicate exists for the 175
benchmark-stream additions that pass through the selected-800 artifact; and
the remaining promoted additions have traceable source membership but no
retained universal selection rule. The absence of a retained selector is itself
important provenance information.

## Annotation lineage and mixed provenance

The release is intentionally richer than the original 658-record annotation
run. `data/final_annotations/stagewise_multi/` contains the retained Stage 1,
Stage 2, and Stage 3 outputs for the base 658 records (458 CONFLICTS + 200
initial TRUST-ALIGN refusals). `data/splits/92p5_7p5/` is a seed-21 internal
609/49 stratified split of exactly this 658-row artifact. It stratifies by
`(refusal | nonrefusal) × normalized conflict type`; it is reproducible
pipeline evidence, not the later public 862/81 split.

The original stagewise protocol is:

1. **Stage 1 — evidence adjudication.** Every query/document pair is assessed
   independently as supports, partially supports, or irrelevant, with a
   grounded fact, quote, reason, and URL-based source-quality field.
2. **Stage 2 — conflict reasoning and answerability.** For CONFLICTS rows, the
   supplied conflict type is retained while the committee votes on whether the
   evidence is answerable. For refusal rows, the committee determines the
   conflict type and the refusal path requires non-answerability.
3. **Stage 3 — expected response.** The committee votes on abstention. The
   winning side contributes a cited, evidence-grounded answer or the mandated
   `CANNOT ANSWER, INSUFFICIENT EVIDENCE` response, together with the retained
   internal trace.

The base artifacts retain a historical committee provenance. Their record-level
vote fields show an OpenRouter committee including Claude Sonnet 4.6, GPT-5.4,
Qwen 3.5-27B, DeepSeek V3.2, and Grok 4.1 Fast; the historical weights recorded
in the corresponding code revision were 0.30, 0.25, 0.20, 0.15, and 0.10.
The source tree has since evolved to current local-committee configurations, so
the current defaults must not be retroactively claimed as the generator of all
released training records.

The 285 later additions inherit benchmark/committee artifacts with mixed audit
provenance. For example, local benchmark records expose local Qwen/DeepSeek/
Mistral vote metadata, while some promoted benchmark-v2 records carry separate
run metadata. The uniform release schema makes the records usable together, but
the provenance fields show they are not a single model run. This distinction is
important for reproducibility and is why exact source/output paths are named
throughout this document.

Detailed current committee configuration, prompt text, weighted-voting logic,
cache behavior, and execution orchestration are documented separately in the
forthcoming annotation-pipeline documentation. The present card records the
dataset-facing lineage only.

## Human conflict-type review

The interactive training review was performed on the original 658-record
stagewise pool. `scripts/training_conflict_type_review_cli.py` assigned every
item to exactly two of three reviewers (seed 62058), yielding 1,316 assignments:
reviewers 1 and 2 received 439 each and reviewer 3 received 438. Pair counts
are 220 for reviewers 1–2, 219 for 1–3, and 219 for 2–3. Reviewers inspected
the query, retrieved evidence, and committee-assigned conflict label; they
could accept it or change it, record confidence, and give a note.

The final released 943-record population is consolidated in
`human_reviews/training/consolidated/training_conflict_type_consensus_943.jsonl`.
Its provenance field distinguishes the 658 rows with recorded two-reviewer
files from the 285 release additions represented through consensus-completion
records. This distinction is preserved in the data and must not be erased when
reporting the review workflow.

The consolidated review file is an audit/reliability artifact. The retained
consolidation code does **not** rewrite `conflict_type` in `train.jsonl` or
`val.jsonl` from a reviewer consensus label; it stores the original release
label together with the two review judgments. Accordingly, the release-label
distribution in this dataset card and the reviewer-label distribution in the
IAA report are different quantities and should never be silently substituted
for one another.

The primary conflict-type reliability statistic over the consolidated population
is 83.46% raw agreement and Cohen's κ = 0.7694 for the nominal five-way
`reviewed_conflict_type` field. The full calculation, field glossary, label-
change diagnostics, formulas, and machine-readable metrics are in
`human_reviews/training/consolidated/training_conflict_type_agreement_report.md`.

## Split integrity and recommended use

- Use `train.jsonl` for fitting and `val.jsonl` for model selection. They have
  disjoint IDs (862 + 81 = 943).
- Do not use the 736-record benchmark holdout for training, tuning, retrieval
  selection, or prompt selection. Direct ID overlap with the released train and
  validation files is zero.
- Preserve `id` as an opaque string. Do not derive source or split membership
  from an ID pattern alone; use the documented source joins where provenance is
  required.
- Evaluate abstention separately from answer quality. `answerable_under_evidence`
  and `expected_response.abstain` identify the intended behavior.
- The 609/49 internal split under `data/splits/92p5_7p5/` is useful for
  reproducing the base committee-validation workflow only. It is not a
  replacement for the public release split.

## Limitations and reporting guidance

The dataset is evidence-conditioned. A label or answer characterizes the
retrieved snippets stored with the record, not unrestricted world truth; source
pages and retrieval conditions can change over time. The conflict taxonomy and
source-quality fields are annotation outputs and should be reported with their
human-review statistics and source provenance.

The release mixes externally sourced conflict examples, externally sourced
refusal examples, and benchmark-stream additions. Report the 458/272/213 source
composition and the 649/294 answerability composition when publishing aggregate
training results. The misinformation stratum has only 20 records, so metrics
for that label should include counts and uncertainty rather than be interpreted
as a large-sample estimate.

## Primary pointers

- Canonical release: `data/releases/training_dataset_v2/`
- Release manifest: `data/releases/release_manifest.json`
- CONFLICTS raw and normalized sources: `data/raw/conflicts.jsonl` and
  `data/normalized/conflicts_normalized.jsonl`
- TRUST-ALIGN normalized refusal sources: `data/normalized/refusals_normalized*.jsonl`
- Base stagewise outputs: `data/final_annotations/stagewise_multi/`
- Internal 658-row split: `data/splits/92p5_7p5/`
- Training review and IAA: `human_reviews/training/`
- Benchmark dataset card: `docs/BENCHMARK_DATASET_DESCRIPTION.md`
