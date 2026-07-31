# Benchmark Dataset Description

## Purpose and canonical files

This document is the dataset card and construction record for the repository's
current benchmark release. It is deliberately explicit about the distinction
between (1) the retained internal construction artifact and (2) the canonical
reviewer-facing release. They are related, but they are not interchangeable
files and should not be reported as though they have the same size.

| Layer | Canonical path | Records | Intended use |
|---|---|---:|---|
| Full sanitized release | `data/releases/benchmark_dataset_v2/benchmark_final_v2.jsonl` | 933 | Complete current benchmark release |
| Reviewer-facing holdout | `data/releases/benchmark_dataset_v2/benchmark_final_v2_holdout_clean_736.jsonl` | 736 | Main evaluation/inspection set |
| Retained internal build artifact | `data/benchmarks/final_benchmark_2026-06-22/benchmark_final.jsonl` | 1,000 | Reproducing the retained benchmark-build run |

The 736-record holdout is an exact ID subset of the 933-record release (736
shared IDs; no holdout-only IDs). The full release and holdout were promoted
into this repository on 2026-07-29; their release manifest is
`data/releases/release_manifest.json`. The 1,000-record artifact was created
on 2026-06-22 and records the earlier retained stagewise build run. Do not use
the internal 1,000-row artifact as a substitute for the current release.

The benchmark evaluates whether a RAG system can reason over a query together
with retrieved evidence that may be mutually supportive, incomplete,
temporally outdated, misleading, or in genuine conflict. The task is not to
infer a label from the source dataset from which a query was drawn: source
membership supplies queries, whereas conflict labels are determined after
retrieval and review of the actual evidence shown for the query.

## What one benchmark record contains

Both release JSONL files use the same normalized top-level schema:

```text
id
query
retrieved_docs
conflict_type
conflict_reason
gold_answer
per_doc_notes
answerable_under_evidence
conflict_category_id
```

`retrieved_docs` contains only `doc_id`, `snippet`, `source_url`, and
`timestamp`. `per_doc_notes` is aligned one-to-one with those documents by
`doc_id` and contains `verdict`, `key_fact`, `quote`, `verdict_reason`,
`source_quality`, `_vote_tally`, `_winner_model`, and `_all_verdicts`.
`doc_id` is canonicalized as `d1` through `dN` within each record. Validation
of both release files finds no document/note ID mismatches.

The labels are:

| ID | `conflict_type` | Interpretation |
|---:|---|---|
| 1 | `No conflict` | Retrieved evidence supports one compatible answer. |
| 2 | `Complementary information` | Documents provide compatible but non-identical facts that must be combined. |
| 3 | `Conflicting opinions or research outcomes` | Credible evidence expresses substantively incompatible claims, findings, or interpretations. |
| 4 | `Conflict due to outdated information` | Evidence differs because one account is temporally stale relative to another. |
| 5 | `Conflict due to misinformation` | Evidence includes a misleading or unreliable claim that should not determine the answer. |
| -1 | Refusal track | Evidence is insufficient to responsibly answer; `answerable_under_evidence` is false. |

For answerable records, `gold_answer` is the evidence-grounded target answer.
For abstention/refusal examples it is intentionally empty; the correct behavior
is to state that the evidence is insufficient rather than fabricate an answer.
`conflict_reason` records why the label applies, and document notes expose the
evidence-level rationale used by the stagewise annotation process.

## Current release statistics

### Full 933-record release

| Conflict type | Count | Share |
|---|---:|---:|
| No conflict | 265 | 28.40% |
| Complementary information | 276 | 29.58% |
| Conflict due to outdated information | 210 | 22.51% |
| Conflicting opinions or research outcomes | 132 | 14.15% |
| Conflict due to misinformation | 50 | 5.36% |

These five `conflict_type` values partition all 933 records. Answerability is a
separate field: a refusal record retains the conflict type of the evidence
configuration from which it was constructed but has `conflict_category_id=-1`
and `answerable_under_evidence=false`.

| Answerability status | Count | Share |
|---|---:|---:|
| Answerable under supplied evidence | 805 | 86.28% |
| Refusal / insufficient evidence | 128 | 13.72% |

Within the 805 answerable records, the label counts are Complementary 231, No
conflict 208, Outdated 197, Conflicting opinions/research outcomes 119, and
Misinformation 50. The 128 refusals retain these source-evidence labels as No
conflict 57, Complementary 45, Outdated 13, and Conflicting opinions/research
outcomes 13. The release contains 4,700 retrieved documents (mean 5.038 and
median 5 documents/record). Document counts are: 2 docs: 2 records; 3: 1; 4:
108; 5: 786; 6: 3; 7: 2; 8: 5; 9: 1; 10: 25. There are 188 empty gold-answer
fields: all 128 refusals and 60 answerable records. Evaluation code must not
silently convert an empty field into a target answer.

Across the 4,700 document notes, 2,771 are marked high source quality and
1,929 low source quality. Evidence verdicts are 1,633 `supports`, 2,698
`partially supports`, and 369 `irrelevant`.

### Reviewer-facing 736-record holdout

| Conflict type | Count | Share |
|---|---:|---:|
| No conflict | 211 | 28.67% |
| Complementary information | 221 | 30.03% |
| Conflict due to outdated information | 158 | 21.47% |
| Conflicting opinions or research outcomes | 109 | 14.81% |
| Conflict due to misinformation | 37 | 5.03% |

As in the full release, these label counts partition the holdout while
answerability is independent:

| Answerability status | Count | Share |
|---|---:|---:|
| Answerable under supplied evidence | 608 | 82.61% |
| Refusal / insufficient evidence | 128 | 17.39% |

Among its answerable records, labels are Complementary 176, No conflict 154,
Outdated 145, Conflicting opinions/research outcomes 96, and Misinformation 37.
The 128 refusal records retain labels No conflict 57, Complementary 45, Outdated
13, and Conflicting opinions/research outcomes 13. The holdout has 3,701
retrieved documents (mean 5.029; median 5) and 188 empty gold answers (all 128
refusals plus 60 answerable records). Its document counts are 2: 2 records; 3:
1; 4: 78; 5: 631; 6: 1; 7: 1; 8: 3; and 10: 19. It has 2,119 high-quality and
1,582 low-quality document notes, with 1,225 `supports`, 2,168 `partially
supports`, and 308 `irrelevant` verdicts.

The release preserves several query-origin prefixes. In the 736 holdout these
are QACC (145), SituatedQA (141), ConflictingQA (137), `trust_align_` / the
TRUST-ALIGN refusal source (128), FreshQA (85), WikiRevision (78), HotpotQA
(19), HealthContradict (2), and Misinformation (1). Prefixes are provenance
indicators, not labels and not a substitute for inspection of evidence or final
annotations.

| Query-origin prefix | Full 933-record release | 736-record holdout |
|---|---:|---:|
| QACC | 196 | 145 |
| SituatedQA | 185 | 141 |
| ConflictingQA | 173 | 137 |
| `trust_align_` (TRUST-ALIGN refusals) | 128 | 128 |
| FreshQA | 115 | 85 |
| WikiRevision | 108 | 78 |
| HotpotQA | 25 | 19 |
| HealthContradict | 2 | 2 |
| Misinformation | 1 | 1 |
| **Total** | **933** | **736** |

## Source queries and candidate-pool construction

The reproducible query-source map is
`data/external_sources/source_manifest.json`, which pins the exact repository
commits or downloaded files. The current builder is
`scripts/build_conflicts_benchmark_candidates.py`.

| Source | Pinned input | Role in this build |
|---|---|---|
| CONFLICTS | `rag_conflicts` commit `81ba921...`; 458 questions | Exclusion set, preventing reuse of retained CONFLICTS/training questions. |
| FreshQA | commit `7d2d368...`; 600-question exported CSV (500 test, 100 dev) | Candidate query source. The CSV has two leading warning/blank lines; the parser locates the header row. |
| SituatedQA | commit `1ef854...`; geographic and temporal train/dev/test JSONL files | Candidate query source. The builder uses ID-aware deduplication because some variants repeat IDs. |
| QACC / ConflictQA | commit `fff5a4...`; 1,617 questions (394 train, 410 dev, 813 test) | Candidate query source. |
| ConflictingQA retrieval windows | RAG convincingness commit `001b6d...`; 4,002 rows, 434 unique queries | Candidate-query/retrieval reference. It is a retrieval-window table, not a conventional QA file. |

The source manifest retains the following raw-file inventory so that source
coverage is auditable before the builder's query-level deduplication:

| Source component | Raw rows | Distinct IDs when recorded |
|---|---:|---:|
| FreshQA export | 600 | — |
| SituatedQA geographic train / dev / test | 3,548 / 1,398 / 506 | 3,548 / 351 / 129 |
| SituatedQA temporal train / dev / test | 6,009 / 3,423 / 2,795 | 2,667 / 1,178 / 876 |
| QACC / ConflictQA train / dev / test | 394 / 410 / 813 | — |
| ConflictingQA retrieval-window table | 4,002 | 434 unique `search_query` values |
| Pinned CONFLICTS exclusion list | 458 | 458 questions |

The candidate builder normalizes a query for comparison by lowercasing,
collapsing whitespace, and removing ASCII punctuation. It constructs stable
record IDs from a SHA-1 digest of source dataset, source ID, and query. It
then excludes normalized overlaps with the pinned CONFLICTS set, current
normalized conflicts data, and retained raw conflicts data; removes duplicate
normalized queries; and makes a source-balanced selection using a seeded,
least-represented-source cycle.

The retained 2,000-query pool was built with seed 17. Starting from 20,330 raw
candidate rows, 598 normalized overlaps were excluded and 8,891 duplicates
were removed, leaving 10,841 unique usable candidates. The final 2,000 query
pool has 272 ConflictingQA queries (all available) and 432 from each of
SituatedQA geographic, SituatedQA temporal, FreshQA, and QACC. This is a
balanced *candidate query* pool, not a balanced final-label distribution.

## Retrieval and evidence preparation

The retained benchmark retrieval run takes the 2,000 candidates and searches
with Tavily, requests 20 results, filters/retains a maximum of 10 documents,
then extracts a query-relevant readable window. The complete protocol,
parameters, caching behavior, limitations, and reproducibility command are in
[`TAVILY_RETRIEVAL_METHODOLOGY.md`](TAVILY_RETRIEVAL_METHODOLOGY.md).

The important retained retrieval artifacts are:

| Artifact | Records | Meaning |
|---|---:|---|
| `full2000_fresh_tavily_20_keep10_tasb_readable_raw_with_fulltext.jsonl` | 2,000 | Raw retained Tavily/search-and-fetch output with full-text fields where available. |
| `full2000_fresh_annotation_candidates_exact10.jsonl` | 1,878 | Exact-ten-document candidates used for deterministic reduced evidence preparation. |
| `full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl` | 1,878 | Five-document review/annotation view. |
| `full2000_fresh_manual100_conflict_quality_audit.md` | 100 manually audited queries | Stratified retrieval-quality audit. |

Of the 2,000 retrieved candidates, 1,878 had exactly ten retained documents;
122 had fewer than ten (118 had zero). The exact-ten set has source counts
ConflictingQA 258, SituatedQA geographic 406, SituatedQA temporal 413,
FreshQA 394, and QACC 407. The 122 underfilled records were excluded from the
strict exact-ten annotation candidate set. This avoids silently comparing
records with radically different evidence exposure.

At this retrieval gate, **1,878/2,000 queries (93.90%)** progressed to the
strict downstream evidence pipeline and **122/2,000 (6.10%)** were rejected
because fewer than ten documents could be retained. The five-document artifact
is a fixed evidence-view reduction of those 1,878 queries, not a further query
rejection. Some historical non-refusal retrieval intermediates are no longer
retained as individually named current artifacts after repository cleanup. The
project's recorded method is nevertheless uniform: non-refusal benchmark
queries were sourced first and then passed through the same Tavily search,
fetch, readable-text, and query-windowing procedure described here and in the
retrieval-methodology document.

For the five-document view, the deterministic subset script preserves two
documents selected from original ranks 1–5 and three from ranks 6–10. It uses
SHA-256 ordering keyed by seed 62002, record ID, rank bucket, document ID, and
original rank; it then restores original result order and retains original-rank
provenance. This design deliberately exposes both highly ranked and lower
ranked evidence while keeping the review display manageable.

A 100-query stratified manual audit (1,000 snippets) sampled all five query
sources and only exact-ten examples. Eighty-eight queries were judged usable;
82 had high or medium-high confidence and 12 were rejected for low retrieval
quality. Geographic SituatedQA queries were the prominent failure mode when a
query was underspecified or the retrieved locale was wrong. This audit is a
retrieval-quality check, not a conflict-labeling gold standard.

## Human preselection and quality selection

The benchmark human-preselection corpus contains **1,454 unique records** and
is the quality-selection population from which the current **736-record
reviewer-facing holdout** was selected. The consolidated file is
`human_reviews/benchmark/consolidated/benchmark_preselection_consensus_1454.jsonl`;
it preserves a common first/second-review schema and `review_provenance` for
each record. Its companion agreement report gives the full reliability analysis
over the 1,454-record selection population (five-way conflict-type raw
agreement 94.77%; Cohen's κ = 0.9217).

Human preselection is documented operationally in
`human_reviews/benchmark/README.md`. The CLI shows the query and retrieved
snippets and asks reviewers to record an acceptance decision, preliminary
conflict type, confidence, retrieval quality, evidence sufficiency, conflict
clarity, query specificity, source reliability, relevant-document count,
gold-answer feasibility, and notes where needed. The consolidated records retain
their provenance, including consensus-completion entries, so the quality-
selection population remains auditable rather than erasing distinct review
histories.

Canonical first-pass inputs are the cleaned JSONL files under
`human_reviews/benchmark/first_pass/reviews/`. The selection artifact is
`human_reviews/benchmark/first_pass/benchmark_selection_final/benchmark_non_refusal_selected_800.jsonl`.
It contains 800 selected non-refusal records:

| Label | Selected count |
|---|---:|
| Complementary information | 231 |
| Conflict due to misinformation | 52 |
| Conflict due to outdated information | 127 |
| Conflicting opinions or research outcomes | 145 |
| No conflict | 245 |

All 555 selected records whose human label was not `No conflict` are retained.
For `No conflict`, 245 are selected from 351 strict top-tier candidates. The
strict predicate requires: accepted; high review confidence; good retrieval
quality; sufficient evidence; clear conflict assessment; specific query; strong
source quality; 4–6 relevant documents; and a possible gold answer. Those
245 are apportioned across sources in proportion to strict-top-tier
availability, with deterministic SHA-256 ordering to make ties reproducible.
The strict `No conflict` pool/selected quota is QACC 133/93, FreshQA 93/65,
ConflictingQA 22/15, SituatedQA temporal 98/68, and SituatedQA geographic 5/4.
The selected 800-source mix is ConflictingQA 221, QACC 193, FreshQA 171,
SituatedQA temporal 165, and SituatedQA geographic 50.

The retained 800-row non-refusal selection is an internal construction-stage
artifact. The current release decision is made over the broader 1,454-record
human-preselection corpus: the 736 holdout is the final quality-selected
evaluation subset. Its selection rationale is therefore the bundle of review
dimensions above—acceptance, confidence, retrieval quality, evidence
sufficiency, clarity, query specificity, source reliability, relevant evidence,
and answerability—not an arbitrary random sampling step.

The final selected 800 received second-pass review assignments under
`human_reviews/benchmark/second_pass/assignments/`; the assignment manifest
uses seed 62002 and assigns every selected ID exactly once across seven
reviewers. Consolidated review artifacts and agreement reports live under
`human_reviews/benchmark/consolidated/`. They are human-study documentation,
not input-label replacements for the released JSONL schema.

## Refusal/insufficient-evidence track

Refusal examples are a separate benchmark track. They assess whether a system
abstains when the supplied evidence does not warrant an answer; they should not
be evaluated as ordinary answerable conflict examples. Current release refusals
are **not** produced by the Tavily retrieval path used for non-refusal benchmark
queries. They originate in the held-out refusal pool of
[TRUST-ALIGN](https://arxiv.org/abs/2409.11242), a RAG trustworthiness resource
for grounded attribution and learning to refuse. The project also draws its
training-side refusal population from that upstream resource, but released
splits are disjoint: the 128 benchmark-refusal IDs and 294 training/validation-
refusal IDs have intersection zero.

The retained refusal-selection artifacts are in
`data/benchmarks/fresh_refusals_selection_2026-06-21/`. The ranked pool has 306
unique queries and records selection signals such as average/max lexical
overlap, snippet length, support-shape summaries, a prior committee-refusal
flag, and manual exclusions. Its preselection label mix is 174 No conflict, 112
Complementary, 11 Outdated, 5 Misinformation, and 4 Conflicting. The curated
strict subset is `refusals_200_fresh_high_quality_strict.jsonl`: 200 records,
each with five evidence documents, label mix 113 No conflict, 75 Complementary,
8 Outdated, 3 Misinformation, and 1 Conflicting. All 30 manually excluded ranked
records are absent from that strict subset.

The retained internal 1,000-row build artifact combined the 800 non-refusal
stagewise records with these 200 refusal records. Refusal normalization is
implemented by `scripts/prepare_refusal_benchmark_stagewise_input.py`. Its
Stage 2/3 refusal prompting fixes `answerable_under_evidence` to false and
requires an abstention response grounded in the inadequacy of the supplied
evidence, even when a snippet is superficially on topic. This is intentionally
more stringent than merely checking topical overlap.

The current 933/736 releases contain 128 refusal records with IDs prefixed
`trust_align_`. Do not infer that they are the same 200 IDs in the earlier
internal artifact without an explicit ID comparison. The 200-row files preserve
the historical construction path; the 128 held-out TRUST-ALIGN rows in the
release files define the current refusal evaluation set.

## Stagewise committee annotation and validation

After evidence preparation and human preselection, the current pipeline uses a
stagewise multi-model committee. `scripts/run_benchmark_stagewise.py` prepares
inputs, validates them, runs Stage 1 document-level assessment, runs benchmark
Stage 2 conflict classification, and validates outputs. Operational committee
configs are in `configs/local_committee/`; prompts are in `prompts/`; Slurm
launchers are in `slurm/sharanga/local_committee/`.

In benchmark mode, Stage 1 annotates each document as `supports`, `partially
supports`, or `irrelevant` and assigns source quality. The prompt explicitly
preserves evidence conflict: a document that is conflicting, outdated, weak,
or incomplete but still about the target is usually `partially supports`, not
`irrelevant`. Stage 2 derives the five-way category from document notes rather
than trusting a source-provided label and requires a concise evidence-grounded
reason. For refusal mode it enforces abstention. `scripts/validate_benchmark_gold.py`
checks schema and document/note alignment; `scripts/audit_benchmark_stage2.py`
supports label and evidence audits.

The detailed model, weighting, cache, voting, and runner description belongs in
the repository's forthcoming annotation-pipeline documentation. This dataset
card records only the benchmark-facing behavior needed to understand the
labels and artifacts.

## Normalization, integrity, and reproducibility

The `benchmark_final_v2` release was normalized into a minimal canonical
schema. Every one of its 933 rows had extra top-level merge provenance removed;
805 rows had noncanonical document IDs remapped to `d1..dN`; one document/note
alignment mismatch caused by HTML escaping was repaired; 10 irrelevant notes
had leftover key facts cleared. The full normalization record is
`data/releases/benchmark_dataset_v2/benchmark_final_v2_manifest.json`.

For reproducibility:

1. Start from the canonical release file appropriate to the claim: use 736 for
   the reviewer-facing holdout and 933 only when the full release is intended.
2. Preserve JSONL line boundaries and do not reorder records before joining
   external predictions by `id`.
3. Treat `retrieved_docs` and `per_doc_notes` as aligned by `doc_id`, never by
   an assumed fixed document count.
4. Use `answerable_under_evidence` to separate answer generation from abstention
   evaluation. Empty gold answers do not license guessing.
5. For reconstruction research, consult the pinned source manifest, candidate
   manifest, retrieval artifacts, selection scripts, and human-review artifacts
   named above. Network search results are time-sensitive, so exact historic
   retrieval cannot be regenerated from a later Tavily call alone.

## Limitations and responsible interpretation

This is an evidence-conditioned benchmark: its labels and answers describe the
provided retrieved evidence, not an unrestricted claim about world truth.
Search ranking, page availability, locale, and temporal drift can change the
evidence observed for an otherwise identical query. The release stores snippets
and source URLs/timestamps, but source pages may subsequently change or vanish.

The candidate sources have different topical, temporal, and question-style
properties. Candidate-pool balancing mitigates source dominance at acquisition
time; it does not guarantee label balance after retrieval and quality filtering.
The relatively small misinformation and credible-conflict strata should be
reported with counts alongside aggregate metrics. Finally, document-level
source-quality fields are committee annotations and should be used as provided
metadata, not treated as independently verified institutional rankings.

## Primary pointers

- Current release entry point: `data/releases/README.md`
- Release manifests: `data/releases/benchmark_dataset_v2/`
- Build overview and commands: `docs/conflicts_benchmark_build.md`
- Tavily retrieval protocol: `docs/TAVILY_RETRIEVAL_METHODOLOGY.md`
- Pinned external sources: `data/external_sources/source_manifest.json`
- Benchmark human-review materials: `human_reviews/benchmark/`
- Internal build artifact: `data/benchmarks/final_benchmark_2026-06-22/`
