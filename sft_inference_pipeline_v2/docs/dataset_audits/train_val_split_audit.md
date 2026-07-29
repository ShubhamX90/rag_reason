# Train / Val Split Audit

## Scope

This audit covers the current files:

- `data/splits/train.jsonl`
- `data/splits/val.jsonl`

All counts below were computed directly from those files.

## Split Sizes

| Split | Rows |
| --- | ---: |
| Train | 862 |
| Val | 81 |

## Shared Structural Notes

Both files:

- have no missing `query`, `retrieved_docs`, `per_doc_notes`, `conflict_type`, or `conflict_reason`
- have exact equality between retrieved doc count and `per_doc_notes` count on every row
- use multiple top-level schema variants rather than a single benchmark-style canonical schema

The observed top-level fields include:

| Field |
| --- |
| `id` |
| `query` |
| `retrieved_docs` |
| `per_doc_notes` |
| `gold_answer` |
| `conflict_type` |
| `conflict_reason` |
| `answerable_under_evidence` |
| `expected_response` |
| `think` |
| `_ans_vote_tally` |
| `_ans_winner_model` |
| `_abstain_vote_tally` |
| `_abstain_winner_model` |

## Train Split

### Core Counts

| Metric | Count | % |
| --- | ---: | ---: |
| Total rows | 862 | 100.00 |
| Non-refusals | 387 | 44.90 |
| Refusals | 475 | 55.10 |
| `answerable_under_evidence = true` | 591 | 68.56 |
| `answerable_under_evidence = false` | 271 | 31.44 |
| Blank gold answers | 455 | 52.78 |
| Non-blank gold answers | 407 | 47.22 |
| `answerable_under_evidence = true` but blank `gold_answer` | 204 | 23.67 |

### Schema / Integrity

| Check | Result |
| --- | --- |
| Same top-level schema across all rows | No |
| Unique top-level schema variants | 4 |
| Missing `query` rows | 0 |
| Missing `retrieved_docs` rows | 0 |
| Missing `per_doc_notes` rows | 0 |
| Missing `conflict_type` rows | 0 |
| Missing `conflict_reason` rows | 0 |
| Retrieved docs count matches `per_doc_notes` count on every row | Yes |

### Conflict Distribution: Overall

| Conflict Type | Count | % |
| --- | ---: | ---: |
| No conflict | 315 | 36.54 |
| Complementary information | 262 | 30.39 |
| Conflicting opinions or research outcomes | 144 | 16.71 |
| Conflict due to outdated information | 123 | 14.27 |
| Conflict due to misinformation | 18 | 2.09 |

### Conflict Distribution: Non-Refusals Only

| Conflict Type | Count | % of 387 |
| --- | ---: | ---: |
| No conflict | 195 | 50.39 |
| Conflict due to outdated information | 100 | 25.84 |
| Complementary information | 55 | 14.21 |
| Conflicting opinions or research outcomes | 21 | 5.43 |
| Conflict due to misinformation | 16 | 4.13 |

### Conflict Distribution: Refusals Only

| Conflict Type | Count | % of 475 |
| --- | ---: | ---: |
| Complementary information | 207 | 43.58 |
| Conflicting opinions or research outcomes | 123 | 25.89 |
| No conflict | 120 | 25.26 |
| Conflict due to outdated information | 23 | 4.84 |
| Conflict due to misinformation | 2 | 0.42 |

### Retrieved Document Distribution

| Docs per row | Rows | % |
| --- | ---: | ---: |
| 5 | 418 | 48.49 |
| 9 | 177 | 20.53 |
| 10 | 101 | 11.72 |
| 8 | 53 | 6.15 |
| 7 | 29 | 3.36 |
| 4 | 24 | 2.78 |
| 6 | 17 | 1.97 |
| 11 | 11 | 1.28 |
| 12 | 8 | 0.93 |
| 13 | 8 | 0.93 |
| 15 | 6 | 0.70 |
| 14 | 5 | 0.58 |
| 16 | 2 | 0.23 |
| 20 | 2 | 0.23 |
| 19 | 1 | 0.12 |

### Document Summary

| Metric | Value |
| --- | ---: |
| Total docs | 6090 |
| Mean docs/row | 7.06 |
| Median docs/row | 5 |
| Min docs/row | 4 |
| Max docs/row | 20 |

### Per-Doc Verdicts

| Verdict | Count | % of docs |
| --- | ---: | ---: |
| Partially supports | 3011 | 49.44 |
| Supports | 2400 | 39.41 |
| Irrelevant | 679 | 11.15 |

### Source Quality

| Source Quality | Count | % of docs |
| --- | ---: | ---: |
| Low | 4034 | 66.24 |
| High | 2056 | 33.76 |

### Top Domains

| Domain | Doc Count | % of docs |
| --- | ---: | ---: |
| `en.wikipedia.org` | 511 | 8.39 |
| `infoarchive.net` | 325 | 5.34 |
| `datasource.org` | 317 | 5.21 |
| `example.com` | 317 | 5.21 |
| `researchhub.ai` | 296 | 4.86 |
| `youtube.com` | 156 | 2.56 |
| `reddit.com` | 114 | 1.87 |
| `quora.com` | 84 | 1.38 |
| `britannica.com` | 63 | 1.03 |
| empty domain | 50 | 0.82 |

### Source-Style Query Prefixes

Many rows use curated numeric IDs, but the source-style prefixes present are:

| Prefix | Count |
| --- | ---: |
| `trust` | 66 |
| `qacc` | 45 |
| `situatedqa` | 42 |
| `freshqa` | 39 |
| `conflictingqa` | 32 |
| `wikirevision` | 24 |
| `hotpotqa` | 5 |

### Timestamp Coverage

| Metric | Count |
| --- | ---: |
| Empty doc timestamps | 3342 |
| Non-empty doc timestamps | 2748 |
| Rows with all doc timestamps empty | 315 |
| Rows with any non-empty timestamp | 547 |

### Note Field Coverage

| Metric | Count |
| --- | ---: |
| Blank `quote` fields in `per_doc_notes` | 677 |
| Blank `key_fact` fields in `per_doc_notes` | 679 |

### Query Length

| Metric | Value |
| --- | ---: |
| Min words | 3 |
| Max words | 48 |
| Mean words | 9.31 |

## Val Split

### Core Counts

| Metric | Count | % |
| --- | ---: | ---: |
| Total rows | 81 | 100.00 |
| Non-refusals | 42 | 51.85 |
| Refusals | 39 | 48.15 |
| `answerable_under_evidence = true` | 59 | 72.84 |
| `answerable_under_evidence = false` | 22 | 27.16 |
| Blank gold answers | 38 | 46.91 |
| Non-blank gold answers | 43 | 53.09 |
| `answerable_under_evidence = true` but blank `gold_answer` | 17 | 20.99 |

### Schema / Integrity

| Check | Result |
| --- | --- |
| Same top-level schema across all rows | No |
| Unique top-level schema variants | 4 |
| Missing `query` rows | 0 |
| Missing `retrieved_docs` rows | 0 |
| Missing `per_doc_notes` rows | 0 |
| Missing `conflict_type` rows | 0 |
| Missing `conflict_reason` rows | 0 |
| Retrieved docs count matches `per_doc_notes` count on every row | Yes |

### Conflict Distribution: Overall

| Conflict Type | Count | % |
| --- | ---: | ---: |
| No conflict | 28 | 34.57 |
| Complementary information | 24 | 29.63 |
| Conflicting opinions or research outcomes | 14 | 17.28 |
| Conflict due to outdated information | 13 | 16.05 |
| Conflict due to misinformation | 2 | 2.47 |

### Conflict Distribution: Non-Refusals Only

| Conflict Type | Count | % of 42 |
| --- | ---: | ---: |
| No conflict | 19 | 45.24 |
| Conflict due to outdated information | 11 | 26.19 |
| Complementary information | 7 | 16.67 |
| Conflicting opinions or research outcomes | 3 | 7.14 |
| Conflict due to misinformation | 2 | 4.76 |

### Conflict Distribution: Refusals Only

| Conflict Type | Count | % of 39 |
| --- | ---: | ---: |
| Complementary information | 17 | 43.59 |
| Conflicting opinions or research outcomes | 11 | 28.21 |
| No conflict | 9 | 23.08 |
| Conflict due to outdated information | 2 | 5.13 |
| Conflict due to misinformation | 0 | 0.00 |

### Retrieved Document Distribution

| Docs per row | Rows | % |
| --- | ---: | ---: |
| 5 | 43 | 53.09 |
| 9 | 13 | 16.05 |
| 10 | 9 | 11.11 |
| 11 | 4 | 4.94 |
| 4 | 4 | 4.94 |
| 8 | 3 | 3.70 |
| 7 | 2 | 2.47 |
| 6 | 1 | 1.23 |
| 12 | 1 | 1.23 |
| 14 | 1 | 1.23 |

### Document Summary

| Metric | Value |
| --- | ---: |
| Total docs | 552 |
| Mean docs/row | 6.81 |
| Median docs/row | 5 |
| Min docs/row | 4 |
| Max docs/row | 14 |

### Per-Doc Verdicts

| Verdict | Count | % of docs |
| --- | ---: | ---: |
| Partially supports | 252 | 45.65 |
| Supports | 223 | 40.40 |
| Irrelevant | 77 | 13.95 |

### Source Quality

| Source Quality | Count | % of docs |
| --- | ---: | ---: |
| Low | 338 | 61.23 |
| High | 214 | 38.77 |

### Top Domains

| Domain | Doc Count | % of docs |
| --- | ---: | ---: |
| `en.wikipedia.org` | 52 | 9.42 |
| `example.com` | 35 | 6.34 |
| `datasource.org` | 26 | 4.71 |
| `infoarchive.net` | 23 | 4.17 |
| `researchhub.ai` | 21 | 3.80 |
| `youtube.com` | 12 | 2.17 |
| empty domain | 10 | 1.81 |
| `britannica.com` | 8 | 1.45 |
| `reddit.com` | 8 | 1.45 |
| `quora.com` | 6 | 1.09 |

### Source-Style Query Prefixes

| Prefix | Count |
| --- | ---: |
| `trust` | 6 |
| `qacc` | 6 |
| `situatedqa` | 6 |
| `freshqa` | 5 |
| `conflictingqa` | 4 |
| `wikirevision` | 4 |
| `hotpotqa` | 1 |

### Timestamp Coverage

| Metric | Count |
| --- | ---: |
| Empty doc timestamps | 358 |
| Non-empty doc timestamps | 194 |
| Rows with all doc timestamps empty | 37 |
| Rows with any non-empty timestamp | 44 |

### Note Field Coverage

| Metric | Count |
| --- | ---: |
| Blank `quote` fields in `per_doc_notes` | 77 |
| Blank `key_fact` fields in `per_doc_notes` | 77 |

### Query Length

| Metric | Value |
| --- | ---: |
| Min words | 4 |
| Max words | 31 |
| Mean words | 9.09 |

## Bottom Line

The current training setup is much more refusal-heavy than the held-out benchmark:

| Split | Non-Refusal % | Refusal % | Mean Docs/Row |
| --- | ---: | ---: | ---: |
| Benchmark 736 | 74.46 | 25.54 | 5.03 |
| Train | 44.90 | 55.10 | 7.06 |
| Val | 51.85 | 48.15 | 6.81 |

That difference is large enough to matter for abstention behavior and benchmark generalization, especially because the train/val splits also contain substantially more long-context retrieval rows than the held-out benchmark.
