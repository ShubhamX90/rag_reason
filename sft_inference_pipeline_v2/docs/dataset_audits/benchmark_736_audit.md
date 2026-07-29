# Benchmark 736 Audit

## Scope

This audit covers the held-out benchmark file:

- `data/Benchmark Dataset/benchmark_final_v2_holdout_clean_736.jsonl`

All counts below were computed directly from that file.

## File / Schema Summary

| Check | Result |
| --- | --- |
| Rows | 736 |
| Unique top-level schema variants | 1 |
| Same top-level schema across all rows | Yes |
| Missing `query` rows | 0 |
| Missing `retrieved_docs` rows | 0 |
| Missing `per_doc_notes` rows | 0 |
| Missing `conflict_type` rows | 0 |
| Missing `conflict_reason` rows | 0 |
| Retrieved doc count matches `per_doc_notes` count on every row | Yes |

Top-level fields:

| Field |
| --- |
| `id` |
| `query` |
| `retrieved_docs` |
| `conflict_type` |
| `conflict_reason` |
| `gold_answer` |
| `per_doc_notes` |
| `answerable_under_evidence` |
| `conflict_category_id` |

## Core Counts

| Metric | Count | % |
| --- | ---: | ---: |
| Total rows | 736 | 100.00 |
| Non-refusals | 548 | 74.46 |
| Refusals | 188 | 25.54 |
| `answerable_under_evidence = true` | 608 | 82.61 |
| `answerable_under_evidence = false` | 128 | 17.39 |
| Blank gold answers | 188 | 25.54 |
| Non-blank gold answers | 548 | 74.46 |

Important caveat:

| Metric | Count |
| --- | ---: |
| `answerable_under_evidence = true` but blank `gold_answer` | 60 |

So the file has `608` rows marked answerable under evidence, but only `548` rows with non-blank gold answers.

## Conflict Distribution

### Overall

| Conflict Type | Count | % |
| --- | ---: | ---: |
| Complementary information | 221 | 30.03 |
| No conflict | 211 | 28.67 |
| Conflict due to outdated information | 158 | 21.47 |
| Conflicting opinions or research outcomes | 109 | 14.81 |
| Conflict due to misinformation | 37 | 5.03 |

### Non-Refusals Only

| Conflict Type | Count | % of 548 |
| --- | ---: | ---: |
| Complementary information | 156 | 28.47 |
| No conflict | 154 | 28.10 |
| Conflict due to outdated information | 140 | 25.55 |
| Conflicting opinions or research outcomes | 61 | 11.13 |
| Conflict due to misinformation | 37 | 6.75 |

### Refusals Only

| Conflict Type | Count | % of 188 |
| --- | ---: | ---: |
| Complementary information | 65 | 34.57 |
| No conflict | 57 | 30.32 |
| Conflicting opinions or research outcomes | 48 | 25.53 |
| Conflict due to outdated information | 18 | 9.57 |
| Conflict due to misinformation | 0 | 0.00 |

## Query Source Distribution

This uses the `id` prefix.

| Query Source | Count | % |
| --- | ---: | ---: |
| `qacc` | 145 | 19.70 |
| `situatedqa` | 141 | 19.16 |
| `conflictingqa` | 137 | 18.61 |
| `trust` | 128 | 17.39 |
| `freshqa` | 85 | 11.55 |
| `wikirevision` | 78 | 10.60 |
| `hotpotqa` | 19 | 2.58 |
| `healthcontradict` | 2 | 0.27 |
| `misinformation` | 1 | 0.14 |

## Retrieved Document Distribution

| Docs per row | Rows | % |
| --- | ---: | ---: |
| 5 | 631 | 85.73 |
| 4 | 78 | 10.60 |
| 10 | 19 | 2.58 |
| 8 | 3 | 0.41 |
| 2 | 2 | 0.27 |
| 3 | 1 | 0.14 |
| 6 | 1 | 0.14 |
| 7 | 1 | 0.14 |

## Document Summary

| Metric | Value |
| --- | ---: |
| Total retrieved docs | 3701 |
| Mean docs/row | 5.03 |
| Median docs/row | 5 |
| Min docs/row | 2 |
| Max docs/row | 10 |

## Per-Doc Verdict Distribution

| Verdict | Count | % of 3701 |
| --- | ---: | ---: |
| Partially supports | 2168 | 58.58 |
| Supports | 1225 | 33.10 |
| Irrelevant | 308 | 8.32 |

## Source Quality Distribution

| Source Quality | Count | % of 3701 |
| --- | ---: | ---: |
| High | 2119 | 57.26 |
| Low | 1582 | 42.74 |

## Top Retrieved Domains

| Domain | Doc Count | % of docs |
| --- | ---: | ---: |
| `en.wikipedia.org` | 502 | 13.56 |
| empty / unparsable domain | 193 | 5.21 |
| `infoarchive.net` | 168 | 4.54 |
| `example.com` | 166 | 4.49 |
| `youtube.com` | 154 | 4.16 |
| `datasource.org` | 154 | 4.16 |
| `researchhub.ai` | 152 | 4.11 |
| `reddit.com` | 120 | 3.24 |
| `quora.com` | 92 | 2.49 |
| `imdb.com` | 41 | 1.11 |

## Timestamp Coverage

| Metric | Count |
| --- | ---: |
| Docs with empty timestamp | 3113 |
| Docs with non-empty timestamp | 588 |
| Rows where all doc timestamps are empty | 529 |
| Rows with at least one non-empty doc timestamp | 207 |

## Note Field Coverage

| Metric | Count |
| --- | ---: |
| Blank `quote` fields in `per_doc_notes` | 300 |
| Blank `key_fact` fields in `per_doc_notes` | 308 |

## Query Length

| Metric | Value |
| --- | ---: |
| Min words | 3 |
| Max words | 40 |
| Mean words | 9.35 |

## Bottom Line

The held-out benchmark is structurally clean at the schema level and heavily concentrated around 5-document retrieval contexts. It is majority non-refusal (`548/736`), but the file also contains a substantial set of blank-gold rows. The practical answerable-with-gold subset is therefore smaller than the `answerable_under_evidence = true` count alone would suggest.
