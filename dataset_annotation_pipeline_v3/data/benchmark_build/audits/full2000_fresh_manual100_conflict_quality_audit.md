# Full 2000 Fresh Retrieval Audit

Input: `data/benchmark_build/retrieved/full2000_fresh_tavily_20_keep10_tasb_readable_raw_with_fulltext.jsonl`

## Full-file retrieval integrity
- Rows: 2000
- Source distribution: {'conflictingqa': 272, 'situatedqa_geo': 432, 'situatedqa_temp': 432, 'freshqa': 432, 'qacc': 432}
- Retrieved doc count distribution: {10: 1878, 0: 118, 1: 1, 3: 2, 7: 1}
- Rows with exactly 10 docs: 1878
- Rows with fewer than 10 docs: 122
- Empty retrieval rows: 118
- Fewer-than-10 rows by source: {'freshqa': 38, 'situatedqa_geo': 26, 'qacc': 25, 'situatedqa_temp': 19, 'conflictingqa': 14}
- Exactly-10 rows by source: {'situatedqa_temp': 413, 'qacc': 407, 'situatedqa_geo': 406, 'freshqa': 394, 'conflictingqa': 258}
- Search cache hits: {False: 2000}
- Tavily usage sum from metadata: 2000
- Provider docs: {'unknown': 18794}
- Fetch status top 12: [('provided_raw_content', 13051), ('fetched', 3454), ('http_403', 1824), ('http_400', 81), ('cache', 79), ('http_404', 34), ("error:('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))", 23), ('http_401', 19), ('http_429', 19), ("error:HTTPSConnectionPool(host='www.sporcle.com', port=443): Max retries exceeded with url: /reference/clu", 16), ("error:HTTPSConnectionPool(host='www.washingtonpost.com', port=443): Read timed out. (read timeout=20.0)", 10), ('http_500', 7)]
- Top hosts: [('en.wikipedia.org', 1542), ('youtube.com', 1321), ('reddit.com', 855), ('quora.com', 590), ('imdb.com', 297), ('britannica.com', 239), ('espn.com', 98), ('pmc.ncbi.nlm.nih.gov', 96), ('ebsco.com', 92), ('statmuse.com', 90), ('sciencedirect.com', 90), ('testbook.com', 86), ('linkedin.com', 80), ('bbc.com', 68), ('history.com', 67), ('medium.com', 64), ('scribd.com', 59), ('statista.com', 58), ('guinnessworldrecords.com', 56), ('brainly.com', 54)]
- Window fallback reasons: {}
- Short-text word lengths: avg=267.2, p05=14, p50=360, p95=427
- Full text chars: avg=19960.0, p05=0, p50=4770, p95=51272, missing_full=2582

## Manual stratified sample
- Read 100 query records from `/private/tmp/full2000_stratified100_review_compact.txt`.
- Each sampled record had 10 retrieved docs, so this pass covered 1000 retrieved snippets/search extracts.
- Sample quota: conflictingqa=25, freshqa=25, situatedqa_temp=20, situatedqa_geo=20, qacc=10.

### Manual conflict bucket distribution
- no_conflict: 32
- ambiguous_complementary: 13
- conflicting_research_or_opinion: 13
- reject_low_quality: 12
- misinformation_false_premise: 8
- outdated_temporal_conflict: 7
- legal_temporal_or_jurisdiction_conflict: 5
- factual_or_scope_conflict: 4
- temporal_or_current_fact: 3
- factual_conflict: 2
- temporal_or_scope_conflict: 1

- Usable for benchmark annotation after manual read: 88/100
- High or medium-high manual confidence: 82/100
- Reject/low-quality in sample: 12/100

### Manual buckets by source
- conflictingqa: {'conflicting_research_or_opinion': 13, 'legal_temporal_or_jurisdiction_conflict': 2, 'no_conflict': 3, 'ambiguous_complementary': 6, 'misinformation_false_premise': 1}
- freshqa: {'no_conflict': 12, 'misinformation_false_premise': 7, 'temporal_or_current_fact': 2, 'factual_or_scope_conflict': 2, 'outdated_temporal_conflict': 1, 'reject_low_quality': 1}
- qacc: {'no_conflict': 7, 'factual_conflict': 1, 'ambiguous_complementary': 2}
- situatedqa_geo: {'ambiguous_complementary': 3, 'temporal_or_scope_conflict': 1, 'reject_low_quality': 11, 'no_conflict': 1, 'temporal_or_current_fact': 1, 'legal_temporal_or_jurisdiction_conflict': 3}
- situatedqa_temp: {'outdated_temporal_conflict': 6, 'ambiguous_complementary': 2, 'no_conflict': 9, 'factual_conflict': 1, 'factual_or_scope_conflict': 2}

### Confidence by source
- conflictingqa: {'high': 25}
- freshqa: {'high': 19, 'medium_high': 3, 'medium': 1, 'medium_low': 1, 'low': 1}
- qacc: {'high': 10}
- situatedqa_geo: {'medium': 2, 'low': 11, 'high': 6, 'medium_high': 1}
- situatedqa_temp: {'high': 16, 'medium_high': 2, 'medium': 2}

## Key findings
- Retrieval integrity is now clean: all 2000 searches report `search_cache_hit=false` and Tavily metadata sums to 2000 credits.
- The strongest conflict-quality source is `conflictingqa`: most sampled rows are high-confidence opinion/research/legal/conditional conflicts.
- `freshqa` is strong but not balanced toward conflict; many rows are clean current facts, while several are excellent false-premise/misinformation-style rows.
- `situatedqa_temp` is useful for temporal drift and currentness conflicts, with fewer outright retrieval failures than `situatedqa_geo`.
- `situatedqa_geo` is the main risk source: many rows contain pronouns or missing geography, causing Tavily to retrieve the wrong country/context. These should be filtered or annotated as low quality before final benchmark selection.
- `qacc` is mostly clean fact QA, with occasional scope ambiguity; it is useful filler but does not naturally create an even conflict-type distribution.

## Recommended next action
- Use `full2000_fresh_annotation_candidates_exact10.jsonl` as the annotation input pool, not the raw 2000 file.
- Keep `full2000_fresh_quality_review_candidates_lt10docs.jsonl` out of automated annotation unless manually rescued.
- Before selecting the final 500, run the multi-LLM annotation pipeline over the 1878 exact-10 rows, then select by high gold-label confidence and balance across conflict types.
- Add a source-aware prefilter or postfilter that heavily penalizes underspecified `situatedqa_geo` questions with terms like `we`, `today`, `the state`, `the country`, `the president`, `the prime minister`, unless the source metadata clearly supplies the missing location/date.

## Files written
- `data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_exact10.jsonl`
- `data/benchmark_build/retrieved/full2000_fresh_quality_review_candidates_lt10docs.jsonl`
- `data/benchmark_build/retrieved/full2000_fresh_manual100_conflict_quality_audit.jsonl`
