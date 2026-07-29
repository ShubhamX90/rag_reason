# Stagewise Val Ceiling Analysis

Run target:
- Dataset: `data/splits/92p5_7p5/stagewise_multi/val/stage3_final.jsonl`
- Evaluated answer field: `expected_response.answer`
- Committee: `Codex CLI (gpt-5.4)` + `DeepSeek v4 Flash`
- Weighting: behavior/STR use weighted voting with priorities `3:2`; FG uses weighted committee support aggregation

## Overall

| Metric | Value | Notes |
|---|---:|---|
| Samples | 49 | Full `92p5_7p5` stagewise val split |
| Correct Refusals | 15 | Counted in GR only; excluded from other sub-metric averages |
| GR Accuracy | 1.000 | 49/49 |
| GR F1 | 1.000 | TP=34, FP=0, FN=0, TN=15 |
| Behavior Adherence | 0.971 | 34 applicable samples |
| Factual Grounding | 0.953 | 34 applicable samples |
| Single-Truth Recall | 0.941 | 17 applicable samples |
| CATS Score | 0.966 | Best current ceiling estimate |

## Per Type

| Type | Samples | Correct Refusals | Behavior | Grounding | Recall | CATS |
|---|---:|---:|---:|---:|---:|---:|
| 1 No Conflict | 19 | 7 | 0.917 | 0.972 | 0.917 | 0.951 |
| 2 Complementary Info | 15 | 7 | 1.000 | 1.000 | n/a | 1.000 |
| 3 Conflicting Opinions | 10 | 1 | 1.000 | 0.861 | n/a | 0.954 |
| 4 Outdated Info | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |

Note:
- No Type 5 samples are present in this split.
- Recall is not applicable for Type 2 / Type 3 in the current setup.

## Committee / Cost

| Item | Value |
|---|---:|
| Total Cost | `$0.0962` |
| Decisions Made | 162 |
| Avg Cost / Decision | `$0.000594` |
| Codex CLI Cost | Unmetered in report |
| DeepSeek Cost | `$0.0962` |

Interpretation:
- The evaluator now appears stable enough to use this as the current ceiling.
- The earlier FG under-scoring issue was mostly claim/citation extraction, which has now been substantially improved.

## Remaining Misses

### Behavior

Only 1 behavior miss remains:

| Sample | Type | Issue |
|---|---:|---|
| `#0408` | 1 | Committee split on whether mentioning informal everyday usage of "champagne" is unnecessary contrast in a no-conflict answer |

### Factual Grounding

Only 6 samples still have unsupported FG claims:

| Sample | Type | FG Score | Reason |
|---|---:|---:|---|
| `#0042` | 1 | 0.667 | `no_supporting_doc_found` |
| `#0175` | 3 | 0.750 | `supporting_doc_not_cited` |
| `#0201` | 3 | 0.750 | `supporting_doc_not_cited` |
| `#0206` | 3 | 0.750 | `supporting_doc_not_cited` |
| `#0399` | 3 | 0.750 | `cross_doc_not_cited` |
| `#0416` | 3 | 0.750 | `supporting_doc_not_cited` |

Interpretation:
- The remaining FG loss is now localized rather than systemic.
- Most residual FG errors are citation-discipline failures, not broad committee or parser failures.

## Bottom Line

This is a strong ceiling run for the current metric design:
- GR is perfect.
- Behavior is near-perfect with one remaining borderline disagreement.
- FG is now high (`0.953`) and substantially cleaner than earlier runs.
- STR is high (`0.941`).

Recommended use:
- Treat this run as the current best stagewise-val ceiling for the `Codex + DeepSeek` committee.
