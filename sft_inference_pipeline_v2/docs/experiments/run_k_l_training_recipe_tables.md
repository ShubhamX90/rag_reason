# Run K And Run L Training Recipe Tables

These tables are the Run K / Run L equivalents of the older Run F training-mixture summary tables.

All counts below are taken from the real rebuilt raw message files and launch recipes:

- [docs/experiments/sft_run_k_short_context_targeted_strategy_and_results.md](../../docs/experiments/sft_run_k_short_context_targeted_strategy_and_results.md)
- [docs/experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md](../../docs/experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md)
- [slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh](../../slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh)
- [slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh](../../slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh)

## Strategy Summary

| Strategy | Rows | Purpose |
| --- | ---: | --- |
| Short-context targeted mix (Run K) | 12,659 | Starts from the Run J benchmark-augmented backbone, adds `27` benchmark-like 5-doc answerable derived rows, turns on doc-verdict boundary drills, and strengthens partial-synthesis drills. The aim is to reduce over-abstention on short answerable cases without shaking the full recipe. |
| Boundary-rebalanced mix (Run L) | 13,349 | Keeps the useful short-context pressure from Run K, but adds `21` short `No conflict` answerable derived rows and slightly stronger `No conflict` / misinformation boundary pressure so the model does not learn that short answerable contexts usually imply conflict. |

## Component-Level Mixture Table

| Component | Run K | Run L |
| --- | ---: | ---: |
| Strict/default E2E trace | 1,778 | 1,820 |
| Strict partial-synthesis E2E drill | 66 | 69 |
| Runtime E2E trace | 889 | 910 |
| Runtime partial-synthesis E2E drill | 132 | 138 |
| Runtime document-verdict task | 889 | 910 |
| Runtime doc-boundary verdict drill | 889 | 910 |
| Runtime conflict-type task | 1,778 | 1,858 |
| Runtime boundary-drill conflict-type task | 1,463 | 1,839 |
| Runtime answer-only task | 889 | 910 |
| Runtime partial-synthesis answer-only drill | 132 | 138 |
| Source-guarded E2E trace | 0 | 0 |
| Minimal E2E trace | 3,556 | 3,640 |
| Minimal partial-synthesis E2E drill | 198 | 207 |
| Total rows | 12,659 | 13,349 |

## Exact Component Arithmetic

| Component | Run K arithmetic | Run L arithmetic |
| --- | --- | --- |
| Strict/default E2E trace | `889 * 2 = 1,778` | `910 * 2 = 1,820` |
| Runtime E2E trace | `889 * 1 = 889` | `910 * 1 = 910` |
| Runtime document-verdict task | `889 * 1 = 889` | `910 * 1 = 910` |
| Runtime answer-only task | `889 * 1 = 889` | `910 * 1 = 910` |
| Minimal E2E trace | `889 * 4 = 3,556` | `910 * 4 = 3,640` |
| Strict partial-synthesis E2E drill | `66 answerable partial-only rows * 1 = 66` | `69 answerable partial-only rows * 1 = 69` |
| Runtime partial-synthesis E2E drill | `66 * 2 = 132` | `69 * 2 = 138` |
| Runtime partial-synthesis answer-only drill | `66 * 2 = 132` | `69 * 2 = 138` |
| Minimal partial-synthesis E2E drill | `66 * 3 = 198` | `69 * 3 = 207` |
| Runtime doc-boundary verdict drill | `889 * 1 = 889` | `910 * 1 = 910` |

## Conflict-Type Boundary Pressure Breakdown

This is where the main K-to-L shift happened.

### Runtime conflict-type rows

| Conflict label | Run K | Run L |
| --- | ---: | ---: |
| No conflict | 630 | 672 |
| Complementary information | 544 | 544 |
| Conflicting opinions or research outcomes | 308 | 308 |
| Conflict due to outdated information | 258 | 258 |
| Conflict due to misinformation | 38 | 76 |
| Total | 1,778 | 1,858 |

Interpretation:

- Run K uses the standard runtime `conflict_type=2` duplication for all labels.
- Run L keeps that, but adds extra runtime label pressure for misinformation, which doubles the misinformation slice again from `38` to `76`.

### Boundary-drill conflict-type rows

| Conflict label | Run K | Run L |
| --- | ---: | ---: |
| No conflict | 315 | 672 |
| Complementary information | 544 | 544 |
| Conflicting opinions or research outcomes | 308 | 308 |
| Conflict due to outdated information | 258 | 258 |
| Conflict due to misinformation | 38 | 57 |
| Total | 1,463 | 1,839 |

Interpretation:

- Run K boundary rows use weights `No conflict=1`, all other labels `=2`.
- Run L boundary rows raise `No conflict` from `1` to `2`, and misinformation from `2` to `3`.
- That is the cleanest recipe-level explanation for why Run L is called boundary-rebalanced rather than just "more of Run K."

## Derived Short-Context Additions

| Derived split statistic | Run K | Run L |
| --- | ---: | ---: |
| Base train rows | 862 | 862 |
| Base val rows | 81 | 81 |
| Derived rows added | 27 | 48 |
| Final train rows | 889 | 910 |
| Final val rows | 81 | 81 |

### Derived rows by origin

| Origin | Run K | Run L |
| --- | ---: | ---: |
| Main backbone | 862 | 862 |
| Short 5-doc answerable support | 27 | 27 |
| Short 5-doc `No conflict` support | 0 | 18 |
| Short 5-doc `No conflict` partial-only | 0 | 3 |
| Total | 889 | 910 |

## Short Reading

| Run | Plain-language summary |
| --- | --- |
| Run K | "Teach the model harder that short 5-doc answerable cases are often still answerable, and add extra drills where partial evidence must be combined." |
| Run L | "Keep that short-answerable teaching, but balance it by also showing short answerable `No conflict` cases so the model does not start treating short answerable contexts as conflict-heavy by default." |
