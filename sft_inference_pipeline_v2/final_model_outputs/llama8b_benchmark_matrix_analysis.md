# Llama 3.1 8B Run L Benchmark Matrix Audit

## Scope

This audit covers the locally completed benchmark generations stored under `final_model_outputs/llama8b` and evaluated against `data/splits/benchmark_final_v2_holdout_clean_736.jsonl`. The matrix structure is baseline vs SFT across `e2e`, `oracle_conflict`, `oracle_notes`, and `oracle_both`, each with `minimal`, `runtime`, and `strict` prompt profiles.

Completed rows available locally: `24` / `24`.

Important interpretation rule: `contract_ok_pct`, abstention behavior, and support completeness are treated as first-gate sanity checks before trusting downstream task metrics.

`Operationally trustworthy` in this audit means `contract_ok_pct >= 80`, `false_abstains <= 25`, and `conflict_support >= 700`.

## Full Matrix

| Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Conflict Support | Doc Micro % | Token F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0 | 82.47 | 1 | 128 | 0 | 0 | 0 | 0.2657 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | e2e | runtime | 55.4 | 83.15 | 2 | 122 | 36.2 | 594 | 47.35 | 0.2533 | support drop |
| baseline | e2e | strict | 0 | 75.14 | 87 | 96 | 34.58 | 720 | 0 | 0.2363 | contract 0; no doc eval; over-abstain; trace gap |
| baseline | oracle_conflict | minimal | 0 | 81.52 | 9 | 127 | 0 | 0 | 0 | 0.2457 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_conflict | runtime | 66.2 | 71.74 | 119 | 89 | 99.3 | 716 | 42.56 | 0.2422 | heavy over-abstain; trace gap |
| baseline | oracle_conflict | strict | 0 | 76.77 | 67 | 104 | 0 | 0 | 55.46 | 0.1545 | contract 0; conflict unparsable; over-abstain; trace gap |
| baseline | oracle_notes | minimal | 0 | 83.29 | 1 | 122 | 0 | 0 | 0 | 0.2022 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_notes | runtime | 82.1 | 82.47 | 92 | 37 | 31.97 | 710 | 88.6 | 0.2281 | over-abstain; trace gap |
| baseline | oracle_notes | strict | 29.9 | 79.21 | 93 | 60 | 41.35 | 728 | 99.86 | 0.2322 | over-abstain; trace gap |
| baseline | oracle_both | minimal | 0 | 82.61 | 7 | 121 | 0 | 0 | 0 | 0.1675 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_both | runtime | 84.1 | 82.2 | 79 | 52 | 97.82 | 733 | 93.35 | 0.2581 | over-abstain; trace gap |
| baseline | oracle_both | strict | 0 | 71.2 | 128 | 84 | 0 | 0 | 100 | 0.2263 | contract 0; conflict unparsable; heavy over-abstain; trace gap |
| sft | e2e | minimal | 93.2 | 97.55 | 17 | 1 | 61.07 | 732 | 79.81 | 0.3516 | clean |
| sft | e2e | runtime | 93.5 | 97.42 | 18 | 1 | 62.89 | 733 | 78.48 | 0.353 | clean |
| sft | e2e | strict | 93.5 | 97.28 | 18 | 2 | 62.98 | 732 | 78.52 | 0.3379 | clean |
| sft | oracle_conflict | minimal | 87.8 | 96.06 | 28 | 1 | 75.24 | 719 | 79.36 | 0.3587 | over-abstain |
| sft | oracle_conflict | runtime | 93.3 | 97.28 | 19 | 1 | 79.18 | 735 | 78.02 | 0.3524 | clean |
| sft | oracle_conflict | strict | 92.9 | 97.42 | 17 | 2 | 78.95 | 727 | 77.87 | 0.338 | clean |
| sft | oracle_notes | minimal | 66.3 | 95.92 | 25 | 5 | 59.13 | 526 | 99.92 | 0.3628 | support drop |
| sft | oracle_notes | runtime | 95.9 | 97.96 | 11 | 4 | 64.54 | 736 | 98.51 | 0.3483 | clean |
| sft | oracle_notes | strict | 92.4 | 97.28 | 16 | 4 | 63.83 | 716 | 99.94 | 0.3461 | clean |
| sft | oracle_both | minimal | 44.3 | 94.7 | 21 | 18 | 74.09 | 359 | 99.78 | 0.3595 | support drop |
| sft | oracle_both | runtime | 95.2 | 97.42 | 13 | 6 | 76.33 | 735 | 98.4 | 0.3499 | trace gap |
| sft | oracle_both | strict | 92.1 | 97.42 | 11 | 8 | 75.7 | 712 | 99.89 | 0.3414 | clean |

## Variant-Level Summary

| Variant | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 12 | 26.47 | 79.31 | 57.08 | 95.17 | 28.43 | 43.93 | 0.226 |
| sft | 12 | 86.7 | 96.98 | 17.83 | 4.42 | 69.49 | 89.04 | 0.35 |

## Variant x Prompt-Family Summary

| Variant | Prompt Mode | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | 3 | 18.47 | 80.25 | 30 | 115.33 | 23.59 | 15.78 | 0.2518 |
| baseline | oracle_conflict | 3 | 22.07 | 76.68 | 65 | 106.67 | 33.1 | 32.67 | 0.2141 |
| baseline | oracle_notes | 3 | 37.33 | 81.66 | 62 | 73 | 24.44 | 62.82 | 0.2208 |
| baseline | oracle_both | 3 | 28.03 | 78.67 | 71.33 | 85.67 | 32.61 | 64.45 | 0.2173 |
| sft | e2e | 3 | 93.4 | 97.42 | 17.67 | 1.33 | 62.31 | 78.94 | 0.3475 |
| sft | oracle_conflict | 3 | 91.33 | 96.92 | 21.33 | 1.33 | 77.79 | 78.42 | 0.3497 |
| sft | oracle_notes | 3 | 84.87 | 97.05 | 17.33 | 4.33 | 62.5 | 99.46 | 0.3524 |
| sft | oracle_both | 3 | 77.2 | 96.51 | 15 | 10.67 | 75.37 | 99.36 | 0.3503 |

## Operationally Trustworthy Top Configurations

| Metric | Variant | Prompt Mode | Profile | Value | False Abstains | Missed Refusals | Conflict Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Contract OK | sft | oracle_notes | runtime | 95.9 | 11 | 4 | 736 |
| Contract OK | sft | oracle_both | runtime | 95.2 | 13 | 6 | 735 |
| Contract OK | sft | e2e | runtime | 93.5 | 18 | 1 | 733 |
| Contract OK | sft | e2e | strict | 93.5 | 18 | 2 | 732 |
| Contract OK | sft | oracle_conflict | runtime | 93.3 | 19 | 1 | 735 |
| Abstain Accuracy | sft | oracle_notes | runtime | 97.96 | 11 | 4 | 736 |
| Abstain Accuracy | sft | e2e | minimal | 97.55 | 17 | 1 | 732 |
| Abstain Accuracy | sft | e2e | runtime | 97.42 | 18 | 1 | 733 |
| Abstain Accuracy | sft | oracle_conflict | strict | 97.42 | 17 | 2 | 727 |
| Abstain Accuracy | sft | oracle_both | runtime | 97.42 | 13 | 6 | 735 |
| Conflict Accuracy | sft | oracle_conflict | runtime | 79.18 | 19 | 1 | 735 |
| Conflict Accuracy | sft | oracle_conflict | strict | 78.95 | 17 | 2 | 727 |
| Conflict Accuracy | sft | oracle_both | runtime | 76.33 | 13 | 6 | 735 |
| Conflict Accuracy | sft | oracle_both | strict | 75.7 | 11 | 8 | 712 |
| Conflict Accuracy | sft | oracle_notes | runtime | 64.54 | 11 | 4 | 736 |
| Doc Micro Accuracy | sft | oracle_notes | strict | 99.94 | 16 | 4 | 716 |
| Doc Micro Accuracy | sft | oracle_both | strict | 99.89 | 11 | 8 | 712 |
| Doc Micro Accuracy | sft | oracle_notes | runtime | 98.51 | 11 | 4 | 736 |
| Doc Micro Accuracy | sft | oracle_both | runtime | 98.4 | 13 | 6 | 735 |
| Doc Micro Accuracy | sft | e2e | minimal | 79.81 | 17 | 1 | 732 |
| Token F1 | sft | e2e | runtime | 0.353 | 18 | 1 | 733 |
| Token F1 | sft | oracle_conflict | runtime | 0.3524 | 19 | 1 | 735 |
| Token F1 | sft | e2e | minimal | 0.3516 | 17 | 1 | 732 |
| Token F1 | sft | oracle_both | runtime | 0.3499 | 13 | 6 | 735 |
| Token F1 | sft | oracle_notes | runtime | 0.3483 | 11 | 4 | 736 |

## Recommended Picks By Use Case

| Use Case | Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 | Why it wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best end-to-end overall | sft | e2e | minimal | 93.2 | 97.55 | 17 | 1 | 61.07 | 79.81 | 0.3516 | Best end-to-end trade-off on abstention control, structural reliability, and answer overlap. |
| Best conflict-type-focused oracle | sft | oracle_conflict | runtime | 93.3 | 97.28 | 19 | 1 | 79.18 | 78.02 | 0.3524 | Highest usable conflict accuracy within the oracle_conflict family while keeping refusal behavior and support sane. |
| Best doc-verdict-focused oracle | sft | oracle_notes | strict | 92.4 | 97.28 | 16 | 4 | 63.83 | 99.94 | 0.3461 | Strongest doc-verdict row once contract integrity and support are treated as first gates. |
| Best all-oracle overall | sft | oracle_both | runtime | 95.2 | 97.42 | 13 | 6 | 76.33 | 98.4 | 0.3499 | Best combined oracle trade-off after filtering for structural reliability. |

## Structural / Parsing Anomalies

| Variant | Prompt Mode | Profile | Contract OK % | Conflict Support | Doc Pairs | Think Count | Sentinel Count | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | e2e | runtime | 55.4 | 594 | 3569 | 736 | 736 | support drop |
| baseline | e2e | strict | 0 | 720 | 0 | 723 | 733 | contract 0; no doc eval; over-abstain; trace gap |
| baseline | oracle_conflict | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_conflict | runtime | 66.2 | 716 | 3510 | 718 | 725 | heavy over-abstain; trace gap |
| baseline | oracle_conflict | strict | 0 | 0 | 1830 | 362 | 695 | contract 0; conflict unparsable; over-abstain; trace gap |
| baseline | oracle_notes | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_notes | runtime | 82.1 | 710 | 3597 | 720 | 731 | over-abstain; trace gap |
| baseline | oracle_notes | strict | 29.9 | 728 | 1470 | 730 | 735 | over-abstain; trace gap |
| baseline | oracle_both | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_both | runtime | 84.1 | 733 | 3684 | 735 | 735 | over-abstain; trace gap |
| baseline | oracle_both | strict | 0 | 0 | 15 | 734 | 736 | contract 0; conflict unparsable; heavy over-abstain; trace gap |
| sft | oracle_conflict | minimal | 87.8 | 719 | 3570 | 736 | 736 | over-abstain |
| sft | oracle_notes | minimal | 66.3 | 526 | 2651 | 736 | 736 | support drop |
| sft | oracle_both | minimal | 44.3 | 359 | 1790 | 736 | 736 | support drop |
| sft | oracle_both | runtime | 95.2 | 735 | 3696 | 735 | 735 | trace gap |

## Dominant SFT Confusion Boundaries

| Gold | Pred | Count |
| --- | --- | --- |
| No conflict | Complementary information | 752 |
| Conflicting opinions or research outcomes | Complementary information | 218 |
| Complementary information | No conflict | 188 |
| Conflict due to misinformation | Complementary information | 146 |
| Complementary information | Conflicting opinions or research outcomes | 125 |
| No conflict | Conflict due to outdated information | 110 |
| Complementary information | Conflict due to outdated information | 69 |
| Conflict due to outdated information | Complementary information | 64 |
| Conflicting opinions or research outcomes | No conflict | 44 |
| No conflict | Conflicting opinions or research outcomes | 42 |

## Key Findings

| Finding | Evidence |
| --- | --- |
| SFT vs baseline separation | Average `contract_ok_pct` is `86.7` for SFT versus `26.47` for baseline. Average abstain accuracy is `96.98%` for SFT versus `79.31%` for baseline. |
| Over-abstention reduction | Average false abstains drop from `57.08` in baseline to `17.83` in SFT. |
| Best SFT e2e row | `sft + e2e + minimal` is the strongest end-to-end row here with `contract_ok_pct = 93.2`, `abstain_acc = 97.55%`, `false_abstains = 17`, `missed_refusals = 1`, and `token_f1 = 0.3516`. |
| Main remaining conflict boundary | The most persistent SFT confusion in the available rows is `No conflict -> Complementary information` with aggregate count `752`. |

## Bottom Line

For this Llama 3.1 8B matrix, the cleanest end-to-end operating point is `sft + e2e + minimal` under the current local results. The oracle families remain useful for stage-specific probing, but the report should be read through the lens of contract integrity, refusal calibration, and support completeness rather than headline conflict accuracy alone.
