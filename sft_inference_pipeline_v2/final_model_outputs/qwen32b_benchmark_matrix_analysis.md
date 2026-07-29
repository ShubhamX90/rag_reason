# Qwen 2.5 32B Run K Benchmark Matrix Audit

## Scope

This audit covers the locally completed benchmark generations stored under `final_model_outputs/qwen32b` and evaluated against `data/splits/benchmark_final_v2_holdout_clean_736.jsonl`. The matrix structure is baseline vs SFT across `e2e`, `oracle_conflict`, `oracle_notes`, and `oracle_both`, each with `minimal`, `runtime`, and `strict` prompt profiles.

Completed rows available locally: `24` / `24`.

Important interpretation rule: `contract_ok_pct`, abstention behavior, and support completeness are treated as first-gate sanity checks before trusting downstream task metrics.

`Operationally trustworthy` in this audit means `contract_ok_pct >= 80`, `false_abstains <= 25`, and `conflict_support >= 700`.

## Full Matrix

| Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Conflict Support | Doc Micro % | Token F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0 | 83.15 | 1 | 123 | 0 | 0 | 0 | 0.2158 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | e2e | runtime | 83.8 | 85.46 | 14 | 93 | 35.05 | 736 | 52.35 | 0.2768 | clean |
| baseline | e2e | strict | 83.7 | 86.01 | 20 | 83 | 40.72 | 727 | 62.85 | 0.2696 | clean |
| baseline | oracle_conflict | minimal | 0 | 82.74 | 1 | 126 | 0 | 0 | 0 | 0.1676 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_conflict | runtime | 82.7 | 84.1 | 50 | 67 | 100 | 736 | 52.13 | 0.2649 | over-abstain |
| baseline | oracle_conflict | strict | 0 | 30.16 | 502 | 12 | 50 | 2 | 63.64 | 0.244 | contract 0; heavy over-abstain; support drop |
| baseline | oracle_notes | minimal | 0 | 82.61 | 1 | 127 | 0 | 0 | 0 | 0.1715 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_notes | runtime | 87.8 | 88.32 | 25 | 61 | 43.75 | 736 | 93.92 | 0.282 | clean |
| baseline | oracle_notes | strict | 86.1 | 87.64 | 9 | 82 | 47.15 | 736 | 100 | 0.2543 | clean |
| baseline | oracle_both | minimal | 0 | 82.88 | 0 | 126 | 0 | 0 | 0 | 0.1668 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_both | runtime | 85.9 | 86.82 | 33 | 64 | 100 | 736 | 99.35 | 0.2763 | over-abstain |
| baseline | oracle_both | strict | 6.5 | 86.55 | 13 | 86 | 100 | 66 | 100 | 0.2571 | support drop |
| sft | e2e | minimal | 93.5 | 96.6 | 25 | 0 | 67.3 | 734 | 83.27 | 0.4068 | clean |
| sft | e2e | runtime | 92.5 | 96.2 | 28 | 0 | 66.71 | 733 | 83.95 | 0.4009 | over-abstain |
| sft | e2e | strict | 92.9 | 96.33 | 27 | 0 | 65.98 | 732 | 84.04 | 0.4105 | over-abstain; trace gap |
| sft | oracle_conflict | minimal | 89.1 | 97.01 | 21 | 1 | 86.53 | 720 | 83.7 | 0.3979 | clean |
| sft | oracle_conflict | runtime | 93.5 | 96.6 | 25 | 0 | 88.68 | 733 | 83.59 | 0.3898 | trace gap |
| sft | oracle_conflict | strict | 92.9 | 95.38 | 34 | 0 | 90.9 | 736 | 84.25 | 0.3919 | over-abstain |
| sft | oracle_notes | minimal | 95.2 | 97.01 | 17 | 5 | 68.48 | 736 | 99.92 | 0.3949 | clean |
| sft | oracle_notes | runtime | 94.7 | 96.33 | 19 | 8 | 66.98 | 736 | 99.43 | 0.4028 | clean |
| sft | oracle_notes | strict | 94 | 95.92 | 25 | 5 | 66.71 | 736 | 99.97 | 0.3986 | clean |
| sft | oracle_both | minimal | 94.4 | 97.69 | 11 | 6 | 92.78 | 734 | 99.86 | 0.3802 | trace gap |
| sft | oracle_both | runtime | 94.2 | 96.2 | 22 | 6 | 92.52 | 735 | 99.54 | 0.4069 | trace gap |
| sft | oracle_both | strict | 93.3 | 95.65 | 30 | 2 | 93.48 | 736 | 99.95 | 0.3852 | over-abstain |

## Variant-Level Summary

| Variant | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 12 | 43.04 | 80.54 | 55.75 | 87.5 | 43.06 | 52.02 | 0.2372 |
| sft | 12 | 93.35 | 96.41 | 23.67 | 2.75 | 78.92 | 91.79 | 0.3972 |

## Variant x Prompt-Family Summary

| Variant | Prompt Mode | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | 3 | 55.83 | 84.87 | 11.67 | 99.67 | 25.26 | 38.4 | 0.2541 |
| baseline | oracle_conflict | 3 | 27.57 | 65.67 | 184.33 | 68.33 | 50 | 38.59 | 0.2255 |
| baseline | oracle_notes | 3 | 57.97 | 86.19 | 11.67 | 90 | 30.3 | 64.64 | 0.2359 |
| baseline | oracle_both | 3 | 30.8 | 85.42 | 15.33 | 92 | 66.67 | 66.45 | 0.2334 |
| sft | e2e | 3 | 92.97 | 96.38 | 26.67 | 0 | 66.66 | 83.75 | 0.4061 |
| sft | oracle_conflict | 3 | 91.83 | 96.33 | 26.67 | 0.33 | 88.7 | 83.85 | 0.3932 |
| sft | oracle_notes | 3 | 94.63 | 96.42 | 20.33 | 6 | 67.39 | 99.77 | 0.3988 |
| sft | oracle_both | 3 | 93.97 | 96.51 | 21 | 4.67 | 92.93 | 99.78 | 0.3908 |

## Operationally Trustworthy Top Configurations

| Metric | Variant | Prompt Mode | Profile | Value | False Abstains | Missed Refusals | Conflict Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Contract OK | sft | oracle_notes | minimal | 95.2 | 17 | 5 | 736 |
| Contract OK | sft | oracle_notes | runtime | 94.7 | 19 | 8 | 736 |
| Contract OK | sft | oracle_both | minimal | 94.4 | 11 | 6 | 734 |
| Contract OK | sft | oracle_both | runtime | 94.2 | 22 | 6 | 735 |
| Contract OK | sft | oracle_notes | strict | 94 | 25 | 5 | 736 |
| Abstain Accuracy | sft | oracle_both | minimal | 97.69 | 11 | 6 | 734 |
| Abstain Accuracy | sft | oracle_conflict | minimal | 97.01 | 21 | 1 | 720 |
| Abstain Accuracy | sft | oracle_notes | minimal | 97.01 | 17 | 5 | 736 |
| Abstain Accuracy | sft | e2e | minimal | 96.6 | 25 | 0 | 734 |
| Abstain Accuracy | sft | oracle_conflict | runtime | 96.6 | 25 | 0 | 733 |
| Conflict Accuracy | sft | oracle_both | minimal | 92.78 | 11 | 6 | 734 |
| Conflict Accuracy | sft | oracle_both | runtime | 92.52 | 22 | 6 | 735 |
| Conflict Accuracy | sft | oracle_conflict | runtime | 88.68 | 25 | 0 | 733 |
| Conflict Accuracy | sft | oracle_conflict | minimal | 86.53 | 21 | 1 | 720 |
| Conflict Accuracy | sft | oracle_notes | minimal | 68.48 | 17 | 5 | 736 |
| Doc Micro Accuracy | baseline | oracle_notes | strict | 100 | 9 | 82 | 736 |
| Doc Micro Accuracy | sft | oracle_notes | strict | 99.97 | 25 | 5 | 736 |
| Doc Micro Accuracy | sft | oracle_notes | minimal | 99.92 | 17 | 5 | 736 |
| Doc Micro Accuracy | sft | oracle_both | minimal | 99.86 | 11 | 6 | 734 |
| Doc Micro Accuracy | sft | oracle_both | runtime | 99.54 | 22 | 6 | 735 |
| Token F1 | sft | oracle_both | runtime | 0.4069 | 22 | 6 | 735 |
| Token F1 | sft | e2e | minimal | 0.4068 | 25 | 0 | 734 |
| Token F1 | sft | oracle_notes | runtime | 0.4028 | 19 | 8 | 736 |
| Token F1 | sft | oracle_notes | strict | 0.3986 | 25 | 5 | 736 |
| Token F1 | sft | oracle_conflict | minimal | 0.3979 | 21 | 1 | 720 |

## Recommended Picks By Use Case

| Use Case | Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 | Why it wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best end-to-end overall | sft | e2e | minimal | 93.5 | 96.6 | 25 | 0 | 67.3 | 83.27 | 0.4068 | Best end-to-end trade-off on abstention control, structural reliability, and answer overlap. |
| Best conflict-type-focused oracle | sft | oracle_conflict | runtime | 93.5 | 96.6 | 25 | 0 | 88.68 | 83.59 | 0.3898 | Highest usable conflict accuracy within the oracle_conflict family while keeping refusal behavior and support sane. |
| Best doc-verdict-focused oracle | sft | oracle_notes | strict | 94 | 95.92 | 25 | 5 | 66.71 | 99.97 | 0.3986 | Strongest doc-verdict row once contract integrity and support are treated as first gates. |
| Best all-oracle overall | sft | oracle_both | minimal | 94.4 | 97.69 | 11 | 6 | 92.78 | 99.86 | 0.3802 | Best combined oracle trade-off after filtering for structural reliability. |

## Structural / Parsing Anomalies

| Variant | Prompt Mode | Profile | Contract OK % | Conflict Support | Doc Pairs | Think Count | Sentinel Count | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_conflict | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_conflict | runtime | 82.7 | 736 | 3670 | 736 | 736 | over-abstain |
| baseline | oracle_conflict | strict | 0 | 2 | 3691 | 736 | 736 | contract 0; heavy over-abstain; support drop |
| baseline | oracle_notes | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_both | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_both | runtime | 85.9 | 736 | 3701 | 736 | 736 | over-abstain |
| baseline | oracle_both | strict | 6.5 | 66 | 3701 | 736 | 736 | support drop |
| sft | e2e | runtime | 92.5 | 733 | 3689 | 736 | 736 | over-abstain |
| sft | e2e | strict | 92.9 | 732 | 3696 | 735 | 736 | over-abstain; trace gap |
| sft | oracle_conflict | runtime | 93.5 | 733 | 3686 | 735 | 736 | trace gap |
| sft | oracle_conflict | strict | 92.9 | 736 | 3701 | 736 | 736 | over-abstain |
| sft | oracle_both | minimal | 94.4 | 734 | 3651 | 735 | 735 | trace gap |
| sft | oracle_both | runtime | 94.2 | 735 | 3696 | 735 | 736 | trace gap |
| sft | oracle_both | strict | 93.3 | 736 | 3701 | 736 | 736 | over-abstain |

## Dominant SFT Confusion Boundaries

| Gold | Pred | Count |
| --- | --- | --- |
| No conflict | Complementary information | 266 |
| Complementary information | No conflict | 214 |
| Complementary information | Conflicting opinions or research outcomes | 110 |
| Conflicting opinions or research outcomes | Complementary information | 94 |
| Conflicting opinions or research outcomes | No conflict | 76 |
| No conflict | Conflict due to outdated information | 71 |
| Conflict due to outdated information | No conflict | 67 |
| Conflict due to misinformation | Complementary information | 61 |
| No conflict | Conflicting opinions or research outcomes | 57 |
| Conflict due to misinformation | No conflict | 46 |

## Key Findings

| Finding | Evidence |
| --- | --- |
| SFT vs baseline separation | Average `contract_ok_pct` is `93.35` for SFT versus `43.04` for baseline. Average abstain accuracy is `96.41%` for SFT versus `80.54%` for baseline. |
| Over-abstention reduction | Average false abstains drop from `55.75` in baseline to `23.67` in SFT. |
| Best SFT e2e row | `sft + e2e + minimal` is the strongest end-to-end row here with `contract_ok_pct = 93.5`, `abstain_acc = 96.6%`, `false_abstains = 25`, `missed_refusals = 0`, and `token_f1 = 0.4068`. |
| Main remaining conflict boundary | The most persistent SFT confusion in the available rows is `No conflict -> Complementary information` with aggregate count `266`. |

## Bottom Line

For this Qwen 2.5 32B matrix, the cleanest end-to-end operating point is `sft + e2e + minimal` under the current local results. The oracle families remain useful for stage-specific probing, but the report should be read through the lens of contract integrity, refusal calibration, and support completeness rather than headline conflict accuracy alone.
