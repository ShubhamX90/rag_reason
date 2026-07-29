# Mistral 7B Run L Benchmark Matrix Audit

## Scope

This audit covers the locally completed benchmark generations stored under `final_model_outputs/mistral7b` and evaluated against `data/splits/benchmark_final_v2_holdout_clean_736.jsonl`. The matrix structure is baseline vs SFT across `e2e`, `oracle_conflict`, `oracle_notes`, and `oracle_both`, each with `minimal`, `runtime`, and `strict` prompt profiles.

Completed rows available locally: `24` / `24`.

Important interpretation rule: `contract_ok_pct`, abstention behavior, and support completeness are treated as first-gate sanity checks before trusting downstream task metrics.

`Operationally trustworthy` in this audit means `contract_ok_pct >= 80`, `false_abstains <= 25`, and `conflict_support >= 700`.

## Full Matrix

| Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Conflict Support | Doc Micro % | Token F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0 | 82.34 | 2 | 128 | 0 | 0 | 0 | 0.26 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | e2e | runtime | 1 | 81.66 | 25 | 110 | 35.29 | 17 | 44 | 0.0946 | trace gap; support drop |
| baseline | e2e | strict | 79.8 | 82.61 | 31 | 97 | 34.56 | 735 | 45.38 | 0.2411 | over-abstain |
| baseline | oracle_conflict | minimal | 0 | 82.34 | 2 | 128 | 0 | 0 | 0 | 0.2132 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_conflict | runtime | 1 | 76.49 | 68 | 105 | 100 | 11 | 60 | 0.0854 | over-abstain; trace gap; support drop |
| baseline | oracle_conflict | strict | 0 | 78.94 | 69 | 86 | 0 | 0 | 47.14 | 0.1433 | contract 0; conflict unparsable; over-abstain; trace gap |
| baseline | oracle_notes | minimal | 0 | 82.47 | 1 | 128 | 0 | 0 | 0 | 0.2214 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_notes | runtime | 5.3 | 81.39 | 48 | 89 | 19.4 | 67 | 97.99 | 0.1048 | over-abstain; trace gap; support drop |
| baseline | oracle_notes | strict | 70.4 | 83.02 | 21 | 104 | 38.58 | 648 | 99.71 | 0.2493 | support drop |
| baseline | oracle_both | minimal | 0 | 82.47 | 2 | 127 | 0 | 0 | 0 | 0.1832 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_both | runtime | 3 | 80.03 | 39 | 108 | 100 | 36 | 98.15 | 0.0885 | over-abstain; trace gap; support drop |
| baseline | oracle_both | strict | 27 | 81.11 | 32 | 107 | 99.63 | 270 | 99.74 | 0.2532 | over-abstain; support drop |
| sft | e2e | minimal | 90.8 | 93.34 | 49 | 0 | 60.73 | 736 | 78.47 | 0.3305 | over-abstain |
| sft | e2e | runtime | 89.8 | 92.8 | 53 | 0 | 61.41 | 736 | 78.38 | 0.3274 | over-abstain |
| sft | e2e | strict | 90.1 | 92.66 | 54 | 0 | 61.01 | 736 | 78.44 | 0.3202 | over-abstain |
| sft | oracle_conflict | minimal | 87 | 93.34 | 49 | 0 | 67.48 | 701 | 78.72 | 0.314 | over-abstain; trace gap |
| sft | oracle_conflict | runtime | 90.5 | 93.34 | 49 | 0 | 66.3 | 736 | 78.14 | 0.3278 | over-abstain |
| sft | oracle_conflict | strict | 88.5 | 91.58 | 62 | 0 | 67.53 | 736 | 78.71 | 0.334 | over-abstain |
| sft | oracle_notes | minimal | 88 | 94.02 | 41 | 3 | 60.45 | 708 | 99.27 | 0.3129 | over-abstain; trace gap |
| sft | oracle_notes | runtime | 92.4 | 94.43 | 38 | 3 | 62.72 | 735 | 95.11 | 0.3232 | over-abstain |
| sft | oracle_notes | strict | 89.8 | 92.12 | 56 | 2 | 60.68 | 735 | 99.32 | 0.3229 | over-abstain |
| sft | oracle_both | minimal | 90.5 | 93.75 | 43 | 3 | 67.31 | 728 | 99.21 | 0.3258 | over-abstain; trace gap |
| sft | oracle_both | runtime | 92.4 | 94.7 | 37 | 2 | 66.62 | 734 | 95.16 | 0.3186 | over-abstain |
| sft | oracle_both | strict | 90.6 | 93.07 | 50 | 1 | 66.58 | 736 | 99.3 | 0.3157 | over-abstain |

## Variant-Level Summary

| Variant | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 12 | 15.62 | 81.24 | 28.33 | 109.75 | 35.62 | 49.34 | 0.1782 |
| sft | 12 | 90.03 | 93.26 | 48.42 | 1.17 | 64.07 | 88.19 | 0.3227 |

## Variant x Prompt-Family Summary

| Variant | Prompt Mode | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | 3 | 26.93 | 82.2 | 19.33 | 111.67 | 23.28 | 29.79 | 0.1986 |
| baseline | oracle_conflict | 3 | 0.33 | 79.26 | 46.33 | 106.33 | 33.33 | 35.71 | 0.1473 |
| baseline | oracle_notes | 3 | 25.23 | 82.29 | 23.33 | 107 | 19.33 | 65.9 | 0.1918 |
| baseline | oracle_both | 3 | 10 | 81.2 | 24.33 | 114 | 66.54 | 65.96 | 0.175 |
| sft | e2e | 3 | 90.23 | 92.93 | 52 | 0 | 61.05 | 78.43 | 0.326 |
| sft | oracle_conflict | 3 | 88.67 | 92.75 | 53.33 | 0 | 67.1 | 78.52 | 0.3253 |
| sft | oracle_notes | 3 | 90.07 | 93.52 | 45 | 2.67 | 61.28 | 97.9 | 0.3197 |
| sft | oracle_both | 3 | 91.17 | 93.84 | 43.33 | 2 | 66.84 | 97.89 | 0.32 |

## Operationally Trustworthy Top Configurations

No rows met the operationally trustworthy threshold.

## Recommended Picks By Use Case

| Use Case | Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 | Why it wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best end-to-end overall | sft | e2e | minimal | 90.8 | 93.34 | 49 | 0 | 60.73 | 78.47 | 0.3305 | Best available row in this family, but it did not clear the report's operational trust threshold. |
| Best conflict-type-focused oracle | sft | oracle_conflict | strict | 88.5 | 91.58 | 62 | 0 | 67.53 | 78.71 | 0.334 | Best available row in this family, but it did not clear the report's operational trust threshold. |
| Best doc-verdict-focused oracle | sft | oracle_notes | strict | 89.8 | 92.12 | 56 | 2 | 60.68 | 99.32 | 0.3229 | Best available row in this family, but it did not clear the report's operational trust threshold. |
| Best all-oracle overall | sft | oracle_both | runtime | 92.4 | 94.7 | 37 | 2 | 66.62 | 95.16 | 0.3186 | Best available row in this family, but it did not clear the report's operational trust threshold. |

## Structural / Parsing Anomalies

| Variant | Prompt Mode | Profile | Contract OK % | Conflict Support | Doc Pairs | Think Count | Sentinel Count | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | e2e | runtime | 1 | 17 | 50 | 17 | 482 | trace gap; support drop |
| baseline | e2e | strict | 79.8 | 735 | 3629 | 736 | 736 | over-abstain |
| baseline | oracle_conflict | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_conflict | runtime | 1 | 11 | 60 | 11 | 585 | over-abstain; trace gap; support drop |
| baseline | oracle_conflict | strict | 0 | 0 | 1292 | 250 | 736 | contract 0; conflict unparsable; over-abstain; trace gap |
| baseline | oracle_notes | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_notes | runtime | 5.3 | 67 | 249 | 67 | 639 | over-abstain; trace gap; support drop |
| baseline | oracle_notes | strict | 70.4 | 648 | 3480 | 736 | 736 | support drop |
| baseline | oracle_both | minimal | 0 | 0 | 0 | 0 | 0 | contract 0; conflict unparsable; no doc eval; trace gap |
| baseline | oracle_both | runtime | 3 | 36 | 162 | 36 | 657 | over-abstain; trace gap; support drop |
| baseline | oracle_both | strict | 27 | 270 | 3509 | 736 | 736 | over-abstain; support drop |
| sft | e2e | minimal | 90.8 | 736 | 3701 | 736 | 736 | over-abstain |
| sft | e2e | runtime | 89.8 | 736 | 3701 | 736 | 736 | over-abstain |
| sft | e2e | strict | 90.1 | 736 | 3701 | 736 | 736 | over-abstain |
| sft | oracle_conflict | minimal | 87 | 701 | 3492 | 701 | 736 | over-abstain; trace gap |
| sft | oracle_conflict | runtime | 90.5 | 736 | 3701 | 736 | 736 | over-abstain |
| sft | oracle_conflict | strict | 88.5 | 736 | 3701 | 736 | 736 | over-abstain |
| sft | oracle_notes | minimal | 88 | 708 | 3577 | 709 | 736 | over-abstain; trace gap |
| sft | oracle_notes | runtime | 92.4 | 735 | 3701 | 736 | 736 | over-abstain |
| sft | oracle_notes | strict | 89.8 | 735 | 3701 | 736 | 736 | over-abstain |
| sft | oracle_both | minimal | 90.5 | 728 | 3663 | 728 | 736 | over-abstain; trace gap |
| sft | oracle_both | runtime | 92.4 | 734 | 3701 | 736 | 736 | over-abstain |
| sft | oracle_both | strict | 90.6 | 736 | 3701 | 736 | 736 | over-abstain |

## Dominant SFT Confusion Boundaries

| Gold | Pred | Count |
| --- | --- | --- |
| No conflict | Complementary information | 770 |
| Complementary information | No conflict | 426 |
| Complementary information | Conflicting opinions or research outcomes | 351 |
| Complementary information | Conflict due to outdated information | 247 |
| Conflicting opinions or research outcomes | Complementary information | 233 |
| No conflict | Conflict due to outdated information | 190 |
| No conflict | Conflicting opinions or research outcomes | 180 |
| Conflict due to misinformation | No conflict | 122 |
| Conflict due to misinformation | Complementary information | 88 |
| Conflict due to outdated information | No conflict | 44 |

## Key Findings

| Finding | Evidence |
| --- | --- |
| SFT vs baseline separation | Average `contract_ok_pct` is `90.03` for SFT versus `15.62` for baseline. Average abstain accuracy is `93.26%` for SFT versus `81.24%` for baseline. |
| Over-abstention warning | Average false abstains rise from `28.33` in baseline to `48.42` in SFT, so this model family is still over-abstaining despite stronger structure and answer quality. |
| Best SFT e2e row | `sft + e2e + minimal` is the strongest end-to-end row here with `contract_ok_pct = 90.8`, `abstain_acc = 93.34%`, `false_abstains = 49`, `missed_refusals = 0`, and `token_f1 = 0.3305`. |
| Main remaining conflict boundary | The most persistent SFT confusion in the available rows is `No conflict -> Complementary information` with aggregate count `770`. |

## Bottom Line

For this Mistral 7B matrix, `sft + e2e + minimal` is only the best available current row, not a fully trustworthy operating point. The main blocker is still refusal calibration rather than stage-2 capability alone.
