# Qwen 2.5 7B Run K Benchmark Matrix Audit

## Scope

This audit covers the full 24-generation Qwen 2.5 7B Run K benchmark matrix on the held-out benchmark `benchmark_final_v2_holdout_clean_736.jsonl`: 12 baseline generations and 12 SFT generations, across `e2e`, `oracle_conflict`, `oracle_notes`, and `oracle_both`, each with `minimal`, `runtime`, and `strict` prompt profiles.

Important interpretation rule: some rows produce inflated conflict or doc metrics despite being structurally weak overall. In this audit, `contract_ok_pct`, abstention behavior, and parse completeness are treated as first-gate sanity checks before trusting downstream task metrics.

`Operationally trustworthy` in the tables below means: parse-complete, `contract_ok_pct >= 80`, and `false_abstains <= 25`. That filter removes rows that spike on a single metric while still being practically unusable.

## Full 24-Run Matrix

| Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Conflict Support | Doc Micro % | Token F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0.0 | 83.02 | 1 | 124 | 0.0 | 0 | 0.0 | 0.2551 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | e2e | runtime | 63.3 | 77.72 | 82 | 82 | 37.06 | 707 | 49.05 | 0.2682 | clean |
| baseline | e2e | strict | 67.4 | 70.92 | 169 | 45 | 35.25 | 729 | 39.73 | 0.254 | heavy over-abstain |
| baseline | oracle_conflict | minimal | 0.0 | 82.61 | 2 | 126 | 0.0 | 0 | 0.0 | 0.2228 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | oracle_conflict | runtime | 55.8 | 62.77 | 198 | 76 | 99.59 | 736 | 45.7 | 0.2811 | heavy over-abstain |
| baseline | oracle_conflict | strict | 0.0 | 65.35 | 207 | 48 | 0.0 | 0 | 43.15 | 0.2914 | contract 0; conflict unparsable; heavy over-abstain |
| baseline | oracle_notes | minimal | 0.0 | 82.61 | 2 | 126 | 0.0 | 0 | 0.0 | 0.2192 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | oracle_notes | runtime | 79.3 | 80.3 | 100 | 45 | 42.99 | 735 | 93.03 | 0.2862 | heavy over-abstain |
| baseline | oracle_notes | strict | 71.3 | 72.96 | 166 | 33 | 44.78 | 728 | 99.81 | 0.227 | heavy over-abstain |
| baseline | oracle_both | minimal | 0.0 | 82.88 | 0 | 126 | 0.0 | 0 | 0.0 | 0.1898 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | oracle_both | runtime | 67.5 | 68.07 | 179 | 56 | 99.45 | 733 | 95.75 | 0.2967 | heavy over-abstain |
| baseline | oracle_both | strict | 1.5 | 69.43 | 198 | 27 | 100.0 | 26 | 99.89 | 0.2248 | heavy over-abstain |
| sft | e2e | minimal | 93.6 | 95.79 | 18 | 13 | 60.14 | 735 | 80.44 | 0.3296 | clean |
| sft | e2e | runtime | 92.8 | 96.47 | 19 | 7 | 60.74 | 731 | 79.74 | 0.3265 | clean |
| sft | e2e | strict | 93.6 | 96.74 | 19 | 5 | 62.43 | 732 | 79.46 | 0.3411 | trace gap |
| sft | oracle_conflict | minimal | 91.7 | 95.52 | 15 | 18 | 75.17 | 725 | 79.63 | 0.3221 | trace gap |
| sft | oracle_conflict | runtime | 93.3 | 96.47 | 13 | 13 | 76.77 | 736 | 80.49 | 0.3201 | clean |
| sft | oracle_conflict | strict | 93.1 | 95.92 | 22 | 8 | 76.6 | 735 | 79.65 | 0.3267 | trace gap |
| sft | oracle_notes | minimal | 65.2 | 94.97 | 7 | 30 | 58.46 | 520 | 99.89 | 0.337 | trace gap; support drop |
| sft | oracle_notes | runtime | 94.6 | 95.52 | 3 | 30 | 64.53 | 733 | 97.57 | 0.332 | clean |
| sft | oracle_notes | strict | 94.2 | 94.84 | 9 | 29 | 62.99 | 735 | 99.78 | 0.3375 | trace gap |
| sft | oracle_both | minimal | 83.7 | 94.02 | 3 | 41 | 76.7 | 661 | 99.82 | 0.3246 | trace gap; support drop |
| sft | oracle_both | runtime | 94.3 | 94.7 | 4 | 35 | 78.91 | 735 | 97.54 | 0.3259 | clean |
| sft | oracle_both | strict | 94.7 | 95.52 | 6 | 27 | 76.77 | 736 | 99.73 | 0.3259 | clean |

## Variant-Level Summary

| Variant | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 12 | 33.84 | 74.89 | 108.67 | 76.17 | 38.26 | 47.18 | 0.2514 |
| sft | 12 | 90.4 | 95.54 | 11.5 | 21.33 | 69.18 | 89.48 | 0.3291 |

## Variant x Prompt-Family Summary

| Variant | Prompt Mode | Rows | Contract OK % | Abstain Acc % | Avg False Abstains | Avg Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | 3 | 43.57 | 77.22 | 84.0 | 83.67 | 24.1 | 29.59 | 0.2591 |
| baseline | oracle_conflict | 3 | 18.6 | 70.24 | 135.67 | 83.33 | 33.2 | 29.62 | 0.2651 |
| baseline | oracle_notes | 3 | 50.2 | 78.62 | 89.33 | 68.0 | 29.26 | 64.28 | 0.2441 |
| baseline | oracle_both | 3 | 23.0 | 73.46 | 125.67 | 69.67 | 66.48 | 65.21 | 0.2371 |
| sft | e2e | 3 | 93.33 | 96.33 | 18.67 | 8.33 | 61.1 | 79.88 | 0.3324 |
| sft | oracle_conflict | 3 | 92.7 | 95.97 | 16.67 | 13.0 | 76.18 | 79.92 | 0.323 |
| sft | oracle_notes | 3 | 84.67 | 95.11 | 6.33 | 29.67 | 61.99 | 99.08 | 0.3355 |
| sft | oracle_both | 3 | 90.9 | 94.75 | 4.33 | 34.33 | 77.46 | 99.03 | 0.3255 |

## Operationally Trustworthy Top Configurations

| Metric | Variant | Prompt Mode | Profile | Value | False Abstains | Missed Refusals | Conflict Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Contract OK | sft | oracle_both | strict | 94.7 | 6 | 27 | 736 |
| Contract OK | sft | oracle_notes | runtime | 94.6 | 3 | 30 | 733 |
| Contract OK | sft | oracle_both | runtime | 94.3 | 4 | 35 | 735 |
| Contract OK | sft | oracle_notes | strict | 94.2 | 9 | 29 | 735 |
| Contract OK | sft | e2e | minimal | 93.6 | 18 | 13 | 735 |
| Abstain Accuracy | sft | e2e | strict | 96.74 | 19 | 5 | 732 |
| Abstain Accuracy | sft | e2e | runtime | 96.47 | 19 | 7 | 731 |
| Abstain Accuracy | sft | oracle_conflict | runtime | 96.47 | 13 | 13 | 736 |
| Abstain Accuracy | sft | oracle_conflict | strict | 95.92 | 22 | 8 | 735 |
| Abstain Accuracy | sft | e2e | minimal | 95.79 | 18 | 13 | 735 |
| Conflict Accuracy | sft | oracle_both | runtime | 78.91 | 4 | 35 | 735 |
| Conflict Accuracy | sft | oracle_conflict | runtime | 76.77 | 13 | 13 | 736 |
| Conflict Accuracy | sft | oracle_both | strict | 76.77 | 6 | 27 | 736 |
| Conflict Accuracy | sft | oracle_conflict | strict | 76.6 | 22 | 8 | 735 |
| Conflict Accuracy | sft | oracle_notes | runtime | 64.53 | 3 | 30 | 733 |
| Doc Micro Accuracy | sft | oracle_notes | strict | 99.78 | 9 | 29 | 735 |
| Doc Micro Accuracy | sft | oracle_both | strict | 99.73 | 6 | 27 | 736 |
| Doc Micro Accuracy | sft | oracle_notes | runtime | 97.57 | 3 | 30 | 733 |
| Doc Micro Accuracy | sft | oracle_both | runtime | 97.54 | 4 | 35 | 735 |
| Doc Micro Accuracy | sft | oracle_conflict | runtime | 80.49 | 13 | 13 | 736 |
| Token F1 | sft | e2e | strict | 0.3411 | 19 | 5 | 732 |
| Token F1 | sft | oracle_notes | strict | 0.3375 | 9 | 29 | 735 |
| Token F1 | sft | oracle_notes | runtime | 0.332 | 3 | 30 | 733 |
| Token F1 | sft | e2e | minimal | 0.3296 | 18 | 13 | 735 |
| Token F1 | sft | oracle_conflict | strict | 0.3267 | 22 | 8 | 735 |

## Recommended Picks By Use Case

| Use Case | Variant | Prompt Mode | Profile | Contract OK % | Abstain Acc % | False Abstains | Missed Refusals | Conflict Acc % | Doc Micro % | Token F1 | Why it wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best end-to-end overall | sft | e2e | strict | 93.6 | 96.74 | 19 | 5 | 62.43 | 79.46 | 0.3411 | Best clean e2e trade-off: highest e2e abstain accuracy, best e2e token overlap, and only 5 missed refusals. |
| Best conflict-type-focused oracle | sft | oracle_conflict | runtime | 93.3 | 96.47 | 13 | 13 | 76.77 | 80.49 | 0.3201 | Highest clean conflict accuracy in the oracle_conflict family with balanced refusal errors and no support collapse. |
| Best doc-verdict-focused oracle | sft | oracle_notes | strict | 94.2 | 94.84 | 9 | 29 | 62.99 | 99.78 | 0.3375 | Strongest high-support oracle_notes row for doc verdicts; chosen over oracle_notes minimal because the minimal row loses too much conflict support. |
| Best all-oracle overall | sft | oracle_both | strict | 94.7 | 95.52 | 6 | 27 | 76.77 | 99.73 | 0.3259 | Best combined oracle setting once contract integrity and abstention sanity are treated as first gates. |

## Structural / Parsing Anomalies

| Variant | Prompt Mode | Profile | Contract OK % | Conflict Support | Doc Pairs | Think Count | Sentinel Count | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | e2e | minimal | 0.0 | 0 | 0 | 0 | 0 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | e2e | strict | 67.4 | 729 | 3695 | 736 | 736 | heavy over-abstain |
| baseline | oracle_conflict | minimal | 0.0 | 0 | 0 | 0 | 0 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | oracle_conflict | runtime | 55.8 | 736 | 3525 | 736 | 736 | heavy over-abstain |
| baseline | oracle_conflict | strict | 0.0 | 0 | 3692 | 736 | 736 | contract 0; conflict unparsable; heavy over-abstain |
| baseline | oracle_notes | minimal | 0.0 | 0 | 0 | 0 | 0 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | oracle_notes | runtime | 79.3 | 735 | 3674 | 736 | 736 | heavy over-abstain |
| baseline | oracle_notes | strict | 71.3 | 728 | 3701 | 736 | 736 | heavy over-abstain |
| baseline | oracle_both | minimal | 0.0 | 0 | 0 | 0 | 0 | trace gap; no doc eval; contract 0; conflict unparsable |
| baseline | oracle_both | runtime | 67.5 | 733 | 3670 | 736 | 736 | heavy over-abstain |
| baseline | oracle_both | strict | 1.5 | 26 | 3701 | 736 | 736 | heavy over-abstain |
| sft | e2e | strict | 93.6 | 732 | 3696 | 735 | 736 | trace gap |
| sft | oracle_conflict | minimal | 91.7 | 725 | 3643 | 726 | 736 | trace gap |
| sft | oracle_conflict | strict | 93.1 | 735 | 3696 | 735 | 736 | trace gap |
| sft | oracle_notes | minimal | 65.2 | 520 | 2676 | 522 | 735 | trace gap; support drop |
| sft | oracle_notes | strict | 94.2 | 735 | 3696 | 735 | 735 | trace gap |
| sft | oracle_both | minimal | 83.7 | 661 | 3351 | 662 | 735 | trace gap; support drop |

## Dominant SFT Confusion Boundaries

This aggregates the top reported confusion pairs across the operationally trustworthy SFT rows only.

| Gold | Pred | Count |
| --- | --- | --- |
| No conflict | Complementary information | 724 |
| Complementary information | No conflict | 222 |
| Conflicting opinions or research outcomes | Complementary information | 221 |
| Complementary information | Conflicting opinions or research outcomes | 129 |
| Conflict due to misinformation | Complementary information | 116 |
| No conflict | Conflict due to outdated information | 31 |
| Conflict due to misinformation | No conflict | 8 |

## Key Findings

| Finding | Evidence |
| --- | --- |
| SFT is the real winner across the matrix. | Average `contract_ok_pct` is `90.4` for SFT versus `33.84` for baseline. Average abstain accuracy is `95.54%` for SFT versus `74.89%` for baseline. |
| Qwen 7B Run K has solved the extreme over-abstention issue relative to the baseline behaviors. | Best SFT e2e rows stay around `18-19` false abstains, while many baseline rows produce `82-207` false abstains. |
| The best end-to-end row is `SFT + e2e + strict`. | It has the highest e2e abstain accuracy (`96.74%`), the lowest e2e missed refusals (`5`), and the best e2e token overlap (`0.3411`) while keeping `contract_ok_pct = 93.6`. |
| `oracle_conflict` is the strongest stage-2 supervision family for SFT. | Its three SFT rows average `76.18%` conflict accuracy with `92.7%` contract OK and only `16.67` average false abstains. |
| `oracle_notes` and `oracle_both` dramatically improve doc-verdict metrics, but they are less refusal-sensitive. | SFT `oracle_notes` and `oracle_both` average around `99%` and `99.03%` doc micro respectively, but also miss more gold refusals on average (`29.67` and `34.33`). |
| Several baseline rows are structurally non-usable, not merely low-performing. | All four baseline minimal rows have `contract_ok_pct = 0`, `doc_pairs = 0`, `conflict_support = 0`, and zero trace presence. |
| Some oracle baseline conflict scores are misleadingly high. | `baseline oracle_both strict` shows `100%` conflict accuracy, but only on `26` supported rows and with `198` false abstains plus `1.5%` contract OK. That is not a trustworthy overall result. |
| The main remaining SFT stage-2 confusion boundary is `No conflict` versus `Complementary information`. | Across operationally trustworthy SFT rows, the biggest aggregate confusion is `No conflict -> Complementary information` (`724`), followed by `Conflicting opinions or research outcomes -> Complementary information` (`221`). |
| `Conflict due to misinformation` remains the hardest conflict label. | Even the strong SFT rows repeatedly confuse it with `Complementary information`; the e2e minimal row, for example, shows just `0.087` F1 for that class in the detailed conflict report. |
| Minimal oracle SFT rows need caution even when headline metrics look good. | `sft oracle_notes minimal` and `sft oracle_both minimal` have support drops (`520` and `661` conflict support respectively) and trace gaps, so they should not be treated as the cleanest reference rows. |

## Bottom Line

For genuine end-to-end benchmarking, the cleanest Qwen 7B Run K choice is `SFT + e2e + strict`, with `SFT + e2e + runtime` as a close second and `SFT + e2e + minimal` still strong but slightly weaker on abstention and overlap. For oracle-style probing, `SFT + oracle_conflict + runtime/strict` is the cleanest stage-2 family, while `SFT + oracle_notes` and `SFT + oracle_both` are best treated as upper-bound stage-specific probes rather than end-to-end replacements.
