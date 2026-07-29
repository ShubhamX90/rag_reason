# Run J Deep Audit - 2026-06-25

Artifacts audited: 7B val, 7B benchmark, 32B val, 32B benchmark for Run J minimal E2E.

## Run J Val Canon

| rows | answerable | refusal | refusal_rate |
| --- | --- | --- | --- |
| 81 | 59 | 22 | 27.16% |

Overall conflict distribution

| label | count |
| --- | --- |
| Conflicting opinions or research outcomes | 14 |
| Complementary information | 24 |
| No conflict | 28 |
| Conflict due to outdated information | 13 |
| Conflict due to misinformation | 2 |

Non-refusal conflict distribution

| label | count |
| --- | --- |
| Conflicting opinions or research outcomes | 12 |
| Complementary information | 15 |
| No conflict | 19 |
| Conflict due to outdated information | 11 |
| Conflict due to misinformation | 2 |

Doc-count distribution

| doc_count | count |
| --- | --- |
| 4 | 4 |
| 5 | 43 |
| 6 | 1 |
| 7 | 2 |
| 8 | 3 |
| 9 | 13 |
| 10 | 9 |
| 11 | 4 |
| 12 | 1 |
| 14 | 1 |

Evidence buckets

| bucket | count |
| --- | --- |
| support_present | 55 |
| partial_only | 25 |
| no_explicit_support_note | 1 |

## Benchmark Holdout Canon

| rows | answerable | refusal | refusal_rate |
| --- | --- | --- | --- |
| 736 | 608 | 128 | 17.39% |

Overall conflict distribution

| label | count |
| --- | --- |
| Complementary information | 221 |
| Conflicting opinions or research outcomes | 109 |
| No conflict | 211 |
| Conflict due to misinformation | 37 |
| Conflict due to outdated information | 158 |

Non-refusal conflict distribution

| label | count |
| --- | --- |
| Complementary information | 176 |
| Conflicting opinions or research outcomes | 96 |
| No conflict | 154 |
| Conflict due to misinformation | 37 |
| Conflict due to outdated information | 145 |

Doc-count distribution

| doc_count | count |
| --- | --- |
| 2 | 2 |
| 3 | 1 |
| 4 | 78 |
| 5 | 631 |
| 6 | 1 |
| 7 | 1 |
| 8 | 3 |
| 10 | 19 |

Evidence buckets

| bucket | count |
| --- | --- |
| support_present | 550 |
| partial_only | 185 |
| no_explicit_support_note | 1 |

## Headline Metrics

| run | contract_ok_pct | citation_pass_pct | contract_abstain_acc_pct | conflict_acc_pct | conflict_support | doc_micro_pct | doc_pairs | final_abstain_acc_pct | pred_abstain | avg_token_f1 | avg_rougeL_f1 | avg_sentence_cov |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7B Val | 98.8 | 55.2 | 98.8 | 64.2 | 81 | 76.09 | 552 | 98.77 | 23 | 0.4334 | 0.3097 | 0.3823 |
| 7B Benchmark | 92.5 | 66.3 | 94.2 | 62.07 | 733 | 77.96 | 3688 | 94.29 | 164 | 0.3555 | 0.3215 | 0.6461 |
| 32B Val | 95.1 | 79.3 | 100.0 | 67.09 | 79 | 83.27 | 532 | 100.0 | 22 | 0.4578 | 0.3362 | 0.6265 |
| 32B Benchmark | 92.8 | 55.1 | 95.0 | 62.55 | 729 | 82.05 | 3687 | 94.97 | 149 | 0.3945 | 0.362 | 0.6163 |

## 7B Training-Time Dev Selection

| epoch | macro_f1 | doc_acc | format_ok | abstain_acc | false_abstain_partial_only | false_abstain_with_support | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.5774 | 0.7663 | 0.9877 | 0.9877 | 1 | 0 | 0.7802 |
| 2.0 | 0.5454 | 0.8007 | 1.0000 | 0.9630 | 1 | 2 | 0.7625 |

Selected checkpoint epoch: 1.0

## 32B Training-Time Dev Selection

| epoch | macro_f1 | doc_acc | format_ok | abstain_acc | false_abstain_partial_only | false_abstain_with_support | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.5740 | 0.8170 | 1.0000 | 1.0000 | 0 | 0 | 0.8180 |
| 2.0 | 0.5994 | 0.8333 | 1.0000 | 0.9753 | 0 | 2 | 0.8133 |

Selected checkpoint epoch: 1.0

## 7B Val

Abstain confusion

| TP | TN | FP | FN | pred_abstain | gold_abstain | accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 22 | 58 | 1 | 0 | 23 | 22 | 98.77 |

False-abstain labels

| gold_label | count |
| --- | --- |
| Conflict due to misinformation | 1 |

Hard subgroup slices

| slice | n | answered_rate_answerable | false_abstain_rate | missed_refusal_rate | parsed_conflict_acc_answerable | avg_doc_acc | avg_token_f1_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| non_refusal | 59 | 98.31% | 1.69% | NA | 66.10% | 0.7613 | 0.4334 |
| answerable_support_present | 54 | 100.00% | 0.00% | NA | 64.81% | 0.7430 | 0.4413 |
| answerable_partial_only | 5 | 80.00% | 20.00% | NA | 80.00% | 0.9600 | 0.3269 |
| answerable_5doc_support_present | 17 | 100.00% | 0.00% | NA | 41.18% | 0.7765 | 0.4067 |
| answerable_5doc_partial_only | 4 | 75.00% | 25.00% | NA | 75.00% | 0.9500 | 0.2494 |
| answerable_5doc_partial_only_complementary | 2 | 100.00% | 0.00% | NA | 100.00% | 1.0000 | 0.2074 |
| answerable_5doc_partial_only_conflicting | 1 | 100.00% | 0.00% | NA | 100.00% | 0.8000 | 0.3333 |
| answerable_5doc_partial_only_misinformation | 1 | 0.00% | 100.00% | NA | 0.00% | 1.0000 | NA |
| refusal | 22 | NA | NA | 0.00% | NA | 0.8364 | NA |

Top answerable conflict confusions

| gold | pred | count |
| --- | --- | --- |
| No conflict | Complementary information | 6 |
| Complementary information | No conflict | 4 |
| No conflict | Conflict due to outdated information | 2 |
| Conflicting opinions or research outcomes | Complementary information | 2 |
| Conflict due to outdated information | No conflict | 2 |
| Complementary information | Conflicting opinions or research outcomes | 1 |
| Conflict due to misinformation | No conflict | 1 |
| Conflicting opinions or research outcomes | No conflict | 1 |
| Conflict due to misinformation | Complementary information | 1 |

Conflict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| No conflict | 19 | 38 | 15 | 9 |
| Complementary information | 15 | 46 | 11 | 9 |
| Conflicting opinions or research outcomes | 9 | 66 | 1 | 5 |
| Conflict due to outdated information | 9 | 66 | 2 | 4 |
| Conflict due to misinformation | 0 | 79 | 0 | 2 |

Doc verdict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| supports | 140 | 310 | 19 | 83 |
| partially supports | 233 | 190 | 110 | 19 |
| irrelevant | 47 | 472 | 3 | 30 |

Worst doc-verdict rows

| id | doc_count | bucket | gold_conflict | doc_acc |
| --- | --- | --- | --- | --- |
| #0206 | 11 | support_present | Conflicting opinions or research outcomes | 0.2727 |
| #0427 | 8 | support_present | Complementary information | 0.3750 |
| hotpotqa_0043 | 10 | support_present | Complementary information | 0.4000 |
| #0343 | 5 | support_present | No conflict | 0.4000 |
| #0638 | 5 | partial_only | No conflict | 0.4000 |
| #0542 | 5 | no_explicit_support_note | No conflict | 0.4000 |
| #0133 | 7 | support_present | Conflicting opinions or research outcomes | 0.4286 |
| #0300 | 9 | support_present | Conflicting opinions or research outcomes | 0.4444 |
| #0333 | 9 | support_present | Conflict due to outdated information | 0.4444 |
| #0394 | 10 | support_present | Conflict due to outdated information | 0.5000 |

Lowest-overlap answered rows

| id | gold_conflict | doc_count | token_f1 | rouge_l_f1 |
| --- | --- | --- | --- | --- |
| conflictingqa_d3b9d370384c | Complementary information | 5 | 0.0000 | 0.0000 |
| hotpotqa_0043 | Complementary information | 10 | 0.0230 | 0.0230 |
| wikirevision_0048 | Conflict due to outdated information | 4 | 0.0278 | 0.0278 |
| #0187 | Complementary information | 9 | 0.0500 | 0.0500 |
| #0104 | Conflict due to outdated information | 9 | 0.0566 | 0.0566 |
| qacc_402e42870e74 | No conflict | 5 | 0.1333 | 0.1333 |
| conflictingqa_f8726a647020 | Complementary information | 5 | 0.1795 | 0.1667 |
| freshqa_98056834d12d | Conflict due to outdated information | 5 | 0.1818 | 0.1818 |
| qacc_e0db7d73e48f | No conflict | 5 | 0.1867 | 0.1867 |
| situatedqa_temp_b30fa7ac21ea | Conflict due to outdated information | 5 | 0.1875 | 0.1250 |

False-abstain example IDs

| id | gold_conflict | doc_count | bucket |
| --- | --- | --- | --- |
| freshqa_6f48c26f2ccd | Conflict due to misinformation | 5 | partial_only |

## 7B Benchmark

Abstain confusion

| TP | TN | FP | FN | pred_abstain | gold_abstain | accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 125 | 569 | 39 | 3 | 164 | 128 | 94.29 |

False-abstain labels

| gold_label | count |
| --- | --- |
| Conflicting opinions or research outcomes | 3 |
| Conflict due to misinformation | 10 |
| Conflict due to outdated information | 4 |
| Complementary information | 15 |
| No conflict | 7 |

Hard subgroup slices

| slice | n | answered_rate_answerable | false_abstain_rate | missed_refusal_rate | parsed_conflict_acc_answerable | avg_doc_acc | avg_token_f1_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| non_refusal | 608 | 93.59% | 6.41% | NA | 63.64% | 0.7656 | 0.3555 |
| answerable_support_present | 531 | 95.86% | 4.14% | NA | 63.02% | 0.7384 | 0.3681 |
| answerable_partial_only | 77 | 77.92% | 22.08% | NA | 68.00% | 0.9532 | 0.2615 |
| answerable_5doc_support_present | 426 | 95.77% | 4.23% | NA | 55.87% | 0.7352 | 0.3432 |
| answerable_5doc_partial_only | 77 | 77.92% | 22.08% | NA | 68.00% | 0.9532 | 0.2615 |
| answerable_5doc_partial_only_complementary | 35 | 77.14% | 22.86% | NA | 66.67% | 0.9200 | 0.2379 |
| answerable_5doc_partial_only_conflicting | 31 | 93.55% | 6.45% | NA | 87.10% | 0.9806 | 0.2819 |
| answerable_5doc_partial_only_misinformation | 11 | 36.36% | 63.64% | NA | 18.18% | 0.9818 | 0.2727 |
| refusal | 128 | NA | NA | 2.34% | NA | 0.8344 | NA |

Top answerable conflict confusions

| gold | pred | count |
| --- | --- | --- |
| No conflict | Complementary information | 41 |
| Complementary information | No conflict | 38 |
| Conflict due to outdated information | No conflict | 26 |
| Conflicting opinions or research outcomes | Complementary information | 23 |
| Conflicting opinions or research outcomes | No conflict | 17 |
| Conflict due to misinformation | Complementary information | 15 |
| Complementary information | Conflicting opinions or research outcomes | 14 |
| Conflict due to misinformation | No conflict | 14 |
| No conflict | Conflict due to outdated information | 8 |
| Conflict due to outdated information | Complementary information | 7 |
| Conflict due to misinformation | Conflict due to outdated information | 4 |
| Complementary information | Conflict due to outdated information | 4 |

Conflict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| No conflict | 140 | 399 | 123 | 71 |
| Complementary information | 143 | 400 | 115 | 75 |
| Conflicting opinions or research outcomes | 54 | 606 | 18 | 55 |
| Conflict due to outdated information | 115 | 558 | 17 | 43 |
| Conflict due to misinformation | 3 | 691 | 5 | 34 |

Doc verdict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| supports | 714 | 2346 | 121 | 507 |
| partially supports | 1965 | 913 | 610 | 200 |
| irrelevant | 196 | 3304 | 82 | 106 |

Worst doc-verdict rows

| id | doc_count | bucket | gold_conflict | doc_acc |
| --- | --- | --- | --- | --- |
| hotpotqa_0064 | 10 | support_present | Complementary information | 0.0000 |
| hotpotqa_0062 | 10 | support_present | Complementary information | 0.2000 |
| conflictingqa_962d8f5d5574 | 5 | support_present | Complementary information | 0.2000 |
| conflictingqa_c34991d9897e | 5 | support_present | Complementary information | 0.2000 |
| conflictingqa_fad0d30903d2 | 5 | support_present | No conflict | 0.2000 |
| qacc_0bd7153f19ad | 5 | support_present | No conflict | 0.2000 |
| qacc_2e1b5edb5e0d | 5 | support_present | Conflicting opinions or research outcomes | 0.2000 |
| qacc_4387048ed24f | 5 | support_present | Conflicting opinions or research outcomes | 0.2000 |
| qacc_883303a2d535 | 5 | support_present | Conflict due to misinformation | 0.2000 |
| qacc_b1bd9518429b | 5 | support_present | No conflict | 0.2000 |

Lowest-overlap answered rows

| id | gold_conflict | doc_count | token_f1 | rouge_l_f1 |
| --- | --- | --- | --- | --- |
| conflictingqa_76956c2fba7d | Conflicting opinions or research outcomes | 5 | 0.0000 | 0.0000 |
| freshqa_4a98eba95e97 | No conflict | 5 | 0.0000 | 0.0000 |
| freshqa_6a45fadeb16b | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| freshqa_7bc92b47dc43 | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| freshqa_c3f10dc1632d | No conflict | 5 | 0.0000 | 0.0000 |
| freshqa_c7315f8b3029 | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| freshqa_c7ac9d61059a | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| qacc_0b75ed799d46 | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| qacc_4387048ed24f | Conflicting opinions or research outcomes | 5 | 0.0000 | 0.0000 |
| qacc_798b6853d20f | No conflict | 5 | 0.0000 | 0.0000 |

False-abstain example IDs

| id | gold_conflict | doc_count | bucket |
| --- | --- | --- | --- |
| freshqa_1fe3ba1b1cba | Conflicting opinions or research outcomes | 5 | support_present |
| freshqa_2e51f51132ee | Conflict due to misinformation | 5 | support_present |
| freshqa_7f1c3aae61a5 | Conflict due to outdated information | 5 | support_present |
| freshqa_a47283064972 | Conflict due to outdated information | 5 | support_present |
| qacc_17dc360eea55 | Complementary information | 5 | support_present |
| qacc_367b09e4ed80 | No conflict | 5 | support_present |
| qacc_51c89636151e | Complementary information | 5 | support_present |
| qacc_883303a2d535 | Conflict due to misinformation | 5 | support_present |

## 32B Val

Abstain confusion

| TP | TN | FP | FN | pred_abstain | gold_abstain | accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 22 | 59 | 0 | 0 | 22 | 22 | 100.0 |

False-abstain labels

| gold_label | count |
| --- | --- |
| none | 0 |

Hard subgroup slices

| slice | n | answered_rate_answerable | false_abstain_rate | missed_refusal_rate | parsed_conflict_acc_answerable | avg_doc_acc | avg_token_f1_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| non_refusal | 59 | 100.00% | 0.00% | NA | 72.41% | 0.8013 | 0.4578 |
| answerable_support_present | 54 | 100.00% | 0.00% | NA | 71.70% | 0.7923 | 0.4733 |
| answerable_partial_only | 5 | 100.00% | 0.00% | NA | 80.00% | 0.8978 | 0.2897 |
| answerable_5doc_support_present | 17 | 100.00% | 0.00% | NA | 47.06% | 0.8118 | 0.4118 |
| answerable_5doc_partial_only | 4 | 100.00% | 0.00% | NA | 75.00% | 0.9000 | 0.2350 |
| answerable_5doc_partial_only_complementary | 2 | 100.00% | 0.00% | NA | 100.00% | 0.9000 | 0.2264 |
| answerable_5doc_partial_only_conflicting | 1 | 100.00% | 0.00% | NA | 100.00% | 1.0000 | 0.2373 |
| answerable_5doc_partial_only_misinformation | 1 | 100.00% | 0.00% | NA | 0.00% | 0.8000 | 0.2500 |
| refusal | 22 | NA | NA | 0.00% | NA | 0.8545 | NA |

Top answerable conflict confusions

| gold | pred | count |
| --- | --- | --- |
| Complementary information | No conflict | 5 |
| No conflict | Complementary information | 4 |
| Conflict due to misinformation | Complementary information | 2 |
| No conflict | Conflict due to outdated information | 1 |
| Conflicting opinions or research outcomes | Complementary information | 1 |
| Complementary information | Conflicting opinions or research outcomes | 1 |
| Conflicting opinions or research outcomes | No conflict | 1 |
| Conflict due to outdated information | No conflict | 1 |

Conflict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| No conflict | 22 | 36 | 15 | 6 |
| Complementary information | 12 | 47 | 9 | 11 |
| Conflicting opinions or research outcomes | 9 | 65 | 1 | 4 |
| Conflict due to outdated information | 10 | 65 | 1 | 3 |
| Conflict due to misinformation | 0 | 77 | 0 | 2 |

Doc verdict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| supports | 178 | 287 | 27 | 40 |
| partially supports | 209 | 236 | 54 | 33 |
| irrelevant | 56 | 452 | 8 | 16 |

Worst doc-verdict rows

| id | doc_count | bucket | gold_conflict | doc_acc |
| --- | --- | --- | --- | --- |
| #0031 | 12 | support_present | Conflicting opinions or research outcomes | 0.0000 |
| #0427 | 8 | support_present | Complementary information | 0.1250 |
| #0638 | 5 | partial_only | No conflict | 0.4000 |
| situatedqa_temp_059a37dd299c | 5 | support_present | No conflict | 0.4000 |
| #0042 | 10 | support_present | No conflict | 0.5000 |
| #0229 | 6 | support_present | No conflict | 0.5000 |
| #0301 | 9 | support_present | Conflict due to outdated information | 0.5556 |
| #0416 | 9 | support_present | Conflicting opinions or research outcomes | 0.5556 |
| #0399 | 7 | support_present | Conflicting opinions or research outcomes | 0.5714 |
| #0203 | 10 | support_present | Complementary information | 0.6000 |

Lowest-overlap answered rows

| id | gold_conflict | doc_count | token_f1 | rouge_l_f1 |
| --- | --- | --- | --- | --- |
| wikirevision_0048 | Conflict due to outdated information | 4 | 0.0278 | 0.0278 |
| situatedqa_temp_b30fa7ac21ea | Conflict due to outdated information | 5 | 0.0458 | 0.0305 |
| qacc_e0db7d73e48f | No conflict | 5 | 0.1176 | 0.1176 |
| wikirevision_0063 | Conflict due to outdated information | 4 | 0.1250 | 0.1250 |
| situatedqa_temp_cd88d70d8f91 | Complementary information | 5 | 0.1429 | 0.1429 |
| freshqa_98056834d12d | Conflict due to outdated information | 5 | 0.1667 | 0.1667 |
| conflictingqa_d3b9d370384c | Complementary information | 5 | 0.1875 | 0.1875 |
| #0031 | Conflicting opinions or research outcomes | 12 | 0.1929 | 0.0971 |
| qacc_602f2d6e8001 | Complementary information | 5 | 0.2222 | 0.2222 |
| conflictingqa_3e34a8ddb07c | Conflicting opinions or research outcomes | 5 | 0.2373 | 0.2373 |

## 32B Benchmark

Abstain confusion

| TP | TN | FP | FN | pred_abstain | gold_abstain | accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 120 | 579 | 29 | 8 | 149 | 128 | 94.97 |

False-abstain labels

| gold_label | count |
| --- | --- |
| Conflict due to misinformation | 10 |
| Conflict due to outdated information | 5 |
| Conflicting opinions or research outcomes | 4 |
| Complementary information | 9 |
| No conflict | 1 |

Hard subgroup slices

| slice | n | answered_rate_answerable | false_abstain_rate | missed_refusal_rate | parsed_conflict_acc_answerable | avg_doc_acc | avg_token_f1_answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| non_refusal | 608 | 95.23% | 4.77% | NA | 63.12% | 0.8237 | 0.3945 |
| answerable_support_present | 531 | 97.18% | 2.82% | NA | 63.50% | 0.8135 | 0.4070 |
| answerable_partial_only | 77 | 81.82% | 18.18% | NA | 60.53% | 0.8935 | 0.3038 |
| answerable_5doc_support_present | 426 | 96.71% | 3.29% | NA | 58.43% | 0.8108 | 0.3890 |
| answerable_5doc_partial_only | 77 | 81.82% | 18.18% | NA | 60.53% | 0.8935 | 0.3038 |
| answerable_5doc_partial_only_complementary | 35 | 85.71% | 14.29% | NA | 61.76% | 0.8286 | 0.2752 |
| answerable_5doc_partial_only_conflicting | 31 | 93.55% | 6.45% | NA | 80.65% | 0.9677 | 0.3011 |
| answerable_5doc_partial_only_misinformation | 11 | 36.36% | 63.64% | NA | 0.00% | 0.8909 | 0.5382 |
| refusal | 128 | NA | NA | 6.25% | NA | 0.8000 | NA |

Top answerable conflict confusions

| gold | pred | count |
| --- | --- | --- |
| Complementary information | No conflict | 70 |
| Conflict due to outdated information | No conflict | 26 |
| No conflict | Complementary information | 24 |
| Conflict due to misinformation | No conflict | 20 |
| Conflicting opinions or research outcomes | No conflict | 18 |
| Conflicting opinions or research outcomes | Complementary information | 17 |
| Complementary information | Conflicting opinions or research outcomes | 14 |
| Conflict due to misinformation | Complementary information | 12 |
| Conflict due to outdated information | Complementary information | 7 |
| No conflict | Conflicting opinions or research outcomes | 3 |
| Conflict due to misinformation | Conflicting opinions or research outcomes | 3 |
| Conflict due to outdated information | Conflicting opinions or research outcomes | 2 |

Conflict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| No conflict | 179 | 344 | 175 | 31 |
| Complementary information | 102 | 444 | 68 | 115 |
| Conflicting opinions or research outcomes | 59 | 599 | 23 | 48 |
| Conflict due to outdated information | 116 | 566 | 5 | 42 |
| Conflict due to misinformation | 0 | 690 | 2 | 37 |

Doc verdict one-vs-rest confusion

| label | TP | TN | FP | FN |
| --- | --- | --- | --- | --- |
| supports | 957 | 2284 | 180 | 266 |
| partially supports | 1830 | 1199 | 332 | 326 |
| irrelevant | 238 | 3229 | 150 | 70 |

Worst doc-verdict rows

| id | doc_count | bucket | gold_conflict | doc_acc |
| --- | --- | --- | --- | --- |
| situatedqa_temp_61a79d74d827 | 5 | support_present | No conflict | 0.0000 |
| trust_align_038 | 5 | partial_only | Conflicting opinions or research outcomes | 0.0000 |
| qacc_0bd7153f19ad | 5 | support_present | No conflict | 0.2000 |
| qacc_9c2f95b14a78 | 5 | support_present | Complementary information | 0.2000 |
| situatedqa_geo_7222d6123c27 | 5 | support_present | Complementary information | 0.2000 |
| conflictingqa_3c835387fe6d | 5 | partial_only | Complementary information | 0.2000 |
| situatedqa_temp_7dd0bea41e4a | 5 | partial_only | Complementary information | 0.2000 |
| trust_align_016 | 5 | partial_only | Conflicting opinions or research outcomes | 0.2000 |
| trust_align_041 | 5 | partial_only | No conflict | 0.2000 |
| trust_align_050 | 5 | partial_only | No conflict | 0.2000 |

Lowest-overlap answered rows

| id | gold_conflict | doc_count | token_f1 | rouge_l_f1 |
| --- | --- | --- | --- | --- |
| conflictingqa_a994724a28e7 | Conflicting opinions or research outcomes | 5 | 0.0000 | 0.0000 |
| conflictingqa_bfbbc2c7a1af | Conflicting opinions or research outcomes | 5 | 0.0000 | 0.0000 |
| freshqa_4a98eba95e97 | No conflict | 5 | 0.0000 | 0.0000 |
| freshqa_64c12116affc | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| freshqa_6a45fadeb16b | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| freshqa_c7315f8b3029 | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| freshqa_c7ac9d61059a | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| freshqa_f6cc6071caa5 | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| qacc_0b75ed799d46 | Conflict due to outdated information | 5 | 0.0000 | 0.0000 |
| qacc_287da9f37864 | No conflict | 5 | 0.0000 | 0.0000 |

False-abstain example IDs

| id | gold_conflict | doc_count | bucket |
| --- | --- | --- | --- |
| freshqa_2e51f51132ee | Conflict due to misinformation | 5 | support_present |
| freshqa_3227ea6c6056 | Conflict due to outdated information | 5 | support_present |
| freshqa_8f302f0bfe82 | Conflict due to outdated information | 5 | support_present |
| freshqa_d4d59d75b206 | Conflict due to misinformation | 5 | support_present |
| qacc_08cf866bcb9b | Conflicting opinions or research outcomes | 5 | support_present |
| qacc_17dc360eea55 | Complementary information | 5 | support_present |
| qacc_367b09e4ed80 | No conflict | 5 | support_present |
| qacc_51c89636151e | Complementary information | 5 | support_present |
