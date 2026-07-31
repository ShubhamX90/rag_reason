# CATS Master Results Matrix

This report is rebuilt directly from the synced `detailed_results.json` files under `outputs/benchmark_local_committee_3judge`. The secondary CATS summaries shown here are recomputed from each file's stored `per_sample` payload using the latest example-level hierarchical aggregation, rather than trusting the historical `summary.conflict_overall.cats_score`.

## Coverage

- Total synced result files included: `108`
- Standard benchmark result files: `96`
- Answer-only SFT result files: `6`
- Other-techniques result files: `6`
- The four redone Mistral/Qwen comparison runs are included from `other_techniques_fixed/{con,cot_fewshot}`; older unfixed Mistral/Qwen comparison JSONs are excluded from the 108-row master scope.
- Complete CATS-H result files: `108`
- Incomplete CATS-H result files: `0`
- Example rows without a computable CATS score: `0`
- Correct refusals contribute their grounded-refusal decision-correctness score; behavior, grounding, and recall remain non-applicable for those examples.
- Complete standard benchmark baseline+SFT pairs: `48`
- Standard benchmark baseline-only configurations still present locally: `0`
- Standard benchmark SFT-only configurations still present locally: `0`

## Model Distribution

| Model | Synced result files |
| --- | ---: |
| llama8b | 26 |
| mistral7b | 26 |
| qwen7b | 26 |
| qwen32b | 24 |
| llama | 2 |
| mistral | 2 |
| qwen | 2 |

## Standard Benchmark Matrix

Within each model table below, rows are ordered by `eval family`, then `prompt`, and then by `run`, with `baseline` always shown before `sft`. The `Delta vs baseline` column is populated only on the `sft` row, so it is visually obvious which row is the baseline and which row is the SFT counterpart.

### llama8b

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal | Delta Prev vs baseline | Delta Bal vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8267 | 0.9967 | 0.9038 | 0.3333 | 0.0078 | 0.0153 | 0.7105 | 0.0737 | 0.7429 | 0.0551 | 0.0289 | 0.0176 | — | — |
| e2e | minimal | sft | 0.9983 | 0.9720 | 0.9850 | 0.8819 | 0.9922 | 0.9338 | 0.7136 | 0.8574 | 0.7947 | 0.6944 | 0.6718 | 0.7316 | 0.6429 | 0.7140 |
| e2e | runtime | baseline | 0.8279 | 0.9967 | 0.9045 | 0.5000 | 0.0156 | 0.0303 | 0.7197 | 0.8971 | 0.7507 | 0.7465 | 0.5105 | 0.3350 | — | — |
| e2e | runtime | sft | 0.9983 | 0.9704 | 0.9842 | 0.8759 | 0.9922 | 0.9304 | 0.7053 | 0.8491 | 0.8062 | 0.6901 | 0.6730 | 0.7478 | 0.1625 | 0.4128 |
| e2e | strict | baseline | 0.8442 | 0.8734 | 0.8585 | 0.2804 | 0.2344 | 0.2553 | 0.6345 | 0.7054 | 0.7210 | 0.5766 | 0.4263 | 0.3627 | — | — |
| e2e | strict | sft | 0.9966 | 0.9704 | 0.9833 | 0.8750 | 0.9844 | 0.9265 | 0.7269 | 0.8516 | 0.8049 | 0.7061 | 0.6801 | 0.7439 | 0.2538 | 0.3812 |
| oracle_both | minimal | baseline | 0.8283 | 1.0000 | 0.9061 | 1.0000 | 0.0156 | 0.0308 | 0.6232 | 0.1563 | 0.5559 | 0.1279 | 0.0793 | 0.0691 | — | — |
| oracle_both | minimal | sft | 0.9702 | 0.9655 | 0.9678 | 0.8397 | 0.8594 | 0.8494 | 0.7033 | 0.8742 | 0.7780 | 0.7080 | 0.6458 | 0.6578 | 0.5664 | 0.5888 |
| oracle_both | runtime | baseline | 0.9076 | 0.8882 | 0.8978 | 0.5177 | 0.5703 | 0.5428 | 0.6992 | 0.5872 | 0.8175 | 0.4903 | 0.4357 | 0.4791 | — | — |
| oracle_both | runtime | sft | 0.9900 | 0.9786 | 0.9843 | 0.9037 | 0.9531 | 0.9278 | 0.7402 | 0.8971 | 0.8192 | 0.7427 | 0.7038 | 0.7534 | 0.2681 | 0.2743 |
| oracle_both | strict | baseline | 0.8412 | 0.8191 | 0.8300 | 0.2361 | 0.2656 | 0.2500 | 0.6745 | 0.5820 | 0.7863 | 0.5116 | 0.4324 | 0.3752 | — | — |
| oracle_both | strict | sft | 0.9868 | 0.9819 | 0.9843 | 0.9160 | 0.9375 | 0.9266 | 0.7485 | 0.8927 | 0.8084 | 0.7407 | 0.6886 | 0.7285 | 0.2562 | 0.3533 |
| oracle_conflict | minimal | baseline | 0.8251 | 0.9852 | 0.8981 | 0.1000 | 0.0078 | 0.0145 | 0.6355 | 0.1115 | 0.6680 | 0.0806 | 0.0504 | 0.0561 | — | — |
| oracle_conflict | minimal | sft | 0.9983 | 0.9556 | 0.9765 | 0.8247 | 0.9922 | 0.9007 | 0.6951 | 0.8424 | 0.8161 | 0.6802 | 0.6698 | 0.7250 | 0.6194 | 0.6690 |
| oracle_conflict | runtime | baseline | 0.8366 | 0.8339 | 0.8353 | 0.2231 | 0.2266 | 0.2248 | 0.5770 | 0.3907 | 0.7751 | 0.3215 | 0.2581 | 0.2916 | — | — |
| oracle_conflict | runtime | sft | 0.9983 | 0.9688 | 0.9833 | 0.8699 | 0.9922 | 0.9270 | 0.7043 | 0.8529 | 0.8079 | 0.6864 | 0.6786 | 0.7365 | 0.4205 | 0.4449 |
| oracle_conflict | strict | baseline | 0.8387 | 0.8980 | 0.8674 | 0.2706 | 0.1797 | 0.2160 | 0.6078 | 0.3489 | 0.6578 | 0.2925 | 0.2566 | 0.2261 | — | — |
| oracle_conflict | strict | sft | 0.9966 | 0.9720 | 0.9842 | 0.8811 | 0.9844 | 0.9299 | 0.7105 | 0.8443 | 0.8098 | 0.6897 | 0.6772 | 0.7350 | 0.4206 | 0.5090 |
| oracle_notes | minimal | baseline | 0.8283 | 1.0000 | 0.9061 | 1.0000 | 0.0156 | 0.0308 | 0.6314 | 0.1593 | 0.6049 | 0.1214 | 0.0773 | 0.0668 | — | — |
| oracle_notes | minimal | sft | 0.9915 | 0.9605 | 0.9758 | 0.8367 | 0.9609 | 0.8945 | 0.7177 | 0.8685 | 0.7879 | 0.7079 | 0.6658 | 0.7029 | 0.5885 | 0.6361 |
| oracle_notes | runtime | baseline | 0.9298 | 0.8717 | 0.8998 | 0.5301 | 0.6875 | 0.5986 | 0.6817 | 0.6754 | 0.8210 | 0.5503 | 0.4836 | 0.5216 | — | — |
| oracle_notes | runtime | sft | 0.9933 | 0.9819 | 0.9876 | 0.9185 | 0.9688 | 0.9430 | 0.7495 | 0.9022 | 0.8186 | 0.7512 | 0.7105 | 0.7657 | 0.2270 | 0.2440 |
| oracle_notes | strict | baseline | 0.8880 | 0.8734 | 0.8806 | 0.4420 | 0.4766 | 0.4586 | 0.6920 | 0.7104 | 0.7674 | 0.6097 | 0.5030 | 0.4928 | — | — |
| oracle_notes | strict | sft | 0.9933 | 0.9737 | 0.9834 | 0.8857 | 0.9688 | 0.9254 | 0.7454 | 0.8879 | 0.8252 | 0.7368 | 0.6991 | 0.7559 | 0.1961 | 0.2631 |

### mistral7b

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal | Delta Prev vs baseline | Delta Bal vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8256 | 0.9967 | 0.9031 | 0.0000 | 0.0000 | 0.0000 | 0.7002 | 0.0795 | 0.7269 | 0.0730 | 0.0477 | 0.0247 | — | — |
| e2e | minimal | sft | 1.0000 | 0.9194 | 0.9580 | 0.7232 | 1.0000 | 0.8393 | 0.6273 | 0.7920 | 0.7862 | 0.6204 | 0.6082 | 0.6628 | 0.5605 | 0.6381 |
| e2e | runtime | baseline | 0.8409 | 0.9737 | 0.9024 | 0.5000 | 0.1250 | 0.2000 | 0.5903 | 0.7393 | 0.6764 | 0.5662 | 0.3977 | 0.2994 | — | — |
| e2e | runtime | sft | 1.0000 | 0.9145 | 0.9553 | 0.7111 | 1.0000 | 0.8312 | 0.6129 | 0.7621 | 0.7599 | 0.5927 | 0.5869 | 0.6571 | 0.1893 | 0.3577 |
| e2e | strict | baseline | 0.8590 | 0.9720 | 0.9120 | 0.6458 | 0.2422 | 0.3523 | 0.5626 | 0.6241 | 0.6596 | 0.4886 | 0.3783 | 0.3032 | — | — |
| e2e | strict | sft | 1.0000 | 0.9112 | 0.9535 | 0.7033 | 1.0000 | 0.8258 | 0.6437 | 0.7955 | 0.7928 | 0.6315 | 0.6199 | 0.6769 | 0.2416 | 0.3738 |
| oracle_both | minimal | baseline | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.6068 | 0.2100 | 0.5435 | 0.1854 | 0.1475 | 0.1145 | — | — |
| oracle_both | minimal | sft | 0.9947 | 0.9293 | 0.9609 | 0.7440 | 0.9766 | 0.8446 | 0.6797 | 0.8250 | 0.8036 | 0.6656 | 0.6469 | 0.7054 | 0.4995 | 0.5909 |
| oracle_both | runtime | baseline | 0.8363 | 0.9408 | 0.8854 | 0.3077 | 0.1250 | 0.1778 | 0.6273 | 0.2983 | 0.7194 | 0.2540 | 0.2206 | 0.1931 | — | — |
| oracle_both | runtime | sft | 0.9965 | 0.9391 | 0.9670 | 0.7730 | 0.9844 | 0.8660 | 0.6622 | 0.8310 | 0.7852 | 0.6615 | 0.6433 | 0.6930 | 0.4227 | 0.4999 |
| oracle_both | strict | baseline | 0.8384 | 0.9556 | 0.8932 | 0.3721 | 0.1250 | 0.1871 | 0.6417 | 0.5824 | 0.6944 | 0.4749 | 0.4024 | 0.3371 | — | — |
| oracle_both | strict | sft | 0.9982 | 0.9178 | 0.9563 | 0.7175 | 0.9922 | 0.8328 | 0.6520 | 0.8071 | 0.7898 | 0.6470 | 0.6250 | 0.6917 | 0.2226 | 0.3546 |
| oracle_conflict | minimal | baseline | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.5400 | 0.1805 | 0.5652 | 0.1544 | 0.1354 | 0.0941 | — | — |
| oracle_conflict | minimal | sft | 1.0000 | 0.9194 | 0.9580 | 0.7232 | 1.0000 | 0.8393 | 0.6366 | 0.7874 | 0.8026 | 0.6076 | 0.6119 | 0.6744 | 0.4765 | 0.5802 |
| oracle_conflict | runtime | baseline | 0.8364 | 0.8997 | 0.8669 | 0.2561 | 0.1641 | 0.2000 | 0.5144 | 0.2963 | 0.7063 | 0.2334 | 0.1996 | 0.1890 | — | — |
| oracle_conflict | runtime | sft | 1.0000 | 0.9194 | 0.9580 | 0.7232 | 1.0000 | 0.8393 | 0.6263 | 0.7766 | 0.7730 | 0.6049 | 0.5968 | 0.6688 | 0.3972 | 0.4798 |
| oracle_conflict | strict | baseline | 0.8623 | 0.9062 | 0.8837 | 0.4124 | 0.3125 | 0.3556 | 0.5862 | 0.2592 | 0.7313 | 0.2061 | 0.2022 | 0.2448 | — | — |
| oracle_conflict | strict | sft | 1.0000 | 0.8980 | 0.9463 | 0.6737 | 1.0000 | 0.8050 | 0.6119 | 0.7495 | 0.7878 | 0.5872 | 0.5915 | 0.6625 | 0.3893 | 0.4177 |
| oracle_notes | minimal | baseline | 0.8270 | 0.9984 | 0.9046 | 0.5000 | 0.0078 | 0.0154 | 0.6940 | 0.2198 | 0.6789 | 0.1886 | 0.1467 | 0.1030 | — | — |
| oracle_notes | minimal | sft | 0.9947 | 0.9326 | 0.9626 | 0.7530 | 0.9766 | 0.8503 | 0.6663 | 0.8230 | 0.7872 | 0.6537 | 0.6240 | 0.6928 | 0.4773 | 0.5898 |
| oracle_notes | runtime | baseline | 0.8569 | 0.9260 | 0.8901 | 0.4304 | 0.2656 | 0.3285 | 0.6057 | 0.3870 | 0.6225 | 0.3273 | 0.2859 | 0.2557 | — | — |
| oracle_notes | runtime | sft | 0.9948 | 0.9375 | 0.9653 | 0.7669 | 0.9766 | 0.8591 | 0.6704 | 0.8280 | 0.7774 | 0.6739 | 0.6364 | 0.6892 | 0.3505 | 0.4335 |
| oracle_notes | strict | baseline | 0.8426 | 0.9770 | 0.9048 | 0.5484 | 0.1328 | 0.2138 | 0.6591 | 0.6146 | 0.6676 | 0.5222 | 0.4125 | 0.3189 | — | — |
| oracle_notes | strict | sft | 0.9964 | 0.9079 | 0.9501 | 0.6923 | 0.9844 | 0.8129 | 0.6376 | 0.8107 | 0.7934 | 0.6337 | 0.6212 | 0.6857 | 0.2087 | 0.3669 |

### qwen7b

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal | Delta Prev vs baseline | Delta Bal vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8281 | 0.9984 | 0.9053 | 0.6667 | 0.0156 | 0.0305 | 0.7567 | 0.0310 | 0.7371 | 0.0334 | 0.0223 | 0.0167 | — | — |
| e2e | minimal | sft | 0.9784 | 0.9704 | 0.9744 | 0.8647 | 0.8984 | 0.8812 | 0.6858 | 0.8375 | 0.7681 | 0.6803 | 0.6167 | 0.6586 | 0.5944 | 0.6419 |
| e2e | runtime | baseline | 0.8650 | 0.8750 | 0.8700 | 0.3719 | 0.3516 | 0.3614 | 0.6478 | 0.6510 | 0.7685 | 0.5415 | 0.4529 | 0.3746 | — | — |
| e2e | runtime | sft | 0.9883 | 0.9704 | 0.9793 | 0.8705 | 0.9453 | 0.9064 | 0.6930 | 0.8291 | 0.7740 | 0.6900 | 0.6294 | 0.6794 | 0.1764 | 0.3049 |
| e2e | strict | baseline | 0.9033 | 0.7220 | 0.8026 | 0.3240 | 0.6328 | 0.4286 | 0.5606 | 0.6168 | 0.8107 | 0.5240 | 0.4498 | 0.4586 | — | — |
| e2e | strict | sft | 0.9916 | 0.9688 | 0.9800 | 0.8662 | 0.9609 | 0.9111 | 0.6869 | 0.8384 | 0.7798 | 0.6773 | 0.6299 | 0.6843 | 0.1801 | 0.2257 |
| oracle_both | minimal | baseline | 0.8270 | 0.9984 | 0.9046 | 0.5000 | 0.0078 | 0.0154 | 0.7300 | 0.1046 | 0.6503 | 0.1106 | 0.0961 | 0.0773 | — | — |
| oracle_both | minimal | sft | 0.9365 | 0.9951 | 0.9649 | 0.9667 | 0.6797 | 0.7982 | 0.7423 | 0.9024 | 0.7904 | 0.7505 | 0.6370 | 0.6212 | 0.5409 | 0.5439 |
| oracle_both | runtime | baseline | 0.8850 | 0.7089 | 0.7872 | 0.2892 | 0.5625 | 0.3820 | 0.6304 | 0.1559 | 0.8795 | 0.1322 | 0.1876 | 0.3376 | — | — |
| oracle_both | runtime | sft | 0.9455 | 0.9984 | 0.9712 | 0.9894 | 0.7266 | 0.8378 | 0.7515 | 0.9082 | 0.7916 | 0.7566 | 0.6596 | 0.6375 | 0.4719 | 0.2998 |
| oracle_both | strict | baseline | 0.9385 | 0.6776 | 0.7870 | 0.3401 | 0.7891 | 0.4753 | 0.5862 | 0.4950 | 0.8331 | 0.4344 | 0.4629 | 0.5192 | — | — |
| oracle_both | strict | sft | 0.9571 | 0.9901 | 0.9733 | 0.9439 | 0.7891 | 0.8596 | 0.7372 | 0.8854 | 0.7906 | 0.7364 | 0.6508 | 0.6579 | 0.1879 | 0.1387 |
| oracle_conflict | minimal | baseline | 0.8301 | 0.9967 | 0.9058 | 0.6667 | 0.0312 | 0.0597 | 0.7136 | 0.0269 | 0.7063 | 0.0230 | 0.0215 | 0.0155 | — | — |
| oracle_conflict | minimal | sft | 0.9705 | 0.9753 | 0.9729 | 0.8800 | 0.8594 | 0.8696 | 0.6951 | 0.8548 | 0.7907 | 0.6937 | 0.6365 | 0.6699 | 0.6150 | 0.6544 |
| oracle_conflict | runtime | baseline | 0.8429 | 0.6793 | 0.7523 | 0.2073 | 0.3984 | 0.2727 | 0.5698 | 0.3382 | 0.8905 | 0.2937 | 0.2886 | 0.3361 | — | — |
| oracle_conflict | runtime | sft | 0.9786 | 0.9786 | 0.9786 | 0.8984 | 0.8984 | 0.8984 | 0.6889 | 0.8235 | 0.7971 | 0.6751 | 0.6291 | 0.6768 | 0.3405 | 0.3407 |
| oracle_conflict | strict | baseline | 0.8971 | 0.7599 | 0.8228 | 0.3394 | 0.5859 | 0.4298 | 0.5565 | 0.3202 | 0.8396 | 0.2594 | 0.2721 | 0.3413 | — | — |
| oracle_conflict | strict | sft | 0.9866 | 0.9655 | 0.9759 | 0.8511 | 0.9375 | 0.8922 | 0.6858 | 0.8295 | 0.8019 | 0.6741 | 0.6248 | 0.6851 | 0.3527 | 0.3438 |
| oracle_notes | minimal | baseline | 0.8265 | 0.9951 | 0.9030 | 0.2500 | 0.0078 | 0.0152 | 0.7669 | 0.0869 | 0.7401 | 0.0871 | 0.0759 | 0.0528 | — | — |
| oracle_notes | minimal | sft | 0.9525 | 0.9885 | 0.9701 | 0.9333 | 0.7656 | 0.8412 | 0.7392 | 0.8951 | 0.7900 | 0.7466 | 0.6429 | 0.6482 | 0.5670 | 0.5954 |
| oracle_notes | runtime | baseline | 0.9173 | 0.8388 | 0.8763 | 0.4556 | 0.6406 | 0.5325 | 0.6889 | 0.2619 | 0.8349 | 0.2091 | 0.2484 | 0.3653 | — | — |
| oracle_notes | runtime | sft | 0.9528 | 0.9967 | 0.9743 | 0.9800 | 0.7656 | 0.8596 | 0.7546 | 0.9051 | 0.7790 | 0.7557 | 0.6533 | 0.6489 | 0.4049 | 0.2836 |
| oracle_notes | strict | baseline | 0.9268 | 0.7286 | 0.8158 | 0.3605 | 0.7266 | 0.4819 | 0.6068 | 0.5932 | 0.7994 | 0.5058 | 0.4865 | 0.5442 | — | — |
| oracle_notes | strict | sft | 0.9538 | 0.9852 | 0.9693 | 0.9167 | 0.7734 | 0.8390 | 0.7392 | 0.8854 | 0.8006 | 0.7384 | 0.6479 | 0.6513 | 0.1615 | 0.1071 |

### qwen32b

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal | Delta Prev vs baseline | Delta Bal vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8270 | 0.9984 | 0.9046 | 0.5000 | 0.0078 | 0.0154 | 0.7731 | 0.1081 | 0.7469 | 0.1052 | 0.0862 | 0.0568 | — | — |
| e2e | minimal | sft | 1.0000 | 0.9589 | 0.9790 | 0.8366 | 1.0000 | 0.9110 | 0.7382 | 0.8220 | 0.8092 | 0.6872 | 0.6520 | 0.7163 | 0.5658 | 0.6595 |
| e2e | runtime | baseline | 0.8636 | 0.9786 | 0.9175 | 0.7234 | 0.2656 | 0.3886 | 0.7803 | 0.8643 | 0.7991 | 0.7597 | 0.6002 | 0.4546 | — | — |
| e2e | runtime | sft | 1.0000 | 0.9539 | 0.9764 | 0.8205 | 1.0000 | 0.9014 | 0.7577 | 0.8136 | 0.8043 | 0.7007 | 0.6583 | 0.7164 | 0.0581 | 0.2618 |
| e2e | strict | baseline | 0.8604 | 0.9836 | 0.9179 | 0.7561 | 0.2422 | 0.3669 | 0.7331 | 0.8511 | 0.7887 | 0.7238 | 0.5739 | 0.4317 | — | — |
| e2e | strict | sft | 1.0000 | 0.9556 | 0.9773 | 0.8258 | 1.0000 | 0.9046 | 0.7495 | 0.8122 | 0.8207 | 0.6859 | 0.6607 | 0.7074 | 0.0867 | 0.2757 |
| oracle_both | minimal | baseline | 0.8329 | 1.0000 | 0.9088 | 1.0000 | 0.0469 | 0.0896 | 0.7248 | 0.1090 | 0.7178 | 0.1043 | 0.0861 | 0.0828 | — | — |
| oracle_both | minimal | sft | 0.9900 | 0.9819 | 0.9860 | 0.9173 | 0.9531 | 0.9349 | 0.7351 | 0.8779 | 0.8127 | 0.7236 | 0.6752 | 0.7284 | 0.5890 | 0.6456 |
| oracle_both | runtime | baseline | 0.8879 | 0.9507 | 0.9182 | 0.6471 | 0.4297 | 0.5164 | 0.7515 | 0.7568 | 0.8443 | 0.6709 | 0.6167 | 0.5312 | — | — |
| oracle_both | runtime | sft | 0.9899 | 0.9638 | 0.9767 | 0.8472 | 0.9531 | 0.8971 | 0.7536 | 0.8715 | 0.8469 | 0.7385 | 0.6930 | 0.7336 | 0.0762 | 0.2024 |
| oracle_both | strict | baseline | 0.8598 | 0.9885 | 0.9197 | 0.8108 | 0.2344 | 0.3636 | 0.8049 | 0.8404 | 0.8116 | 0.7528 | 0.6241 | 0.5093 | — | — |
| oracle_both | strict | sft | 0.9966 | 0.9507 | 0.9731 | 0.8077 | 0.9844 | 0.8873 | 0.7361 | 0.8426 | 0.8180 | 0.7109 | 0.6634 | 0.7149 | 0.0392 | 0.2055 |
| oracle_conflict | minimal | baseline | 0.8283 | 1.0000 | 0.9061 | 1.0000 | 0.0156 | 0.0308 | 0.6468 | 0.0528 | 0.6975 | 0.0516 | 0.0531 | 0.0402 | — | — |
| oracle_conflict | minimal | sft | 0.9983 | 0.9655 | 0.9816 | 0.8581 | 0.9922 | 0.9203 | 0.7721 | 0.0041 | 0.8161 | 0.0016 | 0.1739 | 0.3989 | 0.1208 | 0.3587 |
| oracle_conflict | runtime | baseline | 0.8892 | 0.9243 | 0.9065 | 0.5577 | 0.4531 | 0.5000 | 0.7351 | 0.1347 | 0.8732 | 0.1288 | 0.1952 | 0.2855 | — | — |
| oracle_conflict | runtime | sft | 1.0000 | 0.9589 | 0.9790 | 0.8366 | 1.0000 | 0.9110 | 0.7515 | 0.0047 | 0.8224 | 0.0030 | 0.1769 | 0.4016 | -0.0184 | 0.1162 |
| oracle_conflict | strict | baseline | 0.8543 | 0.9934 | 0.9186 | 0.8621 | 0.1953 | 0.3185 | 0.7515 | 0.7916 | 0.8354 | 0.6878 | 0.5936 | 0.4100 | — | — |
| oracle_conflict | strict | sft | 1.0000 | 0.9457 | 0.9721 | 0.7950 | 1.0000 | 0.8858 | 0.7382 | 0.0016 | 0.8207 | 0.0000 | 0.1739 | 0.4000 | -0.4197 | -0.0100 |
| oracle_notes | minimal | baseline | 0.8313 | 0.9967 | 0.9065 | 0.7143 | 0.0391 | 0.0741 | 0.7361 | 0.1219 | 0.6977 | 0.1175 | 0.0928 | 0.0966 | — | — |
| oracle_notes | minimal | sft | 0.9916 | 0.9737 | 0.9826 | 0.8849 | 0.9609 | 0.9213 | 0.7608 | 0.8785 | 0.8108 | 0.7448 | 0.6794 | 0.7211 | 0.5865 | 0.6245 |
| oracle_notes | runtime | baseline | 0.8935 | 0.9655 | 0.9281 | 0.7342 | 0.4531 | 0.5604 | 0.7536 | 0.7969 | 0.7861 | 0.6995 | 0.5975 | 0.5393 | — | — |
| oracle_notes | runtime | sft | 0.9866 | 0.9688 | 0.9776 | 0.8633 | 0.9375 | 0.8989 | 0.7433 | 0.8800 | 0.8068 | 0.7336 | 0.6709 | 0.7149 | 0.0734 | 0.1756 |
| oracle_notes | strict | baseline | 0.8573 | 0.9984 | 0.9225 | 0.9643 | 0.2109 | 0.3462 | 0.7875 | 0.0268 | 0.7532 | 0.0281 | 0.0594 | 0.1211 | — | — |
| oracle_notes | strict | sft | 0.9915 | 0.9589 | 0.9749 | 0.8311 | 0.9609 | 0.8913 | 0.7341 | 0.8560 | 0.8140 | 0.7142 | 0.6635 | 0.7131 | 0.6041 | 0.5920 |

## Answer-only SFT

These six runs are methodologically distinct from the standard benchmark family, so they are kept in their own section. All of them are `sft` runs.

| Model | Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama8b | e2e | final_only | sft | 1.0000 | 0.9293 | 0.9633 | 0.7485 | 1.0000 | 0.8562 | 0.6817 | 0.7947 | 0.8010 | 0.6390 | 0.6187 | 0.6786 |
| llama8b | e2e | minimal | sft | 1.0000 | 0.9457 | 0.9721 | 0.7950 | 1.0000 | 0.8858 | 0.7064 | 0.8283 | 0.7911 | 0.6784 | 0.6409 | 0.6990 |
| mistral7b | e2e | final_only | sft | 1.0000 | 0.9984 | 0.9992 | 0.9922 | 1.0000 | 0.9961 | 0.6797 | 0.8307 | 0.7418 | 0.6575 | 0.5962 | 0.6798 |
| mistral7b | e2e | minimal | sft | 1.0000 | 0.9984 | 0.9992 | 0.9922 | 1.0000 | 0.9961 | 0.6684 | 0.8256 | 0.7352 | 0.6491 | 0.6002 | 0.6850 |
| qwen7b | e2e | final_only | sft | 0.9983 | 0.9671 | 0.9825 | 0.8639 | 0.9922 | 0.9236 | 0.6520 | 0.8124 | 0.7619 | 0.6249 | 0.5963 | 0.6721 |
| qwen7b | e2e | minimal | sft | 0.9983 | 0.9490 | 0.9730 | 0.8038 | 0.9922 | 0.8881 | 0.6427 | 0.7934 | 0.7718 | 0.6118 | 0.5979 | 0.6722 |

## Other Techniques

These rows summarize the currently synced `CoN` and `CoT fewshot` comparisons. These are committee-evaluated comparison runs, not baseline/SFT prompt-family pairs.

| Model | Technique | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | con | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.6581 | 0.6092 | 0.6318 | 0.5328 | 0.4005 | 0.2758 |
| llama | cot_fewshot | committee_eval | 0.8528 | 0.9145 | 0.8825 | 0.3810 | 0.2500 | 0.3019 | 0.6776 | 0.6348 | 0.7159 | 0.5900 | 0.4705 | 0.4030 |
| mistral | con | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.6756 | 0.5224 | 0.6848 | 0.4595 | 0.3477 | 0.2232 |
| mistral | cot_fewshot | committee_eval | 0.9204 | 0.8174 | 0.8659 | 0.4337 | 0.6641 | 0.5247 | 0.5277 | 0.4165 | 0.6575 | 0.3848 | 0.3890 | 0.4506 |
| qwen | con | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.7187 | 0.6664 | 0.7269 | 0.5947 | 0.4777 | 0.3181 |
| qwen | cot_fewshot | committee_eval | 0.8667 | 0.9622 | 0.9119 | 0.6230 | 0.2969 | 0.4021 | 0.7238 | 0.7338 | 0.7808 | 0.6524 | 0.5424 | 0.4458 |

## Metric Notes

- `GR-answer Precision`, `GR-answer Recall`, and `GR-answer F1` are read directly from `summary.gr_dataset_metrics.{precision, recall, f1}` in each synced `detailed_results.json`.
- `GR-refusal Precision`, `GR-refusal Recall`, and `GR-refusal F1` are read directly from `summary.gr_dataset_metrics.{abstain_precision, abstain_recall, abstain_f1}` in each synced `detailed_results.json`.
- The two GR families come from the same dataset-level answer/refuse confusion table; the answer family treats `answered` as positive, while the refusal family treats `refused` as positive.
- `STR`, `FG`, and `Behavior` are reported from the stored per-sample judgments after recomputing aggregate means.
- `Answer Quality` is the example-level fusion of FG and STR: `sqrt(FG * STR)` when STR applies, else `FG`.
- `CATS-Prev` is the benchmark-prevalence-weighted CATS-Harmonized summary. `CATS-Bal` gives equal weight to conflict types and balances answerable/refusal-required subgroups within each type when both are present.
- Final CATS values are recomputed from per-example hierarchical scores; correct refusals use decision correctness only.
- This Markdown was regenerated directly from the synced JSON result files, not from the derived `cats_master_results_20260731_hierarchical.csv` file.
- `Delta Prev vs baseline` and `Delta Bal vs baseline` are shown only on the `sft` row.
