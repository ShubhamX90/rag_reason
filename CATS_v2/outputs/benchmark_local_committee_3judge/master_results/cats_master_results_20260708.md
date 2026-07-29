# CATS Master Results Matrix

This report is rebuilt directly from the synced `detailed_results.json` files under `outputs/benchmark_local_committee_3judge`, not from any derived CSV. In the standard benchmark section, each exact `model + eval family + prompt` configuration is shown with the `baseline` row immediately followed by the matching `sft` row whenever both are available locally.

## Coverage

- Total synced result files included: `108`
- Standard benchmark result files: `96`
- Answer-only SFT result files: `6`
- Other-techniques result files: `6`
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

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8267 | 0.9967 | 0.9038 | 0.3333 | 0.0078 | 0.0153 | 0.7105 | 0.0737 | 0.7429 | 0.6077 | — |
| e2e | minimal | sft | 0.9983 | 0.9720 | 0.9850 | 0.8819 | 0.9922 | 0.9338 | 0.7136 | 0.8574 | 0.7947 | 0.8377 | 0.2300 |
| e2e | runtime | baseline | 0.8279 | 0.9967 | 0.9045 | 0.5000 | 0.0156 | 0.0303 | 0.7197 | 0.8971 | 0.7507 | 0.8180 | — |
| e2e | runtime | sft | 0.9983 | 0.9704 | 0.9842 | 0.8759 | 0.9922 | 0.9304 | 0.7053 | 0.8491 | 0.8062 | 0.8362 | 0.0182 |
| e2e | strict | baseline | 0.8442 | 0.8734 | 0.8585 | 0.2804 | 0.2344 | 0.2553 | 0.6345 | 0.7054 | 0.7210 | 0.7298 | — |
| e2e | strict | sft | 0.9966 | 0.9704 | 0.9833 | 0.8750 | 0.9844 | 0.9265 | 0.7269 | 0.8516 | 0.8049 | 0.8417 | 0.1119 |
| oracle_both | minimal | baseline | 0.8283 | 1.0000 | 0.9061 | 1.0000 | 0.0156 | 0.0308 | 0.6232 | 0.1563 | 0.5559 | 0.5604 | — |
| oracle_both | minimal | sft | 0.9702 | 0.9655 | 0.9678 | 0.8397 | 0.8594 | 0.8494 | 0.7033 | 0.8742 | 0.7780 | 0.8308 | 0.2705 |
| oracle_both | runtime | baseline | 0.9076 | 0.8882 | 0.8978 | 0.5177 | 0.5703 | 0.5428 | 0.6992 | 0.5872 | 0.8175 | 0.7504 | — |
| oracle_both | runtime | sft | 0.9900 | 0.9786 | 0.9843 | 0.9037 | 0.9531 | 0.9278 | 0.7402 | 0.8971 | 0.8192 | 0.8602 | 0.1098 |
| oracle_both | strict | baseline | 0.8412 | 0.8191 | 0.8300 | 0.2361 | 0.2656 | 0.2500 | 0.6745 | 0.5820 | 0.7863 | 0.7182 | — |
| oracle_both | strict | sft | 0.9868 | 0.9819 | 0.9843 | 0.9160 | 0.9375 | 0.9266 | 0.7485 | 0.8927 | 0.8084 | 0.8585 | 0.1403 |
| oracle_conflict | minimal | baseline | 0.8251 | 0.9852 | 0.8981 | 0.1000 | 0.0078 | 0.0145 | 0.6355 | 0.1115 | 0.6680 | 0.5783 | — |
| oracle_conflict | minimal | sft | 0.9983 | 0.9556 | 0.9765 | 0.8247 | 0.9922 | 0.9007 | 0.6951 | 0.8424 | 0.8161 | 0.8325 | 0.2542 |
| oracle_conflict | runtime | baseline | 0.8366 | 0.8339 | 0.8353 | 0.2231 | 0.2266 | 0.2248 | 0.5770 | 0.3907 | 0.7751 | 0.6445 | — |
| oracle_conflict | runtime | sft | 0.9983 | 0.9688 | 0.9833 | 0.8699 | 0.9922 | 0.9270 | 0.7043 | 0.8529 | 0.8079 | 0.8371 | 0.1926 |
| oracle_conflict | strict | baseline | 0.8387 | 0.8980 | 0.8674 | 0.2706 | 0.1797 | 0.2160 | 0.6078 | 0.3489 | 0.6578 | 0.6205 | — |
| oracle_conflict | strict | sft | 0.9966 | 0.9720 | 0.9842 | 0.8811 | 0.9844 | 0.9299 | 0.7105 | 0.8443 | 0.8098 | 0.8372 | 0.2167 |
| oracle_notes | minimal | baseline | 0.8283 | 1.0000 | 0.9061 | 1.0000 | 0.0156 | 0.0308 | 0.6314 | 0.1593 | 0.6049 | 0.5754 | — |
| oracle_notes | minimal | sft | 0.9915 | 0.9605 | 0.9758 | 0.8367 | 0.9609 | 0.8945 | 0.7177 | 0.8685 | 0.7879 | 0.8375 | 0.2620 |
| oracle_notes | runtime | baseline | 0.9298 | 0.8717 | 0.8998 | 0.5301 | 0.6875 | 0.5986 | 0.6817 | 0.6754 | 0.8210 | 0.7695 | — |
| oracle_notes | runtime | sft | 0.9933 | 0.9819 | 0.9876 | 0.9185 | 0.9688 | 0.9430 | 0.7495 | 0.9022 | 0.8186 | 0.8645 | 0.0950 |
| oracle_notes | strict | baseline | 0.8880 | 0.8734 | 0.8806 | 0.4420 | 0.4766 | 0.4586 | 0.6920 | 0.7104 | 0.7674 | 0.7626 | — |
| oracle_notes | strict | sft | 0.9933 | 0.9737 | 0.9834 | 0.8857 | 0.9688 | 0.9254 | 0.7454 | 0.8879 | 0.8252 | 0.8605 | 0.0979 |

### mistral7b

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8256 | 0.9967 | 0.9031 | 0.0000 | 0.0000 | 0.0000 | 0.7002 | 0.0795 | 0.7269 | 0.6024 | — |
| e2e | minimal | sft | 1.0000 | 0.9194 | 0.9580 | 0.7232 | 1.0000 | 0.8393 | 0.6273 | 0.7920 | 0.7862 | 0.7909 | 0.1884 |
| e2e | runtime | baseline | 0.8409 | 0.9737 | 0.9024 | 0.5000 | 0.1250 | 0.2000 | 0.5903 | 0.7393 | 0.6764 | 0.7271 | — |
| e2e | runtime | sft | 1.0000 | 0.9145 | 0.9553 | 0.7111 | 1.0000 | 0.8312 | 0.6129 | 0.7621 | 0.7599 | 0.7725 | 0.0454 |
| e2e | strict | baseline | 0.8590 | 0.9720 | 0.9120 | 0.6458 | 0.2422 | 0.3523 | 0.5626 | 0.6241 | 0.6596 | 0.6896 | — |
| e2e | strict | sft | 1.0000 | 0.9112 | 0.9535 | 0.7033 | 1.0000 | 0.8258 | 0.6437 | 0.7955 | 0.7928 | 0.7964 | 0.1068 |
| oracle_both | minimal | baseline | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.6068 | 0.2100 | 0.5435 | 0.5663 | — |
| oracle_both | minimal | sft | 0.9947 | 0.9293 | 0.9609 | 0.7440 | 0.9766 | 0.8446 | 0.6797 | 0.8250 | 0.8036 | 0.8173 | 0.2510 |
| oracle_both | runtime | baseline | 0.8363 | 0.9408 | 0.8854 | 0.3077 | 0.1250 | 0.1778 | 0.6273 | 0.2983 | 0.7194 | 0.6326 | — |
| oracle_both | runtime | sft | 0.9965 | 0.9391 | 0.9670 | 0.7730 | 0.9844 | 0.8660 | 0.6622 | 0.8310 | 0.7852 | 0.8114 | 0.1787 |
| oracle_both | strict | baseline | 0.8384 | 0.9556 | 0.8932 | 0.3721 | 0.1250 | 0.1871 | 0.6417 | 0.5824 | 0.6944 | 0.7029 | — |
| oracle_both | strict | sft | 0.9982 | 0.9178 | 0.9563 | 0.7175 | 0.9922 | 0.8328 | 0.6520 | 0.8071 | 0.7898 | 0.8013 | 0.0984 |
| oracle_conflict | minimal | baseline | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.5400 | 0.1805 | 0.5652 | 0.5476 | — |
| oracle_conflict | minimal | sft | 1.0000 | 0.9194 | 0.9580 | 0.7232 | 1.0000 | 0.8393 | 0.6366 | 0.7874 | 0.8026 | 0.7962 | 0.2485 |
| oracle_conflict | runtime | baseline | 0.8364 | 0.8997 | 0.8669 | 0.2561 | 0.1641 | 0.2000 | 0.5144 | 0.2963 | 0.7063 | 0.5960 | — |
| oracle_conflict | runtime | sft | 1.0000 | 0.9194 | 0.9580 | 0.7232 | 1.0000 | 0.8393 | 0.6263 | 0.7766 | 0.7730 | 0.7835 | 0.1875 |
| oracle_conflict | strict | baseline | 0.8623 | 0.9062 | 0.8837 | 0.4124 | 0.3125 | 0.3556 | 0.5862 | 0.2592 | 0.7313 | 0.6151 | — |
| oracle_conflict | strict | sft | 1.0000 | 0.8980 | 0.9463 | 0.6737 | 1.0000 | 0.8050 | 0.6119 | 0.7495 | 0.7878 | 0.7739 | 0.1587 |
| oracle_notes | minimal | baseline | 0.8270 | 0.9984 | 0.9046 | 0.5000 | 0.0078 | 0.0154 | 0.6940 | 0.2198 | 0.6789 | 0.6243 | — |
| oracle_notes | minimal | sft | 0.9947 | 0.9326 | 0.9626 | 0.7530 | 0.9766 | 0.8503 | 0.6663 | 0.8230 | 0.7872 | 0.8098 | 0.1855 |
| oracle_notes | runtime | baseline | 0.8569 | 0.9260 | 0.8901 | 0.4304 | 0.2656 | 0.3285 | 0.6057 | 0.3870 | 0.6225 | 0.6263 | — |
| oracle_notes | runtime | sft | 0.9948 | 0.9375 | 0.9653 | 0.7669 | 0.9766 | 0.8591 | 0.6704 | 0.8280 | 0.7774 | 0.8103 | 0.1839 |
| oracle_notes | strict | baseline | 0.8426 | 0.9770 | 0.9048 | 0.5484 | 0.1328 | 0.2138 | 0.6591 | 0.6146 | 0.6676 | 0.7115 | — |
| oracle_notes | strict | sft | 0.9964 | 0.9079 | 0.9501 | 0.6923 | 0.9844 | 0.8129 | 0.6376 | 0.8107 | 0.7934 | 0.7979 | 0.0864 |

### qwen7b

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8281 | 0.9984 | 0.9053 | 0.6667 | 0.0156 | 0.0305 | 0.7567 | 0.0310 | 0.7371 | 0.6075 | — |
| e2e | minimal | sft | 0.9784 | 0.9704 | 0.9744 | 0.8647 | 0.8984 | 0.8812 | 0.6858 | 0.8375 | 0.7681 | 0.8165 | 0.2089 |
| e2e | runtime | baseline | 0.8650 | 0.8750 | 0.8700 | 0.3719 | 0.3516 | 0.3614 | 0.6478 | 0.6510 | 0.7685 | 0.7343 | — |
| e2e | runtime | sft | 0.9883 | 0.9704 | 0.9793 | 0.8705 | 0.9453 | 0.9064 | 0.6930 | 0.8291 | 0.7740 | 0.8188 | 0.0845 |
| e2e | strict | baseline | 0.9033 | 0.7220 | 0.8026 | 0.3240 | 0.6328 | 0.4286 | 0.5606 | 0.6168 | 0.8107 | 0.6977 | — |
| e2e | strict | sft | 0.9916 | 0.9688 | 0.9800 | 0.8662 | 0.9609 | 0.9111 | 0.6869 | 0.8384 | 0.7798 | 0.8213 | 0.1236 |
| oracle_both | minimal | baseline | 0.8270 | 0.9984 | 0.9046 | 0.5000 | 0.0078 | 0.0154 | 0.7300 | 0.1046 | 0.6503 | 0.5974 | — |
| oracle_both | minimal | sft | 0.9365 | 0.9951 | 0.9649 | 0.9667 | 0.6797 | 0.7982 | 0.7423 | 0.9024 | 0.7904 | 0.8500 | 0.2526 |
| oracle_both | runtime | baseline | 0.8850 | 0.7089 | 0.7872 | 0.2892 | 0.5625 | 0.3820 | 0.6304 | 0.1559 | 0.8795 | 0.6133 | — |
| oracle_both | runtime | sft | 0.9455 | 0.9984 | 0.9712 | 0.9894 | 0.7266 | 0.8378 | 0.7515 | 0.9082 | 0.7916 | 0.8556 | 0.2424 |
| oracle_both | strict | baseline | 0.9385 | 0.6776 | 0.7870 | 0.3401 | 0.7891 | 0.4753 | 0.5862 | 0.4950 | 0.8331 | 0.6753 | — |
| oracle_both | strict | sft | 0.9571 | 0.9901 | 0.9733 | 0.9439 | 0.7891 | 0.8596 | 0.7372 | 0.8854 | 0.7906 | 0.8466 | 0.1713 |
| oracle_conflict | minimal | baseline | 0.8301 | 0.9967 | 0.9058 | 0.6667 | 0.0312 | 0.0597 | 0.7136 | 0.0269 | 0.7063 | 0.5881 | — |
| oracle_conflict | minimal | sft | 0.9705 | 0.9753 | 0.9729 | 0.8800 | 0.8594 | 0.8696 | 0.6951 | 0.8548 | 0.7907 | 0.8284 | 0.2402 |
| oracle_conflict | runtime | baseline | 0.8429 | 0.6793 | 0.7523 | 0.2073 | 0.3984 | 0.2727 | 0.5698 | 0.3382 | 0.8905 | 0.6377 | — |
| oracle_conflict | runtime | sft | 0.9786 | 0.9786 | 0.9786 | 0.8984 | 0.8984 | 0.8984 | 0.6889 | 0.8235 | 0.7971 | 0.8220 | 0.1844 |
| oracle_conflict | strict | baseline | 0.8971 | 0.7599 | 0.8228 | 0.3394 | 0.5859 | 0.4298 | 0.5565 | 0.3202 | 0.8396 | 0.6348 | — |
| oracle_conflict | strict | sft | 0.9866 | 0.9655 | 0.9759 | 0.8511 | 0.9375 | 0.8922 | 0.6858 | 0.8295 | 0.8019 | 0.8233 | 0.1885 |
| oracle_notes | minimal | baseline | 0.8265 | 0.9951 | 0.9030 | 0.2500 | 0.0078 | 0.0152 | 0.7669 | 0.0869 | 0.7401 | 0.6242 | — |
| oracle_notes | minimal | sft | 0.9525 | 0.9885 | 0.9701 | 0.9333 | 0.7656 | 0.8412 | 0.7392 | 0.8951 | 0.7900 | 0.8486 | 0.2244 |
| oracle_notes | runtime | baseline | 0.9173 | 0.8388 | 0.8763 | 0.4556 | 0.6406 | 0.5325 | 0.6889 | 0.2619 | 0.8349 | 0.6655 | — |
| oracle_notes | runtime | sft | 0.9528 | 0.9967 | 0.9743 | 0.9800 | 0.7656 | 0.8596 | 0.7546 | 0.9051 | 0.7790 | 0.8533 | 0.1878 |
| oracle_notes | strict | baseline | 0.9268 | 0.7286 | 0.8158 | 0.3605 | 0.7266 | 0.4819 | 0.6068 | 0.5932 | 0.7994 | 0.7038 | — |
| oracle_notes | strict | sft | 0.9538 | 0.9852 | 0.9693 | 0.9167 | 0.7734 | 0.8390 | 0.7392 | 0.8854 | 0.8006 | 0.8486 | 0.1448 |

### qwen32b

| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e | minimal | baseline | 0.8270 | 0.9984 | 0.9046 | 0.5000 | 0.0078 | 0.0154 | 0.7731 | 0.1081 | 0.7469 | 0.6332 | — |
| e2e | minimal | sft | 1.0000 | 0.9589 | 0.9790 | 0.8366 | 1.0000 | 0.9110 | 0.7382 | 0.8220 | 0.8092 | 0.8371 | 0.2039 |
| e2e | runtime | baseline | 0.8636 | 0.9786 | 0.9175 | 0.7234 | 0.2656 | 0.3886 | 0.7803 | 0.8643 | 0.7991 | 0.8403 | — |
| e2e | runtime | sft | 1.0000 | 0.9539 | 0.9764 | 0.8205 | 1.0000 | 0.9014 | 0.7577 | 0.8136 | 0.8043 | 0.8380 | -0.0023 |
| e2e | strict | baseline | 0.8604 | 0.9836 | 0.9179 | 0.7561 | 0.2422 | 0.3669 | 0.7331 | 0.8511 | 0.7887 | 0.8227 | — |
| e2e | strict | sft | 1.0000 | 0.9556 | 0.9773 | 0.8258 | 1.0000 | 0.9046 | 0.7495 | 0.8122 | 0.8207 | 0.8399 | 0.0173 |
| oracle_both | minimal | baseline | 0.8329 | 1.0000 | 0.9088 | 1.0000 | 0.0469 | 0.0896 | 0.7248 | 0.1090 | 0.7178 | 0.6151 | — |
| oracle_both | minimal | sft | 0.9900 | 0.9819 | 0.9860 | 0.9173 | 0.9531 | 0.9349 | 0.7351 | 0.8779 | 0.8127 | 0.8529 | 0.2378 |
| oracle_both | runtime | baseline | 0.8879 | 0.9507 | 0.9182 | 0.6471 | 0.4297 | 0.5164 | 0.7515 | 0.7568 | 0.8443 | 0.8177 | — |
| oracle_both | runtime | sft | 0.9899 | 0.9638 | 0.9767 | 0.8472 | 0.9531 | 0.8971 | 0.7536 | 0.8715 | 0.8469 | 0.8622 | 0.0444 |
| oracle_both | strict | baseline | 0.8598 | 0.9885 | 0.9197 | 0.8108 | 0.2344 | 0.3636 | 0.8049 | 0.8404 | 0.8116 | 0.8442 | — |
| oracle_both | strict | sft | 0.9966 | 0.9507 | 0.9731 | 0.8077 | 0.9844 | 0.8873 | 0.7361 | 0.8426 | 0.8180 | 0.8425 | -0.0017 |
| oracle_conflict | minimal | baseline | 0.8283 | 1.0000 | 0.9061 | 1.0000 | 0.0156 | 0.0308 | 0.6468 | 0.0528 | 0.6975 | 0.5758 | — |
| oracle_conflict | minimal | sft | 0.9983 | 0.9655 | 0.9816 | 0.8581 | 0.9922 | 0.9203 | 0.7721 | 0.0041 | 0.8161 | 0.6435 | 0.0677 |
| oracle_conflict | runtime | baseline | 0.8892 | 0.9243 | 0.9065 | 0.5577 | 0.4531 | 0.5000 | 0.7351 | 0.1347 | 0.8732 | 0.6623 | — |
| oracle_conflict | runtime | sft | 1.0000 | 0.9589 | 0.9790 | 0.8366 | 1.0000 | 0.9110 | 0.7515 | 0.0047 | 0.8224 | 0.6394 | -0.0230 |
| oracle_conflict | strict | baseline | 0.8543 | 0.9934 | 0.9186 | 0.8621 | 0.1953 | 0.3185 | 0.7515 | 0.7916 | 0.8354 | 0.8243 | — |
| oracle_conflict | strict | sft | 1.0000 | 0.9457 | 0.9721 | 0.7950 | 1.0000 | 0.8858 | 0.7382 | 0.0016 | 0.8207 | 0.6332 | -0.1911 |
| oracle_notes | minimal | baseline | 0.8313 | 0.9967 | 0.9065 | 0.7143 | 0.0391 | 0.0741 | 0.7361 | 0.1219 | 0.6977 | 0.6155 | — |
| oracle_notes | minimal | sft | 0.9916 | 0.9737 | 0.9826 | 0.8849 | 0.9609 | 0.9213 | 0.7608 | 0.8785 | 0.8108 | 0.8581 | 0.2426 |
| oracle_notes | runtime | baseline | 0.8935 | 0.9655 | 0.9281 | 0.7342 | 0.4531 | 0.5604 | 0.7536 | 0.7969 | 0.7861 | 0.8162 | — |
| oracle_notes | runtime | sft | 0.9866 | 0.9688 | 0.9776 | 0.8633 | 0.9375 | 0.8989 | 0.7433 | 0.8800 | 0.8068 | 0.8519 | 0.0358 |
| oracle_notes | strict | baseline | 0.8573 | 0.9984 | 0.9225 | 0.9643 | 0.2109 | 0.3462 | 0.7875 | 0.0268 | 0.7532 | 0.6225 | — |
| oracle_notes | strict | sft | 0.9915 | 0.9589 | 0.9749 | 0.8311 | 0.9609 | 0.8913 | 0.7341 | 0.8560 | 0.8140 | 0.8448 | 0.2223 |

## Answer-only SFT

These six runs are methodologically distinct from the standard benchmark family, so they are kept in their own section. All of them are `sft` runs.

| Model | Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama8b | e2e | final_only | sft | 1.0000 | 0.9293 | 0.9633 | 0.7485 | 1.0000 | 0.8562 | 0.6817 | 0.7947 | 0.8010 | 0.8102 |
| llama8b | e2e | minimal | sft | 1.0000 | 0.9457 | 0.9721 | 0.7950 | 1.0000 | 0.8858 | 0.7064 | 0.8283 | 0.7911 | 0.8245 |
| mistral7b | e2e | final_only | sft | 1.0000 | 0.9984 | 0.9992 | 0.9922 | 1.0000 | 0.9961 | 0.6797 | 0.8307 | 0.7418 | 0.8128 |
| mistral7b | e2e | minimal | sft | 1.0000 | 0.9984 | 0.9992 | 0.9922 | 1.0000 | 0.9961 | 0.6684 | 0.8256 | 0.7352 | 0.8071 |
| qwen7b | e2e | final_only | sft | 0.9983 | 0.9671 | 0.9825 | 0.8639 | 0.9922 | 0.9236 | 0.6520 | 0.8124 | 0.7619 | 0.8022 |
| qwen7b | e2e | minimal | sft | 0.9983 | 0.9490 | 0.9730 | 0.8038 | 0.9922 | 0.8881 | 0.6427 | 0.7934 | 0.7718 | 0.7952 |

## Other Techniques

These rows summarize the currently synced `CoN` and `CoT fewshot` comparisons. These are committee-evaluated comparison runs, not baseline/SFT prompt-family pairs.

| Model | Technique | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | con | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.6581 | 0.6092 | 0.6318 | 0.7010 |
| llama | cot_fewshot | committee_eval | 0.8528 | 0.9145 | 0.8825 | 0.3810 | 0.2500 | 0.3019 | 0.6776 | 0.6348 | 0.7159 | 0.7277 |
| mistral | con | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.6828 | 0.5327 | 0.6726 | 0.6982 |
| mistral | cot_fewshot | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 0.0000 | 0.2263 |
| qwen | con | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.7341 | 0.6830 | 0.7351 | 0.7642 |
| qwen | cot_fewshot | committee_eval | 0.8261 | 1.0000 | 0.9048 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2262 |

## Metric Notes

- `GR-answer Precision`, `GR-answer Recall`, and `GR-answer F1` are read directly from `summary.gr_dataset_metrics.{precision, recall, f1}` in each synced `detailed_results.json`.
- `GR-refusal Precision`, `GR-refusal Recall`, and `GR-refusal F1` are read directly from `summary.gr_dataset_metrics.{abstain_precision, abstain_recall, abstain_f1}` in each synced `detailed_results.json`.
- The two GR families come from the same dataset-level answer/refuse confusion table; the answer family treats `answered` as positive, while the refusal family treats `refused` as positive.
- `STR`, `FG`, `Behavior`, and `Final CATS` are read from `summary.conflict_overall`.
- This Markdown was regenerated directly from the synced JSON result files, not from the derived `cats_master_results_20260708.csv` file.
- `Delta vs baseline` is shown only on the `sft` row and is computed as `sft Final CATS - baseline Final CATS`.
