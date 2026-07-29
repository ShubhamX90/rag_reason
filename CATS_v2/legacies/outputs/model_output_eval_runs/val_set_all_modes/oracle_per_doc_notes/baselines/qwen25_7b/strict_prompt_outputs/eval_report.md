# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 11 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.755 (over 49 samples)

**GR F1** *(used in CATS)*: 0.812

**Behavior Adherence**: 0.632 (over 38 applicable samples)

**Factual Grounding**: 0.605 (over 38 applicable samples)

**Single-Truth Recall**: 0.794 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.711

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.812
- **Precision**: 0.867
- **Recall**: 0.765
- **Accuracy**: 0.755
- TP=26, FP=4, FN=8, TN=11


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (6 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.842
- **GR F1** *(used in CATS)*: 0.870
- **Behavior**: 0.692 (n=13)
- **Grounding**: 0.692 (n=13)
- **Recall**: 0.708 (n=12)
- **CATS**: 0.741

### Type 2: Complementary Info

- **Samples**: 15 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.733
- **GR F1** *(used in CATS)*: 0.778
- **Behavior**: 0.636 (n=11)
- **Grounding**: 0.636 (n=11)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.684

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.500
- **GR F1** *(used in CATS)*: 0.615
- **Behavior**: 0.333 (n=9)
- **Grounding**: 0.278 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.409

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.900 (n=5)
- **Recall**: 1.000 (n=5)
- **CATS**: 0.975


================================================================================

## Cost Summary

**Total Cost**: $0.0422

**Decisions Made**: 108

**Average Cost per Decision**: $0.000391


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 108
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0423
- **Total Requests**: 108
- **Average Cost per Request**: $0.000392


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0104

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This information is confirmed across different ranking systems and websites

### Sample #0127

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This date is corroborated by historical records and authoritative references

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d7, d1, d5
- **Claim**: The evidence collectively indicates that stress does not cause stomach ulcers on its own

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d8
- **Claim**: The evidence provides multiple estimates for the number of employees, with the most specific and recent estimates being 166,000 and 167,397

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3, d8
- **Claim**: Given the slight discrepancy, we can infer that the number of employees is approximately 166,000 to 167,000

### Sample #0201

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d11
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: also supports this conclusion, although it is of lower quality

### Sample #0263

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d7, d4, d2, d1, d5, d8
- **Claim**: Based on the evidence, dogs can understand human language to a certain extent, as supported by studies showing they can recognize specific words and process word meaning and intonation similarly to humans

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d4, d2, d1, d5
- **Claim**: The evidence shows that while some sources recommend against vegan diets for pregnant women due to potential nutritional risks, others support the practice if well-planned and supplemented

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The snippets confirm that club soda is effective for stain removal, particularly for certain types of stains, as seen in d1 and d4

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, none of the documents provide a detailed explanation of why club soda works so well

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the lack of a comprehensive explanation, we cannot definitively answer why club soda works so well for stain removal

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The documents collectively suggest that night vision devices display green images due to the sensitivity of the human eye to green wavelengths and the energy-saving benefits of using green light

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, while the green color is a well-known characteristic of night vision technology, the exact scientific reason for its use remains unclear based on the provided evidence

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d2, d1, d5
- **Supporting Docs Found**: None
- **Claim**: Therefore, we cannot form a definitive answer from the given evidence


================================================================================

*Report generated by CATS v2.0*
