# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.959 (over 49 samples)

**GR F1** *(used in CATS)*: 0.970

**Behavior Adherence**: 0.765 (over 34 applicable samples)

**Factual Grounding**: 0.875 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.858

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.970
- **Precision**: 1.000
- **Recall**: 0.941
- **Accuracy**: 0.959
- TP=32, FP=0, FN=2, TN=15


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.947
- **GR F1** *(used in CATS)*: 0.957
- **Behavior**: 0.583 (n=12)
- **Grounding**: 0.840 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.803

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.933
- **GR F1** *(used in CATS)*: 0.933
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.792 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.867

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.778 (n=9)
- **Grounding**: 0.926 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.901

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.950


================================================================================

## Cost Summary

**Total Cost**: $0.0978

**Decisions Made**: 157

**Average Cost per Decision**: $0.000623


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 157
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0978
- **Total Requests**: 157
- **Average Cost per Request**: $0.000623


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d6, d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: She is Peru's first ever female president, having served as Castillo's vice president before succeeding him

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: Johns Hopkins Medicine confirms that experts previously believed lifestyle factors like stress caused ulcers, but now know that stomach acids and digestive juices are

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Earlier sources confirm additional Princeton-affiliated winners across different years , consistent with the department's long history of producing outstanding mathematical researchers

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: It is worth noting that international jurisdictions have different rules entirely — for example, the General Law on the Rights of Children and Adolescents in Argentina establishes 18

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d6, d1
- **Supporting Docs Found**: d10, d7, d2
- **Claim**: Shoshana Zuboff is the author of a small but influential body of work, with her most recent and widely cited book being *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power*, published in 2019

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d6, d5, d2
- **Claim**: The overall consensus is that a vegan diet can be safe for pregnant women if it is well-planned, diverse includes appropriate fortified foods or supplements — and that all women, whether vegan or not, should avoid nutritional deficiencies throughout pregnancy to protect fetal development

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d10, d8, d9, d6
- **Supporting Docs Found**: None
- **Claim**: This geographic exclusivity is deeply rooted in French law and cultural tradition the Champagne region's unique terroir is considered essential to the wine's distinct character

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE


================================================================================

*Report generated by CATS v2.0*
