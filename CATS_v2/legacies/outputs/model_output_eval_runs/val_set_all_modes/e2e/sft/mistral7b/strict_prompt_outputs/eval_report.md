# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.588 (over 34 applicable samples)

**Factual Grounding**: 0.919 (over 34 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.847

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 1.000
- **Precision**: 1.000
- **Recall**: 1.000
- **Accuracy**: 1.000
- TP=34, FP=0, FN=0, TN=15


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.417 (n=12)
- **Grounding**: 0.861 (n=12)
- **Recall**: 0.875 (n=12)
- **CATS**: 0.788

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.500 (n=8)
- **Grounding**: 0.896 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.799

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.778 (n=9)
- **Grounding**: 1.000 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.926

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.950 (n=5)
- **Recall**: 0.900 (n=5)
- **CATS**: 0.912


================================================================================

## Cost Summary

**Total Cost**: $0.1007

**Decisions Made**: 160

**Average Cost per Decision**: $0.000629


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 160
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.1007
- **Total Requests**: 160
- **Average Cost per Request**: $0.000629


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This victory gave her a $250,000 cash prize and a recording contract with Hollywood Records

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10, d4
- **Supporting Docs Found**: d7
- **Claim**: Her inauguration was met with protests across Peru she has since faced calls for early elections, though she has yet to complete Castillo's term, which extends through July 2026

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d4
- **Supporting Docs Found**: None
- **Claim**: He has held this position since at least early 2022, when he surpassed Alexander Zverev and Carlos Alcaraz to claim the #1 ranking

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d6
- **Supporting Docs Found**: None
- **Claim**: Internationally, the situation differs greatly: in many countries, child marriage is still legal the minimum age can be as low as 12 years old in some places

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: If you are a smoker, quitting is considered an essential preventive measure, as public health campaigns aim to reduce smoking and its harmful effects on RA incidence

### Sample #0324

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d5
- **Supporting Docs Found**: None
- **Claim**: Han Kang is the first South Korean author to receive the Nobel Prize in Literature her work spans multiple genres including fiction, poetry essays

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3, d1
- **Supporting Docs Found**: d10
- **Claim**: No, champagne does not come solely from France

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3, d1
- **Supporting Docs Found**: d10
- **Claim**: While the name 'Champagne' is legally protected under international treaties and French law for sparkling wines produced in the Champagne region of northeastern France, the term is not exclusive to France as a geographic origin

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d6, d7
- **Supporting Docs Found**: None
- **Claim**: The term 'AUV' is also commonly used to refer to a specific type of underwater vehicle that operates autonomously without direct human control, capable of surveying the seafloor and collecting data


================================================================================

*Report generated by CATS v2.0*
