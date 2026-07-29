# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 13 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.755 (over 49 samples)

**GR F1** *(used in CATS)*: 0.800

**Behavior Adherence**: 0.667 (over 36 applicable samples)

**Factual Grounding**: 0.579 (over 36 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.732

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.800
- **Precision**: 0.923
- **Recall**: 0.706
- **Accuracy**: 0.755
- TP=24, FP=2, FN=10, TN=13


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.947
- **GR F1** *(used in CATS)*: 0.957
- **Behavior**: 0.917 (n=12)
- **Grounding**: 0.750 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.864

### Type 2: Complementary Info

- **Samples**: 15 (5 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.733
- **GR F1** *(used in CATS)*: 0.750
- **Behavior**: 0.800 (n=10)
- **Grounding**: 0.550 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.700

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.300
- **GR F1** *(used in CATS)*: 0.364
- **Behavior**: 0.111 (n=9)
- **Grounding**: 0.148 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.208

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 1.000 (n=5)
- **CATS**: 0.950


================================================================================

## Cost Summary

**Total Cost**: $0.0439

**Decisions Made**: 106

**Average Cost per Decision**: $0.000414


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 106
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0439
- **Total Requests**: 106
- **Average Cost per Request**: $0.000414


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d5, d4, d6, d10, d8, d2, d3
- **Claim**: Chilliwack is the artist who sings "Gone Gone Gone She Been Gone So Long" as per the song "My Girl (Gone, Gone, Gone)"

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Princeton University has been affiliated with 15 Fields Medalists

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d4, d6, d8, d7, d2, d3
- **Claim**: Heated gemstones are generally less valuable than unheated ones, but the extent and generality of this effect vary depending on the type of gemstone and treatment level

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d6, d8, d7, d2, d3
- **Claim**: For example, unheated sapphires are rarer and more valuable than heated ones heated gemstones can be less valuable than unheated ones in specific comparisons (d1, d6)

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d7
- **Claim**: However, some sources provide more nuanced or subset-limited answers, such as heated rubies losing value with treatment levels or heat treatment having no effect on price for Tanzanite specifically

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d4, d6, d10, d8, d7, d2, d3
- **Claim**: Overall, the evidence suggests that heat treatment can affect the value of gemstones, but the impact is not uniform across all types of gemstones

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d7, d9
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0201

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d11
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [CANNOT ANSWER, INSUFFICIENT EVIDENCE]

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: However, it is essential to note that a vegan diet may not be suitable for everyone, especially those with certain medical conditions or nutritional deficiencies

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d7, d2
- **Supporting Docs Found**: None
- **Claim**: Pregnant women considering a vegan diet should consult with a healthcare professional to ensure they are getting all the necessary nutrients for a healthy pregnancy

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0427

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This definition is consistent across the relevant documents, with some providing additional context and examples

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The final answer is that night vision devices display a green image, but the exact reason for this is not specified in the provided documents

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The green color but do not provide a scientific explanation


================================================================================

*Report generated by CATS v2.0*
