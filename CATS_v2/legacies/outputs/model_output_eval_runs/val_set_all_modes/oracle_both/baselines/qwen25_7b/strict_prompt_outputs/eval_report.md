# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 14 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.653 (over 49 samples)

**GR F1** *(used in CATS)*: 0.679

**Behavior Adherence**: 0.486 (over 35 applicable samples)

**Factual Grounding**: 0.481 (over 35 applicable samples)

**Single-Truth Recall**: 0.647 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.573

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.679
- **Precision**: 0.947
- **Recall**: 0.529
- **Accuracy**: 0.653
- TP=18, FP=1, FN=16, TN=14


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.895
- **GR F1** *(used in CATS)*: 0.909
- **Behavior**: 0.833 (n=12)
- **Grounding**: 0.750 (n=12)
- **Recall**: 0.750 (n=12)
- **CATS**: 0.811

### Type 2: Complementary Info

- **Samples**: 15 (6 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.444 (n=9)
- **Grounding**: 0.648 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.631

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.100
- **GR F1** *(used in CATS)*: 0.000
- **Behavior**: 0.000 (n=9)
- **Grounding**: 0.000 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.000

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.400
- **GR F1** *(used in CATS)*: 0.571
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.400 (n=5)
- **Recall**: 0.400 (n=5)
- **CATS**: 0.493


================================================================================

## Cost Summary

**Total Cost**: $0.0364

**Decisions Made**: 96

**Average Cost per Decision**: $0.000380


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 96
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0364
- **Total Requests**: 96
- **Average Cost per Request**: $0.000380


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d3, d5, d6, d8, d1, d7, d4, d9, d2
- **Claim**: Therefore, based on the evidence, Dina Boluarte is the answer to the query

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6, d4, d2
- **Supporting Docs Found**: d5, d8, d7, d1
- **Claim**: Therefore, we cannot provide a definitive answer without more specific information

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0201

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0203

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The query asks for the date when Channel 5 started

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d10, d7
- **Claim**: While other documents provide a vague timeframe or an incorrect date, the consensus from the most credible sources is clear

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0263

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d8, d1, d7, d4
- **Claim**: Are most octopuses venomous?

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0333

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d10, d3
- **Claim**: Therefore, we cannot provide a precise count

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d1, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE


================================================================================

*Report generated by CATS v2.0*
