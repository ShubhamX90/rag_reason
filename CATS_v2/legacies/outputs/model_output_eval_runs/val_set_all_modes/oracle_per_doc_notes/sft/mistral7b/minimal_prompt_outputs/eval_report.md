# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.980 (over 49 samples)

**GR F1** *(used in CATS)*: 0.985

**Behavior Adherence**: 0.824 (over 34 applicable samples)

**Factual Grounding**: 0.909 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.885

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.985
- **Precision**: 1.000
- **Recall**: 0.971
- **Accuracy**: 0.980
- TP=33, FP=0, FN=1, TN=15


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.750 (n=12)
- **Grounding**: 0.875 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.865

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.933
- **GR F1** *(used in CATS)*: 0.933
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.875 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.894

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.778 (n=9)
- **Grounding**: 0.972 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.917

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.933 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.933


================================================================================

## Cost Summary

**Total Cost**: $0.0913

**Decisions Made**: 151

**Average Cost per Decision**: $0.000605


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 151
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0913
- **Total Requests**: 151
- **Average Cost per Request**: $0.000605


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6, d10
- **Supporting Docs Found**: None
- **Claim**: The song appears on Chilliwack's 1981 album Anthology and has also been released as a standalone single, with the band's lead vocalists Brian MacLeod and Bill Henderson sharing singing duties throughout

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: This song is not to be confused with a cover version by The Recess Monkeys, which is a separate song with similar wording but different music

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d11, d8, d7, d5, d10, d9, d6, d1, d2
- **Claim**: The overall scientific consensus leans toward acknowledging that prolonged sitting carries significant health risks — particularly for cardiovascular disease, diabetes obesity — but the degree of harm remains a subject of ongoing debate the claim that sitting is literally "the new smoking" is not uniformly supported by all researchers

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: Public health campaigns and smoking cessation interventions are therefore considered essential for preventing and managing RA, particularly for individuals with a family history of the disease

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d8
- **Supporting Docs Found**: None
- **Claim**: This victory was confirmed across multiple sources, with Pegasus Spiele, the game's publisher, also releasing an official press release

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d7, d9, d1, d2
- **Supporting Docs Found**: d10
- **Claim**: No, champagne does not come solely from France. The term 'Champagne' is legally protected as a name exclusively for sparkling wine produced in the Champagne region of northeastern France over 70 countries have adopted similar protections , but this does not mean champagne is produced exclusively in France

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE


================================================================================

*Report generated by CATS v2.0*
