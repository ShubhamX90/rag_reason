# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 12 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.633 (over 49 samples)

**GR F1** *(used in CATS)*: 0.679

**Behavior Adherence**: 0.459 (over 37 applicable samples)

**Factual Grounding**: 0.446 (over 37 applicable samples)

**Single-Truth Recall**: 0.588 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.543

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.679
- **Precision**: 0.864
- **Recall**: 0.559
- **Accuracy**: 0.633
- TP=19, FP=3, FN=15, TN=12


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (5 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.667
- **Behavior**: 0.571 (n=14)
- **Grounding**: 0.512 (n=14)
- **Recall**: 0.583 (n=12)
- **CATS**: 0.583

### Type 2: Complementary Info

- **Samples**: 15 (6 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.667
- **GR F1** *(used in CATS)*: 0.615
- **Behavior**: 0.556 (n=9)
- **Grounding**: 0.444 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.538

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.714
- **Behavior**: 0.000 (n=9)
- **Grounding**: 0.259 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.325

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.750
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.600 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.688


================================================================================

## Cost Summary

**Total Cost**: $0.0364

**Decisions Made**: 101

**Average Cost per Decision**: $0.000360


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 101
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0364
- **Total Requests**: 101
- **Average Cost per Request**: $0.000360


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Stomach ulcers are primarily caused by H. pylori infection and the use of nonsteroidal anti-inflammatory drugs (NSAIDs)

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0187

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0201

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3, d6, d4, d1
- **Claim**: Can smoking cause Rheumatoid Arthritis?

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d2, d8, d3, d6, d9, d1
- **Claim**: The evidence shows that in many cities, public transportation is slower than driving, but there are instances where public transportation can be faster, particularly in cities with dedicated bus lanes or robust transit networks

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d2, d8, d3, d6, d4, d1
- **Claim**: Based on the evidence provided, dogs can understand human language to some extent, particularly through familiar words and commands

### Sample #0300

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact extent and nature of this understanding varies among the sources

### Sample #0320

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d12, d14, d8, d3, d4, d11, d9, d13, d1
- **Claim**: The evidence collectively supports the idea that melting land ice, particularly from glaciers and ice sheets, is the primary contributor to sea level rise

### Sample #0333

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0373

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d10, d9, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d2, d6, d4
- **Claim**: A well-planned vegan diet can be safe and beneficial during pregnancy, as supported by the evidence from

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d2, d6, d4, d1
- **Claim**: These sources emphasize the importance of ensuring adequate intake of essential nutrients such as vitamin B12 and iron

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

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This explains why they seem to get more absorbent over time

### Sample #0509

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the regular broadcast schedule is not mentioned in the retrieved evidence

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Dorothy lived in Kansas according to the information provided in


================================================================================

*Report generated by CATS v2.0*
