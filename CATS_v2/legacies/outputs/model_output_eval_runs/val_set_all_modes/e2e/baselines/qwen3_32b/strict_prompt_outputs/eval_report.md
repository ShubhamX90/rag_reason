# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 8 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.633 (over 49 samples)

**GR F1** *(used in CATS)*: 0.719

**Behavior Adherence**: 0.561 (over 41 applicable samples)

**Factual Grounding**: 0.635 (over 41 applicable samples)

**Single-Truth Recall**: 0.706 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.655

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.719
- **Precision**: 0.767
- **Recall**: 0.676
- **Accuracy**: 0.633
- TP=23, FP=7, FN=11, TN=8


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.737
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.733 (n=15)
- **Grounding**: 0.729 (n=15)
- **Recall**: 0.750 (n=12)
- **CATS**: 0.753

### Type 2: Complementary Info

- **Samples**: 15 (3 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.588
- **Behavior**: 0.667 (n=12)
- **Grounding**: 0.604 (n=12)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.620

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.400
- **GR F1** *(used in CATS)*: 0.500
- **Behavior**: 0.000 (n=9)
- **Grounding**: 0.315 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.272

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.850


================================================================================

## Cost Summary

**Total Cost**: $0.0693

**Decisions Made**: 156

**Average Cost per Decision**: $0.000444


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 156
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0693
- **Total Requests**: 156
- **Average Cost per Request**: $0.000444


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d6, d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0102

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This date is consistently mentioned across various sources, including educational websites and news articles, confirming its accuracy

### Sample #0159

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

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The exact count cannot be determined from the retrieved evidence for the entire Mercedes-Benz Group as different documents refer to different entities within the group

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6
- **Supporting Docs Found**: d5
- **Claim**: However, the updated law now sets the minimum age at 18, aligning Massachusetts with other states that have eliminated child marriage

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
- **Cited Docs**: d9
- **Supporting Docs Found**: d1
- **Claim**: However, dogs may rely more on body language than verbal commands

### Sample #0320

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5
- **Claim**: Spider-Man did not originally have organic web shooters in the comics

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d8
- **Claim**: Some signers, like Elbridge Gerry and Thomas McKean, added their names later

### Sample #0373

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The complete list of signers includes representatives from all 13 colonies

### Sample #0392

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0399

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
- **Claim**: This process enhances their ability to absorb water efficiently

### Sample #0509

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, ongoing air dates are not consistently provided across all documents it is unclear if this schedule is current

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Our brains register most metals as silver in color because they reflect light uniformly across the visible spectrum, which results in a neutral or silver appearance

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: This uniform reflectivity of most metals leads the brain to perceive them as silver

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The book "Oh Canada!

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not provide additional titles of books authored by him


================================================================================

*Report generated by CATS v2.0*
