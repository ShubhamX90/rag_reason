# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.676 (over 34 applicable samples)

**Factual Grounding**: 0.922 (over 34 applicable samples)

**Single-Truth Recall**: 0.853 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.863

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
- **Behavior**: 0.583 (n=12)
- **Grounding**: 0.896 (n=12)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.849

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.625 (n=8)
- **Grounding**: 0.896 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.840

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.778 (n=9)
- **Grounding**: 0.963 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.914

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.950 (n=5)
- **Recall**: 0.700 (n=5)
- **CATS**: 0.863


================================================================================

## Cost Summary

**Total Cost**: $0.0941

**Decisions Made**: 156

**Average Cost per Decision**: $0.000603


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 156
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0941
- **Total Requests**: 156
- **Average Cost per Request**: $0.000603


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d6
- **Supporting Docs Found**: None
- **Claim**: The song appears on their 1981 album *Gone, Gone, Gone* was also released as a standalone single with "My Girl" on the B-side

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: If you smoke, it is essential to seek medical advice about quitting, as doing so can reduce your risk of developing RA and may improve outcomes for those already diagnosed

### Sample #0333

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This appointment was confirmed despite Republican opposition Jackson became the first Black woman to serve on the Court

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10, d1
- **Supporting Docs Found**: d7
- **Claim**: Her most recent and influential work, *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power*, was published in 2019 and has been translated into 25 languages

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The organic web-shooter concept was further explored in the 2012 reboot starring Andrew Garfield, who used a chemical formula to create artificial web-shooters in the 2017 animated film 'Spider-Man: Into the Spider-Verse,' where Miles Morales inherits the power of organic web-shooters from a spider-inspired suit

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d11
- **Supporting Docs Found**: d10, d7
- **Claim**: While France is the sole home of Champagne AOC, the word 'champagne' is also used as a general term for any sparkling wine worldwide some producers in other countries — such as California, Australia New Zealand — have established their own regional appellations (such as 'California Champagne' or 'Australian Sparkling Wine') to distinguish their products from French Champagne

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1
- **Claim**: At that point, it became binding as a commandment for all Church members

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d8, d7, d6
- **Supporting Docs Found**: None
- **Claim**: In oceanography and underwater research, however, AUV (Autonomous Underwater Vehicle) is a different type of unmanned, untethered robot that operates independently under computer control


================================================================================

*Report generated by CATS v2.0*
