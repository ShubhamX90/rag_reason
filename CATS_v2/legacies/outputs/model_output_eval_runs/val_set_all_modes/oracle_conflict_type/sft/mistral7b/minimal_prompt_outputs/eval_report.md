# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.765 (over 34 applicable samples)

**Factual Grounding**: 0.917 (over 34 applicable samples)

**Single-Truth Recall**: 0.765 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.862

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
- **Behavior**: 0.667 (n=12)
- **Grounding**: 0.910 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.852

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.917 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.931

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.667 (n=9)
- **Grounding**: 0.907 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.858

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.950 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.888


================================================================================

## Cost Summary

**Total Cost**: $0.0886

**Decisions Made**: 148

**Average Cost per Decision**: $0.000599


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 148
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0886
- **Total Requests**: 148
- **Average Cost per Request**: $0.000599


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, some lower-credibility sources present a different view: a peer-reviewed study notes that during chronic stress, noradrenaline secretion can constrict capillaries in

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: This brings the total number of Princeton Fields Medalists to four, as a fourth recipient (Hugo Duminil-Copin of the Université de Genève) was also awarded the 2022 medal

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d9, d10, d4, d5, d6
- **Claim**: The overall scientific consensus is that prolonged sitting is harmful to health, but the degree of risk remains a subject of ongoing debate the claim that sitting is

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: If you smoke and have risk factors for RA such as a family history or certain genetic markers, quitting smoking is considered an essential preventive measure, as doing so has been associated

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7, d8
- **Supporting Docs Found**: d3, d5
- **Claim**: This victory was confirmed when the jury announced the winners at a live awards ceremony on July 21, 2024, with Sky Team taking home the top prize

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This organic web-shooter design was also carried over into the Marvel Cinematic Universe, with Tom Holland's Spider-Man using organic webbing in the MCU films, further cementing the departure from the original comic-book web shooters

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d3, d7, d4, d6
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that 'AUV' also has a separate technical meaning in oceanography: Autonomous Underwater Vehicle (AUV), which refers to unmanned, untethered robots used for underwater research

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These two definitions are valid but apply to different domains, so it is important to distinguish between them when discussing AUV in cars versus AUV in underwater science


================================================================================

*Report generated by CATS v2.0*
