# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.959 (over 49 samples)

**GR F1** *(used in CATS)*: 0.970

**Behavior Adherence**: 0.824 (over 34 applicable samples)

**Factual Grounding**: 0.853 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.867

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
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.833 (n=12)
- **Grounding**: 0.854 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.880

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.867
- **GR F1** *(used in CATS)*: 0.857
- **Behavior**: 0.750 (n=8)
- **Grounding**: 0.708 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.772

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
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.883 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.921


================================================================================

## Cost Summary

**Total Cost**: $0.0882

**Decisions Made**: 149

**Average Cost per Decision**: $0.000592


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 149
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0883
- **Total Requests**: 149
- **Average Cost per Request**: $0.000593


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This victory gave her a $250,000 prize and a recording contract with Hollywood Records

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10, d6
- **Supporting Docs Found**: d7
- **Claim**: While Carter took home the trophy for season 22, the results for the current year's season — season 23 — have not yet been revealed, as the show is still airing

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9, d6, d1
- **Supporting Docs Found**: None
- **Claim**: The song appears on Chilliwack's 1981 album *Gone by the Moon* has also been included on various compilation albums and music video collections

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d3
- **Supporting Docs Found**: d4
- **Claim**: India became the sixth country in the world to adopt postal codes, following Germany (1944), Singapore (1950), Argentina (1958), the U.S. (1963) Switzerland (1964) the system has since been expanded to 8 postal regions covering over 1.5 lakh post offices

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d7
- **Supporting Docs Found**: None
- **Claim**: He has held this position since approximately late 2024, surpassing both Novak Djokovic and Alexander Zverev in the rankings

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d7, d6, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: If you smoke, it is essential to seek medical advice about quitting, as doing so can reduce your risk and improve outcomes for those already diagnosed with RA

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: This victory was confirmed across multiple sources, with Pegasus Spiele, the game's publisher, also releasing an official press release

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d2
- **Supporting Docs Found**: d10, d7
- **Claim**: Her most recent and influential work, *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power*, was published in 2019 and has been translated into 25 languages

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: This distinction between mechanical and organic web shooters is consistent across multiple sources, with the organic version becoming a notable hallmark of the Raimi films and subsequent cinematic iterations

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d8, d7
- **Claim**: The Word of Wisdom was first revealed on February 27, 1833, when Joseph Smith dictated the revelation as a principle with promise, not a commandment

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE


================================================================================

*Report generated by CATS v2.0*
