# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.959 (over 49 samples)

**GR F1** *(used in CATS)*: 0.970

**Behavior Adherence**: 0.765 (over 34 applicable samples)

**Factual Grounding**: 0.853 (over 34 applicable samples)

**Single-Truth Recall**: 0.794 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.845

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
- **Behavior**: 0.667 (n=12)
- **Grounding**: 0.826 (n=12)
- **Recall**: 0.750 (n=12)
- **CATS**: 0.800

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.933
- **GR F1** *(used in CATS)*: 0.933
- **Behavior**: 0.750 (n=8)
- **Grounding**: 0.833 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.839

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.778 (n=9)
- **Grounding**: 0.935 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.904

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.800 (n=5)
- **Recall**: 0.900 (n=5)
- **CATS**: 0.925


================================================================================

## Cost Summary

**Total Cost**: $0.0895

**Decisions Made**: 149

**Average Cost per Decision**: $0.000601


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 149
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0895
- **Total Requests**: 149
- **Average Cost per Request**: $0.000601


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This victory gave her a $250,000 prize and a recording contract with Hollywood Records

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7, d8
- **Supporting Docs Found**: None
- **Claim**: He has held this position since the 2024 season, when he became the youngest-ever ATP year-end number one, surpassing the previous record held by Novak Djokovic

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d11, d7
- **Supporting Docs Found**: d9, d3
- **Claim**: A middle-ground view, supported by the lean startup movement, holds that while a business plan is not strictly mandatory, it significantly improves a startup's chances of success by forcing founders to think through their ideas thoroughly and test their assumptions before launching .

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d3
- **Supporting Docs Found**: None
- **Claim**: If you are a smoker, quitting is an important step to reduce your risk of developing RA public health campaigns are encouraged to educate smokers about this preventable risk factor

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: This victory was confirmed across multiple sources, with Pegasus Spiele, the game's publisher, also releasing an official press release

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d2, d6
- **Claim**: However, there are important nuances and exceptions to this rule

### Sample #0324

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d6
- **Supporting Docs Found**: None
- **Claim**: This is confirmed unanimously across multiple authoritative sources, including the official Nobel Prize website and Britannica

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d9
- **Supporting Docs Found**: d7
- **Claim**: Her most recent and influential work, *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power*, was published in 2019 and has been translated into 25 languages

### Sample #0373

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7, d6
- **Supporting Docs Found**: None
- **Claim**: The other 55 delegates signed the document beginning at the right with their names arranged by state, from New Hampshire to Georgia

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d9
- **Supporting Docs Found**: d1, d7, d6, d3, d8
- **Claim**: This is confirmed by multiple sources, with the NFL's official website and Wikipedia both confirming Caesars Superdome as the host stadium for the 2024 season's championship game

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE


================================================================================

*Report generated by CATS v2.0*
