# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 14 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.980 (over 49 samples)

**GR F1** *(used in CATS)*: 0.986

**Behavior Adherence**: 0.743 (over 35 applicable samples)

**Factual Grounding**: 0.898 (over 35 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.862

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.986
- **Precision**: 0.971
- **Recall**: 1.000
- **Accuracy**: 0.980
- TP=34, FP=1, FN=0, TN=14


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (6 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.947
- **GR F1** *(used in CATS)*: 0.960
- **Behavior**: 0.538 (n=13)
- **Grounding**: 0.833 (n=13)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.791

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.906 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.927

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.889 (n=9)
- **Grounding**: 0.963 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.951

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.933 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.883


================================================================================

## Cost Summary

**Total Cost**: $0.0990

**Decisions Made**: 157

**Average Cost per Decision**: $0.000631


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 157
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0990
- **Total Requests**: 157
- **Average Cost per Request**: $0.000631


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d9
- **Supporting Docs Found**: d2
- **Claim**: She was crowned on May 19, 2024, at the season finale, becoming the first woman to win in four consecutive seasons

### Sample #0015

- **Reason**: cross_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: This is consistently confirmed across multiple sources, all of which agree on the same result

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d11
- **Supporting Docs Found**: d3, d1, d8, d6, d4, d5, d7, d2
- **Claim**: Heated gemstones are generally less valuable than their unheated counterparts, though the extent varies by gem type and quality level

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This victory marks a first for the award, as Sky Team is the first two-player game to win the traditionally family-oriented Spiel des Jahres prize

### Sample #0324

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: The 2024 award marks the first time a South Korean writer has received the Nobel Prize in Literature, making it particularly historic for Korean literature

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d6
- **Supporting Docs Found**: d3, d8, d9, d4, d5, d7
- **Claim**: Yes, champagne comes solely from France, specifically from the Champagne region, which is about 90 minutes from Paris

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1, d7, d5, d6
- **Claim**: While the original revelation (received 27 February 1833) stated it was 'not by commandment or constraint,' the 1851 conference proposal made compliance binding violation is not normally cause for church discipline

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d8, d7
- **Supporting Docs Found**: None
- **Claim**: In the marine world, AUV stands for Autonomous Underwater Vehicle, which is an unmanned, untethered robot capable of operating underwater without human guidance, equipped with sensors such as cameras, sonar depth sensors used for research, surveying military applications


================================================================================

*Report generated by CATS v2.0*
