# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.735 (over 34 applicable samples)

**Factual Grounding**: 0.909 (over 34 applicable samples)

**Single-Truth Recall**: 0.853 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.874

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
- **Grounding**: 0.889 (n=12)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.868

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.823 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.899

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.667 (n=9)
- **Grounding**: 1.000 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.889

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.933 (n=5)
- **Recall**: 0.700 (n=5)
- **CATS**: 0.858


================================================================================

## Cost Summary

**Total Cost**: $0.0888

**Decisions Made**: 153

**Average Cost per Decision**: $0.000581


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 153
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0889
- **Total Requests**: 153
- **Average Cost per Request**: $0.000581


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: 15 Princeton University alumni have won the Fields Medal , making Princeton second only to Harvard (with 18 Fields Medalists) on the all-time list

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d11
- **Supporting Docs Found**: d7, d8, d1, d4, d2, d6, d3
- **Claim**: Heated gemstones are generally less valuable than their unheated counterparts, though the extent depends on the specific gemstone variety and treatment intensity

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d6
- **Supporting Docs Found**: d2, d1
- **Claim**: The channel officially went on air in the UK to reach approximately 65% of the population a month later it launched on satellite to expand coverage nationwide

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7, d6
- **Supporting Docs Found**: d8
- **Claim**: Han Kang becomes the first Nobel laureate in literature from South Korea her unique poetic style has been described as an innovator in contemporary prose

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: The Senate Judiciary Committee had held hearings on her nomination President Biden nominated her following the retirement of Justice Stephen Breyer

### Sample #0334

- **Reason**: cross_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The discrepancy reflects different contextual scopes — the three-book figure likely refers to her most prominent works, while Goodreads tracks all published titles including academic articles and edited volumes

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: Stan Lee's original vision for the character included Peter designing the web-shooters himself as a demonstration of his intellect, which is why the comics also eventually retroactively established that Peter developed organic web shooters as a practical evolution of his powers

### Sample #0392

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: This is corroborated by the FIFA/Coca-Cola Men's World Ranking, which explicitly lists Argentina as the top ranked nation is consistent with the USA's continued dominance in the game

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7, d8, d3, d4
- **Supporting Docs Found**: None
- **Claim**: In the marine robotics field, AUV stands for Autonomous Underwater Vehicle — a free-swimming, untethered robot controlled by an onboard computer, used for underwater research, surveying military operations


================================================================================

*Report generated by CATS v2.0*
