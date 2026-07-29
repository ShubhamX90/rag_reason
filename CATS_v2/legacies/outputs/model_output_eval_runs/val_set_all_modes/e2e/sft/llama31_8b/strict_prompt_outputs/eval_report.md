# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.980 (over 49 samples)

**GR F1** *(used in CATS)*: 0.985

**Behavior Adherence**: 0.735 (over 34 applicable samples)

**Factual Grounding**: 0.900 (over 34 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.876

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
- **GR Accuracy**: 0.947
- **GR F1** *(used in CATS)*: 0.957
- **Behavior**: 0.667 (n=12)
- **Grounding**: 0.847 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.826

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.750 (n=8)
- **Grounding**: 0.875 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.875

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
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.933 (n=5)
- **Recall**: 1.000 (n=5)
- **CATS**: 0.933


================================================================================

## Cost Summary

**Total Cost**: $0.0925

**Decisions Made**: 154

**Average Cost per Decision**: $0.000601


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 154
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0925
- **Total Requests**: 154
- **Average Cost per Request**: $0.000601


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d1, d3
- **Claim**: This result is consistent across multiple sources, with Wikipedia confirming Carter as the season 22 winner and ET and E!

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d9, d10
- **Claim**: Risk estimates for smoking far outweigh those for sitting except in the case of type 2 diabetes, further complicating the comparison

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Research also confirms that perceived travel time is a critical factor in mode choice — a 15-minute reduction in commute time corresponded to about 25% higher ridership in a New York study — suggesting that improving public transit speed is key to increasing ridership and competitiveness with driving

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8
- **Supporting Docs Found**: d4
- **Claim**: As the most recently appointed justice, Jackson's appointment date is corroborated by the Center for American Women and Politics at Rutgers University, which notes that she joined the Court on June 30, 2022

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0373

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6, d1
- **Supporting Docs Found**: None
- **Claim**: The signers were arranged by state, with New Hampshire delegates (Josiah Bartlett and Matthew Thornton) signing last due to space constraints

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d9
- **Claim**: The Americans reclaimed the top spot after overtaking Spain in the September 2023 update, a move also confirmed by AIPS Media's rankings that list USA at 832.2 points , with Spain dropping to second at 580 points and Germany at 740.2 in third

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d7, d4, d3
- **Supporting Docs Found**: None
- **Claim**: In the marine science context, AUV stands for Autonomous Underwater Vehicle — a unmanned, untethered robot operated by an onboard computer that conducts underwater research, surveying mapping missions

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: The two definitions are mutually exclusive: AUVs used in oceanography are entirely separate from the Asian Utility Vehicle designation found in some Asian car markets


================================================================================

*Report generated by CATS v2.0*
