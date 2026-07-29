# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.706 (over 34 applicable samples)

**Factual Grounding**: 0.885 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.854

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
- **Behavior**: 0.500 (n=12)
- **Grounding**: 0.889 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.806

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.844 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.906

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
- **Grounding**: 0.800 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.850


================================================================================

## Cost Summary

**Total Cost**: $0.0931

**Decisions Made**: 157

**Average Cost per Decision**: $0.000593


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 157
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0931
- **Total Requests**: 157
- **Average Cost per Request**: $0.000593


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d6
- **Supporting Docs Found**: d9, d4
- **Claim**: This date, describing it as the day the silver jubilee of India's independence was observed confirming that the system came into effect nationwide on that date

### Sample #0103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d9, d8, d6, d7, d5
- **Supporting Docs Found**: None
- **Claim**: Multiple authoritative sources confirm Boluarte as the definitive answer to this query, with no contradictions across any retrieved document

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d7
- **Supporting Docs Found**: d9, d8, d5, d4
- **Claim**: Jannik Sinner is the current world No. 1 ATP-ranked men's singles player, having surpassed Novak Djokovic as the top-ranked player after clinching his seventh ATP Finals title in Turin on 19 November 2023

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The ATP Live Rankings page further confirms Sinner's position at #1, reflecting the post-November 2023 update

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d4
- **Supporting Docs Found**: d9
- **Claim**: However, some color varieties—such as red sapphires (garnets) or blue zircon—are actually more valuable when heated, as heating can deepen or improve their color

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Sky Team beat out strong contenders like *Captain Flip* and *Daybreak* to claim the award, marking the first time a two-player game has won the traditional family-focused Kennerspiel des Jahres category

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The most up-to-date and authoritative source, Goodreads, confirms the highest count of 16 books published by Shoshana Zuboff

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d5, d4
- **Claim**: This cinematic divergence — where the comics show a gradual evolution to organic shooters while the Raimi film presents them as the default — reflects different creative choices about how to portray Peter Parker's natural integration with spider-like abilities is further explored in various comic continuities where Peter eventually develops organic shooters organically

### Sample #0392

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: Argentina is currently the top-ranked country

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d6, d7, d1, d5
- **Claim**: While the original revelation (received 27 February 1833) described the Word of Wisdom as "not by commandment or constraint," President John Smith's 1851 proposal made it a binding commandment for all Saints

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d7
- **Supporting Docs Found**: None
- **Claim**: The term AUV typically refers to an **autonomous underwater vehicle** (AUV), which is an unmanned, untethered vehicle designed to operate underwater without direct human control

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d4
- **Supporting Docs Found**: None
- **Claim**: These vehicles are programmed using preloaded instructions and can carry a variety of equipment such as cameras, sonar depth sensors, making them useful for underwater research, surveying military operations


================================================================================

*Report generated by CATS v2.0*
