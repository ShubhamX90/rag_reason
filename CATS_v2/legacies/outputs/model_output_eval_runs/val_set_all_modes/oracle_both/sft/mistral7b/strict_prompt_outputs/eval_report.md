# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.980 (over 49 samples)

**GR F1** *(used in CATS)*: 0.985

**Behavior Adherence**: 0.824 (over 34 applicable samples)

**Factual Grounding**: 0.870 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

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
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.667 (n=12)
- **Grounding**: 0.785 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.821

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.933
- **GR F1** *(used in CATS)*: 0.933
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.802 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.870

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.889 (n=9)
- **Grounding**: 1.000 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.963

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.950 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.938


================================================================================

## Cost Summary

**Total Cost**: $0.0911

**Decisions Made**: 153

**Average Cost per Decision**: $0.000596


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 153
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0911
- **Total Requests**: 148
- **Average Cost per Request**: $0.000616


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This victory gave her a $250,000 cash prize and a recording contract with Hollywood Records

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6, d10
- **Supporting Docs Found**: None
- **Claim**: The song appears on Chilliwack's 1981 album Anthology and has also been released as a standalone single, with the band's lead vocalists Brian MacLeod and Bill Henderson sharing singing duties throughout

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: This song is not to be confused with a cover version by The Recess Monkeys, which is a separate track with similar wording but different lyrics

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d3
- **Supporting Docs Found**: d4
- **Claim**: India became the sixth country in the world to adopt postal codes, following Germany (1944), Singapore (1950), Argentina (1958), the U.S. (1963) Switzerland (1964) the system has since been expanded to 8 postal regions covering over 1.5 lakh post offices

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d10
- **Claim**: In the UK, the path to Channel 5's launch was longer

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: If you smoke, it is essential to discuss smoking cessation with your healthcare provider, as quitting can reduce your risk and improve disease outcomes

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: This victory was confirmed across multiple sources, with Pegasus Spiele, the game's publisher, also releasing an official press release

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d8
- **Claim**: Even deep-sea octopuses, which are rarely studied, have been found to possess venom glands and use them for defense and prey capture

### Sample #0324

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d7
- **Supporting Docs Found**: None
- **Claim**: Han Kang, born in 1970, is the first Nobel laureate in literature from South Korea her work spans many genres including fiction, poetry essays

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

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE


================================================================================

*Report generated by CATS v2.0*
