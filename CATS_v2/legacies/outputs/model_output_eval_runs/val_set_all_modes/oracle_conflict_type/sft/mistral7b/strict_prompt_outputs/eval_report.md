# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.618 (over 34 applicable samples)

**Factual Grounding**: 0.897 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.835

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
- **Behavior**: 0.417 (n=12)
- **Grounding**: 0.875 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.781

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.625 (n=8)
- **Grounding**: 0.958 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.861

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.667 (n=9)
- **Grounding**: 0.852 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.840

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.933 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.933


================================================================================

## Cost Summary

**Total Cost**: $0.0937

**Decisions Made**: 153

**Average Cost per Decision**: $0.000613


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 153
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0937
- **Total Requests**: 153
- **Average Cost per Request**: $0.000613


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d10, d6
- **Supporting Docs Found**: None
- **Claim**: While Carter's Season 22 win is the most recent documented, it is worth noting that Season 23 of American Idol has since premiered, featuring a new group of contestants

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d9
- **Supporting Docs Found**: None
- **Claim**: The song appears on their 1981 album *Gone, Gone, Gone* has also been covered by The Recess Monkeys

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d10, d7, d3
- **Supporting Docs Found**: d9, d4
- **Claim**: The lyrics 'gone gone gone she been gone so long' are repeated throughout the song, with Chilliwack's lead vocalist Brian MacLeod delivering the powerful, emotive performance

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d4
- **Supporting Docs Found**: None
- **Claim**: He has held this position since at least March 2022, when he surpassed Alexander Zverev for the #1 spot has maintained his lead through the 2024 season

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3, d7, d6
- **Claim**: The consensus among high-credibility sources is that stress does not cause stomach ulcers on its own: the American Psychological Association and Johns Hopkins Medicine both state that stress does not increase stomach acid production or cause ulcers that ulcers are actually caused by bacterial infections (H. pylori) or NSAID medications; a peer-reviewed study also notes that stress may constrict capillaries in the stomach lining and impair mucosal production, but these changes do not directly cause ulcers

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d4, d6
- **Claim**: However, some sources present a different view — stress ulcers are a recognized clinical condition caused by physiological stress stress can impair the body's ability to heal existing ulcers or make it more prone to developing them , so while stress may not be the direct culprit, it may still play a role in ulcer formation

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d9, d4
- **Supporting Docs Found**: d5
- **Claim**: At least two people from Princeton University have won the Fields Medal, the most prestigious award in mathematics

### Sample #0187

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This figure includes employees at its headquarters in Stuttgart, production facilities, research and development centers sales operations around the world

### Sample #0206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: The best-powered evidence from large prospective studies suggests that replacing prolonged sitting with moderate-intensity exercise (60–75 minutes per day) can mitigate the risks significantly, though complete elimination of sitting remains uncertain

### Sample #0320

- **Reason**: cross_doc_not_cited
- **Cited Docs**: d1, d2, d7, d4
- **Supporting Docs Found**: None
- **Claim**: The blue-ringed octopus is the most infamous example, carrying a venom so powerful it can kill a human in minutes, while the venom of common octopuses (Octopus vulgaris) contains cephalotoxin, a toxin capable of causing paralysis and other harmful effects in humans


================================================================================

*Report generated by CATS v2.0*
