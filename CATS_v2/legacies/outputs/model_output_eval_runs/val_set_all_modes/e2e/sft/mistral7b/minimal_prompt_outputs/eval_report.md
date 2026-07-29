# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.588 (over 34 applicable samples)

**Factual Grounding**: 0.890 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.825

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
- **Grounding**: 0.896 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.807

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.750 (n=8)
- **Grounding**: 0.812 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.854

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.444 (n=9)
- **Grounding**: 0.963 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.802

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.867 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.867


================================================================================

## Cost Summary

**Total Cost**: $0.0889

**Decisions Made**: 151

**Average Cost per Decision**: $0.000589


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 149
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0889
- **Total Requests**: 149
- **Average Cost per Request**: $0.000596


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This victory gave her a $250,000 cash prize and a recording contract with Hollywood Records

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6, d10
- **Supporting Docs Found**: None
- **Claim**: The song appears on their 1981 album *Gone, Gone, Gone* was also released as a standalone single with "My Girl" on the A-side and "Lonesome Loser" on the B-side

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7, d1
- **Supporting Docs Found**: None
- **Claim**: He has consistently held the #1 spot since approximately November 2023, when he surpassed Novak Djokovic to claim the ATP Finals title in Turin

### Sample #0104

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This ranking is corroborated by multiple

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9, d2, d4, d1
- **Supporting Docs Found**: None
- **Claim**: Finland, three people affiliated with Princeton University have been awarded the medal: June Huh, Hugo Duminil-Copin James Maynard

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d2, d8
- **Claim**: Ultimately, the value difference between heated and unheated gemstones depends heavily on the specific stone

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d7
- **Supporting Docs Found**: d8
- **Claim**: In addition to its global workforce, Mercedes-Benz USA has over 300 dealerships across the United States, each with its own staff, bringing the total number of people employed under the Mercedes-Benz brand to approximately 167,000–168,000

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d14, d4, d11, d12, d3, d10, d13, d5, d1
- **Claim**: In the general case, melting sea ice does not contribute to sea level rise because icebergs and sea ice are already floating in the ocean and displace the same volume of water they add when they melt — the volume of water they occupy is the same whether they are solid or liquid the weight of the ice does not change the water level (Archimedes' principle)

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d4, d7, d2, d5, d1
- **Claim**: This is consistent with the consensus across multiple sources, which variously report 3–16 books depending on the time period and source consulted

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d7, d4, d1
- **Supporting Docs Found**: d10
- **Claim**: No, champagne does not come solely from France

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d4, d6, d7, d3
- **Supporting Docs Found**: None
- **Claim**: In a broader underwater context, AUV also means Autonomous Underwater Vehicle (AUV), which are unmanned, untethered robots programmed to operate underwater without human control, used for research, survey missions military operations


================================================================================

*Report generated by CATS v2.0*
