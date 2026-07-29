# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 7 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.837 (over 49 samples)

**GR F1** *(used in CATS)*: 0.895

**Behavior Adherence**: 0.786 (over 42 applicable samples)

**Factual Grounding**: 0.831 (over 42 applicable samples)

**Single-Truth Recall**: 0.853 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.841

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.895
- **Precision**: 0.810
- **Recall**: 1.000
- **Accuracy**: 0.837
- TP=34, FP=8, FN=0, TN=7


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.842
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.867 (n=15)
- **Grounding**: 0.878 (n=15)
- **Recall**: 0.875 (n=12)
- **CATS**: 0.877

### Type 2: Complementary Info

- **Samples**: 15 (3 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.733
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.917 (n=12)
- **Grounding**: 0.785 (n=12)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.834

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.500 (n=10)
- **Grounding**: 0.730 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.726

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.900


================================================================================

## Cost Summary

**Total Cost**: $0.0626

**Decisions Made**: 142

**Average Cost per Decision**: $0.000441


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 142
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0626
- **Total Requests**: 142
- **Average Cost per Request**: $0.000441


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d3, d2, d4, d8, d7
- **Claim**: Children can learn language skills from television, but the effectiveness depends on several factors

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d3, d2, d4, d7
- **Claim**: High-quality educational programs can promote language acquisition when accompanied by adult interaction and discussion

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d10
- **Supporting Docs Found**: None
- **Claim**: Further research is required to determine the exact number of Princeton University affiliates who have won the Fields Medal

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2
- **Claim**: The question of whether humans have innate knowledge is a matter of debate among philosophers and researchers

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3, d2, d4
- **Claim**: Proponents of innate knowledge, such as Plato and Descartes, argue that certain concepts and ideas are inherent to human cognition, as evidenced by the ability to understand mathematical truths and recognize objects without explicit learning ()

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d9
- **Claim**: On the other hand, empiricists like John Locke contend that all knowledge arises from sensory experience, suggesting that the mind begins as a blank slate ()

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9
- **Supporting Docs Found**: None
- **Claim**: The legal age for marriage in the United States varies by state, but as of June 1, 2020, the minimum marriageable age in all states is 18 without exceptions

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9, d6
- **Supporting Docs Found**: None
- **Claim**: Other states such as Connecticut, Delaware, New Jersey, New York, Pennsylvania, Rhode Island Vermont have similar laws setting the minimum age at 18 with no exceptions

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d11, d10, d2, d8
- **Claim**: The comparison of sitting to smoking is a contentious topic

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d9, d11, d6, d3, d10, d2, d4, d8, d7
- **Claim**: While some sources support the idea that prolonged sitting can be harmful and liken it to smoking, others argue that the risks are not equivalent

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d3, d4, d7
- **Claim**: However, all sources agree that prolonged sitting can lead to various health issues, including obesity, diabetes cardiovascular disease

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d7
- **Claim**: Public transportation is generally slower than driving in cities

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d3, d2, d4, d8, d7
- **Claim**: Dogs can understand human language to a certain extent

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d6, d2, d4, d7
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Continental Congress

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d2, d4, d7
- **Claim**: Pregnant women can follow a vegan diet, but it requires careful planning and supplementation to ensure all necessary nutrients are met

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d9, d4
- **Claim**: The Word of Wisdom became increasingly mandatory over time

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d2
- **Claim**: The term AUV in the context of cars stands for Asian utility vehicle

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: It refers to a type of vehicle primarily used in Asia, particularly in third-world countries, designed to seat 8-10 people, haul goods be used for commercial purposes

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: AUVs were first seen on the roads in the late '60s and early '70s, with examples including the Toyota Tamaraw, Ford Tiera series GM Harabas

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, the specific reasons behind its effectiveness are not clearly explained in the provided documents

### Sample #0561

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Jamie Oliver has appeared as a cast member in various television series, including his appearances as a sous chef on his own shows

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d3, d2, d4
- **Supporting Docs Found**: None
- **Claim**: In-play odds are adjusted quickly based on live events and changing probabilities, though the exact methods used for rapid recalculations aren't detailed in the provided documents

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Salt softens water by flushing through the resin inside the water softener to clean it and remove minerals


================================================================================

*Report generated by CATS v2.0*
