# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 5 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.776 (over 49 samples)

**GR F1** *(used in CATS)*: 0.857

**Behavior Adherence**: 0.773 (over 44 applicable samples)

**Factual Grounding**: 0.782 (over 44 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.809

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.857
- **Precision**: 0.767
- **Recall**: 0.971
- **Accuracy**: 0.776
- TP=33, FP=10, FN=1, TN=5


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.842
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.667 (n=15)
- **Grounding**: 0.822 (n=15)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.824

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.929 (n=14)
- **Grounding**: 0.748 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.801

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.700 (n=10)
- **Grounding**: 0.728 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.772

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.867 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.817


================================================================================

## Cost Summary

**Total Cost**: $0.0843

**Decisions Made**: 179

**Average Cost per Decision**: $0.000471


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 179
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0843
- **Total Requests**: 178
- **Average Cost per Request**: $0.000474


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the conflicting information, it is important to consider the credibility of each source

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d5, d7, d3
- **Claim**: Stomach ulcers are not primarily caused by stress

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d5, d4, d3, d7, d8, d2
- **Claim**: The retrieved evidence suggests that heated gemstones are often less valuable than unheated ones, though the extent varies by type and specific circumstances

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2, d5
- **Claim**: The retrieved evidence presents conflicting views on whether humans have innate knowledge

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d1, d6, d7
- **Claim**: The legal age for marriage varies significantly by jurisdiction

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: The retrieved evidence presents conflicting views on whether every startup needs a business plan

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d11, d10, d8, d2, d9
- **Claim**: Citing the conflicting evidence, the claim that "sitting is the new smoking" is contested

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d11, d5, d4, d3, d7, d10, d9
- **Claim**: While some sources argue that sitting poses health risks comparable to smoking, others, including high-quality evidence from d1, indicate that sitting and smoking are distinct behaviors with different levels of associated risk

### Sample #0324

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6, d8, d2, d5
- **Supporting Docs Found**: None
- **Claim**: This information is confirmed by multiple high-quality sources

### Sample #0334

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The exact number of books published by Shoshana Zuboff cannot be determined from the retrieved evidence, as the sources provide conflicting counts

### Sample #0334

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the exact count varies based on the source consulted

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d5
- **Claim**: Spider-Man originally did not have organic web shooters; in the comics, he had mechanical web shooters that he designed himself

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d7, d4, d3, d8, d9
- **Claim**: The current world population is over 8 billion people

### Sample #0392

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The top-ranked country in the FIBA Men's World Ranking is currently disputed

### Sample #0399

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The evidence presents conflicting opinions on whether pregnant women should follow a vegan diet

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d4, d5, d2
- **Claim**: Other sources suggest that a vegan diet can be safe during pregnancy if carefully planned and supplemented with essential nutrients [d3-d7]

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d4, d5, d2
- **Claim**: Therefore, while a vegan diet may be safe for pregnant women, it requires careful planning and monitoring to ensure adequate nutrient intake [d4-d7]

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: AUV stands for Asian Utility Vehicle in the context of cars

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: It refers to vehicles designed for use in the Asian market, typically seating 8-10 people, capable of hauling goods serving commercial purposes

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, the exact reason for this phenomenon is not provided in the retrieved evidence

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: However, the specific reason why reflective surfaces like metal appear silver rather than showing all reflected colours remains unclear based on the provided evidence

### Sample #0561

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: While Contaldo appeared on Jamie Oliver's television show The Naked Chef, this does not confirm Oliver's role as a cast member

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: However, the exact total number of championships won by UCLA cannot be determined from the provided information

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: However, the specific methods and speed at which bookmakers adjust odds in real-time during live events are not detailed in the provided evidence

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence partially supports the query by identifying one book written by Mordecai Richler, titled "Oh Canada!

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, the evidence does not provide a complete list of his works

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d3
- **Claim**: The retrieved documents provide complementary information about water softening processes but do not fully explain the mechanism of how salt softens water


================================================================================

*Report generated by CATS v2.0*
