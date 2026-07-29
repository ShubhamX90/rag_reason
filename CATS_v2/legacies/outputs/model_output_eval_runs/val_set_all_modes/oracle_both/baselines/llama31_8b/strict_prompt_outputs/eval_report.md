# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 6 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.571 (over 49 samples)

**GR F1** *(used in CATS)*: 0.677

**Behavior Adherence**: 0.558 (over 43 applicable samples)

**Factual Grounding**: 0.484 (over 43 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.636

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.677
- **Precision**: 0.710
- **Recall**: 0.647
- **Accuracy**: 0.571
- TP=22, FP=9, FN=12, TN=6


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.789
- **GR F1** *(used in CATS)*: 0.846
- **Behavior**: 0.600 (n=15)
- **Grounding**: 0.678 (n=15)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.739

### Type 2: Complementary Info

- **Samples**: 15 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.467
- **GR F1** *(used in CATS)*: 0.556
- **Behavior**: 0.692 (n=13)
- **Grounding**: 0.523 (n=13)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.590

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.100
- **GR F1** *(used in CATS)*: 0.182
- **Behavior**: 0.200 (n=10)
- **Grounding**: 0.075 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.152

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.617 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.804


================================================================================

## Cost Summary

**Total Cost**: $0.0587

**Decisions Made**: 136

**Average Cost per Decision**: $0.000432


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 136
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0587
- **Total Requests**: 136
- **Average Cost per Request**: $0.000432


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d9, d8
- **Claim**: Jannik Sinner is the current ATP top-ranked men's singles tennis player, according to the majority of the retrieved documents ()

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5, d9, d8
- **Supporting Docs Found**: d7
- **Claim**: However, documents indicate that Novak Djokovic was the top-ranked player in the past, but this information is outdated

### Sample #0127

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The East India Company's legacy continues to be felt today, with its impact on global trade and politics still studied and debated by historians

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [CANNOT ANSWER, INSUFFICIENT EVIDENCE]

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d3, d5
- **Claim**: However, we can say that the most frequently cited employee count for Mercedes-Benz Group is 166,000 there is another count of 167,397

### Sample #0187

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The exact count cannot be determined from the retrieved evidence

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d9, d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0201

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: The evidence is consistent across all sources, with no conflicting information

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d5, d4, d6
- **Claim**: Therefore, the answer is that smoking can cause rheumatoid arthritis

### Sample #0263

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0301

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict due to outdated information is resolved by prioritizing the more recent evidence

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [CANNOT ANSWER, INSUFFICIENT EVIDENCE]

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3
- **Claim**: The most recent information indicates that Ketanji Brown Jackson is the correct answer

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Therefore, the final answer is based on the most recent estimate provided by d1

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0416

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The LDS Church's official stance on the matter is unclear further research is needed to determine the exact date when the Word of Wisdom became mandatory

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5
- **Claim**: AUV stands for Asian Utility Vehicle, a term used in the automotive industry to describe vehicles that are designed to seat 8-10 people, haul goods serve commercial purposes

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: The retrieved documents collectively confirm that club soda can be used for stain removal, but none of them provide a clear explanation of why it works so well

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved documents do not provide a clear explanation for why regular bath towels become more absorbent with repeated washing

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, d3 provides a related fact about Turkish cotton towels becoming more absorbent and feathery with washing

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The brain perceives reflective surfaces as silver because of the way light interacts with the surface and the selective reflection of certain wavelengths

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact reason why reflective metals appear silver rather than showing all reflected colours is not explicitly stated in the retrieved documents

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Jamie Oliver is a cast member of the documentary series Jamie's School Dinners, broadcast on Channel 4 in 2005

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: However, the total number of championships won by UCLA cannot be determined from the retrieved evidence

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The final answer is that the mechanism of how salt softens water is not fully explained by the retrieved documents

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2, d3
- **Claim**: However, they do provide some information on water softening methods, including salt-based systems

### Sample #0654

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A definition of gravity that can be easily explained is not clearly established in the retrieved documents

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: While d1 suggests a definition based on motion and bombardment d5 provides a concise definition of gravity as a natural phenomenon, neither offers a clear, easily explained definition

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The documents present conflicting views on the nature of gravity the query's request for an easily explained definition is not met


================================================================================

*Report generated by CATS v2.0*
