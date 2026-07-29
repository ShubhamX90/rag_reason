# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 9 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.531 (over 49 samples)

**GR F1** *(used in CATS)*: 0.596

**Behavior Adherence**: 0.600 (over 40 applicable samples)

**Factual Grounding**: 0.325 (over 40 applicable samples)

**Single-Truth Recall**: 0.588 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.527

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.596
- **Precision**: 0.739
- **Recall**: 0.500
- **Accuracy**: 0.531
- TP=17, FP=6, FN=17, TN=9


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.696
- **Behavior**: 0.533 (n=15)
- **Grounding**: 0.400 (n=15)
- **Recall**: 0.583 (n=12)
- **CATS**: 0.553

### Type 2: Complementary Info

- **Samples**: 15 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.667
- **GR F1** *(used in CATS)*: 0.706
- **Behavior**: 0.818 (n=11)
- **Grounding**: 0.455 (n=11)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.660

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.200
- **GR F1** *(used in CATS)*: 0.200
- **Behavior**: 0.444 (n=9)
- **Grounding**: 0.111 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.252

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.400
- **GR F1** *(used in CATS)*: 0.571
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.200 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.493


================================================================================

## Cost Summary

**Total Cost**: $0.0430

**Decisions Made**: 112

**Average Cost per Decision**: $0.000384


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 112
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0430
- **Total Requests**: 112
- **Average Cost per Request**: $0.000384


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
Based on the conflicting opinions presented in the documents, it is unclear whether children learn language skills from television

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d3, d2, d10, d1, d6, d9, d4, d7
- **Claim**: Some studies support the benefits of educational programs and co-viewing, while others emphasize the negative impact of child-directed viewing and the importance of real play and social interactions

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d2, d5, d10, d1, d6, d9, d4
- **Claim**: Chilliwack sings "My Girl (Gone, Gone, Gone)," and the lyrics confirm the specific phrase "Gone gone gone, she been gone so long."

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d5, d4, d8
- **Claim**: The current ATP top-ranked men's singles tennis player is Jannik Sinner, according to the most recent and credible evidence

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1, d6, d7
- **Claim**: However, the provided documents are outdated and do not reflect this information

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d2, d1, d6, d4, d7
- **Claim**: Heated gemstones are generally less valuable than unheated ones, especially for specific gemstones like sapphires and rubies

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d8
- **Claim**: However, the information provided does not cover all gemstones comprehensively, so a definitive answer for all gemstones cannot be given based on the available evidence

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d8, d1
- **Claim**: The legal age for marriage in the United States is generally 16, 17 18 years old, with recent changes in Massachusetts raising the age to 18 with no exceptions

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d11, d10
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
Based on the conflicting opinions presented in the documents, it is not definitively clear whether every startup needs a business plan

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d8, d3, d5, d10, d1, d6, d9, d4, d7
- **Claim**: Some sources emphasize the importance of a business plan for startups, while others suggest that it is not always necessary, especially for those not seeking outside funding

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d3, d10, d1, d6, d9, d4
- **Claim**: Therefore, the answer depends on the specific circumstances and goals of the startup

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d1, d8, d2, d10
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
The evidence shows conflicting opinions on whether sitting is the new smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d8, d3, d2, d5, d10, d1, d6, d9, d4, d7
- **Claim**: Some documents emphasize the health risks of sitting, comparing it to smoking, while others argue that the risks of smoking are significantly higher and that sitting is not as dangerous

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d9, d10, d1
- **Claim**: A comprehensive answer would need to consider the relative risks and benefits of both behaviors, taking into account the specific health outcomes and economic impacts discussed in the documents

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d2, d1, d6, d9, d7
- **Claim**: Final answer with proper citations:
Public transportation can be faster or slower than driving, depending on the specific context and location

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d8
- **Claim**: For example, in some cities and during certain times, public transportation may be faster due to dedicated lanes or reduced traffic, as seen in Mexico City

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d1
- **Claim**: However, in many metro areas, driving is significantly faster, as indicated by Governing's study ()

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d6, d9, d7
- **Claim**: Therefore, the answer to whether public transportation is faster than driving in cities depends on the specific urban area and circumstances

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d2, d5, d1, d6, d4, d7
- **Claim**: Based on the evidence, dogs can understand human language to some degree, but the extent of this understanding varies

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d5, d1, d6, d4, d7
- **Claim**: Some studies suggest that dogs can understand specific words and commands, while others emphasize the importance of context and body language

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d1, d6, d4, d7
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
All octopuses are venomous, as stated in multiple reliable sources

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d5, d1, d6, d4, d7
- **Claim**: Therefore, the answer to whether most octopuses are venomous is yes

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d5, d6, d7
- **Claim**: The 2024 Nobel Prize in Literature is awarded to the South Korean author Han Kang, as confirmed by multiple reliable sources

### Sample #0333

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d2, d5, d10, d1, d6, d4, d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d7, d1
- **Claim**: The Declaration of Independence was adopted on July 4, 1776, but the exact date of signing is disputed; some sources suggest August 2, 1776, while others indicate that signing occurred over time

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d2, d5, d1, d6, d9, d4, d7
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
The current world population is approximately 8 billion, as indicated by multiple recent sources

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d2, d10, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
CANNOT ANSWER

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d5, d9, d4, d7
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
Champagne comes solely from France, specifically the Champagne region

### Sample #0416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The meaning of AUV in the context of cars is not addressed by the retrieved documents

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, based on the evidence, AUV stands for autonomous underwater vehicle, which is used for underwater research and survey missions

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Club soda works well for stain removal, especially for beer and red wine stains, as supported by multiple documents

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The documents collectively suggest that UCLA has won at least ten NCAA basketball championships, with the most recent being in 1974-75

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact number of championships is not definitively stated across all documents

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The documents collectively agree that Dorothy lives somewhere other than the Emerald City, as she embarks on a journey to meet the Wizard of Oz there

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the specific location of her home is not mentioned in the provided evidence


================================================================================

*Report generated by CATS v2.0*
