# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 4 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.735 (over 49 samples)

**GR F1** *(used in CATS)*: 0.831

**Behavior Adherence**: 0.622 (over 45 applicable samples)

**Factual Grounding**: 0.752 (over 45 applicable samples)

**Single-Truth Recall**: 0.853 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.765

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.831
- **Precision**: 0.744
- **Recall**: 0.941
- **Accuracy**: 0.735
- TP=32, FP=11, FN=2, TN=4


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (3 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.737
- **GR F1** *(used in CATS)*: 0.815
- **Behavior**: 0.625 (n=16)
- **Grounding**: 0.724 (n=16)
- **Recall**: 0.875 (n=12)
- **CATS**: 0.760

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.786 (n=14)
- **Grounding**: 0.756 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.756

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.400 (n=10)
- **Grounding**: 0.792 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.713

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.750 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.760


================================================================================

## Cost Summary

**Total Cost**: $0.0816

**Decisions Made**: 174

**Average Cost per Decision**: $0.000469


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 174
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0816
- **Total Requests**: 174
- **Average Cost per Request**: $0.000469


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d1, d4, d5, d11, d2, d8, d3
- **Claim**: Children can learn language skills from television, but excessive screen time can be detrimental to language development

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2, d7, d1
- **Claim**: Overall, a balanced approach to screen time and social interaction is essential for promoting language development in children

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d1, d4, d5, d10, d9, d2, d8, d3
- **Claim**: 2024, she remains in office, but I have insufficient evidence to confirm whether she is the most recent woman to become President of Peru

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Princeton University has been affiliated with 15 Fields Medalists , but this information is not verified by other sources

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The exact number of Princeton University Fields Medal winners is not known with certainty

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d6, d3, d5
- **Claim**: The concept of innate knowledge is complex and multifaceted, with different philosophers and researchers holding different perspectives on the issue

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Overall, the evidence suggests that the relationship between innate knowledge and sensory experience is not straightforward that different theories and perspectives are necessary to fully understand the issue

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d4, d10, d11, d9, d3
- **Claim**: While there is no consensus on whether every startup needs a business plan, most sources agree that a business plan can be helpful in getting financing and attracting investors

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d10, d11, d9, d8
- **Claim**: The comparison between sitting and smoking is not supported by the available scientific evidence

### Sample #0229

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: Public health campaigns and smoking cessation interventions are essential to reduce the incidence of RA

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d1, d9, d2, d8, d3
- **Claim**: Ultimately, the choice between public transportation and driving depends on various factors, including the specific city, transportation options individual preferences

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d1
- **Supporting Docs Found**: d5
- **Claim**: However, the extent to which dogs understand human language is still a topic of debate more research is needed to fully understand canine language capabilities

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d1, d4, d8, d3
- **Claim**: All octopuses are venomous, but the severity of their venom varies

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d8
- **Claim**: Although some signers were not present on that day, including Elbridge Gerry, Oliver Wolcott, Lewis Morris, Thomas McKean Matthew Thornton, they eventually signed the document

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: This represents roughly 6% of the estimated 106 billion people who have ever lived on Earth

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d10, d9, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d4, d3, d7
- **Supporting Docs Found**: None
- **Claim**: AUV stands for Autonomous Underwater Vehicle, an unmanned underwater robot used for conducting underwater research and exploration

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This definition is widely accepted and supported by high-credibility sources, including scientific organizations and dictionaries

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This is consistent with the physical properties of silver, which has a reflectivity of 95%-99% over the visible spectrum

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: George Gershwin is likely the writer of the jazz classic "I Got Rhythm", based on the indirect connections between

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact authorship is not directly confirmed in the retrieved documents

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This series features Oliver as a main character, although it is not clear if it meets the query's criteria for a series with a large cast

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: This is why many outdoor signs are yellow-green, as it's easier for the eye to pick up on this color in low-light conditions

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This is the highest number of gold medals mentioned in any of the documents, but it does not provide a clear answer to the question of who won the most gold medals overall

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: There may be additional championships not mentioned in these documents

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This equation takes into account the decimal odds for both options and calculates the margin as a percentage

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Dorothy lives outside of the Emerald City in the magical Land of Oz, as she embarks on a journey with Toto on the Yellow Brick Road to meet the Wizard of Oz in the Emerald City

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is implied by her desire to return home, which suggests that she has a residence elsewhere in the Land of Oz

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: This process removes the minerals that cause water to be hard, resulting in soft water

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This definition is straightforward and easy to understand, making it an accurate and easily explained concept of gravity


================================================================================

*Report generated by CATS v2.0*
