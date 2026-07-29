# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 7 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.816 (over 49 samples)

**GR F1** *(used in CATS)*: 0.880

**Behavior Adherence**: 0.714 (over 42 applicable samples)

**Factual Grounding**: 0.621 (over 42 applicable samples)

**Single-Truth Recall**: 0.794 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.752

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.880
- **Precision**: 0.805
- **Recall**: 0.971
- **Accuracy**: 0.816
- TP=33, FP=8, FN=1, TN=7


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (3 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.789
- **GR F1** *(used in CATS)*: 0.857
- **Behavior**: 0.875 (n=16)
- **Grounding**: 0.583 (n=16)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.808

### Type 2: Complementary Info

- **Samples**: 15 (4 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.842
- **Behavior**: 0.818 (n=11)
- **Grounding**: 0.652 (n=11)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.771

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.400 (n=10)
- **Grounding**: 0.558 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.635

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.800 (n=5)
- **Recall**: 0.500 (n=5)
- **CATS**: 0.697


================================================================================

## Cost Summary

**Total Cost**: $0.0691

**Decisions Made**: 148

**Average Cost per Decision**: $0.000467


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 148
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0691
- **Total Requests**: 148
- **Average Cost per Request**: $0.000467


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d7, d12, d2
- **Claim**: Research indicates that children's language skills are significantly influenced by the quality of interactions they have with adults

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d6, d2
- **Claim**: This date marks the beginning of the company's operations as a monopolistic trading body in the East Indies

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d6, d1
- **Claim**: Therefore, stress can play a role in the exacerbation of ulcers but is not the primary cause

### Sample #0139

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This number is based on the information provided by multiple high-credibility sources, including Princeton University and Wikipedia

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d2, d3, d5, d6
- **Claim**: The concept of innate knowledge is complex and debated among philosophers and psychologists

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d8, d5, d6, d1
- **Claim**: Some historical figures like Plato and Descartes argued for the existence of innate knowledge, while others, such as John Locke, rejected this idea

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d9
- **Supporting Docs Found**: None
- **Claim**: The legal age for marriage in the United States is 18, with no exceptions, as of June 1, 2020, based on the most recent and authoritative sources

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d9
- **Claim**: This information is consistent with the general trend of raising the minimum age for marriage across the country

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d4, d10, d11, d3, d6, d1
- **Claim**: While a business plan is beneficial for most startups, especially those seeking outside funding and alignment with team goals, it is not strictly necessary for every single startup

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d1, d9
- **Claim**: While there is a popular belief that "sitting is the new smoking," the evidence suggests that the health risks associated with smoking are much higher than those of sitting

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3, d5, d6, d1
- **Claim**: Smoking can cause Rheumatoid Arthritis (RA)

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d6, d1
- **Claim**: These sources include governmental health organizations and peer-reviewed journals, providing strong evidence for this causal relationship

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d2, d4, d8, d3, d5, d6, d1
- **Claim**: Dogs can understand human language to some degree, particularly in recognizing familiar words and tones

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d8, d3, d5, d6, d1
- **Claim**: However, the exact proportion of venomous octopuses is not quantified in the documents, suggesting that while the majority are venomous, not all species have equally potent venom

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d13, d9, d12, d4, d11, d8, d3, d14, d1
- **Claim**: The primary contributors to sea level rise are the melting of land ice, such as glaciers and ice sheets from Greenland and Antarctica

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: These sources list her major works, which align with the consistent count of 4 books

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9
- **Supporting Docs Found**: d4
- **Claim**: The signatories included future presidents, vice presidents members of the United States Congress

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d2, d10, d6, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: Therefore, pregnant women considering a vegan diet should consult with a healthcare provider and a registered dietitian to ensure they meet all necessary nutrient requirements

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d2, d4, d8, d3, d5, d6, d1
- **Claim**: Champagne comes solely from the Champagne region of France

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d2, d4, d8, d3, d5, d6, d1
- **Claim**: Other regions can produce sparkling wine, but only those from the Champagne region can be called Champagne

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d7
- **Supporting Docs Found**: d6
- **Claim**: The Word of Wisdom became a commandment in 1921, when it was required for temple recommends

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d7
- **Supporting Docs Found**: d1, d6
- **Claim**: Prior to this, it was seen as a covenant proposed by President Brigham Young in 1851, but it did not become a mandatory commandment until 1921

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2
- **Claim**: In the context of cars, AUV stands for Asian Utility Vehicle, a type of passenger vehicle designed primarily for carrying a large number of passengers

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Club soda works well for stain removal because it contains carbon dioxide gas dissolved in water, creating a slightly acidic environment that helps break down protein-based stains

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The carbonation in club soda also aids in lifting the stain from the fabric surface

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bath towels, especially those made of pima cotton, seem to get more absorbent the more they are washed over time

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is because the fibers in these towels become more intertwined and better at wicking water, leading to increased absorbency

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: While the exact differences between socialism and communism are not explicitly detailed in the documents, it is clear that socialism is seen as a stepping stone to achieving the ultimate goal of communism

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This implies that socialism focuses on creating a more equitable society through state intervention, whereas communism aims for a classless, stateless society where resources are distributed based on need rather than individual merit

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Night vision devices display images in green because green wavelengths fall within the part of the light spectrum where the human eye is most sensitive in low-light conditions

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d2
- **Supporting Docs Found**: None
- **Claim**: UCLA has won 4 NCAA basketball championships

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The additional 6 championships mentioned in are not specified as basketball championships, so they are not included in this count

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Dorothy lived in Kansas before embarking on her journey to the Emerald City in the magical Land of Oz

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is where she started her adventure with Toto and met the Scarecrow, Tin Thing Cowardly Lion

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This definition is straightforward and aligns with the basic understanding of gravity without introducing overly complex concepts


================================================================================

*Report generated by CATS v2.0*
