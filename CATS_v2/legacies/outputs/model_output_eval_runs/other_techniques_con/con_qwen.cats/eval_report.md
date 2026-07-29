# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**GR Accuracy**: 0.694 (over 49 samples)

**GR F1** *(used in CATS)*: 0.819

**Behavior Adherence**: 0.633 (over 49 applicable samples)

**Factual Grounding**: 0.017 (over 49 applicable samples)

**Single-Truth Recall**: 0.765 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.558

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.819
- **Precision**: 0.694
- **Recall**: 1.000
- **Accuracy**: 0.694
- TP=34, FP=15, FN=0, TN=0


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.774
- **Behavior**: 0.737 (n=19)
- **Grounding**: 0.000 (n=19)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.607

### Type 2: Complementary Info

- **Samples**: 15
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.696
- **Behavior**: 0.733 (n=15)
- **Grounding**: 0.022 (n=15)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.484

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.400 (n=10)
- **Grounding**: 0.050 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.466

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.400 (n=5)
- **Grounding**: 0.000 (n=5)
- **Recall**: 0.400 (n=5)
- **CATS**: 0.450


================================================================================

## Cost Summary

**Total Cost**: $0.0725

**Decisions Made**: 155

**Average Cost per Decision**: $0.000468


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 155
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0725
- **Total Requests**: 155
- **Average Cost per Request**: $0.000468


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d7, d3, d1, d2
- **Claim**: Abi Carter won the 22nd season of American Idol, which took place in 2024

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d6, d4, d7, d3, d1, d2
- **Claim**: Children can learn language skills from television, particularly when it is educational and accompanied by adult interaction

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d4, d7, d3, d1, d2
- **Claim**: However, the quality of the programming and the presence of a competent adult are crucial

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d6, d10, d7, d11, d1
- **Claim**: Child-directed viewing may negatively impact language development, while adult-directed viewing and co-viewing can promote language acquisition

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d12, d10, d1
- **Claim**: Therefore, while television can be a tool for language learning, it is not as effective as real-life interactions and play

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d10, d4, d3, d1, d2
- **Claim**: Chilliwack sings "My Girl (Gone, Gone, Gone)."

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d4, d7, d3, d1, d2
- **Claim**: The PIN code system was introduced in India on August 15, 1972

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d10, d4, d7, d3, d1, d2
- **Claim**: Dina Boluarte is the most recent woman to become President of Peru, having taken office on December 7, 2022, after her predecessor Pedro Castillo was impeached

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: Novak Djokovic is the current ATP top-ranked men's singles tennis player

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d7, d3, d1, d2
- **Claim**: The East India Company was founded on December 31, 1600, under a royal charter granted by Queen Elizabeth I

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d7, d3, d1
- **Claim**: Based on the information provided, stress does not cause stomach ulcers on its own

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d4
- **Claim**: However, it can contribute to their development by prompting the use of NSAIDs and other behaviors that exacerbate ulcers

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The primary causes of ulcers are H. pylori infection and the use of NSAIDs

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Based on the information provided, Princeton University has 15 Fields Medalists as of 2022

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d4, d7, d3, d1, d2
- **Claim**: Heated gemstones are generally less valuable than unheated gemstones, unless they are of the same quality in all aspects

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d3, d6
- **Claim**: The value difference can be significant, especially for larger stones

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d10, d4, d7, d3, d11, d1, d2
- **Claim**: Unheated gemstones are rarer and more valuable, but heat treatment is a common practice that can improve the appearance and value of many gemstones

### Sample #0175

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Based on the passages, there is evidence supporting both the existence and the non-existence of innate knowledge

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d7, d1
- **Claim**: Plato and Descartes argue for innate knowledge, while John Locke and other empiricists argue that knowledge is derived from sensory experience

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The passages also discuss biases against the idea of innate knowledge and ongoing research to understand innate abilities

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The number of employees in the Mercedes-Benz Group is 166,056 as of December 31, 2023

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 2020, the legal age for marriage in the United States is 18 years old in all states without exceptions

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d4, d7, d3, d1, d2
- **Claim**: However, some states have specific restrictions on minors marrying under 18

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d10, d4, d7, d3, d11, d1
- **Claim**: Based on the passages, while many emphasize the importance of a business plan for startups, others argue that it is not always necessary, especially for those not seeking outside funding

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d5, d6, d10, d4, d7, d3, d11, d1
- **Claim**: However, a business plan can provide a roadmap for success, help secure funding serve as a communication tool for stakeholders

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d10, d4, d7, d3, d11, d1
- **Claim**: Therefore, while not every startup necessarily needs a business plan, it can be beneficial for many

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d6, d10, d4, d7, d1, d2
- **Claim**: Channel 5 started broadcasting on March 30, 1997

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d10, d4, d7, d3, d11, d1, d2
- **Claim**: The passages provide mixed information on whether sitting is the new smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d10, d4, d7, d3, d11, d1, d2
- **Claim**: Some passages argue that sitting is a significant health risk and should be compared to smoking, while others refute this claim, stating that the health risks of sitting are much lower than those of smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d10, d11, d1
- **Claim**: Therefore, the answer is that sitting is not the new smoking, as supported by the evidence from multiple passages

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d4, d3, d1, d2
- **Claim**: Yes, smoking can cause Rheumatoid Arthritis

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d4, d3, d1, d2
- **Claim**: Smoking is a significant risk factor for developing RA and can exacerbate the disease

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Studies have shown that smoking interacts with genetic factors and can lead to an increased risk of developing anti-citrullinated protein antibody-positive RA

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d3, d1, d2
- **Claim**: Based on the information provided, public transportation is often slower than driving in most cities

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d9
- **Claim**: However, there are instances where public transportation can be faster, particularly with improvements in service and infrastructure

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d6, d7, d1, d2
- **Claim**: Therefore, the answer depends on the specific city and its transportation network

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d6, d4, d7, d3, d1, d2
- **Claim**: Dogs can understand human language to a certain extent, particularly through frequent commands and words they are familiar with

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d8, d1
- **Claim**: They can differentiate between words and intonation their understanding is influenced by both linguistic content and emotional content

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d3, d1
- **Claim**: However, their comprehension is not as extensive as human language they rely heavily on body language and tone

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d6, d8, d1
- **Claim**: Dorfromantik: The Board Game won the Spiel des Jahres award in 2023

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d6, d4, d7, d3, d1
- **Claim**: Yes, most octopuses are venomous

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d4, d7, d3, d1
- **Claim**: The passages consistently state that all octopuses are venomous, with the blue-ringed octopuses being the most deadly

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d4
- **Claim**: However, the venom of most octopuses is not harmful to humans

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d5, d13, d10, d4, d14, d12, d3, d11, d1
- **Claim**: Melting sea ice does not contribute to sea level rise because it is already in the water

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2
- **Claim**: However, the volume of meltwater from sea ice is slightly larger than the volume of water it displaces due to changes in salinity and density, which can cause a small increase in sea level

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d6, d7, d2
- **Claim**: Han Kang won the latest Nobel Prize in Literature in 2024

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d4
- **Claim**: The last person appointed to the U.S. Supreme Court was Ketanji Brown Jackson, who took her seat on June 30, 2022

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d10, d4, d3, d1
- **Claim**: Based on the information provided in the passages, Shoshana Zuboff has published at least 4 books, with the earliest being "In the Age of the Smart Machine: The Future of Work and Power" in 1988 and the latest being "The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power" in 2018

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d6
- **Claim**: Additional books are mentioned, but the exact number is not specified

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1, d2
- **Claim**: Based on the information provided, Spider-Man originally did not have organic web shooters in the comics

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: He acquired them later in the series

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Sam Raimi's Spider-Man movies, however, introduced the idea of organic web shooters from the beginning, which deviated from the comics' storyline

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d6, d4, d7, d1, d2
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Continental Congress

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d1, d4
- **Claim**: The signing process began on August 2, 1776, with John Hancock being the first to sign

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8
- **Claim**: Some delegates signed later Thomas McKean was the last to sign

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d4, d7, d3, d1, d2
- **Claim**: The current world population, as of January 8, 2025, is 8,198,260,420 people

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9
- **Claim**: Historical estimates suggest that around 108 billion people have ever lived on Earth the world population is expected to peak by the end of the century

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d1, d4
- **Claim**: The current top-ranked country in the FIBA Men's World Ranking is the USA

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d4, d7, d2
- **Claim**: Pregnant women can follow a vegan diet if it is well-planned and supplemented appropriately to meet all nutritional needs

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d4, d7, d3, d1, d2
- **Claim**: While some studies suggest that vegan diets during pregnancy can be safe and even beneficial, others caution about potential nutritional deficiencies

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1, d6
- **Claim**: Therefore, it is important for pregnant women to work with healthcare professionals to ensure they are meeting all their nutritional requirements

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d4, d7, d3, d1, d2
- **Claim**: Based on the information provided in the passages, Champagne can only come from the Champagne region of France

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d5, d6, d4, d7, d3, d1, d2
- **Claim**: The passages consistently emphasize that only sparkling wine from this region can be called Champagne that the name is protected by law

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d6, d5, d1
- **Claim**: The Word of Wisdom became a commandment in 1851, when President Brigham Young proposed that all Saints formally covenant to keep it

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: However, it did not become a requirement for temple recommends until 1921

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The term AUV in the context of cars refers to Asian Utility Vehicles, which are primarily used for carrying passengers and are not designed for off-road use

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: They are typically found in the Asian market and are designed to seat 8-10 people and haul goods for commercial purposes

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Club soda works well for stain removal due to its carbonation and acidity, which help break down and lift stains

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The carbonation creates bubbles that can help push the stain away from the fabric, while the acidity can help dissolve certain types of stains

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bath towels seem to get more absorbent over time because the fibers become more aligned and the towel becomes more compact with repeated washing

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This process enhances the towel's ability to hold and retain water, making it more absorbent

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Our brains perceive reflective surfaces as silver because they reflect most of the visible light spectrum the brain interprets this reflection as a silver color

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: When white light hits a reflective surface, it reflects most of the colors the brain perceives this as silver

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passages do not provide a clear distinction between socialism and communism

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, based on the information given, socialism is described as a stage on the way to communism in Marxist theory, suggesting that communism is a more advanced form of socialism

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: But without a direct comparison, the exact differences remain unclear

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Based on the provided passages, "I Got Rhythm" was composed and arranged by George Gershwin, as mentioned in

### Sample #0561

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Based on the provided passages, there is no clear indication of which television series Jamie Oliver is a member of the cast for

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The passages mention his involvement in "Jamie's School Dinners" and his appearances on other chefs' shows, but not as a cast member

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Night vision is green because the human eye is more sensitive to the yellow-green part of the light spectrum, which is where the cones are more sensitive

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This sensitivity makes green the optimal color for night vision devices to use in order to enhance visibility in low-light conditions

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2
- **Claim**: Based on the provided passages, UCLA has won at least three NCAA basketball championships (1964-65, 1967 1974-75)

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact number of championships won by UCLA is not fully detailed in the given passages

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1, d2
- **Claim**: Bookmakers calculate odds by considering the probability of an event occurring, laying bets on outcomes not happening ensuring a balanced book

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: They use odds to represent the value of a bet and adjust these odds as events progress

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The exact methods used can vary, but the goal is to make a profit

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1, d2
- **Claim**: The provided passages do not give a detailed step-by-step process but suggest that bookmakers use a combination of probability, value assessment risk management to set odds

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: An accurate and easily explained definition of gravity is that it is a natural phenomenon by which all material bodies attract each other


================================================================================

*Report generated by CATS v2.0*
