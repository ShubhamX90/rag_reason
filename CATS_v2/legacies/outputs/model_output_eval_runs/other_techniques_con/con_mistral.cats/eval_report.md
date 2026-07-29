# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**GR Accuracy**: 0.673 (over 49 samples)

**GR F1** *(used in CATS)*: 0.805

**Behavior Adherence**: 0.388 (over 49 applicable samples)

**Factual Grounding**: 0.010 (over 49 applicable samples)

**Single-Truth Recall**: 0.647 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.462

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.805
- **Precision**: 0.688
- **Recall**: 0.971
- **Accuracy**: 0.673
- TP=33, FP=15, FN=1, TN=0


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.774
- **Behavior**: 0.421 (n=19)
- **Grounding**: 0.000 (n=19)
- **Recall**: 0.667 (n=12)
- **CATS**: 0.465

### Type 2: Complementary Info

- **Samples**: 15
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.696
- **Behavior**: 0.533 (n=15)
- **Grounding**: 0.000 (n=15)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.410

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.100 (n=10)
- **Grounding**: 0.000 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.330

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.400 (n=5)
- **Grounding**: 0.100 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.525


================================================================================

## Cost Summary

**Total Cost**: $0.0557

**Decisions Made**: 131

**Average Cost per Decision**: $0.000425


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 131
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0558
- **Total Requests**: 131
- **Average Cost per Request**: $0.000426


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d11, d4, d7, d9
- **Claim**: While some studies suggest that well-designed educational television programs can improve cognitive, literacy social outcomes for children aged 3 to 5, the evidence for benefits of media for children younger than 2 years is limited

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d12, d11, d10, d7, d1, d9
- **Claim**: Back-and-forth conversation between children and caregivers is the most significant factor contributing to a child's language development, while adult monologuing and television viewing have no effect

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The given documents do not provide information about who sings the song "Gone Gone Gone She Been Gone So Long"

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d3, d6, d7, d4, d1, d9
- **Claim**: The PIN code system was introduced in India on August 15, 1972 was designed to streamline the sorting and delivery of mail

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d10, d3, d4, d6, d7, d9
- **Claim**: Dina Boluarte is the first female president of Peru, who took office following the impeachment and arrest of Pedro Castillo

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: She has pledged to fight corruption and has faced protests calling for her resignation and the scheduling of general elections

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d1
- **Claim**: The given documents do not provide the current ATP top-ranked men's singles tennis player

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d7, d6, d9
- **Claim**: The East India Company was established in 1600 under a royal charter by Queen Elizabeth I, with the primary purpose of trading with East and Southeast Asia and India

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d7, d6
- **Claim**: It became a powerful commercial and political organization its supremacy in India was confirmed in 1765

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d6, d7, d4, d1
- **Claim**: Stress does not directly cause stomach ulcers, but it can impair the body's ability to heal, making one more prone to developing a peptic ulcer

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d6, d4, d1
- **Claim**: The main causes of peptic ulcers are H. pylori infection and NSAIDs

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The Fields Medal is a prestigious award given to mathematicians under the age of 40 for outstanding mathematical achievement

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d4, d9
- **Claim**: In 2022, June Huh from Princeton University was one of four recipients of the Fields Medal

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d11, d10, d3, d6, d7, d4, d1, d9
- **Claim**: Heated gemstones are commonly treated to improve their color and clarity their value depends on the type of treatment and the quality of the gemstone

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d3, d6, d7, d4, d1, d9
- **Claim**: The documents discuss the concept of innate knowledge, with some arguing for its existence (Plato, Leibniz) and others against it (Locke, empiricists)

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d9
- **Claim**: The evidence suggests that there is a debate about whether we are born with certain knowledge or if all knowledge is acquired through experience

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3, d8
- **Claim**: The number of employees at Mercedes-Benz Group is approximately 166,000, as of 2023

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The minimum legal age for marriage in the United States is 18 without exceptions, as of June 1, 2020

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d4, d8, d7, d1, d9
- **Claim**: Every startup needs a business plan, as it helps them achieve their goals, secure funding communicate their business proposition to stakeholders

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d10, d3, d4, d6, d9
- **Claim**: However, some startups may not need a traditional business plan if they are not seeking external funding

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d10, d4, d7, d6, d1, d9
- **Claim**: Channel 5 was launched on March 30, 1997

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9
- **Claim**: Sitting for prolonged periods can increase the risk of premature death and some chronic diseases, but this risk is significantly less than the risks associated with smoking

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d3, d6, d4, d1
- **Claim**: Smoking increases a person's risk of developing Rheumatoid Arthritis (RA) and can make the disease worse

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d3, d6, d7, d4, d1, d9
- **Claim**: Public transportation is generally slower than driving in cities, but there are solutions like ridesharing and improving infrastructure that can make public transportation more convenient and faster

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d3, d6, d7, d4, d1
- **Claim**: Dogs can understand human words to some extent and can associate them with specific actions or objects

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d3, d1
- **Claim**: They can also understand intonation and distinguish between their native language and a foreign language

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d6, d1
- **Claim**: The most recently awarded Spiel des Jahres is Dorfromantik: The Board Game, which won in 2023

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d6
- **Claim**: The blue-ringed octopus is the world's most venomous marine animal and can paralyze and kill an adult human with a single bite

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d6, d7
- **Claim**: The latest Nobel Prize in Literature was awarded to South Korean author Han Kang in 2024

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d4, d1
- **Claim**: The last person appointed to the U.S. Supreme Court is Ketanji Brown Jackson, who was appointed by President Joe Biden and confirmed on April 7, 2022

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d10, d3, d6, d7, d4, d1, d9
- **Claim**: Shoshana Zuboff has written several books, with "The Age of Surveillance Capitalism" being one of her most recent and well-known works

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Spider-Man originally had mechanical web shooters in the comics, but in some film adaptations, such as the Sam Raimi's trilogy, he had organic web shooters

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The passages do not provide a clear answer to the question of when the Declaration of Independence was signed by all 56 delegates

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d1, d9
- **Claim**: While some passages mention specific dates, such as July 4 for the adoption and August 2 for John Hancock's signature, they do not confirm that all delegates signed on those dates

### Sample #0381

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The rough estimate of the total number of people who have ever lived on Earth is 117 billion

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1, d9
- **Claim**: Based on the provided passages, the current top-ranked team at the FIBA Men's World Ranking is the USA

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d3, d8, d7, d1
- **Claim**: The host stadium of this year's Super Bowl is the Caesars Superdome in New Orleans, Louisiana

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3
- **Claim**: Super Bowl LVIX will take place on February 9, 2025

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d10, d3, d8, d7, d1, d9
- **Claim**: Future Super Bowl locations include Levi's Stadium in Santa Clara, California (Super Bowl LX in 2026), SoFi Stadium in Los Angeles, California (Super Bowl LXI in 2027) Mercedes-Benz Stadium in Atlanta, Georgia (Super Bowl LXII in 2028)

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d4, d6, d7
- **Claim**: A well-planned vegan diet can be safe and beneficial during pregnancy, as long as nutritional requirements are met

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: It may offer benefits such as reduced risk of preeclampsia and gestational diabetes

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d5, d3, d6, d7, d4, d1, d9
- **Claim**: Champagne is a sparkling wine produced solely in the Champagne region of France, using specific grape varieties and following a labor-intensive production process that includes a secondary fermentation in the bottle

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1
- **Claim**: The Word of Wisdom became a commandment for all Church members in 1851, as proposed by Brigham Young and accepted unanimously in a general conference

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: They are unmanned, untethered vehicles designed to operate underwater without human control

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The passages suggest that club soda can be effective at removing stains, particularly beer and red wine stains, but the study conducted by the Dry-cleaning & Laundry Institute (DLI) does not provide conclusive evidence on its effectiveness

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, they do not directly answer the question about why bath towels seem to get more absorbent the more they get washed over time

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The answer to this question may require additional information or an explanation of the specific mechanism behind towel absorbency

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d1
- **Claim**: America's Got Talent season 11 premiered on May 31, 2016

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The question asks why our brains register reflective surfaces like metal as silver in color when they're actually reflecting all of the colors around them

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The provided passages do not directly answer this question, but they do explain that the colors we perceive are dependent on the absorption and re-emission of light that metals like silver, gold copper have specific reflectivity properties that can influence the colors we perceive

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the question asks for a specific explanation about why our brains perceive reflective surfaces as silver the passages do not provide this information

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The given documents do not provide a detailed comparison between socialism and communism

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The given documents do not provide any information about the jazz classic "I Got Rhythm" being written by AC/DC or any connection between the two

### Sample #0561

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The given documents do not provide information about Jamie Oliver's involvement in any television series

### Sample #0588

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Night vision devices often produce a green image due to the sensitivity of the human eye's rods and cones to the yellow-green light spectrum

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is because rods are more sensitive to the blue-green section of the light spectrum, while cones are more sensitive to the yellow-green section

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: At the 2014 Commonwealth Games, England won the most medals, with a total of 27 including 13 silver medals

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Based on the provided passages, the exact number of NCAA basketball championships won by UCLA is not explicitly stated

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, we can infer that UCLA won at least four championships between 1964 and 1975

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d1
- **Claim**: The passages provide information on how bookmakers calculate odds, based on probability and percentages how they aim to achieve a balanced book by having bettors compete against each other

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: They also explain that odds are not always an accurate reflection of the actual result and provide an equation for calculating margins on a two-outcome market

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, they do not directly answer the question about where Dorothy lived

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The given documents do not provide specific information about the books written by Mordecai Richler

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: The given passages do not provide a clear explanation of how salt softens water

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passages do not directly answer the question about an accurate definition of Gravity that can be easily explained

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, they do provide some insights into different theories and perspectives on gravity, such as the possibility of explaining it through motion, data gravity as an analogy, the process of developing scientific theories modified gravity theories


================================================================================

*Report generated by CATS v2.0*
