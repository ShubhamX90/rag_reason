# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 1 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.714 (over 49 samples)

**GR F1** *(used in CATS)*: 0.829

**Behavior Adherence**: 0.688 (over 48 applicable samples)

**Factual Grounding**: 0.106 (over 48 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.612

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.829
- **Precision**: 0.708
- **Recall**: 1.000
- **Accuracy**: 0.714
- TP=34, FP=14, FN=0, TN=1


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.774
- **Behavior**: 0.842 (n=19)
- **Grounding**: 0.018 (n=19)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.617

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.857 (n=14)
- **Grounding**: 0.196 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.594

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.000 (n=10)
- **Grounding**: 0.100 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.349

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 0.200 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.750


================================================================================

## Cost Summary

**Total Cost**: $0.0490

**Decisions Made**: 132

**Average Cost per Decision**: $0.000371


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 132
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0490
- **Total Requests**: 132
- **Average Cost per Request**: $0.000371


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d1, d3, d2
- **Claim**: The winner of American Idol this year is Abi Carter []

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d6, d7, d1, d5, d3, d2
- **Claim**: Children can learn language skills from television, but it is essential to use it in moderation and with parental guidance

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d7, d1, d3, d2
- **Claim**: Well-designed television programs, such as Sesame Street and Dora the Explorer, can be beneficial for child language skills, especially when they are accompanied by adult interaction and co-viewing

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8
- **Claim**: Additionally, watching TV with subtitles can improve children's reading skills and literacy

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d7, d6, d1, d11
- **Claim**: However, excessive screen time can displace other forms of language development, such as conversation and play can have negative effects on children's language skills

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d7, d6, d1, d2, d12
- **Claim**: Therefore, parents should engage in back-and-forth conversation with their children and encourage them to participate in real-time social interactions to promote language development

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d10, d8, d6, d9, d1, d5, d3, d2
- **Claim**: The artist who sings "Gone gone gone, she been gone so long" is Chilliwack []

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d6, d7, d9, d1, d5, d3, d2
- **Claim**: The PIN code system was introduced in India on August 15, 1972 []

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d10, d8, d6, d7, d9, d1, d5, d3, d2
- **Claim**: The most recent woman to become President of Peru is Dina Boluarte []

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9, d8, d5
- **Claim**: The current ATP top-ranked men's singles tennis player is Jannik Sinner []

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d7, d1, d3, d2
- **Claim**: The East India Company was founded in 1600 []

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d6
- **Claim**: The final answer is: Stress does not directly cause stomach ulcers, but it can contribute to the development of ulcers by impairing the body's ability to heal itself or increasing stomach acidity []

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The final answer is:
At least 16 people from Princeton University have won the Fields Medal, including June Huh, William P. Thurston 14 other individuals mentioned in d1 and d5 []

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d6, d7, d1, d3, d2
- **Claim**: The final answer is: Heated gemstones are generally less valuable than unheated ones, especially for high-quality stones []

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3, d2
- **Claim**: We have innate knowledge []

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d3, d2
- **Claim**: The final answer is: Mercedes-Benz has approximately 166,000 employees []

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The minimum age for marriage in the United States is 18, with no exceptions []

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d7, d1, d3, d5
- **Claim**: Every startup needs a business plan, as it serves as a roadmap for achieving goals, helps with fundraising provides a clear direction for the company

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d3
- **Claim**: However, some startups may benefit from a lean start-up approach, which involves testing and revising hypotheses, gathering customer feedback rapidly iterating on and reengineering products

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: Additionally, a pitch deck may be more effective in fundraising than a traditional business plan. []

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d10, d7, d6, d9, d1, d2
- **Claim**: The final answer is:
Channel 5 launched on 30 March 1997 []

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d10, d8, d6, d7, d9, d1, d11, d3, d5
- **Claim**: The final answer is:
Sitting is not as bad as smoking, but excessive sitting can still have negative health consequences. []

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d5, d3, d2
- **Claim**: The final answer is: Smoking is a risk factor for developing rheumatoid arthritis []

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d7, d9, d1, d3, d2
- **Claim**: Public transportation is generally slower than driving in cities, but it can be faster with better infrastructure and services []

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d6, d7, d1, d5, d3, d2
- **Claim**: Dogs can understand human language to some extent, with the ability to learn to associate certain words with specific actions or objects to understand the meaning of certain words and intonation []

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9
- **Supporting Docs Found**: d4, d8, d6, d7, d1, d3, d5
- **Claim**: While dogs may not understand human language in the same way as humans, they are capable of learning to recognize and respond to certain words and phrases

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: The 2024 Spiel des Jahres winner is Sky Team [d3, d5]

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d7, d6, d1, d3
- **Claim**: The final answer is: All octopuses are venomous []

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d10, d9, d1, d13, d11, d14, d3, d5, d12
- **Claim**: The final answer is: Melting sea ice does not contribute to sea level rise []

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d9, d1, d13, d11, d14, d3, d12
- **Claim**: However, the rate of sea level rise is expected to increase due to the melting of glaciers and ice sheets on land [d8, d14]

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d7, d5, d2
- **Claim**: The winner of the 2024 Nobel Prize in Literature is Han Kang []

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3, d5
- **Claim**: The most recent person appointed to the U.S. Supreme Court is Ketanji Brown Jackson [d3, d5]

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d5
- **Claim**: The final answer is: Shoshana Zuboff has published at least 4 books []

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3, d5
- **Claim**: The final answer is: Spider-Man originally had organic web shooters in some versions []

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The web shooters were later replaced with mechanical ones in some versions

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d7, d9, d1, d5, d2
- **Claim**: The 56 delegates to the Continental Congress who signed the Declaration of Independence []

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9, d1, d5
- **Claim**: The country currently at the top rank at the FIBA Men's World Ranking is the USA []

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d6, d7, d1, d3
- **Claim**: The host stadium of this year's Super Bowl is Caesars Superdome []

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d6, d5, d2
- **Claim**: A well-planned vegan diet can be safe for pregnant women, as long as they ensure adequate nutrient intake, including proteins, amino acids, iron, calcium, vitamins D and B12 []

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d6, d7, d9, d1, d5, d3, d2
- **Claim**: The final answer is:
Champagne comes solely from the Champagne region of France []

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d6
- **Claim**: The Word of Wisdom became a requirement for temple recommends in 1919 or 1921 []

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2
- **Claim**: The final answer is: AUV stands for Asian Utility Vehicle in the context of cars and Autonomous Underwater Vehicle in the context of underwater exploration []

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: The final answer is that club soda can be effective in removing certain stains []

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bath towels seem to get more absorbent the more they get washed over time [d3, d4]

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5, d2
- **Claim**: The season of America's Got Talent premiered on May 31, 2016 []

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The brain registers reflective surfaces as silver because of the high reflectivity of silver and the way it interacts with light, which is a trick of the eye convincing the brain to see reflections that aren't actually there []

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The major differences between socialism and communism are not explicitly stated in the provided documents

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: However, based on the information in d3, which discusses the differences between Marxism, Socialism Communism, we can infer that socialism is often seen as a step towards communism, but the exact differences between the two ideologies are not clearly defined

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Socialism typically involves a mixed economy with a strong public sector, while communism aims for a classless, stateless society

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact nature of these differences is not explicitly stated in the provided documents [d3, d4]

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The composer of the jazz classic "I Got Rhythm" is George Gershwin

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The UCLA Bruins have won at least 10 NCAA basketball championships []

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3, d2
- **Claim**: Bookmakers calculate odds in play by using probability-based calculations and aiming to achieve a balanced book, where they try to have an equal number of people betting on the happening and non-happening of an event []

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Emerald City in the Land of Oz []

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Mordecai Richler wrote essays and a book, including "Oh Canada!

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: He was also the stepfather of Daniel Richler

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: The salt is used to soften water by replacing minerals in the water through a process called ion exchange, which is done using a resin that collects the minerals and is regularly regenerated by flushing salt through it [d2, d3]

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: This process is used in salt-based water softeners [d2, d4]


================================================================================

*Report generated by CATS v2.0*
