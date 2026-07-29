# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 1 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.653 (over 49 samples)

**GR F1** *(used in CATS)*: 0.785

**Behavior Adherence**: 0.521 (over 48 applicable samples)

**Factual Grounding**: 0.446 (over 48 applicable samples)

**Single-Truth Recall**: 0.853 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.651

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.785
- **Precision**: 0.689
- **Recall**: 0.912
- **Accuracy**: 0.653
- TP=31, FP=14, FN=3, TN=1


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.774
- **Behavior**: 0.579 (n=19)
- **Grounding**: 0.500 (n=19)
- **Recall**: 0.958 (n=12)
- **CATS**: 0.703

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.667
- **Behavior**: 0.571 (n=14)
- **Grounding**: 0.518 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.585

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.700
- **GR F1** *(used in CATS)*: 0.824
- **Behavior**: 0.300 (n=10)
- **Grounding**: 0.317 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.480

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.300 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.625


================================================================================

## Cost Summary

**Total Cost**: $0.0596

**Decisions Made**: 144

**Average Cost per Decision**: $0.000414


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 144
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0596
- **Total Requests**: 144
- **Average Cost per Request**: $0.000414


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d7, d5, d4, d1, d2, d8
- **Claim**: Children learn language skills from television to some extent, but the evidence is mixed

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d10, d11, d7, d2, d4, d12
- **Claim**: Some documents suggest that educational television can help language development, particularly for preschoolers, while others find no effect or even negative consequences

### Sample #0104

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved documents provide conflicting information about the current ATP top-ranked men's singles tennis player

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While some documents () state that Jannik Sinner is the current top-ranked player, another document states that Novak Djokovic is the current top-ranked player, but the information in d7 is older than the other documents

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9, d5, d8
- **Claim**: The most recent and credible evidence points to Jannik Sinner as the current top-ranked player

### Sample #0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, it is important to note that the information in d7 may still be relevant as it was the most recent available information at the time it was published

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Stomach ulcers are primarily caused by bacterial infection (H. pylori) and the use of nonsteroidal anti-inflammatory drugs (NSAIDs)

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d9, d5, d4, d1, d8
- **Claim**: In the United States, the legal age for marriage varies by state, with some states setting the age at 16, 17 18, while others have no minimum age with parental or judicial consent

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d5, d7, d1
- **Claim**: Some states have raised the legal age for marriage to 18, but the query asks for a general answer the documents do not provide a single definitive answer to the query

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d9, d10, d5, d11, d4, d1, d8
- **Claim**: Startups may or may not need a business plan, as some documents argue for their necessity while others suggest they are not always required

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: 1947, but this is a different channel and not the one being queried

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d7, d5, d4, d1, d2, d8
- **Claim**: - d3: A study found that dogs process words with the left hemisphere and intonation with the right hemisphere, just like humans, indicating they understand human language. [source_quality: high]
- d8: Scientists found that dogs can understand human speech, processing word meaning and intonation using brain mechanisms similar to humans. [source_quality: low]
- d6: Dogs can understand what humans say to a certain extent, with an average vocabulary of up to 165 words, similar to a 2-3 year old child. [source_quality: low]
- d7: Scientists have proven that some dogs can understand spoken language, with studies showing border collies understanding hundreds of object names. [source_quality: low]

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d6, d8
- **Claim**: Dorfromantik: The Board Game won the Spiel des Jahres award most recently in 2023, as confirmed by multiple documents

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7, d1, d6, d8
- **Supporting Docs Found**: d3, d5
- **Claim**: However, some documents contain outdated information, citing Sky Team (2024) as the most recent winner

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d7, d5, d4, d1, d8
- **Claim**: Most octopuses are venomous, as all octopuses possess venom ()

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8
- **Claim**: The venom in octopuses is primarily used for self-defense and hunting

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d6
- **Claim**: While not all octopuses are deadly to humans, the blue-ringed octopus is known to be particularly dangerous (d1, d6)

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Ketanji Brown Jackson was the most recent person appointed to the U.S. Supreme Court, confirmed on April 7, 2022

### Sample #0334

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d6, d2
- **Claim**: The Declaration was signed by 56 delegates in total

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d9, d7, d4, d1, d2, d8
- **Claim**: 8 billion people are currently living on Earth, according to the most recent estimates from the United Nations ()

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d8
- **Claim**: The world population reached 8 billion on November 15, 2022, according to the U.S. Census Bureau

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d8
- **Claim**: However, an older estimate from mid-November 2022 suggests the world population exceeded eight billion

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The US Census Bureau estimated the global population as of September 2022 was approximately 7,922,312,800 people and was expected to reach 8 billion

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: UNICEF Data addresses the question of how many people are in the world, drawing on data from 236 countries/areas using WPP2022, but does not provide a specific population figure in the snippet

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d9, d7, d5, d4, d1, d2, d8
- **Claim**: Champagne comes solely from the Champagne region of France, as all retrieved documents agree on this point

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Toyota Innova and Crosswind are examples of Asian Utility Vehicles (AUVs)

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: AUVs are vehicles predominantly sold in Asia, designed for use in the Asian market as a vehicle that could seat 8-10 people, could haul goods and be used for commercial purposes

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: The term Asian Utility Vehicle is just a coined term, as AUVs are predominantly sold in Asia

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: Club soda is often used to remove stains, as illustrated by a flight attendant using it on a tie in one of the documents

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The documents agree that club soda can be effective, but they do not provide a comprehensive explanation of why it works so well

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For a more detailed explanation, further research may be necessary

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bath towels may appear to get more absorbent over time due to the fibers in cotton breaking down during washing, creating more spaces for water to be absorbed

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d4, d1, d2
- **Claim**: When does American's Got Talent come on?

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The brain perceives reflective surfaces as silver due to the selective reflection of certain wavelengths, but the exact mechanism remains unclear

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, it is important to note that there are various interpretations and nuances of both socialism and communism the specific policies and practices implemented under these ideologies can vary greatly depending on the country and historical context

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The jazz classic "I Got Rhythm" was written by George Gershwin

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Night vision devices display a green image due to a combination of factors, including rod/cone sensitivity and energy efficiency, as suggested by the documents

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d1
- **Supporting Docs Found**: None
- **Claim**: However, they do not provide a definitive explanation for the green color

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: To answer this question, one would need additional information or sources that specifically discuss the real-time calculation process for in-play odds

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: <1-2 concise sentences explaining how the cited evidence yields the final answer why the evidence requires abstention.>
The provided documents do not directly address Dorothy's home location in the original 'Wizard of Oz' story

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Mordecai Richler wrote at least one book, 'Oh Canada!

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The retrieved documents do not provide a comprehensive list of his works

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Salt softens water by replacing minerals in the water with sodium ions during the ion-exchange process

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This process is carried out in a salt-based water softener, which uses a resin to collect minerals and salt to clean the resin

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2
- **Claim**: However, the provided documents do not explain this mechanism in detail

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Gravity is a natural phenomenon by which all material bodies attract each other

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: While there are conflicting opinions on how to explain gravity, such as through motion and bombardment, a simple and accurate definition can be derived from the documents provided


================================================================================

*Report generated by CATS v2.0*
