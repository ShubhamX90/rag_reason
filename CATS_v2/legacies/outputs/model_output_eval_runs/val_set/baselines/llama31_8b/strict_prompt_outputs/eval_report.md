# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 3 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.571 (over 49 samples)

**GR F1** *(used in CATS)*: 0.704

**Behavior Adherence**: 0.391 (over 46 applicable samples)

**Factual Grounding**: 0.349 (over 46 applicable samples)

**Single-Truth Recall**: 0.706 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.537

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.704
- **Precision**: 0.676
- **Recall**: 0.735
- **Accuracy**: 0.571
- TP=25, FP=12, FN=9, TN=3


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.759
- **Behavior**: 0.500 (n=18)
- **Grounding**: 0.449 (n=18)
- **Recall**: 0.750 (n=12)
- **CATS**: 0.614

### Type 2: Complementary Info

- **Samples**: 15 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.632
- **Behavior**: 0.462 (n=13)
- **Grounding**: 0.365 (n=13)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.486

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.300
- **GR F1** *(used in CATS)*: 0.462
- **Behavior**: 0.100 (n=10)
- **Grounding**: 0.020 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.194

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.400 (n=5)
- **Grounding**: 0.600 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.650


================================================================================

## Cost Summary

**Total Cost**: $0.0589

**Decisions Made**: 146

**Average Cost per Decision**: $0.000403


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 146
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0589
- **Total Requests**: 145
- **Average Cost per Request**: $0.000406


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d9, d2, d7, d1
- **Claim**: Abi Carter was the winner of American Idol season 22, as stated in multiple reliable sources, including []

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d6, d8, d3, d9, d2, d7, d1
- **Claim**: The PIN code system was introduced in India on 15th August 1972, as consistently reported by all retrieved documents []

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d6, d8, d3, d9, d2, d7, d10, d1
- **Claim**: The retrieved documents provide consistent information about Dina Boluarte being the first female president of Peru, with some documents providing additional details about her background and presidency

### Sample #0127

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d9, d8
- **Claim**: The final answer is:
Based on the evidence from documents , 64 people have been awarded the Fields Medal June Huh, a Princeton University professor, was awarded the 2022 Fields Medal

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Princeton University has been affiliated with 15 Fields Medalists

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3, d8
- **Claim**: The final answer is: Based on the retrieved documents, the number of employees at Mercedes-Benz is around 166,000, with some minor discrepancies

### Sample #0201

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d9, d2, d7, d10, d1
- **Claim**: Channel 5 launched on 30 March 1997 []

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d8, d3, d6, d7, d1
- **Claim**: Dogs can understand human language to a certain extent, including familiar words and their meanings can even learn to associate words with actions or objects

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d9
- **Claim**: However, the extent of their understanding varies they do not understand any particular human language

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d6, d8, d3, d2, d7, d1
- **Claim**: This is supported by documents , which all show that dogs can understand human language to some extent

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: However, document d9 suggests that dogs do not understand human language, but can be influenced by body language

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: The most recent winner of the Spiel des Jahres award is Sky Team

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d8, d2, d7
- **Claim**: Han Kang is the winner of the latest Nobel Prize in Literature, awarded in 2024 []

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d8, d9, d2, d7, d1
- **Claim**: Shoshana Zuboff has published at least 3 books, with some sources citing 4 or 16 books

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: The exact number of books she has published cannot be determined from the retrieved evidence []

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d9, d2, d7, d1
- **Claim**: The final answer is:
56 delegates to the Continental Congress signed the Declaration of Independence, but the date of signing is disputed, with some documents suggesting August 2, 1776 others suggesting July 4, 1776

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1
- **Claim**: The FIBA Men's World Ranking is currently sponsored by Nike

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d9, d10, d1
- **Claim**: The host stadium of this year's Super Bowl is not specified in the retrieved documents, as the query asks for the host stadium of "this year's" Super Bowl, but the documents only provide information about future Super Bowls

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d8
- **Claim**: However, the documents do agree on the host stadium of Super Bowl LXI in 2027, which is SoFi Stadium

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d6, d2, d7
- **Claim**: A well-planned vegan diet is considered safe for pregnant women, with some documents highlighting the importance of careful planning to ensure adequate nutrient intake

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: A well-planned vegan diet may reduce the risk of preeclampsia and gestational diabetes during pregnancy []

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d2, d6, d1
- **Claim**: However, it is essential to note that a vegan diet can be challenging to follow and requires careful monitoring of nutrient intake

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d6
- **Supporting Docs Found**: None
- **Claim**: Pregnant women should consult with a registered dietitian specializing in plant-based diets to ensure they are getting all the necessary nutrients

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d7, d1
- **Claim**: The Word of Wisdom was first proposed as a commandment in 1851 [] it became a requirement for temple recommends in 1921 []

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d6, d8, d9, d2, d7, d1
- **Claim**: The Church of Jesus Christ of Latter-day Saints has a clear timeline of the Word of Wisdom's development and implementation []

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d8, d3, d6, d7
- **Supporting Docs Found**: None
- **Claim**: However, in the context of underwater exploration and research, AUV stands for Autonomous Underwater Vehicle, as stated in

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The term AUV is used in different contexts its meaning depends on the context in which it is used

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The absorbency of bath towels can increase with washing, as stated in d3

### Sample #0509

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The season 11 premiere date is consistently reported across multiple sources []

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact reason why our brains register reflective surfaces as silver is not explicitly stated in the retrieved documents

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: It is possible that the brain is tricked into perceiving reflective surfaces as silver due to the way that light is reflected and perceived, but this is not directly addressed in the provided documents

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: I Got Rhythm was written and recorded by Johnny Cash

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The final answer is: Night vision is green because the image shown through night vision devices is always green, as supported by documents d1 and the collective evidence from the other documents

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is consistent with the idea that the human eye is more sensitive to the yellow-green section of the light spectrum, as mentioned in document d4

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The use of colored LED flashlights, including those that emit yellow-green and blue-green light, is also mentioned in documents d2 and d4 as a way to preserve and protect night vision

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The exact count cannot be determined from the retrieved evidence, but UCLA has won at least 10 championships

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3, d1
- **Claim**: Bookmakers calculate odds based on probability, but they also try to achieve a balanced book to have odds in their favor [d2, d4]

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: This means that the odds are not always based on the actual probability of an event happening

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact method of calculating odds is not explicitly stated in the retrieved documents

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Dorothy lives in the Land of Oz, as stated in and

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Emerald City, where the Wizard of Oz resides, is located in the centre of Oz, as mentioned in

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The Emerald City is also the capital of Oz, as stated in

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The books written by Mordecai Richler are not explicitly listed in the retrieved documents, but one book, "Oh Canada!

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Water softening can be achieved through various methods, including salt-based systems, reverse osmosis systems, magnetic systems salt-free systems [d1, d4]

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Data gravity is an analogy of the nature of data and its ability to attract additional applications and services

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This definition is a clear and easily explained concept that can be understood by a wide range of audiences


================================================================================

*Report generated by CATS v2.0*
