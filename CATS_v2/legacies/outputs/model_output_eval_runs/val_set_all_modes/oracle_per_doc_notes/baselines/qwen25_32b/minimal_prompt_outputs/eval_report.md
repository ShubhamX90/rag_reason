# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 2 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.735 (over 49 samples)

**GR F1** *(used in CATS)*: 0.840

**Behavior Adherence**: 0.574 (over 47 applicable samples)

**Factual Grounding**: 0.094 (over 47 applicable samples)

**Single-Truth Recall**: 0.794 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.575

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.840
- **Precision**: 0.723
- **Recall**: 1.000
- **Accuracy**: 0.735
- TP=34, FP=13, FN=0, TN=2


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.774
- **Behavior**: 0.684 (n=19)
- **Grounding**: 0.063 (n=19)
- **Recall**: 0.875 (n=12)
- **CATS**: 0.599

### Type 2: Complementary Info

- **Samples**: 15 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.667
- **GR F1** *(used in CATS)*: 0.762
- **Behavior**: 0.462 (n=13)
- **Grounding**: 0.110 (n=13)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.445

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.400 (n=10)
- **Grounding**: 0.067 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.471

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.220 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.655


================================================================================

## Cost Summary

**Total Cost**: $0.0579

**Decisions Made**: 145

**Average Cost per Decision**: $0.000400


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 145
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0579
- **Total Requests**: 145
- **Average Cost per Request**: $0.000400


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d9, d7
- **Claim**: Therefore, the winner of American Idol this year is Abi Carter

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d11, d5, d6, d1, d2, d10, d9, d4, d12, d7, d8
- **Claim**: In conclusion, while television can potentially aid in language learning under specific circumstances, such as when paired with adult interaction and high-quality educational content, it is not a definitive or universally beneficial tool for language skill development in children

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d2, d10, d12, d7
- **Claim**: The overall consensus leans towards the importance of direct human interaction and social engagement over passive screen time for optimal language development

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d10, d9, d4, d8
- **Claim**: Therefore, the answer to the query "Who sings 'gone gone gone she been gone so long?'" is Chilliwack

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d10, d9, d4, d7, d8
- **Claim**: Based on the retrieved documents, the most recent woman to become President of Peru is Dina Boluarte

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d10, d9, d4, d7, d8
- **Claim**: She became the first female president in Peru's history when she was sworn in on December 7, 2022, after her predecessor, Pedro Castillo, was impeached by Congress

### Sample #0103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Each document explicitly identifies Dina Boluarte as the first and most recent female president of Peru

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d8, d4
- **Claim**: Therefore, the most up-to-date answer is that Jannik Sinner is the current ATP top-ranked men's singles tennis player

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d7, d8
- **Claim**: Therefore, the East India Company came into existence on December 31, 1600

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d4
- **Claim**: In summary, although stress does not directly cause stomach ulcers, it can act as a contributing factor in their development and worsening

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Therefore, the answer to the query is that 15 people from Princeton University have won the Fields Medal

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d10, d4, d7, d8
- **Claim**: In conclusion, while heated gemstones can be less valuable than their unheated counterparts, this is not universally true for all gemstones

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d4, d7, d8
- **Claim**: The value difference depends on the specific type of gemstone and the extent of the treatment

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2
- **Claim**: Given the mixed perspectives presented, the documents suggest that the existence of innate knowledge remains a topic of debate among philosophers and researchers

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d9, d4, d7
- **Claim**: Given the variability across jurisdictions, the legal age for marriage depends on the specific location

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d9, d4, d7, d8
- **Claim**: For a definitive answer, one would need to check the laws of the particular state or country in question

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d11, d6, d1, d10, d9, d4, d7
- **Claim**: Given the mixed responses, the consensus seems to be that while a business plan is highly recommended and often necessary for certain purposes (like securing funding), it is not an absolute requirement for every startup

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d11, d6, d1, d10, d9, d4, d7
- **Claim**: Therefore, the answer is that not every startup needs a business plan, although it can be a valuable tool under specific circumstances

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d11, d10
- **Claim**: In conclusion, while sitting can pose health risks, the evidence presented in the documents does not support the claim that sitting is equivalent to smoking in terms of health hazards

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d4
- **Claim**: Therefore, the answer to the query is yes, smoking can cause an increased risk of Rheumatoid Arthritis

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d7
- **Claim**: Based on the retrieved documents, public transportation is generally not faster than driving in cities

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d7
- **Claim**: Overall, the evidence strongly suggests that driving is faster than public transportation in most cities

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d4, d7, d8
- **Claim**: In conclusion, while dogs do not fully understand human language in the same way humans do, they can understand specific words and tones their brains process language similarly to humans

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Citations:
- doc_id: d3, Source URL: https://boardgamegumbo.wordpress.com/2024/07/22/spiel-des-jahres-winners-2024/
- doc_id: d5, Source URL: https://boardgoats.org/2024/07/21/spiel-des-jahres-winners-2024/

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d1, d4, d7, d8
- **Claim**: Therefore, the documents collectively provide strong support that all octopuses are venomous, which is a stronger statement than the query asking if most are venomous

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d10, d13, d14, d9, d4, d12, d11
- **Claim**: In summary, while the majority of the documents indicate that melting sea ice does not significantly contribute to sea level rise, there are notable exceptions that suggest a minor contribution due to the density differences between freshwater and saltwater

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d2, d7, d8
- **Claim**: Therefore, Han Kang is the winner of the latest Nobel Prize in Literature

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3, d4
- **Claim**: Based on the retrieved documents, the last person appointed to the U.S. Supreme Court was Ketanji Brown Jackson

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3, d4
- **Claim**: Therefore, the last person appointed to the U.S. Supreme Court is Ketanji Brown Jackson

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: Given the discrepancy in the number of books across the sources, the exact number of books published by Shoshana Zuboff cannot be definitively determined from the provided documents alone

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d7, d9
- **Claim**: However, the most consistent answer based on multiple sources is that she has published three major books

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d2, d4
- **Claim**: Therefore, the documents collectively support the conclusion that Spiderman originally did not have organic web shooters in the comics this feature was introduced later in film adaptations

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d9, d4, d7, d8
- **Claim**: Therefore, the current world population is approximately 8 billion people

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d9, d4
- **Claim**: Therefore, the USA is currently at the top rank in the FIBA Men's World Ranking

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: Therefore, the host stadium for this year's Super Bowl is Allegiant Stadium

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d1, d2, d4, d7
- **Claim**: In conclusion, while a vegan diet can be safe and potentially beneficial during pregnancy, it necessitates meticulous planning and supplementation to ensure adequate intake of essential nutrients such as proteins, amino acids, iron, calcium vitamins D and B12

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6
- **Claim**: Therefore, pregnant women considering a vegan diet should consult with healthcare professionals to ensure they and their babies remain healthy

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d1, d2, d9, d4, d7, d8
- **Claim**: In summary, Champagne, when referring to the legally protected and traditionally recognized sparkling wine, comes solely from the Champagne region of France

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d9, d4, d7, d8
- **Claim**: Therefore, the Word of Wisdom transitioned into a mandatory commandment in two significant phases: first in 1851 as a formal covenant and later in 1919 as a requirement for temple worthiness

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d2
- **Claim**: Sources:
- [AutoDeal](https://www.autodeal.com.ph/articles/car-features/auv-mpv-or-suv-what-are-differences)
- [Vigattin Insurance](https://www.vigattininsurance.com/news-and-promos/Knowing-Your-Wheels-Differentiating-SUV-AUV-and-MPV/81)
- [Tsikot.com](https://www.tsikot.com/forums/pick-ups-trucks-buses-other-vehicles-talk-7/what-does-auv-xuv-mean-1133/index2.html)

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: In summary, while the documents confirm that club soda is effective for stain removal, they do not provide the scientific reasoning behind its efficacy

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Therefore, the documents are insufficient to fully answer the query regarding why club soda works so well for getting stains out

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Based on the retrieved documents, the information provided is insufficient to fully explain why bath towels seem to get more absorbent the more they get washed over time

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Therefore, while there is partial support for the idea that certain types of towels can become more absorbent with repeated washing, the underlying reason remains unexplained by the given documents

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the information available, the core question about why reflective surfaces appear silver rather than a mix of reflected colours remains unanswered

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide enough detail to fully elaborate on all the major differences between socialism and communism

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Based on the retrieved documents and the provided per-document notes, there is no information available regarding who wrote the jazz classic "I Got Rhythm." All the documents are deemed irrelevant to the query none provide details about the composer of the song in question

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Based on the retrieved documents, Jamie Oliver is a member of the cast of the documentary series "Jamie's School Dinners," which was broadcast on Channel 4 in the United Kingdom from 23 February to 16 March 2005

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Therefore, the definitive answer based on the provided documents is that Jamie Oliver is a cast member of "Jamie's School Dinners."

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: From these points, we can infer that the green color in night vision devices may be chosen because it aligns with the sensitivity of the human eye's rods and cones, particularly in low-light conditions

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Additionally, green might be selected because it is easier on the eyes and can help preserve night vision

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, none of the documents provide a definitive explanation for why green is specifically used in night vision technology

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Based on the retrieved documents, Kenya and Jamaica each won 10 gold medals at a particular meet, which could potentially refer to the 2014 Commonwealth Games, although the exact event is not explicitly confirmed in the snippets provided

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Therefore, while we can infer that Kenya and Jamaica likely tied for the most gold medals, the documents do not provide conclusive evidence to answer the query with certainty

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d4
- **Claim**: Based on the retrieved documents, we can confirm that UCLA has won multiple NCAA basketball championships, including at least the following years: 1964-65, 1967 1971, with the 1974-75 season marking their tenth championship in twelve years

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide the exact total number of NCAA basketball championships UCLA has won overall

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, while we know UCLA has won numerous championships, the specific total count cannot be determined from the given information

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Based on the retrieved documents, there is no direct information about where Dorothy lived in the original "Wizard of Oz" story

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, this does not answer the query regarding her living place in the original story

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Typically, Dorothy is known to live in Kansas, but this detail is not found within the provided documents

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Therefore, the documents do not sufficiently answer the query

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide a comprehensive list of all the books written by Mordecai Richler

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Therefore, while we can confirm that "Oh Canada!

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Oh Quebec!" is one of his books, the available documents are insufficient to list all of his works

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: While the documents partially support this explanation, they do not explicitly describe the ion exchange mechanism

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Therefore, the documents are insufficient to fully answer the query

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This definition captures the essence of gravity as an attractive force between masses, which is straightforward and easy to understand

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide a comprehensive explanation suitable for all educational levels the quality of the sources is low


================================================================================

*Report generated by CATS v2.0*
