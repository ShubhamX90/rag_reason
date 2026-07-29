# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 2 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.714 (over 49 samples)

**GR F1** *(used in CATS)*: 0.825

**Behavior Adherence**: 0.383 (over 47 applicable samples)

**Factual Grounding**: 0.090 (over 47 applicable samples)

**Single-Truth Recall**: 0.765 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.516

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.825
- **Precision**: 0.717
- **Recall**: 0.971
- **Accuracy**: 0.714
- TP=33, FP=13, FN=1, TN=2


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.684
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.667 (n=18)
- **Grounding**: 0.125 (n=18)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.627

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.143 (n=14)
- **Grounding**: 0.071 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.314

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.200 (n=10)
- **Grounding**: 0.000 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.382

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.400 (n=5)
- **Grounding**: 0.200 (n=5)
- **Recall**: 0.400 (n=5)
- **CATS**: 0.472


================================================================================

## Cost Summary

**Total Cost**: $0.0442

**Decisions Made**: 121

**Average Cost per Decision**: $0.000365


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 121
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0442
- **Total Requests**: 121
- **Average Cost per Request**: $0.000365


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d7, d1, d9, d3
- **Claim**: The winner of American Idol this year is Abi Carter

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d1, d7
- **Claim**: Overall, the retrieved documents suggest that the relationship between television and language development is complex and influenced by various factors, including the type of content, adult interaction the child's age

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9
- **Claim**: The song that matches the query "gone gone gone she been gone so long" is "My Girl (Gone, Gone, Gone)" by Chilliwack

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d7, d1, d8, d9, d4, d3, d6
- **Claim**: The PIN code system was introduced in India on August 15, 1972

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d7, d10, d1, d8, d9, d4, d3, d6
- **Claim**: The most recent woman to become President of Peru is Dina Boluarte

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d3, d1
- **Claim**: The current ATP top-ranked men's singles tennis player is not explicitly stated in the retrieved documents

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: However, the most recent information available is from 2023, which states that Novak Djokovic was the number one ranked player as of November 20, 2023

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d7, d1, d8, d3, d6
- **Claim**: The East India Company was founded in 1600

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1, d4, d3, d6
- **Claim**: Therefore, based on the retrieved documents, the answer to the query is that there is conflicting evidence on whether stress can cause stomach ulcers some documents suggest that stress may contribute to the development of stomach ulcers, while others state that stress alone does not cause stomach ulcers

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d8, d9
- **Claim**: June Huh (2022)
2

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Akshay Venkatesh (2018)
3

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Manjul Bhargava is not mentioned as a Princeton University alumnus, but rather as an alumnus of the Institute for Advanced Study at Princeton

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d10, d8, d3
- **Claim**: However, it's worth noting that some documents also mention that the value difference between heated and unheated gemstones can be relatively small that some heated gemstones can still be valuable and desirable

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d6, d9
- **Claim**: <CONFLICT_LABEL>Conflicting opinions or research outcomes</CONFLICT_LABEL>

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The legal age for marriage in Massachusetts is 18, with no exceptions

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d11, d10, d9
- **Claim**: Therefore, the answer to the question is that every startup does not necessarily need a business plan, but rather a well-thought-out plan that suits their specific needs and goals

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d10, d1, d9, d4, d6
- **Claim**: The channel 5 in the United Kingdom started broadcasting on March 30th, 1997

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Therefore, the answer to the query is CONFLICTING OPINIONS, as there is no clear consensus on whether sitting is as bad for your health as smoking

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d1, d4, d3, d6
- **Claim**: Therefore, based on the retrieved documents, the answer to the question "Can smoking cause Rheumatoid Arthritis?" is that smoking is associated with an increased risk of developing RA, but it is not a direct cause of the disease

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d7, d1, d8, d9, d4, d3, d6
- **Claim**: Overall, the majority of the documents suggest that public transportation is generally slower than driving in cities, but there are some exceptions and potential solutions to improve public transportation efficiency

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d7, d1, d8, d4, d3, d6
- **Claim**: Overall, the evidence suggests that dogs can understand human language to some extent, but the extent to which they understand is still a topic of debate

### Sample #0320

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: There is no document that contradicts the statement that most octopuses are venomous

### Sample #0322

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict label indicates that there are conflicting opinions or research outcomes on this topic, which is consistent with the retrieved documents

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d7, d8, d6
- **Claim**: The winner of the latest Nobel Prize in Literature is Han Kang

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3, d1
- **Claim**: The last person appointed to the U.S. Supreme Court is Ketanji Brown Jackson

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d7, d9
- **Claim**: The number of books Shoshana Zuboff has published is 3

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1
- **Claim**: Therefore, based on the retrieved documents, the answer to the question is that Spiderman originally did not have organic web shooters

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d7
- **Claim**: The signers of the Declaration of Independence were 56 delegates to the Continental Congress

### Sample #0381

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Since the documents provide conflicting information, I am unable to provide a definitive answer to the question of how many people are in the world

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1, d9
- **Claim**: The country currently at the top rank at the FIBA Men's World Ranking is the USA

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: The host stadium of this year's Super Bowl is Allegiant Stadium in Las Vegas, Nevada

### Sample #0399

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer to the query is marked as "Conflicting opinions or research outcomes" due to the mixed and inconclusive evidence presented in the retrieved documents

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d7
- **Claim**: The Word of Wisdom became a commandment in 1851, when President Brigham Young proposed to the general conference of the Church that all Saints formally covenant to keep the Word of Wisdom the proposal was unanimously upheld by the membership of the Church

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The meaning of AUV in the context of cars is "Asian Utility Vehicle." It refers to a type of vehicle that is designed for use in the Asian market, typically seating 8-10 people capable of hauling goods and being used for commercial purposes

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Club soda works well at getting stains out because it is a mild acid that helps break down the stain, making it easier to remove

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The retrieved documents suggest that bath towels, particularly those made of cotton, become more absorbent over time as they are washed

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1
- **Claim**: The show "America's Got Talent" premiered on May 31, 2016

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Overall, the retrieved documents do not provide a direct answer to the question, but they do provide some relevant information about colour perception and reflection

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The jazz classic "I Got Rhythm" was written by George Gershwin

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer to the question "why is night vision green?" cannot be definitively answered based on the provided documents

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: There may be additional championships not mentioned in the provided documents

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3, d1
- **Claim**: In terms of how quickly bookmakers can adjust their odds, it's often done in real-time, based on the number of bets placed and the likelihood of the event occurring

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This can happen very quickly, often in a matter of seconds or minutes, depending on the event and the bookmaker's systems

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: From the documents, we can infer that Dorothy lives in the Land of Oz, specifically in the Emerald City, which is the capital of Oz

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This is mentioned in

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: * "The Nose From Jupiter" (1998)
* "A Nose For Adventure"
* "Noses Are Red"
* "The Boy From Earth"
* "The Way to Schenectady" (1998)
* "Of Mice and Nutcrackers" (2001)
* "Mystical Rose" (2000)
* "Oh Canada!

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Salt is used in water softening systems to replace the minerals that are removed from the fluid

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: In a salt ion exchange system, salt is used to regenerate the resin, which collects the minerals from the water

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This process involves flushing salt through the resin to clean it, allowing the softener to provide soft water

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Therefore, while there is a general consensus on the definition of gravity, there are ongoing debates and alternative theories that may not be widely accepted


================================================================================

*Report generated by CATS v2.0*
