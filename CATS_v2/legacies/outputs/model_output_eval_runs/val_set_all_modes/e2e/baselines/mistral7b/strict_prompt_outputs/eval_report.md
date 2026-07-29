# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 2 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.735 (over 49 samples)

**GR F1** *(used in CATS)*: 0.840

**Behavior Adherence**: 0.340 (over 47 applicable samples)

**Factual Grounding**: 0.434 (over 47 applicable samples)

**Single-Truth Recall**: 0.471 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.521

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

- **Samples**: 19 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.684
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.500 (n=18)
- **Grounding**: 0.522 (n=18)
- **Recall**: 0.667 (n=12)
- **CATS**: 0.622

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.357 (n=14)
- **Grounding**: 0.411 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.498

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.200 (n=10)
- **Grounding**: 0.325 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.491

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.000 (n=5)
- **Grounding**: 0.400 (n=5)
- **Recall**: 0.000 (n=5)
- **CATS**: 0.350


================================================================================

## Cost Summary

**Total Cost**: $0.0733

**Decisions Made**: 159

**Average Cost per Decision**: $0.000461


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 159
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0733
- **Total Requests**: 159
- **Average Cost per Request**: $0.000461


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d7, d1, d9
- **Claim**: Abi Carter won American Idol this year, as confirmed by multiple high-credibility sources, including Entertainment Tonight, ABC News E!

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d7, d1, d4
- **Claim**: Children can learn language skills from television, but it is most effective when accompanied by social interactions, appropriate programming parental involvement

### Sample #0102

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The first three digits together indicate a central sorting point and distribution area

### Sample #0102

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The last three digits identify the location of the delivery post office served by that central sorting point

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d6, d1
- **Claim**: The person behind the initiative was Shriram Bhikaji Velankar

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: Novak Djokovic was the ATP top-ranked men's singles tennis player as of 19 November 2023, according to the ATP Tour's official rankings

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d5, d8, d9, d4
- **Claim**: However, the rankings may have changed since then some documents may be outdated

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d6, d1, d7, d8, d9, d5
- **Claim**: When was the East India Company founded?

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d3
- **Claim**: On December 31, 1600

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d6, d1, d7, d5, d4
- **Claim**: Can stress cause stomach ulcers?

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d8, d9, d4
- **Claim**: The snippets collectively establish that June Huh, a Princeton mathematician, was awarded the 2022 Fields Medal

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9
- **Claim**: However, they do not provide a clear answer to the question of how many people from Princeton University have won the Fields Medal

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The documents disagree on the total number of Princeton University alumni who have won the Fields Medal, with some suggesting it is 15 and others implying it is more than 15

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: As a result, we cannot definitively answer the question based on the provided evidence

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d6, d1, d7, d8, d4
- **Claim**: Are heated gemstones less valuable?

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d4
- **Claim**: Do we have innate knowledge?

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d9
- **Claim**: However, some documents also discuss the empiricist view that knowledge is acquired only through sensory experience, which seems to contradict the existence of innate knowledge

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The minimum age for marriage in the United States is 18 without exceptions, as of June 1, 2020

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9, d1, d5
- **Supporting Docs Found**: None
- **Claim**: This is based on the information provided in documents

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d1
- **Claim**: - d1: The mass-media enthusiasm for condemning sitting by making comparisons to smoking has far outpaced the available scientific evidence

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d1, d11, d9, d10
- **Claim**: It is obvious from examination of smoking research that sitting and smoking are distinct behaviors with different levels of associated risk (Table 1).
- d2: Over 25% of American adults sit for more than 8 hours every day

### Sample #0206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: 44% of those people get little to no exercise

### Sample #0206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The average American watches approximately 3 hours of television every day

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The average American is active less than 20 minutes every day.
- d3: James Levine, a professor at the Mayo Clinic, claims that sitting is more dangerous than smoking, kills more people than HIV is more treacherous than parachuting

### Sample #0206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A 2018 study found that 82% of people who suffer from blood clots, sat for a significantly greater amount of time than the remaining 18%.
- d4: The human body is not designed for long periods of sitting

### Sample #0206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Instead, it works best when it spends a significant portion of time throughout the day standing and walking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d7, d10, d8, d5, d4
- **Claim**: These activities keep the blood moving, aid digestion allow the body to metabolize nutrients.
- d5: A growing body of research shows that people who spend many hours of the day glued to a seat die at an earlier age than those who sit less, even if those sitters exercise.
- d6: Not moving for prolonged periods of time can lead to serious health problems like obesity, diabetes, high cholesterol, high blood pressure even heart disease and stroke.
- d7: Physical inactivity is linked to more than 3 million preventable deaths every year globally prolonged sitting constitutes physical inactivity.
- d8: One of the most important ways to avoid work related postural pain with desk jobs is to reduce the amount of time you are continuously in the same position

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d6, d1, d7, d8, d9, d5
- **Claim**: In cities, public transportation can be slower than driving, but it can also be made faster through ridesharing or improved infrastructure

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d6, d1, d7, d8, d5, d4
- **Claim**: Canines can understand human language to some extent, as they can learn the names of objects, associate words with actions distinguish languages

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6, d8
- **Supporting Docs Found**: d2, d4
- **Claim**: However, other documents do not explicitly state the year, so it is not clear if it is the most recent winner

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d14, d3, d13, d1, d11, d12, d8, d9, d4
- **Claim**: Melting land ice contributes to sea level rise because the water eventually runs into the ocean

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d14, d3, d13, d1, d11, d12, d10, d9, d5, d4
- **Claim**: Melting sea ice does not contribute to sea level rise because it has the same volume as the water it displaces when it melts. ()

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d1, d4
- **Claim**: It is possible that Ketanji Brown Jackson is the last person appointed to the U.S. Supreme Court, but the documents do not provide conclusive evidence to confirm this

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1, d7, d10, d8, d9, d5, d4
- **Claim**: How many books has Shoshana Zuboff published?

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d6, d7, d1, d9, d10
- **Claim**: Some sources mention three books, while others do not specify the exact number

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: The documents suggest that Spider-Man's web shooters in the Raimi trilogy are organic, but it is not clear whether this is consistent with the original comics

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d6, d1, d7, d8, d9, d5, d4
- **Claim**: 8,198,260,420 people 10.2 billion people

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d7, d5, d4
- **Claim**: Pregnant women can follow a vegan diet, but it is important to ensure that the diet provides all essential nutrients, particularly protein, iron, calcium, vitamins D and B12

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7, d5
- **Supporting Docs Found**: None
- **Claim**: Pregnant women following a vegan diet should consult with a healthcare provider or a dietitian to ensure they are meeting their nutritional needs

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d8, d9, d5, d4
- **Claim**: Champagne is a sparkling wine that can only come from the Champagne region of France

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d7, d1, d8, d9, d5
- **Claim**: When did the Word of Wisdom become a commandment?

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d4
- **Claim**: The Word of Wisdom was given as a revelation by Joseph Smith in 1833

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d5
- **Claim**: In 1851, it became a commandment for all Church members, as proposed by Brigham Young and accepted unanimously by the membership of the Church

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: In cars, AUVs (Asian Utility Vehicles) are passenger vehicles primarily used for carrying passengers

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide specific information about AUVs in cars

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Why does club soda work so well getting stains out?

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: The documents suggest that club soda can be effective at removing certain stains, such as beer and red wine stains

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3, d1, d5
- **Supporting Docs Found**: None
- **Claim**: (mentions cotton towels but does not discuss their absorbency over time) (discusses microfiber towels and their absorbency compared to regular cotton towels)

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The question seems to be asking for a psychological or neurological explanation, which is not directly addressed in the provided documents

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The documents suggest that George Garland composed and arranged "i got rhythm" before offering it to Miller, but they do not provide conclusive evidence about who wrote the original version of the song

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Some documents suggest that the choice of colors for LED flashlights, such as yellow-green and blue-green, can help preserve night vision. 3

### Sample #0592

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, neither document explicitly states that they won the most gold medals

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d4
- **Claim**: In play odds are calculated by bookmakers based on the probability of an event happening and their focus on achieving a balanced book

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the provided documents do not directly explain how bookmakers calculate odds in play

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: To get a more detailed explanation, we would need to consult additional sources that specifically address the calculation of odds in play

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3, d1, d4
- **Supporting Docs Found**: None
- **Claim**: One possible answer: Dorothy lived somewhere other than the Emerald City before her journey to meet the Wizard of Oz, but the documents do not specify where she lived

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: The mechanism by which salt softens water is not explicitly explained in the provided documents

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: The documents discuss various types of water softeners, their costs installation processes, but they do not provide a clear explanation of how salt softens water

### Sample #0650

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For a detailed explanation, it is recommended to consult a reliable source that specifically addresses the question of how salt softens water

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Gravity is a natural phenomenon by which all material bodies attract each other

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It can also be associated with the ability of data to attract additional applications and services (Data Gravity)

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, these definitions are not simple or easily explained, so a more concise definition may be needed for a clear understanding


================================================================================

*Report generated by CATS v2.0*
