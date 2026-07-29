# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 1 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.714 (over 49 samples)

**GR F1** *(used in CATS)*: 0.829

**Behavior Adherence**: 0.500 (over 48 applicable samples)

**Factual Grounding**: 0.000 (over 48 applicable samples)

**Single-Truth Recall**: 0.765 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.523

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
- **Behavior**: 0.684 (n=19)
- **Grounding**: 0.000 (n=19)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.573

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.357 (n=14)
- **Grounding**: 0.000 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.361

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
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.000 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.600


================================================================================

## Cost Summary

**Total Cost**: $0.0543

**Decisions Made**: 139

**Average Cost per Decision**: $0.000391


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 139
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0543
- **Total Requests**: 139
- **Average Cost per Request**: $0.000391


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d9, d2, d1
- **Claim**: The winner of American Idol this year is Abi Carter

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d6, d8, d7, d1
- **Claim**: In summary, while children can learn language skills from television, the effectiveness of this learning depends on factors such as the type of program, the presence of a competent adult the amount of screen time

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9
- **Claim**: The song "My Girl (Gone, Gone, Gone)" by Chilliwack contains the lyrics "Gone gone gone, she been gone so long."

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d6, d8, d7, d9, d2, d1
- **Claim**: The PIN code system was introduced in India on August 15, 1972

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d6, d8, d7, d10, d9, d2, d1
- **Claim**: The most recent woman to become President of Peru is Dina Boluarte

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d5, d9
- **Claim**: The current ATP top-ranked men's singles tennis player is Jannik Sinner, with 10,330 points

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d6, d8, d7, d2, d1
- **Claim**: The East India Company was founded in 1600

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1
- **Claim**: Therefore, while stress may not be the primary cause of stomach ulcers, it can play a contributing role in their development

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d8, d2, d9
- **Claim**: June Huh (2022)
2

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d6, d8, d7, d2, d1
- **Claim**: Overall, while heated gemstones can still be valuable, they are generally considered to be less valuable than their natural, unheated counterparts

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d8, d7, d9, d2, d1
- **Claim**: Yes, there is evidence to suggest that humans may have innate knowledge, but the concept is still a topic of debate

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d8, d5, d2
- **Claim**: The Mercedes-Benz Group has approximately 166,000 employees, according to multiple sources

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d9, d2, d1
- **Claim**: The legal age for marriage varies by state in the United States

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: 22 states have made marriage before 18 illegal, while another 10 allow it under certain circumstances

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: However, as of June 1, 2020, marriageable age in all states is 18 without exceptions

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d6, d7, d10, d11, d9, d1
- **Claim**: In summary, while a business plan can be a useful tool for startups, it is not a requirement for all startups

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d6, d7, d10, d11, d9, d1
- **Claim**: The decision to create a business plan should be based on the specific needs and goals of the startup, as well as the type of funding or support it is seeking

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d7, d10, d9, d2, d1
- **Claim**: Channel 5 started on March 30th, 1997

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d10, d11, d9, d1
- **Claim**: Therefore, based on the retrieved documents, it appears that the comparison between sitting and smoking is not entirely accurate that sitting is not as bad as smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8
- **Claim**: However, it is also clear that excessive sitting can have negative health effects that regular physical activity can help mitigate these effects

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d6, d2, d1
- **Claim**: Based on the retrieved documents, smoking is associated with an increased risk of developing Rheumatoid Arthritis (RA)

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d6, d1
- **Claim**: The documents suggest that smoking can make the disease worse and that it interacts with other risk factors such as genetics and obesity to increase the risk of developing RA

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: However, document d9 from BI Team mentions that in Mexico City, the bus rapid transit system has decreased CO2 by 35,000 tons annually and reduced travel times for riders by 40 percent, making it faster than cars in some areas

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d6, d8, d7, d2, d1
- **Claim**: In summary, the retrieved documents suggest that dogs can understand human language to some extent, including learning to associate specific words with specific actions or objects, understanding emotional content even processing language in a similar way to humans

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d5, d6, d1
- **Claim**: However, it's also clear that dogs do not understand human language in the same way as humans do that they have their own unique way of communicating

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: The most recent Spiel des Jahres award winner is Sky Team

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d6, d8, d7, d1
- **Claim**: All octopuses are venomous

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d13, d14, d12, d11, d9, d1
- **Claim**: However, this effect is relatively small compared to the contribution of melting land ice to sea level rise

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d8, d13, d14, d12, d9, d1
- **Claim**: The melting of glaciers and ice sheets on land is a major contributor to sea level rise is expected to continue to increase in the future due to climate change

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d8, d7, d2
- **Claim**: The winner of the latest Nobel Prize in Literature is Han Kang, a South Korean author

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: The last person appointed to the U.S. Supreme Court is Ketanji Brown Jackson

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: In the Age of the Smart Machine: The Future of Work and Power (1988)
2

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d2, d1
- **Claim**: The Support Economy: Why Corporations Are Failing Individuals and the Next Episode of Capitalism (co-authored with James Maxmin)
3

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d7, d10, d9, d2, d1
- **Claim**: The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power (2019)
4

### Sample #0334

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Age of the Smart Machine: The Future of Work and Power (co-authored with James Maxmin)

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: It's worth noting that the idea of organic web shooters was not well-received by some fans, who felt that it made Spider-Man less relatable and more like a superhero with a gimmick

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, the concept was also seen as a way to make the character more human and vulnerable, as it would require him to constantly replenish his webbing and deal with the consequences of using it

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d9, d2, d1
- **Claim**: The signers of the Declaration of Independence were 56 delegates to the Continental Congress

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d9, d6, d1
- **Claim**: The current world population is approximately 8.2 billion people, as of January 8, 2025, according to the United Nations estimates elaborated by Worldometer (document d1)

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9, d5, d1
- **Claim**: The current top-ranked country in the FIBA Men's World Ranking is the USA, according to the document with doc_id "d9"

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The host stadium of this year's Super Bowl is Allegiant Stadium in Las Vegas, Nevada

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d6, d7, d2
- **Claim**: In conclusion, while there is some variation in the opinions, most of the documents suggest that a well-planned vegan diet can be safe for pregnant women if they receive proper nutrition and supplements

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, it is essential for pregnant women to consult with a healthcare professional or a registered dietitian to ensure they are getting all the necessary nutrients for a healthy pregnancy

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d8, d7, d9
- **Claim**: Champagne comes solely from the Champagne region of France

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d1
- **Claim**: The Word of Wisdom became mandatory in 1851, when President Brigham Young proposed to the general conference of the Church that all Saints formally covenant to keep the Word of Wisdom the proposal was unanimously upheld by the membership of the Church

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1
- **Claim**: The meaning of AUV in cars is Asian Utility Vehicle

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: It refers to a type of vehicle that is designed for use in the Asian market, typically seating 8-10 people, hauling goods being used for commercial purposes

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, in the context of underwater vehicles, AUV stands for Autonomous Underwater Vehicle, which is an unmanned, untethered vehicle used to conduct underwater research

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer to the question of why club soda works well for stain removal is not explicitly stated in the provided documents

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Based on the retrieved documents, it appears that bath towels made of certain materials, such as Turkish cotton (document d1) and microfiber (document d5), become more absorbent over time

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Document d3 specifically mentions that Turkish cotton towels become more absorbent and feathery after being washed multiple times

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d2, d1
- **Claim**: The premiere of America's Got Talent aired on May 31, 2016

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: In summary, the retrieved documents do not provide a direct answer to the question, but they do provide some related information about colour perception and the properties of metals

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, without more specific information on the differences between socialism and communism, it is difficult to provide a detailed comparison

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The jazz classic "I Got Rhythm" was written by George Gershwin

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: - Jamie's School Dinners (document d2)
- The Naked Chef (document d5)

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: The documents suggest that night vision is often green because it is the color that is most easily perceived by the human eye in low-light conditions

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is due to the sensitivity of the rods in the eye to the blue-green section of the light spectrum

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Document d1 mentions that UCLA won its second consecutive championship in 1965, implying that they won at least one championship before that

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Document d2 mentions that UCLA won the national title in 1967

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Document d3 mentions that UCLA won its tenth National Championship in 1975

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Document d4 mentions that UCLA won the national title in 1971

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: By constantly monitoring these factors and adjusting their odds accordingly, bookmakers can ensure that they maintain a balanced book and make a profit, even in the short term

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Dorothy lived in the Land of Oz, specifically in the area where the story begins, but the exact location is not specified in the provided documents

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, it is mentioned that she wants to return home, implying that she is not from the Land of Oz

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: "The Nose From Jupiter" (1998)
2

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: "A Nose For Adventure"
3

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: "Noses Are Red"
4

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: "The Boy From Earth"
5

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: "The Way to Schenectady" (1998)
6

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: "Of Mice and Nutcrackers" (2001)
7

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: "Mystical Rose" (2000)
8

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2, d1
- **Claim**: Based on the retrieved documents, it appears that salt is used in the process of softening water, specifically in salt-based systems

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: "The softener will regularly 'regenerate' itself by flushing salt through the resin to clean it." Additionally, document d2 mentions that "a softener that uses salt" is an option for softening water that the cost of salt refills is around $5.00-$10.00 for a 40-pound bag

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This definition is from document d5, which provides a clear and concise explanation of gravity as a force that exists between material bodies


================================================================================

*Report generated by CATS v2.0*
