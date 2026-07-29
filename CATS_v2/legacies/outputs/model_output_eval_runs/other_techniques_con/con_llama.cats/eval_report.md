# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**GR Accuracy**: 0.673 (over 49 samples)

**GR F1** *(used in CATS)*: 0.805

**Behavior Adherence**: 0.469 (over 49 applicable samples)

**Factual Grounding**: 0.094 (over 49 applicable samples)

**Single-Truth Recall**: 0.706 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.518

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
- **Behavior**: 0.526 (n=19)
- **Grounding**: 0.066 (n=19)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.550

### Type 2: Complementary Info

- **Samples**: 15
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.696
- **Behavior**: 0.467 (n=15)
- **Grounding**: 0.156 (n=15)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.439

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.300 (n=10)
- **Grounding**: 0.000 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.396

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.200 (n=5)
- **Recall**: 0.400 (n=5)
- **CATS**: 0.550


================================================================================

## Cost Summary

**Total Cost**: $0.0696

**Decisions Made**: 152

**Average Cost per Decision**: $0.000458


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 152
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0696
- **Total Requests**: 151
- **Average Cost per Request**: $0.000461


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d9, d2, d7
- **Claim**: The winner of American Idol this year is Abi Carter, who won the 22nd season of the show, which was crowned on May 19, 2024

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d6, d2, d7
- **Claim**: The answer to the question of whether children learn language skills from television is complex and depends on various factors

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d12, d8, d10, d5, d6, d2, d4, d7
- **Claim**: While some studies suggest that educational television can be beneficial for language development, others note that excessive screen time can displace opportunities for language and motor skill development that caregiver-child interactions are critical for enhancing child language

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d6, d2, d4, d7
- **Claim**: Additionally, the quality of the programming and the level of adult interaction during viewing are also important factors to consider

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d5, d6, d9, d2, d4, d7
- **Claim**: The PIN code system was introduced in India on August 15, 1972, to ease the process of mail sorting and delivery in a country with diverse languages and similar place names

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d10, d5, d6, d9, d2, d4, d7
- **Claim**: Dina Boluarte is the most recent woman to become President of Peru, taking office on December 7, 2022, after her predecessor, Pedro Castillo, was impeached

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d2, d7
- **Claim**: The passages provide conflicting and outdated information about the current ATP top-ranked men's singles tennis player

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d9, d2, d4, d7
- **Claim**: While some passages mention Novak Djokovic as the top-ranked player, others mention Jannik Sinner or provide outdated rankings

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d6
- **Claim**: Given the lack of a clear and up-to-date answer, the answer is unknown

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d5, d6, d2, d4, d7
- **Claim**: While some passages suggest that stress can make ulcers more bothersome or impair the body's ability to heal itself, most passages agree that stress is not a direct cause of stomach ulcers

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The majority of the passages point to H. pylori infection and NSAIDs as the main causes of ulcers

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passage from doc_id=d1 describes the Fields Medal as the most prestigious award in mathematics, but does not provide information about the number of Fields Medal winners from Princeton University

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d5
- **Claim**: Additionally, doc_id=d5 mentions that Princeton University has been affiliated with 15 Fields Medalists doc_id=d10 mentions that almost half of the medalists have held appointments at the Institute for Advanced Study, Princeton

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Therefore, it can be inferred that Princeton University has at least 15 Fields Medal winners

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d6, d2, d4, d7
- **Claim**: The passage in doc_id: d6 directly states that unheated gemstones are rarer and more valuable than heated ones, which implies that heated gemstones are less valuable

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d5, d6, d9, d2, d4, d7
- **Claim**: The concept of innate knowledge is a topic of debate among philosophers, with some arguing that we are born with certain knowledge and others arguing that all knowledge is acquired through experience

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d5, d6, d9, d2, d4, d7
- **Claim**: While some passages suggest that humans have innate knowledge, such as the understanding of three-dimensional objects and the concept of permanent objects, others argue that all knowledge is acquired through sensory experience

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d7, d9, d5
- **Claim**: The minimum age for marriage varies by state in the United States, but as of June 1, 2020, it is 18 without exceptions in all states

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d11, d1, d8, d10, d5, d6, d9, d4, d7
- **Claim**: While some passages suggest that a business plan is not necessary for every startup, the majority of the passages emphasize the importance of a business plan in a startup's success

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d9, d4, d7
- **Claim**: A business plan serves as a roadmap for achieving goals, helps entrepreneurs focus on company goals, obtain feedback from potential partners keep the team on the same page provides a clear understanding of the business model, target market financial projections

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d8, d10, d6, d9, d2, d4, d7
- **Claim**: Channel 5 launched on 30 March 1997

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d11, d10
- **Claim**: The answer to the question "Is sitting the new smoking?" is not supported by scientific evidence, according to some studies

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d10
- **Claim**: While prolonged sitting has been linked to various health risks, the risks associated with smoking are significantly higher

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d5, d6, d2, d4
- **Claim**: Smoking is a risk factor for developing Rheumatoid Arthritis (RA) it can make the disease worse

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6
- **Claim**: It is associated with several pathogenic mechanisms on RA development, including oxidative stress, inflammation epigenetic changes

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d5, d9, d2, d4, d7
- **Claim**: Public transportation is generally slower than driving in cities, but there are examples of successful public transportation systems that have reduced travel times and increased ridership

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: These systems often prioritize convenience, affordability ease of use, making them more attractive to users

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d5, d6, d2, d4, d7
- **Claim**: Dogs can understand human language to a certain extent, including learning to associate specific words with specific meanings and responding to them

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d8
- **Claim**: They can also understand the emotional content of human speech, such as tone and pitch

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d5, d6
- **Claim**: However, they do not understand human language in the same way that humans do their ability to understand is limited by their cognitive abilities and the complexity of human language

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: The most recent Spiel des Jahres award winner is Sky Team, announced in 2024

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d7, d4
- **Claim**: The passages confirm that all octopuses are venomous, but not all their venom is deadly to humans

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d7, d6
- **Claim**: The blue-ringed octopus is an exception, with venom that can be fatal to humans

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d5, d6, d2, d7
- **Claim**: The latest Nobel Prize in Literature was awarded to South Korean author Han Kang in 2024 for her intense poetic prose that confronts historical traumas

### Sample #0333

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passages do not provide a clear answer to the question of who the last person appointed to the U.S. Supreme Court was

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d10, d5, d6, d9, d2, d4, d7
- **Claim**: The passages provide varying information about Shoshana Zuboff's published works, but none of them provide a comprehensive list of her books

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d8, d10, d5, d6, d9, d2, d4, d7
- **Claim**: Some passages mention specific titles, while others mention the number of books she has written or the number of languages in which her works have been translated

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d10, d6
- **Claim**: However, the information is not consistent across all passages some passages appear to be incomplete

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d4
- **Claim**: The passages suggest that Spider-Man originally had artificial web shooters in the comics, but the idea of organic web shooters was explored in the 2002 film with Tobey Maguire's version

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, the comic book origin of Spider-Man's powers and web-shooters is that the spider bite gave him the knowledge of how spider webs work and allowed him to make them

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d7
- **Claim**: The passages suggest that the Declaration of Independence was adopted on July 4, 1776, but not signed until August 2, 1776

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d8
- **Claim**: This is also confirmed by other documents, such as d3 and d8, which report the world population reaching 8 billion on November 15, 2022

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d5, d4
- **Claim**: The current top rank at the FIBA Men's World Ranking is the USA, according to the latest information available

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d10, d6, d9, d2
- **Claim**: The passages do not provide information about the host stadium for this year's Super Bowl

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d2, d4, d7
- **Claim**: A well-planned vegan diet can be safe for pregnant women if they meet their nutrient needs through a variety of plant foods and reliable sources of essential nutrients

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d5, d6, d2, d4, d7
- **Claim**: However, it is essential to be aware of potential nutritional shortcomings and take necessary precautions to avoid deficiencies

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d11, d1, d8, d10, d5, d6, d9, d2, d4, d7
- **Claim**: The passages consistently emphasize that only sparkling wine from the Champagne region of France can be called Champagne that the term is protected by law

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d11, d1, d8, d5, d6, d9, d2, d4, d7
- **Claim**: This suggests that Champagne can only come from France

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d7, d5, d6
- **Claim**: The passages suggest that the Word of Wisdom was first proposed as a commandment by President Brigham Young in 1851, but it did not become a requirement for temple recommends until 1921

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8
- **Claim**: The exact date when it became mandatory for all church members is unclear, but it is mentioned that it was not considered a commandment until 1919, when it became a requirement for temple recommends

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: The first two passages describe AUVs as a type of vehicle that is similar to SUVs but lacks off-road capability as passenger vehicles that are not designed for off-road routes

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, these passages do not provide a clear definition of AUVs in the context of cars

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The remaining passages define AUVs as autonomous underwater vehicles, which are not relevant to the question about AUVs in the context of cars

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d4
- **Claim**: The passages do not provide a clear explanation for why club soda works well for stain removal

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d4
- **Claim**: While some passages mention its effectiveness, they do not provide the underlying reason

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The passage in doc_id=d3 suggests that cotton towels can become more absorbent and feathery with repeated washing, but it does not provide a clear explanation for this phenomenon

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d4
- **Claim**: The passages do not provide a clear answer to the question of when America's Got Talent comes on

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d4
- **Claim**: While they mention the premiere date of season 11 (May 31, 2016), they do not specify the exact time

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The passages do not directly address the question of why our brains register reflective surfaces as a specific colour, such as silver

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: They discuss various aspects of colour perception, light materials, but do not provide a clear explanation for this phenomenon

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passages provided do not directly address the question about the differences between socialism and communism

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While they mention socialism and communism in passing, they do not provide a clear explanation of the differences between the two ideologies

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer is unknown

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passages do not provide direct information about the writer of the jazz classic "I Got Rhythm"

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While some passages mention the song, they do not specify the writer

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer is unknown

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Jamie Oliver is a member of the cast of the television series "Jamie's School Dinners"

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The passages provide some hints about why night vision is often associated with the color green, but they do not provide a clear explanation

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: However, this is not explicitly stated the other passages do not provide a clear answer

### Sample #0592

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passages do not provide a clear answer to the question about the team that won the most gold medals at the 2014 Commonwealth Games

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, they do mention that Kenya and Jamaica won 10 gold medals each Australia won 8 gold medals

### Sample #0592

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Since the passages do not provide a clear answer, the answer is unknown

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d2, d4
- **Claim**: The first passage mentions UCLA winning their second consecutive championship, the third passage mentions UCLA winning their tenth championship the third and fourth passages confirm that UCLA won championships in 1967 and 1971, respectively

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Based on this information, UCLA has won at least 10 NCAA basketball championships

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d2
- **Claim**: The passages do not provide a clear explanation of how bookmakers calculate the initial odds for a certain event

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d4
- **Claim**: The passages focus on other aspects of bookmaking, such as setting starting price, achieving a balanced book calculating margins

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The passages do not explicitly state where Dorothy lives in the Land of Oz

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The third passage confirms that Mordecai Richler is the author in question, but it does not list any specific books written by him

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, the passage does not provide a comprehensive list of his works

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the lack of specific information, the answer is unknown

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: The passages do not provide a clear explanation of how salt softens water

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: They discuss various aspects of water softeners, such as costs, installation different types of systems, but do not address the specific question of how salt softens water

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: A simple and accurate definition of gravity that can be easily explained is that gravity is a natural phenomenon by which all material bodies attract each other


================================================================================

*Report generated by CATS v2.0*
