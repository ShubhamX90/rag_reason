# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 2 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.735 (over 49 samples)

**GR F1** *(used in CATS)*: 0.840

**Behavior Adherence**: 0.681 (over 47 applicable samples)

**Factual Grounding**: 0.023 (over 47 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.592

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
- **Behavior**: 0.722 (n=18)
- **Grounding**: 0.000 (n=18)
- **Recall**: 0.875 (n=12)
- **CATS**: 0.599

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.714 (n=14)
- **Grounding**: 0.000 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.481

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.500 (n=10)
- **Grounding**: 0.000 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.482

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.220 (n=5)
- **Recall**: 0.700 (n=5)
- **CATS**: 0.680


================================================================================

## Cost Summary

**Total Cost**: $0.0537

**Decisions Made**: 137

**Average Cost per Decision**: $0.000392


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 137
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0538
- **Total Requests**: 137
- **Average Cost per Request**: $0.000392


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d3, d1, d2, d7
- **Claim**: Therefore, the winner of American Idol this year is Abi Carter

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d6, d2, d7
- **Claim**: In conclusion, the documents present a mix of supportive and contradictory evidence, indicating that the impact of television on children's language skills is nuanced and depends on various factors such as the nature of the content, the presence of adult interaction the age of the child

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d3, d8, d1, d6, d2, d7
- **Claim**: Therefore, the answer to whether children learn language skills from television is not straightforward and varies based on the context and conditions under which the viewing occurs

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d3, d8, d4, d1, d6, d2, d10
- **Claim**: Therefore, the answer to the query "Who sings 'gone gone gone she been gone so long?'" is Chilliwack

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d3, d8, d4, d1, d6, d2, d10, d7
- **Claim**: Therefore, Dina Boluarte is the most recent woman to become President of Peru

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d2
- **Claim**: Therefore, the East India Company came into existence on December 31, 1600

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d4, d1, d6, d7
- **Claim**: Given the conflicting opinions and research outcomes, the consensus from the documents is that while stress can play a role in exacerbating conditions that lead to ulcers, it is not the primary cause

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The main causes of ulcers are identified as H. pylori infection and the use of NSAIDs

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Therefore, the answer to the query is that 15 people from Princeton University have won the Fields Medal

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d8, d4, d1, d6, d2, d7
- **Claim**: In conclusion, while heated gemstones can be less valuable than their unheated counterparts, the extent and applicability of this rule depend on the specific type of gemstone and the context of the comparison

### Sample #0175

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the conflicting opinions presented in the documents, it is clear that the question of whether humans have innate knowledge remains unresolved, with strong arguments on both sides of the debate

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d1, d6, d7
- **Claim**: Given the variability across jurisdictions, there is no single universal legal age for marriage

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, the trend towards raising the minimum age to 18 without exceptions is growing in some states

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d11, d3, d4, d1, d6, d10, d7
- **Claim**: Given the conflicting opinions, the conclusion is that while business plans are highly recommended and can be crucial for certain aspects of a startup's development, they are not necessarily required for every startup

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d11, d3, d4, d1, d6, d10
- **Claim**: The necessity depends on the specific circumstances, such as the need for external funding or the preference for a lean startup approach

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d11, d3, d8, d1, d2, d10
- **Claim**: Given the conflicting opinions and research outcomes, the answer to the query "Is sitting the new smoking?" is nuanced

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9, d1, d11
- **Claim**: While sitting can indeed pose health risks, the evidence does not support equating it to the severity of smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d11, d8, d1, d10
- **Claim**: Therefore, the claim that "sitting is the new smoking" is not fully supported by the available scientific evidence

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d3, d8, d4, d1, d6, d2, d7
- **Claim**: Overall, the documents suggest that public transportation is generally slower than driving in cities, with some exceptions and conditions under which public transport can be faster

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d3, d8, d4, d1, d6, d2, d7
- **Claim**: Given the conflicting opinions and research outcomes, it is clear that while dogs can understand some human words and interpret tone and body language, the extent to which they fully comprehend human language remains debated

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: Based on the provided documents, the most recent winner of the Spiel des Jahres award is **Sky Team**, which won in 2024

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d8, d4, d1, d6, d7
- **Claim**: Therefore, the answer to the query "Are most octopuses venomous?" is yes, as all octopuses are venomous according to the provided documents

### Sample #0322

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the conflicting evidence, it is evident that there is disagreement in the scientific community regarding the exact impact of melting sea ice on sea level rise

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d8, d6, d2, d7
- **Claim**: Therefore, the latest Nobel Prize in Literature winner is Han Kang

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d1
- **Claim**: Based on the provided documents, the last person appointed to the U.S. Supreme Court was Ketanji Brown Jackson

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3, d1
- **Claim**: Therefore, the most recent appointment to the U.S. Supreme Court is Ketanji Brown Jackson

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d3, d8, d4, d1, d6, d2, d7
- **Claim**: Given the complementary information across the documents, the most consistent answer is that Shoshana Zuboff has published three major books, though additional titles exist under her name

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d2
- **Claim**: Based on the retrieved documents, Spiderman originally did not have organic web shooters

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d2
- **Claim**: Instead, in the original comic books, Spiderman's web-shooters were mechanical devices designed by Peter Parker himself

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Organic web shooters were a change introduced specifically for film adaptations, such as in Sam Raimi's 2002 Spider-Man film

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d2
- **Claim**: Therefore, the answer to the query is no, Spiderman did not originally have organic web shooters

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d8, d4, d1, d2, d7
- **Claim**: While these documents confirm the number of signers and mention some notable signatories, they do not provide a complete list of all 56 signers

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d9, d6, d2
- **Claim**: For a comprehensive list, one would need to refer to the full document from the National Archives or another detailed source

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Therefore, the best answer to the query "How many people are there in the world?" based on the provided documents is **8,198,260,420**

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d9
- **Claim**: Therefore, the USA is currently at the top rank in the FIBA Men's World Ranking

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: Since the query asks for the host stadium of "this year's" Super Bowl the documents indicate that Super Bowl LVIII was held in 2024 at Allegiant Stadium, the host stadium for this year's Super Bowl is **Allegiant Stadium** in Las Vegas, NV

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d6, d2, d7
- **Claim**: In summary, while there are differing opinions on the safety and advisability of a vegan diet during pregnancy, the consensus seems to be that a well-planned vegan diet can be safe and beneficial, provided that it is carefully managed to meet all nutritional requirements

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d3, d8, d4, d1, d2, d7
- **Claim**: Therefore, based on the provided documents, champagne, when referring to the legally protected and traditionally recognized sparkling wine, comes solely from France

### Sample #0416

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the conflicting dates provided by the documents, it is clear that there is disagreement over the exact year the Word of Wisdom became mandatory

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9, d8, d1, d6, d7
- **Claim**: While some sources point to 1851 as the year it became a commandment, others suggest it became a requirement for temple recommends in either 1919 or 1921

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: None of the documents offer a scientific explanation for why club soda is effective at removing stains

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the documents are insufficient to answer the query fully

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Based on the provided documents, the information regarding why bath towels seem to get more absorbent the more they get washed over time is limited and specific to certain types of towels

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Therefore, the given documents do not sufficiently explain why bath towels in general become more absorbent with repeated washing

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: In summary, while the documents provide some context about colour perception and the nature of metallic colours, they do not sufficiently explain why reflective surfaces are registered as silver in colour by our brains despite reflecting all surrounding colours

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Given the lack of detailed information in the provided documents, they are insufficient to fully answer the query about the major differences between socialism and communism

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Based on the retrieved documents, Jamie Oliver is a member of the cast of the documentary series "Jamie's School Dinners," which aired on Channel 4 in the United Kingdom from 23 February to 16 March 2005

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Additionally, there is partial support indicating that Jamie Oliver has his own television show called "The Naked Chef," where another chef, Contaldo, appeared

### Sample #0561

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, the primary and most direct answer to the query is that Jamie Oliver is a cast member of "Jamie's School Dinners."

### Sample #0588

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Combining these pieces of information, we can infer that night vision displays are green because green light falls within the range of wavelengths that the human eye is most sensitive to in low-light conditions

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: Additionally, green may be chosen because it is energy-efficient and effectively enhances visibility without overwhelming the viewer's night vision

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the specific technical details regarding phosphor screens or other components that cause the green color are not provided in the given documents

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Based on the given documents, we can infer that Kenya and Jamaica each won 10 gold medals at a particular meet, which could potentially refer to the 2014 Commonwealth Games, although this is not explicitly confirmed in the snippets provided

### Sample #0592

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the documents do not provide sufficient information to definitively answer who won the most gold medals in the 2014 Commonwealth Games

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d2
- **Claim**: Based on the provided documents, we can confirm that UCLA has won multiple NCAA basketball championships, including at least the following years: 1964-65, 1967 1971, with the 1974-75 season marking their tenth championship in twelve years

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide a comprehensive or exact total number of championships won by UCLA

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Therefore, while we know UCLA has won numerous championships, the specific total count cannot be definitively stated from the given information

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The provided documents do not sufficiently address the specific query regarding how bookmakers calculate odds in play, particularly the rapid adjustments made during live events for specific outcomes like a player scoring or an event happening

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d1, d2
- **Claim**: Each document provides some foundational understanding of how bookmakers set odds based on probabilities and percentages, but none delve into the specifics of in-play betting or the speed at which odds are recalculated in real-time

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the documents are insufficient to fully answer the query

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Based on the provided documents, there isn't sufficient information to definitively answer where Dorothy lived in "The Wizard of Oz." While one document mentions Dorothy being in California during an earthquake in the book "Dorothy and the Wizard of Oz," this does not address the original setting of Dorothy's home in the primary "Wizard of Oz" story

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: None of the snippets directly state that Dorothy lived in Kansas, which is the well-known starting point for Dorothy's adventure in the original story

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Therefore, the documents do not provide enough evidence to answer the query accurately

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide a comprehensive list of all the books written by Mordecai Richler

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the available information is insufficient to answer the query fully

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: In summary, although the documents do not provide a complete explanation, they suggest that salt softens water through an ion-exchange process where sodium ions from the salt replace the hardness-causing minerals on the resin, thus softening the water

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This definition captures the essence of gravity in simple terms, stating that it is a force of attraction between all material objects

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: However, the documents do not provide a comprehensive explanation that includes the causes or mechanisms behind gravity in a simplified manner suitable for easy understanding

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Therefore, while we can provide a definition, a full, easily explained description of gravity's workings is not fully covered by the given documents


================================================================================

*Report generated by CATS v2.0*
