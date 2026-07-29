# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 9 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.796 (over 49 samples)

**GR F1** *(used in CATS)*: 0.857

**Behavior Adherence**: 0.475 (over 40 applicable samples)

**Factual Grounding**: 0.428 (over 40 applicable samples)

**Single-Truth Recall**: 0.559 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.580

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.857
- **Precision**: 0.833
- **Recall**: 0.882
- **Accuracy**: 0.796
- TP=30, FP=6, FN=4, TN=9


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (5 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.842
- **GR F1** *(used in CATS)*: 0.880
- **Behavior**: 0.500 (n=14)
- **Grounding**: 0.407 (n=14)
- **Recall**: 0.792 (n=12)
- **CATS**: 0.645

### Type 2: Complementary Info

- **Samples**: 15 (3 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.733
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.583 (n=12)
- **Grounding**: 0.556 (n=12)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.646

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.556 (n=9)
- **Grounding**: 0.436 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.664

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.400
- **GR F1** *(used in CATS)*: 0.571
- **Behavior**: 0.000 (n=5)
- **Grounding**: 0.167 (n=5)
- **Recall**: 0.000 (n=5)
- **CATS**: 0.185


================================================================================

## Cost Summary

**Total Cost**: $0.0967

**Decisions Made**: 184

**Average Cost per Decision**: $0.000526


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 184
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0967
- **Total Requests**: 184
- **Average Cost per Request**: $0.000526


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The most recent and specific information comes from d2

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d5, d3, d7, d1, d4, d8, d2
- **Claim**: Children can learn language skills from television, but the quality of the programming and the level of adult interaction during viewing are crucial factors

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The song "Gone Gone Gone (She Been Gone So Long)" by Chilliwack was written by Brian MacLeod and Bill Henderson, but the documents do not provide explicit information about who sings the song

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, it is reasonable to assume that MacLeod and Henderson are the singers since they wrote the song

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The system is structured as follows: the first digit of the PIN code indicates the region, the second digit specifies the sub-region, the third digit points to the sorting district within the region the last three digits are assigned to individual post offices within the district

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d6, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d8
- **Claim**: The company's primary aim was to secure spices, but it soon expanded its business to trade in cotton, silk, indigo, tea saltpeter

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d6, d7, d8, d2
- **Claim**: The East India Company's monopoly on English trade with India soon expanded to China, importing porcelain and tea to Great Britain and its colonies

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d5, d3, d7, d1, d4, d2
- **Claim**: While stress can serve as a backdrop to stomach ulcers, it does not necessarily cause them

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The number of Fields Medal winners from Princeton University is a contested fact across sources

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, d3 does not specify the year of the award for William P. Thurston, so it is unclear if he should be included in the count

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The evidence remains mixed, with different sources providing different numbers

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d3, d7, d1, d4, d8, d2
- **Claim**: Heated gemstones are generally less valuable than unheated ones due to their lack of natural appeal and the process used to achieve their color

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The relationship between innate knowledge and human cognition is a complex and contested topic

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The evidence landscape is complex, with different sources presenting conflicting views

### Sample #0175

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While some documents support the idea that humans have innate knowledge, others reject the idea

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d5, d2
- **Claim**: The question of whether humans have innate knowledge remains a topic of ongoing debate

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, it is important to note that this number may have changed due to fluctuations in the workforce size over time

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d7, d1, d4, d8
- **Claim**: Every startup should have a business plan, as it serves as a roadmap for achieving business goals, helps secure funding focuses the team on important decisions

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: A good business plan includes an executive summary, a description of the business model, financial information a SWOT analysis

### Sample #0201

- **Reason**: cross_doc_not_cited
- **Cited Docs**: d4, d11
- **Supporting Docs Found**: None
- **Claim**: Some investors may prefer pitch decks over business plans, but a business plan is still necessary during due diligence

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d7, d10, d2
- **Claim**: This date marks the launch of the fifth national terrestrial channel in the United Kingdom

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d5, d3, d7, d4, d10
- **Claim**: Prolonged sitting can have negative health effects, including obesity, metabolic syndrome increased risk of several diseases

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d10
- **Claim**: However, the risks associated with smoking are substantially higher than for sitting

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d10
- **Claim**: A high-credibility source found that the risks of chronic disease and premature death associated with smoking are substantially higher than for sitting

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d11, d1, d10
- **Claim**: While some documents state that sitting is as dangerous as smoking, the evidence does not support this claim

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d8
- **Claim**: It is important to engage in regular physical activity to counteract the negative effects of prolonged sitting

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d1, d2
- **Claim**: In many cities, public transportation is slower than driving due to factors such as traffic congestion, fixed schedules routes

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d2
- **Claim**: However, the evidence is not consistent across all cities and sources

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d9, d8
- **Claim**: Some cities have made investments in rapid mass transit, such as bus rapid transit systems, to reduce travel times for public transportation riders

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d5, d3, d7, d1, d4, d8, d2
- **Claim**: Dogs can understand human language to some extent, focusing on both words and intonation

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d6
- **Claim**: The Spiel des Jahres is a prestigious German award for the best board game of the year, focusing on family-friendly games

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: Dorfromantik is a cooperative tile-placement game about building beautiful, pastoral landscapes

### Sample #0320

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d9
- **Claim**: This is because sea ice is already floating in the ocean the amount of water displaced as ice is about the same as the amount of water added to the ocean when it melts

### Sample #0333

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: In the Raimi trilogy of Spiderman films, Spiderman is depicted with organic web shooters

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d5, d3, d7, d1, d4, d8, d2
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Continental Congress, including John Hancock, Thomas Jefferson Benjamin Franklin

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The signatories represented the new states as follows: New Hampshire (George Read, Caesar Rodney, Thomas McKean), Pennsylvania (George Clymer, Benjamin Franklin, Robert Morris, John Morton, Benjamin Rush, George Ross), Delaware (Thomas McKean, George Read, Caesar Rodney), Maryland (Samuel Chase, William Paca, Thomas Stone, Charles Carroll of Carrollton), Virginia (George Wythe, Richard Henry Lee, Thomas Jefferson, Benjamin Harrison, Thomas Nelson Jr., Francis Lightfoot Lee, Carter Braxton), North Carolina (William Hooper, Joseph Hewes, John Penn), South Carolina (Edward Rutledge, Thomas Heyward Jr., Thomas Lynch Jr., Arthur Middleton), Georgia (Button Gwinnett, Lyman Hall, George Walton), New York (William Floyd, Philip Livingston, Francis Lewis, Lewis Morris), New Jersey (Richard Stockton, John Witherspoon, Francis Hopkinson, John Hart, Abraham Clark), Connecticut (Roger Sherman, Samuel Huntington, William Williams, Oliver Wolcott, Matthew Griswold), Massachusetts (John Hancock, Samuel Adams, John Adams, Robert Treat Paine, Elbridge Gerry), Rhode Island (Stephen Hopkins, William Ellery), Vermont (Thomas Chittenden, Arthur Young, Stephen R. Bradley), New Hampshire (Matthew Thornton), Delaware (George Read, Thomas McKean, Caesar Rodney), Pennsylvania (George Clymer, Benjamin Rush, Benjamin Franklin, Robert Morris, George Ross, James Smith, George Taylor), Virginia (George Wythe, Thomas Jefferson, Benjamin Harrison, Thomas Nelson Jr., Francis Lightfoot Lee, Carter Braxton, Richard Henry Lee), North Carolina (William Hooper, Joseph Hewes, John Penn), South Carolina (Edward Rutledge, Thomas Heyward Jr., Thomas Lynch Jr., Arthur Middleton), Georgia (Button Gwinnett, Lyman Hall, George Walton), Maryland (Samuel Chase, William Paca, Thomas Stone, Charles Carroll of Carrollton) New York (William Floyd, Philip Livingston, Francis Lewis, Lewis Morris)

### Sample #0381

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0392

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, it is unclear whether this information is still accurate as the timestamp is empty

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: To confirm the current rankings, please refer to the official FIBA website

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d8, d2
- **Claim**: Super Bowl LXI will be held at SoFi Stadium in Los Angeles in 2027 Super Bowl LX at Levi's Stadium in Santa Clara in 2026

### Sample #0394

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The host city for Super Bowl LXI has not been specified in the provided documents

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d5, d3, d7, d1, d4, d2
- **Claim**: While some sources caution against vegan diets during pregnancy due to potential nutritional shortcomings, other sources suggest that well-planned vegan diets can meet all nutrient needs during pregnancy

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d5, d7, d1, d4, d2
- **Claim**: For example, d2 (NCBI) states that well-planned vegan diets during pregnancy, lactation, infancy childhood can meet all nutrient needs, as long as they include a variety of plant foods and reliable sources of vitamin B12 and vitamin D. On the other hand, d1 (CBS News) discusses the Belgian Royal Academy of Medicine's recommendation against vegan diets for children, teens, pregnant women nursing mothers due to potential nutritional shortcomings and risks of stunted development

### Sample #0399

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Overall, the evidence landscape is conflicting, with some sources supporting vegan diets during pregnancy and others cautioning against them

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is important for pregnant women to consult with a healthcare provider to ensure they are meeting all of their nutritional needs

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d5, d3, d7, d1, d4, d8, d2
- **Claim**: Champagne is a sparkling wine that can only come from the Champagne region of France

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d5, d3, d7, d4, d8
- **Claim**: It is produced from grapes grown only in the Champagne region, the most northerly wine region of France, north-east of Paris

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Only three grape varieties are used in Champagne production: Chardonnay, Pinot Noir Pinot Meunier

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d5, d3, d7, d10, d1, d4, d8, d2
- **Claim**: Other sparkling wines, such as Prosecco and Cava, are not considered Champagne

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The relationship between the Word of Wisdom and its mandatory status is contested across historical interpretations

### Sample #0416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d7, d8, d5
- **Supporting Docs Found**: None
- **Claim**: The most credible sources do not provide a definitive answer the less credible sources provide conflicting information

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d9, d5, d3, d7, d1, d4, d8, d2
- **Claim**: The Word of Wisdom is a commandment for members of The Church of Jesus Christ of Latter-day Saints, but the exact timeline of when it became mandatory remains unclear

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d7, d8, d1
- **Supporting Docs Found**: None
- **Claim**: An AUV (Autonomous Underwater Vehicle) is an unmanned, untethered vehicle designed to operate underwater without guidance according to preprogrammed instructions

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d7
- **Supporting Docs Found**: None
- **Claim**: AUVs are commonly used for underwater research and exploration

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: The perception of reflective surfaces as silver might be due to the specific wavelengths of light that are reflected by certain metals, such as gold and copper

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: However, this hypothesis is not a definitive answer, as the documents do not provide a clear and direct explanation for the perception of silver

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d5
- **Supporting Docs Found**: None
- **Claim**: Further research may be needed to fully understand this phenomenon

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Socialism and communism are two distinct political and economic ideologies that have been debated and defined in various ways throughout history

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Socialism generally advocates for collective ownership and control of the means of production, while allowing for some private property and a role for the market

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Communism, on the other hand, advocates for the complete elimination of private property and the establishment of a classless, stateless society where all goods are owned in common

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While there are many nuances and variations within these ideologies, this general overview provides a starting point for understanding the differences between socialism and communism

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This was also confirmed in the production of the television show Battlestar Galactica, where real night vision footage was used

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The sensitivity of rods and cones in the human eye to different colors may also play a role in why night vision is green, but this is not directly related to the question

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the sources are not highly credible there is no clear indication of the most recent or authoritative source

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Further research may be necessary to confirm the exact number of titles

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Salt softens water by replacing calcium and magnesium ions in hard water with sodium ions through a process called ion exchange

### Sample #0650

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This makes the water easier to clean with soap and prevents mineral buildup

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d2
- **Claim**: While the retrieved documents do not directly explain this process, it is a well-known principle in water treatment


================================================================================

*Report generated by CATS v2.0*
