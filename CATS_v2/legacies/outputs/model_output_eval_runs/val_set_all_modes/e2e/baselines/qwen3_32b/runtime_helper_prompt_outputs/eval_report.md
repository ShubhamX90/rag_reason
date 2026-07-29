# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 3 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.673 (over 49 samples)

**GR F1** *(used in CATS)*: 0.789

**Behavior Adherence**: 0.761 (over 46 applicable samples)

**Factual Grounding**: 0.597 (over 46 applicable samples)

**Single-Truth Recall**: 0.794 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.735

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.789
- **Precision**: 0.714
- **Recall**: 0.882
- **Accuracy**: 0.673
- TP=30, FP=12, FN=4, TN=3


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.684
- **GR F1** *(used in CATS)*: 0.786
- **Behavior**: 0.647 (n=17)
- **Grounding**: 0.578 (n=17)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.711

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 1.000 (n=14)
- **Grounding**: 0.737 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.821

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.750
- **Behavior**: 0.600 (n=10)
- **Grounding**: 0.298 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.549

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.867 (n=5)
- **Recall**: 0.700 (n=5)
- **CATS**: 0.842


================================================================================

## Cost Summary

**Total Cost**: $0.0960

**Decisions Made**: 193

**Average Cost per Decision**: $0.000497


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 193
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0960
- **Total Requests**: 193
- **Average Cost per Request**: $0.000497


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d3, d6
- **Claim**: Children can learn language skills from television **when paired with adult co-viewing, educational content interactive engagement**

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d12, d2, d1, d7, d9, d6
- **Claim**: However, conflicting evidence indicates that unstructured screen time, especially for children under 2, may hinder development or show no benefit [d9–d12]

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d11, d2, d1, d7, d9, d3, d6, d8
- **Claim**: The effectiveness depends on factors like program quality, parental involvement age-appropriateness

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d5, d9, d6
- **Claim**: This six-digit Postal Index Number system was implemented to streamline mail sorting and delivery by assigning unique codes to post offices nationwide

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d4, d2, d1, d7, d5, d9, d3, d6, d8
- **Claim**: The most recent woman to become President of Peru is **Dina Boluarte**, who assumed office on December 7, 2022, following the impeachment of her predecessor, Pedro Castillo

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d3, d6
- **Claim**: Heated gemstones are generally less valuable than untreated ones when comparing stones of equal quality, as untreated gemstones are rarer and often preferred for their natural attributes

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d1
- **Claim**: Citations support the existence of innate knowledge through historical theories (Plato, Descartes, Leibniz) and empirical studies on infant cognition (d2, d5)

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d5, d2
- **Claim**: The conflict reflects enduring philosophical and scientific debates

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6
- **Supporting Docs Found**: None
- **Claim**: This aligns with other states like New York and Pennsylvania that also require 18 as the minimum age

### Sample #0201

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d4, d7, d5, d3, d6
- **Claim**: Sitting has been compared to smoking in public health discourse due to its association with increased risks of heart disease, diabetes mortality (d3-d7)

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9
- **Claim**: However, scientific evidence indicates that smoking poses significantly greater risks

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9
- **Claim**: For example, smoking increases all-cause mortality risk by approximately 180%, while sitting elevates it by ~25% (d9, d10)

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: Smoking is linked to over 1 billion projected deaths in the 21st century and $467 billion in annual global costs, far exceeding sitting's impact (d1, d10)

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9, d11, d1
- **Claim**: While prolonged sitting is harmful, experts emphasize that it is neither addictive nor as universally dangerous as smoking (d10, d11)

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d1, d3, d8
- **Claim**: The metaphor oversimplifies nuanced health risks but underscores the importance of reducing sedentary behavior through movement and activity (d8, d11)

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d5, d3, d6
- **Claim**: Smoking increases the risk of developing rheumatoid arthritis (RA) and can worsen its severity

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d2, d1
- **Claim**: In most cities, driving is faster than public transportation due to factors like traffic congestion, waiting times limited transit infrastructure

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1
- **Claim**: For example, in New York and Los Angeles, public transit commutes average 1.9 times longer than driving , while Michigan metro areas show similar disparities

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d2, d1
- **Claim**: Studies confirm car-based travel is often faster, with driving being twice as quick as transit in the U.S

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d9
- **Claim**: However, exceptions exist where dedicated infrastructure, such as bus rapid transit systems, can make public transport faster—for instance, Mexico City’s Metrobús reduces travel times by 40% compared to driving

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d5, d9, d3, d6, d8
- **Claim**: Overall, while driving remains the faster option in most urban contexts, targeted improvements to transit systems could close this gap

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d5, d3, d6, d8
- **Claim**: Dogs can understand human language to a limited extent, primarily through associating specific words with actions or objects and interpreting tone and body language

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d5, d3, d8
- **Claim**: Studies show dogs process words using brain mechanisms similar to humans (d3, d8), distinguish known from unknown words learn thousands of object names (d4, d7)

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: They also rely on intonation and context for praise (d3, d8)

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d7, d3, d6, d8
- **Claim**: While dogs do not comprehend language as humans do, their ability to map words to meanings and respond to vocal cues demonstrates a foundational understanding (d1, d6)

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d5, d9, d3, d6, d8
- **Claim**: Non-verbal communication remains critical, as emphasized by d9, but verbal cues are undeniably part of their social learning

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This cooperative two-player game involves working together to land an airplane at various airports worldwide

### Sample #0301

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: It marks the first time the award has gone to a two-player game

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d4, d12
- **Supporting Docs Found**: d13
- **Claim**: The overwhelming consensus attributes sea level rise to melting land ice (e.g., Greenland and Antarctica) and thermal expansion

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d8, d4
- **Claim**: Some sources like Goodreads and Amazon list higher numbers, but these likely include different editions, compilations anthologies rather than distinct authored works

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d5, d9, d6, d8
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Second Continental Congress

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d9
- **Claim**: Earlier rankings (e.g., Spain first in August 2023 ) are outdated

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d4, d2, d1, d7, d5, d9, d3, d6, d8
- **Claim**: Champagne comes solely from the Champagne region of France due to legal protections and appellation rules

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d5, d9, d3, d6, d8
- **Claim**: These documents highlight that other sparkling wines, regardless of quality, cannot legally be called Champagne unless they originate from this French region

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d5, d1
- **Claim**: The Word of Wisdom was first formally proposed as a binding covenant in 1851 by Brigham Young, accepted unanimously by the Church ()

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d9, d2
- **Claim**: However, strict enforcement as a requirement for temple recommends and priesthood ordination occurred later

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d9, d2
- **Claim**: By 1919–1921, the First Presidency mandated adherence to the Word of Wisdom for temple participation, aligning with broader societal shifts like Prohibition ()

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d7, d5, d9, d6, d8
- **Claim**: While 1851 marked its acceptance as a commandment, practical enforcement as a mandatory standard intensified in the early 20th century

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d1
- **Claim**: AUV in cars stands for Asian Utility Vehicle, a category of passenger-focused vehicles designed for commercial and family use in Asian markets

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d1
- **Claim**: These vehicles prioritize passenger capacity and cargo flexibility over off-road capabilities, differing from SUVs which emphasize rugged terrain performance

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Cotton bath towels often become more absorbent with repeated washing because their fibers undergo structural changes over time

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This process contrasts with microfiber or synthetic materials, which may retain or lose absorbency differently

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The brain perceives reflective surfaces like silver-colored metals as neutral (silver) because they reflect most wavelengths of visible light equally

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Thus, the even reflectivity of metals like silver creates a perceptual "default" neutral color

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The major difference between socialism and communism, according to Marxist theory, is that socialism is viewed as a transitional stage leading to communism

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: In this framework, socialism involves collective ownership and wealth redistribution, while communism represents a classless, stateless society where resources are communally owned

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Night vision appears green because the human eye is most sensitive to green/yellow-green wavelengths in low-light conditions

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: This sensitivity allows green light to be perceived more efficiently than other colors, reducing eye strain while maximizing visibility

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: This real-time adjustment ensures profitability regardless of the event’s outcome

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Mordecai Richler wrote the book "Oh Canada!

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Salt softens water by using a resin inside the softener to collect minerals like calcium and magnesium

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This fundamental force explains why objects fall toward Earth and governs celestial motion


================================================================================

*Report generated by CATS v2.0*
