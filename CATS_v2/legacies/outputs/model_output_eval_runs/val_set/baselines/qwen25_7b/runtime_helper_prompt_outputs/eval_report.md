# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 6 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.633 (over 49 samples)

**GR F1** *(used in CATS)*: 0.735

**Behavior Adherence**: 0.558 (over 43 applicable samples)

**Factual Grounding**: 0.357 (over 43 applicable samples)

**Single-Truth Recall**: 0.706 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.589

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.735
- **Precision**: 0.735
- **Recall**: 0.735
- **Accuracy**: 0.633
- TP=25, FP=9, FN=9, TN=6


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.579
- **GR F1** *(used in CATS)*: 0.692
- **Behavior**: 0.706 (n=17)
- **Grounding**: 0.353 (n=17)
- **Recall**: 0.750 (n=12)
- **CATS**: 0.625

### Type 2: Complementary Info

- **Samples**: 15 (3 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.467
- **GR F1** *(used in CATS)*: 0.500
- **Behavior**: 0.500 (n=12)
- **Grounding**: 0.354 (n=12)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.451

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.941
- **Behavior**: 0.333 (n=9)
- **Grounding**: 0.231 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.502

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.600 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.672


================================================================================

## Cost Summary

**Total Cost**: $0.0502

**Decisions Made**: 123

**Average Cost per Decision**: $0.000408


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 123
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0502
- **Total Requests**: 123
- **Average Cost per Request**: $0.000408


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d3, d4, d5, d8, d7
- **Claim**: Children can learn language skills from television, particularly from educational programs and shows that are designed to foster literacy and language skills

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d12, d2, d1, d7
- **Claim**: However, the effectiveness of television in language learning is limited compared to real-life interactions with adults

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [CAN'T ANSWER, INSUFFICIENT EVIDENCE]

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9, d2, d1, d6, d3, d4, d5, d8, d7
- **Claim**: Dina Boluarte is the most recent woman to become President of Peru. []

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1
- **Claim**: Stress does not cause stomach ulcers on its own, but it can contribute to their development, particularly when combined with the use of NSAIDs or an H. pylori infection. []

### Sample #0139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d9, d2, d6, d1, d3, d4, d8, d7
- **Claim**: Heated gemstones are generally less valuable than unheated gemstones, as the heat treatment process is often used to improve the color, clarity other characteristics of the gemstone

### Sample #0175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE [d7, d9] for the total number of employees across all divisions

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d9, d6, d1, d4, d7
- **Claim**: While a business plan is crucial for guiding a startup, securing funding outlining strategic goals, it is becoming less necessary for startups that do not seek outside investment

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: Instead, pitch decks are increasingly preferred by investors as a more concise and visually appealing way to present a business idea

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9, d6, d3, d4, d5, d7
- **Claim**: Sitting is indeed a significant health risk there is evidence suggesting that it can contribute to various health issues similar to those associated with smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d10
- **Claim**: Smoking poses a more severe and widespread threat to public health, with higher risks and costs

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d6, d3, d4, d5
- **Claim**: Smoking can indeed cause Rheumatoid Arthritis

### Sample #0263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [CAN'T ANSWER, INSUFFICIENT EVIDENCE FOR ALL CASES]

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d6, d3, d4, d5, d8, d7
- **Claim**: Dogs can understand human language to a certain extent

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d3, d4, d5, d8, d7
- **Claim**: They can recognize specific words and commands, such as "sit," "walk," and "eat." Research indicates that dogs can differentiate between known and unknown words and that their brains process words and intonation similarly to humans

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7
- **Claim**: Some dogs, like Chaser, have been trained to understand a large vocabulary and use words contextually

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d3, d4, d8, d7
- **Claim**: [CANONICAL ANSWER] All octopuses are venomous. []

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d12, d9, d13, d1, d14, d3, d4, d5
- **Claim**: Melting sea ice does not directly contribute to sea level rise due to Archimedes' principle, as the volume of water displaced by the ice is the same as the volume of water it adds when it melts

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6
- **Claim**: However, some studies suggest that changes in salinity and density due to melting sea ice can indirectly affect sea level

### Sample #0334

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE []

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d2, d1, d6, d3, d4, d5, d8, d7
- **Claim**: The current world population is over 8 billion, as of November 15, 2022 it is estimated that over 108 billion people have ever lived on Earth. []

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d5, d4, d1
- **Claim**: The current top-ranked country at the FIBA Men's World Ranking is the USA []

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d4, d5, d7
- **Claim**: Pregnant women can follow a vegan diet if it is well-planned and monitored to ensure adequate intake of essential nutrients

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d6, d3, d4, d5, d7
- **Claim**: While some sources, such as the Royal Academy of Medicine in Belgium and the article on the vegan keto diet , caution against vegan diets due to potential nutritional deficiencies, other studies and organizations () support the safety and benefits of vegan diets during pregnancy, provided they are carefully planned and supplemented as necessary

### Sample #0399

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d6, d4, d5, d7
- **Supporting Docs Found**: None
- **Claim**: It is crucial for pregnant women considering a vegan diet to consult with healthcare professionals and registered dietitians to ensure they meet all nutritional requirements for themselves and their developing babies

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d6, d1
- **Claim**: The Word of Wisdom became a formal covenant in 1851 when President Brigham Young proposed that all Saints formally covenant to keep it

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6, d3, d4, d8, d7
- **Supporting Docs Found**: None
- **Claim**: An AUV stands for Autonomous Underwater Vehicle, which is an unmanned vehicle designed to operate underwater without guidance according to preprogrammed instructions

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Club soda works well for getting stains out because it is effective at removing stains from various beverages such as beer and red wine

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Cotton towels, especially those made of pima cotton, become more absorbent and feathery with each wash

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is because the fibers in the towel soften and align better with each washing, enhancing their ability to absorb water

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: George Gershwin wrote the jazz classic "I Got Rhythm."

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Night vision is green because the human eye is more sensitive to green light in low-light conditions

### Sample #0592

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The most gold medals in the 2014 Commonwealth Games were won by Ian Thorpe, an Australian swimmer, with 10 gold medals

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Bookmakers calculate odds primarily based on the probability of an event occurring

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: They use historical data, expert analysis current market conditions to estimate these probabilities

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: For instance, in a simple scenario like a dice roll, each side has a 16.67% chance of landing face up, which translates into odds

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Dorothy lived in Kansas


================================================================================

*Report generated by CATS v2.0*
