# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 1 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.673 (over 49 samples)

**GR F1** *(used in CATS)*: 0.800

**Behavior Adherence**: 0.625 (over 48 applicable samples)

**Factual Grounding**: 0.483 (over 48 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.697

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.800
- **Precision**: 0.696
- **Recall**: 0.941
- **Accuracy**: 0.673
- TP=32, FP=14, FN=2, TN=1


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.684
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.722 (n=18)
- **Grounding**: 0.366 (n=18)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.701

### Type 2: Complementary Info

- **Samples**: 15
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.696
- **Behavior**: 0.667 (n=15)
- **Grounding**: 0.567 (n=15)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.643

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.700
- **GR F1** *(used in CATS)*: 0.824
- **Behavior**: 0.300 (n=10)
- **Grounding**: 0.508 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.544

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.600 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.800


================================================================================

## Cost Summary

**Total Cost**: $0.0602

**Decisions Made**: 143

**Average Cost per Decision**: $0.000421


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 143
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0605
- **Total Requests**: 143
- **Average Cost per Request**: $0.000423


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d11, d12, d2, d3, d7, d10, d8, d9, d4
- **Claim**: Children learn language skills from television to varying degrees, with some research suggesting benefits and others finding no effect or even negative impacts

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d10, d8, d9, d4
- **Claim**: My Girl (Gone, Gone, Gone) is a song performed by Chilliwack

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d4
- **Claim**: The song contains the lyrics 'gone gone gone she's been gone so long' or 'gone gone gone, she been gone so long.' The song was released in 1981

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d8, d9, d4
- **Claim**: The PIN code system was introduced on August 15, 1972, in India

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d10, d8, d9, d4
- **Claim**: Peru's First Woman President Takes the Reins Amid Unrest Peru inaugurated its first women president into office, Dina Boluarte

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3, d7, d10, d9, d4
- **Claim**: This comes after the impeachment, arrest and removal from office of former President

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d10, d8, d9, d4
- **Claim**: News Peru’s First Woman President Takes the Reins Amid Unrest Just before the start of the new year Peru inaugurated its first women president into office, Dina Boluarte

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d8, d9, d4
- **Claim**: The current ATP top-ranked men's singles tennis player is Jannik Sinner

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d4
- **Claim**: Can stomach ulcers be caused by stress?

### Sample #0201

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d11, d10, d8, d9
- **Claim**: Is sitting the new smoking?

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d8, d4
- **Claim**: Can dogs understand human language?

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d8, d4
- **Claim**: To a certain extent, yes

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d8, d4
- **Claim**: Research shows that dogs can process words and intonation similarly to humans some dogs can understand hundreds of object names

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d3, d7, d8, d4
- **Claim**: Most octopuses are venomous, with all species possessing venom

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d8, d9, d4
- **Claim**: Shoshana Zuboff has published multiple books, with varying counts ranging from 3 to 16 across the retrieved documents

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: The exact number of books she has published cannot be definitively determined from the provided evidence

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d2
- **Claim**: Did Spiderman originally have organic web shooters?

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d2
- **Claim**: No. Spider-Man originally had mechanical web shooters in the comics organic web shooters were a change introduced for film adaptations ()

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d9, d2
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Continental Congress, with John Hancock being the first signer

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: 8,198,260,420 people (as of January 8, 2025, according to the most recent United Nations estimates) are currently living on Earth

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d9, d4
- **Claim**: However, the query asks for the current population in general terms all documents agree that the current world population is more than 8 billion

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d6, d2, d3, d7, d8, d9, d4
- **Claim**: Champagne comes solely from the Champagne region of France, as all retrieved documents agree that it is legally protected as a name for sparkling wine produced in that region

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The term 'Asian Utility Vehicle' is used for vehicles predominantly sold in Asia, designed to seat 8-10 people, haul goods serve commercial purposes

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Examples include the Toyota Innova and Crosswind

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Why does club soda work so well getting stains out?

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: The documents suggest that club soda can effectively remove certain stains, but they do not provide a clear explanation of why it works so well

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Towels like Turkish and pima cotton towels may become more absorbent with repeated washing, although the exact reason for this is not explained in the provided documents

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is likely that the increased absorbency is due to the fibers in the towels becoming more intertwined and better at wicking water after multiple washes

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Why do our brains register reflective surfaces such as metal as silver in colour if they're actually reflecting all of the colours around them?

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: The documents provided offer insights into colour perception and the properties of metals, but they do not directly address the question of why the brain perceives reflective surfaces as silver rather than as a mix of reflected colours

### Sample #0517

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: To answer this question, we would need to delve into the specific neural mechanisms that allow the brain to interpret reflected light as a single colour how these mechanisms might be influenced by the properties of reflective surfaces

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Socialism is a stage on the way to communism

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, to fully understand the major differences between socialism and communism, it is necessary to consult additional resources that focus specifically on these ideologies and their key distinctions

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Who wrote the jazz classic "I Got Rhythm"?

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Night vision devices display a green image due to a combination of factors related to human eye sensitivity and energy efficiency, as suggested by the documents

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d2, d3, d4
- **Supporting Docs Found**: None
- **Claim**: The bookmakers calculate in-play odds using a combination of historical data, real-time statistics algorithms that adjust the odds based on the flow of the game

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The speed of these adjustments can be rapid, as bookmakers aim to keep their odds as accurate as possible

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: However, the retrieved documents do not provide enough information to determine the full list of books written by Mordecai Richler

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is likely that he wrote more books than just this one

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: How does salt soften water?

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The retrieved documents suggest that salt is used in water softening systems the mechanism involves ion exchange or resin collection

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d2
- **Claim**: However, they do not provide a clear explanation of the process

### Sample #0650

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: For a detailed understanding, further research is needed to find a source that explains the ion-exchange mechanism in detail

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Gravity is a natural phenomenon by which all material bodies attract each other

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: It may be partially explained by motion and bombardment by subparticles, as suggested in some theories

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, a simple, easily explained definition of gravity that can be easily understood may not be found in the provided documents


================================================================================

*Report generated by CATS v2.0*
