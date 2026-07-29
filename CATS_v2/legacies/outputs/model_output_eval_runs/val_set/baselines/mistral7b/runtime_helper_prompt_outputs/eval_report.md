# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 1 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.673 (over 49 samples)

**GR F1** *(used in CATS)*: 0.800

**Behavior Adherence**: 0.500 (over 48 applicable samples)

**Factual Grounding**: 0.021 (over 48 applicable samples)

**Single-Truth Recall**: 0.647 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.492

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

- **Samples**: 19
- **GR Accuracy**: 0.579
- **GR F1** *(used in CATS)*: 0.733
- **Behavior**: 0.684 (n=19)
- **Grounding**: 0.053 (n=19)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.576

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.571 (n=14)
- **Grounding**: 0.000 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.433

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.100 (n=10)
- **Grounding**: 0.000 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.330

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.400 (n=5)
- **Grounding**: 0.000 (n=5)
- **Recall**: 0.200 (n=5)
- **CATS**: 0.400


================================================================================

## Cost Summary

**Total Cost**: $0.0505

**Decisions Made**: 131

**Average Cost per Decision**: $0.000386


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 131
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0505
- **Total Requests**: 131
- **Average Cost per Request**: $0.000386


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d7, d2, d3, d1
- **Claim**: Abi Carter won American Idol this year ()

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d2, d4, d6, d3, d5, d1
- **Claim**: Children can learn language skills from television, but the effectiveness of this learning depends on the quality of the educational programs, the presence of social interactions the appropriateness of the content

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d2, d6, d12, d1
- **Claim**: Co-viewing and parent-child conversations are particularly important for promoting language acquisition

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d8, d9, d2, d4, d6, d3, d5, d1
- **Claim**: Chilliwack sings "Gone Gone Gone" []

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d7, d2, d4, d6, d3, d5, d1
- **Claim**: The PIN code system was introduced in India on August 15, 1972 []

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d7, d2, d4, d10, d3, d5, d1
- **Claim**: Dina Boluarte became the most recent woman to become president of Peru on December 7, 2022 []

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d3, d1
- **Claim**: The current ATP top-ranked men's singles tennis player cannot be definitively determined based on the provided documents, as they contain conflicting information. []

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d3
- **Claim**: The East India Company was founded on December 31, 1600 []

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d4, d1
- **Claim**: Stress can contribute to the development of stomach ulcers, but it does not cause them on its own. []

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d2, d4, d3, d5
- **Claim**: At least one person from Princeton University has won the Fields Medal. []

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d2, d4, d6, d3, d1
- **Claim**: Are heated gemstones less valuable?

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d2, d4, d6, d3, d1
- **Claim**: Yes, heated gemstones can be less valuable than natural gemstones of the same quality in terms of color and clarity. []

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d7, d2, d4, d6, d3, d5, d1
- **Claim**: Some documents support the existence of innate knowledge, while others argue against it

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Based on the evidence presented in these documents, it is not possible to definitively answer the question of whether humans have innate knowledge or not. []

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d5, d8
- **Claim**: Mercedes-Benz Group had around 166,000 employees in 2023 []

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d9
- **Claim**: The documents suggest that as of June 1, 2020, the minimum marriageable age in all states, including Massachusetts, is 18 without exceptions

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d7, d2, d4, d6, d3, d1
- **Claim**: However, the documents do not all specify the current legal age for marriage in Massachusetts there is some inconsistency and lack of clarity about the matter

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Therefore, a definitive answer about the current legal age for marriage in Massachusetts cannot be provided based on the available evidence. []

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d4, d3, d5, d1
- **Claim**: Every startup needs a business plan, as it provides guidance, attracts investors serves as a roadmap for success

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: However, a pitch deck may be more effective for fundraising. []

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d7, d2, d4, d6, d10, d1
- **Claim**: Channel 5 started on March 30, 1997. []

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE []

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d6, d3, d5, d1
- **Claim**: Smoking can cause Rheumatoid Arthritis, as multiple sources agree that smoking increases a person's risk of developing Rheumatoid Arthritis and can make the disease worse. []

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d7, d1
- **Claim**: In cities, driving is generally faster than public transportation

### Sample #0263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, ridesharing can potentially make public transportation faster than driving. []

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d2, d4, d6, d3, d5, d1
- **Claim**: Dogs can understand certain words and associate them with specific actions or objects they can distinguish between human words they've previously heard and words they haven't

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Dorfromantik: The Board Game won the Spiel des Jahres award most recently in 2023 []

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d4, d6, d3, d5, d1
- **Claim**: Most octopuses are venomous, but only the blue-ringed octopus is known to be deadly to humans. []

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d8
- **Claim**: Both melting land ice and sea ice contribute to sea level rise, but the primary contributor may vary depending on the specific location and circumstances. []

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d7, d2, d6, d5
- **Claim**: Han Kang won the Nobel Prize in Literature 2024 []

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d4, d1
- **Claim**: The documents suggest that Ketanji Brown Jackson was the last person appointed to the Supreme Court, but they do not provide consistent information about the last person appointed. [d3, d5]

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: The exact number of books Shoshana Zuboff has published cannot be determined with the provided documents. []

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d7, d2, d4, d6, d3, d5, d1
- **Claim**: The 56 signatories of the Declaration of Independence are (in alphabetical order):
[] (Note that all provide the same list of signatories.)

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d7, d2, d4, d6, d3, d5, d1
- **Claim**: The current world population is 8,198,260,420 as of 2025

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d9
- **Claim**: The USA is currently at the top rank at the FIBA Men's World Ranking. []

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d8
- **Claim**: Super Bowl LXI (2027) will be held at SoFi Stadium. []

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d2, d4, d6, d5
- **Claim**: A well-planned vegan diet can be safe and beneficial during pregnancy, but it requires careful planning and monitoring to ensure nutritional needs are met. []

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d8, d9, d7, d2, d4, d6, d3, d5, d1
- **Claim**: Champagne comes solely from the Champagne region of France. []

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d5, d1
- **Claim**: The Word of Wisdom became a commandment for Church members in 1851, but it is unclear when it became a requirement for temple recommends. [d1, d7]

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: In the context of cars, the meaning of AUV is not clear, as the term "Asian Utility Vehicle" is not widely defined or accepted. []

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Club soda can be effective for removing stains, as supported by personal anecdotes and a study conducted by the Dry-cleaning & Laundry Institute (DLI)

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, the effectiveness may depend on the type of stain and the method used. []

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bath towels can become more absorbent with each wash, although the documents do not specify the type of bath towel that exhibits this behavior. []

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1
- **Claim**: America's Got Talent season 11 premiered on May 31, 2016. []

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Our brains register reflective surfaces such as metal as silver in color because of the selective reflection of light by the metal, with gold having low reflectivity at short wavelengths and preferentially reflecting yellow and red. []

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Socialism and Communism share some similarities, such as a focus on collective ownership and a concern for the welfare of the community

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, there are significant differences between the two ideologies

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Communism, as proposed by Marx, is the final stage of societal evolution, where all property is owned in common the state has withered away

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Socialism, on the other hand, is a transitional stage between capitalism and communism, where the means of production are owned and controlled by the workers, but the state still exists to manage the economy and ensure fair distribution of resources. []

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: George Gershwin wrote the jazz classic "I Got Rhythm"

### Sample #0561

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The evidence is inconclusive as to whether Jamie Oliver is a member of a television series cast. []

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Night vision is green because of the technology used in night vision devices, which amplifies a specific range of light that is more visible to the human eye in low light conditions

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is supported by documents , which provide additional context about the sensitivity of the human eye to different colors and the use of LED flashlights to preserve night vision. []

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: UCLA has won 4 NCAA basketball championships []

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d1
- **Claim**: Bookmakers calculate odds in play by using probability and trying to achieve a balanced book

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d1
- **Claim**: They offer odds based on how likely they believe an event is to happen they adjust the odds as more bets are placed to try to achieve a balanced book

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This means that they will offer different odds for the same event depending on the distribution of bets among different outcomes

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d1
- **Claim**: The specific calculation of odds may vary between bookmakers, but the general approach is to use probability and try to achieve a balanced book. []

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Dorothy does not live in the Emerald City in the Wizard of Oz

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Further research is needed to determine where she lives. []

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Salt softens water by replacing the minerals in hard water with sodium ions

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This process is known as ion exchange

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: When hard water passes through a resin bed containing sodium ions, the calcium and magnesium ions in the hard water are exchanged for sodium ions, resulting in soft water. []

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Gravity is a natural phenomenon by which all material bodies attract each other, as defined in document d5

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide a simple, easy-to-understand definition of gravity


================================================================================

*Report generated by CATS v2.0*
