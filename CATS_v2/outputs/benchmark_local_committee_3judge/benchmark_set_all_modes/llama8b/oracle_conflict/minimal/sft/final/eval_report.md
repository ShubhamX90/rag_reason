# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.962 (over 736 samples)

**GR F1** *(used in CATS)*: 0.976

**Behavior Adherence**: 0.816 (over 609 applicable samples)

**Factual Grounding**: 0.842 (over 609 applicable samples)

**Single-Truth Recall**: 0.695 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.832

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.976
- **Precision**: 0.998
- **Recall**: 0.956
- **Accuracy**: 0.962
- TP=581, FP=1, FN=27, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.825
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.901
- **Specificity**: 0.956
- Abstain TP=127, FP=27, FN=1, TN=581


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.955 (n=154)
- **Grounding**: 0.892 (n=154)
- **Recall**: 0.821 (n=154)
- **CATS**: 0.915

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.941
- **GR F1** *(used in CATS)*: 0.962
- **Behavior**: 0.893 (n=177)
- **Grounding**: 0.825 (n=177)
- **Recall**: 0.590 (n=156)
- **CATS**: 0.817

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.688 (n=96)
- **Grounding**: 0.819 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.830

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.703 (n=145)
- **Grounding**: 0.875 (n=145)
- **Recall**: 0.682 (n=140)
- **CATS**: 0.812

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.649 (n=37)
- **Grounding**: 0.653 (n=37)
- **Recall**: 0.662 (n=37)
- **CATS**: 0.719


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2043

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/qwen3.5-397b-a17b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000

#### local/mistral-small-4

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000

#### local/deepseek-r1-distill-32b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3, d4
- **Claim**: Peeling an apple reduces some of its nutritional value by removing dietary fiber and certain antioxidants, but not all nutrients — vitamins like vitamin C remain unchanged per 100 grams of apple the peel itself contains a significant amount of fiber, antioxidants flavonoids

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Some producers have begun to adopt sustainable practices such as RSPO certification, but the environmental damage remains a major concern

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Therefore, while the Silurian was a pivotal period for terrestrial plant evolution and diversification, it cannot be proven to have been the absolute 'birth' of the first land plants

### Sample conflictingqa_24fa0020a521

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: In addition, children who drink primarily non-fortified milk alternatives without adequate dietary compensation or who are underweight or failing to thrive may also benefit from supplementation

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: Given this methodological divergence and the presence of credible sources on both sides, the weight of available evidence does not firmly resolve the question of whether fluoride in drinking water is dangerous in general; rather, it points to a conditional answer where safety is tied to the level of exposure

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: The retrieved evidence is mixed. Some sources argue that chlorine does not directly turn hair green and that copper from algaecides is the primary culprit, while others argue that chlorine can cause bleaching and contribute to the greenish appearance

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: These opposing conclusions reflect methodological divergence across philosophical arguments about the limits of thought and personal experience

### Sample conflictingqa_2c0ea18839df

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The conflict is interpretive and methodological, reflecting differing research conclusions on the same biological question

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2
- **Claim**: IPv6 is not fundamentally more secure than IPv4; both protocols are equally secure because IPv4 can also use IPSec and most security incidents result from implementation flaws rather than protocol weaknesses

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict type is complementary information: each document contributes a distinct contextual perspective on the question without providing a universal, unambiguous answer

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: The question of whether audiobooks count as "real" reading is genuinely contested. Some sources argue that listening to an audiobook is every bit as legitimate a form of reading as holding a physical book, pointing to brain science that shows narratives are processed equally whether read visually or aurally noting that oral storytelling has long been the original form of literature. Others, though, are skeptical: a recent NPR-Ipsos poll found that 41% of adults do not consider audiobooks to be reading some argue that the medium difference matters because authors intend their work to be read visually, not heard

### Sample conflictingqa_3c835387fe6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A study from The Australian National University further corroborated this, noting that the dragon was already large when it originated in Australia and only later crossed over to Indonesia before eventually becoming extinct there

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: One Tree Planted similarly concludes that real trees are more eco-friendly because they can have negligible or even negative emissions when recycled or kept growing, whereas artificial trees produce up to 40 kg of greenhouse gas emissions and are mostly made from plastic and metal

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: On the other hand, critics argue that trophy hunting can perpetuate wildlife killing and habitat destruction , that the IUCN's own research shows that 68% of trophy hunting revenue in Namibia goes to the state and only 22% to local communities, suggesting state capture rather than community benefit that the IUCN's claimed benefits are based on a narrow, conditional definition of 'well-managed' that does not reflect the broader reality of most trophy hunting operations

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: The evidence is mixed and the answer depends on the stage of CKD and the dose used

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: They make up roughly ten percent of the colony's population and have no corbiculae or scopae on their legs, abdomen thorax, meaning they do not collect pollen deliberately like female worker bees do

### Sample conflictingqa_613a0093714b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The first recorded use was in a 1651 poem Jonathan Swift popularized the modern version in 1738 , but whether Swift coined it or borrowed a pre-existing expression remains unclear

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Most products that claim to repair split ends can only temporarily mask the damage, smoothing the cuticle or adding weight to frayed ends, but these effects do not last beyond the next shampoo

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: A fair and complete answer would need to address all these perspectives and the underlying scriptural and historical arguments, which the available evidence does not collectively provide

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Additionally, the cosmic microwave background radiation and large-scale structure of the universe are both consistent with the existence of dark matter, which is estimated to make up approximately 27% of the universe's mass-energy density

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: In short, while individual birds contribute distinct sounds to the overall vocal repertoire of their species, the calls themselves are not necessarily unique to each individual

### Sample conflictingqa_962d8f5d5574

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological: sources differ in the scope of species sampled and the definition of'swim,' with some authors including any movement in water as swimming and others reserving the term for more active or efficient swimming behaviors

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: These opposing views reflect a methodological divergence in how'reverse discrimination' is defined and understood, with one side viewing affirmative action as a remedial tool and the other seeing it as a new form of discriminatory favoritism

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3
- **Claim**: Stalactites form through the downward growth of calcite crystals from water drips while underwater caves may seem to lack dripping water, the same process can occur through other means such as sea water flowing through a submerged cave's ceiling or side walls

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d5, d1
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because these sources present directly opposing conclusions on the same question of whether cold water makes hair shinier, with no single definitive experimental or expert consensus in the retrieved evidence

### Sample conflictingqa_c418fecfc1e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Some plants, such as shade-loving species like Chinese evergreen and cast iron plant, can tolerate low light or artificial light from regular bulbs and still thrive , while others like Orobanche (broomrape) have lost photosynthesis altogether and obtain nutrients parasitically

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A 1991 Gallup poll cited by the Oral Cancer Foundation found that Americans almost never think about death or think of it only occasionally, reflecting a broader cultural tendency to avoid the subject

### Sample conflictingqa_ece626a6cba9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological: differing interpretive frameworks for the same historical evidence yield opposing conclusions about the same event

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: d3, d4
- **Claim**: The conflict is interpretive and philosophical, rooted in differing definitions of'religion' and the nature of yoga itself

### Sample conflictingqa_f43b2c51deea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The evidence is further complicated by a 1988 study that found no correlation between pet disappearance notices in a newspaper and earthquake dates by the fact that animals can only detect the P wave a few seconds before an earthquake—too little time for people to take action

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: None
- **Claim**: However, other research has suggested that yerba mate may also possess anti-cancer properties — for example, a study found that yerba mate had a cytotoxic effect on cancer cells in lab experiments Dr. Axe notes that it has been scientifically shown to kill colon cancer cells — though these findings have not been conclusively confirmed in human trials

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d5, d1
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because these interpretive differences on the same issue reflect methodological and definitional disagreements across sources

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: This year's Passover began at sundown on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: She is described as the 'genius child' who overcame significant obstacles to achieve this milestone her award was recognized for outstanding contributions to the dynamics and geometry of Riemann surfaces and their moduli spaces

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: Earlier versions such as.NET Framework 4.8 and.NET Core 3.1 are outdated and no longer recommended for new development

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: A prior record held by Wikipedia's ancient DNA article identified 1 million-year-old mammoth DNA from Siberia as the oldest, but this has since been superseded by the 2022 Greenland discovery

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Academy Award for Best Picture was won by *Anora* (2025), directed by Sean Baker, at the 99th Academy Awards ceremony

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Nebula Award for Best Novel is 'The Dragonfly Gambit' (2025), as listed on BookBrowse's current and historical winners page

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: All available evidence is fully consistent, with no contradictions across any source

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 68

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The green anaconda (Eunectes murinus) is the heaviest reptile in the world, with a maximum recorded weight of 550 pounds in females

### Sample freshqa_c7ac9d61059a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The Tesla Model Y Premium All-Wheel Drive is priced at $51,630

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 9 minutes

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Slugs have one lung

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The current year's Ramadan is 2026, with the month expected to run through sundown on Wednesday, March 18

### Sample healthcontradict_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A 2019 meta-analysis of 17 RCTs reported that yoga significantly reduced the need for rescue inhalers and improved lung function in asthma patients , but a 2020 systematic review and meta-analysis found yoga had no significant effect on asthma symptoms or quality of life

### Sample healthcontradict_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Methodological differences across studies—such as variations in yoga style, population demographics outcome measures—likely contribute to these conflicting research outcomes

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The jingle was written by Pharrell Williams

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3, d4
- **Supporting Docs Found**: None
- **Claim**: Bartholdi's design was commissioned by French historian Édouard de Laboulaye to commemorate the upcoming U.S. centennial of independence and symbolize the friendship between France and the United States

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The Allies followed up their North African campaign by invading Sicily and Italy, then pushing into France and Germany — in other words, they crossed the Mediterranean and advanced into Europe

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Princess of Wales Theatre

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The gesture was also associated with the early Christian fish symbol (Ichthys), used to invoke protection and recognition it was not until later that the solo finger-crossing practice became widespread

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central blunt-ended lymphatic capillaries in intestinal villi responsible for absorbing dietary lipids

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: NASA's official page confirms Gagarin's achievement and identifies him as the first human in space, with his spacecraft completing 108 minutes of orbit around the Earth before returning safely

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: The Airdrome

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The 63rd and final volume of the Fairy Tail manga was published on January 23, 2018 the anime reached the end of its story in 2019, confirming the 2018–2019 timeframe as the correct window

### Sample qacc_9c2f95b14a78

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Research has shown that the Duluth Model produces fewer reports of violence and lower recidivism among participants compared to untreated offenders it is widely replicated across the country as the most used domestic violence intervention program

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This was the first time humans lived and worked on the station, marking its effective deployment into space

### Sample qacc_a1c73439eca9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Production on the season began in 2025, with the cast and crew working towards a July 2026 premiere

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: 245

### Sample qacc_aa94588b9477

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5, d2
- **Claim**: Over time, the Commonwealth Bank assumed increasing central banking responsibilities — including note issue from 1924 and formal central banking powers after World War II — before the Reserve Bank Act 1959 formally separated its commercial and central banking functions, renaming the Commonwealth Bank the Reserve Bank of Australia

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d4
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d4
- **Supporting Docs Found**: None
- **Claim**: This date, including the official U.S. Treasury Department's reference to New Mexico as the 47th state the Green Papers election database, which lists New Mexico as a state without a date, implying it was admitted prior to its appearance there

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: President Hoover and his wife watched from the West Terrace as firefighters battled the blaze, which was brought under control by around 10:30 PM

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Elton Hayes

### Sample qacc_cb5bcdb1ef9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: This distinction: Elliott Gould was the film version Wayne Rogers was the TV series version

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The 'Sc' classification indicates that the galaxy has a small central bulge and well-defined spiral arms, while the 'SBc' notation indicates that it also possesses a bar feature

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The balance sheet (also known as the statement of financial position or statement of financial condition) is the financial statement that involves all aspects of the accounting equation. It presents a company's total assets, total liabilities total equity at a specific point in time, directly representing the equation Assets = Liabilities + Equity. By showing the interrelationship between these three components, the balance sheet provides a comprehensive snapshot of a company's financial position, making it the central financial statement tied to the accounting equation

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_ecd3d9c0ca11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: An initialism is an abbreviation formed from the initial letters of a phrase or name is pronounced individual letters rather than as a word. Examples include DNA, RT-PCR FBI

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: 7

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These are the most authoritative sources for the two countries the answer depends on which country is being asked about

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d2
- **Supporting Docs Found**: None
- **Claim**: Other high-casualty nations include China (20 million total), Germany (6.6–8.8 million), Japan (2.6–3.1 million) Poland (5.6–5.8 million), reflecting the war's widespread devastation across Europe and Asia

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: grizzly bear

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: The Continental Army, led by General George Washington, was defeated by a British army of about 16,000 troops under General Sir William Howe near Chadds Ford in southeastern Pennsylvania, about 25 miles southwest of Philadelphia

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: This British victory opened the way for the conquest of Philadelphia, the American capital, which Howe occupied just two weeks later

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: These differences reflect the distinction between absolute output and standard of living: Nigeria's large population gives it the largest total GDP, while Seychelles' small population and high-value services sector drive the highest GDP per capita

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_40e6764f611f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: He belongs to the family Cheirogaleidae and is one of the smallest primates in the world, with over 20 species recognized

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: In the circle of fifths, moving up a fifth from B brings us to F#, so B Major is the key associated with five sharps

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7cd18101326e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1
- **Claim**: This result, with the 2026 Indiana Hoosiers' victory serving as the most recent data point in the tournament's history

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 km

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d3, d4
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This breakdown is confirmed by the California Energy Commission and is the most recent data cited in the retrieved evidence

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: It was a pivotal battle in which a Muslim army of approximately 313 fighters defeated a larger Meccan force of around 1,000 is considered one of the most important battles in Islamic history

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Shay Mitchell (Emily Fields) is 36 years old

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: 164

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: The retrieved evidence supports 1274 BC as the date of the Battle of Kadesh

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample situatedqa_temp_dcdb77c2f3d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Paris, France

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: The record for the most titles overall remains with Brad Pitt, who has been named Sexiest Man Alive four times, but Jonathan Bailey holds the most recent title and is the face of the 2025 edition

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Twitter is no longer called Twitter; it has been rebranded as X. This change was confirmed when Elon Musk terminated the Twitter deal in July 2025 and instead closed the acquisition of X Holdings, merging Twitter into X Corp. The platform's official name is now X, though it is still widely referred to as Twitter

### Sample wikirevision_0007

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Sundar Pichai serves as CEO of both Alphabet and Google Google's class A and C shares are listed on Nasdaq under the symbols GOOGL and GOOG respectively

### Sample wikirevision_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This ownership structure is further corroborated by LinkedIn's revenue reporting of $17.8 billion in 2025, filed with the U.S. Securities and Exchange Commission

### Sample wikirevision_0097

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Bangalore is still commonly used as a synonym for the city, but the official civic body of the city is called the Bruhat Bengaluru Mahanagara Palike (BBMP)

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The current Prime Minister of Canada is Mark Carney, who became the 24th Prime Minister on 14 March 2025. He was inaugurated in office on that date, serving as the head of government and the leader of the party with the greatest number of seats in the House of Commons

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has been in office since 23 May 2022. He is the 31st person to hold the role, serving as the leader of the Australian Government and the Cabinet of Australia, with the support of the majority of the House of Representatives

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Australia

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who defended his title at the 2025 French Open by defeating Jannik Sinner in the final to win his second French Open and fifth major. This is consistent across multiple sources, with the most recent information pointing to his 2025 victory


================================================================================

*Report generated by CATS v2.0*
