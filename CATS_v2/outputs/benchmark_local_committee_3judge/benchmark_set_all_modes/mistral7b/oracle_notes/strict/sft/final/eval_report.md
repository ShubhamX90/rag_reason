# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 126 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.921 (over 736 samples)

**GR F1** *(used in CATS)*: 0.950

**Behavior Adherence**: 0.793 (over 610 applicable samples)

**Factual Grounding**: 0.811 (over 610 applicable samples)

**Single-Truth Recall**: 0.638 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.798

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.950
- **Precision**: 0.996
- **Recall**: 0.908
- **Accuracy**: 0.921
- TP=552, FP=2, FN=56, TN=126

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.692
- **Abstain Recall**: 0.984
- **Abstain F1**: 0.813
- **Specificity**: 0.908
- Abstain TP=126, FP=56, FN=2, TN=552


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.903 (n=155)
- **Grounding**: 0.924 (n=155)
- **Recall**: 0.782 (n=154)
- **CATS**: 0.900

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.855
- **GR F1** *(used in CATS)*: 0.901
- **Behavior**: 0.921 (n=177)
- **Grounding**: 0.702 (n=177)
- **Recall**: 0.513 (n=156)
- **CATS**: 0.759

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.625 (n=96)
- **Grounding**: 0.832 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.812

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.949
- **GR F1** *(used in CATS)*: 0.972
- **Behavior**: 0.669 (n=145)
- **Grounding**: 0.864 (n=145)
- **Recall**: 0.664 (n=140)
- **CATS**: 0.792

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.703
- **GR F1** *(used in CATS)*: 0.825
- **Behavior**: 0.649 (n=37)
- **Grounding**: 0.595 (n=37)
- **Recall**: 0.459 (n=37)
- **CATS**: 0.632


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2108

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

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Designers should also consider registering their work under the Design Piracy Proposal Act (DPPA) or seeking trademark protection for logos, brand names distinctive visual elements, as these provide separate and complementary forms of protection

### Sample conflictingqa_07e290a954ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d3, d2
- **Supporting Docs Found**: None
- **Claim**: Additional sources note that St. John's wort may interact with other medications and can have side effects such as dry mouth, dizziness photosensitivity its safety for certain populations, including pregnant women, is not fully established

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: However, iodine deficiency is also associated with thyroid dysfunction adequate iodine intake is considered essential for thyroid hormone synthesis, meaning the appropriate iodine level is crucial

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d4
- **Claim**: It is estimated to be over 2,000 years old and consists of both above-ground mushrooms and a vast underground network of mycelia

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Overall, while artificial sweeteners are widely used and FDA-approved, the evidence remains mixed and the safety of long-term use in diabetic individuals continues to be evaluated

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d1, d4
- **Claim**: However, the more common agricultural and anatomical consensus is that cows have four stomachs, with each compartment performing a specific digestive role — such as fermenting grass in the rumen or regurgitating cud in the reticulum — before the final digestion of food in the abomasum

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: This methodological divergence in defining 'first' versus subsequent radiation produces conflicting research outcomes the evidence as a whole suggests that the Ordovician–Silurian transition was a critical period for terrestrial life but does not resolve the exact timing of the very first land plants

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2
- **Claim**: The retrieved evidence is only partially supportive and does not supply a complete answer

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3, d2
- **Claim**: Taken together, the available evidence points to epigenetic inheritance being a real but incompletely understood phenomenon, with the strongest evidence pointing to specific transgenerational epigenetic markers rather than general epigenetic information

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, as they have a lower carbon footprint and support forest conservation efforts

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d2
- **Supporting Docs Found**: None
- **Claim**: On the other hand, critics — including some scientists — argue that the industry is morally inappropriate, that its economic benefits are overstated that a ban would not be detrimental to wildlife on balance , while some field evidence suggests trophy hunting can contribute to animal cruelty and ecological disruption

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Overall, the evidence does not support a definitive yes/no conclusion — rather than being a panacea, trophy hunting appears to work only in certain well-managed contexts the broader debate around its conservation benefits remains unresolved

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: The most reliable conclusion from the available evidence is that while full moons may influence the probability of large, specific events, they do not appear to cause more earthquakes generally

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: However, other high-credibility sources caution that taking large amounts of vitamin C does not constitute proven prevention and can even increase the risk of kidney stones in certain populations , underscoring the need for individualized advice from a healthcare provider before starting any supplement regimen

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d4
- **Claim**: However, organic farming does offer environmental and health benefits — it produces fewer pollutants, causes less soil loss uses less water — making it a more sustainable long-term choice, even if it is not the most efficient in terms of raw yield

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Religious truth claims are inherently theological opinions rather than historically verifiable facts, so the question of whether the Catholic Church is the true church is properly understood as a theological debate rather than a historical proof

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: In practice, whether a snake can swim depends heavily on species and context the available evidence reflects a genuine scientific gap rather than a definitive yes/no answer

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2
- **Claim**: Contrary to a common misconception, stalactites do not require dripping water in dry air to grow — instead, they initiate as soda straws where water flow deposits calcite crystals along their length this process can continue underwater once the structure has initiated

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: The PETM onset coincides with a mercury low, suggesting at least one other carbon reservoir released significant greenhouse gases in response to initial warming some models propose that methane releases from organic-rich permafrost or methane-rich ocean sediments could have contributed to the warming , indicating ongoing scientific debate about the precise mechanism

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d2
- **Supporting Docs Found**: None
- **Claim**: The consensus is that any shine effect is negligible that the use of conditioners and styling products containing silicones and oils remains the best way to achieve shiny hair

### Sample conflictingqa_c34991d9897e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d2
- **Supporting Docs Found**: None
- **Claim**: For broader protection, many businesses also register their logos as trademarks, which provide exclusive rights to use the mark for specific goods or services and do not depend on proving copying

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Religious and philosophical views differ; science has not established the Bible as historically or scientifically infallible

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d2
- **Claim**: Relativism; epistemological paradoxes; Gettier-style counterexamples; Bayesian epistemology

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: Folklore states that Macbeth was cursed from the beginning: a coven of witches objected to Shakespeare's use of real incantations in the play the actor playing Lady Macbeth died during the first performance around 1606

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5, d2
- **Supporting Docs Found**: None
- **Claim**: Some sources also attribute the play's notorious accident record to this curse, noting that the Astor Place Riot in 1849 resulted in at least 20 deaths the 1937 Old Vic production was plagued by injuries and a fire

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, some research suggests yerba mate may also have anti-cancer properties, with cytotoxic effects observed on cancer cells in laboratory studies, though human clinical evidence to confirm these benefits is currently lacking

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1, d2
- **Claim**: Overall, while the available evidence points to an association between yerba mate use and cancer risk under specific conditions—particularly when consumed at very high temperatures—the evidence does not establish definitive causation the substance's overall effect may depend on how it is prepared and consumed

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: A specific black hole, located approximately 1,560 light-years away in the constellation Virgo, has been directly imaged using a network of radio telescopes, appearing as a black circle surrounded by a thick, orange ring of material

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: Religion is contextually defined Mormons consider themselves Christians — but their definition may differ from that of historical Christianity

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2d7eb41139aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d5, d3, d4
- **Claim**: The first atomic bomb test in the United States took place at the Trinity Site in New Mexico on July 16, 1945. The test, codenamed 'Trinity,' was conducted at a site 210 miles south of Los Alamos on the Alamogordo Bombing Range, in the New Mexican desert. This was the world's first nuclear explosion the site is now part of the White Sands Missile Range

### Sample freshqa_39dcd7b38c39

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3, d5
- **Claim**: This is consistently confirmed across multiple sources, with the coin program featuring twenty-four notable women over four years Angelou being selected as the first honoree

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: Portugal won the 2017 Eurovision Song Contest, marking the country's first victory since 1964. The contest took place in Kyiv, Ukraine Portugal's winner was Salvador Sobral, who performed the song "Amar pelos dois"

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This figure is corroborated by Costco's own website, which states that Executive Members earn an annual 2% reward on qualified purchases up to $1,250

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The film, directed by Paul Thomas Anderson, also won Best Director and Best Adapted Screenplay, making it a major sweep of multiple categories

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d3, d4
- **Supporting Docs Found**: None
- **Claim**: This result, superseding earlier reports that listed *CODA* (2022) or *Sinners* (2024) as the most recent winners, as those awards have since been surpassed by the 2026 ceremony

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: Bayonne is a city in New Jersey Martin has often spoken about growing up there, noting that his childhood world was limited to First Street to Fifth Street between his grade school and home

### Sample freshqa_8eca5bd62ae0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The 2022 Winter Olympics were the first ever to be held in China the country used the experience gained from the 2008 Games to ensure a seamless event

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel was won by *When We Were Real* by Daryl Gregory, published by Saga Press. The award was announced at the 2025 Nebula Conference, making this the most recent recognition for a novel published in 2025

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: A boating accident in Chesapeake Bay on July 28, 1971

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: Queen Elizabeth II of England died on 8 September 2022

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c7ac9d61059a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4, d1, d2
- **Claim**: Additional sources report slightly different figures due to destination charges and optional equipment, but the core pricing structure remains the same across all sources

### Sample freshqa_cf331ed7d09f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This update introduced new features such as improved performance, enhanced security optimizations for Apple's latest hardware

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample healthcontradict_0060

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: A multi-institutional study of 114,000 children found that tepid water sponging did not lower fever any more than keeping children dressed

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Instead, parents should monitor their child's temperature and use acetaminophen or ibuprofen as directed by a healthcare provider

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Additional research has reported yoga's effectiveness in reducing symptoms and improving exercise capacity in specific subgroups, such as children with exercise-induced bronchoconstriction some sources suggest yoga may help reduce the need for medication ; however, these findings are limited to certain populations and specific outcomes the overall scientific consensus remains inconclusive

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d8
- **Claim**: Lucas di Grassi

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d7, d4, d5, d6, d1
- **Claim**: Pusha T

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b75ed799d46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: The ceremony streamed live on Netflix at 8:00 p.m. ET/5:00 p.m. PT, hosted by Kristen Bell for the third time

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Madhuri Dixit

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_213701765f94

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The landing was confirmed by multiple sources, including the National Transportation Safety Board and the Federal Aviation Administration

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The tree was described as a 'festive centrepiece' at her 1800 party the custom of decorating trees had previously been practised in her native Germany

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, individual travelers should verify the specific entry rules and requirements for their intended destinations, as visa policies and eligibility criteria vary by country

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: October 1, 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nana is an Australian Shepherd

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: The ISS had been under development since the mid-1980s the specific 1998 launch date marked the transition from the Russian Mir space station to the new international collaboration

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912, when President William Taft signed the New Mexico statehood bill into law

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5
- **Claim**: The territory had sought statehood since 1850 the signing ceremony took place in the Palace of Governors in Santa Fe

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: California's Mojave Desert (near Parker, Arizona and Vidal Junction)

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d2
- **Supporting Docs Found**: None
- **Claim**: This joint allows for movement in two planes — flexion, extension, abduction circumduction — and is crucial for sound transmission from the eardrum to the inner ear

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Most effigy mounds were built between 700 and 1200 A.D., with the majority constructed during the Woodland period (approximately 750–1050 A.D.)

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple authoritative sources, including the official Social Security Administration website and the St. Louis Federal Reserve Economic Data (FRED)

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This means it possesses a central bulge surrounded by spiral arms, with the bulge extending beyond the plane of the disk

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Additionally, federal highways with tolled sections are identified by the suffix 'D' (e.g., Fed

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Their most recent appearance in the competition was the 2025–26 season , which saw them finish second in their group and reach the play-off round before being eliminated by Villarreal

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells played the character Wez in Mad Max 2: The Road Warrior

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: 407,000

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_897e47478bbc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that within this total, there are approximately 593,615 inhabited villages, with the remaining villages being uninhabited

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: In areas with mixed federal and local ownership, responsibility for maintenance may be jointly held by both the USACE and the local entity funding for levee repairs may come from a combination of federal, state local sources

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: President Kennedy was the first to send a significant number of military advisers to Vietnam, authorizing the deployment of 16,000 advisers in 1961

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features the California grizzly bear, which is an extinct animal. The flag's design originated in 1846, when California was part of Mexico the bear became a symbol of the Bear Flag Republic — a short-lived attempt by U.S. settlers to break away from Mexico. The bear on the flag is depicted as a grizzly bear, not a polar bear or any other type of bear

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Scotland (defeated England 25-13 at Murrayfield on 2 February 2018)

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: He is a senior BJP leader and has held the position since 2019, consistently appearing in the Cabinet listings of the official Wikipedia page for the Ministry of Law and Justice

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: However, the Articles proved to be too weak to address the needs of a growing nation — such as raising armies, collecting taxes regulating commerce — leading the states to convene a constitutional convention in Philadelphia in 1787, which produced the U.S. Constitution

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: England are the current champions after winning the 2019 World Cup, with Australia, India Pakistan being the other recent winners

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This victory gave the Blues a 2-1 edge in the 2024 series, after Queensland had won the first game 18-6

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Mort is a Goodman's mouse lemur, a small primate native to Madagascar, though a 2014 spin-off series reveals he is also part bear, spider starfish

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Argentina defeated France 4-2 in the final held in Lusail Stadium in Qatar, claiming their third title

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The next Avatar comic is *Avatar: The Last Airbender — Kyoshi Warriors*, a three-issue series written by Brandon Hoáng with art by BellBessa, Xanthe Bouma lettered by Jimmy Betancourt

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: PTI's victory gave Imran Khan his first term as Prime Minister, after which he promised to investigate the allegations of electoral wrongdoing

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 km

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d5, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d4
- **Supporting Docs Found**: None
- **Claim**: Some sources cite 1532 as the year of the Spanish conquest and Atahualpa's death , while others place it in 1533 after additional military resistance ; these variations reflect the gradual consolidation of Spanish rule rather than factual disagreement

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d1
- **Claim**: It was originally formed from the Old German name Gerhard, which itself consisted of the elements gēr ('spear') and hard ('hardy, brave strong')

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Another theory suggests that the surname may have originated as a patronymic name formed by adding the suffix -ard to the personal name Gerard

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d5, d2
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin & Perry Go Large

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2
- **Claim**: Older sources citing higher populations reflect outdated data, as the island has experienced consistent population growth over the past decade

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Argentina (defeated Italy 3–2 in a penalty shootout at the Rose Bowl, Pasadena, California)

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Argentina defeated France 4–2 on penalties after extra time in the final, securing their first World Cup title since 1986

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the latest ICC Men's Cricket World Cup, the 2023 tournament hosted in India

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed by the Haryana government's official gazette notification in April 2023, superseding the older name of Gurgaon. The change is consistently reflected across all sources, with the newer Wikipedia revision of Gurgaon explicitly stating 'Gurugram' as the official name the older revision also confirming the same fact

### Sample wikirevision_0162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d2
- **Supporting Docs Found**: None
- **Claim**: The current FIFA World Cup champion is Argentina, who won the 2026 tournament, defeating Italy 3-0 in the final held at the Lusail Stadium in Qatar on December 18, 2022

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d4
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who won the 2026 French Open by defeating Jannik Sinner in the final


================================================================================

*Report generated by CATS v2.0*
