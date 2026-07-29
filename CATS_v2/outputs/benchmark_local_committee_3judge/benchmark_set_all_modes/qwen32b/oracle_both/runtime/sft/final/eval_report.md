# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 122 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.962 (over 736 samples)

**GR F1** *(used in CATS)*: 0.977

**Behavior Adherence**: 0.847 (over 614 applicable samples)

**Factual Grounding**: 0.871 (over 614 applicable samples)

**Single-Truth Recall**: 0.754 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.862

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.977
- **Precision**: 0.990
- **Recall**: 0.964
- **Accuracy**: 0.962
- TP=586, FP=6, FN=22, TN=122

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.847
- **Abstain Recall**: 0.953
- **Abstain F1**: 0.897
- **Specificity**: 0.964
- Abstain TP=122, FP=22, FN=6, TN=586


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (54 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.968 (n=157)
- **Grounding**: 0.930 (n=157)
- **Recall**: 0.838 (n=154)
- **CATS**: 0.931

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.941
- **GR F1** *(used in CATS)*: 0.962
- **Behavior**: 0.949 (n=178)
- **Grounding**: 0.815 (n=178)
- **Recall**: 0.676 (n=156)
- **CATS**: 0.851

### Type 3: Conflicting Opinions

- **Samples**: 109 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.680 (n=97)
- **Grounding**: 0.861 (n=97)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.840

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.724 (n=145)
- **Grounding**: 0.928 (n=145)
- **Recall**: 0.754 (n=140)
- **CATS**: 0.850

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.757 (n=37)
- **Grounding**: 0.703 (n=37)
- **Recall**: 0.730 (n=37)
- **CATS**: 0.775


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1865

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

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: Yes — palm oil production causes significant environmental harm through deforestation, habitat destruction, biodiversity loss, soil erosion air pollution

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Not really — chlorine doesn't turn hair green, but copper in pool water can

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Philosophers and researchers hold differing views on whether we can know anything beyond our minds; no settled scientific or logical consensus exists on this question

### Sample conflictingqa_35491baf4f4b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This exosphere is far less dense than Earth's atmosphere and is constantly replenished through processes such as solar wind ion-sputtering and meteorite impacts

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d4, d5
- **Claim**: It depends on whether the artificial tree is reused sufficiently; real trees are more sustainable if disposed of via recycling or composting, while artificial trees are environmentally harmful if discarded after short-term use

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: On-balance beneficial — IUCN and conservation scientists argue well-managed trophy hunting provides revenue and incentives to conserve wild populations and protect wildlife from poaching; however, critics contend trophy hunting is morally inappropriate and call for reform, while some experts like Amy Dickman argue blanket bans could actually increase animal killings

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are valuable and should be pursued, while others argue that software patents are controversial and may hinder innovation

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d1
- **Claim**: Yes

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: It depends on jurisdiction; in the U.S. federal law S.J.Res. 34 generally allows ISPs to sell browsing history without consent, though some states now require opt-in or opt-out rights other countries maintain separate rules

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: Some studies suggest a dose-dependent benefit—reducing the duration of symptoms by about half a day in adults—while others, particularly in healthy populations with few colds, show little to no preventive effect high-quality clinical reviews conclude that regular vitamin C supplementation does not meaningfully reduce the incidence of colds

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: In summary, while the overall consensus leans toward limiting saturated fats, particularly for those at high cardiovascular risk, the evidence is not universally conclusive some studies question the degree of risk attributable to saturated fat alone

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The answer depends on how efficiency is defined. Studies cited by McGill University found that organic systems can produce sufficient protein and calories with 27% less energy than conventional farming organic farming can also reduce environmental costs such as pesticide pollution and carbon emissions ; however, the main finding of the meta-analysis is that organic farming yields are generally 20-25% lower than conventional yields across most crop types , a conclusion echoed by the UN's Food and Agriculture Organization, which notes that organic farming's yield gaps make it difficult to scale while preserving ecosystem functions

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: In short, organic farming may be less efficient in terms of land use and yield, but more efficient in terms of energy use and environmental impact

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Multiculturalism's impact on unity is contested in the evidence: d2 argues it can hinder civic cohesion by reinforcing cultural affiliations that resist assimilation, while d3 and d5 present opposing research suggesting it may facilitate immigrant political integration and civic engagement when values are accepted

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Bird calls are not entirely unique to each individual bird, as many species share calls and respond to conspecific and even heterospecific signals

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A middle-ground approach is widely recommended: moderate green tea consumption (1–2 cups per day) is generally considered safe for most people, while those with a history of calcium oxalate stones may want to limit intake or choose low-oxalate beverage alternatives

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The weight of evidence suggests that neither option is clearly superior across all dimensions the most environmentally friendly choice remains refusing straws altogether when possible

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: The official military explanation attributes the Phoenix Lights to military flares dropped during a training exercise, but many witnesses and even former officials were not convinced by this account

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2b9ba7e192e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2, d5
- **Claim**: She had previously served as vice president under Castillo's administration and was next in line to assume the presidency

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d5
- **Supporting Docs Found**: None
- **Claim**: The site is now part of the White Sands Missile Range and is owned by the U.S. Department of Defense, with ground zero marked by a black lava rock obelisk

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: No chemical reaction between lead and any other element produces gold as a byproduct; the query is based on a misconception. Gold can be obtained from lead only through highly impractical nuclear transmutation experiments, not chemical reactions much of the gold produced via nuclear routes is radioactive and unsuitable for commercial use

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The official ceremony report further details that the film swept multiple categories, winning six Oscars in total, including Best Director and Best Adapted Screenplay for Anderson

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel was won by *When We Were Real* by Daryl Gregory at the 2025 awards. This result is corroborated across multiple sources, with the award ceremony and results page listing the novel as the winner, while related pages reference the same 2025 outcome. Older records showing only nominees or citing prior winners like 'Someone You Can Build a Nest In' (2024) are superseded by this 2025 update

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 9 September 2022

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: Jeff Bezos did not sell Amazon. He stepped down as CEO in 2021 but remained executive chairman later in 2025 he only sold shares worth approximately $736–863.5 million while retaining more than 900 million shares valued at close to $200 billion

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: A permanent cure for cancer has not been developed; the retrieved evidence indicates that researchers are still actively working toward that goal

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: However, a 2014 meta-analysis by Cramer et al. concluded that yoga cannot yet be considered a routine intervention for asthma patients, suggesting it may serve as an ancillary intervention or alternative to breathing exercises for those interested in complementary treatments

### Sample hotpotqa_0196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5, d4
- **Supporting Docs Found**: None
- **Claim**: John Speed's work on the 1610 map of Monmouth is further corroborated by additional records showing the town's topography and listed buildings during the same era

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Norma Koch

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Following the North African campaign, the Allies moved eastward across North Africa into Tunisia

### Sample qacc_1a764b8b6cf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This term is universally applied across all sources, including high-credibility references the reign is consistently identified as spanning 632–661 CE

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: These early fish were characterized by the presence of a vertebral column and pharyngeal slits, distinguishing them from earlier chordates such as lancelets and tunicates

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Among the earliest true vertebrates, the jawless fish (Agnathans) are recognized as the most primitive group, predating the evolution of jaws by approximately 50 million years

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d5
- **Claim**: As a player, Bill Russell holds the record with 11 NBA championships; as a coach, Phil Jackson holds the record with 11 rings — making them tied for the most in NBA history within their respective categories

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Canada's independence from Great Britain was an evolutionary process rather than a single moment scholars and sources differ on which milestone to emphasize. The Dominion of Canada was formed on July 1, 1867, marking the first major step toward self-government when the British North America Act established a federal union of Ontario, Quebec, Nova Scotia New Brunswick ; this date is celebrated as Canada Day and is often cited as the country's founding/independence day

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: October 1, 1968

### Sample qacc_883303a2d535

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8882ab46be5d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: 35

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a1c73439eca9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This tenth season begins filming

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane plays Carter Pewterschmidt, Lois's dad, on Family Guy

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Most effigy mounds were built between approximately 700 and 1200 A.D., with the majority constructed between A.D. 750 and 1050, making them some of the most recently built precontact earthen monuments in North America

### Sample qacc_d03e85bdc95a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: None
- **Claim**: It was flown by a hand-picked squadron under the command of U.S. Army Air Forces Colonel Paul Tibbets, Jr., who named the aircraft after his mother, Enola Gay Tibbets

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: 2023/24

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: ICD-10 codes have a minimum length of 3 characters and a maximum length of 7 characters

### Sample qacc_f2218f8c979e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: Prime rib comes from the rib primal, specifically the portion of the cow situated under the front section of the backbone between the fifth and sixth ribs and the twelfth and thirteenth ribs. It is cut from the same anatomical section as ribeye steak, spanning the area between the chuck and the loin

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: It depends on location; in the UK, it is illegal for children under 5 to drink alcohol those aged 16 can drink wine, beer cider with a meal at a licensed premises, while in the United States the minimum legal drinking age is 21

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4
- **Claim**: The welfare state was introduced at different times across nations, with scholars citing Germany's social insurance laws of the 1880s as a pioneering起点始于德国学者所称的1880年代的社会保险法律，然后是英国的自由改革（1906-1914），美国的社会保障于1935年建立，以及战后时期的全面巩固。

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4
- **Claim**: Scotland banned smoking in pubs on March 26, 2006, making it the first part of the UK to do so; Wales, Northern Ireland England followed the next year, with England's full ban taking effect on July 1, 2007. New York City banned smoking in enclosed workplaces including bars on March 30, 2003, while North Carolina enacted a statewide bar smoking ban on January 2, 2010. For other jurisdictions, including pub garden bans in the UK, the government is currently considering such a measure rather than having already implemented it

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Since 1965, the bulk of U.S. immigrants have come from Latin America (about half) and Asia (about a quarter), with Mexico alone accounting for roughly 25% of all immigrants

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: However, immigration patterns have evolved over time: in the 1840–1889 period, about 90% of immigrants came from Europe, while from 1890 to 1919 nearly 60% came from Italy, Austria-Hungary Russia-Poland

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Eisenhower

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features the California grizzly bear (Ursus arctos californicus), an extinct population of the brown bear

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: This cultural shift was further cemented in 1865, when Union soldiers returned home from the Civil War having consumed coffee as part of their standard rations, fully eclipsing hot tea in the United States by the 20th century, coffee dominated American daily life

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: A standard UNO deck now contains 112 cards, though the original base configuration had 108 cards

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: It was first made available on Google Pixel phones and has since rolled out to other manufacturers including Samsung

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances released into the blood when the heart is damaged or stressed they are used to diagnose and monitor heart disease

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Los Angeles (1932, 1984, 2028 Summer), Lake Placid (1932, 1980 Winter), Atlanta (1996 Summer), Palisades Tahoe (1960 Winter), St. Louis (1904 Summer) Salt Lake City (2002 Winter, 2034 Winter)

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 133

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: If you're looking for the most current information, consider checking McDonald's official promotion materials or recent news sources covering the current game period

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Simultaneously, inhalants displace oxygen in the lungs and central nervous system, causing users to lose consciousness and stop breathing

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India in the 2023 final by six wickets on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad. This was Australia's sixth Cricket World Cup title the 2023 tournament was the 13th edition, hosted across ten venues in India. Wikipedia's revision history confirms this update: an older March 2026 snapshot of the Cricket World Cup page listed India as the 2023 champion, but the more recent May 2026 revision corrected this to name Australia as the winner

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This name change was confirmed by the Government of Haryana in 2016 the city is now officially known by this new name. As a result, Gurgaon is no longer the official name of the city


================================================================================

*Report generated by CATS v2.0*
