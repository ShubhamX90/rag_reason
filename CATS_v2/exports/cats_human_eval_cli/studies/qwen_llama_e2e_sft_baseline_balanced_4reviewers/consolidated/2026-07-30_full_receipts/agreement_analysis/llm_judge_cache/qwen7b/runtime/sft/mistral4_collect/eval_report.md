# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 121 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.966 (over 736 samples)

**GR F1** *(used in CATS)*: 0.979

**Behavior Adherence**: 0.641 (over 615 applicable samples)

**Factual Grounding**: 0.867 (over 615 applicable samples)

**Single-Truth Recall**: 0.532 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.755

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.979
- **Precision**: 0.988
- **Recall**: 0.970
- **Accuracy**: 0.966
- TP=590, FP=7, FN=18, TN=121

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.871
- **Abstain Recall**: 0.945
- **Abstain F1**: 0.906
- **Specificity**: 0.970
- Abstain TP=121, FP=18, FN=7, TN=590


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (53 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.967
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.835 (n=158)
- **Grounding**: 0.926 (n=158)
- **Recall**: 0.500 (n=154)
- **CATS**: 0.810

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.964
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.719 (n=178)
- **Grounding**: 0.854 (n=178)
- **Recall**: 0.500 (n=156)
- **CATS**: 0.763

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.995
- **Behavior**: 0.542 (n=96)
- **Grounding**: 0.874 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.804

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.486 (n=146)
- **Grounding**: 0.854 (n=146)
- **Recall**: 0.671 (n=140)
- **CATS**: 0.749

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.297 (n=37)
- **Grounding**: 0.712 (n=37)
- **Recall**: 0.270 (n=37)
- **CATS**: 0.552


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2047

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/mistral-small-4

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2047
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_04e1627e9fc5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: Fertilizer type further modulates these effects—organic fertilizers tend to increase free-living nematode populations and nutrient availability, whereas inorganic fertilizers can negatively affect nematode communities, particularly plant-parasitic species

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: Lawrence Ferlinghetti, Ginsberg's publisher, further framed the legal victory as a protection of First Amendment rights, noting that the court case "paved the way by setting a precedent for freedom of speech in art"

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: Yes — anime is a specific style of cartoon, originating in Japan with distinct artistic and narrative characteristics

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: It is worth noting that some scholars and commentators further distinguish between religious observance and ethnic identity, recognizing that many Jews are irreligious or post-denominational that Jewish cultural practices vary significantly across geographic and demographic subgroups

### Sample conflictingqa_114c06976f62

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The religion's own official statement asserts it is "as legitimate as any other," though its primary origin as a satirical protest against intelligent design creates ongoing debate about whether it qualifies as a serious faith

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: However, research specifically focused on people with diabetes presents a more nuanced picture: a large cohort study found that high intake of artificial sweeteners was associated with higher all-cause mortality, cardiovascular risk cancer risk in diabetic patients, while lower intake was not consistently linked to harm

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d3
- **Supporting Docs Found**: None
- **Claim**: However, it is worth noting that palm oil is not inherently evil — it is highly profitable and widely used due to its high yield some certified sustainable palm oil (via RSPO standards) can help mitigate the worst environmental harms

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Proponents of responsible breeding argue that it is necessary to maintain unique dog breeds and that ethical practices minimize harm through stringent health screening and humane treatment of breeding dogs. Opponents contend that breeding inherently exploits animals for profit, causes widespread inherited health conditions contributes to pet overpopulation

### Sample conflictingqa_34fef928d452

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: The scientific consensus, supported by multiple studies, is that Archaeopteryx was indeed a capable flyer, though its exact mode of flight — whether gliding or powered by active muscle contraction — remains an area of ongoing scholarly debate .

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: In cases where data is extremely limited, specialized techniques such as transfer learning or active learning can help bridge the gap, though these approaches add additional computational and engineering challenges

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: Some researchers further note that emojis function more like internet slang or emoticons — as convenient pictorial abbreviations — and are not universally recognized as a separate linguistic medium

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The question of whether patents should apply to software is actively debated. A high-credibility source notes that while software patents are technically patentable under U.S. law, recent Supreme Court rulings make them difficult to enforce, leading many inventors to conclude that the costs outweigh the benefits

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Some clinical trials have shown bicarbonate may reduce urinary fibrotic biomarkers and preserve eGFR in early-stage CKD, particularly in patients with hypertension nephropathy, while higher-dose regimens have been associated with reduced potassium concentrations and only a catabolic state improvement in some reports

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: Single 'r' sounds in the middle of words are typically produced with a soft tap (similar to the English 'd' in 'butter'), while double 'rr' requires a strong trill

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d1
- **Supporting Docs Found**: None
- **Claim**: However, some states have pushed back against this: Maine passed a law prohibiting ISPs from selling personal data without express individual permission the California Consumer Privacy Act (CCPA) grants California residents the right to opt out of having their data sold

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: The exact threshold depends on the intensity of the rain and the individual bee's circumstances — for example, bees in wetter climates may be more tolerant of rain than those in drier areas

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d3, d2
- **Claim**: Spelunking and caving are traditionally understood as distinct activities: caving is the broader term encompassing professional exploration, scientific study supervised recreational tours, while spelunking specifically refers to casual, often unsupervised cave exploration — essentially caving done by hobbyists or unprepared individuals

### Sample conflictingqa_894f4a4b9552

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Yes — dark matter is inferred to exist because galaxies rotate too fast for their visible matter its gravitational effects are observed in strong and weak lensing of distant light, gravitational lensing large-scale structure formation; it makes up roughly 85–90% of the total matter in the universe

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Research has further shown that birds' vocal frequencies are influenced by their habitat, body mass beak shape, adding another layer of complexity to how individual birds produce sounds

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d5
- **Claim**: Proponents of the idea that fish do feel pain point to the fact that they have pain receptors (nociceptors) that are activated by noxious stimuli that they exhibit behavioral responses such as attempting to avoid harmful situations

### Sample conflictingqa_9261438d6ee2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Research with Rainbow Trout has shown that injecting bee venom or acid into their lips produces a nervous response and abnormal behavior that returns to normal after administration of the painkiller morphine, suggesting a pain experience comparable to that of mammals

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Underwater stalactites are possible but require specific conditions — they form when water pressure pushes calcite-laden water up through a hollow stalactite structure and then deposits the calcite radially, rather than the typical dripping process — and are rarely observed

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: Merriam-Webster further notes that the one-word spelling is first recorded in the late 19th century and carries no logical grammatical objection, though it remains nonstandard in formal contexts

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Wikipedia notes that 'all the hard evidence suggests there was no mass exodus,' and that newspapers likely exaggerated the scale of the panic to discredit radio as a news source, while some witnesses reported thousands of people fleeing their homes in genuine terror

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Foundationalist accounts attempt to resolve the regress by positing basic beliefs justified through experience or sensory evidence, though critics reject this as implausible

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d1
- **Claim**: However, some studies caution that these differences are not uniform across all contexts — organic farming can nearly rival conventional yields in specific warm temperate regions and for certain crop types, suggesting the yield gap is largely conditional rather than absolute

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: The play has been plagued with accidents ever since — including the 1937 production at the Old Vic where Laurence Olivier nearly lost his life when a 25-pound stage weight crashed near him

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: However, the evidence does not confirm that the Dutch were the sole or definitive discoverers of Australia, as other European powers would later challenge their claims

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some research offers a more nuanced picture: a laboratory study found that yerba mate exhibits a cytotoxic effect on cancer cells, suggesting it may possess anti-cancer properties, though these findings do not translate to a proven preventive or therapeutic role in humans

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d5
- **Claim**: No, Brontosaurus and Apatosaurus were not the same dinosaur

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: While the original promotional plan was for 50,000 attendees, tens of thousands more showed up, creating a spontaneous, inclusive moment that transcended its initial scope

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: Aryna Sabalenka (2025 US Open women's singles champion); Jessica Pegula (2024 US Open women's singles finalist)

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d4, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani (2014 — first and only female recipient; passed away 2017)

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: In summary, the latest major version of .NET depends on which edition (Framework, Core Mono) you are considering

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Season 1 premiered on November 12, 2019, Season 2 on October 30, 2020 Season 3 on March 1, 2023

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5eb89aae15f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d2
- **Claim**: Kantara (Rishab Shetty, 2022)

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: This is corroborated by the official White House history, which confirms his presidency spans from January 20, 2025 to January 20, 2029

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, older sources may reflect Joe Biden's prior presidency (2021–2025) some contemporary sources still reference Biden's tenure as recently as 2026 , underscoring that the most current information places Trump as the incumbent

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has not won the Ballon d'Or

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: When We Were Real by Daryl Gregory (Saga), published by Harper Voyager US and Harper Voyager UK

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: His passing dealt a significant blow to the emerging field of artificial intelligence, as Rosenblatt was the primary defender of the perceptron concept without his advocacy, funding dried up for over a decade

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Her output has steadily grown over her decade-long career, with her novels continuing to appear on the New York Times bestseller lists

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: J June 2025

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f11e8d758fc6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Some sources also point out that certain aquatic and marine slug families have evolved gill-like structures within their lungs, further illustrating the remarkable respiratory adaptations of these creatures

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: Pfizer's vaccine is approved for people ages 5 and older Novavax's vaccine is authorized for ages 12 and older

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: In addition, the Washington State Department of Health notes that pediatric vaccine studies follow patients' health up to two years after vaccination the CDC recommends that children ages 6 months to 17 years may receive the vaccine based on parent preference and clinical judgment

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: These divergent findings reflect methodological differences between a single RCT and a broader meta-analysis the weight of clinical opinion remains unsettled

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0192

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d6, d7, d4, d5, d8
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not support a confident answer to the query

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d4, d1, d5
- **Claim**: Sakshi Malik (Haryana); Madhuri Dixit (India); Parineeti Chopra (Haryana); Bhawna Dehariya and her daughter Siddhi Mishra (Madhya Pradesh); Avani Lekhara (Rajasthan)

### Sample qacc_15ffab2466f7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: For the latest or upcoming productions, Toronto-based ticketing platforms such as SeatGeek and Ticketmaster are useful resources to check current availability across various venues

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: As a coach: Red Auerbach (16)
As a player: Bill Russell (11)
Combined: Phil Jackson (11 as a player + 11 as a coach)

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Canada did not gain full independence on a single date; rather, it was achieved gradually through key milestones

### Sample qacc_6837d86d03ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5
- **Claim**: Following Prince William are his children: Prince George of Wales (2nd), Princess Charlotte of Wales (3rd) Prince Louis of Wales (4th)

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Australian Shepherd

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a44267c115d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: It is worth noting that different organs hold water in varying amounts — for example, the brain and heart are about 75% water muscles and kidneys are around 79% — but the majority of the body's water is still sequestered within the cellular structure

### Sample qacc_aa94588b9477

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: The bank gradually assumed central banking functions over the following decades these responsibilities were formalized under the Commonwealth Bank Act 1945

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Syndicated internationally; in the US, streamed on Paramount+

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_bfbb5f55a63f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Timothy Talbott

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Roger Miller

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: The First Epistle of John was likely written between 95 and 110 C.E., though some scholars place the composition as early as 85–90 C.E

### Sample situatedqa_geo_4cb699778b59

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Current average Senate service stands at approximately 11.2 years as of the 119th Congress (2025–2026), though this reflects cumulative service duration rather than the official statutory term length, which remains six years

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: More broadly, Wikipedia notes that the main types of modern political systems include democracies, totalitarian regimes authoritarian regimes, with democracies being one of the primary categories under which the U.S. falls

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: 2018

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The switch from tea to coffee in the United States was a gradual process tied to historical events rather than a single definitive date

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The Philadelphia Eagles have won the Super Bowl multiple times. The most recent was Super Bowl LIX (59th) in 2025, when they defeated the Kansas City Chiefs. Earlier, the Eagles won Super Bowl LII (52nd) in 2018, earning their first NFL title since 1960 added another championship with a 40-22 victory over the Chiefs in Super Bowl LVII (57th) in 2023

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: New South Wales last won the State of Origin series in 2021

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4
- **Claim**: 1980 (established as a national park); previously designated as a national monument in 1978

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d2
- **Claim**: Since the battle itself is the primary focus of the query, I will highlight the most direct examples from the provided evidence:
- “Corpse-maker” (21, lines 286)
- “Shadow-stalker” (47, lines 704)
- “Shepherd of evil” (a40)
- “Whale-road” for the sea
- “Bone-house” for the human body (though not specific to the battle itself)
- “Sea-wood” for ‘ship’ (used in the journey to Grendel’s lair)

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: For context, this places Australia among the top-ranked countries globally for coastline length, alongside nations like the United States (approximately 19,924 kilometres) and New Zealand (15,134 kilometres)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d4, d3, d2
- **Claim**: These discrepancies are typical of religious organizations experiencing growth, as successive membership tallies supersede prior records without a single authoritative source for all time periods

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Tsinghua University further clarifies that the revolution was driven by intellectuals and students demanding constitutional reforms, who were inspired by the 1905 Russian Revolution

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Shay Mitchell was 23 when she portrayed 16-year-old Emily Fields in the show's pilot episode, meaning Emily's fictional age differed from her real age throughout the series

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 670–680 nm

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: The United States has hosted the Olympics nine times: four Summer Games and five Winter Games

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The first U.S. Olympics were the 1904 Summer Games in St. Louis, Missouri, which were actually part of the St. Louis World's Fair

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: 2020

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, newer 2025 data references suggest India's standing has improved further, with the 2025 index showing only partial rankings for some categories

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Over time, the name spread to other regions including England, where it is associated with the family of Gerard (died 21 May 1108), who served as Lord Chancellor of England and later became Archbishop of York

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc. and is structured as a wholly owned subsidiary of Alphabet

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The President of Germany is Frank-Walter Steinmeier, who has served as the Federal President of the Federal Republic of Germany since taking office on that date. This is confirmed by the official Wikipedia entries on both the President of Germany and the Federal President, which list his incumbency from 19 March 2017. Since German reunification in 1990, the President has been the head of state for all of Germany, serving as the Bundespräsident der Bundesrepublik Deutschland

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: Jannik Sinner (most recent data available indicates he is the champion, though the tournament has since taken place and he may have been succeeded)

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Claudia Sheinbaum Pardo has been serving as the 66th President of Mexico, making her the country's latest head of state and government. This is confirmed by the official Wikipedia revision that superseded the older version in April 2025, which explicitly names her as incumbent since her election. She is the first woman and the first Jewish person to hold the office, having previously served as Head of Government of Mexico City from 2018 to 2023

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The current President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022. He is the 16th President of the Philippines and serves as both head of state and head of government

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of India is Droupadi Murmu, who took office on 20 July 2022. She is the 15th and current President of India, serving as head of state and supreme commander of the Indian Armed Forces. While the older Wikipedia revision (2025) also confirmed her presidency, the newer 2026 revision consistently reinforces that Droupadi Murmu is indeed the current President of India

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: The men's singles champion is **Carlos Alcaraz**, who defended his title by defeating Jannik Sinner in the final. This result is corroborated by Wikipedia's page on the 2026 French Open, which notes that two-time defending champion Carlos Alcaraz withdrew before the tournament began due to a wrist injury


================================================================================

*Report generated by CATS v2.0*
