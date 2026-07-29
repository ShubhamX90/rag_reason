# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 120 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.963 (over 736 samples)

**GR F1** *(used in CATS)*: 0.978

**Behavior Adherence**: 0.807 (over 616 applicable samples)

**Factual Grounding**: 0.880 (over 616 applicable samples)

**Single-Truth Recall**: 0.743 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.852

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.978
- **Precision**: 0.987
- **Recall**: 0.969
- **Accuracy**: 0.963
- TP=589, FP=8, FN=19, TN=120

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.863
- **Abstain Recall**: 0.938
- **Abstain F1**: 0.899
- **Specificity**: 0.969
- Abstain TP=120, FP=19, FN=8, TN=589


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (52 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.981
- **Behavior**: 0.956 (n=159)
- **Grounding**: 0.949 (n=159)
- **Recall**: 0.828 (n=154)
- **CATS**: 0.928

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.937
- **GR F1** *(used in CATS)*: 0.959
- **Behavior**: 0.927 (n=178)
- **Grounding**: 0.801 (n=178)
- **Recall**: 0.670 (n=156)
- **CATS**: 0.839

### Type 3: Conflicting Opinions

- **Samples**: 109 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.546 (n=97)
- **Grounding**: 0.866 (n=97)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.797

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.697 (n=145)
- **Grounding**: 0.931 (n=145)
- **Recall**: 0.750 (n=140)
- **CATS**: 0.844

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.802 (n=37)
- **Recall**: 0.676 (n=37)
- **CATS**: 0.784


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

### Sample conflictingqa_05b33f4ca156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: These toxins can affect the nervous system, leading to symptoms such as numbness, dizziness muscle weakness the severity varies by species

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Partially — peeling reduces dietary fiber, vitamin C antioxidant flavonoids, but does not decrease vitamin C or other vitamin amounts per 100 grams

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: Yes — palm oil production causes significant environmental harm through deforestation, habitat destruction pollution

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Dog breeding is not universally considered unethical, but some practices are

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Cows do not have four stomachs; technically, they have one stomach that is divided into four distinct compartments: the rumen, reticulum, omasum abomasum

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: Not directly from chlorine — the green color is caused by copper (from algaecides) bonding with chlorine and attaching to hair proteins; chlorine alone bleaches hair lighter but doesn't turn it green

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: It depends on usage duration: real trees are more sustainable if the artificial tree is used for fewer than about 20 years; artificial trees become more sustainable only if reused for 20+ years

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Overall, the debate remains unresolved, with opposing factions weighing the financial incentives against ethical and welfare concerns

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are valuable and should be pursued, while others argue that software patents are controversial and may hinder innovation

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2
- **Claim**: However, the available sources do not provide a definitive global death toll or an explicit comparison with all other historical eruptions to conclusively confirm it was the single deadliest eruption of all time

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: It depends on jurisdiction; in the U.S. federal law S.J.Res. 34 generally allows ISPs to sell browsing history without consent, though some states now require opt-in permission

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Not all bird calls are unique to individuals; many are shared at the species or family level

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, the American Academy of Orthopaedic Surgeons notes that knee braces are categorized as prophylactic, functional rehabilitative, each serving distinct but limited roles in injury prevention and management it is generally advised that individuals consult a healthcare provider before relying on a knee brace as a substitute for professional care

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The underlying debate thus centers on whether nociception (the physiological detection of harmful stimuli) should be equated with pain (the subjective experience of that detection), a question that remains unresolved in the scientific community

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5
- **Claim**: The environmental comparison between paper and plastic straws depends on the metric considered. Paper straws are generally biodegradable and compostable, which is an advantage over plastic straws that persist in the environment for centuries; one analysis found that plastic straws break down slowly in oceans and release microplastics, while paper straws decompose more readily

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The evidence is further complicated by the fact that some experts consider reusable options—such as glass or metal straws—as more sustainable long-term solutions despite their own environmental trade-offs

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: While most plant-based proteins contain only some essential amino acids, leaving a gap that needs to be filled through dietary variety, nutritional yeast stands out as a rare vegan source that meets the full complement of amino acid requirements on its own

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Traditionally, plants require light for photosynthesis, but research suggests they can also grow using electricity to produce acetate as a photosynthesis substitute

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The answer depends on which tradition or interpretation you follow — folklore generally allows voluntary or curse-based transformation regardless of the moon, while popular culture (especially film) introduced the full moon as a transformative trigger

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes — barefoot running is healthier for joints and injury prevention, but shoes offer protection from cuts, bruises extreme temperatures, so the answer depends on context and individual factors

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d1
- **Claim**: The Phoenix Lights incident was officially explained as military flares, specifically LUU-2B/B rescue flares dropped by A-10C Thunderbolt IIs during a training exercise over the Barry Goldwater Range

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Religious scholars and commentators present competing views on whether Mormons are Christians. Mormons self-identify as Christians because they believe in Jesus Christ and draw on the New Testament, but some Christian denominations — particularly evangelical groups — contend that key Mormon doctrines (such as the nature of God, divine potential scriptural additions) constitute fundamental departures from traditional Christian theology. This definitional dispute — whether shared belief in Jesus Christ is sufficient or whether adherence to specific doctrines is also required — drives the ongoing disagreement, with no universal consensus across all religious authorities

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: August 16, 1977

### Sample freshqa_2b9ba7e192e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Boluarte's presidency is further notable for occurring in a period of intense political instability, as she became the sixth Peruvian president in less than five years

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel was won by *When We Were Real* by Daryl Gregory at the 2025 awards. This result is corroborated across multiple sources, with the award ceremony and results page listing Gregory as the winner for the 2025 category. Earlier records show John Wiswell's 'Someone You Can Build a Nest In' won the 2024 award, making Gregory's win the most recent

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

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: No permanent cure for cancer has been developed

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Overall, while yoga may offer meaningful benefits—such as improved breathing, reduced stress enhanced quality of life—it is not established as a standalone replacement for conventional asthma management

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Norma Koch won the Oscar for Whatever Happened to Baby Jane

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The Tunisia chapter was the most immediate destination after Algeria and Morocco, culminating in the surrender of 250,000 German and Italian troops on May 12, 1943

### Sample qacc_213701765f94

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: All 155 crew and passengers survived the incident, which occurred after both engines were disabled by bird strikes during takeoff from LaGuardia Airport

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Historians also theorize that the Christian adaptation involved forming the ichthys (fish) symbol by touching thumbs and crossing index fingers, further linking the gesture to religious invocations of divine protection

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: As a player, Bill Russell holds the record with 11 NBA championships; as a coach, Phil Jackson holds the record with 11 rings — making them tied for the most in NBA history within their respective categories

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Canada's independence from Great Britain is a nuanced, multi-stage process rather than a single date, though July 1, 1867 — when the Dominion of Canada was formed — is often cited as the foundational moment

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: October 1968

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

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: XXXTENTACION

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: ICD-10 codes generally consist of three to seven characters

### Sample qacc_f2218f8c979e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5, d3
- **Claim**: Prime rib comes from the rib primal, specifically the portion of the cow situated under the front section of the backbone between the fifth and sixth ribs and the twelfth and thirteenth ribs. It is cut from the same anatomical section as ribeye steak, spanning the area between the chuck and the loin

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Nassau County, located immediately east of New York City, contains 64 incorporated villages and over 60 unincorporated hamlets within its borders

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

### Sample situatedqa_geo_7222d6123c27

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Calcutta became the capital of British India in 1772, when Warren Hastings transferred all important offices there

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Scotland banned smoking in pubs on March 26, 2006, making it the first part of the UK to do so; Wales followed on April 2, 2007 England and Northern Ireland on July 1, 2007

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3
- **Claim**: Since 1965, most immigrants have come from Latin America (about half) and Asia (about a quarter), with Mexico alone accounting for roughly 25% of all immigrants

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Eisenhower

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
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

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Kiren Rijiju, who serves as the Cabinet Minister for Law & Justice

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: This document created a weak central government described as a 'league of friendship' between the states, with a single legislative body and no separate executive or judicial branches

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This cultural shift was further cemented in 1865, when Union soldiers returned home from the Civil War with a taste for coffee after it was issued as part of their standard rations, completely eclipsing hot tea and establishing coffee as the dominant American beverage for the 20th century and beyond

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Todd Monken

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: The Cumberland River begins at the confluence of its forks—Poor Fork and Clover Fork—in Harlan County, Kentucky flows generally westward through the Cumberland Plateau before turning north into Tennessee

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that appear in the blood when the heart is stressed or damaged they are used to diagnose and monitor heart disease

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

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 133

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India in the 2023 final by six wickets on 19 November at the Narendra Modi Stadium in Ahmedabad. This was Australia's sixth Cricket World Cup title the tournament was the 13th edition, held in India from 5 October to 19 November 2023. While India had previously won the 2023 tournament according to an older Wikipedia revision, the most current information confirms Australia as the 2023 champion

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was formally adopted in 2016 the city is now known by this official name

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025. He is the 24th person to serve in the role

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This name change was confirmed by the Government of Haryana in 2016 the city is now officially known by this new name. As a result, Gurgaon is no longer the official name of the city

### Sample wikirevision_0171

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Jannik Sinner


================================================================================

*Report generated by CATS v2.0*
