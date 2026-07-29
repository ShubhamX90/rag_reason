# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.999 (over 736 samples)

**GR F1** *(used in CATS)*: 0.999

**Behavior Adherence**: 0.742 (over 608 applicable samples)

**Factual Grounding**: 0.831 (over 608 applicable samples)

**Single-Truth Recall**: 0.680 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.813

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.999
- **Precision**: 1.000
- **Recall**: 0.998
- **Accuracy**: 0.999
- TP=607, FP=0, FN=1, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.992
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.996
- **Specificity**: 0.998
- Abstain TP=128, FP=1, FN=0, TN=607


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.995
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.955 (n=154)
- **Grounding**: 0.857 (n=154)
- **Recall**: 0.815 (n=154)
- **CATS**: 0.906

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.869 (n=176)
- **Grounding**: 0.841 (n=176)
- **Recall**: 0.532 (n=156)
- **CATS**: 0.811

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.302 (n=96)
- **Grounding**: 0.868 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.723

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.669 (n=145)
- **Grounding**: 0.806 (n=145)
- **Recall**: 0.693 (n=140)
- **CATS**: 0.792

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.671 (n=37)
- **Recall**: 0.689 (n=37)
- **CATS**: 0.759


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1967

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

### Sample conflictingqa_04e1627e9fc5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Research has shown that organic fertilizers that stimulate plant growth also increase the population of plant-parasitic nematodes (PPNs), while organic matter and balanced composts with a low C:N ratio can suppress PPNs by releasing ammonia and isothiocyanates , indicating that nematode communities are directly involved in nutrient cycling and plant-root interactions

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: No, weight lifting does not cause high blood pressure

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Yes, anime is a form of cartoon — it is animation produced in Japan, though it originated in the United States

### Sample conflictingqa_1d741779a31e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Additional studies are needed to fully understand the long-term effects of artificial sweeteners in patients with diabetes some researchers and regulatory bodies recommend individualized dosing based on blood glucose and weight tracking

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Yes

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: No, we cannot know anything beyond our minds

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Yes, the Great Pacific Garbage Patch is larger than Texas

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: CLS Bank decision has not eliminated software patents entirely — only software implementations of abstract ideas are generally not patentable

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: In short, software patents are a valuable form of protection for inventions that are new, non-obvious sufficiently described, regardless of the medium on which the software is recorded

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d9, d4
- **Supporting Docs Found**: d5
- **Claim**: However, other studies have produced mixed or null results: a recently published study with a mean follow-up period of 1.35 ± 0.75 years found no effect of bicarbonate administration on kidney failure progression a study of advanced-stage diabetic CKD with normal bicarbonate levels found that a low dose of 0.5 mEq/kg/day sodium bicarbonate did not significantly reduce urinary TGF-β over six months , while a separate study found that oral bicarbonate supplementation had no effect on eGFR decline in stage 5 CKD

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6, d5
- **Supporting Docs Found**: d2
- **Claim**: The KDIGO guidelines, while recommending sodium bicarbonate for normalizing blood bicarbonate levels in CKD patients, acknowledge that the evidence remains uncertain a large, multicenter study of CKD progression is currently being conducted to further evaluate the role of bicarbonate supplementation in this population

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The retrieved evidence is conflicting. Some sources argue that large earthquakes are more likely during full and new moons, while others argue that a new study found no relationship between lunar phase and earthquake occurrence

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: The retrieved evidence presents competing views. Some sources argue that the Catholic Church is the true church because it claims to be the one, holy, catholic apostolic Church established by Jesus Christ, while others argue that the Catholic Church is not clearly mentioned in the Bible and that other denominations also claim to be the true church, making the question of which is the 'one true church' a matter of genuine theological debate

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Yes; volcanic activity—specifically, elevated levels of mercury relative to organic carbon in North Sea sediments—has been identified as a direct proxy for volcanism multiple studies confirm its role as the dominant carbon source driving the Paleocene-Eocene Thermal Maximum (PETM)

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence is divided. Some sources argue that werewolves can transform during a full moon, while others argue that full moon transformations are largely a product of cinematic storytelling and not rooted in ancient myths

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: The retrieved evidence presents a genuine barefoot running debate. Some sources argue that running barefoot is healthier than running with shoes, while others argue that shoes provide arch support and cushioning that barefoot running does not, making shoes the healthier option

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Yes, humans did evolve from apes — specifically from a common ancestor with chimpanzees and other primates

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Over the following decades, other Dutch explorers like Dirk Hartog, Frederik de Houtman Pieter Nuyts further charted Australia's western and southern coastlines the VOC officially labeled the land as 'New Holland'

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Aryna Sabalenka (6-3, 7-6(3))

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: In 2026, the first Seder is held on the evening of April 1 the second Seder is held on April 2

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Hillary Clinton enacted at least 1 executive order during her tenure as Secretary of State, though the specific number is not explicitly stated in the available evidence

### Sample freshqa_263eca8e024e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This feat, with Box Office India reporting ₹1,810 crore the Times of India noting ₹1,750 crore

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy won the latest Grammy Award for Best Jazz Performance, taking home the award for "Twinkle Twinkle Little Me" featuring Sullivan Fortner at the 67th Annual Grammy Awards in 2025

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest major version of the .NET Framework is 4.8.1, released on August 9, 2022

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Microsoft .NET Framework download page, which lists 4.8.1 as the latest release is corroborated by additional sources that note it was released on August 9, 2022

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Multiple other sources, including Stack Overflow, confirm that 4.8.1 is the most recent .NET Framework version, with d3dcompiler fixes and improved performance

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: The U.S. Army conducted the test at approximately 5:30 a.m., detonating a plutonium-powered implosion device atop a 100-foot steel tower, releasing 18.6 kilotons of energy

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 3 seasons — Season 1 premiered November 12, 2019, Season 2 premiered October 30, 2020 Season 3 premiered March 1, 2023

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: No single chemical reaction directly produces gold as a byproduct from lead; rather, gold is produced from lead indirectly through nuclear transmutation, which requires a particle accelerator and large amounts of energy

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The Federal Reserve cut interest rates by 50 basis points from August to December 2022, bringing the federal funds rate down to 3.50%–3.75%

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: 2023

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The last player to win the Ballon d'Or before the Messi–Ronaldo dominance was Luka Modric, who claimed the award in 2018

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Laika

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: George R.R. Martin was born in Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The most recent Nebula Award for Best Novel was won by *The Incandescent* by Emily Tesh, published by Tor/Orbit UK, in 2025

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Frank Rosenblatt died in a boating accident on July 28, 1971

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: No, the Raptors do not have a winning record in the 2023–24 NBA season

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4, d1
- **Claim**: Queen Elizabeth II of England died on 8 September 2022, at Balmoral Castle in Scotland, where she had been staying with her family

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: She was 96 years old

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Jeff Bezos did not fully sell Amazon; rather, he executed a series of stock sales over time

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Jiangsu and Zhejiang provinces

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 8

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The NFL announced that the Bills vs. Bengals game on January 2, 2023 resumed play on January 6, 2023

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3, d5
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3, d5
- **Supporting Docs Found**: d4
- **Claim**: This figure is confirmed by multiple sources reporting on the same Yamagata University-led study, which found 303 previously unknown geoglyphs using AI technology, bringing the total to 893

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d7
- **Claim**: 2016

### Sample hotpotqa_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9
- **Supporting Docs Found**: None
- **Claim**: She represented the United States and set the American record with a time of 10.8 seconds

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 2011

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d10
- **Supporting Docs Found**: None
- **Claim**: The program also included recruitment of scientists from other countries, such as Austria and Italy, bringing the total to over 1,800 it was confirmed that at least 10 German scientists were detained under Operation Epsilon , suggesting the broader scope of the recruitment effort

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3, d6, d7
- **Supporting Docs Found**: d4
- **Claim**: The jingle appears on Timberlake's 2006 album *FutureSex/LoveSound* and is also included on his 2008 compilation *Greatest Hits: FutureSex/LoveSound*, but Pusha T's role as the jingle's composer is consistently confirmed across multiple sources

### Sample qacc_160a528ae07e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4
- **Claim**: Tom Brady has won the NFL MVP award three times, in 2007, 2010 2017, when he was with the New England Patriots

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Oliver Stark plays Buck on 9-1-1

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Wood Harris plays Ace/Azie Faison, Mekhi Phifer plays Mitch/Rich Porter Cam'ron plays Rico/Alpo Martinez in the 2002 film

### Sample qacc_252987b8054c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Tori Spelling played Violet Anne Bickerstaff in Saved by the Bell

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3, d5
- **Claim**: The retrieved evidence suggests that crossing fingers for luck has its roots in pre-Christian pagan beliefs and early Christian practices, though the exact origins remain uncertain

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The discrepancy reflectsing the nuanced distinction between coaching and playing the fact that Auerbach's role as an executive is often overlooked in NBA history, means the most accurate count depends on how 'ring' is defined

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 1999

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Peyer's patches

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Alice Kremelberg plays Bill Pullman's wife in Season 4 of _The Sinner_

### Sample qacc_6837d86d03ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The retrieved evidence places Prince Charles, the Prince of Wales, as the next in line to be the monarch of England, following the death of Queen Elizabeth II on September 8, 2022

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: October 1, 1968

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: A third source, History Adventuring, corroborates that the first McDonald's in Phoenix represents a milestone in the evolution of fast food and the growth of McDonald's as a global brand

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4, d1
- **Claim**: Additionally, both Argentina and Uruguay share similar economic and ethnic patterns the European heritage of most of the population ties them to Europe as early trading partners

### Sample qacc_946ecfb478b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: Max Martin

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: 2026

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: "God Gave Rock and Roll to You" is sung by multiple artists, but the song was written by Russ Ballard of Argent in 1971

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The International Space Station first went into space on November 20, 2000, when the Russian module Zarya successfully launched and docked with the station, marking the beginning of continuous human presence in space

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3, d5
- **Claim**: Hosanna is an expression of praise and a cry for salvation, meaning "save us now" or "help, please." It is most commonly associated with Palm Sunday, when crowds greeted Jesus riding into Jerusalem with the cry, "Hosanna!

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Celebrity Big Brother is not on any major U.S. channel; it airs on ITV in the UK

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d5
- **Claim**: This followed decades of territorial status and political maneuvering, including the 1850 creation of the New Mexico Territory and the 1910 drafting of a state constitution

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Rio de Janeiro, Brazil

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Isaiah Mustafa plays the coach in Old Spice commercials

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This joint allows for movement in two planes — flexion, extension, abduction, adduction circumduction — and is surrounded by a joint capsule filled with synovial fluid

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Carter Pewterschmidt is Lois's father, voiced by Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Elton Hayes

### Sample qacc_d00b0063e747

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The twins were born on November 1, 2022 are both healthy and happy

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: Cadbury's products are sold in over 50 countries across six continents, though the exact number varies by product and region. The UK and Ireland are the brand's largest markets, with Australia, New Zealand, South Africa the US also significant sales regions

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: 1996

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The balance sheet is the financial statement that involves all aspects of the accounting equation, showing the sum of assets, liabilities equity (Assets = Liabilities + Equity)

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: They are officially called cuotas (fees) drivers pay in Mexican pesos

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d4
- **Claim**: 2022-2023

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3, d5
- **Claim**: Season 7 of Game of Thrones consists of seven episodes — confirmed by HBO's episode listing, which places all seven episodes on the schedule from July 16 through August 27, 2017

### Sample qacc_ff2cb00f4c03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Entertainment Weekly and Vanity Fair, meanwhile, note that Season 7's average episode length is roughly 59 minutes, with the finale running at 81 minutes — the longest episode of the series to date — and the show's overall runtime coming to 7 hours and 20 minutes

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d5
- **Supporting Docs Found**: None
- **Claim**: Alaska is the 3rd largest U.S. state by area, covering approximately 665,384 square miles (1,723,337 km²)

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d3, d5
- **Supporting Docs Found**: None
- **Claim**: All available evidence consistently confirms this ranking, with no contradictions across sources

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: The retrieved evidence places the number of fronts fought by the Axis in World War II at three: the Eastern Front (also called the Russia Front), the Western Front (also called the France and the Low Countries Front) the Mediterranean Front (also called the Italy Front)

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The official Guinness World Records title for the 'land farthest from the sea' is held by the Eurasian pole of inaccessibility in northwestern China, over 500km from any ocean

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This is a representative democracy, meaning citizens vote for representatives who then make laws on their behalf

### Sample situatedqa_geo_85af31651715

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This was confirmed by Andy Kerr, Scotland's former health minister, who spearheaded the smoking ban campaign

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: Pew Research estimates that in 2065, those who identify as Asian will make up some 38% of all immigrants, surpassing the 31% share of Hispanics , while the Bush Institute notes that more than 60% of immigrants today live in the West and South

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d1
- **Supporting Docs Found**: None
- **Claim**: This deployment was part of the broader U.S. foreign policy during the Cold War, aimed at preventing the spread of communism in Southeast Asia

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: The California flag is a bear flag, specifically the California grizzly bear flag. The bear is the official state animal of California and appears on the state flag as a symbol of strength and unyielding resistance

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: 2026

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Shri Kiren Rijiju, who serves as the Minister of State for Law and Justice in the Government of India. He is a senior BJP leader and has held the position since 2020

### Sample situatedqa_geo_f26078ec6467

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The conflict was largely triggered by U.S. support of Cuba's struggle for independence from Spain and the refusal of the Spanish government to withdraw from the island

### Sample situatedqa_geo_f26078ec6467

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: The war was conducted across Cuba and the Philippines apart from guaranteeing the independence of Cuba, the U.S. also acquired Guam, Puerto Rico the Philippines

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5
- **Claim**: This document established a 'league of friendship' between the states, creating a weak central government with significant democratic limitations compared to later constitutional evolution

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: At the state level, governors and state legislatures play critical roles in implementing federal environmental regulations and addressing local environmental priorities at the local level, cities and towns often have their own environmental policies and programs

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: 1995

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 2025

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d5
- **Claim**: This win also made Narang the third shooter from India to medal at the Olympics, joining an elite group that includes Abhinav Bindra and Sanjeev Rajput

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Gavin Creel won the 2009 Tony Award for Best Actor in a Leading Role in a Musical, for his performance in Hair

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: The current Chief Justice of the Sindh High Court is Justice Zafar Ahmed Rajput, who became the acting Chief Justice on 06 December 2025. He is the 12th incumbent to serve as Chief Justice of the SHC since its inception in 1993

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Jordan Ridgeway on Days of Our Lives (Dool) and Bethany Bryant on The Young and the Restless (Y&R)

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This version brings Live Updates, lock screen widgets, grouped notifications improved performance for larger-screen devices

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: This as the most recent Android release, with d2 and d3 being outdated and superseded by the June 2025 update

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d1
- **Claim**: This season ran from October 3, 2018 to May 22, 2019 the show was renewed for a third season on February 23, 2017

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: PTI's victory gave the party the largest number of seats in the National Assembly, though some reports note that the election was marked by widespread allegations of rigging

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Todd Monken

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The current Health Minister of India is Jagat Prakash Nadda, who has served in office since 2026–27

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8808c106a115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5, d1
- **Supporting Docs Found**: None
- **Claim**: This deficiency leads to the abnormal accumulation of GM2-ganglioside in brain and nerve cells, eventually causing the progressive deterioration of the central nervous system

### Sample situatedqa_temp_901be1437bc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: India won the 2018 Test series 2-1, with Virat Kohli scoring the most runs (286)

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Carnie Wilson, Wendy Wilson, Chynna Phillips

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: These are the colors humans perceive as red, orange yellow when looking at a rainbow or a prism

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4, d5
- **Claim**: The U.S. will host the 2028 Summer Olympics in Los Angeles, making it the ninth time the U.S. has played host

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3, d5
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin & Perry Go Large

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Riyad Mahrez won the PFA Player of the Year award for 2015–16, beating team-mates Jamie Vardy and N'Golo Kante, Tottenham's Harry Kane, West Ham's Dimitri Payet and Arsenal's Mesut Ozil

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: 1982–83

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d1
- **Supporting Docs Found**: None
- **Claim**: The current FIFA World Cup champion is Argentina, who defeated France 4–2 on penalties in the 2022 final, securing their third title and their first since 1986

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3, d1
- **Supporting Docs Found**: None
- **Claim**: Argentina's victory gave the country its third title and the first since 1986, making them the current champions

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The current Indian Premier League champion is Chennai Super Kings, who defeated Gujarat Titans by five wickets (DLS method) in the 2023 final to win their fifth league title. This result is consistently confirmed across multiple sources, with Chennai Super Kings' victory confirmed across the 2023 season, making them the most recent champions

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Google is owned by Alphabet Inc., the parent company of Google LLC, which is itself owned by its founders Larry Page and Sergey Brin. Alphabet Inc. is a public company with Nasdaq symbols GOOGL (Class A share) and GOOG (Class C share) Google LLC is a wholly owned subsidiary of Alphabet Inc

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This win was confirmed across multiple sources, with Dembélé beating the likes of Lionel Messi and Cristiano Ronaldo to claim the award

### Sample wikirevision_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Benjamin Netanyahu is the current Prime Minister of Israel, having assumed office on 29 December 2022. This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of Prime Minister of Israel, the list of Israeli prime ministers the older and newer Wikipedia revisions of Alternate Prime Minister of Israel

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d3
- **Claim**: This change was confirmed across multiple sources, including the newer Wikipedia revision of Twitter and the official X Wikipedia revision , as well as corroborated by the broader context of the takeover by Elon Musk

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This victory gave Australia their sixth title, the most successful record held by any team in the history of the tournament

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025. He is the 24th and current Prime Minister, serving as the official head of government of Canada. This is consistent across multiple sources, including the older and newer Wikipedia revisions of Prime Minister of Canada, as well as the list of Canadian prime ministers

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who became incumbent on 23 May 2022. He is the 31st person to serve in the role since the office was created in 1901. This is consistent across multiple sources, including the official Australian Government website and Wikipedia's list of Australian prime ministers

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This victory marked Sinner's third major overall, completing the career Grand Slam

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Sinner's victory gave him his first major title and the 2026 Wimbledon Championships are scheduled to take place from 29 June to 12 July 2026, marking the first time video reviews will be used in the tournament's history

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current President of India is Droupadi Murmu, who became the incumbent on 25 July 2022. She is the 15th President of India since the post was established in 1950 and serves as the head of state and supreme commander of the Indian Armed Forces. This is consistent across multiple sources, including the official Government of India website and Wikipedia's list of presidents of India

### Sample wikirevision_0162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: Argentina's victory gave the country its third title, adding to the ones won in 1978 and 1986, making them the most successful team with five titles in total

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The latest French Open men's singles champion is Carlos Alcaraz, who defeated world No. 1 Jannik Sinner in the 2025 final to win his second French Open title and fifth major. The 2025 French Open was the 124th edition of the tournament, held at the Stade Roland Garros in Paris, France, from 25 May to 8 June 2025, with Carlos Alcaraz defending his title from the 2024 champion Jannik Sinner


================================================================================

*Report generated by CATS v2.0*
