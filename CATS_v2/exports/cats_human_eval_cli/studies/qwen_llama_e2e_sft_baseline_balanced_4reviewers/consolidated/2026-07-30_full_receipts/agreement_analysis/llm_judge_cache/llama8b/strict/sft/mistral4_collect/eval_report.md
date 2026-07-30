# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 126 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.973 (over 736 samples)

**GR F1** *(used in CATS)*: 0.983

**Behavior Adherence**: 0.667 (over 610 applicable samples)

**Factual Grounding**: 0.881 (over 610 applicable samples)

**Single-Truth Recall**: 0.495 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.757

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.983
- **Precision**: 0.997
- **Recall**: 0.970
- **Accuracy**: 0.973
- TP=590, FP=2, FN=18, TN=126

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.875
- **Abstain Recall**: 0.984
- **Abstain F1**: 0.926
- **Specificity**: 0.970
- Abstain TP=126, FP=18, FN=2, TN=590


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.995
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.818 (n=154)
- **Grounding**: 0.929 (n=154)
- **Recall**: 0.448 (n=154)
- **CATS**: 0.798

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.950
- **GR F1** *(used in CATS)*: 0.968
- **Behavior**: 0.702 (n=178)
- **Grounding**: 0.876 (n=178)
- **Recall**: 0.436 (n=156)
- **CATS**: 0.746

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.740 (n=96)
- **Grounding**: 0.888 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.872

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.503 (n=145)
- **Grounding**: 0.861 (n=145)
- **Recall**: 0.650 (n=140)
- **CATS**: 0.751

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.324 (n=37)
- **Grounding**: 0.766 (n=37)
- **Recall**: 0.351 (n=37)
- **CATS**: 0.600


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2119

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/mistral-small-4

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2119
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: In the European Union, however, fashion designs can be protected for up to three or five years under the Creative Designs Directive, depending on whether they are new and have individual character

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d4
- **Claim**: This fungus is estimated to be over 2,000 years old and weighs approximately 440 tons, making it as massive as three blue whales

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Multiple authoritative sources, including the Oregon Field Guide and the US Forest Service, confirm that this fungus holds the record for the largest single living organism on Earth, surpassing all other contenders such as giant trees and blue whales

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Research on apple peel supplementation in mice showed it produced greater effects on nitric oxide levels and endothelin-1 than apple flesh alone, suggesting the peel retains a significant portion of the apple's nutritional value , a finding corroborated by the fact that vitamins like vitamin C are evenly distributed throughout the apple and are not reduced by peeling

### Sample conflictingqa_114c06976f62

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: By its own founding statement, the FSM is described as a'real, legitimate religion, as much as any other,' though its roots trace back to a satirical 2005 letter protesting intelligent design education policies

### Sample conflictingqa_11c5ef7c4545

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The debate is further nuanced by research showing that nearly three quarters of entrepreneurs start their businesses in pursuit of opportunity rather than necessity that regions like North America have the highest proportion of entrepreneurs globally , suggesting that entrepreneurship is not only possible but also increasingly appealing to a broad cross-section of people

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: In summary, while the Silurian was a period of significant terrestrial plant evolution and diversification, it is not explicitly identified as the birth of the first land plants, which had already emerged earlier in the Ordovician

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d4, d2
- **Claim**: However, the mechanisms are not fully understood the extent to which epigenetic changes are hereditary remains an active area of research

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The 2024 KDIGO guidelines recommend bicarbonate supplementation only when serum bicarbonate is less than 18 mEq/L results from a large randomized controlled trial published in the New England Journal of Medicine found that bicarbonate supplementation did not affect the composite outcome of CKD progression, cardiovascular events death over a median of 2.4 years , though the study was limited to patients with normal or high serum bicarbonate levels at baseline

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: The 1815 eruption of Mount Tambora is widely considered the most powerful volcanic eruption in recorded history it is also consistently described as the deadliest

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: A fair and complete answer would need to address all these perspectives and the scriptural evidence they cite

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: It can affect other parts of the body contacted by semen or vaginal fluid such as the anus, throat eyes

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Research has shown that plants can survive in complete darkness for up to 30 days some species can even regrow their leaves after being in darkness for up to 60 days , though continuous darkness will eventually kill all plants

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The evidence is divided on whether cold water makes hair shinier. Some sources argue that cold water closes the hair cuticle and can make hair appear shinier and smoother, while others argue that the effect is negligible and that hot air from a hair dryer simply opens the cuticle back up again that hair is dead tissue and cold water has the same effect as warm water

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: In summary, the evidence presents conflicting research outcomes: while some studies confirm a recent brain size decrease, others attribute brain stability to metabolic constraints or describe a long-term trend of brain enlargement, making the answer complex and contested

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d1, d2
- **Claim**: Ultimately, the question of whether the Bible is infallible is a matter of theological doctrine and interpretation different Christian traditions hold differing views on the extent to which the Bible's contents reflect God's word perfectly

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: A nuanced view holds that yoga predates modern religion and offers a distinct path to enlightenment that does not depend on adherence to a particular deity or scripture , though it acknowledges that yoga practices such as mantra chanting and ritual can evoke strong religious associations

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The evidence does not support a definitive yes/no answer; rather, it indicates that emojis are best understood as a distinct mode of expression that augments written language rather than replacing it

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the British were the ones who established the first permanent settlement and are commonly credited with the "discovery" of Australia in the modern sense , while the Dutch eventually abandoned their colony at New Holland (now South Australia) in 1773

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: d3
- **Claim**: The evidence does not support attributing the discovery of Australia solely to either the Dutch or the British — rather, it was a shared European endeavor with both nations playing distinct roles

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: The question of whether Mormons are Christian is genuinely contested. Some sources argue that Mormonism is Christian because it professes belief in Jesus Christ and affirms itself as part of the body of Christ, while others argue that Mormonism's distinctive doctrines—such as the godhead, the concept of a pre-mortal existence the restorationist view of the Christian church—represent significant departures from historic, orthodox Christianity

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This year's Passover began at sundown on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: She is described as the 'genius child' who overcame significant obstacles to achieve this milestone her award was recognized for her outstanding contributions to the dynamics and geometry of Riemann surfaces and their moduli spaces

### Sample freshqa_2130eea851fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This figure is directly reported from a citation tracking profile and is the most recent and specific data provided in the retrieved set

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d1
- **Supporting Docs Found**: None
- **Claim**: Joe Biden visited Russia as U.S. president on June 16, 2021, when he met with Russian President Vladimir Putin at Villa La Grange in Geneva, Switzerland

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The film follows a comic, multi-generational American saga of political resistance and was Anderson's first Oscar win after being nominated multiple times before

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The first animal to land on the moon was a Russian tortoise, as confirmed by NASA's Apollo 17 mission in 1972

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: All available evidence aligns on this fact without contradiction, confirming Bayonne, New Jersey as his birthplace

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: Beijing

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: The latest Nebula Award for Best Novel is 'Someone You Can Build a Nest In' by John Wiswell, which won in 2024

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2
- **Claim**: This is directly confirmed by a Congressional Record entry from July 28, 1971, which references his death in a boating accident is corroborated by the Cornell Chronicle's obituary, which notes that Rosenblatt died in a boating accident on his 43rd birthday

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is the most recent data in the retrieved evidence, superseding older records from 2019–20 and later that had described the team as a playoff contender

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: The 2023–24 season was a disappointing year for the Raptors, who had hoped to build on earlier successes including their 2019 NBA championship , but were unable to find consistent winning form

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: d4
- **Claim**: This total represents the most recent and comprehensive count available, superseding earlier estimates cited in other sources

### Sample healthcontradict_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Methodological differences between the two studies—such as sample size, population outcome measures—contribute to these opposing research outcomes, making it difficult to draw a definitive conclusion about whether yoga broadly improves asthma management

### Sample hotpotqa_0062

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0063

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8
- **Supporting Docs Found**: d10
- **Claim**: Stanford University, on the other hand, is a private research university located in Stanford, California is not associated with Chestnut Hill, Massachusetts

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: He also starred in other notable films including 'One Million B.C.' (1940), 'My Darling Clementine' (1946), 'Kiss of Death' (1947) 'The Robe' (1953) was known for his dark good looks and charismatic stage presence

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d2
- **Supporting Docs Found**: d9
- **Claim**: It is a member of MedStar Health, the not-for-profit healthcare network serves as a teaching hospital for Georgetown University School of Medicine

### Sample hotpotqa_0192

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 26 episodes — Curse of Oak Island Season 5 has 26 episodes

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d2
- **Claim**: Bill Russell

### Sample qacc_403a59870dc2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This as their second championship under Sean McVay, with the first being Super Bowl LIII in 2018

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central lymphatic capillaries in intestinal villi responsible for absorbing dietary fats and fat-soluble vitamins

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: These origins are not fixed sequences — in metazoans, a specific consensus sequence has not been identified their selection is largely epigenetic — and they fall into three main classes: constitutive, flexible dormant, with the vast majority (flexible origins) being used in an apparently stochastic manner in each cell cycle

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d1
- **Claim**: This as the complete run of the TV series, with no further seasons announced

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The International Space Station (ISS) did not have a single launch date into space; it was built in stages over several years

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: 245

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The music for Disney's Robin Hood was created by a combination of composers across two different film adaptations. The 1952 live-action Robin Hood featured music composed by Elton Hayes, who drew upon medieval English melodies and wrote original songs for the film. The 1973 animated Robin Hood, on the other hand, featured music and lyrics by Roger Miller for songs like 'Whistle-Stop' and 'Oo-de-lally', as well as music by Floyd Huddleston for the character 'Love' the majority of the score was composed by George Bruns. These contributions were further confirmed by the soundtrack album releases for both films, which list the respective composers and lyricists for each track

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: These dates reflect the gradual development of effigy mound culture over several centuries, with the custom dying out about 800 years ago

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The Balance Sheet (also known as the Statement of Financial Position) is the financial statement that involves all aspects of the accounting equation. It displays a company's total assets, total liabilities total equity at a single point in time, directly representing the equation Assets = Liabilities + Equity. The Balance Sheet is central to the accounting system the accounting equation is often described as the foundation of double-entry bookkeeping because every transaction is recorded in a way that maintains this balance between assets, liabilities equity

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d1, d2
- **Claim**: Tradition also places John in Ephesus at the time of writing, though the exact date remains a matter of scholarly debate

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: An initialism is a type of abbreviation that is formed from the initial letters of a phrase and is pronounced letter by letter, rather than as a word. Examples include CEO, DNA FBI. While the term 'acronym' is sometimes used broadly to refer to any abbreviation formed from the first letters of a phrase, technically an acronym is pronounced as a word rather than individual letters, making initialism a distinct category

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Sushma Swaraj also held other firsts to her name, including being the youngest Cabinet Minister in Haryana in 1977 and the first woman Chief Minister of Delhi in 1998

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: 7

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A global breakdown by country further contextualizes the scale: the Soviet Union lost 8.8–10.7 million soldiers and 10.4–13.3 million civilians, while Germany, Japan the United States each suffered between 5–6 million military deaths, with China's military losses alone ranging from 3–4 million

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: 6 fronts

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: The answer depends on the time period: earlier waves brought mostly Europeans (especially from Germany, Ireland the UK), while more recent immigration has shifted toward Latin America and Asia, with Mexico being the largest single origin country

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The present Law Minister of India is Arjun Ram Meghwal, who serves as the Minister of Law and Justice. He is a prominent BJP leader and a former Minister of State for Parliamentary Affairs, appointed to the full Cabinet position of Minister of Law and Justice. This is consistent across multiple sources, including the official Law Ministry of India website and other current affairs outlets

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: At the state level, California's own environmental policies, such as its cap-and-trade program, have served as a model for the nation many other states have their own environmental agencies and regulations

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Together, these complementary federal, state local efforts form a comprehensive framework for environmental protection across the country

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: New South Wales last won the State of Origin series in 2026 (most recent data in retrieved evidence)

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida Gators

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The 2022 final was confirmed as the most recent World Cup, with the tournament taking place in Qatar from November 20 to December 18, 2022

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: The latest Android version is Android 16, released on June 10, 2025. It was first released on Google Pixel phones and has since rolled out to Samsung Galaxy and other devices. Android 16 does not have a dessert nickname like earlier versions; its internal codename is 'Baklava'

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d4
- **Claim**: These distinct definitions reflect temporal and classification differences within the same abbreviation

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 km

### Sample situatedqa_temp_901be1437bc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Kent County, Maryland

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: 164

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: He defeated Anthony Joshua to win the WBA title in September 2021, Tyson Fury to win the WBC in May 2024 Daniel Dubois to win the IBF in July 2025, unifying all four belts

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 Indian Premier League (IPL) is the 19th edition of the Indian Premier League, a professional Twenty20 cricket league

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d2
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the older and newer Wikipedia revisions of the President of Indonesia article, as well as the dedicated biography of Prabowo Subianto

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Australia

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving a five-year term that is renewable once consecutively his incumbency began on 19 March 2017

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has been in office since 23 May 2022. He is the 31st person to hold the role since the office was created in 1901 is appointed by the Governor-General on the advice of the incumbent

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Australia


================================================================================

*Report generated by CATS v2.0*
