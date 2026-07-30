# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 126 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.973 (over 736 samples)

**GR F1** *(used in CATS)*: 0.983

**Behavior Adherence**: 0.803 (over 610 applicable samples)

**Factual Grounding**: 0.868 (over 610 applicable samples)

**Single-Truth Recall**: 0.717 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.843

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
- **Behavior**: 0.916 (n=154)
- **Grounding**: 0.910 (n=154)
- **Recall**: 0.838 (n=154)
- **CATS**: 0.915

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.950
- **GR F1** *(used in CATS)*: 0.968
- **Behavior**: 0.893 (n=178)
- **Grounding**: 0.860 (n=178)
- **Recall**: 0.609 (n=156)
- **CATS**: 0.832

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.688 (n=96)
- **Grounding**: 0.892 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.856

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.662 (n=145)
- **Grounding**: 0.852 (n=145)
- **Recall**: 0.707 (n=140)
- **CATS**: 0.803

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.757 (n=37)
- **Grounding**: 0.730 (n=37)
- **Recall**: 0.703 (n=37)
- **CATS**: 0.787


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2115

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/qwen3.5-397b-a17b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2119
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_05b33f4ca156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Some salamanders are poisonous to touch because they have toxin-secreting glands in their skin handling them can cause serious illness

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: In the European Union, however, fashion designs can be protected for up to three or five years under the Creative Designs Directive, depending on whether they are new and have individual character

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d4
- **Claim**: This fungus is estimated to be over 2,000 years old and weighs approximately 440 tons, making it as massive as three blue whales

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: Research on apple peel supplementation in mice showed it produced greater effects on nitric oxide levels and endothelin-1 than apple flesh alone, suggesting the peel retains a significant portion of the apple's nutritional value , a finding corroborated by the fact that vitamins like vitamin C are evenly distributed throughout the apple and are not reduced by peeling

### Sample conflictingqa_11c5ef7c4545

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The debate is further nuanced by research showing that nearly three quarters of entrepreneurs start their businesses in pursuit of opportunity rather than necessity that regions like North America have the highest proportion of entrepreneurs globally , suggesting that entrepreneurship is not only possible but also increasingly appealing to a broad cross-section of people

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: However, the mechanisms are not fully understood the extent to which epigenetic changes are hereditary remains an active area of research

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: The retrieved evidence presents a genuine debate on whether audiobooks count as real reading, with some sources arguing they are fully equivalent to physical books and others doubting their legitimacy

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d5
- **Claim**: The Moon has long been considered geologically inactive, with volcanism ceasing about 3 billion years ago and tectonic activity slowing to near-zero

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

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, these yield gaps also mean that organic farms require more land to produce the same amount of food as conventional farms , which can have its own environmental costs

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2, d5
- **Supporting Docs Found**: None
- **Claim**: Some sources note that high-yield conventional farming can use less land and produce fewer pollutants per unit of output , while others argue that organic farming offers distinct environmental benefits through soil health, biodiversity lower synthetic input use , suggesting that 'efficiency' is a multidimensional concept where organic farming excels in some areas while falling short in others

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: A fair and complete answer would need to address all these perspectives and the scriptural evidence they cite

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: It can affect other parts of the body contacted by semen or vaginal fluid such as the anus, throat eyes

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: They require a secure, well-ventilated tank with a heat source and humidity control, as well as a varied diet of leafy greens and fresh vegetables

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3, d2
- **Claim**: Research has shown that plants can survive in complete darkness for up to 30 days some species can even regrow their leaves after being in darkness for up to 60 days , though continuous darkness will eventually kill all plants

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: The debate between barefoot and shoed running is ongoing, with research yielding conflicting results. Some studies suggest that barefoot running may reduce the risk of chronic injuries by encouraging a mid-foot strike and strengthening foot muscles, while others argue that shoes provide benefits such as cushioning and arch support that reduce the high-impact stress on the body

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: A nuanced view holds that yoga predates modern religion and offers a distinct path to enlightenment that does not depend on adherence to a particular deity or scripture , though it acknowledges that yoga practices such as mantra chanting and ritual can evoke strong religious associations

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the British were the ones who established the first permanent settlement and are commonly credited with the "discovery" of Australia in the modern sense , while the Dutch eventually abandoned their colony at New Holland (now South Australia) in 1773

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d5
- **Supporting Docs Found**: None
- **Claim**: The evidence does not support attributing the discovery of Australia solely to either the Dutch or the British — rather, it was a shared European endeavor with both nations playing distinct roles

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This year's Passover began at sundown on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d5
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

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The war had reached its 1,000th day, with UN estimates suggesting Ukraine's population had declined by over 10 million people — roughly 25% of its total population — and over 1 million people had been killed or grievously injured

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The conflict has caused hundreds of thousands of deaths and millions of displaced persons, with Russia holding approximately 20% of Ukraine's territory

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Bismuth

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d5
- **Supporting Docs Found**: None
- **Claim**: Joe Biden visited Russia as U.S. president on June 16, 2021, when he met with Russian President Vladimir Putin at Villa La Grange in Geneva, Switzerland

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: This finding superseded the previous record of ~1 million-year-old DNA from a mammoth tooth is considered the oldest DNA discovered so far

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: She won over Team Kelly's Liv Ciara and Lucas West (Team Legend) to claim the title, earning a recording contract with Universal Music Group and a $100,000 cash prize

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The film follows a comic, multi-generational American saga of political resistance and was Anderson's first Oscar win after being nominated multiple times before

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Their most recent World Series appearance was in 2022, when they lost to the Philadelphia Phillies, making their total count remain at one

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Andrés Iniesta (2012)

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The first animal to land on the moon was a Russian tortoise, as confirmed by NASA's Apollo 17 mission in 1972

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: This is directly confirmed by a Congressional Record entry from July 28, 1971, which references his death in a boating accident is corroborated by the Cornell Chronicle's obituary, which notes that Rosenblatt died in a boating accident on his 43rd birthday

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The Toronto Raptors' latest season record is the 2023–24 season, in which they finished 25–57 and missed the playoffs

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: This is the most recent data in the retrieved evidence, superseding older records from 2019–20 and later that had described the team as a playoff contender

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
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
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 9 minutes

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: Stanford University, on the other hand, is a private research university located in Stanford, California is not associated with Chestnut Hill, Massachusetts

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Bartholdi was elected by French abolitionist and politician Édouard de Laboulaye to build a monument commemorating the kinship between France and the United States and the appreciation for President Abraham Lincoln's work

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: After defeating the Axis powers in North Africa, the Allies launched a further invasion of Italy

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 26 episodes — Curse of Oak Island Season 5 has 26 episodes

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2
- **Claim**: The 1972 win is the most recent FA Cup triumph, making it the definitive answer to when Leeds United won the FA Cup

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The film's production designer, Tamara Deverell, confirmed that the team shot on location in Montegut, Louisiana the bayou country, bringing the fictional Bathtub to life through a combination of real locations and practical effects

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d5
- **Claim**: Bill Russell

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central lymphatic capillaries in intestinal villi responsible for absorbing dietary fats and fat-soluble vitamins

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3, d1, d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
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
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: 245

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This joint allows movement in two planes, facilitating the transmission of sound vibrations from the tympanic membrane to the inner ear

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Carter Pewterschmidt (voiced by Alex Borstein)

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The Tavarez name has been carried by notable figures across a range of fields and has significant connections to the British peerage, tracing its presence in England since the medieval period

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: These dates reflect the gradual development of effigy mound culture over several centuries, with the custom dying out about 800 years ago

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: The Balance Sheet (also known as the Statement of Financial Position) is the financial statement that involves all aspects of the accounting equation. It displays a company's total assets, total liabilities total equity at a single point in time, directly representing the equation Assets = Liabilities + Equity. The Balance Sheet is central to the accounting system the accounting equation is often described as the foundation of double-entry bookkeeping because every transaction is recorded in a way that maintains this balance between assets, liabilities equity

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Sushma Swaraj also held other firsts to her name, including being the youngest Cabinet Minister in Haryana in 1977 and the first woman Chief Minister of Delhi in 1998

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: 7

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 6 fronts

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The U.S. is a representative democracy, a form of government in which the people elect representatives to make decisions on their behalf

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
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

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d5
- **Supporting Docs Found**: None
- **Claim**: It is composed of twelve members — seven from the Board of Governors and five presidents from Federal Reserve Banks — and meets regularly to decide on interest rates and the money supply through open market operations

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: At the state level, California's own environmental policies, such as its cap-and-trade program, have served as a model for the nation many other states have their own environmental agencies and regulations

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Together, these complementary federal, state local efforts form a comprehensive framework for environmental protection across the country

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: New South Wales last won the State of Origin series in 2026 (most recent data in retrieved evidence)

### Sample situatedqa_temp_35156c8be377

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: These differences reflect the distinction between absolute wealth (GDP) and standard of living (GDP per capita) both measurements have their own valid uses in different contexts

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
- **Supporting Docs Found**: d2, d5
- **Claim**: The 2022 final was confirmed as the most recent World Cup, with the tournament taking place in Qatar from November 20 to December 18, 2022

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: The latest Android version is Android 16, released on June 10, 2025. It was first released on Google Pixel phones and has since rolled out to Samsung Galaxy and other devices. Android 16 does not have a dessert nickname like earlier versions; its internal codename is 'Baklava'

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d1
- **Claim**: These distinct definitions reflect temporal and classification differences within the same abbreviation

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: 59,681 km

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: LeBron James and Anthony Davis led the team to its 17th title, briefly tying the Boston Celtics for the most in league history before the Celtics won their 18th in 2024

### Sample situatedqa_temp_901be1437bc7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Kent County, Maryland

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This figure is confirmed by the U.S. Energy Information Administration, which also notes that California's state gasoline excise tax is the highest in the United States, at $0.60 per gallon

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: It was a pivotal battle in which the Muslim army, numbering around 313 men, defeated a larger force of approximately 1,000 from the Quraysh tribe

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This is the first carrier of the Queen Elizabeth class, with her sister ship HMS Prince of Wales (R09) following in 2019

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The official UK Government publications confirm that the Queen Elizabeth class carriers are the future flagship and second carrier of the Royal Navy , with HMS Queen Elizabeth being the lead ship

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: 164

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: The retrieved evidence supports 1274 BC as the date of the Battle of Kadesh

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: He defeated Anthony Joshua to win the WBA title in September 2021, Tyson Fury to win the WBC in May 2024 Daniel Dubois to win the IBF in July 2025, unifying all four belts

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Their most recent playoff run came in the 2020-21 season, where they were defeated by the Atlanta Hawks in the first round

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: Twitter is known as X

### Sample wikirevision_0003

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: The rebranding was confirmed when Twitter, Inc. became part of X Corp after the merger the platform's name was updated to X

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 Indian Premier League (IPL) is the 19th edition of the Indian Premier League, a professional Twenty20 cricket league

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, including the newer Wikipedia revision of the Ballon d'Or article, which supersedes an older version that had described Messi as the winner

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Australia

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The 2026 tournament was the 139th edition of Wimbledon, making it the most recent men's singles competition at the event

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The current Prime Minister of Japan is Sanae Takaichi, who became Japan's first female Prime Minister after assuming office on 21 October 2025. She is the 32nd Prime Minister of Japan and the incumbent, serving in office from the date of her appointment by the National Diet


================================================================================

*Report generated by CATS v2.0*
