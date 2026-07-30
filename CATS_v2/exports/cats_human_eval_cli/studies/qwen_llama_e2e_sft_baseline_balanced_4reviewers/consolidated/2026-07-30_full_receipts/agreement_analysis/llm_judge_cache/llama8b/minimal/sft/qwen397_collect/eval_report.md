# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.976 (over 736 samples)

**GR F1** *(used in CATS)*: 0.985

**Behavior Adherence**: 0.791 (over 609 applicable samples)

**Factual Grounding**: 0.872 (over 609 applicable samples)

**Single-Truth Recall**: 0.706 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.839

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.985
- **Precision**: 0.998
- **Recall**: 0.972
- **Accuracy**: 0.976
- TP=591, FP=1, FN=17, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.882
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.934
- **Specificity**: 0.972
- Abstain TP=127, FP=17, FN=1, TN=591


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.995
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.942 (n=154)
- **Grounding**: 0.897 (n=154)
- **Recall**: 0.818 (n=154)
- **CATS**: 0.913

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.876 (n=177)
- **Grounding**: 0.868 (n=177)
- **Recall**: 0.615 (n=156)
- **CATS**: 0.833

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.635 (n=96)
- **Grounding**: 0.847 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.824

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.662 (n=145)
- **Grounding**: 0.901 (n=145)
- **Recall**: 0.707 (n=140)
- **CATS**: 0.816

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.892
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.730 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.742


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2037

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/qwen3.5-397b-a17b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2040
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: Peeling an apple reduces some of its nutritional value by removing dietary fiber and certain antioxidants, but not all nutrients — vitamins like vitamin C and compounds like flavonoids remain in the flesh

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d4, d1
- **Claim**: Yes, palm oil is bad for the environment; it causes deforestation, habitat destruction, biodiversity loss, greenhouse gas emissions soil erosion

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: A multi-faceted approach including stricter regulations, better enforcement public awareness is considered necessary to address the harms caused by unethical breeding while supporting responsible breeders

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: The Harvard School of Public Health cautions that fluoride's benefits on dental health do not necessarily justify adding it to drinking water, given these risks — particularly for bottle-fed infants, the elderly people with kidney disease — and calls for further research on safe dosing and vulnerable populations

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d4, d1, d2
- **Claim**: The conflict is factual, with d4 and d1 affirming inheritance via germline transmission, d5 denying it due to demethylation d2 and d3 offering partial mechanistic context without resolving the hereditary question

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Scaled feet and rigid first two digits in the hand, consistent with frequent ground walking, but with a third digit capable of movement suggesting tree climbing was also possible

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: The retrieved evidence presents a genuine debate on whether audiobooks count as real reading, with some sources arguing they are fully equivalent to physical books and others doubting their legitimacy

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Moon has long been considered geologically inactive, with volcanism ceasing ~3 billion years ago and the core dynamo shutting down ~2.5–1 billion years ago

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Additionally, researchers have found 266 previously undocumented small ridges on the Moon's far side that are younger than those on the near side, suggesting recent geological activity there as well

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The Kidney Disease Improving Global Outcomes (KDIGO) guidelines recommend bicarbonate supplementation only when serum bicarbonate levels fall below 18 mEq/L the evidence is considered insufficient to fully support routine use in all CKD stages

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Multiple sources consistently support this ranking, with no contradictions across documents

### Sample conflictingqa_63fde268aa8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: The conflict here reflects a methodological divergence between philosophical speculation and scientific evidence, with no single definitive answer

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Most products marketed as split end repairs work by temporarily coating the hair cuticle, adding weight to frayed ends creating a temporary 'glue' effect, but these effects do not last beyond the next shampoo

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The retrieved evidence is mixed. Some randomized controlled trials have shown that vitamin C can shorten the duration of common colds and reduce their severity by about 15%, while other research indicates that taking high doses of vitamin C does not prevent colds and that most people already get enough from their diet

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: From a Bahá'í perspective, cultural differences are not hindrances but rather secondary to humanity's spiritual unity the faith teaches that these differences can be overcome

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d4
- **Claim**: A middle-ground view holds that the benefits and risks must be weighed individually for each pet depending on factors like breed, sex, age health status

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Methodological differences—such as the EPA's larger dataset of 15 carcinogenicity studies compared to IARC's 8—underlie these opposing research conclusions, with high-credibility sources on both sides contributing to a nuanced and contested body of evidence

### Sample conflictingqa_a9bed39d234d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d3
- **Supporting Docs Found**: None
- **Claim**: The conflict arises from methodological divergence: some sources rely on theoretical models or limited studies suggesting net calorie deficits are biologically possible , while others rely on the weight of scientific consensus and measured data showing no food meets the negative-calorie threshold

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: One source notes that yeast protein biomass contains a mean of 47 grams per 100 grams, which is nearly 100% of the recommended daily intake for adults that yeast protein amino acids include lysine, isoleucine, leucine, phenylalanine, threonine valine in amounts exceeding FAO/WHO recommendations

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: The debate between barefoot and shoed running is ongoing, with research yielding conflicting results. Some studies suggest that barefoot running may reduce the risk of chronic injuries by encouraging a mid-foot strike and strengthening foot muscles, while others argue that shoes provide specific benefits such as greater arch support and stiffness, which foot muscles must work harder to compensate for

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: The evidence is mixed. Some high-credibility sources (NIH/NLM, Verywell Health, GoodRx) report associations between hot yerba mate consumption and increased risk of esophageal cancer, while also noting that yerba mate may have anti-cancer properties and could potentially reduce the risk of some cancers. The NIH review found that studies linking mate to cancer risk were methodologically similar but produced widely varying results, with the strongest association seen in women drinking large quantities at high temperature. Overall, the most credible and comprehensive evidence suggests that drinking yerba mate at very hot temperatures is the primary driver of cancer risk, rather than yerba mate itself

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: English

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: This result is corroborated by ESPN's coverage of the 2025 tournament, which explicitly names Sabalenka as the women's champion and Anisimova as the runner-up

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: This year's Passover (Pesach) begins at sundown on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d4, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Her citation recognized outstanding contributions to the dynamics and geometry of Riemann surfaces and their moduli spaces her work involved calculating the number of simple closed geodesics on hyperbolic surfaces

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It has lasted over 1,000 days as of mid-2026, caused over 1 million casualties resulted in Ukraine's population declining by over 10 million — roughly a quarter of its total population

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Bismuth

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Joe Biden visited Russia as U.S. President on June 16, 2021, when he met with Russian President Vladimir Putin at Villa La Grange in Geneva, Switzerland

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: This finding superseded earlier records of oldest DNA, including a million-year-old mammoth tooth and environmental DNA from Antarctic sediments is currently considered the oldest DNA discovered so far

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: She won over Team Kelly's Liv Ciara and Lucas West (Team Legend) to claim the title, earning a recording contract with Universal Music Group and a $100,000 cash prize

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This victory is further corroborated by the fact that the subsequent Season 30 announcement references her win as the most recent champion

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This film, which follows a multi-generational American saga of political resistance, won six Oscars including Best Director and Best Adapted Screenplay, marking Anderson's first Academy trophy

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The 98th Academy Awards also saw other notable wins, including Michael B. Jordan taking Best Actor and Autumn Durald Arkapaw becoming the first female director of photography to win the award

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Andrés Iniesta (2012)

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Laika

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: This birthplace, including his biography on Wikipedia, his entry on DBpedia his full birth certificate

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

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

### Sample freshqa_d4d59d75b206

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
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Johnson's own Senate election in 1875 is a separate event from his presidency, as noted by the source

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10
- **Supporting Docs Found**: d5, d7
- **Claim**: Japanese colonial rule of Korea ended in 1945, at the conclusion of World War II

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: Stanford University, on the other hand, is a private research university located in Stanford, California is thus not the institution referenced by the query

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

### Sample hotpotqa_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: 569

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Bartholdi's design was commissioned by French historian Édouard de Laboulaye, who proposed the monument to commemorate the upcoming centennial of U.S. independence in 1876

### Sample qacc_2243f17ccc38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Leeds United's FA Cup win in 1972 is further corroborated by their broader history, which lists the 1972-73 season as the year of their FA Cup runner-up and League Champions (2nd time)

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The film's director, Benh Zeitlin, aimed to keep the production as authentic as possible, meaning that when actors appear in bodies of water, it is actually the ocean live animals were used on set

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central lymphatic capillaries in intestinal villi responsible for absorbing dietary fats and fat-soluble vitamins

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The Canada Act of 1982 is also considered a key milestone on the path to full independence, as it provided for the first time a process by which Canada's basic constitutional laws could be legally amended without action by the British Parliament

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d4, d1, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: These numbers reflect different levels of organization and complexity across the eukaryotic kingdom, with d1 referring to a general count in complex eukaryotes and d4 specifying the range for humans

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Argentina is the largest country in the region both in population and land area, accounting for nearly half of the total population of the Southern Cone, further influencing the region's dominant ethnic makeup

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: 245

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: President Hoover and his wife continued the party in another area of the house after the child guests left about 10:00 pm

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Seth MacFarlane

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The balance sheet is central to the accounting system the accounting equation is fundamental to understanding how transactions affect it

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Rangers last participated in the Champions League during the 2022–23 season

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells played Wez, the character with a mohawk in The Road Warrior

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: 7

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: The Villages is an active retirement community with its own internal governance and infrastructure, though it is zoned to local school districts in Sumter, Lake Marion counties for educational purposes

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

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 2006

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This cultural shift was reinforced by 19th- and 20th-century immigration patterns and the development of a vast American coffee infrastructure, cementing coffee as the dominant morning beverage in the United States

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The switch was not universal — Southern Americans continued to drink sweet tea and some immigrant communities retained their traditional tea-drinking habits — but it became the mainstream American preference

### Sample situatedqa_temp_32d33d503f69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This is a one-off film featuring Jace Norman reprising his role as Henry Hart/Kid Danger also includes Ella Anderson, Sean Ryan Fox Frankie Grande

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: These differences reflect the distinction between absolute output and standard of living: Nigeria's large population gives it the largest total GDP, while Seychelles' small population and high-value services sector drive the highest GDP per capita

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida Gators

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Mort is a Goodman's mouse lemur (also known as a Goodman's mouse lemur)

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: The latest Android version is Android 16

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Understanding this is important for music theory and sheet music reading

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is confirmed by the episode title "An Astounding, Great Transformation!!

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: These distinct definitions reflect different eras and classification systems used by the same type of vessel

### Sample situatedqa_temp_7cd18101326e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: This result, noting that no player has repeated as MVP in consecutive years, making the 2026 winners the most recent

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d4, d1
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2
- **Claim**: HMS Queen Elizabeth was commissioned on 7 December 2017 , with the White Ensign raised at Portsmouth naval base was formally declared operational in 2020

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is the lead ship of the Queen Elizabeth class her commissioning was described as the start of a 'hugely significant chapter for the Royal Navy'

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The name is traced to the grandson of Edward the Confessor and was first recorded in the Domesday Book of 1086 , with early bearers including a Lord Chancellor of England and an Archbishop of York

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: 164

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: The retrieved evidence supports 1274 BC as the date of the Battle of Kadesh

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: He achieved this by defeating Anthony Joshua to win the WBA and IBF titles in 2021 later added the WBO title after defeating Daniel Dubois in July 2025

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. (formerly Google)

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Australia

### Sample wikirevision_0111

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The rebranding from Facebook, Inc. to Meta Platforms, Inc. occurred in 2021, as noted in the snippet from Meta Platforms is further corroborated by the 2026 timestamp on the newer Wikipedia revision of Facebook's article


================================================================================

*Report generated by CATS v2.0*
