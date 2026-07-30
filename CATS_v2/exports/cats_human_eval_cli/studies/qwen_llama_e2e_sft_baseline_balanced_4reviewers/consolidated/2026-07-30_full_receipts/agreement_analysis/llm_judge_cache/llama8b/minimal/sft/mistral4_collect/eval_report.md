# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.976 (over 736 samples)

**GR F1** *(used in CATS)*: 0.985

**Behavior Adherence**: 0.658 (over 609 applicable samples)

**Factual Grounding**: 0.883 (over 609 applicable samples)

**Single-Truth Recall**: 0.485 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.753

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
- **Behavior**: 0.877 (n=154)
- **Grounding**: 0.930 (n=154)
- **Recall**: 0.435 (n=154)
- **CATS**: 0.810

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.718 (n=177)
- **Grounding**: 0.881 (n=177)
- **Recall**: 0.429 (n=156)
- **CATS**: 0.750

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.698 (n=96)
- **Grounding**: 0.862 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.850

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.400 (n=145)
- **Grounding**: 0.877 (n=145)
- **Recall**: 0.629 (n=140)
- **CATS**: 0.725

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.892
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.378 (n=37)
- **Grounding**: 0.768 (n=37)
- **Recall**: 0.378 (n=37)
- **CATS**: 0.617


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2040

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/mistral-small-4

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2040
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

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
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: The conflict is factual, with d4 and d1 affirming inheritance via germline transmission, d5 denying it due to demethylation d2 and d3 offering partial mechanistic context without resolving the hereditary question

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: One security area where IPv6 is clearly superior is in privacy protection — with 340 undecillion possible addresses, it is mathematically impossible for IPv6 to be less private than IPv4

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Additionally, researchers have found 266 previously undocumented small ridges on the Moon's far side that are younger than those on the near side, suggesting recent geological activity there as well

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3, d2
- **Claim**: A middle-ground view holds that trophy hunting can be beneficial when rigorously regulated and compared to other land uses, but that unmanaged or poorly regulated hunting can be harmful

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: The Kidney Disease Improving Global Outcomes (KDIGO) guidelines recommend bicarbonate supplementation only when serum bicarbonate levels fall below 18 mEq/L the evidence is considered insufficient to fully support routine use in all CKD stages

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Multiple sources consistently support this ranking, with no contradictions across documents

### Sample conflictingqa_63fde268aa8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict here reflects a methodological divergence between philosophical speculation and scientific evidence, with no single definitive answer

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, supporters of organic farming argue that these yield differences do not necessarily translate into greater environmental benefits, as both systems use chemicals—approved and unapproved—and that organic farming offers distinct sustainability advantages in areas like soil health and biodiversity

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The question of whether the Catholic Church is the true church is therefore a matter of faith and doctrine that requires a deeper examination of scriptural teaching, church history theological tradition

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Spelunking and caving are related but distinct terms: spelunking typically refers to casual, recreational cave exploration by hobbyists, while caving encompasses a broader range of activities from family-friendly tours to advanced expeditions

### Sample conflictingqa_8efa53ba7c60

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This is further confirmed by the Las Vegas Natural History Museum, which states that paleontologists have proven dinosaurs are the ancestors of modern birds that both T-Rex and Velociraptors were feathered theropods

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: A middle-ground view holds that the benefits and risks must be weighed individually for each pet depending on factors like breed, sex, age health status

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: A 2010 book by Victoria Braithwaite argues that fish are more intelligent than previously thought and possess brain structures that allow them to feel pain, while Dr. J. Rose of the University of Wyoming counters that fish perceive pain and fear very differently from humans, distinguishing between the physical reception of harmful stimuli (nociception) and the subjective experience of pain

### Sample conflictingqa_9275911a2961

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, it is worth noting that at normal doses, the risk is typically not a concern kidney stone formation is more closely tied to the total amount of calcium (from antacids and supplements combined) rather than antacid use alone

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Affirmative action can be argued to constitute reverse discrimination when it gives preferential treatment to historically disadvantaged groups at the expense of those who are not similarly situated, such as working-class white Americans. This view holds that affirmative action discriminates against those who are not receiving the benefits, even if the overall goal is to address past racism

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Methodological differences—such as the EPA's larger dataset of 15 carcinogenicity studies compared to IARC's 8—underlie these opposing research conclusions, with high-credibility sources on both sides contributing to a nuanced and contested body of evidence

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: A balanced approach that considers the full lifecycle of both materials—production, use disposal—suggests that neither paper nor plastic straws are definitively better than the other that reusable alternatives like metal or glass straws may be the most environmentally responsible choice of all

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: The evidence is mixed. Some high-credibility sources (NIH/NLM, Verywell Health, GoodRx) report associations between hot yerba mate consumption and increased risk of esophageal cancer, while also noting that yerba mate may have anti-cancer properties and could potentially reduce the risk of some cancers. The NIH review found that studies linking mate to cancer risk were methodologically similar but produced widely varying results, with the strongest association seen in women drinking large quantities at high temperature. Overall, the most credible and comprehensive evidence suggests that drinking yerba mate at very hot temperatures is the primary driver of cancer risk, rather than yerba mate itself

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: English

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: This result is corroborated by ESPN's coverage of the 2025 tournament, which explicitly names Sabalenka as the women's champion and Anisimova as the runner-up

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This year's Passover (Pesach) begins at sundown on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d1, d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Her citation recognized outstanding contributions to the dynamics and geometry of Riemann surfaces and their moduli spaces her work involved calculating the number of simple closed geodesics on hyperbolic surfaces

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Joe Biden visited Russia as U.S. President on June 16, 2021, when he met with Russian President Vladimir Putin at Villa La Grange in Geneva, Switzerland

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This victory is further corroborated by the fact that the subsequent Season 30 announcement references her win as the most recent champion

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The 98th Academy Awards also saw other notable wins, including Michael B. Jordan taking Best Actor and Autumn Durald Arkapaw becoming the first female director of photography to win the award

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2
- **Claim**: Their most recent World Series appearance was in 2022, when they lost to the Philadelphia Phillies, making their total count remain at one title

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2
- **Claim**: This birthplace, including his biography on Wikipedia, his entry on DBpedia his full birth certificate

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
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

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is the latest data available in the retrieved evidence, superseding older records from 2019–20 and beyond

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
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Ta-Nehisi Coates

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0100

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9
- **Supporting Docs Found**: d6
- **Claim**: Children's National Medical Center is a notable not-for-profit children's hospital in D.C. that is ranked among the top 10 children's hospitals in the country by U.S. News & World Report, but it is not the largest private hospital overall in D.C

### Sample hotpotqa_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d1
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

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The ultimate goal throughout had always been to defeat Germany the North African campaign had provided a crucial stepping stone for that final push into Europe

### Sample qacc_0d85f1089c4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: These are distinct designations for different geographic areas no single person has been chosen as the universal brand ambassador for the campaign as a whole

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The film's director, Benh Zeitlin, aimed to keep the production as authentic as possible, meaning that when actors appear in bodies of water, it is actually the ocean live animals were used on set

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central lymphatic capillaries in intestinal villi responsible for absorbing dietary fats and fat-soluble vitamins

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2, d3, d1
- **Claim**: Gagarin's feat of orbiting the Earth marked a significant milestone in the competition the Soviets maintained the lead in human spaceflight until the U.S. eventually caught up and surpassed them with the Apollo program

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The Canada Act of 1982 is also considered a key milestone on the path to full independence, as it provided for the first time a process by which Canada's basic constitutional laws could be legally amended without action by the British Parliament

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: These numbers reflect different levels of organization and complexity across the eukaryotic kingdom, with d1 referring to a general count in complex eukaryotes and d4 specifying the range for humans

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Together, the evidence supports a contextual scope model where origin number is not fixed across all eukaryotes but rather reflects species-specific biology

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Argentina is the largest country in the region both in population and land area, accounting for nearly half of the total population of the Southern Cone, further influencing the region's dominant ethnic makeup

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: 245

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Hosanna is a Hebrew word (hosha‘na') used as a cry for salvation or help, often translated into English as "save now" or "save us now." In both Hebrew and Greek, the word is consistently rendered as a plea for deliverance it is this salvific meaning that underlies its use in the New Testament accounts of Jesus's entry into Jerusalem

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
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

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: July 4, 1776

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The balance sheet is central to the accounting system the accounting equation is fundamental to understanding how transactions affect it

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: XXXTENTACION

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Rangers last participated in the Champions League during the 2022–23 season

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells played Wez, the character with a mohawk in The Road Warrior

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: 7

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

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: The Articles of Confederation was eventually superseded by the United States Constitution, which was drafted in 1787 and ratified in 1788, though the exact date when the Constitution replaced the Articles of Confederation is not explicitly stated in the available evidence

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida Gators

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Understanding this is important for music theory and sheet music reading

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This is confirmed by the episode title "An Astounding, Great Transformation!!

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: These distinct definitions reflect different eras and classification systems used by the same type of vessel

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This result, noting that no player has repeated as MVP in consecutive years, making the 2026 winners the most recent

### Sample situatedqa_temp_8808c106a115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4, d2
- **Supporting Docs Found**: None
- **Claim**: The disease causes progressive neurological damage due to the accumulation of GM2-ganglioside within brain and nerve cells it is most commonly found in Ashkenazi Jewish, French Canadian Cajun populations, though it can affect anyone

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 9 U.S. cities have hosted the Olympics and Winter Games

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: 164

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: He achieved this by defeating Anthony Joshua to win the WBA and IBF titles in 2021 later added the WBO title after defeating Daniel Dubois in July 2025

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving a five-year term that is renewable once consecutively resides at Bellevue Palace in Berlin

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has been in office since 23 May 2022. He is the 31st person to hold the role since the office was created in 1901 is appointed by the Governor-General on the advice of the incumbent

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The latest President of Mexico is Claudia Sheinbaum, who became the 66th President of Mexico on 1 October 2024. She is the first woman and the first Jewish person to hold the office she previously served as Head of Government of Mexico City from 2018 to 2023

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The current President of the Philippines is Bongbong Marcos, who has been in office since June 30, 2022. He is the 17th President of the Philippines and serves as both head of state and head of government, as well as commander-in-chief of the country's armed forces. This is consistent across multiple sources, including the Wikipedia article on the President of the Philippines, which describes him as the incumbent with an incumbency date of June 30, 2022

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d4
- **Claim**: This victory is confirmed across multiple sources, with the 2026 French Open representing the current year's edition and Alcaraz's 2025 title defense making him the latest champion


================================================================================

*Report generated by CATS v2.0*
