# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.976 (over 736 samples)

**GR F1** *(used in CATS)*: 0.985

**Behavior Adherence**: 0.877 (over 609 applicable samples)

**Factual Grounding**: 0.858 (over 609 applicable samples)

**Single-Truth Recall**: 0.727 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.862

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
- **Behavior**: 0.961 (n=154)
- **Grounding**: 0.917 (n=154)
- **Recall**: 0.877 (n=154)
- **CATS**: 0.938

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.876 (n=177)
- **Grounding**: 0.842 (n=177)
- **Recall**: 0.641 (n=156)
- **CATS**: 0.833

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.729 (n=96)
- **Grounding**: 0.831 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.850

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.876 (n=145)
- **Grounding**: 0.859 (n=145)
- **Recall**: 0.707 (n=140)
- **CATS**: 0.859

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.892
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.919 (n=37)
- **Grounding**: 0.748 (n=37)
- **Recall**: 0.541 (n=37)
- **CATS**: 0.788


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2040

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/deepseek-r1-distill-32b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2040
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: Weight lifting temporarily increases blood pressure during the actual lift — sometimes reaching 200 mmHg or higher — but regular weight training is not generally a cause of high blood pressure; it can help reduce both systolic and diastolic pressure over the long term

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: The retrieved evidence is mixed. Some sources argue that anyone can become an entrepreneur with the right mindset and preparation, while others argue that entrepreneurship is not for everyone due to the high level of risk, uncertainty difficulty

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: A multi-faceted approach including stricter regulations, better enforcement public awareness is considered necessary to address the harms caused by unethical breeding while supporting responsible breeders

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Yes, cows do have four stomach compartments — the rumen, reticulum, omasum abomasum — but they are technically considered to have one stomach

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: The Harvard School of Public Health cautions that fluoride's benefits on dental health do not necessarily justify adding it to drinking water, given these risks — particularly for bottle-fed infants, the elderly people with kidney disease — and calls for further research on safe dosing and vulnerable populations

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: The conflict is factual, with d4 and d1 affirming inheritance via germline transmission, d5 denying it due to demethylation d2 and d3 offering partial mechanistic context without resolving the hereditary question

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Some sources note that IPv6's built-in security features give it a 'basic' edge over IPv4, though real-world performance tests have shown similar results and IPv4 has still been faster in some instances

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Scaled feet and rigid first two digits in the hand, consistent with frequent ground walking, but with a third digit capable of movement suggesting tree climbing was also possible

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence presents a genuine debate on whether audiobooks count as real reading, with some sources arguing they are fully equivalent to physical books and others doubting their legitimacy

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Additionally, researchers have found 266 previously undocumented small ridges on the Moon's far side that are younger than those on the near side, suggesting recent geological activity there as well

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: A middle-ground view holds that trophy hunting can be beneficial when rigorously regulated and compared to other land uses, but that unmanaged or poorly regulated hunting can be harmful

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d3
- **Claim**: At the same time, students and teachers do retain certain constitutional rights: the Supreme Court has upheld the Pledge of Allegiance as constitutional and allowed Bible studies during non-instructional time as long as they are not led by school personnel and do not constitute an overt expression of a particular religion students are permitted to pray privately and quietly by themselves, dress according to their religious faith participate in student-led prayer groups outside of class without school sponsorship or organization

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: The Kidney Disease Improving Global Outcomes (KDIGO) guidelines recommend bicarbonate supplementation only when serum bicarbonate levels fall below 18 mEq/L the evidence is considered insufficient to fully support routine use in all CKD stages

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Multiple sources consistently support this ranking, with no contradictions across documents

### Sample conflictingqa_63fde268aa8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict here reflects a methodological divergence between philosophical speculation and scientific evidence, with no single definitive answer

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3
- **Claim**: The Gutenberg Bible was not the first book ever printed with movable type — that distinction belongs to Jikji, a Korean Buddhist text printed in 1377 — but it was the first major book printed in Europe using mass-produced metal movable type the first to be commercially successful in the West

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bees can fly in light rain or when absolutely necessary, but prefer to stay dry and return to the hive during heavy rain

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The question of whether the Catholic Church is the true church is therefore a matter of faith and doctrine that requires a deeper examination of scriptural teaching, church history theological tradition

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: From a Bahá'í perspective, cultural differences are not hindrances but rather secondary to humanity's spiritual unity the faith teaches that these differences can be overcome

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: Spelunking and caving are related but distinct terms: spelunking typically refers to casual, recreational cave exploration by hobbyists, while caving encompasses a broader range of activities from family-friendly tours to advanced expeditions

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d4
- **Claim**: A middle-ground view holds that the benefits and risks must be weighed individually for each pet depending on factors like breed, sex, age health status

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2, d3
- **Claim**: A 2010 book by Victoria Braithwaite argues that fish are more intelligent than previously thought and possess brain structures that allow them to feel pain, while Dr. J. Rose of the University of Wyoming counters that fish perceive pain and fear very differently from humans, distinguishing between the physical reception of harmful stimuli (nociception) and the subjective experience of pain

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Methodological differences—such as the EPA's larger dataset of 15 carcinogenicity studies compared to IARC's 8—underlie these opposing research conclusions, with high-credibility sources on both sides contributing to a nuanced and contested body of evidence

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The conflicting expert opinions and descriptions of different mechanisms mean there is no single definitive answer; cold water may have some limited benefits for hair appearance but will not fundamentally change its natural shine

### Sample conflictingqa_be17259fe5c0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: While penguins do not live in Antarctica today , their modern distribution across the Southern Hemisphere is consistent with an Australian/New Zealand origin, as earlier scientists had proposed

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: A balanced approach that considers the full lifecycle of both materials—production, use disposal—suggests that neither paper nor plastic straws are definitively better than the other that reusable alternatives like metal or glass straws may be the most environmentally responsible choice of all

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: The evidence is divided on whether coffee grounds are an effective slug and snail deterrent. Some sources argue that coffee grounds are a useful, non-toxic environmentally friendly way to repel slugs and snails, while others argue that the caffeine content in coffee grounds is too low to reliably deter slugs and snails that cold coffee or coffee extracts are more effective

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2, d1, d3
- **Claim**: The yield gap is not universal, as some crop types and management practices perform more closely to conventional standards, but the overall body of evidence points to a significant and widespread productivity advantage of conventional farming

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The retrieved evidence is mixed. Some sources argue that solar panels can produce more energy than they consume over their lifetime

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
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

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: This year's Passover (Pesach) begins at sundown on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
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
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The 98th Academy Awards also saw other notable wins, including Michael B. Jordan taking Best Actor and Autumn Durald Arkapaw becoming the first female director of photography to win the award

### Sample freshqa_7dce5d575302

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
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

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: 8 September 2022

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Jiangsu

### Sample freshqa_c479e83e408f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
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
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 9 minutes

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5, d3
- **Claim**: Ta-Nehisi Coates

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10
- **Supporting Docs Found**: d7, d5
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

### Sample hotpotqa_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d5, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: 569

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4, d2
- **Claim**: The ultimate goal throughout had always been to defeat Germany the North African campaign had provided a crucial stepping stone for that final push into Europe

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The film's director, Benh Zeitlin, aimed to keep the production as authentic as possible, meaning that when actors appear in bodies of water, it is actually the ocean live animals were used on set

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Phil Jackson (coaching), Bill Russell (playing), Red Auerbach (coaching/executive)

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The St. Louis Rams won Super Bowl XXXIV on January 30, 2000, defeating the Tennessee Titans 23-16

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5, d3
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central lymphatic capillaries in intestinal villi responsible for absorbing dietary fats and fat-soluble vitamins

### Sample qacc_4387048ed24f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2, d3
- **Claim**: The Academy data confirms Bette Davis was only nominated for Best Actress for the film, not that she won it Ryan Murphy's account further corroborates that Davis was devastated by the loss

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

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Argentina is the largest country in the region both in population and land area, accounting for nearly half of the total population of the Southern Cone, further influencing the region's dominant ethnic makeup

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: 245

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5, d3
- **Claim**: Hosanna is a Hebrew word (hosha‘na') used as a cry for salvation or help, often translated into English as "save now" or "save us now." In both Hebrew and Greek, the word is consistently rendered as a plea for deliverance it is this salvific meaning that underlies its use in the New Testament accounts of Jesus's entry into Jerusalem

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: President Hoover and his wife continued the party in another area of the house after the child guests left about 10:00 pm

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Elton Hayes

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5, d3
- **Claim**: July 4, 1776

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The balance sheet is central to the accounting system the accounting equation is fundamental to understanding how transactions affect it

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
- **Supporting Docs Found**: d4, d3
- **Claim**: Vernon Wells played Wez, the character with a mohawk in The Road Warrior

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5, d3
- **Claim**: An initialism is a type of abbreviation formed from the initial letters of a phrase, but unlike acronyms, initialisms are pronounced letter by letter rather than as a word. Examples of initialisms include DNA, RT-PCR FBI, where each letter is pronounced individually. The term 'initialism' is older than 'acronym' and refers specifically to abbreviations that are pronounced as individual letters, not as a word

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5, d3
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

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 2006

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The answer is limited to cities and does not include urban agglomerations or metropolitan areas, though some sources (such as Tokyo) include metro data to reflect the full population

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

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Novak Djokovic (Serbia) has won the most Grand Slam titles in tennis history with 24

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

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
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

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d1
- **Claim**: These distinct definitions reflect different eras and classification systems used by the same type of vessel

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This result, noting that no player has repeated as MVP in consecutive years, making the 2026 winners the most recent

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence supports 1274 BC as the date of the Battle of Kadesh

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: He achieved this by defeating Anthony Joshua to win the WBA and IBF titles in 2021 later added the WBO title after defeating Daniel Dubois in July 2025

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d5, d3
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

### Sample wikirevision_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Microsoft owns LinkedIn. Microsoft acquired LinkedIn in December 2016, making LinkedIn a subsidiary of Microsoft. LinkedIn Corporation is a subsidiary of Microsoft, with Jeff Weiner serving as Executive Chairman and Daniel Shapero as CEO

### Sample wikirevision_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current President of Turkey is Recep Tayyip Erdoğan, who has been in office since 28 August 2014. He is the 12th President of Turkey and serves as both head of state and head of government, with Cevdet Yılmaz as his Vice President since 4 June 2023

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Ballon d'Or winner is Ousmane Dembélé (France), who won the 69th Ballon d'Or ceremony in 2025, marking his first win. This is confirmed across multiple sources, including the high-credibility Wikipedia articles on the 2025 Ballon d'Or and the award's main entry

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Calcutta is officially called Kolkata. The city officially changed its name from Calcutta to Kolkata in 2001 this change is recognized across all sources. The current official name is Kolkata it has been so since 2001 when the city changed its name from Calcutta

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Australia

### Sample wikirevision_0111

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
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
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has been in office since 23 May 2022. He is the 31st person to hold the role since the office was created in 1901 is appointed by the Governor-General on the advice of the incumbent

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The latest Ballon d'Or winner is Ousmane Dembélé (France, 2025), the holder of the 69th Ballon d'Or awarded by France Football. He earned his first Ballon d'Or at the 2025 ceremony, surpassing Lionel Messi's record of 8 awards

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The current President of the Philippines is Bongbong Marcos, who has been in office since June 30, 2022. He is the 17th President of the Philippines and serves as both head of state and head of government, as well as commander-in-chief of the country's armed forces. This is consistent across multiple sources, including the Wikipedia article on the President of the Philippines, which describes him as the incumbent with an incumbency date of June 30, 2022

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current President of India is Droupadi Murmu, who took office in 2022. She is the 15th President of India and serves as the head of state of the Republic of India and the supreme commander of the Indian Armed Forces. This is consistent across multiple sources, including the official Government of India Press Information Bureau and the newer Wikipedia revision of the President of India article

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This victory is confirmed across multiple sources, with the 2026 French Open representing the next iteration of the tournament


================================================================================

*Report generated by CATS v2.0*
