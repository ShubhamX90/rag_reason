# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 124 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.973 (over 736 samples)

**GR F1** *(used in CATS)*: 0.983

**Behavior Adherence**: 0.825 (over 612 applicable samples)

**Factual Grounding**: 0.888 (over 612 applicable samples)

**Single-Truth Recall**: 0.745 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.860

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.983
- **Precision**: 0.993
- **Recall**: 0.974
- **Accuracy**: 0.973
- TP=592, FP=4, FN=16, TN=124

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.886
- **Abstain Recall**: 0.969
- **Abstain F1**: 0.925
- **Specificity**: 0.974
- Abstain TP=124, FP=16, FN=4, TN=592


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (55 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.949 (n=156)
- **Grounding**: 0.926 (n=156)
- **Recall**: 0.844 (n=154)
- **CATS**: 0.927

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.964
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.882 (n=178)
- **Grounding**: 0.851 (n=178)
- **Recall**: 0.660 (n=156)
- **CATS**: 0.842

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.698 (n=96)
- **Grounding**: 0.874 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.854

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.724 (n=145)
- **Grounding**: 0.926 (n=145)
- **Recall**: 0.743 (n=140)
- **CATS**: 0.847

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.757 (n=37)
- **Grounding**: 0.793 (n=37)
- **Recall**: 0.703 (n=37)
- **CATS**: 0.795


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2139

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes, many salamanders are poisonous to touch because they possess skin glands that produce toxins to defend themselves handling them can transfer those toxins to humans; however, the amount of toxin varies by species some — like tiger salamanders — are considered harmless to humans with proper hand-washing

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: Weight lifting causes a temporary increase in blood pressure during the actual lifting action — sometimes climbing as high as 200 mmHg — but the long-term effects are generally positive: regular strength training can lower resting blood pressure and reduce cardiovascular risk

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The conflicting views reflect differing opinions on whether entrepreneurship is innate or learned the answer to whether anyone can become an entrepreneur depends on which perspective one accepts

### Sample conflictingqa_151865dc414b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: A multi-disciplinary approach involving interventional radiology, otolaryngology audiology is typically required to determine the cause and develop an effective treatment plan

### Sample conflictingqa_2500bfd10608

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d2
- **Supporting Docs Found**: None
- **Claim**: The debate is further complicated by methodological differences—most studies linking fluoride to harm have used populations with exposures far higher than typical U.S. water supplies, making the evidence for safe levels context-dependent and contested

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: The evidence is mixed. Some studies and experts claim wrist rests can reduce wrist extension angles by 10–15 degrees and lower reported discomfort by up to 30% when used correctly, while others argue that wrist rests are not necessary for good ergonomics and can be harmful if used improperly or for extended periods

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The debate is further complicated by the fact that many schools and institutions do not count audiobooks toward reading goals some sources note that while audiobooks may count for personal reading goals, they do not substitute for the focused attention required to read physical books

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: Ultimately, whether or not audiobooks are considered "real" reading depends on how one defines reading—whether by medium, by attention by functional purpose

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Moon has long been considered geologically inactive, with volcanism ceasing about 3 billion years ago and a magnetic field disappearing between 2.5 and 1 billion years ago

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: The evidence on fish oil for heart disease is mixed and the strongest evidence comes from eating fish directly rather than taking supplements

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The debate is ongoing, with no consensus yet on whether emoji will eventually develop into a fully recognized language or remain a supplementary tool for digital communication

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: The evidence is mixed. Some sources argue that trophy hunting can benefit conservation by generating revenue, controlling wildlife populations funding anti-poaching efforts, while others argue it is morally inappropriate and that bans are not harmful but rather beneficial to conservation

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: A 2020 Pew Research Center analysis found that across 22 developed economies, the median earnings gap was 15% among workers aged 25–54 in the United States, the gap was 18% among women and men with a bachelor's degree or higher

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d1, d3
- **Claim**: Multiple sources note that the 1815 eruption triggered the 'Year Without a Summer' of 1816, which caused widespread crop failures and famine across the globe, further complicating direct comparison with other events

### Sample conflictingqa_613a0093714b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: The phrase itself was popularized by Jonathan Swift's 1738 satire, but whether Swift coined it or was reusing a common expression remains unclear

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Additionally, bees use their sensitive body hairs to detect approaching rain and will often return to the hive before the rain becomes heavy, as nectar can become diluted and pollen can wash away in heavy rain

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: A third perspective holds that the key issue is not yield per se, but rather land use and biodiversity conservation, where high-yield conventional farming can be better for the environment even if it uses more resources

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Overall, the evidence does not establish a clear, general answer that organic farming is universally more or less efficient than conventional farming

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: However, other sources note that neutering also provides significant benefits—such as preventing ovarian and prostate cancers, reducing roaming and aggression eliminating life-threatening diseases like pyometras —so the overall health impact depends heavily on factors like age, breed sex

### Sample conflictingqa_962d8f5d5574

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological and interpretive, arising because sources differ in the scope of species studied and the criteria used to define'swimming ability,' with some sources counting partial or undetected ability as a negative result and others treating it as absence of evidence

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Nutritional yeast is not a complete protein source for vegans on its own. It is, however, an excellent source of highly digestible protein and B vitamins when consumed as part of a varied diet with other plant-based foods, it can help fill the protein gap for vegans

### Sample conflictingqa_f22b389be1d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The debate between barefoot and shoed running is ongoing; there is no settled scientific consensus on which is healthier

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: The retrieved evidence is mixed and does not support a definitive answer

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: Scientists and researchers have long been interested in whether animals can predict earthquakes, with some studies suggesting that animals can detect the P-wave vibrations from an earthquake before the larger S-wave causes damage

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence supports both Dutch and British claims to have first discovered Australia, with the Dutch being the first Europeans to land on the continent and the British being the first to establish a colony

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: The evidence is mixed and the answer depends on the temperature at which yerba mate is consumed. Some studies suggest that drinking very hot mate increases the risk of esophageal cancer, while other research indicates that yerba mate may have anti-cancer properties and could help kill cancer cells

### Sample conflictingqa_f970957c5e52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d2, d1, d5
- **Supporting Docs Found**: None
- **Claim**: The conflict arises because present a nuanced, context-dependent view, while d4 and d5 present opposing interpretive conclusions on the same issue

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: This ranking is consistent across multiple sources, including the most recent data from Ethnologue and Visual Capitalist, which place Hindi at #3 with 600 million+ total speakers , surpassing both Spanish (#4 with 560 million+) and French (#6 with 310 million+)

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_0436c0b3a9d7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable Android version is Android 16

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

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The latest Academy Award for Best Picture was won by *Anora* (2025), directed by Sean Baker, which won the award at the 98th Academy Awards

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
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: He spent his early years in Bayonne, a city in New Jersey it was there that his world was shaped by the five blocks between his grade school and home

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The cause of death is explicitly stated as a boating accident, not a heart attack as sometimes incorrectly claimed

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: Jeff Bezos has not sold Amazon; he sold a portion of his Amazon shares in 2025

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3, d5
- **Claim**: Earlier releases, such as macOS Sonoma 14 and macOS Monterey 12 , are no longer the latest versions and have been superseded by macOS 26 Tahoe

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: After defeating the Axis powers in North Africa, the Allies proceeded in several directions. The most immediate next step was the invasion of Sicily, which began on July 10, 1943, just a few months after the North African campaign ended

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d1, d5
- **Claim**: These venues for different productions, with the Pantages being the historic home of the show and the Princess of Wales hosting a more recent touring production

### Sample qacc_1a764b8b6cf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This period spanned from 632 CE, when Abu Bakr became the first Muslim Caliph following Muhammad's death, until 661 CE, when Ali's reign was cut short

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: All vertebrates — including amphibians, reptiles, birds mammals — descended from these fish, making them the foundational group of vertebrates

### Sample qacc_2e1b5edb5e0d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2
- **Claim**: This is consistent with the broader understanding that skin thickness varies by body region, with the presence or absence of the stratum lucidum being a key distinguishing feature between thick and thin skin types

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The film's director, Benh Zeitlin, aimed to keep the production entirely authentic, using real water and live animals whenever possible, further grounding the story in the region

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: Additional contextual support comes from the Hollywood Reporter's confirmation that the film is set in a fictional fishing colony off the Louisiana coast known as 'the Bathtub' from the broader New Orleans setting referenced throughout the review

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

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: The Airdrome

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The two modules docked in space on December 5, 1998, marking the initial assembly of the ISS

### Sample qacc_a3c882e062c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The basilica has never been fully finished, as Gaudí died in 1926 before completing the design successive builders have worked on the project in stages

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The remaining one third is located in the extracellular space, which includes fluids outside cells and in blood plasma

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: All body systems rely on this water distribution, as intracellular fluid bathes and nourishes cells directly, while extracellular fluid helps regulate blood pressure and remove waste

### Sample qacc_a4dde83c35da

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The government relied on a bureaucracy of scholars selected through rigorous examinations, as well as a network of eunuchs and spies, to manage provincial affairs and contain any dissent

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: The Rajya Sabha currently has 245 members, of which 233 are elected and 12 are nominated by the President. This is consistent with the constitutional limit of 250 members, which is divided between 238 elected by state legislative assemblies and 12 nominated for exceptional contributions to art, literature, science social service

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The statehood date is consistent across multiple sources, with the Gazette of the Republic and the Congressional Record both confirming January 6, 1912 as the date when President William Taft signed the New Mexico statehood bill

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This joint allows movement in two planes, providing flexion, extension, abduction, adduction circumduction is further characterized by a joint capsule filled with synovial fluid that lubricates the articulating surfaces

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

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

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: While the income statement and cash flow statements are also important financial statements, they do not display the balance sheet equation in the same way that the balance sheet does

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Sushma Swaraj also served as the Chief Minister of Delhi in 1998 and was the first woman spokesperson for any political party in India, further cementing her status as a trailblazer for women in Indian politics

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: At the most granular level, The Villages comprises over 50 named villages across these counties, each with its own recreation centers, town squares shopping districts

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is consistent across all 50 states, as established by the National Minimum Drinking Age Act of 1984

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: For those under 21, it is illegal to purchase alcohol and doing so can result in fines, community service even jail time

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: By 2006

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Nixon

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, the U.S. Code establishes the National Oceanic and Atmospheric Administration (NOAA) within the Department of Commerce to coordinate federal programs related to environmental quality, reflecting a broader federal commitment to environmental protection

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: British General Sir William Howe's army of about 16,000 troops defeated the Continental Army of about 15,000 in the vicinity of Chadds Ford, Pennsylvania, near Philadelphia

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Multiple election observers noted that the post-election environment was marred by allegations of widespread rigging, which PTI's incoming Prime Minister Khan promised to investigate

### Sample situatedqa_temp_6f777dda5314

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The current status is further corroborated by the fact that the older 2025 roster page from the same tracker no longer lists Kevin Stefanski as head coach, confirming Monken's succession

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that in the U.S. Navy, the prefix "USS" (United States Ship) is used for all ships, including those with the SS designation, which can sometimes cause confusion about what SS actually stands for in those contexts

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Australia has a very long coastline, but methodological differences across measurement scales and years explain the divergence in figures

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4, d3, d5
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: The three women are the daughters of famous musicians — Carnie and Wendy are Brian Wilson's daughters and Chynna is John Phillips' daughter — and their harmonies were described as'smooth, melodic lush'

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This was the first major battle between Muslims and the Quraysh Muslims emerged victorious with the help of Allah

### Sample situatedqa_temp_b797de4c6610

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: These biomarkers can help distinguish between heart attacks and other conditions with similar symptoms, such as angina or non-cardiac chest pain are used in combination with electrocardiograms (ECGs) and other tests to guide treatment and monitor cardiac function

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: This result, noting that the Panthers are the 10th franchise to win consecutive championships and the first since Tampa Bay in 2020-21

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is the lead ship of the Queen Elizabeth class, with her sister HMS Prince of Wales following in 2019

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4, d5
- **Claim**: 164

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

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The Philadelphia 76ers last made the playoffs in 2021

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Their most recent playoff appearance was in the 2020-21 NBA season, where they played a six-game series against the Atlanta Hawks

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not justify a complete answer

### Sample wikirevision_0003

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The older Twitter Wikipedia revision is still accessible but is superseded by the newer revision that confirms the name change to X

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This was their third World Cup title, their first since 1986 their first from outside of Europe since 2002

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the latest Cricket World Cup, the 2023 ICC Men's Cricket World Cup, defeating India in the final on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad. This was Australia's sixth Cricket World Cup title, making them the most successful team in the history of the tournament

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This is explicitly recognized in the Wikipedia infobox for Gurgaon, which lists its official name as Gurugram. The name change from Gurgaon to Gurugram was confirmed by the Haryana government the city is now officially known by this name

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 tournament took place from 29 June to 12 July 2026 at the All England Lawn Tennis and Croquet Club in Wimbledon, London

### Sample wikirevision_0120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This victory was his second French Open title and fifth major overall

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The 2026 French Open is scheduled to take place from 24 May to 7 June 2026, but the snippet from that event does not identify the actual 2026 men's singles champion, only noting that Carlos Alcaraz withdrew due to a wrist injury


================================================================================

*Report generated by CATS v2.0*
