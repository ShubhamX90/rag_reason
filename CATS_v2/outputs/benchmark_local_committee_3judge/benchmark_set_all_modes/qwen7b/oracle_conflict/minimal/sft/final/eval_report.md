# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 110 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.955 (over 736 samples)

**GR F1** *(used in CATS)*: 0.973

**Behavior Adherence**: 0.791 (over 626 applicable samples)

**Factual Grounding**: 0.855 (over 626 applicable samples)

**Single-Truth Recall**: 0.695 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.828

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.973
- **Precision**: 0.971
- **Recall**: 0.975
- **Accuracy**: 0.955
- TP=593, FP=18, FN=15, TN=110

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.880
- **Abstain Recall**: 0.859
- **Abstain F1**: 0.870
- **Specificity**: 0.975
- Abstain TP=110, FP=15, FN=18, TN=593


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (49 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.948
- **GR F1** *(used in CATS)*: 0.965
- **Behavior**: 0.907 (n=162)
- **Grounding**: 0.878 (n=162)
- **Recall**: 0.753 (n=154)
- **CATS**: 0.876

### Type 2: Complementary Info

- **Samples**: 221 (39 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.941
- **GR F1** *(used in CATS)*: 0.963
- **Behavior**: 0.923 (n=182)
- **Grounding**: 0.821 (n=182)
- **Recall**: 0.612 (n=156)
- **CATS**: 0.830

### Type 3: Conflicting Opinions

- **Samples**: 109 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.673 (n=98)
- **Grounding**: 0.866 (n=98)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.840

### Type 4: Outdated Info

- **Samples**: 158 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.626 (n=147)
- **Grounding**: 0.863 (n=147)
- **Recall**: 0.743 (n=140)
- **CATS**: 0.805

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.946
- **GR F1** *(used in CATS)*: 0.972
- **Behavior**: 0.595 (n=37)
- **Grounding**: 0.856 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.761


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2135

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

### Sample conflictingqa_07e290a954ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Importantly, most clinical studies used standardized extracts preparations vary significantly in potency, meaning dose recommendations can differ substantially between brands

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2, d4
- **Claim**: Chlorine can, however, bleach hair and increase its porosity, making it more susceptible to metal absorption and accelerating color fading

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: We cannot know anything beyond our minds

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: A central debate revolves around whether the financial returns justify the risks of poaching incentives and the suffering inflicted on individual animals, particularly when compared to alternative conservation strategies such as ecotourism, which has its own documented negative impacts

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The question of whether patents should apply to software is genuinely contested, with strong arguments on both sides. Opponents note that the Supreme Court's Alice v

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CLS Bank decision made software patents significantly harder to obtain and enforce that the USPTO's own guidelines treat 'abstract ideas' as ineligible unless they are 'tied to a specific machine or apparatus,' meaning software alone rarely qualifies

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: However, many products can temporarily mask the damage by smoothing the cuticle, adding weight to frayed ends creating temporary bonds between split fibers — though these effects typically persist only until the next shampoo

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: However, some states have pushed back against this: Maine passed a law prohibiting ISPs from selling personal data without individual express permission the California Consumer Privacy Act (CCPA) grants California residents the right to opt out of having their data sold

### Sample conflictingqa_80857a692531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4, d3
- **Supporting Docs Found**: None
- **Claim**: The conflicting evidence reflects genuine scientific and methodological differences in how researchers and advocacy groups interpret data on nutrient composition and contaminant levels, leading to divergent conclusions about whether farmed salmon can be considered as nutritious as its wild counterpart

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Research further suggests that most politicians exploit cultural differences to fuel divisions empirical evidence from multiple countries indicates that integration policies are difficult to implement given the sheer scale of demographic pluralism

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, some scientists argue that while the evidence for dark matter's effects is robust, the specific identity of dark matter particles remains unknown alternative explanations—such as modifications to general relativity—have not been entirely ruled out, reflecting ongoing scientific debate

### Sample conflictingqa_9251c7a33ec5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For cats, the picture is similarly complex: while spaying reduces risks of ovarian and uterine cancers and pyometra, it is also associated with increased risks of urinary incontinence and certain types of obesity the overall net health impact is subject to ongoing scientific debate

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Not all snakes can swim

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d5, d2, d3
- **Claim**: Overall, while sexual activity represents the most common and efficient route of transmission, gonorrhea demonstrates that infection can occur via non-sexual means, albeit far less frequently

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Some plants can survive without light for limited periods — typically a few weeks — by relying on stored energy in their roots, stems leaves, though they will eventually die if light is permanently denied

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Stalactites can form underwater, though not directly from dripping water as they do in caves above sea level

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Contemporary data collected immediately after the broadcast showed that virtually no one thought it was real surveys found only about 2–3 percent of the national audience was tuned in to Mercury Theatre on the Air on the evening of 30 October 1938

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some researchers remain cautious, acknowledging that while volcanic carbon release was probable, it has not yet been definitively established as the sole or exclusive trigger, as the PETM onset also coincided with a mercury low, hinting at the possible involvement of additional carbon reservoirs such as methane-rich ocean sediments or organic-rich permafrost

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Research has further shown that green tea extract can change the shape of calcium oxalate crystals, making them flatter and more fragile — characteristics that the authors argue could prevent the formation of clinical stones that the caffeine in green tea may also help flush the bladder and reduce oxalate absorption

### Sample conflictingqa_bd2e652cd64d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: They are especially beneficial for people with arthritis, orthodontic appliances limited hand mobility, as the powered head eliminates the need for precise manual control

### Sample conflictingqa_be17259fe5c0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d3
- **Claim**: Penguins did not originate in Antarctica

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1
- **Claim**: For comprehensive legal coverage, most brands also pursue trademark registration, which complements copyright by protecting the functional aspects of the logo and preventing consumer confusion

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Research published in Nature confirms caffeine acts as a neurotoxin on slugs, supporting the mechanism behind why it works as a deterrent, though strongly brewed coffee solutions also carry risks of damaging other garden organisms

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Religious bodies hold differing views: some treat the Bible as inerrant (meaning it contains no factual errors whatsoever), while others view it as infallible (meaning it cannot be contradicted where it speaks to matters of faith and practice, even if scientific or historical details may still contain errors)

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Additional sources note that barefoot running may increase the risk of stress fractures and foot injuries among those increasing speed or mileage too quickly that the overall incidence of running injuries has remained stable regardless of whether runners are barefoot or shod

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Proponents of the curse tradition point to numerous subsequent accidents across centuries — including the 1937 fire at the Old Vic that narrowly missed Laurence Olivier the 1980 production at the Old Vic that disbanded after poor reviews — as further evidence of the play's supernatural reputation

### Sample conflictingqa_f3ba8599e370

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: In short, whether yoga counts as a religion depends on how one defines 'religion' and whether one identifies the non-dogmatic, direct-experience core as a spiritual path distinct from organized faith

### Sample conflictingqa_f4693bea2c31

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, some researchers argue that emoji are technically ideograms — pictures representing ideas rather than single words — and do not participate in morphological or grammatical processes like words do, suggesting they are not yet actual words in the traditional sense

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, it is worth noting that the International Agency for Research on Cancer (IARC) classifies unfiltered mate as "probably carcinogenic to humans," largely due to polycyclic aromatic hydrocarbons (PAHs) present in the smoke generated during traditional preparation that some sources distinguish this risk to the esophagus specifically from the direct effects of yerba mate itself

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d5, d2, d3
- **Claim**: As a result, the incident remains open to interpretation, with some researchers and witnesses concluding that the official military explanation does not fully account for all observed phenomena

### Sample conflictingqa_f97fef94decc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, VR can also offer vision benefits: under professional guidance, it can improve eye coordination, hand-eye coordination, depth perception reaction time, as well as assist with conditions like amblyopia

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Hindi

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: Aryna Sabalenka (2025 US Open women's singles champion); Jessica Pegula was the 2024 finalist

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Maryam Mirzakhani (2014 — first and, to date, only woman to receive the Fields Medal)

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The latest major version of .NET depends on the specific framework generation. For .NET Framework, the most recent release is 4.8.0 Build 4115 , released in 2024, though .NET Framework 4.8.1 is also noted as a recent release

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d1, d5
- **Claim**: Multiple authoritative sources confirm Biden's presidency, including the official White House website which describes him as the 46th President

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2
- **Claim**: This is confirmed by multiple sources, including the official Costco membership page and various reviews, which also detail the additional benefits such as two percent cash back on up to $1,250 of eligible purchases each year

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: CODA (2021)

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: This is further corroborated by the MLB timeline, which confirms the 2017 World Series victory and notes the 2022 appearance as their second championship, implying a win was achieved

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Luka Modric (2018) — the last Ballon d'Or winner before Messi–Ronaldo dominance

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Bayonne, New Jersey

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: He was 43 years old at the time, tragically losing his life just two years after the influential book "Perceptrons" by Marvin Minsky and Seymour Papert was published — a work that initially seemed to doom the field of neural network research

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1, d4
- **Claim**: Queen Elizabeth II died on 8 September 2022 at Balmoral Castle in Aberdeenshire, Scotland

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: There is no permanent cure for cancer; however, significant milestones in cancer treatment have been achieved over time

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: However, multiple sources cite a figure closer to 16 minutes before the suspension , making the most accurate answer 16–21 minutes depending on the timeframe referenced

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3
- **Claim**: The FDA approved the Pfizer vaccine for people ages 5 and older, making 5 the youngest age eligible for COVID-19 vaccination in the United States

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, the FDA's recent approval of updated COVID-19 shots further narrowed eligibility, limiting Pfizer's vaccine to children with at least one high-risk health condition, while Moderna remains open to all children 6 months and older

### Sample hotpotqa_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d6
- **Claim**: Additional context on the album includes that it was preceded by the lead single "Single Again" released on November 6, 2007 followed by the final single "Look Back at Me" featuring Killer Mike

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d7
- **Supporting Docs Found**: None
- **Claim**: Their self-titled debut album, *Lit*, was recorded in 1995, though it was not released until 2000

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d5, d4, d6, d7, d8
- **Claim**: 506

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: Anne Bancroft

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The statue was dedicated on October 28, 1886, becoming a iconic symbol of freedom and democracy

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d5, d4, d1
- **Claim**: Sakshi Malik (Haryana); Madhuri Dixit (India); Parineeti Chopra (Haryana); Avani Lekhara (Rajasthan); Bhawna Dehariya & Siddhi Mishra (Madhya Pradesh)

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Princess of Wales Theatre

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Some sources suggest that the gesture evolved from early Christian practices, in which believers crossed their thumbs and index fingers to form an 'L' symbol (the Ichthys or fish symbol), before simplifying to the modern one-handed cross

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5, d3
- **Claim**: Over time, the gesture became associated with invoking divine protection rather than merely seeking luck was further adopted as a sign of solidarity and hope during periods of persecution

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Phil Jackson (11 championships) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach: Red Auerbach (16 championships as a coach and executive) As a player: Bill Russell (11 championships) As a coach

### Sample qacc_403a59870dc2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2
- **Claim**: The Rams' 2021 win was their second championship overall

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Lacteals

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d3
- **Claim**: Gagarin's mission, which circled Earth in a Vostok 1 spacecraft, marked the USSR as the clear front-runner in the global space competition, a fact corroborated by additional reporting on the mission

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Canada did not achieve full independence on a single date; rather, it was a gradual process spanning several key milestones

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: McEwan

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 180

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: October 1, 1968

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Russ Ballard ()

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: NASA commemorates this milestone as the point when uninterrupted human presence in low-Earth orbit transitioned from the Mir station to the ISS

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Fortunately, no one was injured in the blaze the following Christmas White House staff gathered again to celebrate, receiving toy fire trucks as gifts from the Hoovers

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5
- **Claim**: Combining these figures, we estimate that Cadbury sells its products in approximately 50–100 countries globally, though an exact hard count is not provided anywhere in the retrieved evidence

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTENTATION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: In Mexico, toll roads are called **autopistas**

### Sample qacc_e7318f6f3bbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: This word is well-documented in English language vocabulary records and is consistently cited across multiple sources

### Sample qacc_e7318f6f3bbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A close second is **strengthlessnesses**, which is technically the longest word with only one vowel repeated, but 'strengths' remains the standard answer

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d4
- **Claim**: The last time Rangers were in the Champions League was during the 2022–2023 season, when they qualified through the group stages

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4, d3
- **Supporting Docs Found**: d1
- **Claim**: The range reflects genuine scholarly disagreement about the precise timing, with factors such as the nature of the heresies addressed and historical traditions contributing to differing conclusions

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: From three to seven

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d1
- **Claim**: Within Florida, The Villages further divides into more than 50 named villages, each ranging in size up to approximately 1,000 homes, all located exclusively in the state of Florida

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: For Germany, figures range from 5.3–6.6 million Japan reported 1–3.1 million casualties

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For commercial motor vehicles, the Federal Motor Carrier Safety Administration requires drivers to be at least 21 years old pipeline companies similarly require operators of hazardous materials transport to be at least 21

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The concept evolved further during the interwar period and was cemented by the Beveridge Report in 1942, which served as the blueprint for the post-War welfare state

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: This occurred during his presidency, as he steadily increased the number of military advisers in South Vietnam throughout his term

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Insufficient evidence to answer the query confidently

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: The Battle of Brandywine was won by the British general Howe, though the Americans managed to retreat intact rather than being destroyed

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Additionally, the Eagles claimed their first-ever NFL Super Bowl title in 1981 (Super Bowl XV) participated in Super Bowl XXXIX in 2005

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: New South Wales last won the State of Origin series in 2021

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_301378915064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Merritt Wever (Nurse Jackie)

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The next Avatar comic is *Avatar: The High Ground Omnibus*, set to release on September 30, 2025

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d1
- **Claim**: The show was originally ordered on March 27, 2018 season two renewal was confirmed , though the specific broadcast start date is not explicitly stated in the combined evidence

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The specific episode in which Goku becomes Super Saiyan 3 is titled "An Astounding, Great Transformation!!

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: This transformation is also referenced in Dragon Ball Z: Battle of Gods 2, where it is noted as Goku's strongest Super Saiyan transformation

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken (2026–present)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The group, formed in Los Angeles in 1989, quickly rose to fame with hits such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Shay Mitchell was 23 years old when she portrayed 16-year-old Emily Fields in the show's pilot episode, meaning Emily is approximately 23 years older than her on-screen character

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 670–680

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This record was confirmed when the film surpassed its predecessor, **Rewind**, just 10 days after its release

### Sample trust_align_009

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence indicates that the Black Death did not have a single definitive start date in the UK, but rather appeared in various forms across different centuries

### Sample trust_align_026

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d5, d4, d1
- **Claim**: These plans reflect the ongoing development of Mars exploration capabilities, with no single definitive 'first' mission identified across the available evidence

### Sample trust_align_039

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Society generally accepts euthanasia for animals because it is viewed as a humane way to relieve intractable suffering animal owners are trusted to make decisions in their pets' best interests

### Sample trust_align_039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: In contrast, human euthanasia is subject to far stricter criteria because humans can communicate their own wishes there is a strong ethical consensus that another person should not terminate someone else's life without explicit, voluntary consent

### Sample trust_align_115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The Boston Celtics last won the NBA championship in 2008, when they defeated the Detroit Pistons in six games

### Sample trust_align_173

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Fruit contains fructose (a natural sugar) combined with fiber, vitamins antioxidants, which mitigates the blood sugar impact compared to isolated added sugars like those in candy and soda

### Sample trust_align_175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Additionally, the South Pole's higher elevation (approximately 2,500 meters) contributes significantly to its extreme temperatures, as higher altitude correlates with lower atmospheric pressure and colder conditions

### Sample trust_align_175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: In contrast, the North Pole is often warmer than expected due to its lower elevation — approximately 3.5 meters — which allows for more rapid heat transfer from the surrounding ocean — a mechanism not present at the higher-altitude South Pole

### Sample wikirevision_0004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This was confirmed when the company officially rebranded itself in October 2022, updating its former name of Facebook, Inc

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc. as of 2015, when Google was reorganized as a wholly owned subsidiary of Alphabet Inc. (formerly known as Alphabet). The two companies are often treated as distinct entities, though Google remains the more visible brand

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed when the Haryana government passed a resolution changing the city's official name from Gurgaon to Gurugram in 2016, which took effect on 1 July 2017. As a result, while 'Gurgaon' is still commonly used, official communications, road signs civic records are required to use 'Gurugram'

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The President of Germany is Frank-Walter Steinmeier, who has served as the Federal President of the Federal Republic of Germany since taking office on that date. This is confirmed by the official Wikipedia entries on both the President of Germany and the Federal President, which list his incumbency from 19 March 2017. While the older revision (timestamped 2025-09-22) and the newer revision (timestamped 2026-04-28) both consistently identify him as the current President, the newer revision provides the most recent confirmation

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 edition — the 139th iteration of the prestigious tournament — took place from 29 June to 12 July 2026, confirming Sinner's title as the most recent record

### Sample wikirevision_0154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in April 2025, which explicitly lists her as incumbent with a detailed biography

### Sample wikirevision_0171

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Jannik Sinner was the defending men's singles champion at the 2026 Australian Open, but he lost in the semifinals

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The men's singles champion is *Carlos Alcaraz*, who defended his title by defeating Jannik Sinner in the final. This supersedes earlier information from the 2025 edition, in which Alcaraz also won the title but is now the defending champion at the 2026 tournament


================================================================================

*Report generated by CATS v2.0*
