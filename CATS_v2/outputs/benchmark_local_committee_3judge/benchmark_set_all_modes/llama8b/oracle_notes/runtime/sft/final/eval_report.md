# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 124 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.980 (over 736 samples)

**GR F1** *(used in CATS)*: 0.988

**Behavior Adherence**: 0.819 (over 612 applicable samples)

**Factual Grounding**: 0.902 (over 612 applicable samples)

**Single-Truth Recall**: 0.749 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.864

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.988
- **Precision**: 0.993
- **Recall**: 0.982
- **Accuracy**: 0.980
- TP=597, FP=4, FN=11, TN=124

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.919
- **Abstain Recall**: 0.969
- **Abstain F1**: 0.943
- **Specificity**: 0.982
- Abstain TP=124, FP=11, FN=4, TN=597


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (54 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.955 (n=157)
- **Grounding**: 0.954 (n=157)
- **Recall**: 0.864 (n=154)
- **CATS**: 0.941

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.904 (n=177)
- **Grounding**: 0.862 (n=177)
- **Recall**: 0.644 (n=156)
- **CATS**: 0.848

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.656 (n=96)
- **Grounding**: 0.895 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.847

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.697 (n=145)
- **Grounding**: 0.915 (n=145)
- **Recall**: 0.732 (n=140)
- **CATS**: 0.834

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.973
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.730 (n=37)
- **Grounding**: 0.842 (n=37)
- **Recall**: 0.784 (n=37)
- **CATS**: 0.836


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2116

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

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, designers can seek protection under trademark law for logos, labels brand names, as well as through design rights and patents in some jurisdictions

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: For individuals with prehypertension or hypertension, the temporary blood pressure spikes during lifting are not a cause for concern in most cases, but they do warrant caution and may require modifications such as lifting lighter loads or avoiding Valsalva maneuvers

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while it is the largest single living organism, it is not the heaviest; a single aspen colony called Pando in Utah spans 106 acres but weighs more than the Humongous Fungus

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, the exact total impact on overall nutritional value depends on the specific nutrients of interest and portion sizes consumed

### Sample conflictingqa_24c25ef3a801

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A third perspective holds that money can buy moments of happiness but not lasting fulfillment, as witnessed by the fact that 40% of respondents in a survey reported being unhappy despite having sufficient material resources

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The weight of the evidence is mixed and the relationship is considered complex, with some sources noting that money is more powerful at reducing misery than at increasing genuine happiness that people tend to adapt quickly to new possessions and experiences, making them less sources of lasting happiness

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The conflicting research outcomes and expert opinions make it difficult to give a clear general answer; any wrist rest use carries risks and should be evaluated on a case-by-case basis

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2
- **Claim**: IPv6 is not fundamentally more secure than IPv4; both protocols are equally secure because the majority of security incidents stem from implementation and human error rather than protocol weaknesses IPv6's security advantages—such as mandatory IPsec support—are not a inherent feature of the protocol itself but rather a design choice that can also be applied to IPv4

### Sample conflictingqa_34610226ee3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The answer depends on which perspective one trusts more

### Sample conflictingqa_411445406724

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Cycads did not replace flowering plants as ecologically dominant species until well after the Mesozoic ended, more than 100 million years ago

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The most credible and directly relevant source in the retrieved set is the Harvard study cited in , which found that the wage gap among MBTA workers could be fully explained by workplace choices rather than discrimination, but this finding is presented as a nuanced rather than universal explanation other studies present conflicting conclusions

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Weisman, 1992; Santa Fe, 2000), while faculty prayer groups and personal religious expression during non-instructional time are permitted (Mergens, 1990; Good News Club v. Milford Central School, 2001) the U.S. Department of Education's 2026 guidance further clarified that schools must allow students and staff to act in accordance with their faith without favoritism

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Overall, the evidence does not establish bicarbonate supplementation as a universally preventive measure for all stages of CKD

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d1
- **Claim**: However, this right to sell is not absolute: in 2020, California's Consumer Privacy Act gave California residents the right to opt out of data sales Maine and other states have since enacted their own laws requiring explicit permission before personal data can be sold

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological: d2 and d4 focus on civic identity barriers and ethnocentrism, while d3 and d1 focus on citizenship/political integration and diversity acceptance, with no source providing a complete, direct answer to whether multiculturalism hinders unity overall

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: An experiment described in d3 challenges plants to survive in zero light for 30 days to test their ability to survive without light

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The evidence does not establish a clear, universal answer; rather, it highlights that plant survival without light is possible for some species under specific conditions

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Overall, the evidence does not support a clear, confident answer that coffee grounds are an effective general-purpose slug and snail deterrent

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological: sources differ in how they define 'taboo' and in their interpretation of cultural trends over time

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: The retrieved evidence presents a genuine curse legend surrounding Macbeth, with folklore claiming a coven of witches objected to Shakespeare using real incantations and cursed the play from its first performance around 1606

### Sample conflictingqa_f3ba8599e370

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The debate is further complicated by the fact that the word 'yoga' itself comes from the Sanskrit root meaning 'to yoke' and was used in non-religious contexts before being absorbed into Hindu and Buddhist spiritual practices , making the question one of interpretation and classification rather than a straightforward factual one

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3, d1, d2
- **Claim**: The Phoenix Lights incident on March 13, 1997, was officially explained by the Department of Defense as military flares from A-10C Thunderbolt IIs, though witnesses and some sources, including former Governor Fife Symington, disputed this explanation, describing the formation as unexplained and unlike any man-made object

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A definitive answer will depend on resolving this methodological and interpretive conflict

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This ranking is based on the most recent data from Ethnologue and Visual Capitalist (2025), which places Hindi at number 3 ahead of Spanish and Arabic

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 2025 US Open was Aryna Sabalenka's second consecutive US Open title, having also won in 2024

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Prince Harry's HRH title was removed from the official Royal Family website in a quiet update, but the Duke of Sussex title remains in use

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: August 16, 1977

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The war between Russia and Ukraine, which began in 2022, has caused over 1 million casualties and resulted in Ukraine's population declining by over 10 million people — roughly a quarter of its total population

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: Harry Maguire has never won the Ballon d'Or; no confirmed Ballon d'Or win year exists in the evidence

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This film, which follows a multi-generational American saga of political resistance, won six Oscars including Best Director and Best Adapted Screenplay, marking Anderson's first Academy trophy

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The 98th Academy Awards also saw Ryan Coogler's *Sinners* lead all films with 16 nominations, further confirming the validity of the Best Picture result

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

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel is "When We Were Real" by Daryl Gregory, which won the 2025 Nebula Award. This is explicitly listed on the official Nebula Awards page with a 2025 designation, making it the most recent winner for the Best Novel category

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: His death was a significant loss for the field of AI research, as funding for perceptron-related projects dried up in the following years

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The 2023–24 result is further corroborated by the fact that the Raptors missed the playoffs in that season earlier records from 2015–16 only provide context on the team's historical performance without contradicting the most recent data

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Jeff Bezos sold Amazon shares in late June and July 2025, but did not sell the entire company

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The green anaconda (Eunectes murinus) is the heaviest reptile in the world, with the largest specimen ever recorded weighing 550 pounds

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9, d7, d2
- **Supporting Docs Found**: None
- **Claim**: These details contextualize the geographic relationship between the two streets on the specific map referenced, but do not contradict any other sources

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Bartholdi's design was chosen by Édouard de Laboulaye, the founder of the French-American Union, who proposed the statue as a monument to commemorate the upcoming centennial of U.S. independence

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Sakshi Malik

### Sample qacc_132167e66120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_15ffab2466f7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: These venues for different productions, with no contradictions

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

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: These origins are not randomly distributed along the genome — they are organized in clusters of consecutive origins that are synchronously activated in metazoans (animals) the selection mechanism appears to be mainly epigenetic rather than relying on a specific consensus DNA sequence

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: The Airdrome

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: The two modules docked in space on December 5, 1998, marking the official beginning of the ISS's assembly

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: By weight, the average adult human is approximately 60% water, with the brain and muscles being the most water-dense organs at around 75% water each adipose tissue being the least at about 20%

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This distribution is consistent across multiple sources, with the intracellular compartment (ICF) holding approximately 28 L of water in a 70 kg man the extracellular compartment (ECF) holding about 14 L

### Sample qacc_a4dde83c35da

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: The government also relied on a civil service system where officials entered through examinations provincial affairs were managed by three agencies reporting to the central government, further reflecting its imperial character

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2
- **Claim**: The Rajya Sabha currently has 245 members, consisting of 233 elected members and 12 nominated members

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: The total allowed capacity is 250, but the current strength is 245 due to the Jammu and Kashmir Reorganisation Act

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: The 47th position is further corroborated by the sequential statehood timeline, where Arizona was admitted as the 48th state just a month later on February 14, 1912

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This figure, noting that he appointed justices including Hugo Black, Stanley Reed, Felix Frankfurter, William Douglas, Frank Murphy, James Byrnes, Robert Jackson Wiley Rutledge

### Sample qacc_eb7c676e133e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: December 19, 1972

### Sample qacc_ec5b0067c29a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1, d2
- **Supporting Docs Found**: d5
- **Claim**: It is located on the grounds of the U.S. Naval Observatory, a historic building dating to 1830

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: 3–7

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

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: This 1970 Act was the most significant federal legislation ever enacted to address air pollution, shifting the focus from state-oriented regulation to a comprehensive national program under the newly created Environmental Protection Agency (EPA)

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Eisenhower

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3, d1
- **Supporting Docs Found**: None
- **Claim**: Multiple other documents provide partial context on desert geography and border regions without contradicting this conclusion

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: These are distinct elections in different countries with different systems there is no single universally applicable answer to 'the first election' without knowing which country is meant

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, it is worth noting that tea never fully disappeared — Southern Americans continued to drink sweet tea and other regional varieties tea remained popular in parts of the country like the South and Asian-American communities, meaning the shift was never fully complete

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: At the state level, environmental policy is also a primary area of responsibility, with many state governments operating their own environmental protection agencies and programs in areas such as water quality, waste management climate action

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Additionally, local governments and tribes play important roles in implementing and enforcing federal and state environmental regulations the U.S. Supreme Court has confirmed that local governments can also use their land-use powers to protect the environment when state and federal authorities are insufficient

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: British General Sir William Howe's army of about 16,000 troops defeated the Continental Army of about 15,000 in the vicinity of Chadds Ford, Pennsylvania, near Philadelphia

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This was confirmed when Djokovic won his 24th Grand Slam at the 2023 US Open, the final tournament of his career

### Sample situatedqa_temp_40e6764f611f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: He is a recurring character in the Madagascar franchise, often serving as comic relief is also the main character in the spin-off series All Hail King Julien

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: 2022 (most recent in the retrieved evidence)

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: 1980

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Todd Monken

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: This result, noting it was the first time a team has won consecutive championships since the Tampa Bay Lightning did so in 2020 and 2021

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This is the lead ship of the Queen Elizabeth class, the largest carriers ever built for the Royal Navy was formally declared operational in 2020 after completing her maiden deployment

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The UK Government's official publications confirm that the carrier is now in service and has conducted its first operational deployment

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Their victory over France in the final — won 4–2 in a penalty shootout after the match ended 3–3 after extra time — was their first title since 1986 and made them the first nation outside of Europe to win the tournament since 2002

### Sample wikirevision_0097

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Bangalore is officially called Bengaluru. Bengaluru is the capital and largest city of the southern Indian state of Karnataka its official name changed from Bangalore on 1 November 2014

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the 2023 Cricket World Cup, defeating India in the final on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad to claim their sixth Cricket World Cup title

### Sample wikirevision_0142

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: This is confirmed across multiple authoritative sources, including the newer Wikipedia revision of the Vice President of the United States article, which supersedes an older version that had described him as the 51st Vice President

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: He earned his first Ballon d'Or, surpassing the record of 8 held by Lionel Messi


================================================================================

*Report generated by CATS v2.0*
