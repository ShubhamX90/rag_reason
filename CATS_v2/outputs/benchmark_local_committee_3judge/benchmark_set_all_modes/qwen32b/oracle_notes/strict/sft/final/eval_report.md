# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 123 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.959 (over 736 samples)

**GR F1** *(used in CATS)*: 0.975

**Behavior Adherence**: 0.814 (over 613 applicable samples)

**Factual Grounding**: 0.856 (over 613 applicable samples)

**Single-Truth Recall**: 0.734 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.845

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.975
- **Precision**: 0.991
- **Recall**: 0.959
- **Accuracy**: 0.959
- TP=583, FP=5, FN=25, TN=123

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.831
- **Abstain Recall**: 0.961
- **Abstain F1**: 0.891
- **Specificity**: 0.959
- Abstain TP=123, FP=25, FN=5, TN=583


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (54 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.955 (n=157)
- **Grounding**: 0.941 (n=157)
- **Recall**: 0.828 (n=154)
- **CATS**: 0.929

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.923
- **GR F1** *(used in CATS)*: 0.950
- **Behavior**: 0.904 (n=178)
- **Grounding**: 0.784 (n=178)
- **Recall**: 0.651 (n=156)
- **CATS**: 0.822

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.573 (n=96)
- **Grounding**: 0.837 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.800

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.710 (n=145)
- **Grounding**: 0.914 (n=145)
- **Recall**: 0.746 (n=140)
- **CATS**: 0.841

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.811 (n=37)
- **Grounding**: 0.667 (n=37)
- **Recall**: 0.649 (n=37)
- **CATS**: 0.759


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1861

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

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Partially — peeling removes most fiber and some antioxidants but does not reduce vitamin C content; the degree of nutrient loss depends on the specific nutrient

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: Yes — palm oil production causes deforestation, habitat destruction for many endangered species substantial greenhouse gas emissions; however, some sources note that sustainably-produced palm oil carries lower environmental costs

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Dog breeding is not universally unethical, but some practices are

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Cows technically have one stomach that is divided into four compartments — the rumen, reticulum, omasum abomasum — rather than four distinct stomachs

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: Overall, the evidence suggests that the common perception of milk increasing mucus is largely due to the texture of milk itself and the mouth's enzymatic response, rather than any actual physiological increase in mucus secretion

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Not directly from chlorine — chlorine actually lightens hair, but copper (from algaecide or tap water) oxidizes and binds to hair proteins, causing the green discoloration

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: It depends on usage duration: real trees are more sustainable when disposed of via recycling or composting, while artificial trees become more sustainable only after being reused for 15-20 years — making the overall comparison conditional on how long the artificial tree is kept

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d1
- **Claim**: However, other evidence suggests that these benefits are conditional on rigorous management and community involvement that trophy hunting is increasingly criticized for ethical reasons and concerns about whether it delivers meaningful conservation outcomes in practice

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: The constitutional status of prayer in U.S. public schools is nuanced — the Supreme Court has ruled that school-led or endorsed prayer (including participation by faculty and staff) is unconstitutional, while the First Amendment protects the rights of individual students to pray privately and quietly, dress according to their faith form religious student groups on equal terms with nonreligious groups. The 2026 U.S. Department of Education guidance clarifies that schools must maintain a stance of neutrality toward all faiths and permit students and employees to express religious beliefs without coercion, though the guidance itself is rooted in settled court precedent rather than establishing a new legal standard

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are valuable for protecting core algorithms and functional innovations, while others argue that patents should not apply to software or that current rules already sufficiently limit patentability to only technical or novel inventions

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: It depends on jurisdiction; in the U.S. federal law generally allows ISPs to sell anonymized browsing data without consent, but some states like Maine and California require opt-in or opt-out consent other states are considering similar protections

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Bees generally avoid flying in the rain because wet wings reduce lift and maneuverability, but they are capable of flying in light rain and will do so when driven by urgent needs such as defending the hive or finding emergency food

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Additional research from the University of Guelph and Harvard University further note that farmed salmon can have higher levels of certain contaminants and lower levels of beneficial carotenoids, though the magnitude of these differences varies considerably depending on farming practices and wild salmon species

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Not all bird calls are unique to individuals; many are shared within species or even recognized across species

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Within this clade, scientists have identified a specific subgroup called Paraves as the most likely ancestor to modern birds, rather than T-Rex itself

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Overall, while the capability to swim appears widespread among snakes, the universal claim is regarded with scientific caution due to incomplete evidence for the full extent of snake diversity

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: In summary, regulatory bodies and some scientific reviews consider glyphosate safe when used as directed, whereas other researchers — particularly those focusing on long-term or cumulative effects — argue that the evidence of harm is compelling and warrants stricter regulation

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Overall, while volcanic activity clearly contributed to the event, the scientific consensus leans toward a nuanced scenario involving multiple interacting factors rather than a single definitive trigger

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: This reduction is linked to a shift from brute-force information processing to more metabolically efficient symbolic thinking, as well as declining body size and reduced energetic pressures as humans transitioned to complex societies and began storing information externally

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Religious views differ; the Bible is infallible for some denominations (especially Catholicism and many evangelicals), but not all scholars or traditions agree on infallibility extending to historical or scientific details

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The retrieved evidence shows that justified false beliefs are not only possible but also commonly acknowledged in epistemological debate, particularly in the context of the JTB (justified true belief) theory of knowledge and its limitations

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Emoji serve as a visual supplement to written language but do not constitute a distinct written language themselves

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Religious scholars and Christian denominations differ on whether to classify Mormons as Christians; the answer depends heavily on the definition of Christianity being applied

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: Prince Harry was not formally stripped of the title Duke of Sussex by King Charles III. The Duke and Duchess of Sussex stopped using the formal title 'His/Her Royal Highness' as part of their 2020 agreement to step down as senior working royals, though Prince Harry's biography on the official Royal Family website was updated to remove HRH references only in late 2022. Multiple sources indicate that Prince William has been pressuring his father to strip Harry and Meghan of their titles that such discussions are ongoing at the highest levels of the royal family, but no official removal has been announced

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: August 16, 1977

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The latest Grammy Award for Best Jazz Performance was won by Samara Joy at the 67th Annual Grammy Awards in February 2025. She received the award for "Twinkle Twinkle Little Me," a duet with Sullivan Fortner is now five-for-five in her Grammy nominations. This win superscedes earlier records, including the 2026 award listed in some sources, as the 67th ceremony took place in 2025

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: The site is now part of the White Sands Missile Range and is owned by the U.S. Department of Defense the Trinity test remains the most widely recognized example of U.S. nuclear testing

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

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Costco Executive membership costs $130 annually

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Beijing is the first city in the world to have hosted both the Summer and Winter Olympics. This distinction was confirmed when Beijing hosted the 2022 Winter Olympics, becoming the first city in Olympic history to have hosted both Games. The city had previously hosted the 2008 Summer Olympics, making it a dual-Olympic host

### Sample freshqa_97f3c1fe1fd4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This is further contextualized by the fact that the team's most recent confirmed playoff appearance was in the 2019-20 season, where they held a 53-19 record , making the 2023-24 season a significant downturn in performance

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

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This figure includes principal photography, reshoots, post-production on-set costs, though it excludes global marketing expenses

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d2
- **Claim**: February 19, 2026

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Norma Koch

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The Allies moved to Italy, beginning with the invasion of Sicily in July 1943, followed by the Italian Campaign on the mainland from 1943 to 1945

### Sample qacc_132167e66120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 1968

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2
- **Claim**: Additionally, crossing fingers became associated with early Christianity, as persecuted followers reportedly used the gesture (along with the ichthys or fish symbol) to recognize one another and invoke divine protection some theorists suggest the solo version crystallized during the Hundred Years' War period

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: October 1968

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The scheduled completion date for the Sagrada Familia has been pushed back from 2026 to the early 2030s

### Sample qacc_a6b48b7accc4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Celebrity Big Brother is currently available on CBS in the USA

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Most effigy mounds were built between approximately 700 and 1200 CE, with the majority constructed within the Late Woodland period (roughly 750–1050 CE)

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

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: More specifically, it is often described as type SBbc, indicating a barred spiral with intermediate-sized central bulge and loosely wound arms

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple authoritative sources, including the NASA Extragalactic Database (NED) and the Two Micron All-Sky Survey (2MASS) , which both confirm the SBbc designation

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_f2218f8c979e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Prime rib comes from the rib primal, specifically ribs 6–12, located under the front section of the backbone between the chuck and the loin

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: 407,000

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_362420ec2416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: These findings are consistent with the broader 2011 census data, which also records larger states such as Uttar Pradesh with a population of around 19,960,000

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

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It depends on location; check your state's rate or enter a zip code — or see all rates by state

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The query is ambiguous because it does not specify which country or type of election is meant. The first U.S. presidential election was held on February 4, 1789, in which George Washington was unanimously chosen as the first president by electors from 10 of the 13 states

### Sample situatedqa_geo_d982055a66d8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: For context, the first unified U.S. presidential Election Day under the 1845 federal law was held on November 7, 1848 earlier elections for the U.S. House of Representatives and Senate predated both of these milestones

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d1
- **Claim**: The present Law Minister of India is Kiren Rijiju, who serves as the Cabinet Minister for Law and Parliamentary Affairs. He is a senior Bharatiya Janata Party (BJP) leader and has held the portfolio since 2014, making him one of the longest-serving ministers in the Government of India. In addition, Senator Azam Nazeer Tarar serves as the Federal Law Minister of Pakistan

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5
- **Claim**: This document created a loose confederation of states with a weak central government, establishing a single legislative body where each state held one vote lacking a separate executive or judicial branch

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: The FOMC meets regularly — typically every six weeks — to discuss the economic outlook and make decisions on key instruments such as interest rates and the money supply, with the goal of promoting maximum employment, stable prices moderate long-term interest rates

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: The latest stable version of Android is Android 16, released on June 10, 2025. It was first made available to Pixel phones and has since rolled out to other manufacturers including Samsung. This supersedes earlier information identifying Android 15 as the latest version, as d4 and d5 both confirm Android 16 as the current release

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: 1980

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: On naval ships, SS most commonly stands for Submersible Ship, indicating a submarine

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that enter the bloodstream when the heart is damaged or stressed they are used to diagnose and monitor heart conditions

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 133

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held on 19 November at the Narendra Modi Stadium in Ahmedabad. This was Australia's sixth Cricket World Cup title the 2023 tournament was the 13th edition, hosted across ten venues in India. Wikipedia's up-to-date revision of the Cricket World Cup page confirms Australia as champions, superseding an older revision that had listed India as the 2023 winner

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was confirmed in April 2016, when the Haryana Government officially renamed the city from Gurgaon to Gurugram. As a result, all official references to the city now use the name Gurugram, though 'Gurgaon' is still commonly used in everyday speech and in some older documentation

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This name change was confirmed by the Government of Haryana in 2016 the city is now officially known by this new name. As a result, Gurgaon is no longer the official name of the city


================================================================================

*Report generated by CATS v2.0*
