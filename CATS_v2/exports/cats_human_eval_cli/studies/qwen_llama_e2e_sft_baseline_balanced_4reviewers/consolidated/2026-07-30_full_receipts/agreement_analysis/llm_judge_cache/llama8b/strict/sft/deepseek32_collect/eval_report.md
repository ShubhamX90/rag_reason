# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 126 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.973 (over 736 samples)

**GR F1** *(used in CATS)*: 0.983

**Behavior Adherence**: 0.882 (over 610 applicable samples)

**Factual Grounding**: 0.838 (over 610 applicable samples)

**Single-Truth Recall**: 0.739 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.861

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
- **Behavior**: 0.942 (n=154)
- **Grounding**: 0.908 (n=154)
- **Recall**: 0.870 (n=154)
- **CATS**: 0.929

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.950
- **GR F1** *(used in CATS)*: 0.968
- **Behavior**: 0.865 (n=178)
- **Grounding**: 0.818 (n=178)
- **Recall**: 0.641 (n=156)
- **CATS**: 0.823

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.792 (n=96)
- **Grounding**: 0.839 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.874

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.897 (n=145)
- **Grounding**: 0.814 (n=145)
- **Recall**: 0.714 (n=140)
- **CATS**: 0.854

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.892 (n=37)
- **Grounding**: 0.730 (n=37)
- **Recall**: 0.703 (n=37)
- **CATS**: 0.821


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2119

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/deepseek-r1-distill-32b

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

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Weight lifting can cause temporary increases in blood pressure, especially during heavy lifts or Valsalva maneuvers, but research indicates that regular weight training is generally not a cause of high blood pressure and may even help lower it

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: That legal ruling has since been celebrated as a landmark victory for free expression, though some continue to debate the poem's content and the broader implications of its publication

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Traditional definition: yes, anime is a form of cartoon

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d4
- **Claim**: This fungus is estimated to be over 2,000 years old and weighs approximately 440 tons, making it as massive as three blue whales

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Multiple authoritative sources, including the Oregon Field Guide and the US Forest Service, confirm that this fungus holds the record for the largest single living organism on Earth, surpassing all other contenders such as giant trees and blue whales

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: By its own founding statement, the FSM is described as a'real, legitimate religion, as much as any other,' though its roots trace back to a satirical 2005 letter protesting intelligent design education policies

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: A third perspective holds that entrepreneurship is not a personality trait but a practice that can be learned that the vast majority of entrepreneurs were not born but rather learned how to become entrepreneurs

### Sample conflictingqa_11c5ef7c4545

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The debate is further nuanced by research showing that nearly three quarters of entrepreneurs start their businesses in pursuit of opportunity rather than necessity that regions like North America have the highest proportion of entrepreneurs globally , suggesting that entrepreneurship is not only possible but also increasingly appealing to a broad cross-section of people

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: Ethical concerns are further compounded when breeders focus more on profit than on the health and welfare of dogs, as is often the case with puppy mills and designer dog breeding

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The retrieved evidence is mixed. Some sources state that cows have four stomachs, while others clarify that they have one stomach divided into four compartments: the rumen, reticulum, omasum abomasum

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: In summary, while the Silurian was a period of significant terrestrial plant evolution and diversification, it is not explicitly identified as the birth of the first land plants, which had already emerged earlier in the Ordovician

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: However, the mechanisms are not fully understood the extent to which epigenetic changes are hereditary remains an active area of research

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence presents a genuine debate on whether audiobooks count as real reading, with some sources arguing they are fully equivalent to physical books and others doubting their legitimacy

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial trees, according to multiple sources. Christmas tree farms are agricultural operations that sequester carbon, reduce erosion provide wildlife habitat the trees are typically farmed on a cycle where one harvested tree is replaced by another, making the practice carbon neutral. In contrast, artificial trees are made from nonrenewable plastics and metals, have a carbon-intensive manufacturing process are often shipped long distances, producing up to 40 kg of greenhouse gas emissions per tree. However, a 2009 study cited by The Guardian found that an artificial tree only becomes more environmentally friendly after about 20 Christmases of reuse, meaning that if an artificial tree is used for 20 or more years it will have a lower carbon footprint than a real tree harvested and discarded after a single season

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The 2024 KDIGO guidelines recommend bicarbonate supplementation only when serum bicarbonate is less than 18 mEq/L results from a large randomized controlled trial published in the New England Journal of Medicine found that bicarbonate supplementation did not affect the composite outcome of CKD progression, cardiovascular events death over a median of 2.4 years , though the study was limited to patients with normal or high serum bicarbonate levels at baseline

### Sample conflictingqa_63fde268aa8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The conflict between these perspectives is further reflected in the mind-body problem's two subproblems—causal interaction (how minds affect matter) and consciousness (how matter gives rise to subjective experience)—which remain unresolved, making the answer to whether the mind is separate from the body a matter of ongoing philosophical debate

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3
- **Claim**: However, some states are pushing back against this rule — for example, Maine passed a law requiring ISPs to obtain individual consent before selling personal data California's Consumer Privacy Act gives California residents the right to opt out of data sales — so the answer is not universal and varies by jurisdiction

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: Taking extra vitamin C to prevent colds has not proven true and that most cases respond to adequate dietary intake rather than supplements the same source warns that high doses can have side effects and interact with medications

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: However, these yield gaps also mean that organic farms require more land to produce the same amount of food as conventional farms , which can have its own environmental costs

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: A fair and complete answer would need to address all these perspectives and the scriptural evidence they cite

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Spelunking is related to caving but not exactly the same: it can refer to any unprepared or recreational cave entry, while caving typically implies a higher level of expertise and safety measures

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d1
- **Claim**: At the species level, most birds have a unique set of calls that are learned from parents or other members of their species, but the calls themselves are not necessarily unique to each individual bird — for example, a single species may have multiple call variants used in different contexts

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2
- **Claim**: It can affect other parts of the body contacted by semen or vaginal fluid such as the anus, throat eyes

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Research has shown that plants can survive in complete darkness for up to 30 days some species can even regrow their leaves after being in darkness for up to 60 days , though continuous darkness will eventually kill all plants

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: In practice, plants in low-light conditions may still photosynthesize and grow, albeit slowly plants in complete darkness will eventually die from nutrient starvation

### Sample conflictingqa_a864ff85e648

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3, d2
- **Claim**: Research using GH-transgenic mice has suggested that growth hormone may actually accelerate aging under certain conditions human studies have found that while GH treatment may increase muscle mass in healthy older adults, it does not necessarily translate into increased strength or overall health improvement

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The evidence is divided on whether cold water makes hair shinier. Some sources argue that cold water closes the hair cuticle and can make hair appear shinier and smoother, while others argue that the effect is negligible and that hot air from a hair dryer simply opens the cuticle back up again that hair is dead tissue and cold water has the same effect as warm water

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: However, most meteors are formed when Earth passes through a relatively fixed debris trail left by a comet the pieces are scattered in all directions, making a direct hit unlikely

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: In summary, the evidence presents conflicting research outcomes: while some studies confirm a recent brain size decrease, others attribute brain stability to metabolic constraints or describe a long-term trend of brain enlargement, making the answer complex and contested

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Ultimately, the question of whether the Bible is infallible is a matter of theological doctrine and interpretation different Christian traditions hold differing views on the extent to which the Bible's contents reflect God's word perfectly

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The retrieved evidence is mixed. Some sources argue that solar panels can produce more energy than they consume over their lifetime

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

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The question of whether Mormons are Christian is genuinely contested. Some sources argue that Mormonism is Christian because it professes belief in Jesus Christ and affirms itself as part of the body of Christ, while others argue that Mormonism's distinctive doctrines—such as the godhead, the concept of a pre-mortal existence the restorationist view of the Christian church—represent significant departures from historic, orthodox Christianity

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: This year's Passover began at sundown on April 1, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Their most recent World Series appearance was in 2022, when they lost to the Philadelphia Phillies, making their total count remain at one

### Sample freshqa_7dce5d575302

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Andrés Iniesta (2012)

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: This is the most recent data in the retrieved evidence, superseding older records from 2019–20 and later that had described the team as a playoff contender

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
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 9 minutes

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6
- **Supporting Docs Found**: d3
- **Claim**: John Speed

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

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The statue's design was further influenced by the idea of Libertas holding a torch and a tablet inscribed with the date of American independence, July 4, 1776

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

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Peyer's patches and lacteals are both lymphatic vessels found in the small intestine, but they are distinct structures with different functions: Peyer's patches are organized lymphoid follicles in the ileum involved in immune surveillance, while lacteals are central lymphatic capillaries in intestinal villi responsible for absorbing dietary fats and fat-soluble vitamins

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: These origins are not fixed sequences — in metazoans, a specific consensus sequence has not been identified their selection is largely epigenetic — and they fall into three main classes: constitutive, flexible dormant, with the vast majority (flexible origins) being used in an apparently stochastic manner in each cell cycle

### Sample qacc_8ef7b3cf5c3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: European ethnic groups dominate the Southern Cone region, which includes Argentina, Uruguay Chile

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

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: The first component, the Russian module Zarya, was launched on November 20, 1998 the first US component, the Unity Module, was launched on December 4, 1998 docked to Zarya on December 5, 1998

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: 245

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This joint allows movement in two planes, facilitating the transmission of sound vibrations from the tympanic membrane to the inner ear

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d2
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that one source incorrectly describes this joint as a hinge joint, but this classification is inconsistent with the established consensus across multiple sources, including high-credibility references

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Carter Pewterschmidt (voiced by Alex Borstein)

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The music for Disney's Robin Hood was created by a combination of composers across two different film adaptations. The 1952 live-action Robin Hood featured music composed by Elton Hayes, who drew upon medieval English melodies and wrote original songs for the film. The 1973 animated Robin Hood, on the other hand, featured music and lyrics by Roger Miller for songs like 'Whistle-Stop' and 'Oo-de-lally', as well as music by Floyd Huddleston for the character 'Love' the majority of the score was composed by George Bruns. These contributions were further confirmed by the soundtrack album releases for both films, which list the respective composers and lyricists for each track

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The Tavarez name has been carried by notable figures across a range of fields and has significant connections to the British peerage, tracing its presence in England since the medieval period

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: These dates reflect the gradual development of effigy mound culture over several centuries, with the custom dying out about 800 years ago

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: 1996 (the year)

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

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: An initialism is a type of abbreviation that is formed from the initial letters of a phrase and is pronounced letter by letter, rather than as a word. Examples include CEO, DNA FBI. While the term 'acronym' is sometimes used broadly to refer to any abbreviation formed from the first letters of a phrase, technically an acronym is pronounced as a word rather than individual letters, making initialism a distinct category

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Sushma Swaraj also held other firsts to her name, including being the youngest Cabinet Minister in Haryana in 1977 and the first woman Chief Minister of Delhi in 1998

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
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

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
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

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d2
- **Supporting Docs Found**: None
- **Claim**: It is composed of twelve members — seven from the Board of Governors and five presidents from Federal Reserve Banks — and meets regularly to decide on interest rates and the money supply through open market operations

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2
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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida Gators

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Arizona and Oklahoma follow with eight titles each, making UCLA the clear all-time leader

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d2
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

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 5 sharps in a key signature identify the key of B Major

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d4
- **Claim**: These distinct definitions reflect temporal and classification differences within the same abbreviation

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This result, superseding older data that had described the game as undecided

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 km

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3, d2
- **Claim**: The HEXA gene encodes the HEXA protein inheriting two variant copies of the HEXA gene causes deficiency of the hexosaminidase A enzyme, which is necessary to break down GM2-ganglioside within cells — its abnormal accumulation in brain and nerve cells eventually causes progressive central nervous system deterioration

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The Los Angeles Lakers last won a championship in 2020, when they defeated the Miami Heat 4-2 in the NBA Finals

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

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

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: It was a pivotal battle in which the Muslim army, numbering around 313 men, defeated a larger force of approximately 1,000 from the Quraysh tribe

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3, d1
- **Claim**: The longest wavelengths in the visible spectrum are approximately 700 nanometers, which is the range of red light

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence supports 1274 BC as the date of the Battle of Kadesh

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: He defeated Anthony Joshua to win the WBA title in September 2021, Tyson Fury to win the WBC in May 2024 Daniel Dubois to win the IBF in July 2025, unifying all four belts

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d4
- **Claim**: Previous winners like Brad Pitt, who earned the title twice in 1995 and 2000, are noted for their historic achievements, but Bailey's 2025 selection supersedes those records as the most recent winner

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Twitter is known as X

### Sample wikirevision_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: LinkedIn is owned by Microsoft, which acquired LinkedIn in December 2016. Microsoft's ownership of LinkedIn is further corroborated by the fact that LinkedIn's revenue is reported in Microsoft's annual reports

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

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Ballon d'Or winner is Ousmane Dembélé (France), who won the 69th Ballon d'Or ceremony in 2025. He earned his first Ballon d'Or, surpassing Lionel Messi's record of 8 wins is the first French player to win the award

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Calcutta is officially called Kolkata. The city officially changed its name from Calcutta to Kolkata in 2001 this change is recognized across all sources. The current official name is Kolkata it has been so since 2001 when the city changed its name from Calcutta

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Australia

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, including the current Wikipedia page on the 2026 Wimbledon Championships, which names Sinner as the men's singles champion with Julian Cash as his doubles partner

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The 2026 tournament was the 139th edition of Wimbledon, making it the most recent men's singles competition at the event

### Sample wikirevision_0120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, with the 2026 French Open representing the next iteration of the tournament

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

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, including the Wikipedia article on the 2026 Wimbledon Championships, which explicitly names Jannik Sinner as the men's singles champion, with Julian Cash and Lloyd Glasspool as the doubles champions

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The 2026 tournament was the 139th edition of Wimbledon, taking place from 29 June to 12 July 2026 Sinner's victory there makes him the current champion

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Australia

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest Ballon d'Or winner is Ousmane Dembélé, who won the 69th Ballon d'Or in 2025, marking his first win. This is confirmed across multiple sources, including the high-credibility Wikipedia articles on the 2025 Ballon d'Or and the Ballon d'Or award itself

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, with the 2026 French Open representing the next iteration of the tournament

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This victory is confirmed across multiple sources, with the 2026 French Open representing the next iteration of the tournament


================================================================================

*Report generated by CATS v2.0*
