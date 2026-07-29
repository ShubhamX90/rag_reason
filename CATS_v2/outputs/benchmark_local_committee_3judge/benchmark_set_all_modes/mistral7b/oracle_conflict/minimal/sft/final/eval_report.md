# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.933 (over 736 samples)

**GR F1** *(used in CATS)*: 0.958

**Behavior Adherence**: 0.803 (over 608 applicable samples)

**Factual Grounding**: 0.787 (over 608 applicable samples)

**Single-Truth Recall**: 0.637 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.796

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.958
- **Precision**: 1.000
- **Recall**: 0.919
- **Accuracy**: 0.933
- TP=559, FP=0, FN=49, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.723
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.839
- **Specificity**: 0.919
- Abstain TP=128, FP=49, FN=0, TN=559


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.877 (n=154)
- **Grounding**: 0.863 (n=154)
- **Recall**: 0.766 (n=154)
- **CATS**: 0.873

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.923
- **GR F1** *(used in CATS)*: 0.949
- **Behavior**: 0.892 (n=176)
- **Grounding**: 0.774 (n=176)
- **Recall**: 0.532 (n=156)
- **CATS**: 0.787

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.945
- **GR F1** *(used in CATS)*: 0.968
- **Behavior**: 0.656 (n=96)
- **Grounding**: 0.786 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.803

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.905
- **GR F1** *(used in CATS)*: 0.945
- **Behavior**: 0.759 (n=145)
- **Grounding**: 0.775 (n=145)
- **Recall**: 0.650 (n=140)
- **CATS**: 0.782

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.811
- **GR F1** *(used in CATS)*: 0.896
- **Behavior**: 0.622 (n=37)
- **Grounding**: 0.590 (n=37)
- **Recall**: 0.486 (n=37)
- **CATS**: 0.648


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2065

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
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: Internationally, the protection of fashion designs varies greatly by country, with many nations — including the European Union — offering more comprehensive protection under their own design-specific directives the World Intellectual Property Organization has registered international designs to facilitate cross-border enforcement

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3
- **Claim**: Overall, the evidence suggests that while peeling does not completely eliminate vitamins or fiber, it does reduce their concentration significantly, making it a trade-off between palatability and nutritional value

### Sample conflictingqa_151865dc414b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: However, when the cause cannot be changed or adequately treated, tinnitus management focuses on reducing its impact on daily life through sound therapy, hearing aids other self-management techniques

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d1, d4
- **Claim**: The answer depends on how you count them: cows have four stomachs if you combine all four compartments (rumen, reticulum, omasum abomasum), but they technically have only one stomach if you treat each compartment as a distinct organ

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Ultimately, whether a multivitamin is appropriate depends heavily on a child's individual dietary profile parents should always consult their physician before starting any supplement

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Yes, hair can turn green from swimming pools — but not directly from chlorine. The real culprit is copper, which is commonly found in algaecides used to control algae growth in pools. When copper oxidizes (exposed to air), it turns from shiny orange to a dull green when it comes into contact with hair, it sticks to the proteins and causes green discoloration. Chlorine actually works to lighten hair, not turn it green — this lightening effect is what sets the stage for copper to oxidize and cause the green color

### Sample conflictingqa_287debe5bb90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Religious and philosophical views differ; science has not established that humans can know anything beyond their minds with certainty

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, as they have negligible or negative carbon emissions and can be recycled or planted again, while artificial trees require large amounts of fossil fuels for manufacturing and end up in landfills

### Sample conflictingqa_411445406724

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The conflicting opinions reflect ongoing scientific debate about how to define 'dominance' and which fossil groups to highlight, making the answer incomplete and contested

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, the Supreme Court's Alice decision has created ongoing uncertainty about whether software should be patentable subject matter at all the EPC excludes computer programs 'as such' from patentability, though the Board of Appeals has narrowly interpreted this to apply only to non-technical programs

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: In short, whether software qualifies for patent protection depends heavily on the specific facts of each case the legal framework remains a subject of active debate

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d5, d2, d4
- **Claim**: Overall, while the evidence suggests a

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: The conflicting findings may reflect methodological differences in how moon's tidal effects are measured and interpreted, as some researchers argue that full moons may increase the probability of a large earthquake growing to a destructive size rather than causing a separate, standalone event

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Split ends cannot be permanently repaired or healed. The hair shaft is dead tissue that cannot regenerate, so any damage once formed will remain. The best you can do is to temporarily mask or coat the damage with protective products to prevent split ends from forming in the first place through proper care

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: However, some states have enacted stricter rules: Maine forbids ISPs from selling personal data without express consent California residents can opt out of having their data sold under the state's Consumer Privacy Act

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3
- **Claim**: However, bees can tolerate light rain and will forage in it if absolutely necessary research has shown they can distinguish a light shower from a full-blown storm

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d3
- **Claim**: The overall picture is that saturated fats may elevate heart disease risk markers such as LDL cholesterol, but whether this translates to a substantially increased risk of heart disease incidence remains an open scientific question requiring further investigation

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: However, other research presents a more nuanced picture — a comparison of 140 conventional and organic farms in the U.S. found that while organic farming is less efficient in terms of yield, it is matched or surpassed conventional farming in other sustainability measures such as biodiversity, soil health carbon footprint a meta-analysis of 51 studies found that organic farming ranks as the more sustainable method overall

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d5
- **Supporting Docs Found**: None
- **Claim**: The key seems to be context: organic farming's lower yield efficiency is more than compensated by its environmental and health benefits, making it broadly comparable to conventional farming when those factors are taken into account

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Religious organizations and scholars hold differing views. Some sources argue that the Catholic Church is the one true church because it traces its origins to Jesus Christ and holds an unbroken apostolic succession, while others argue that 'one true church' in the New Testament refers to a church that aligns with Scripture rather than being the historically first church

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence presents multiple perspectives: d2 and d4 argue multiculturalism is a hindrance to unity via ethnic segregation, while present opposing scholarly perspectives that multiculturalism can facilitate unity d3 highlights that research is insufficient on its broader effects. [[

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The overall picture is complex and contested across species, ages health outcomes, making it impossible to give a single definitive answer

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2, d3
- **Claim**: In short, while sexual contact is the primary route of transmission, gonorrhea can spread in a variety of ways safe sex practices remain essential regardless of the specific activity

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: First-time pet owners should also be aware that giant African land snails are hermaphrodites and can lay eggs without mating, meaning a single snail can produce offspring — which can quickly multiply if not managed appropriately

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d5, d2, d4
- **Claim**: [[

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The most current evidence points to the industrial era as the source of this rapid recent rise — with CO2 concentrations stabilizing at around 280 ppm before the 1760s and reaching 430 ppm by 2

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: Taken together, the evidence suggests that human brain size has not uniformly decreased — rather, it has changed and varied in response to shifts in body size, diet social complexity

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Religion and cultural perspectives differ; science has not eliminated death as a taboo topic in modern society

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Relatively few philosophers would argue that a belief can be justified and yet false; the majority view is that justification requires truth. The classic Gettier-style counterexample shows that even a well-supported belief can be false (Tom may live in San Francisco when you justify that he lives in California) this remains a live epistemological concern: if justification is possible for false propositions, then knowledge cannot be identified with justified true belief

### Sample conflictingqa_f22b389be1d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: Additional factors such as injury incidence, gait mechanics individual biomechanics further complicate a definitive answer, making it important for each runner to consider their own needs and risks

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: However, other research suggests yerba mate may also have anti-cancer properties: in vitro studies have shown it has cytotoxic effects on cancer cells no direct causal link has been established between yerba mate consumption and cancer incidence in humans

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Overall, the evidence does not support yerba mate as a proven cancer cause

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Religion is contextually defined Mormons consider themselves Christians in the sense of believing in Jesus Christ and following His teachings; however, they are not recognized as Christians by many mainstream Christian organizations because their theology is seen as fundamentally alien to biblical orthodoxy

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Hindi is the third largest language by total number of speakers, after English and Mandarin Chinese

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence indicates that Prince Harry's Duke of Sussex title was stripped by King Charles III in the aftermath of the Sussexes' departure from their royal roles in 2020

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: King Charles III was reportedly considering stripping Harry of his titles the Duke of Sussex himself acknowledged that such a move would not make a significant difference

### Sample freshqa_049cc3f14d5e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: April 1, 2026; April 2, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani (1977–2017)

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The latest stable Android version is Android 16, released on June 10, 2025. This version is currently available for Google Pixel phones and Samsung Galaxy devices, with wider rollout to other manufacturers expected

### Sample freshqa_2d7eb41139aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This count is corroborated by the official Ace Attorney website, which lists Phoenix Wright, Apollo Justice Spirit of Justice as the most recent installments

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: The device, code-named 'Gadget,' was detonated at approximately 5:30 a.m. on a 100-foot steel tower, releasing approximately 18.6 kilotons of energy

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d5
- **Supporting Docs Found**: None
- **Claim**: This test was part of the Manhattan Project the site is now part of the White Sands Missile Range

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: This escalated to a full-scale war that has resulted in hundreds of thousands of deaths and millions of displaced people

### Sample freshqa_3cd0b514193b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d3
- **Claim**: For national context, the average minimum wage across all 47 prefectures is approximately ¥1,121 per hour, with rural prefectures like Okinawa having the lowest rate at ¥1,023

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Three seasons — the show has not yet released Season 4

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: Portugal won the 2017 Eurovision Song Contest, marking the country's first victory since 1964. The winner was Salvador Sobral, representing Portugal, with the song "Amar pelos dois"

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: She was crowned the champion after receiving the most votes from an in-studio audience of past contestants and superfans, giving Adam Levine his fourth win as a coach

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5
- **Claim**: Runner-up was Liv Ciara from Team Kelly, with Lucas West and Mikenley Brown placing third and fourth, respectively

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Two

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Laika

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Martin himself has also repeatedly confirmed his birthplace, stating that growing up in Bayonne provided the inspiration for many of the settings in his novels

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a8b908895e11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The city is named in honor of Saint Joseph and has a rich cultural history, featuring notable museums such as the Gold Museum and the Jade Museum

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Bezos still held over 900 million Amazon shares, valued at close to $200 billion

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Star Wars: The Rise of Skywalker is the most expensive film ever made, with a net production budget of roughly $490 million

### Sample freshqa_dd85dcbc2262

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Iga Swiatek

### Sample freshqa_dd87e1e3ad3d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 9

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f5eef5c415e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Romeo Beckham

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: 1864

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Fever is a common symptom of scarlet fever, which can be life-threatening if not treated promptly

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Research suggests that specific yoga techniques may help reduce symptoms and improve exercise capacity in some individuals, but commercial yoga programs and general yoga practice have not been shown to improve asthma control

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1, d3, d6
- **Supporting Docs Found**: d10
- **Claim**: It has been the home of Premier League club Everton since the stadium's completion in 1892 the club itself is based in Liverpool, Merseyside

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Some sources suggest that bleach can kill bacteria and treat infections when applied topically or used as a disinfectant, but these claims refer to external use on surfaces or in controlled sanitation settings, not human ingestion

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d5, d7, d6, d4
- **Claim**: Pusha T

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d7, d2, d8, d6, d4
- **Claim**: 506

### Sample qacc_08cf866bcb9b

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: The retrieved evidence indicates that after landing in North Africa, the Allies (British and American forces) continued to push eastward across the continent, eventually reaching Tunisia and defeating the Axis powers there in May 1943

### Sample qacc_160a528ae07e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Tom Brady has never won the NFL's regular season MVP award, giving him a total of zero MVPs for his career

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Oliver Stark plays Buck on the TV show 9-1-1

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The film is set in a fictional southern Louisiana town called the Bathtub, which is described as a marshland community on the edge of the ocean

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: The exact origins of crossing fingers for good luck are not definitively known, but the evidence points to pre-Christian times and pagan beliefs as the root. Ancient Europeans practiced elaborate hand gestures, believing that crossing their index and middle fingers formed a potent magical sigil associated with binding and securing outcomes. This cross-finger position was also used in pre-Christian oath-swearing ceremonies among Norse and Germanic pagans, Anglo-Saxons early Christians, who crossed their thumbs and index fingers to form an 'L' shape — a symbol later evolved into the Christian fish symbol (Ichthys) used for recognition and worship

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_4387048ed24f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Queen Charlotte's tree was described as a 'festive centrepiece' at her 1800 party the practice continued throughout her lifetime

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9, d3, d5
- **Supporting Docs Found**: None
- **Claim**: These figures reflect different levels of resolution and complexity, as some organisms may have fewer origins (such as yeast) the number of origins can change during development and in response to stress

### Sample qacc_8882ab46be5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The show's writer Charlie Covell explicitly confirmed that the finale was shot on the Isle of Sheppey, explaining that they were trying to create a sense of expanse akin to the American Midwest

### Sample qacc_946ecfb478b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: Justin Timberlake wrote the song "Can't Stop the Feeling!" for the 2016 DreamWorks Animation film *Trolls*, together with producers Max Martin and Shellback

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This clinching came after a dramatic final week of the regular season, where the Red Sox defeated the Yankees in a Game 162 tiebreaker to secure their division title

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: Russ Ballard

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2
- **Claim**: The model asserts that the primary responsibility for controlling abusers belongs to the community and the individual abuser, not the victim defines battering as a pattern of coercion and violence used to intentionally control or dominate an intimate partner

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: 2003 was the birth year of T20 cricket, with the format quickly gaining popularity before the first official competition

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2, d5
- **Claim**: In practice, a yellow 35 mph sign indicates the measured safe speed for a specific curve or series of curves drivers who exceed this suggested speed risk a ticket — though the specific speed limit at that point may be higher

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Celebrity Big Brother is typically covered by CBS in the USA

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912, when President William Taft signed the New Mexico statehood bill

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d4
- **Supporting Docs Found**: None
- **Claim**: Some sources cite slightly different figures due to temporal updates — one source refers to New Mexico as the 48th state another source notes it was the 47th state admitted west of the Mississippi — but these reflect minor variations in counting methodology, not factual disagreement about the core admission date

### Sample qacc_b1bd9518429b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The conflict has simmered for centuries and came to a boiling point in 2025 when Spain insisted on imposing border checks at Gibraltar's land frontier with Spain, prompting the UK to announce legal action at the UN

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: [[

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Elton Hayes

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple authoritative sources, including the official Social Security Administration website and the St. Louis Federal Reserve Economic Data (FLED)

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The equation is derived from the basic accounting equation, which expands to Assets = Liabilities + Capital + Revenue − Expenses − Drawings, further clarifying the role of each component

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ecd3d9c0ca11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d5
- **Claim**: The first character is always alphabetic, identifying the organ system or body part affected, while the remaining characters provide additional specificity such as etiology, anatomic site procedure code

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: seven

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: In the UK, the minimum age to purchase and consume alcohol is 18 years old

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1, d3
- **Claim**: The exact meaning depends on the region and context, making it important to consider the specific jurisdiction when interpreting a red license plate

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by multiple sources reporting 24 million Soviet casualties, 5.3 million German military deaths 2.12 million Japanese military deaths, with broader estimates ranging from 50 to 56 million

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

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Coton in the Elms

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The program has since expanded to cover additional categories such as spouses and children today provides benefits to millions of Americans

### Sample situatedqa_geo_779fd84224fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: The fleet, which consisted of 11 ships and carried over 1,500 people, including crew, soldiers convicts, had set sail from Portsmouth, England in May 1787

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4, d5
- **Supporting Docs Found**: None
- **Claim**: The U.S. is not the only country with this form of government, as other nations similarly organize their administrative structures around these three basic components

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: In short, the 'bulk' of immigrants has shifted significantly over time, reflecting changing global and demographic realities rather than a single constant origin [d

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: The bear on the flag is a symbol of strength and unyielding resistance, originating in 1846 when California was part of Mexico

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: After several years of experimenting with this form of self-government, delegates from five states met in Annapolis in September 1786 to discuss revising the Articles, eventually deciding instead to draft a new Constitution that replaced it in 1787

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: The White House was set on fire on August 24, 1814, during the War of 1812, when British troops occupied Washington, D.C. President James Madison and his wife Dolley had fled the city the British troops reportedly sat down to eat a meal using White House dishes before setting the building ablaze. This burning marked the only time in U.S. history that Washington, D.C. was occupied by a foreign military force

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: By the 19th century, coffee had fully eclipsed tea as the dominant beverage in the U.S., with Italian, Cuban, Mexican Eastern European immigrants bringing their coffee traditions and reinforcing American coffee culture

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: In Europe, the shift was similarly gradual — influenced by changing tastes, shifting production patterns the rise of coffeehouses — with some nations like France and Italy retaining strong tea-drinking traditions alongside coffee

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_657c130afab6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This makes it one of the more recent national parks in the United States, spanning over 13.2 million acres in Alaska's Southcentral region

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, Beowulf uses the kenning 'sea-wood' to refer to the boat he and his men travel on, emphasizing its material simplicity

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1, d3, d5
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: [[

### Sample situatedqa_temp_901be1437bc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d5
- **Supporting Docs Found**: None
- **Claim**: This location is further corroborated by population distribution data showing that New York State had a uniform population spread in 1790 , indicating that the center of population lay within the Chesapeake Bay region of Maryland

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Beyond red, the human eye can perceive near-infrared light (up to about 1100 nm) and ultraviolet light (below 400 nm), but these are not considered part of the visible spectrum

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: In any case, the evidence consistently points to the name's Germanic linguistic root, with early records showing it was first used as a personal name in the High Middle Ages before becoming established as a surname in the Late Middle Ages

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. The snippet directly states that Alphabet Inc. is the company formerly known as GOOG that Google is a subsidiary of Alphabet Inc. The document further corroborates this by stating that Alphabet Inc. acquired Wiz, Inc. in March 2026 that Google is the largest subsidiary of Alphabet Inc. The answer is consistent across both documents, with the newer revision explicitly naming Alphabet Inc. as the owner of Google

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Argentina (defeated France 4-2 on penalties after a 3-3 draw)

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 ceremony was the 70th annual edition, covering the 2025–26 season, with the nominees announced on 7 August 2026 and the ceremony taking place on 22 September 2026

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2, d1
- **Claim**: This is consistent across multiple sources, including the official Wikipedia entries on both the Leader of the Labour Party and the Deputy Leader of the Labour Party, as well as external references

### Sample wikirevision_0093

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia (defeated India by six wickets on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad)

### Sample wikirevision_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The latest Wimbledon men's singles champion is Nick Kyrgios, who defeated Dan Evans in the 2026 final to win his first major title. The 2026 championships were the 139th edition of the tournament, held at the All England Lawn Tennis and Croquet Club in Wimbledon from 29 June to 12 July 2026. This result is corroborated by the Wikipedia page on the 2026 Wimbledon Championships, which confirms Kyrgios's victory

### Sample wikirevision_0150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia (defeated India by six wickets on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad)

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The current President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022. He is the 37th President of the Philippines and serves as both head of state and head of government. This is consistent across multiple sources, including the official Wikipedia article on the President of the Philippines, which confirms his incumbency from June 30, 2022

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of India is Droupadi Murmu, who became the country's head of state on 24 July 2022. This is confirmed by the official Wikipedia revision that superseded the older version in February 2025, which explicitly names her as the current president with a detailed biography. As of 2026, she is the 15th President of India, having taken office after being elected by the Electoral College comprising members of Parliament and state legislative assemblies


================================================================================

*Report generated by CATS v2.0*
