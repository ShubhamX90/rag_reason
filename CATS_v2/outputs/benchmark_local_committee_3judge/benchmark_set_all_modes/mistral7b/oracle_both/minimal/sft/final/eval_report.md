# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 125 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.938 (over 736 samples)

**GR F1** *(used in CATS)*: 0.961

**Behavior Adherence**: 0.804 (over 611 applicable samples)

**Factual Grounding**: 0.825 (over 611 applicable samples)

**Single-Truth Recall**: 0.680 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.817

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.961
- **Precision**: 0.995
- **Recall**: 0.929
- **Accuracy**: 0.938
- TP=565, FP=3, FN=43, TN=125

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.744
- **Abstain Recall**: 0.977
- **Abstain F1**: 0.845
- **Specificity**: 0.929
- Abstain TP=125, FP=43, FN=3, TN=565


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (55 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.994
- **Behavior**: 0.904 (n=156)
- **Grounding**: 0.939 (n=156)
- **Recall**: 0.808 (n=154)
- **CATS**: 0.911

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.891
- **GR F1** *(used in CATS)*: 0.927
- **Behavior**: 0.915 (n=177)
- **Grounding**: 0.765 (n=177)
- **Recall**: 0.590 (n=156)
- **CATS**: 0.799

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.954
- **GR F1** *(used in CATS)*: 0.973
- **Behavior**: 0.667 (n=96)
- **Grounding**: 0.790 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.810

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.956
- **GR F1** *(used in CATS)*: 0.975
- **Behavior**: 0.683 (n=145)
- **Grounding**: 0.853 (n=145)
- **Recall**: 0.668 (n=140)
- **CATS**: 0.795

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.608 (n=37)
- **Recall**: 0.568 (n=37)
- **CATS**: 0.683


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2123

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

### Sample conflictingqa_04e1627e9fc5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2
- **Claim**: However, researchers consistently note that nematodes can respond to fertilizers in ways that are not necessarily associated with increased fertility, such as changes in trophic structure or disease transmission the specific mechanisms vary considerably depending on nematode group and fertilizer type

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1
- **Claim**: The poem itself explores intense human emotion and experience, including references to sex and drugs, which were cited by some critics as grounds for obscenity

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: However, iodine deficiency is a distinct and separate concern adequate intake is considered essential for thyroid hormone synthesis, meaning the appropriate balance is key

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Overall, the evidence is mixed and the potential risks are debated, with the most consistent finding being that artificial sweeteners help reduce sugar intake without causing hyperglycemia, rather than definitively resolving whether they are harmful

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: However, it is worth noting that certified sustainable palm oil (via RSPO standards) can mitigate some of these harms palm oil does have some benefits—such as lowering greenhouse gas emissions compared to other vegetable oils—meaning boycott is not necessarily the only solution

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d4, d2
- **Claim**: Related evidence

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, particularly if you plan to use them for more than 20 years. The main advantage is that real trees are grown as renewable crops — harvested trees are replaced with new seedlings, making them carbon sinks that absorb CO2 and produce oxygen, while artificial trees are non-biodegradable plastic and metal structures with high manufacturing and transport emissions

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, prominent scientists and advocacy groups contend that the practice normalizes animal cruelty, diverts funding from more effective conservation strategies that the evidence on its overall conservation impact is both mixed and largely promotional , with some experts arguing that blanket bans could actually reduce animal suffering and that the IUCN itself calls for trophy hunting to be replaced by more sustainable alternatives

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Internationally, the WIPO Magazine notes that excluding software from patent protection could hamper technical development and innovation, suggesting a role for software patents even within mixed legal landscapes

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: These conflicting findings reflect methodological differences in how researchers interpret the same phenomena, as probabilistic links to full moons do not necessarily constitute definitive causation

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Overall, the evidence points to vitamin C having a modest beneficial effect on cold symptoms, but the evidence is not definitive and the potential risks of high-dose supplementation should be carefully considered

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, some sources argue that organic farming has other sustainability advantages — such as lower fossil fuel use and less nitrogen pollution — that make it comparable to or even superior to conventional farming for certain metrics , while others note that organic farming's lower yield potential can be offset if food waste is reduced

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The debate thus remains contested and depends on how 'efficiency' is defined

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d5, d4
- **Claim**: Overall, the evidence points to complex and context-dependent effects, rather than a clear yes/no answer

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Overall, the evidence suggests that the health impacts of spaying or neutering depend heavily on species, sex, age individual animal factors, rather than being a one-size-fits-all conclusion

### Sample conflictingqa_9ceca2645833

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Others argue that affirmative action can have discriminatory effects: it may disproportionately benefit minority students over non-minority ones in the employment context, the Supreme Court has treated voluntarily adopted programs with greater severity than those adopted after findings of discrimination

### Sample conflictingqa_9ceca2645833

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d1
- **Claim**: Still others argue that affirmative action is inherently discriminatory because it gives preferential treatment to certain groups over others, even if the intent is to correct historical injustices

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2
- **Claim**: Contrary to a common misconception, stalactites do not require dry conditions to grow — they can form underwater as long as there is sufficient calcium carbonate-rich water flowing over a suitable surface

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: However, some researchers also acknowledge that methane release from organic-rich permafrost or peatlands could have been involved, suggesting a more complex sequence of feedbacks

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: If you do choose to use coffee grounds as a deterrent, it is recommended to test their effectiveness first, as results can vary depending on concentration and slug appetite

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, experts note that investors can protect themselves by focusing on transparent liquidity, verified project fundamentals reliable exchanges regulators are increasingly cracking down on manipulative practices

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A third perspective, called coherentism, holds that beliefs are justified solely through their consistency with each other within a coherent system, which could potentially allow for false beliefs to be justified alongside true ones

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: The retrieved evidence presents a genuine curse: folklore states that witches objected to Shakespeare's use of their real incantations in the play, causing the actor playing Lady Macbeth to die during the first performance (around 1606)

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Emojis are widely used as a supplement to written language, but whether they constitute a distinct form of written language themselves is a genuinely contested question. Most language specialists agree that emojis function as a complex system of pictographs

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Some low-credibility sources suggest yerba mate may have anti-cancer properties that could protect against some cancers, though this evidence is limited and does not contradict the carcinogenic findings from epidemiological studies ; the overall picture remains complex and incomplete

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d1, d5, d2
- **Claim**: The retrieved evidence is conflicting. Some sources say the Phoenix Lights were explained as military flares deployed during a training exercise, while others say witnesses believed the lights were UFOs or not flares at all

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1
- **Claim**: Brontosaurus and Apatosaurus are not the same dinosaur. They were once classified as the same species (Apatosaurus excelsus), but a 2015 study found that they are distinct genera with consistent differences in skull shape, making the earlier classification invalid. The current scientific consensus is that Brontosaurus is a valid genus distinct from Apatosaurus, though the two remain closely related sauropods

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4, d5
- **Claim**: Religion scholars and theologians hold differing views. Some sources argue that Mormons are Christians because they believe in Jesus Christ and follow his teachings, while others argue that Mormon theology and practice are fundamentally at odds with the historic Christian faith

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: This ranking is corroborated by multiple sources reporting the same sequence of languages by global reach

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: King Charles III did not immediately strip Prince Harry of his title as the Duke of Sussex when he became king; the two brothers first discussed the matter during the Sandringham summit in January 2020

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Maryam Mirzakhani (born May 3, 1977, Tehrān, Iran—died July 14, 2017, Palo Alto, California, U.S.)

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2d7eb41139aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This escalated to a full-scale war that has displaced millions of people and resulted in thousands of deaths

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Three seasons — seasons 1 through 3 — of The Mandalorian have been released, premiering on November 12, 2019, October 30, 2020 March 1, 2023 respectively

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4e635a2542a8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This lineup was assembled in 1955 Garland remained the pianist even after Cannonball Adderley joined in 1958

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The film, directed by Paul Thomas Anderson, also won Best Director and Best Adapted Screenplay, making it a major sweep of multiple categories

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d3, d2
- **Supporting Docs Found**: None
- **Claim**: This result, superseding earlier reports that listed *CODA* (2022) or *Sinners* (2024) as the most recent winners, as those ceremonies have since been surpassed by the 2026 event

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
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: He has consistently confirmed this birthplace across multiple sources, including his own website and biographical information about him

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Beijing is the first city to have hosted both the Summer and Winter Olympic Games

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel was won by *When We Were Real* by Daryl Gregory, published by Saga Press. The award was announced at the 2025 Nebula Conference, making this the most recent recognition for the work

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Frank Rosenblatt died in a boating accident on July 28, 1971, in Chesapeake Bay

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by Britannica, which identifies 26 books and notes that Hoover self-published many of them on social media platforms

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Bezos still held approximately 900 million Amazon shares, valued at close to $200 billion

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_cf331ed7d09f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This is the 22nd major version of macOS and marks the transition away from the older 2020 release scheme

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

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d5
- **Claim**: The new discoveries include a wide variety of figures, such as humans, animals geometric shapes, spread across the Nazca Pampa desert

### Sample freshqa_fd00b29e848c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample healthcontradict_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Additional sources suggest that yoga's effectiveness may depend on the type of yoga practiced — with hatha yoga showing mixed results and therapeutic yoga showing more consistent benefits — and that people with asthma should exercise caution when trying new physical activities without medical supervision

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

### Sample hotpotqa_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0073

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: However, d4 (a low-quality source) falsely claims that El Nuevo Cojo Ilustrado is also owned by Time Inc., creating a misinformation that is directly contradicted by the correct answer

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d8
- **Claim**: Lucas di Grassi

### Sample hotpotqa_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d7
- **Supporting Docs Found**: d9
- **Claim**: The band formed in 1995 in Fullerton, California *A Place in the Sun* was their second studio album, recorded in Anaheim and Fullerton

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9
- **Supporting Docs Found**: None
- **Claim**: Starks played in 76 games for the Jazz that season, averaging 7.9 points per game

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d6, d5, d4, d7
- **Claim**: Justin Timberlake and Pusha T wrote the song together; Pusha T co-wrote the jingle and Timberlake sang it, but neither exclusively wrote the entire composition

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Madhuri Dixit

### Sample qacc_132167e66120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_213701765f94

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The landing was confirmed by multiple sources, including the National Transportation Safety Board and the Federal Aviation Administration

### Sample qacc_2e1b5edb5e0d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: It is a translucent layer composed of dead, keratinized cells that provides an additional barrier against friction and shear forces in areas of high mechanical stress

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: A separate theory suggests the practice originated among early Christians, who used the crossed-finger gesture as a secret sign to recognize each other and invoke the power of the Christian cross for protection

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3
- **Claim**: Some historians further propose that the modern one-handed cross was popularized around the Hundred Years' War, when people crossed their middle finger over their index finger to invoke God's protection, eventually evolving into the familiar solo gesture

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_44b315f6f4bb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This initial union was further expanded by the addition of Manitoba and the Northwest Territories in 1870, British Columbia in 1871, Prince Edward Island in 1873, Yukon in 1898 Alberta and Saskatchewan in 1905

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by both historical records and contemporary accounts, which describe how the tree was decorated with candles and sweets in anticipation of a Christmas party

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Additionally, certain countries such as those in the Schengen Area allow U.S. passport holders to stay for up to 90 days within a 180-day period without a visa

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: October 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nana in Snow Dogs is an Australian Shepherd

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This clinching came after a dramatic Game 162, where the Red Sox defeated the Yankees 2-1 at Fenway Park

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Russ Ballard; the song was also covered by Kiss and Petra

### Sample qacc_a1c73439eca9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: Production for season 10 began in September 2023 the official trailer was released on 11 January 2024 , but the premiere date was confirmed much later as July 2026

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2, d4
- **Claim**: 2003 was the birth year of T20 cricket, with the inaugural match taking place between the same two counties at a ground in England

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912, when President William Taft signed the New Mexico statehood bill

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d4
- **Supporting Docs Found**: None
- **Claim**: This is consistently confirmed across multiple sources, with New Mexico entering the Union on the same date as Arizona, forming the 47th and 48th contiguous states

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: This joint allows for movement and sound transmission between the two bones, which is essential for hearing mechanics

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_c88807a22775

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: These rifles are specifically designed for biathlon competition and are tested at the range before each race to ensure they are accurate and reliable

### Sample qacc_c88807a22775

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The cartridges are also carefully monitored to ensure they meet the specifications required by the International Biathlon Union

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1
- **Claim**: The most intensive period of mound construction occurred between 800 and 1200 A.D., after which the practice died out approximately 800 years ago

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple authoritative sources, including the official Social Security Administration website and the St. Louis Federal Reserve Economic Data (FRED)

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: 45D is the tolled version of Federal Highway 45, while Fed

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: 45 is the free version

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: In that campaign, Rangers finished second in their group and were eliminated in the round of 16, marking their best performance since the 1990s

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells

### Sample qacc_fbdae168fc6f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d3, d1
- **Claim**: Sushma Swaraj's tenure as the first full-time woman MEA is consistently confirmed across multiple sources, superseding earlier partial information

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d1
- **Claim**: While the program's origins trace back to earlier proposals and the 1934 Committee on Economic Security, the modern Social Security system was established by the 1935 Act

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Looking further back, the 20th-century experience was marked by high immigration from Eastern and Southern Europe, including large numbers from Italy, Austria-Hungary, Russia Poland , while the more recent shift toward South America and the Caribbean has been accelerated by economic and political conditions in those regions

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This figure is derived from the most authoritative source of administrative boundary data in the country and is further corroborated by additional sources reporting similarly high figures

### Sample situatedqa_geo_897e47478bbc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: It is important to note that this count refers to inhabited villages specifically the broader definition of villages (including uninhabited settlements) would likely result in a higher total count

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a grizzly bear, specifically the California grizzly bear (Ursus arctos californicus), as its official state animal. This extinct population of the brown bear was once found throughout the state the bear's image on the flag is based on the California grizzly bear, which became the official state animal in 1953

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: He is a senior BJP leader and has served in various legal and political roles, including as Minister of State for Home Affairs and as a Member of Parliament

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: In response, delegates from five states met in Annapolis in September 1786 to discuss revising the Articles, eventually agreeing to call a constitutional convention in Philadelphia

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: That convention produced the U.S. Constitution, which was ratified in 1788, replacing the Articles of Confederation as the definitive national government

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: Today, federal environmental policy covers a wide range of issues including air and water quality, hazardous waste management climate change mitigation, with the EPA playing a central role in implementing these policies

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: England are the current champions, having won the 2019 World Cup, with New Zealand as the runner-up

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Mort is a Goodman's mouse lemur, a small primate native to Madagascar, though a spin-off series reveals he is also part bear, spider starfish

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Argentina defeated France 4-2 in the final held at the Lusail Stadium in Qatar, claiming their third title

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: It is written by Brandon Hoáng and illustrated by BellBessa, serving as a direct sequel to the animated series

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The election was marked by widespread allegations of rigging from several parties, including the Pakistan Muslim League-Nawaz (PML-N) and Pakistan People's Party Parliamentarians (PPPP), which were denied by the PTI and the army

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The PTI's victory gave Imran Khan his first term as Prime Minister, after which he promised to investigate the allegations of irregularities

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 km

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: Regarding the North American division specifically, the most current data points to approximately 1.2 million members across the United States and Canada

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5, d1
- **Claim**: It was originally formed from the Old German name Gerhard, which itself consisted of the elements gēr ('spear') and hard ('hardy, brave strong')

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4, d5
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin & Perry Go Large

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

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The X branding is consistent across all platforms, including the official X website, which confirms that the service is now known by that name

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. The retrieved evidence consistently identifies Alphabet Inc. as the company that owns Google. The newer Wikipedia revision directly states that Alphabet Inc. is the parent company to Google, while the older revision and additional sources further confirm this ownership through a 2015 reorganization that made Google a wholly owned subsidiary of Alphabet Inc

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Argentina defeated France 4-2 on penalties after extra time in the final, becoming the first nation from outside of Europe to win the tournament since 2002

### Sample wikirevision_0097

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This name change is consistently reflected across all sources, with the older name 'Bangalore' used primarily as a historical or cultural reference

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This victory gave India their sixth Cricket World Cup title, making them the most successful team in the tournament's history

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed by the Haryana government in 2023 the name change is consistently reflected across all sources including the newer Wikipedia revision of Gurgaon. The official government website of Gurugram further corroborates this, stating that the city is located in Haryana and officially known by that name

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore is officially called Bengaluru. The city officially changed its name from Bangalore to Bengaluru on 1 November 2014 this is confirmed by the Bangalore City Municipal Corporation resolution of 2014-11-01. As a result, Bangalore is no longer the official name of the city, which is now officially called Bengaluru

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The latest Wimbledon men's singles champion is Jannik Sinner, who defeated Matteo Berrettini in the 2025 final to win his first major title

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 Wimbledon Championships are scheduled to take place from 29 June to 12 July 2026, at which point the current champion will be crowned

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4, d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 French Open was the most recent Grand Slam tournament, held at the Stade Roland Garros in Paris from 24 May to 7 June 2026 Alcaraz's victory there marked his second French Open title and fifth major. Multiple sources confirm his 2026 championship, superseding older records that identified him as the 2025 or 2023 winner, making him the latest French Open champion


================================================================================

*Report generated by CATS v2.0*
