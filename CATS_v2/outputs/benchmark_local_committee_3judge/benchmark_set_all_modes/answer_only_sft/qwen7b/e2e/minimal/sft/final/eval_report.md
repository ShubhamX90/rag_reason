# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.957 (over 736 samples)

**GR F1** *(used in CATS)*: 0.973

**Behavior Adherence**: 0.772 (over 609 applicable samples)

**Factual Grounding**: 0.793 (over 609 applicable samples)

**Single-Truth Recall**: 0.643 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.795

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.973
- **Precision**: 0.998
- **Recall**: 0.949
- **Accuracy**: 0.957
- TP=577, FP=1, FN=31, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.804
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.888
- **Specificity**: 0.949
- Abstain TP=127, FP=31, FN=1, TN=577


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.903 (n=155)
- **Grounding**: 0.835 (n=155)
- **Recall**: 0.773 (n=154)
- **CATS**: 0.874

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.946
- **GR F1** *(used in CATS)*: 0.965
- **Behavior**: 0.875 (n=176)
- **Grounding**: 0.773 (n=176)
- **Recall**: 0.526 (n=156)
- **CATS**: 0.785

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.479 (n=96)
- **Grounding**: 0.826 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.765

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.956
- **GR F1** *(used in CATS)*: 0.975
- **Behavior**: 0.738 (n=145)
- **Grounding**: 0.791 (n=145)
- **Recall**: 0.671 (n=140)
- **CATS**: 0.794

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.811
- **GR F1** *(used in CATS)*: 0.896
- **Behavior**: 0.622 (n=37)
- **Grounding**: 0.640 (n=37)
- **Recall**: 0.486 (n=37)
- **CATS**: 0.661


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2031

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
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: The overall effect varies by fertilization type — organic fertilizers generally support greater nematode diversity and abundance than inorganic ones, though excessive inorganic fertilizer application can reduce the abundance of sensitive nematode groups

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d2
- **Claim**: The conflicting perspectives reflect methodological differences: some studies focus on short-term blood pressure spikes during individual lifts, while others examine the net effect of chronic strength training, which tends to be lowering

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: Yes, peeling an apple does remove some nutritional value — approximately 50% of the total dietary fiber and around 30% of vitamin C are lost when the peel is removed

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: However, a peer-reviewed study published in Frontiers in Nutrition warns that high intake of artificial sweeteners is associated with all-cause mortality, cardiovascular risk, coronary artery disease risk, cerebrovascular risk cancer risk that the replacement of sugar with artificial sweeteners may worsen glycemic control due to altered gut microbiota

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d3
- **Claim**: A middle-ground view holds that whether epigenetic changes are hereditary depends on the specific mechanism: for example, DNA methylation can be transmitted via sperm in some animal models, but the broader evidence base remains inconclusive

### Sample conflictingqa_37ab7146eb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: True pain involves conscious experience and emotional response, which are currently understood as functions of biological nervous systems rather than computational frameworks

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The debate is further complicated by medium differences: while content is identical, the format — physical book vs. spoken word — alters the reading experience, which some critics argue changes the nature of engagement

### Sample conflictingqa_3c835387fe6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: A peer-reviewed study published on PLoS ONE corroborates that Australia served as a hub for lizard evolution, with fossilized bones from Queensland matching those of present-day Komodo dragons

### Sample conflictingqa_42d60ecaee9f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: A socio-semantic analysis further complicates the picture, suggesting that emoji's meaning is drawn from context rather than a fixed rulebook, which is also true of hieroglyphs and cuneiform

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: Contested — some research (IUCN, Royal Society) argues trophy hunting can provide revenue and incentives for conservation, while critics argue it selects against rare individuals, can drive local extinction is morally questionable; context matters greatly

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: It is not unconstitutional for students to pray in school; the U.S. Supreme Court has made clear that the Establishment Clause does not prohibit students from praying individually or in groups during non-instructional time

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The question of whether patents should apply to software is a deeply contested normative debate with no clear resolution; the debate is instead driven by practical considerations. The U.S. Patent Office issues approximately 62% of its patents to software-related inventions, indicating that software is already treated as patentable subject matter under current U.S. law, though the Supreme Court's Alice Corp. v. CLS Bank (2014) decision and its progeny have created significant uncertainty about which software inventions are patentable

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: No, the Gutenberg Bible was not the first book printed with movable type

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: However, many products can temporarily make split ends look better by coating the hair with ingredients that smooth the cuticle, add weight to frayed ends create temporary bonds between split fibers, though these effects typically last only until the next shampoo

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Religious belief; cannot be answered objectively

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: Yes

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Yes, stalactites can form underwater, but not through dripping water — instead, they form when water flows along the external surface of an already-existing structure, depositing calcite or other minerals and thickening it over time

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Research has confirmed that stalactites can form ca

### Sample conflictingqa_a3980a2921cf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Research supports these practical guidelines: advanced dermatological studies show that oil penetration correlates with improvements in hair thickness and tensile strength different oils contain distinct vitamins that target specific concerns

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Additionally, some sources emphasize that the question remains under active scientific debate that no conclusive evidence has been established for any particular meteorite's cometary origin

### Sample conflictingqa_bd2e652cd64d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: None
- **Claim**: That said, manual toothbrushes are still better than none at all some sources note that built-in timers and pressure sensors in electric brushes are the most valuable features rather than the brush heads themselves

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Yes, Michael Jackson did compose songs for Sonic the Hedgeog 3; game creator Yuji Naka confirmed he wrote music for the 1994 Sonic the Hedgeog 3 soundtrack

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Overall, while coffee grounds may offer some benefit, they should be used in conjunction with other methods for the best results

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Religious and theological views differ; science cannot definitively confirm or deny historical existence

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Yes, a belief can be justified even if it is false

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: There is limited scientific evidence exploring the potential benefits of bee stings for arthritis what exists is largely outdated or inconclusive

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The most widely supported scientific view holds that humans and apes share a common ancestor who lived millions of years ago, though this is presented as a conclusion from Darwinian evolution rather than a proven fact accepted by all sources

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Yes

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Aryna Sabalenka and Amanda Anisimova were the US Open women's singles finalists last year (2025)

### Sample freshqa_049cc3f14d5e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This year's Passover (2026) began at sundown on Wednesday, April 1, 2026

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The first seder was held that evening the holiday ran through April 9, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Maryam Mirzakhani is the first and only female recipient of the Fields Medal to date

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The result ended Verstappen's dominant four-year reign at the top of Formula 1, as Norris became the 11th British driver to win the championship

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Android 16 is the latest stable version of Android, released on June 10, 2025

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: The official U.S. government energy website confirms that the Trinity test, which detonated a plutonium implosion device code-named 'Gadget' atop a 100-foot tower, occurred on July 16, 1945, releasing 18.6 kilotons of power

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Two million years ago

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: She delivered performances of "Lady Marmalade" and "One and Only" alongside Levine during the finale the in-studio audience voted her the champion

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The Houston Astros have won 2 World Series titles in their history

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Luka Modric

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Laika (dog), Sputnik 2, November 3, 1957

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Someone You Can Build a Nest In

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem holds the current Guinness World Record for the fastest rap in a hit single, as recognized by both Guinness World Records and the official OkayPlayer verification

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: Rosenblatt's death was widely mourned, with former Senator Eugene McCarthy and others paying tribute in Congress

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Minsky and Papert's 1969 book, *Perceptrons*, had already dealt a significant blow to the field Rosenblatt's passing further diminished interest in the perceptron for over a decade

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: Queen Elizabeth II of England died on 8 September 2022

### Sample freshqa_ab11b5dce00e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: The FIFA World Cup 2026 will be co-hosted by the United States, Mexico Canada

### Sample freshqa_b0ffe73c1789

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Colleen Hoover has written a total of 34 books

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: June 2025

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Zhejiang Province

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Saltwater crocodile (Crocodylus porosus) — 1.2 m (4 ft) thick

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: The latest stable macOS release is macOS 26 Tahoe, as of September 2025

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ef3ad40c6540

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1
- **Claim**: He signed with the Lakers in 2018 and has been a member of the team throughout their title runs in 2020 and 2023

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: Earlier counts reported 190 known geoglyphs before the 2022 Yamagata University find 672 after the 2024 AI expansion ; these older counts have since been superseded by the most recent data

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0031

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d7, d4, d5, d6, d1
- **Claim**: Pusha T

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d7, d4, d5, d2, d6, d8
- **Claim**: 506

### Sample qacc_0b3b6452e224

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5, d1, d4
- **Supporting Docs Found**: d2
- **Claim**: A low-quality Reddit comment incorrectly claims the statue was designed after an Egyptian woman, but this is contradicted by the high-credibility sources, which all confirm Bartholdi's role as the statue's sole designer

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 15

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: Oliver Stark

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The main characters of Paid in Full are Ace, Mitch Rico, played by Wood Harris, Mekhi Phifer Cam'ron respectively

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The show is currently in early development, with no writer or network/streamer attached, but if it moves forward, iconic roles like Ace, Mitch Rico are likely to be recast with actors such as Damson Idris, Algee Smith Joey BADA$$

### Sample qacc_252987b8054c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Tori Spelling

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Urochordata (tunicates) — Urochordata (also known as tunicates or sea squirts) are the earliest known vertebrates, with fossil evidence dating back approximately 570 million years ago

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: The other theory points to early Christianity: followers developed hand gestures to recognize each other in secret crossing fingers (or forming an 'L') was one such sign; over time, the gesture evolved from requiring a partner to being performed solo

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5
- **Claim**: As a coach, **Red Auerbach** holds the record with 11 championships, while as a player, **Bill Russell** leads with 11 rings

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Peyer's patches, while also lymphoid structures in the small intestine, are not the same as lacteals — Peyer's patches are lymphoid nodules containing immune cells, whereas lacteals are lymphatic vessels specifically dedicated to lipid absorption

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Canada first gained responsible government in the 1840s, gradually building autonomy over internal affairs then gained full international recognition as an autonomous community within the British Commonwealth in 1926, when the Balfour Declaration was issued

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, symbolically important legal traces of Canada's colonial status were not fully resolved until the passing of the Canada Act in 1982, which declared no British law would henceforth apply to Canada

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: McEwan

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 180

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: Multiple; there is no single number across sources

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: 5.88

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: For the most up-to-date information on the first McDonald's in Phoenix's current status, consult local sources or official McDonald's communications

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: The song was written by Russ Ballard; Argent recorded the original version Kiss also recorded a major cover version in 1991

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Uninterrupted human presence on the ISS began with Expedition 1 in October 2000 , making that the critical date for when the ISS went into space permanently

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Season 10 of El Señor de los Cielos premiered on 13 February 2024

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5, d1, d2
- **Claim**: This apparent contradiction reflects differing perspectives: some sources focus on the most visible milestones (like the completion of the Tower of Jesus), while others emphasize that the overall project has many more pieces still to be finished

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: This distribution holds true for the average adult male (about 60% body water), though the proportion is slightly lower in adult females (about 55%) due to their higher proportion of body fat it varies by age, sex adiposity

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: The basic governmental structure of the Ming was continued by the subsequent Qing dynasty and lasted until the imperial institution was abolished in 1911/12

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: 245

### Sample qacc_a78a32b7b9a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Joseph McCarthy is widely recognized as the most prominent figure who stoked the Red Scare in the 1950s, though not the sole originator

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: President Hoover and his staff were hosting the children in the East Wing at the time the party reportedly continued in another area of the house

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This joint is crucial for the transmission of sound vibrations in the middle ear

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The stapes, the third ossicle, articulates with the incus via a ball and socket joint

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Roger Miller

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Tavarez is a Hispanic surname derived from the Portuguese surname Tavares, with its roots tracing back to medieval Portugal

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Over 50

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: September 28, 1998

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The balance sheet (statement of financial position/statement of financial condition)

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: In Mexico, toll roads are called 'autopistas' or 'cuota' highways

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The last time Rangers were in the Champions League was the 2022–23 season, when they were drawn in Group A alongside Napoli and Liverpool

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: ICD-10 codes vary by edition and use, but the standard ICD-10-CM code is typically 6–7 characters long

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: 7

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: It depends on the jurisdiction; in Ontario (Canada), red licence plates are used by motor vehicle dealers (white background with red lettering) and diplomats/embassy staff (red background with white lettering) — see <d3>

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: In Spain, red license plates are for vehicles in circulation during registration processing, temporarily out of service used for research and tests — see <d2>

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: For Japan, a red stripe on a license plate typically indicates a rental car — see <d5>

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: September 1, 1935

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: A grizzly bear (Ursus arctos californicus), also known as the California brown bear or chaparral bear

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The retrieved evidence supports multiple interpretations of 'last time won.' The Calcutta Cup is the annual England-Scotland Six Nations match, so 'last time won' could refer to the most recent match between those two teams it could refer to the most recent time either team won the trophy in general

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: In the United States, environmental policy can be set at both the federal and state levels. The federal government, through agencies such as the Environmental Protection Agency (EPA), sets national standards and regulations to protect the environment, while individual states also have their own environmental agencies and can enact their own rules within their borders. This dual-level approach allows for a balanced framework where national goals are established while also accounting for local differences and circumstances

### Sample situatedqa_temp_05d714be23fd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, including the official iHeart content and the Yahoo Entertainment coverage of the 2026 iHeartRadio Music Awards

### Sample situatedqa_temp_14a587def215

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Earlier reports referenced the 2025 Stanley Cup Finals loss , but that result does not contradict the 2026 season being the most recent overall

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_32d33d503f69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: January 17, 2025 (United States)

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Jordan Ridgeway

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The official FIFA website confirms Argentina's victory over France in the final, giving them their first title since 1986

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Prior to that, the Avalanche's most recent Cup win was the 2001 Stanley Cup, also their third overall, which they claimed by sweeping the New Jersey Devils in the 2001 Stanley Cup Finals

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_61a79d74d827

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Düsseldorf

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Wrangell-St. Elias National Park was established in 1978, when President Jimmy Carter declared it a national monument; later, in 1980, its status was changed to a national park

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d1
- **Claim**: This makes it the largest national park in the United States

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Australia's mainland coastline is 35,821 km (22,292 miles), while the total Australian coastline (mainland and islands) is 59,681 km (37,069 miles)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Along its course, the river receives several major tributaries, including the Obey River, Caney Fork Stones River forms the Cumberland Falls near Baxter, Kentucky

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This victory came during the NBA bubble in Orlando, with LeBron James and Anthony Davis leading the team

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The Lakers' previous championship had come in 2010, making the 2020 run their first title in nearly a decade

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The population of Belgium on January 1st 2018 was approximately 11,552,967

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Shay Mitchell, the actress who played Emily Fields, was 23 years old when the show premiered in 2010, making her significantly older than her in-character age of 16

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: HMS Queen Elizabeth's commissioning was celebrated with a ceremony in Portsmouth attended by Her Majesty the Queen, Prince Charles senior military leaders

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d2
- **Claim**: The GPI is produced by the Institute for Economics and Peace (IEP), an independent think tank that measures peacefulness through three thematic domains: societal safety and security, ongoing domestic and international conflict militarization ; the 2018 report specifically covers data from the preceding year, meaning it reflects India's peacefulness in 2017

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: In the broader context, Gerard is a masculine Germanic forename that varies across languages and regions the surname Gerard is also found in Haiti, making it a name of wide international significance

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Anthony Joshua is the IBF champion (having defeated Daniel Dubois on July 19, 2025), but Usyk also holds the WBA, WBO IBO titles

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Riyad Mahrez won the PFA Player of the Year award in 2015–16, scoring 17 goals and 11 assists for Leicester City in the Premier League

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Hello, Love, Again

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Outside the U.S., McDonald's Monopoly has been run under different names and with different eligible items, such as 'Get Your Bag' (2025) and 'Coast To Coast' (2015–2024), reflecting regional variations in the promotion

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The Philadelphia 76ers last made the playoffs in the 2021–22 NBA season, when they reached the Eastern Conference Finals

### Sample trust_align_124

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Additionally, some cemeteries may also generate revenue from memorial contributions, service fees investment income to supplement their perpetual care funds

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The platform, originally created in March 2006 as Twitter by Jack Dorsey, Noah Glass, Biz Stone Evan Williams, underwent a significant corporate transition when it merged with X Holdings, ceasing to be an independent company and becoming a part of X Corp. This rebranding represents the latest chapter in Twitter's corporate history, superseding its previous identity as Twitter, Inc

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc., having been reorganized as a wholly owned subsidiary of Alphabet in 2015

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0082

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
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Australia are the latest Cricket World Cup champions, having won the 2023 edition in November 2023. Australia's victory in the final against India on 19 November 2023 confirmed their status as the most successful team in the history of the tournament, having won the title six times in total. The 2027 edition (14th edition) is also mentioned, but that is still future-oriented and does not supersede the 2023 result

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore is officially called Bengaluru. Bengaluru is the current official name of the city, while Bangalore was its official name until 1 November 2014, when the city government officially changed the name from Bangalore to Bengaluru. This change was confirmed by the Karnataka Government Order No. 114 dated 1 November 2014, which stated that henceforth the city would be known officially as Bengaluru

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: He is also the current record holder for the most men's singles titles at Wimbledon with 8 titles

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 Wimbledon Championships (the 139th edition) are scheduled to take place in 2026, but no results or champions have been announced for that future event

### Sample wikirevision_0150

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The latest US Open men's singles champion is Carlos Alcaraz, who won the 2025 US Open, the 145th edition of the tournament. He defeated Jannik Sinner in the final, claiming his second US Open title. The 2024 US Open (144th edition) was the previous edition, which Novak Djokovic won, but that title has since been superseded by Alcaraz's 2025 victory

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This victory gave Alcaraz his second French Open title and fifth major


================================================================================

*Report generated by CATS v2.0*
