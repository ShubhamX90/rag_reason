# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.966 (over 736 samples)

**GR F1** *(used in CATS)*: 0.979

**Behavior Adherence**: 0.809 (over 608 applicable samples)

**Factual Grounding**: 0.822 (over 608 applicable samples)

**Single-Truth Recall**: 0.738 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.837

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.979
- **Precision**: 1.000
- **Recall**: 0.959
- **Accuracy**: 0.966
- TP=583, FP=0, FN=25, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.837
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.911
- **Specificity**: 0.959
- Abstain TP=128, FP=25, FN=0, TN=583


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.935 (n=154)
- **Grounding**: 0.913 (n=154)
- **Recall**: 0.815 (n=154)
- **CATS**: 0.914

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.920 (n=176)
- **Grounding**: 0.766 (n=176)
- **Recall**: 0.676 (n=156)
- **CATS**: 0.834

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.954
- **GR F1** *(used in CATS)*: 0.973
- **Behavior**: 0.531 (n=96)
- **Grounding**: 0.823 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.776

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.738 (n=145)
- **Grounding**: 0.846 (n=145)
- **Recall**: 0.732 (n=140)
- **CATS**: 0.826

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.757 (n=37)
- **Grounding**: 0.613 (n=37)
- **Recall**: 0.703 (n=37)
- **CATS**: 0.746


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1843

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
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d4
- **Claim**: Partially — peeling reduces dietary fiber and some antioxidants (like vitamin E and folate), but does not reduce vitamin C or other nutrients found in the flesh

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Yes, palm oil production is generally considered bad for the environment due to deforestation, loss of biodiversity substantial greenhouse gas emissions

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Dog breeding is not universally considered unethical, but neither is it unambiguously ethical. The practice is ethically contentious depending on the context and standards followed — for example, d2 argues that responsible breeding can be acceptable if regulations are improved and red flags are recognized, while d4 contends that unethical breeding causes severe animal suffering and societal harm. Some opponents argue that breeding is inherently exploitative or contributes to overpopulation, while proponents maintain that careful, regulated breeding can preserve desirable traits and improve animal welfare

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d4, d5
- **Claim**: Most healthy children do not need multivitamins if they are growing normally and eating a varied, well-balanced diet

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2
- **Claim**: The ongoing scientific debate underscores the need for continued rigorous investigation, especially regarding dosage and vulnerable subpopulations, as cited by Harvard researchers and others

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Not really — chlorine doesn't turn hair green, but copper does copper is often present in pool water

### Sample conflictingqa_29f69e16a0c3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Yes, flowers do communicate with bees through multiple channels including sound, scent, color electric fields

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1
- **Claim**: However, the quantity of data needed is not universal: ML algorithms can perform well with small to medium-sized datasets when feature engineering is strong adding more data follows the law of diminishing returns where performance improvements decrease significantly as the dataset grows large

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Some sources further note that completely dataless ML is not possible , while others highlight that active learning and few-shot learning frameworks can approximate useful models with only a handful of examples , suggesting that the strict minimum requirement is some non-zero amount of relevant training data

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Overall, while saturated fats may affect cholesterol levels, the evidence on whether they directly cause heart disease in the general population is mixed and contested

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Multiculturalism's impact on unity is contested in the evidence: some sources argue it undermines civic cohesion by amplifying cultural differences , while others argue it facilitates unity by promoting citizenship integration and political participation among immigrants

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5, d2
- **Claim**: In essence, the two terms reflect the same underground exploration practice, with 'caving' becoming the preferred modern terminology

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: Functional and rehabilitative braces, meanwhile, are generally considered effective for supporting healing and managing existing conditions like osteoarthritis or ligament injuries , but their role in preventing new injuries in healthy individuals is less established

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d1, d2
- **Claim**: Gonorrhea is almost always transmitted through sexual contact, but rare non-sexual transmission routes exist, such as mother-to-baby during childbirth or through contaminated objects

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3, d1
- **Supporting Docs Found**: None
- **Claim**: While some researchers attribute this decrease to reduced body size and changes in lifestyle following the Neolithic transition , others propose that the shift from brute-force information processing to more metabolically efficient symbolic thinking made larger brains unnecessary

### Sample conflictingqa_be17259fe5c0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4, d2
- **Supporting Docs Found**: d3, d1
- **Claim**: While penguins are primarily found in the Southern Hemisphere and some species do live on the Antarctic continent , the most peer-reviewed evidence points to Australo-New Zealand origins rather than Antarctic ones

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Religious and theological views differ; science offers no settled answer

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Religious views differ; scholarship defines biblical infallibility as making no false statements regarding faith and practice

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d4
- **Claim**: The retrieved evidence is conflicting. Some sources claim that a full moon can create werewolves or trigger transformations, while others argue that this is a cinematic myth lacking historical basis

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The concept is further illustrated by examples where nondeductive evidence justifies a conclusion without entailing its truth by the broader debate between coherentists and foundationalists about the nature of justified belief

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes — barefoot running is healthier

### Sample conflictingqa_f3ba8599e370

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Jules

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Emoji are widely used as a supplement to written language, but whether they constitute a distinct form of written language is subject to debate

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence indicates that the Dutch were among the first Europeans to explore and map Australia, with Willem Janszoon's 1606 voyage to Cape York Peninsula representing one of the earliest recorded European contacts. However, the documents do not explicitly confirm that the Dutch 'discovered' Australia in the exclusive sense, as later British expeditions and the question of prior Indigenous occupation complicate that claim

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Additionally, yerba mate contains polycyclic aromatic hydrocarbons (PAHs), which are recognized as carcinogens smoking in conjunction with yerba mate use has been shown to significantly heighten cancer risk

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, laboratory research has also demonstrated that yerba mate possesses anti-cancer properties—capable of inducing cell death in certain cancer models—and some sources note that the evidence for cancer risk in humans remains inconclusive and requires further study

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1
- **Claim**: Once considered the same dinosaur, Apatosaurus and Brontosaurus were reclassified as distinct genera in a 2015 study

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d2
- **Claim**: Religious scholars and commentators differ on whether to classify Mormons as Christians; the answer depends significantly on how one defines Christianity

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: Aryna Sabalenka and Amanda Anisimova

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: This year's Passover (Pesach) began on April 2, 2026

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable version of Android is Android 16, released in June 2025

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The latest major version of .NET is 10.0, released as .NET 10.0.8 on May 12, 2026

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The site is now part of the White Sands Missile Range and is owned by the U.S. Department of Defense, with ground zero marked by a black lava rock obelisk

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Nuclear reactions between lead and bismuth (or mercury) can theoretically produce gold, though the process is highly energy-intensive and unfeasible for practical or commercial use

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 2

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The first animal to circle the Moon was Zond 5, which carried two Russian tortoises in September 1968

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Bayonne, New Jersey

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

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: 9 September 2022

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The heaviest reptile in the world is the saltwater crocodile (Crocodylus porosus), which can grow to over 7 meters (23 feet) in length and weigh well over 1,000 kilograms

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The Bills vs. Bengals game on January 2, 2023 suspended for approximately 82 minutes after Damar Hamlin suffered cardiac arrest on the field

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: Slugs do not have lungs in the same sense as higher vertebrates — rather than a discrete organ, they possess a lung-like cavity derived from the mantle that communicates with the outside through the pneumostome

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: February 19, 2026

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10
- **Supporting Docs Found**: d5, d7
- **Claim**: World War II

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d7
- **Claim**: 1988

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6
- **Supporting Docs Found**: d3
- **Claim**: Stuart

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

### Sample qacc_0b3b6452e224

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d2
- **Claim**: The Statue of Liberty was designed by French sculptor Frédéric Auguste Bartholdi the statue's face is said to have been modeled after his mother

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: Following the North Africa campaign, the Allies moved eastward into Tunisia and eventually into Europe via Italy

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: India won the Cricket World Cup in 1983, 2007, 2024 2026

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Reviews for the Toronto production have been overwhelmingly positive, with the show described as 'breathtaking' and 'bigger and better than ever before'

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Canada did not gain independence from Great Britain on a single date, as the process was gradual. The two most cited milestones are the Statute of Westminster in 1931, which gave Canada legislative independence the Canada Act of 1982, which fully severed remaining constitutional ties — including the monarchy's formal role in amending laws

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d5
- **Claim**: Eukaryotic chromosomes initiate replication at multiple origins rather than a single origin

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: October 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d1
- **Claim**: Nana in Snow Dogs is a collie

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9a9a28d7e159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: This season completed the anime adaptation of Hiro Mashima's original manga, which had concluded earlier in 2017

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a1c73439eca9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: The most recent available evidence indicates that the new (tenth) season of El Señor de los Cielos began production in early 2024, though no specific air date has been publicly confirmed yet

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Most effigy mounds were built between 750 and 1050 AD, though construction spanned a broader period from approximately 700 to 1200 AD

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: July 4, 1776

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The aircraft was named after Enola Gay Tibbets, the mother of its pilot, Colonel Paul Tibbets

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Autopistas are distinguished from free routes via the suffix 'D' (for Directo), meaning that a road numbered, say, Fed

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: 45D is the tolled version of Fed

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2025–26

### Sample situatedqa_geo_082db791e263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d4, d5
- **Claim**: It depends on jurisdiction; federal US law generally requires 18 to buy a shotgun, though some states and other countries like the UK have higher minimum ages

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: For a broader Allied (WE) total, the data becomes more aggregated: the Soviet Union suffered between 8.8 and 10.7 million military deaths alone, while Poland's dead ranged from 5.6 to 5.8 million , underscoring the scale of loss across the entire conflict

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: The welfare state was introduced at different times across nations, with Germany pioneering social insurance in the 1880s, Britain's Liberal government enacting major reforms in 1906–1914 the United States formally establishing its framework through the 1935 Social Security Act

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d1
- **Claim**: This makes 1935 the most commonly cited year for Social Security's inception, though the program developed in stages throughout the late 1930s

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: For example, California has the highest state gasoline tax at 61 cents per gallon, bringing the total there to roughly 79.4 cents per gallon, while Alaska's total tax is among the lowest at approximately 27.4 cents per gallon

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Mexico

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a California grizzly bear (Ursus arctos californicus), which is the official state animal of California

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: In modern times, the switch is often motivated by individual health considerations, such as seeking lower caffeine content or more antioxidants, rather than any unified historical milestone

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: The FOMC meets approximately every six weeks to discuss the economic outlook and make decisions on key tools such as interest rates and the money supply, with the goal of promoting maximum employment, stable prices moderate long-term interest rates

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: 1972

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Darren Criss

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 1939

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Five sharps in a key signature signify the key of B major or its relative minor, G# minor. The sequence of sharps always follows the order F–C–G–D–A–E–B, so the presence of five sharps means all notes F, C, G, D A are raised by a half step throughout the piece. This key signature is confirmed across multiple sources, including the circle of fifths and key signature calculation methods

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: shepherd of evil

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d1
- **Claim**: It is caused by a deficiency of the hexosaminidase A (HEX A) enzyme, which is necessary for breaking down GM2-ganglioside within cells; the disease occurs when harmful quantities of gangliosides accumulate in the nerve cells of the brain, leading to progressive neurological damage

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that appear in the blood when the heart is stressed or damaged they are used to diagnose and assess the severity of heart conditions

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: None
- **Claim**: Some sources, including the Ancient History Encyclopedia, previously cited 1275 BCE as the date , but this is now considered less accurate than the 1274 BCE consensus derived from Egyptian royal records

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 133

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Hello, Love, Again

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2026

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held in Ahmedabad on 19 November 2023. This was Australia's sixth Cricket World Cup title the tournament was the 13th edition organized by the ICC. The 2027 Cricket World Cup is next scheduled for 2027, but no later champion than 2023 has been established

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022. He is the 31st person to hold the office since its creation in 1901

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The current President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022. He is the 17th President of the Philippines and serves as both head of state and head of government. This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of the Philippines page, as well as the list of presidents of the Philippines


================================================================================

*Report generated by CATS v2.0*
