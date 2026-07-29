# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 126 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.957 (over 736 samples)

**GR F1** *(used in CATS)*: 0.973

**Behavior Adherence**: 0.818 (over 610 applicable samples)

**Factual Grounding**: 0.843 (over 610 applicable samples)

**Single-Truth Recall**: 0.736 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.842

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.973
- **Precision**: 0.997
- **Recall**: 0.951
- **Accuracy**: 0.957
- TP=578, FP=2, FN=30, TN=126

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.808
- **Abstain Recall**: 0.984
- **Abstain F1**: 0.887
- **Specificity**: 0.951
- Abstain TP=126, FP=30, FN=2, TN=578


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.994
- **Behavior**: 0.961 (n=155)
- **Grounding**: 0.920 (n=155)
- **Recall**: 0.808 (n=154)
- **CATS**: 0.921

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.923
- **GR F1** *(used in CATS)*: 0.950
- **Behavior**: 0.921 (n=177)
- **Grounding**: 0.784 (n=177)
- **Recall**: 0.657 (n=156)
- **CATS**: 0.828

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.604 (n=96)
- **Grounding**: 0.840 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.810

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.690 (n=145)
- **Grounding**: 0.898 (n=145)
- **Recall**: 0.768 (n=140)
- **CATS**: 0.837

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.784 (n=37)
- **Grounding**: 0.586 (n=37)
- **Recall**: 0.649 (n=37)
- **CATS**: 0.724


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1857

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
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Yes — many salamanders secrete poisonous toxins from their skin that can cause serious illness if ingested after handling the toxin's effects can range from numbness to paralysis; however, a few species such as tiger salamanders and yellow-spotted salamanders are considered safe to handle gently experts generally recommend admiring salamanders from a distance and always washing hands after handling any species

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2
- **Claim**: The evidence presents conflicting findings. Some sources say peeling removes much of the fiber and antioxidants, while others say it does not reduce vitamin levels and that eating a peeled apple is still healthier than not eating one at all

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Cows technically have one stomach that is divided into four distinct compartments — the rumen, reticulum, omasum abomasum — rather than four separate stomachs

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Not really — chlorine doesn't turn hair green, but copper in pool water can many sources falsely attribute the effect to chlorine

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: d3
- **Claim**: The consensus leans toward equivalence—particularly when reading for pleasure rather than studying—and several libraries and major literary figures, including Neil Gaiman, formally recognize audiobooks as reading , but the debate continues, with usage and definition remaining open to individual interpretation

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Continuous slow cooling and associated tectonic movement are the primary drivers of this activity, causing the Moon's diameter to shrink by approximately 5.5 cm per century

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: It depends on whether the artificial tree is used for at least 20 years; otherwise, real trees are more sustainable

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: On balance, the evidence is mixed. Some conservation scientists and the IUCN argue that well-managed trophy hunting can raise significant revenue, provide incentives to restore and protect wildlife habitat fund anti-poaching efforts — particularly when alternative revenue streams are insufficient. Others argue that trophy hunting is inherently unethical, that revenue benefits are often overstated or captured by outfitters rather than local communities that a complete ban would not necessarily eliminate conservation funding if替代方案被探索和证明可行。

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are valuable for protecting core algorithms and providing exclusionary rights, while others argue that patents are controversial, with many inventions falling outside patent eligibility and the patentability standard for software being legally uncertain

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: However, experts note that the healing is gradual—fully repairing the ozone layer could take another 30–50 years new threats such as rocket launches are slowing the recovery process

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: It depends on jurisdiction; in the U.S. federal law generally allows ISPs to sell anonymized browsing data without consent, but some states like Maine and California require opt-in or opt-out consent other states are considering similar protections

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Bees generally avoid flying in the rain because wet wings reduce lift and maneuverability, but they are capable of flying in light rain and will do so in emergencies such as defending the hive or finding emergency food

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d2
- **Claim**: Heavy raindrops can severely damage bee wings and disrupt flight, making light rain the threshold above which bees are mostly unwilling to forage ; d2 similarly notes bees cannot manage in heavy rain, while d1 and d3 confirm bees may venture out in light rain or when driven by colony needs

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Multiculturalism's impact on unity is contested in the evidence: some sources argue it undermines civic cohesion and social integration , while others argue it facilitates political inclusion and democratic participation

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The weight of the evidence leans toward the latter view — that multiculturalism is not a hindrance to unity and may even strengthen it — though the debate remains unresolved

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Yes — the scientific consensus, based on multiple peer-reviewed studies, is that fish do possess pain receptors and respond to noxious stimuli in ways similar to mammals

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Overall, while volcanic activity clearly contributed to the event, the degree of volcanic responsibility relative to other factors remains an active question of scientific research

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Current CO2 levels are not unprecedented in Earth's history as CO2 has fluctuated within natural bounds for hundreds of millions of years

### Sample conflictingqa_be17259fe5c0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Yes — molecular phylogenetics and Bayesian inference place the most recent common ancestor of modern penguins in Antarctica ~71 million years ago; genetic evidence from 18 species corroborates this Antarctic origin before penguins spread poleward

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: Some sources further note that paper straws may be less environmentally friendly due to production costs and that the ideal alternative is often a reusable, non-plastic option used sparingly

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Traditionally, plants require sunlight for photosynthesis, but some research suggests they could grow using electricity instead

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Research presents conflicting findings. Some studies report that barefoot running reduces the risk of certain injuries—such as knee and hip pain—by encouraging a forefoot strike, while also potentially improving running efficiency and strengthening intrinsic foot muscles. However, other research, including a University of Queensland study, found that running shoes cause foot muscles to work harder than barefoot running, which may contribute to overall leg stiffness and balance. Additionally, barefoot running carries its own risks, such as stress fractures from sudden intensity increases and exposure to road debris there is currently no clear consensus among experts that barefoot running is categorically healthier for all runners

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Emoji serve as a visual supplement to written language but do not constitute a distinct written language themselves

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence indicates that the Dutch were among the earliest European discoverers of Australia, with Willem Janszoon's 1606 voyage to the Cape York Peninsula representing one of the first recorded European encounters with the continent. However, the evidence does not confirm that the Dutch were the absolute first discoverers of Australia, as prior unrecorded visits or other navigational histories are not definitively ruled out the document title itself asks 'Who Really Discovered Australia?' without providing a definitive answer

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The official military explanation attributes the Phoenix Lights to military flares dropped by A-10 aircraft during a training exercise, though many witnesses and researchers reject this claim, citing temporal inconsistencies, the object's solid appearance complete silence — characteristics they argue flares cannot replicate. The debate remains unresolved, with some sources suggesting the event may have involved a classified military craft or covert exercise, while others maintain it represents one of the most credible UFO incidents in recent history

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Maryam Mirzakhani was the first woman to receive the Fields Medal, awarded in 2014, but she is not the only female recipient. Maryna Viazovska became the second female Fields Medalist at the 2022 International Congress of Mathematicians. This makes the query's premise obsolete: there are now two female Fields Medalists, not one

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Beijing is the first city in the world to have hosted both the Summer and Winter Olympics. It hosted the 2008 Summer Olympics and then the 2022 Winter Olympics, making it the first city to hold both games. This distinction is confirmed by multiple authoritative sources, including Wikipedia and NPR, which describe Beijing as the first city in Olympic history to host both events

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This figure is consistent across multiple sources, including her Forbes profile and Goodreads page, though some older sources may cite different numbers due to the continuous addition of new publications

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Jeff Bezos did not sell Amazon. He founded the company in 1994 and continued to lead it as CEO until 2021, when he transitioned to the role of Executive Chairman. In 2025, he did sell millions of shares worth over $736 million, but remained the largest shareholder and chairman of the company

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: February 19, 2026; March 19, 2026

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Nomura Koch

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The Allies subsequently moved eastward across North Africa before entering Europe via Italy, fighting a major campaign there from 1943 to 1945

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d4, d5, d1
- **Claim**: These distinct appointments reflect the campaign's structure, where each state has selected its own ambassadors to promote the cause locally

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

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: October 1, 1968

### Sample qacc_8882ab46be5d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: 35

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The planned completion of the Sagrada Família has been updated to the early 2030s, superseding the earlier 2026 target

### Sample qacc_a6b48b7accc4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane plays Carter Pewterschmidt, Lois's father, in Family Guy

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Most effigy mounds were built between approximately 700 and 1200 CE, with the majority constructed within the Late Woodland period (roughly 750–1050 CE)

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The aircraft was named after Enola Gay Tibbets, the mother of the pilot, Colonel Paul Tibbets

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

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: It depends on location; in the UK, those aged 16 can drink wine, beer cider with a meal at a licensed premises, while in the U.S. the minimum legal drinking age is 21 years old

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

### Sample situatedqa_geo_7222d6123c27

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Calcutta became the capital of British India in 1772, when Warren Hastings transferred all important offices there

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3
- **Claim**: Historically, the bulk of U.S. immigrants came from Europe, but this shifted to Latin America and Asia since 1965

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features the California grizzly bear (Ursus arctos californicus), making it the most directly relevant answer to your query

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Kiren Rijiju, who serves as the Cabinet Minister for Law & Parliamentary Affairs. He is also the Minister of the Ministry of Law and Justice, making him the top law official in the country. This is confirmed by the official Punjab government website, which lists him as the Minister for Law & Parliamentary Affairs

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5
- **Claim**: This document established a loose confederation of states with a weak central government, creating a 'league of friendship' in which state powers were largely preserved and there was no separate executive or judicial branch

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The switch was complete by around 1900, when coffee consumption far exceeded that of tea across all American demographics

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The FOMC meets regularly to decide on key monetary tools such as interest rates and open market operations, with its decisions published in minutes and meeting records

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: 112

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: It was first made available to Google Pixel phones and has since rolled out to other manufacturers including Samsung

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
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3, d2
- **Supporting Docs Found**: None
- **Claim**: This relationship is consistent across all sources detailing key signatures and their construction

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: From the battle with Grendel in Beowulf, Grendel is described using kennings such as "twilight-spoiler" and "shepherd of evil," while the narrator also uses "battle-sweat" as a kenning for blood in the context of their fight

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: The most recent nominal GDP figure for the United States is **$31.819 trillion** (seasonally adjusted at an annual rate), reported by Moody's Analytics for Q1 2026. This is corroborated by YCharts, which reports the current level at **31.82 trillion** as of March 2026, representing a 1.26% increase from the prior quarter. Earlier data from Wikipedia and the Bureau of Economic Analysis show a slightly lower figure of **$30.762 trillion** for the full calendar year 2025, which is superseded by the more recent 2026 data

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d4
- **Claim**: The mainland coastline alone measures about 35,821 km (roughly 22,256 miles), making it one of the longest in the world

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_9a91e43d8136

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
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin and Perry Go Large

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

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Royal Challengers Bengaluru (RCB) is the current Indian Premier League (IPL) champion, having won their first title in the 2026 IPL season. This supersedes older information from the 2026 Wikipedia revision that listed RCB as the 2025 champion, which was outdated. The 2026 IPL season was the 19th edition of the tournament and featured 10 teams competing across 74 matches

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4
- **Claim**: Google is owned by Alphabet Inc., which serves as its parent company. Alphabet is a publicly traded company listed on Nasdaq under the ticker symbols GOOGL (Class A shares) and GOOG (Class C shares), making Google indirectly owned by Alphabet's shareholders. Within Alphabet, Google's founders Larry Page and Sergey Brin together hold approximately 14% of publicly listed shares and control about 56% of the company's stockholder voting power through super-voting stock, giving them significant ownership influence. This dual-class share structure means that while Alphabet's Class A and Class C shares are held by millions of investors, Page and Brin retain effective control over the company

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held on 19 November at the Narendra Modi Stadium in Ahmedabad, India. This was Australia's sixth Cricket World Cup title the 2023 tournament was the 13th edition of the ICC Men's Cricket World Cup. Wikipedia's up-to-date revision confirms Australia as the 2023 champion, superseding an older revision that had incorrectly listed India as the latest winner

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was confirmed in April 2016, when the Haryana Government officially renamed the city. As a result, Gurgaon is now known officially as Gurugram, though the older name is still commonly used

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This name change was confirmed by the Government of Haryana in 2016 the city is now officially known by this new name. As a result, Gurgaon is no longer the official name of the city


================================================================================

*Report generated by CATS v2.0*
