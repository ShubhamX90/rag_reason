# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.962 (over 736 samples)

**GR F1** *(used in CATS)*: 0.976

**Behavior Adherence**: 0.804 (over 608 applicable samples)

**Factual Grounding**: 0.814 (over 608 applicable samples)

**Single-Truth Recall**: 0.758 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.838

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.976
- **Precision**: 1.000
- **Recall**: 0.954
- **Accuracy**: 0.962
- TP=580, FP=0, FN=28, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.821
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.901
- **Specificity**: 0.954
- Abstain TP=128, FP=28, FN=0, TN=580


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.974 (n=154)
- **Grounding**: 0.900 (n=154)
- **Recall**: 0.860 (n=154)
- **CATS**: 0.930

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.920 (n=176)
- **Grounding**: 0.766 (n=176)
- **Recall**: 0.708 (n=156)
- **CATS**: 0.842

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.954
- **GR F1** *(used in CATS)*: 0.973
- **Behavior**: 0.521 (n=96)
- **Grounding**: 0.811 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.768

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.683 (n=145)
- **Grounding**: 0.832 (n=145)
- **Recall**: 0.736 (n=140)
- **CATS**: 0.811

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.757 (n=37)
- **Grounding**: 0.613 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.717


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1845

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
- **Claim**: Yes — many salamanders are poisonous to touch, as nearly all species possess toxins in their skin that can cause serious illness if absorbed through the hands or transferred to the mouth or eyes. The most dangerous toxin is produced by the fire salamander, which can cause numbness, dizziness, muscle weakness life-threatening paralysis; Japanese researchers documented a fatal case involving a 4-year-old boy who handled a fire salamander without washing his hands before eating. However, some sources contradict this broad finding: a peer-reviewed study notes that most toxic reactions require ingestion, not mere handling one source states that tiger salamanders are not poisonous to humans. In practice, whether salamander toxins constitute a meaningful touch hazard depends on handwashing habits and the species involved — but the consensus across multiple sources is that handling salamanders carries real risk and should be avoided whenever possible

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: As a result, fashion designers seeking broader protection often turn to alternative intellectual property tools, such as trademark law for logos and brand identifiers the sui generis design rights available in jurisdictions like the European Union

### Sample conflictingqa_08757808c000

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: Not directly — weight lifting causes temporary blood pressure spikes during exertion, especially with heavy loads, but research consistently shows that regular strength training improves long-term blood pressure measures

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3, d4
- **Claim**: Yes, peeling an apple removes a significant portion of its nutritional value, particularly fiber and antioxidants, although it does not reduce vitamin C content

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The answer depends on how much weight one gives to innate talent versus acquired capability

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Dog breeding is not universally unethical, but some practices are

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Cows technically have one stomach that is divided into four distinct compartments — the rumen, reticulum, omasum abomasum — rather than four separate stomachs

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d3, d4
- **Claim**: Most healthy children do not need multivitamins if they are growing normally and eating a varied, well-balanced diet

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Not directly from chlorine — chlorine actually lightens hair, but copper (from algaecides or tap water) oxidizes and binds to hair proteins, causing the green color

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: Yes — machine learning always requires some data to train models, but the volume depends heavily on the algorithm, problem complexity data quality

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: However, other researchers argue that the Moon's magnetic field — once driven by a liquid outer core — crystallized and disappeared between 2.5 and 1 billion years ago, effectively ending most volcanic activity around 3 billion years ago that any observed features reflect ancient processes rather than current activity

### Sample conflictingqa_3c835387fe6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Multiple lines of scientific research, including a study from The Australian National University, confirm this evolutionary origin story, noting that the Komodo dragon interbred with an Australian lizard ancestor before crossing over to Indonesia

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: Yes — the Great Pacific Garbage Patch is roughly twice the size of Texas

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are valuable for protecting investments and encouraging innovation, while others argue that software is too abstract or that patents stifle competition and innovation

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3
- **Claim**: A Nature article also noted that geologic strain from tides during full and new moons could increase tremor magnitude , but the overall scientific consensus remains unsettled, with methods and magnitude thresholds playing a significant role in differing research outcomes

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: On the other hand, other sources contend that certain bond-building products — such as those containing keratin or amino acids — can help repair the disulfide bonds within the hair shaft, offering a more lasting solution than conventional conditioners ; additionally, some sources suggest that humid environments may also contribute to temporary repair by facilitating bond reformation

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Ultimately, the degree of repair depends on the severity of the split ends and the specific method used, with regular trims and humidity being among the more universally recommended approaches

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3
- **Claim**: As a result, bees generally prefer to remain in the hive during inclement weather, returning to forage only when the rain subsides or when the colony faces an urgent need for resources

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Uncertainty — some researchers conclude yes, others conclude no

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1, d3
- **Supporting Docs Found**: d2
- **Claim**: If kept legally, they require careful monitoring of temperature, humidity diet can live 5–7 years, meaning owners must be prepared for a long-term commitment

### Sample conflictingqa_a3980a2921cf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Not all hair types and concerns benefit from hair oil in the same way; some people with fine hair may find heavy oils weigh their hair down or cause greasiness, while others with dry or damaged hair may notice significant improvement

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: This reduction has been attributed to a shift towards more metabolically efficient, symbol-based information processing as humans developed language and complex societies, meaning that smaller brains could perform increasingly sophisticated cognitive tasks without the need for the larger brains of earlier hominins

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: The evidence is further complicated by the fact that some single-use plastic items, including straws, may be among the most environmentally efficient packaging options when evaluated across all lifecycle stages , while single-use paper straws are estimated to account for 43% of total life-cycle emissions compared to 39% for plastic

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Experts and official bodies such as the UN Environment Programme emphasize that the most sustainable choice is to reduce straw use altogether or opt for reusable options

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: While most sources agree on these benefits, some nuance applies: unfortified nutritional yeast contains only modest B-vitamins not all products may carry the same complete-profile guarantee , so always check the label or opt for fortified versions to maximize nutritional coverage

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: A Reddit user similarly stated that coffee grounds are unlikely to affect slugs or snails, though they may benefit the garden as a fertilizer

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Not really — plants can survive in darkness or low light for a while, but they cannot grow properly without any light

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Religious and theological views differ; science offers no settled answer

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Religious authorities and theologians hold differing views on the Bible's infallibility; there is no single unified answer accepted by all traditions

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d3, d4, d2
- **Claim**: In folklore and pop culture, full moons are frequently associated with werewolf transformations, but they do not typically create werewolves

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: There is no definitive scientific proof that bee stings treat arthritis; most evidence is anecdotal or preliminary

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: Heads differ on whether barefoot running is healthier; the evidence is mixed and the answer depends on what you mean by 'healthier.'

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The retrieved evidence indicates that the Dutch were among the earliest European discoverers of Australia, with Willem Janszoon's 1606 voyage to Cape York Peninsula being a particularly significant early contact. However, the evidence is partial and does not conclusively establish that the Dutch 'discovered' Australia in the exclusive sense, as later Dutch explorers charted portions of the coastline without fully recognizing the continent's extent competing claims exist regarding who first discovered Australia

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d5
- **Claim**: Once considered the same dinosaur, Apatosaurus and Brontosaurus were reclassified as distinct genera in a 2015 study

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: Aryna Sabalenka and Amanda Anisimova

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: This year's Passover (Pesach) began on April 2, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable version of Android is Android 16, released in December 2025

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The site is now part of the White Sands Missile Range and is owned by the U.S. Department of Defense, with ground zero marked by a black lava rock obelisk

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The first animal to circle the Moon was Zond 5, which carried two Russian tortoises in September 1968

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: No animal has ever landed on the Moon, as no mission carrying a lander with animals has been recorded

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
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

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: No, the Raptors do not have a winning record in the latest NBA season. The Britannica record shows a 25–57 mark in 2023–24, which is a losing record, though the text also notes that the Raptors made the playoffs in that season

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 9 September 2022

### Sample freshqa_b0ffe73c1789

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: 34

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The heaviest reptile in the world is the saltwater crocodile (Crocodylus porosus), which can grow to over 7 meters (23 feet) in length and weigh well over 1,000 kilograms (2,200 pounds). This is confirmed by Britannica, which notes that both reticulated pythons and saltwater crocodiles are among the largest living reptiles, though the saltwater crocodile is generally considered the heaviest. While the Komodo dragon (Varanus komodoensis) is the largest living lizard, it is significantly smaller than the saltwater crocodile, growing to only about 3 meters (10 feet) in length

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: 13

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: A permanent cure for cancer has not been developed. Most cancer treatments only achieve remission the search for a universal cure remains an active area of research

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: A total of 893 Nazca geoglyphs have been discovered, comprising 430 previously known examples plus 248 additional geoglyphs found through AI-supported field surveys conducted in 2023 and 2024. These figures reflect the most recent confirmed count, as earlier reports citing approximately 358 or 430 geoglyphs have been superseded by the latest Japanese-led research

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: February 19, 2026

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10
- **Supporting Docs Found**: d7, d5
- **Claim**: World War II

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d7
- **Claim**: 1988

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
- **Supporting Docs Found**: d4, d2
- **Claim**: Following the North Africa campaign, Allied forces proceeded to Italy, with the most notable subsequent theater being the invasion of Sicily in July 1943, which served as a stepping stone to the mainland

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3
- **Claim**: Additionally, the film's plot and production are further tied to these men, as Azie Faison himself served as a producer the characters' real-life counterparts were directly referenced in promotional materials and adaptations

### Sample qacc_2243f17ccc38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: This practice was later adopted by early Christians, who used a modified two-finger cross (thumb and index finger touching to form an 'L') as a secret recognition symbol during periods of persecution the gesture eventually evolved into the modern one-handed practice

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Canada did not gain independence from Great Britain on a single date, as the transition was an evolutionary process spanning the early 20th century. The country first became a self-governing dominion under the 1867 Constitution, gaining legislative autonomy while retaining formal ties to Britain. Further milestones included the Balfour Declaration of 1926, which recognized Canada as an autonomous community within the British Empire the Statute of Westminster in 1931, which granted full legislative independence. The final legal vestiges of colonialism were removed in 1982 with the Canada Act, which gave Canada full constitutional sovereignty

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: October 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d5
- **Claim**: Nana in Snow Dogs is a collie

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: The song was first performed and recorded by Argent, the British rock band founded by Russ Ballard, who also wrote the song. It later became a cover hit in 1991 when American hard rock band Kiss recorded 'God Gave Rock 'n' Roll to You II', which featured Ace Frehley on lead vocals. Both versions are distinct performances of the same composition

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: For the 1973 Disney Animated Classic: George Bruns composed the score, Roger Miller performed the songs Floyd Huddleston contributed additional music (including for the song 'Love')

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: July 4, 1776

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The Enola Gay was named after Enola Gay Tibbets, the mother of its pilot, Colonel Paul Tibbets

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_e064a7a717ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Federal highways designated as toll roads typically carry the letter suffix 'D' (for Directo), meaning that Fed

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: 45D, for example, is a tolled version of the free Fed

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2025–26

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 407,000

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_66684169f016

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d3, d2
- **Claim**: The Dandi March was led by Mahatma Gandhi and involved thousands of participants, including notable figures such as Mithuben Petit and Pyare Lal Nayar, as well as seventy-nine Ashramites who accompanied Gandhi from Sabarmati to Dandi

### Sample situatedqa_geo_779fd84224fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Sydney Cove (also referred to as Port Jackson or Botany Bay)

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Since 1965, the bulk of immigrants coming to the United States have come from Latin America and Asia, with about half from Latin America and a quarter from Asia

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: This 1970 version superseded earlier federal air pollution laws passed in 1955 and 1963 established the modern framework of federal air quality regulation under the newly created Environmental Protection Agency

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a California grizzly bear (Ursus arctos californicus), which is the official state animal of California. The grizzly bear was placed on the flag during the 1846 Bear Flag Revolt, when American settlers captured Sonoma and raised a flag with a bear as a symbol of strength and resistance. It is worth noting that the California grizzly bear is now extinct, making California the only state to carry the image of an extinct animal on its state flag

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

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d5
- **Claim**: This document established a loose confederation of states with a weak central government, creating a 'league of friendship' in which state powers were largely preserved

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This broad trajectory is further corroborated by data showing that by 2025, approximately 75% of American adults drink coffee daily, while tea drinkers remain a minority

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 1939

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: The latest stable version of Android is Android 16, which was released on June 10, 2025. This supersedes earlier reports that had identified Android 15 as the most recent version, making Android 16 the current standard

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Wrangell–St. Elias National Park was established in 1980, designated as a national park under the Alaska National Interest Lands Conservation Act (ANILCA) signed by President Jimmy Carter on December 2, 1980. This makes it the largest national park in the United States, covering approximately 13.2 million acres in Southcentral Alaska

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Five sharps in a key signature signify the key of B major (or its relative minor, G# minor). Using the standard order of sharps—F♯, C♯, G♯, D♯, A♯—the presence of all five indicates the tonic is B, with the key signature serving as a shorthand for the B major scale. This relationship is confirmed across multiple sources, including the circle of fifths, which places B major at the point where five sharps accumulate

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

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d1, d3
- **Claim**: It is caused by a deficiency of the hexosaminidase A (HEX A) enzyme, which is necessary for breaking down GM2-ganglioside within cells of the body, particularly in the brain and nerve cells

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: California's total gas tax is approximately 70 cents per gallon, making it the highest in the country. This figure has grown from approximately 90 cents per gallon reported earlier in 2026, reflecting recent changes in tax policy and reporting timelines. The official California Department of Tax and Fee Administration data further breaks down the current tax structure, listing the excise tax at $0.612 per gallon for the period beginning July 2025

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that appear in the blood when the heart is stressed or damaged they are used to diagnose and monitor heart disease

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The Battle of Kadesh reportedly began on May 1274 BCE (Year 5 III Shemu day 9 of Ramesses II), though sources differ on the year, with one citing 1275 BCE; no source provides a specific end date for the battle

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

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Hello, Love, Again

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc., a parent company it spun off into in 2015. Alphabet is a public company traded on the Nasdaq under the symbols GOOGL (Class A share) and GOOG (Class C share). Google was founded in 1998 by Larry Page and Sergey Brin, who together still own about 14% of Alphabet's publicly listed shares and control 56% of its stockholder voting power

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held in Ahmedabad on 19 November 2023. This was Australia's sixth Cricket World Cup title the tournament was the 13th edition organized by the ICC, held entirely in India from 5 October to 19 November 2023. Multiple sources confirm that Australia's 2023 victory is the most recent championship, with the next tournament scheduled for 2027 in South Africa, Zimbabwe Namibia

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was confirmed when the city officially changed its name from Gurgaon to Gurugram the change is consistently reflected across all sources

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025. He is the 24th person to serve in the role and is affiliated with the Conservative Party. This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Canada page, as well as the list of prime ministers of Canada

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The current President of the Philippines is Bongbong Marcos (Ferdinand R. Marcos Jr.), who assumed office on June 30, 2022. He is the 17th President of the Philippines and serves as both head of state and head of government. This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of the Philippines page, as well as the list of presidents of the Philippines

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Argentina (Argentina national football team) — the current 2026 FIFA World Cup champion (3rd title): defeated France in the 2022 final to claim their most recent World Cup title


================================================================================

*Report generated by CATS v2.0*
