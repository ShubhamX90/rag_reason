# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.931 (over 736 samples)

**GR F1** *(used in CATS)*: 0.956

**Behavior Adherence**: 0.790 (over 609 applicable samples)

**Factual Grounding**: 0.807 (over 609 applicable samples)

**Single-Truth Recall**: 0.652 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.801

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.956
- **Precision**: 0.998
- **Recall**: 0.918
- **Accuracy**: 0.931
- TP=558, FP=1, FN=50, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.718
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.833
- **Specificity**: 0.918
- Abstain TP=127, FP=50, FN=1, TN=558


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.897 (n=155)
- **Grounding**: 0.901 (n=155)
- **Recall**: 0.786 (n=154)
- **CATS**: 0.893

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.873
- **GR F1** *(used in CATS)*: 0.914
- **Behavior**: 0.886 (n=176)
- **Grounding**: 0.705 (n=176)
- **Recall**: 0.516 (n=156)
- **CATS**: 0.755

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.954
- **GR F1** *(used in CATS)*: 0.973
- **Behavior**: 0.656 (n=96)
- **Grounding**: 0.833 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.821

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.962
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.697 (n=145)
- **Grounding**: 0.862 (n=145)
- **Recall**: 0.696 (n=140)
- **CATS**: 0.808

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.595 (n=37)
- **Grounding**: 0.617 (n=37)
- **Recall**: 0.500 (n=37)
- **CATS**: 0.648


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2090

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
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: None
- **Claim**: Additional sources indicate that St. John's Wort may interact with other medications and is not recommended for patients with bipolar disorder or phototoxicity sensitivity

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: If you have high blood pressure or other cardiovascular risks, you should exercise caution when engaging in weight lifting activities and consider discussing your specific situation with a healthcare provider

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: A parent objected to it being read in a Colorado high school in 2019-2020

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: At the same time, Jewish identity is not solely defined by religion or culture, as it is practiced and understood differently across distinct communities and generations

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: However, iodine deficiency remains a global health concern physiologic iodine intake is necessary for thyroid hormone synthesis, meaning that iodine supplementation should not be avoided indiscriminately — particularly in iodine-deficient populations — but rather targeted to adequate levels to avoid excessive intake

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: The key takeaway is that the benefits of wrist rests depend heavily on proper alignment and use individuals should exercise caution when selecting a product or implementing this ergonomic practice

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: For example, simple linear models may perform well with relatively small datasets, while complex deep neural networks require large amounts of structured or unstructured data to train effectively

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d4
- **Claim**: Related evidence

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, particularly if you plan to use them for more than two years. The reason is simple: real trees are grown as renewable crops that absorb carbon dioxide and provide habitat for wildlife, while artificial trees are made from non-renewable plastic and metal materials and have a large carbon footprint from manufacturing and shipping

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: On the other hand, some scientists and conservation organizations argue that the practice normalizes animal cruelty, concentrates benefits primarily among wealthy individuals that the evidence on long-term conservation outcomes is insufficient to justify the ongoing harvest of animals for trophies

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The conflicting outcomes reflect methodological differences in how researchers define 'earthquake likelihood' and the statistical models they use, making definitive answers elusive for now

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: However, bees can distinguish light rain from heavy storms and may continue to forage in light rain, while remaining cautious about flying in inclement weather

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Religious truth claims are inherently theological opinions rather than historically verifiable facts, so the question of whether the Catholic Church is the true church is properly understood as a theological debate rather than a historical proof

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The answer is not straightforward because the evidence is divided across research outcomes and philosophical perspectives on the same issue

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1
- **Claim**: However, some variation exists — for example, ducks are born with innate vocalization skills that allow them to quack perfectly regardless of their upbringing individual birds may develop slight differences in their call repertoire depending on their personal needs and experiences

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2
- **Claim**: Contrary to a common misconception, stalactites do not require dripping water in dry air to grow — instead, they initiate as soda straws where water flow deposits calcite crystals along their length this process can continue underwater once the structure has initiated

### Sample conflictingqa_b2524e4883ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: For smaller meteors, the primary risk is that they disperse radioactive dust rather than causing physical damage , while for larger events like the Taurid shower, scientists continue to assess the likelihood of a catastrophic impact

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The conflict arises from differing methodological scopes—per-use emissions vs. full life-cycle impacts—and the evidence remains mixed, with experts suggesting that refusing straws altogether is a better near-term solution than debating which material is definitively greener

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Religious and philosophical views differ; science has not established the Bible as historically or scientifically infallible

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: The retrieved evidence presents a genuine curse: folklore holds that witches objected to Shakespeare's use of their real incantations in Macbeth, causing the actor playing Lady Macbeth to die during the first performance the play has been plagued with accidents and deaths ever since

### Sample conflictingqa_f43b2c51deea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3, d2
- **Supporting Docs Found**: None
- **Claim**: In contrast, scientific analysis of a famous 1988 earthquake found no correlation between unusual animal behavior and the event another study concluded that dogs' behavior before earthquakes does not clearly prove they are sensing impending events ; most scientists pursuing the question are in China or Japan, where conflicting theories on seismic-escape behavior abound

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, some lower-credibility sources claim yerba mate may have anticancer properties and that its cytotoxic effects on cancer cells in vitro suggest it could help prevent cancer rather than cause it ; however, these benefits remain under investigation experts have long debated whether yerba mate's complex mix of compounds—some of which are carcinogenic while others are anticancer—can be reconciled into a single definitive answer

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3, d1, d2
- **Claim**: The retrieved evidence is conflicting. Some sources say the Phoenix Lights were explained as military flares dropped during a training exercise, while others say witnesses believed the lights were UFOs rather than flares

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d1
- **Claim**: This revision of the diplodocid family tree was further supported by additional research published in 2020, which confirmed that Brontosaurus was a valid genus distinct from Apatosaurus

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Religion is complex and contested the answer to whether Mormons are Christian depends on the definition of 'Christian' used. Many Mormons self-identify as Christians — believing in Jesus Christ and following His teachings — though they are often rejected as such by evangelical Christians who argue that Mormon theology is fundamentally at odds with historic orthodoxy

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: King Charles III did not immediately strip Prince Harry of his title as Duke of Sussex when he became king; the two brothers first discussed the matter during the Sandringham summit in January 2020

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: It runs through April 9, with the first Seder on the evening of April 1 and the second Seder on April 2

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2130eea851fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: His total citation count on the platform continues to grow rapidly, reflecting his seminal contributions to machine learning and artificial intelligence research

### Sample freshqa_2877cf4bd00f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: 2025, Donald Trump was 78 years, 7 months and 6 days old

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

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: d3
- **Claim**: This version supersedes all older major versions, including .NET 5.0 and .NET 4.8.1, which are no longer the most recent releases

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d1, d4
- **Claim**: The first atomic bomb test in the United States took place at the Trinity Site in New Mexico on July 16, 1945. The test, code-named 'Trinity,' was conducted at a site 210 miles south of Los Alamos on the Alamogordo Bombing Range, in the New Mexican desert. This was the world's first nuclear explosion the site is now part of the White Sands Missile Range

### Sample freshqa_3ad16f379533

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
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

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This figure is corroborated by Costco's own website, which states that Executive Memberships are valid for one year and come with an annual 2% reward of up to $1,250

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2, d4
- **Supporting Docs Found**: None
- **Claim**: This result, superseding earlier reports that listed *CODA* (2022) or *Sinners* (2024) as the most recent winners, as those awards have since been surpassed by the 2026 ceremony

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The 2022 Winter Olympics were the first ever to be held in China the country used the experience gained from the 2008 Games to ensure a seamless event

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel was won by *When We Were Real* by Daryl Gregory, published by Saga Press in 2025. The award was announced at the 2026 Nebula Conference, held in San Antonio, Texas

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: A boating accident in Chesapeake Bay on July 28, 1971

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: Queen Elizabeth II of England died on 8 September 2022

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: CANCER CURES: A Brief Historical Overview

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The answer depends on the specific taxonomic group and definition of 'lung' being considered

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, eligibility depends on specific vaccine type and individual health conditions, so it is important to check with a healthcare provider before administering any vaccine

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample healthcontradict_0060

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: A multi-institutional study of over 1,000 children found that tepid water sponging did not lower fever any more than simple clothing removal did that both were ineffective when compared to standard fever management with acetaminophen or ibuprofen

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Parents should instead focus on keeping their child hydrated, monitoring their temperature regularly consulting a healthcare provider if fever persists

### Sample hotpotqa_0025

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

### Sample hotpotqa_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, d4 and d8 (low-quality sources) falsely claim that El Nuevo Cojo is also owned by Time Inc., creating a factual accuracy mechanism where unreliable sources contradict the credible Time Inc

### Sample hotpotqa_0079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d8
- **Claim**: Lucas di Grassi

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b75ed799d46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: The ceremony streamed live on Netflix at 8:00 p.m. ET/5:00 p.m. PT, hosted by Kristen Bell for the third time

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The retrieved evidence indicates that after the North Africa campaign, the Allies moved eastward across North Africa and entered Europe via Italy

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

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The exact timing and sequence of events leading to the emergence of the first vertebrates is still a subject of ongoing scientific investigation, with researchers continually refining their understanding of this foundational chapter in Earth's history

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Early Christians adopted the gesture as a secret sign among themselves — touching thumb and index fingers to form an 'L' shape — which later evolved into the modern one-handed cross

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

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by both historical records and contemporary accounts, with the tree being decorated using candles and sweets in anticipation of a Christmas party

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, these figures are subject to change as passport accessibility for U.S. citizens continues to evolve over time

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: October 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nana is an Australian Shepherd

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Russ Ballard

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d3
- **Claim**: The ISS was designed between 1984 and 1993 its first occupation by astronauts occurred in October 2000, when Expedition 1 began

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
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: This admission date is consistently confirmed across multiple sources, with New Mexico becoming the 47th state in the Union

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: President Hoover and his family were hosting a children's party in the East Wing at the time the firefighting effort was so intense that one firefighter was scorched by a back draft of flames

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: This joint allows for movement and sound transmission between the two bones, which are essential for hearing

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Most effigy mounds were built between 700 and 1200 A.D., with the majority constructed during the Woodland period (approximately 750–1050 A.D.)

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: This date is consistently confirmed across multiple authoritative sources, including the official Social Security Administration website and the St. Louis Federal Reserve Economic Data

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: More recent research has suggested that the Milky Way may also exhibit features of an elliptical galaxy (E1), particularly in its outer regions, though it remains classified primarily as a barred spiral

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: Specific segments are identified by a suffix 'D' (e.g., Fed

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Their most recent appearance in the competition was the 2025–26 season , which was their first time qualifying since the 1990s

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells

### Sample qacc_f2218f8c979e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while 'prime rib' is a general term for this cut, it does not exclusively denote USDA Prime-graded beef prime rib roasts can also be selected from Choice or Select gradings

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: With a population of approximately 39 million people, California is the most populous state, making it the most crowded of the three largest states by a wide margin

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that when ranked by population, California is the first largest state, while Texas is the second largest , creating a slight contradiction depending on the metric used to define 'largest.'

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_779fd84224fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This date and location are confirmed by multiple authoritative sources, including the official 1788 census report by Governor Phillip

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: At the same time, the U.S. is described as a representative democracy, where citizens elect representatives to govern on their behalf the country's political system is further shaped by a system of checks and balances designed to prevent any single branch from accumulating too much power

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: This was confirmed by Secretary of State Dean Rusk, who stated that Kennedy's commitment to South Vietnam was 'neither desirable nor necessary' after assuming the presidency following Kennedy's assassination

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a grizzly bear, which is also the official state animal of California. The bear on the flag is the California grizzly bear (Ursus arctos californicus), an extinct population of the brown bear its inclusion on the flag dates back to 1846 when California was part of Mexico

### Sample situatedqa_geo_c76a815bb275

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d1
- **Claim**: At the global level, broad definitions of 'chief commercial tree crops' could include industrial timber species like teak or mahogany, as well as fruit trees like cocoa, oil palm rubber, depending on the source

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: He is a senior BJP leader and has served in various legal and political roles, including as a Member of Parliament and as a Minister of State for Home Affairs

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: However, the Articles proved to be too weak to address the needs of a growing nation — such as raising armies, collecting taxes regulating commerce — leading the states to convene a constitutional convention in Philadelphia in 1787, which produced the U.S. Constitution

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d2, d4
- **Supporting Docs Found**: None
- **Claim**: All four documents consistently reinforce this conclusion, with no contradictions across sources

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: England are the current champions after winning the 2019 World Cup, with New Zealand as the runner-up

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This victory gave the Blues a 2-1 edge in the 2024 series, after Queensland had won the first game 18-6

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, the official NBA stats page for the 2026 season introduces Victor Wembanyama as the current leader in regular season scoring with 31.9 points per game, suggesting the all-time record may eventually be surpassed

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_301378915064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This was her first Emmy win and second consecutive nomination for the show, making it a significant career milestone

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
- **Claim**: Argentina defeated France 4-2 in the final held in Lusail Stadium in Qatar, claiming their third title

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: It is written by Brandon Hoáng and illustrated by BellBessa, serving as a direct sequel to the original series that takes place after the events of Book 3

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: Multiple sources further corroborate this release, noting that the first issue will be available on May 6, 2026, with the entire series spanning three monthly installments

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: None
- **Claim**: This two-step process of first declaring it a monument and then upgrading it to a park is confirmed by multiple authoritative sources, with the official park website and travel Alaska both confirming the 1980 founding date

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: PTI's victory gave Imran Khan his first term as Prime Minister, after which he promised to investigate the allegations of electoral misconduct

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

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: These discrepancies arise because different sources capture different components of the total burden—federal taxes are uniform across all states, state excise taxes vary by state local sales taxes differ by region—creating temporal and definitional inconsistencies

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d1
- **Claim**: High-credibility sources consistently reinforce that red represents the longest wavelengths humans can perceive, with the full spectrum extending from ultraviolet through visible light into infrared wavelengths

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1
- **Claim**: It was originally formed from the Old German name Gerhard, which itself consisted of the elements gēr ('spear') and hard ('hardy, brave strong')

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Another theory proposes that the surname was first recorded in Lancashire in the Domesday Book of 1086, tracing back to the grandson of Edward the Confessor

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin & Perry Go Large

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

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Argentina (defeated Italy 3–2 in a penalty shootout at the Rose Bowl, Pasadena, California)

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Argentina defeated France 4–2 on penalties after extra time in the final, securing their first title since 1986

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the latest ICC Men's Cricket World Cup, the 2023 edition, by beating India in the final by six wickets at the Narendra Modi Stadium in Ahmedabad. This victory gave Australia their sixth World Cup title, with Glenn McGrath as the most successful bowler and Sachin Tendulkar as the all-time leading run-scorer. The 2023 tournament was hosted in India across ten venues from 5 October to 19 November 2023

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed by the Haryana government's official gazette notification in April 2023, superseding the older name of Gurgaon. The change is consistently reflected across all sources, with the newer Wikipedia revision of Gurgaon explicitly stating 'Gurugram' as the official name the older revision also confirming the same fact

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Argentina (defeated France 4–2 on penalties after a 3–3 draw)

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: His 2026 victory, superseding his 2025 title as the latest result


================================================================================

*Report generated by CATS v2.0*
