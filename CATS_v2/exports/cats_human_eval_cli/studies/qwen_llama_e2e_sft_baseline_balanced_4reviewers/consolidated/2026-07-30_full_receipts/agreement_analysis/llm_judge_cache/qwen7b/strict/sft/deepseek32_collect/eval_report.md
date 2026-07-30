# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 123 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.967 (over 736 samples)

**GR F1** *(used in CATS)*: 0.980

**Behavior Adherence**: 0.837 (over 613 applicable samples)

**Factual Grounding**: 0.844 (over 613 applicable samples)

**Single-Truth Recall**: 0.710 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.843

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.980
- **Precision**: 0.992
- **Recall**: 0.969
- **Accuracy**: 0.967
- TP=589, FP=5, FN=19, TN=123

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.866
- **Abstain Recall**: 0.961
- **Abstain F1**: 0.911
- **Specificity**: 0.969
- Abstain TP=123, FP=19, FN=5, TN=589


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (54 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.967
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.949 (n=157)
- **Grounding**: 0.887 (n=157)
- **Recall**: 0.799 (n=154)
- **CATS**: 0.903

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.876 (n=177)
- **Grounding**: 0.808 (n=177)
- **Recall**: 0.679 (n=156)
- **CATS**: 0.836

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.646 (n=96)
- **Grounding**: 0.841 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.825

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.781 (n=146)
- **Grounding**: 0.878 (n=146)
- **Recall**: 0.700 (n=140)
- **CATS**: 0.838

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.892 (n=37)
- **Grounding**: 0.707 (n=37)
- **Recall**: 0.514 (n=37)
- **CATS**: 0.756


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2040

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/deepseek-r1-distill-32b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2040
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: Anime is widely considered to be a specific style of cartoon, though some anime fans and critics distinguish anime as a separate art form

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Unlike animals or trees, fungi reproduce via genetic cloning, allowing them to expand extensively underground and claim the title of largest organism

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2, d3
- **Claim**: The primary nutrient concern, therefore, revolves around fiber — a food processor removes roughly half of it — while vitamins remain largely intact, making the extent of nutritional value loss context-dependent on one's dietary priorities

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: However, the U.S. federal court in Cavanaugh v

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: This divergence in legal and interpretive outcomes reflects ongoing scholarly and judicial debate about whether satire can constitute a serious, protected faith

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Yes — anyone can start a business, though not everyone will succeed

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Technically, no — cows have one stomach, but it is divided into four distinct compartments: the rumen, the reticulum, the omasum the abomasum

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Overall, whether an epigenetic change is heritable appears to depend on the specific mechanism — such as rare genomic sites that preserve methylation across reprogramming events — and current scientific consensus has not established a general rule confirming epigenetic heritability across all contexts

### Sample conflictingqa_39fe5c441657

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Beliefs differ depending on who you ask

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Yes, the Moon exhibits evidence of recent geological activity

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Yes — real Christmas trees are generally more sustainable than artificial ones, as they are grown like agricultural crops and absorb CO2 during their lifetime, whereas artificial trees are made from plastic and metal and release greenhouse gases during manufacturing and transportation

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Frontline scientists studying human-wildlife conflict in regions like Ruaha National Park are increasingly vocal about the probable harms of banning trophy hunting, noting that the current evidence base is fragmented and dominated by anecdotal and industry-aligned perspectives, with genuine scientific consensus on the net benefits remaining absent

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Proponents further contend that software patents are still valuable tools for protecting core algorithms and functions through the Doctrine of Equivalents that pursuing such patents remains essential for technology companies despite Supreme Court rulings like Alice v

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Clinicians generally agree that the risk of regrowth is reduced when surgery is performed later in childhood and when thorough removal of the adenoid tissue is achieved

### Sample conflictingqa_62b1aff6586d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: However, regional holes persist, persistent warming threatens recovery emerging concerns like rocket launches could slow the healing process

### Sample conflictingqa_63fde268aa8c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Religious and philosophical views differ; scientifically, there is no established evidence that the mind exists separately from the body

### Sample conflictingqa_676188bf8139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: Yes — the Lantern Festival is traditionally a Buddhist holiday honoring deceased ancestors is also associated with peace and reconciliation

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d4
- **Claim**: A Nature article further notes that while the Moon's pull could increase the magnitude of tremors, the link breaks down for smaller earthquakes, suggesting any effect is limited to large events ; the original GeoQuake.org claim that major earthquakes are more frequent during full moons is therefore contested, with scientific consensus leaning toward no definitive causal link

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The Gutenberg Bible was not the first book printed with movable type, though it is widely recognized as the first major book produced using mass-produced metal movable type in Europe

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, some clinical evidence suggests it may have a modest effect on reducing the severity and duration of cold symptoms

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Strong wind can also discourage bees from flying, regardless of rainfall intensity, as it disrupts their navigation and makes it harder to collect nectar and pollen

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: However, organic farming can be more sustainable in terms of environmental impact per unit of output, producing fewer emissions and causing less soil erosion, even if it cannot match conventional yields

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Some sources further note that organic standards prohibit synthetic pesticides, meaning organic farms do use pesticides — albeit naturally derived ones — and that the trade-off between reduced chemical use and lower efficiency is a subject of ongoing debate

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Partially — most bird species can distinguish conspecific calls, but researchers differ on whether calls are unique to each individual

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Yes — modern birds descended from theropod dinosaurs, which include tyrannosaurines like T. rex, though T. rex itself was not a bird

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Not all snakes can swim

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: However, some formal and academic writing styles prefer the two-word spelling 'all right,' and many style guides recommend reserving 'alright' for informal contexts

### Sample conflictingqa_bac0f4d62f96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Some researchers estimate that cometary meteoroids make up about 95% of observed meteors and 38% of observed fireballs, though no conclusive evidence links any specific meteorite to a particular comet due to the near-impossibility of preserving cometary material intact through atmospheric entry

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5, d1
- **Supporting Docs Found**: None
- **Claim**: A study by the Radio Project found that less than one-third of panicked listeners mistook the Martians for German invaders or a natural catastrophe historian W. Joseph Campbell has noted that the narrative of mass panic was primarily advanced by newspapers at the time as a means to discredit radio as a credible news source

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It depends significantly on the specific lifecycle stage and disposal method considered

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Some plants can survive without sunlight for short periods, such as during a prolonged night or in very dark indoor environments, as the RHS confirms

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Religious bodies hold differing views: some treat the Bible as inerrant (meaning it contains no factual errors), while others view it as infallible (meaning it cannot be contradicted by truth) still others see it as a historically conditioned text

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: However, experts differ on how strongly to interpret these findings — the same researchers note that bee venom is highly variable and can cause serious side effects other sources caution that no definitive evidence confirms bee sting therapy cures or even reliably alleviates arthritis symptoms

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: However, the evidence does not definitively confirm that the Dutch were the sole discoverers of Australia, as earlier encounters by Portuguese navigators and later British expeditions are also referenced

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some research also highlights that yerba mate possesses cytotoxic properties in laboratory settings, suggesting it may have both pro-cancerous and anti-cancerous effects simultaneously

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Black holes themselves are invisible, but their effects can be observed with telescopes

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Aryna Sabalenka and Amanda Anisimova

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Maryam Mirzakhani (2014 — first and only female recipient; passed away 2017)

### Sample freshqa_25b286cb2af1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: The idea that Venus could have had a moon in the past is also discussed, as scientists are not entirely sure why Venus lacks a moon today

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
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: This result, with the ceremony also seeing wins for Michael B. Jordan in Best Actor and Autumn Durald Arkapaw as the first female cinematographer to win an Oscar

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Luka Modric was the only player to beat Ronaldo and Messi in 2018, further contextualizing Kaka's win as the one immediately preceding the supercycle

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: When We Were Real by Daryl Gregory (Saga), published by Harper Voyager US and Harper Voyager UK

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: This untimely death dealt a significant blow to the field of neural networks, as Rosenblatt was the original proponent of the perceptron concept his passing occurred in 1971

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Jiangsu

### Sample freshqa_d4d59d75b206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: 2015–2016

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These discrepancies reflect methodological differences in how 'cost' is defined — some sources use nominal production budgets, while others adjust for inflation and additional factors like reshoots and marketing — and the evolving nature of record-keeping in the film industry

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: No permanent cure for cancer has been developed; however, several milestone treatments have been achieved

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d7, d10, d5
- **Claim**: The May 16 coup in 1961 marked a significant internal Korean political event, but the external historical framework of Japan's rule and WWII's conclusion provides the most direct and comprehensive answer to the query

### Sample hotpotqa_0056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: Therefore, the answer to the query is England

### Sample hotpotqa_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Thal appeared in a large number of musicals as well, further fitting the description of the actor sought

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Separately, Brad Robert Wenstrup was treated at MedStar Washington Hospital Center following the Steve Scalise shooting , further contextualizing that institution within the D.C. hospital landscape

### Sample hotpotqa_0192

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6
- **Supporting Docs Found**: d3
- **Claim**: English cartographer and historian John Speed (1551–1629), best known as a mapmaker of the Stuart period

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d6, d4, d7, d8
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not support a confident answer to the query

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: While the name shares linguistic roots with German Hans and Dutch Jan, its most direct derivation traces back to the Danish masculine given name Hans

### Sample qacc_0b3b6452e224

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d4
- **Claim**: An earlier Reddit discussion also mentions that the statue was initially designed as an Egyptian woman, though this is presented as a historical note rather than a definitive statement the French channel video confirms that Bartholdi himself modeled the statue's face after his mother

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d1, d3, d4
- **Claim**: Sakshi Malik (Haryana); Madhuri Dixit (India); Parineeti Chopra (Haryana); Avani Lekhara (Rajasthan); Bhawna Dehariya and her daughter Siddhi Mishra (Madhya Pradesh)

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_367b09e4ed80

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: The gesture of crossing fingers for good luck has its roots in pre-Christian European pagan beliefs, where the cross shape was thought to concentrate good spirits and serve as a focal point for anchoring wishes. Over time, this practice evolved into a one-handed habit was further associated with early Christianity — specifically, as a secret sign (L-shaped fingers touching) used by early Christian followers to identify and gather with one another during a time when their faith was persecuted. The gesture was then simplified to the familiar index-finger-over-middle-finger form came to be associated with invoking divine protection rather than merely seeking luck

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4
- **Claim**: As a coach: Red Auerbach (16 championships as a coach and executive, though the query asks about players vs. coaches broadly so this is partial support) — as a player: Bill Russell (11 championships as a player, the most in NBA history) —

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: Lacteals

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: These constitutional shifts were confirmed when Canada signed the Treaty of Versailles in 1919 as an equal partner the Canada Act of 1982 formally ended all remaining legal ties to the British Parliament

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Multiple

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a1c73439eca9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Season 9 of *El Señor de los Cielos* premiered on 13 February 2024 , making that the most recent season

### Sample qacc_aa94588b9477

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The original bank began operations in mid-1912, with the note issue administered by the Australian Department of the Treasury

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2, d5
- **Claim**: President Hoover was hosting a young people's party for the children of his staff at the time the blaze damaged the Executive Offices, including the President's office and the offices of his three secretaries

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Roger Miller

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Notable bearer Christopher Tavarez is an American actor

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4
- **Claim**: Surname origin data indicates Spanish-Portuguese ancestry predominates among people with this surname

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Japan: 1996; in the US, January 9, 1999

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: In Mexico, toll roads are commonly called **autopistas**

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The First Epistle of John was likely written between 95 and 110 AD, though some scholars place the composition as early as the 90s or as late as the first decade of the 2nd century

### Sample qacc_fbdae168fc6f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d3
- **Claim**: Sushma Swaraj

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Red license plates have different meanings depending on the jurisdiction and context

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The minimum age to drive a transport vehicle varies by jurisdiction and vehicle type

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Initially, benefit payments were scheduled to commence in 1942, with the first monthly check issued to Ida M. Fuller in January 1940 — nearly five years ahead of schedule

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This form of government is further characterized by the principle of checks and balances, ensuring no single branch accrues too much power by the requirement that all State governments maintain a republican form, though not necessarily the three-branch structure

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is further corroborated by additional sources that list his cabinet position under the Union Cabinet

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that some regional governments maintain their own Law Ministers — for example, Gujarat's Law Minister is Shri Hitendra K. Desai — but the most current and authoritative answer at the national level is Shri Kiren Rijiju

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Following growing dissatisfaction with the Confederation's limitations, representatives from several states convened in Philadelphia in 1787 to revise the Articles, ultimately producing the U.S. Constitution as the nation's enduring framework of government

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This cultural shift was further reinforced through the 19th and 20th centuries as immigration brought strong coffee traditions to America by the 2025, approximately 75% of American adults drink coffee daily

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: New South Wales last won the State of Origin series in 2021

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5
- **Claim**: While d2 references an all-time scoring list and d5 lists current scoring leaders through the 2028 NBA draft, neither definitively names the top-ranked player, making them partial support compared to the fully verified career totals of

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_301378915064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Merritt Wever (Nurse Jackie)

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1
- **Claim**: Gagan Narang (10m Air Rifle)

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida (won 2025)

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: 16 teams have finished in the top 16, with UCLA among them the official NCAA WCWS history lists multiple UCLA championships from the 1980s through 2019

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: Android 16 (Baklava).

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Syndicated comics based on Avatar: The Last Airbender are coming out in 2025–2026, though the specific next title depends on which series you are asking about

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: This figure represents a 5.92% increase from the previous year and is the latest quarterly update available from the official U.S. government source

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2, d3
- **Claim**: It is classified as a lysosomal storage disorder because the missing enzyme prevents the breakdown of GM2-ganglioside within cells

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4
- **Claim**: The Cumberland River begins on Poor Fork in Letcher County, Kentucky, near Flat Gap on Pine Mountain ends at Smithland, Kentucky, where it merges with the Ohio River

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2
- **Claim**: The group was formed in Los Angeles in 1989 and quickly rose to fame with hits such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Shay Mitchell, who portrays Emily Fields in Pretty Little Liars, was born in November 1993, making Emily 31 years old according to the show's current timeline

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 670–680 nm

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: It is also worth noting that the name was first recorded in the Domesday Book of 1086 the Gerrard family name is traced to the grandson of Edward the Confessor, further attesting to its ancient roots

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Earlier in their history, the 76ers also reached the 2001 NBA Finals, where they lost to the Los Angeles Lakers, representing their most recent championship appearance at the time

### Sample trust_align_076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d5, d4
- **Claim**: The mechanism involves both direct toxic effects on the heart muscle and cardiovascular system, as well as chemical-induced spasms of the coronary arteries, which can trigger fatal heart failure within seconds of inhalation

### Sample trust_align_173

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Fruit contains fructose (a natural sugar) combined with fiber, vitamins antioxidants, which mitigates the blood sugar impact compared to isolated added sugars in candy and soda

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Ousmane Dembélé was the winner. The 2025 Ballon d'Or (the 69th annual award) was presented to Dembélé, who earned his first win, for excellence in association football during the 2024–25 season. This is the most recent Ballon d'Or awarded, making Dembélé the current winner

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Mark Carney is the current Prime Minister of Canada, having assumed office on 14 March 2025. This is confirmed by the official Wikipedia revision that superseded the older version in March 2026, which also notes his inauguration ceremony took place at the Office of the Prime Minister and Privy Council building. Additionally, the list of Canadian prime ministers explicitly ranks him as the 24th and most recent holder of the office, succeeding Justin Trudeau following the latter's resignation

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has served as the Federal President of the Federal Republic of Germany since taking office on that date. This is confirmed by the official Wikipedia entries on both the President of Germany and the Federal President, which list his incumbency from 19 March 2017. Since German reunification in 1990, the President has been the head of state for all of Germany, following the creation of the office by the Basic Law for the Federal Republic of Germany of May 1949

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022. This is confirmed by the official Parliament of Australia website, which lists his incumbency from that date. Wikipedia's list of Prime Ministers of Australia further contextualizes his tenure, noting that he leads a Labor government following the 2022 federal election

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: 2022, Bongbong Marcos is the President of the Philippines, serving as both head of state and government and commander-in-chief of the armed forces. He is also the incumbent President of the Senate, serving as the third highest and most powerful official in the government

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Droupadi Murmu is the current President of India, serving as the country's head of state. She was elected by the Electoral College comprising members of both houses of Parliament and assumed office on 05 July 2022. Her term is scheduled to conclude on 24 May 2027, after which her vice president is expected to succeed her

### Sample wikirevision_0166

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Prime Minister of India is Narendra Modi, who has served in office since 26 May 2014. He is the Honourable Mr. Prime Minister and holds the highest office of the Government of India, being appointed by the President and responsible to the Lok Sabha


================================================================================

*Report generated by CATS v2.0*
