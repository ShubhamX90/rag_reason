# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 123 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.967 (over 736 samples)

**GR F1** *(used in CATS)*: 0.980

**Behavior Adherence**: 0.622 (over 613 applicable samples)

**Factual Grounding**: 0.872 (over 613 applicable samples)

**Single-Truth Recall**: 0.491 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.741

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
- **Behavior**: 0.828 (n=157)
- **Grounding**: 0.907 (n=157)
- **Recall**: 0.448 (n=154)
- **CATS**: 0.790

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.672 (n=177)
- **Grounding**: 0.866 (n=177)
- **Recall**: 0.442 (n=156)
- **CATS**: 0.740

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.583 (n=96)
- **Grounding**: 0.869 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.814

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.445 (n=146)
- **Grounding**: 0.876 (n=146)
- **Recall**: 0.636 (n=140)
- **CATS**: 0.737

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.297 (n=37)
- **Grounding**: 0.748 (n=37)
- **Recall**: 0.324 (n=37)
- **CATS**: 0.570


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2040

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/mistral-small-4

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 2040
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d1
- **Claim**: However, the legal battle itself — initiated when the San Francisco police seized the poem and charged Lawrence Ferlinghetti with obscenity — demonstrated that "Howl" was widely regarded as containing graphic sexual content and coarse language, with the judge's ruling hinging on the work's artistic merit and social value

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: However, a non-iodine-deficient population can tolerate excess iodine quite well the majority of cases do not result in clinically fatal consequences — though susceptibility can be increased in specific groups such as those with autoimmune thyroid disease, the elderly fetuses

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: The primary nutrient concern, therefore, revolves around fiber — a food processor removes roughly half of it — while vitamins remain largely intact, making the extent of nutritional value loss context-dependent on one's dietary priorities

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: However, the U.S. federal court in Cavanaugh v

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: This divergence in legal and interpretive outcomes reflects ongoing scholarly and judicial debate about whether satire can constitute a serious, protected faith

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Technically, no — cows have one stomach, but it is divided into four distinct compartments: the rumen, the reticulum, the omasum the abomasum

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Overall, whether an epigenetic change is heritable appears to depend on the specific mechanism — such as rare genomic sites that preserve methylation across reprogramming events — and current scientific consensus has not established a general rule confirming epigenetic heritability across all contexts

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beliefs differ depending on who you ask

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: Yes — real Christmas trees are generally more sustainable than artificial ones, as they are grown like agricultural crops and absorb CO2 during their lifetime, whereas artificial trees are made from plastic and metal and release greenhouse gases during manufacturing and transportation

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: However, critics such as Oxford University researcher Amy Dickman argue that blanket bans could lead to more animals being killed, as those generating revenue through hunting are more likely to be protected the practice is frequently marked by corruption, illegal activities poor regulation

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Frontline scientists studying human-wildlife conflict in regions like Ruaha National Park are increasingly vocal about the probable harms of banning trophy hunting, noting that the current evidence base is fragmented and dominated by anecdotal and industry-aligned perspectives, with genuine scientific consensus on the net benefits remaining absent

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Opposing interpretations of the same data reflect the enduring ideological debate between conservatives and progressives on the root causes of wage disparities

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Proponents further contend that software patents are still valuable tools for protecting core algorithms and functions through the Doctrine of Equivalents that pursuing such patents remains essential for technology companies despite Supreme Court rulings like Alice v

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: However, some clinical evidence suggests it may have a modest effect on reducing the severity and duration of cold symptoms

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, organic farming can be more sustainable in terms of environmental impact per unit of output, producing fewer emissions and causing less soil erosion, even if it cannot match conventional yields

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
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
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Yes — modern birds descended from theropod dinosaurs, which include tyrannosaurines like T. rex, though T. rex itself was not a bird

### Sample conflictingqa_bac0f4d62f96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Some researchers estimate that cometary meteoroids make up about 95% of observed meteors and 38% of observed fireballs, though no conclusive evidence links any specific meteorite to a particular comet due to the near-impossibility of preserving cometary material intact through atmospheric entry

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: None
- **Claim**: A study by the Radio Project found that less than one-third of panicked listeners mistook the Martians for German invaders or a natural catastrophe historian W. Joseph Campbell has noted that the narrative of mass panic was primarily advanced by newspapers at the time as a means to discredit radio as a credible news source

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d1, d4, d2
- **Claim**: It is particularly valuable for vegans because it is high in B12, which is otherwise difficult to obtain from plant-based diets also provides 5–6 grams of protein per 100g

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, Sega officially denies his involvement some sources treat the claim as unconfirmed speculation, making the matter unresolved

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: However, experts differ on how strongly to interpret these findings — the same researchers note that bee venom is highly variable and can cause serious side effects other sources caution that no definitive evidence confirms bee sting therapy cures or even reliably alleviates arthritis symptoms

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: There is a common belief that animals can predict earthquakes some studies and anecdotal evidence suggest they can detect P-waves a few seconds before a quake strikes — because P-waves travel faster through solids, liquids gases than S-waves, animals can often sense the former before the ground begins to shake

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Emojis are widely regarded as supplementary to written language rather than a replacement

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some research also highlights that yerba mate possesses cytotoxic properties in laboratory settings, suggesting it may have both pro-cancerous and anti-cancerous effects simultaneously

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5, d2
- **Claim**: Symptoms such as eye strain, dryness, headaches blurred vision are common some researchers note that prolonged use may worsen conditions like convergence insufficiency

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Black holes themselves are invisible, but their effects can be observed with telescopes

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

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Maryam Mirzakhani (2014 — first and only female recipient; passed away 2017)

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3
- **Claim**: It is worth noting that while d3 lists several children who died in the disaster, it does not explicitly state the exact age of the youngest passenger, which is confirmed to be Millvina Dean through multiple corroborating sources

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 2026, the President of the United States is Joe Biden, who is serving his first term in office alongside Vice President Kamala Harris. This is corroborated by the official White House history, which confirms that the 117th Congress coincided with the Biden administration from January 20, 2021 to January 3, 2025 the 118th Congress continues under his presidency

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: These discrepancies reflect methodological differences in how 'cost' is defined — some sources use nominal production budgets, while others adjust for inflation and additional factors like reshoots and marketing — and the evolving nature of record-keeping in the film industry

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d10
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
- **Supporting Docs Found**: d6, d2, d8, d5, d7, d4
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not support a confident answer to the query

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: While the name shares linguistic roots with German Hans and Dutch Jan, its most direct derivation traces back to the Danish masculine given name Hans

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5, d4
- **Claim**: The retrieved evidence indicates that following the liberation of North Africa, Allied forces advanced into Tunisia, where they encircled and defeated approximately 250,000 German and Italian troops, ending the battle for North Africa by mid-1943

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d5, d3, d4
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

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: As a coach: Red Auerbach (16 championships as a coach and executive, though the query asks about players vs. coaches broadly so this is partial support) — as a player: Bill Russell (11 championships as a player, the most in NBA history) —

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4, d2
- **Claim**: Lacteals

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while the main Fairy Tail manga concluded in 2017, a sequel manga titled 'Fairy Tail: 100 Years Quest' continued serialization through 2026, with the most recent chapter released in May 2026

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
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The original bank began operations in mid-1912, with the note issue administered by the Australian Department of the Treasury

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5, d2
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
- **Supporting Docs Found**: d3, d4
- **Claim**: Roger Miller

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Notable bearer Christopher Tavarez is an American actor

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d2
- **Claim**: Surname origin data indicates Spanish-Portuguese ancestry predominates among people with this surname

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: Japan: 1996; in the US, January 9, 1999

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: In Mexico, toll roads are commonly called **autopistas**

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: The First Epistle of John was likely written between 95 and 110 AD, though some scholars place the composition as early as the 90s or as late as the first decade of the 2nd century

### Sample qacc_fbdae168fc6f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d1
- **Claim**: Sushma Swaraj

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Initially, benefit payments were scheduled to commence in 1942, with the first monthly check issued to Ida M. Fuller in January 1940 — nearly five years ahead of schedule

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The current Law Minister of India is Shri Kiren Rijiju, as confirmed by the Ministry of Law and Justice website

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is further corroborated by additional sources that list his cabinet position under the Union Cabinet

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Following growing dissatisfaction with the Confederation's limitations, representatives from several states convened in Philadelphia in 1787 to revise the Articles, ultimately producing the U.S. Constitution as the nation's enduring framework of government

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: New South Wales last won the State of Origin series in 2021

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: While d2 references an all-time scoring list and d5 lists current scoring leaders through the 2028 NBA draft, neither definitively names the top-ranked player, making them partial support compared to the fully verified career totals of

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_301378915064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Merritt Wever (Nurse Jackie)

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Gagan Narang (10m Air Rifle)

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida (won 2025)

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: None
- **Claim**: While Texas A&M won the 1986 WCWS and other programs like Arizona and Florida have claimed titles , no other team has approached UCLA's record of dominance in the sport

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: Android 16 (Baklava).

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: This figure represents a 5.92% increase from the previous year and is the latest quarterly update available from the official U.S. government source

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: The group was formed in Los Angeles in 1989 and quickly rose to fame with hits such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d4
- **Claim**: Shay Mitchell, who portrays Emily Fields in Pretty Little Liars, was born in November 1993, making Emily 31 years old according to the show's current timeline

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 670–680 nm

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: It is also worth noting that the name was first recorded in the Domesday Book of 1086 the Gerrard family name is traced to the grandson of Edward the Confessor, further attesting to its ancient roots

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d4
- **Claim**: Earlier in their history, the 76ers also reached the 2001 NBA Finals, where they lost to the Los Angeles Lakers, representing their most recent championship appearance at the time

### Sample trust_align_076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The mechanism involves both direct toxic effects on the heart muscle and cardiovascular system, as well as chemical-induced spasms of the coronary arteries, which can trigger fatal heart failure within seconds of inhalation

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Twitter is now known as X. This was confirmed when Twitter merged with X Holdings in April 2023, causing the company to cease being an independent entity and become part of X Corp. Wikipedia's redirect from Twitter to X reflects the current branding the platform itself is also referred to simply as X

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0072

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Twitter is currently known as X. This was confirmed when Twitter merged with X Holdings in April 2023, causing the company to cease being an independent entity and become part of X Corp. As a result, the platform's official name changed from Twitter to X, though the domain twitter.com remained unchanged

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Mark Carney is the current Prime Minister of Canada, having assumed office on 14 March 2025. This is confirmed by the official Wikipedia revision that superseded the older version in March 2026, which also notes his inauguration ceremony took place at the Office of the Prime Minister and Privy Council building. Additionally, the list of Canadian prime ministers explicitly ranks him as the 24th and most recent holder of the office, succeeding Justin Trudeau following the latter's resignation

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has served as the Federal President of the Federal Republic of Germany since taking office on that date. This is confirmed by the official Wikipedia entries on both the President of Germany and the Federal President, which list his incumbency from 19 March 2017. Since German reunification in 1990, the President has been the head of state for all of Germany, following the creation of the office by the Basic Law for the Federal Republic of Germany of May 1949

### Sample wikirevision_0154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Claudia Sheinbaum Pardo has been serving as the 66th President of Mexico, making her the latest President. This is confirmed by the official Wikipedia revision that superseded the older version in April 2025, which explicitly names her as incumbent with a detailed biography. She is the first woman and the first Jewish person to hold the office, having previously served as Head of Government of Mexico City from 2018 to 2023

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: 2022, Bongbong Marcos is the President of the Philippines, serving as both head of state and government and commander-in-chief of the armed forces. He is also the incumbent President of the Senate, serving as the third highest and most powerful official in the government

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Carlos Alcaraz (defending champion did not participate; Jannik Sinner won instead) — see 2026 French Open for current champion


================================================================================

*Report generated by CATS v2.0*
