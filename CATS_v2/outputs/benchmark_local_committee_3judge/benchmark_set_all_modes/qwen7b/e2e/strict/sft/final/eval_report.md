# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 123 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.967 (over 736 samples)

**GR F1** *(used in CATS)*: 0.980

**Behavior Adherence**: 0.780 (over 613 applicable samples)

**Factual Grounding**: 0.838 (over 613 applicable samples)

**Single-Truth Recall**: 0.687 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.821

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
- **Behavior**: 0.924 (n=157)
- **Grounding**: 0.870 (n=157)
- **Recall**: 0.760 (n=154)
- **CATS**: 0.883

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.932 (n=177)
- **Grounding**: 0.823 (n=177)
- **Recall**: 0.628 (n=156)
- **CATS**: 0.841

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.531 (n=96)
- **Grounding**: 0.829 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.783

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.637 (n=146)
- **Grounding**: 0.857 (n=146)
- **Recall**: 0.711 (n=140)
- **CATS**: 0.800

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.649 (n=37)
- **Grounding**: 0.725 (n=37)
- **Recall**: 0.541 (n=37)
- **CATS**: 0.707


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2040

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

### Sample conflictingqa_0875b5f3262a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1
- **Claim**: However, the legal battle itself — initiated when the San Francisco police seized the poem and charged Lawrence Ferlinghetti with obscenity — demonstrated that "Howl" was widely regarded as containing graphic sexual content and coarse language, with the judge's ruling hinging on the work's artistic merit and social value

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
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
- **Supporting Docs Found**: d5, d2, d3, d1
- **Claim**: This divergence in legal and interpretive outcomes reflects ongoing scholarly and judicial debate about whether satire can constitute a serious, protected faith

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Technically, no — cows have one stomach, but it is divided into four distinct compartments: the rumen, the reticulum, the omasum the abomasum

### Sample conflictingqa_264e97231363

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Some sources describe the process as follows: when copper is oxidized (exposed to air), it turns from orange to dull green this green compound then adheres to the hair shaft

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Overall, whether an epigenetic change is heritable appears to depend on the specific mechanism — such as rare genomic sites that preserve methylation across reprogramming events — and current scientific consensus has not established a general rule confirming epigenetic heritability across all contexts

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beliefs differ depending on who you ask

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: Multiple studies confirm it had its origins in Queensland about four million years ago and persisted there until approximately 300,000 years ago, after which it crossed over to the Indonesian archipelago

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: Yes — real Christmas trees are generally more sustainable than artificial ones, as they are grown like agricultural crops and absorb CO2 during their lifetime, whereas artificial trees are made from plastic and metal and release greenhouse gases during manufacturing and transportation

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Frontline scientists studying human-wildlife conflict in regions like Ruaha National Park are increasingly vocal about the probable harms of banning trophy hunting, noting that the current evidence base is fragmented and dominated by anecdotal and industry-aligned perspectives, with genuine scientific consensus on the net benefits remaining absent

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Opposing interpretations of the same data reflect the enduring ideological debate between conservatives and progressives on the root causes of wage disparities

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Proponents further contend that software patents are still valuable tools for protecting core algorithms and functions through the Doctrine of Equivalents that pursuing such patents remains essential for technology companies despite Supreme Court rulings like Alice v

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d5
- **Claim**: However, regional holes persist, persistent warming threatens recovery emerging concerns like rocket launches could slow the healing process

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: However, many products can temporarily mask the damage by smoothing the cuticle, adding weight to frayed ends creating temporary glue-like bonds between split fibers — though the effects typically persist only until the next shampoo

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: However, some clinical evidence suggests it may have a modest effect on reducing the severity and duration of cold symptoms

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Strong wind can also discourage bees from flying, regardless of rainfall intensity, as it disrupts their navigation and makes it harder to collect nectar and pollen

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, organic farming can be more sustainable in terms of environmental impact per unit of output, producing fewer emissions and causing less soil erosion, even if it cannot match conventional yields

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Some sources further note that organic standards prohibit synthetic pesticides, meaning organic farms do use pesticides — albeit naturally derived ones — and that the trade-off between reduced chemical use and lower efficiency is a subject of ongoing debate

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: The retrieved evidence presents competing views. Some sources argue that the Catholic Church is the One True Church founded by Jesus Christ, while others argue that Scripture determines a church's truth based on its doctrine and practice rather than its temporal precedence

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Partially — most bird species can distinguish conspecific calls, but researchers differ on whether calls are unique to each individual

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Yes — modern birds descended from theropod dinosaurs, which include tyrannosaurines like T. rex, though T. rex itself was not a bird

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Not all snakes can swim

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3, d4
- **Supporting Docs Found**: d5
- **Claim**: Current regulatory practice in the U.S. treats glyphosate as minimally absorbed into the body and primarily excreted unchanged, with permitted exposure levels set more than one hundred times lower than the amount that might cause harm , though some researchers and advocacy groups call for further regulation and stricter safety standards

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, the PETM onset also coincided with a mercury low, suggesting at least one other carbon reservoir was simultaneously released in response to initial warming , making the full causal picture more complex than a single trigger mechanism

### Sample conflictingqa_b7fd50f9f980

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: 'alright' is a valid spelling the Oxford Dictionary notes there is no logical reason to insist on writing it as two words since other merged forms like 'already' and 'altogether' are widely accepted

### Sample conflictingqa_bac0f4d62f96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Some researchers estimate that cometary meteoroids make up about 95% of observed meteors and 38% of observed fireballs, though no conclusive evidence links any specific meteorite to a particular comet due to the near-impossibility of preserving cometary material intact through atmospheric entry

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4, d1
- **Supporting Docs Found**: None
- **Claim**: A study by the Radio Project found that less than one-third of panicked listeners mistook the Martians for German invaders or a natural catastrophe historian W. Joseph Campbell has noted that the narrative of mass panic was primarily advanced by newspapers at the time as a means to discredit radio as a credible news source

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d2, d4, d1
- **Claim**: It is particularly valuable for vegans because it is high in B12, which is otherwise difficult to obtain from plant-based diets also provides 5–6 grams of protein per 100g

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, Sega officially denies his involvement some sources treat the claim as unconfirmed speculation, making the matter unresolved

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: However, research indicates that dry grounds alone are only partially effective, with a University of Nebraska study finding snails are deterred when the soil contains a caffeine concentration of more than 0.1% are even killed at 1% or above

### Sample conflictingqa_c418fecfc1e2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Some plants can survive without sunlight for short periods, such as during a prolonged night or in very dark indoor environments, as the RHS confirms

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
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

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: Symptoms such as eye strain, dryness, headaches blurred vision are common some researchers note that prolonged use may worsen conditions like convergence insufficiency

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Black holes themselves are invisible, but their effects can be observed with telescopes

### Sample conflictingqa_fbedb688b1d9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, scholars and some religious observers differ: Robert Millet, a Brigham Young University professor, argues that Mormons should be considered Christians because they worship Jesus Christ as the Son of God, though the official definition of Christianity emphasized correct belief over mere good works or social conservatism

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
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani (2014 — first and only female recipient; passed away 2017)

### Sample freshqa_25b286cb2af1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2
- **Claim**: The idea that Venus could have had a moon in the past is also discussed, as scientists are not entirely sure why Venus lacks a moon today

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d3
- **Claim**: It is worth noting that while d3 lists several children who died in the disaster, it does not explicitly state the exact age of the youngest passenger, which is confirmed to be Millvina Dean through multiple corroborating sources

### Sample freshqa_64c12116affc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 2026, the President of the United States is Joe Biden, who is serving his first term in office alongside Vice President Kamala Harris. This is corroborated by the official White House history, which confirms that the 117th Congress coincided with the Biden administration from January 20, 2021 to January 3, 2025 the 118th Congress continues under his presidency

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: This result, with the ceremony also seeing wins for Michael B. Jordan in Best Actor and Autumn Durald Arkapaw as the first female cinematographer to win an Oscar

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Luka Modric was the only player to beat Ronaldo and Messi in 2018, further contextualizing Kaka's win as the one immediately preceding the supercycle

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Bayonne, New Jersey

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

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4, d1
- **Claim**: Queen Elizabeth II of England died on 8 September 2022 at Balmoral Castle in Aberdeenshire, Scotland

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

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: No permanent cure for cancer has been developed; however, several milestone treatments have been achieved

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
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

### Sample hotpotqa_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The song initially reached number 17 on the Modern Rock Tracks chart before achieving mainstream success, peaking at number 51 on the *Billboard* Hot 100 and number one on the Modern Rock Tracks chart

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6
- **Supporting Docs Found**: d3
- **Claim**: English cartographer and historian John Speed (1551–1629), best known as a mapmaker of the Stuart period

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d7, d8, d2, d6, d4, d5
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not support a confident answer to the query

### Sample qacc_0ac549afb037

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: While the name shares linguistic roots with German Hans and Dutch Jan, its most direct derivation traces back to the Danish masculine given name Hans

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5, d4
- **Claim**: The retrieved evidence indicates that following the liberation of North Africa, Allied forces advanced into Tunisia, where they encircled and defeated approximately 250,000 German and Italian troops, ending the battle for North Africa by mid-1943

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d4
- **Claim**: This success paved the way for the Allies to continue their campaign in Italy, as Axis forces were expelled from the African continent by late 1943

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d4, d5, d3
- **Claim**: Sakshi Malik (Haryana); Madhuri Dixit (India); Parineeti Chopra (Haryana); Avani Lekhara (Rajasthan); Bhawna Dehariya and her daughter Siddhi Mishra (Madhya Pradesh)

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Wood Harris (Ace), Mekhi Phifer (Mitch), Cam'ron (Rico), Kevin Carroll (Calvin)

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
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: The gesture of crossing fingers for good luck has its roots in pre-Christian European pagan beliefs, where the cross shape was thought to concentrate good spirits and serve as a focal point for anchoring wishes. Over time, this practice evolved into a one-handed habit was further associated with early Christianity — specifically, as a secret sign (L-shaped fingers touching) used by early Christian followers to identify and gather with one another during a time when their faith was persecuted. The gesture was then simplified to the familiar index-finger-over-middle-finger form came to be associated with invoking divine protection rather than merely seeking luck

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: As a coach: Red Auerbach (16 championships as a coach and executive, though the query asks about players vs. coaches broadly so this is partial support) — as a player: Bill Russell (11 championships as a player, the most in NBA history) —

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: Lacteals

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 180

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: October 1968

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
- **Supporting Docs Found**: d2
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
- **Supporting Docs Found**: d4, d3
- **Claim**: Roger Miller

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Notable bearer Christopher Tavarez is an American actor

### Sample qacc_cbf25273f973

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Surname origin data indicates Spanish-Portuguese ancestry predominates among people with this surname

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Japan: 1996; in the US, January 9, 1999

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: In Mexico, toll roads are commonly called **autopistas**

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2025/26

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4
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

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This form of government is further characterized by the principle of checks and balances, ensuring no single branch accrues too much power by the requirement that all State governments maintain a republican form, though not necessarily the three-branch structure

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5
- **Claim**: Looking at longer-term trends, Pew's 2024 analysis shows that since 1965, about half of U.S. immigrants have come from Latin America and about a quarter from Asia , reflecting a gradual shift in the bulk of immigrant origins away from Europe and toward the Americas and Asia

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a California grizzly bear (Ursus arctos californicus), which is also known as the California brown bear or California golden bear

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

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

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
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Gagan Narang (10m Air Rifle)

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida (won 2025)

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: Android 16 (Baklava).

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d4
- **Claim**: Syndicated comics based on Avatar: The Last Airbender are coming out in 2025–2026, though the specific next title depends on which series you are asking about

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken (2026–present)

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

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: It is classified as a lysosomal storage disorder because the missing enzyme prevents the breakdown of GM2-ganglioside within cells

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The group was formed in Los Angeles in 1989 and quickly rose to fame with hits such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Shay Mitchell, who portrays Emily Fields in Pretty Little Liars, was born in November 1993, making Emily 31 years old according to the show's current timeline

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: The empire officially ended when Francisco Pizarro captured the last Sapa Inca, Atahualpa, at Cajamarca on November 16, 1532, though the remnants of the empire persisted briefly in Vilcabamba before being fully conquered by 1572

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

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Additional historical sources corroborate this date, confirming that the conflict took place in the same year as Ramesses II's fifth campaign

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d1
- **Claim**: Earlier in their history, the 76ers also reached the 2001 NBA Finals, where they lost to the Los Angeles Lakers, representing their most recent championship appearance at the time

### Sample trust_align_076

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d5
- **Claim**: The mechanism involves both direct toxic effects on the heart muscle and cardiovascular system, as well as chemical-induced spasms of the coronary arteries, which can trigger fatal heart failure within seconds of inhalation

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Twitter is currently known as X. The platform was rebranded as X in November 2025, following its merger with X Holdings in April 2023. This supersedes its former name, which dated back to its founding in 2006

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

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore's official name was changed to Bengaluru. This change was confirmed by the Karnataka Government, making Bengaluru the official name of the city, while Bangalore remains the commonly used name

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

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: 2022, Bongbong Marcos is the President of the Philippines, serving as both head of state and government and commander-in-chief of the armed forces. He is also the incumbent President of the Senate, serving as the third highest and most powerful official in the government

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The men's singles champion is Carlos Alcaraz, who successfully defended his title by defeating world No. 1 Jannik Sinner in the final. This victory marked his second French Open title and fifth major


================================================================================

*Report generated by CATS v2.0*
