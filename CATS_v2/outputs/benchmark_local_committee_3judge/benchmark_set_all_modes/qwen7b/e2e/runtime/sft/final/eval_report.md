# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 121 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.966 (over 736 samples)

**GR F1** *(used in CATS)*: 0.979

**Behavior Adherence**: 0.774 (over 615 applicable samples)

**Factual Grounding**: 0.829 (over 615 applicable samples)

**Single-Truth Recall**: 0.693 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.819

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.979
- **Precision**: 0.988
- **Recall**: 0.970
- **Accuracy**: 0.966
- TP=590, FP=7, FN=18, TN=121

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.871
- **Abstain Recall**: 0.945
- **Abstain F1**: 0.906
- **Specificity**: 0.970
- Abstain TP=121, FP=18, FN=7, TN=590


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (53 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.967
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.911 (n=158)
- **Grounding**: 0.886 (n=158)
- **Recall**: 0.760 (n=154)
- **CATS**: 0.884

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.964
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.961 (n=178)
- **Grounding**: 0.820 (n=178)
- **Recall**: 0.622 (n=156)
- **CATS**: 0.845

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.995
- **Behavior**: 0.479 (n=96)
- **Grounding**: 0.878 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.784

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.616 (n=146)
- **Grounding**: 0.783 (n=146)
- **Recall**: 0.739 (n=140)
- **CATS**: 0.781

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.685 (n=37)
- **Recall**: 0.541 (n=37)
- **CATS**: 0.707


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2047

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
- **Claim**: Fertilizer type further modulates these effects—organic fertilizers tend to increase free-living nematode populations and nutrient availability, whereas inorganic fertilizers can negatively affect nematode communities, particularly plant-parasitic species

### Sample conflictingqa_05b33f4ca156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: Handling any salamander is not recommended, as all 600+ species carry varying levels of poison and ingestion or absorption of their toxins can cause numbness, dizziness, muscle weakness in severe cases paralysis

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Yes — anime is a specific style of cartoon, originating in Japan with distinct artistic and narrative characteristics

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: It is worth noting that some scholars and commentators further distinguish between religious observance and ethnic identity, recognizing that many Jews are irreligious or post-denominational that Jewish cultural practices vary significantly across geographic and demographic subgroups

### Sample conflictingqa_1d741779a31e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Preclinical and some clinical studies suggest artificial sweeteners may raise the risk of cardiovascular events, incident stroke all-cause mortality in the general population

### Sample conflictingqa_1d741779a31e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, research specifically focused on people with diabetes presents a more nuanced picture: a large cohort study found that high intake of artificial sweeteners was associated with higher all-cause mortality, cardiovascular risk cancer risk in diabetic patients, while lower intake was not consistently linked to harm

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Proponents of responsible breeding argue that it is necessary to maintain unique dog breeds and that ethical practices minimize harm through stringent health screening and humane treatment of breeding dogs. Opponents contend that breeding inherently exploits animals for profit, causes widespread inherited health conditions contributes to pet overpopulation

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Yes, but usually only in plants — in animals, the germline undergoes reprogramming during meiosis that erases most epigenetic marks, though some marks (maternal DNA methylation in mammals) can escape reprogramming, leading to parental imprinting

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1
- **Claim**: However, the amount of data needed varies significantly depending on the specific problem, model complexity the quality of features — with some smaller ML models capable of performing well even with modest datasets

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: In cases where data is extremely limited, specialized techniques such as transfer learning or active learning can help bridge the gap, though these approaches add additional computational and engineering challenges

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The question of whether patents should apply to software is actively debated. A high-credibility source notes that while software patents are technically patentable under U.S. law, recent Supreme Court rulings make them difficult to enforce, leading many inventors to conclude that the costs outweigh the benefits

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Some clinical trials have shown bicarbonate may reduce urinary fibrotic biomarkers and preserve eGFR in early-stage CKD, particularly in patients with hypertension nephropathy, while higher-dose regimens have been associated with reduced potassium concentrations and only a catabolic state improvement in some reports

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: However, many products can temporarily make split ends appear better by smoothing the cuticle, adding weight to frayed ends creating temporary bonds between split fibers — though these effects typically last only through one shampoo

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d1
- **Supporting Docs Found**: None
- **Claim**: However, some states have pushed back against this: Maine passed a law prohibiting ISPs from selling personal data without express individual permission the California Consumer Privacy Act (CCPA) grants California residents the right to opt out of having their data sold

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Proponents of the idea that fish do feel pain point to the fact that they have pain receptors (nociceptors) that are activated by noxious stimuli that they exhibit behavioral responses such as attempting to avoid harmful situations

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Underwater stalactites are possible but require specific conditions — they form when water pressure pushes calcite-laden water up through a hollow stalactite structure and then deposits the calcite radially, rather than the typical dripping process — and are rarely observed

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: The claim is further supported by the fact that once hair has grown past the scalp, it is technically dead tissue that neither contains blood vessels nor nerves, meaning it cannot react to changes in water temperature

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2
- **Claim**: Merriam-Webster further notes that the one-word spelling is first recorded in the late 19th century and carries no logical grammatical objection, though it remains nonstandard in formal contexts

### Sample conflictingqa_bdee100fa8e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: Wikipedia notes that 'all the hard evidence suggests there was no mass exodus,' and that newspapers likely exaggerated the scale of the panic to discredit radio as a news source, while some witnesses reported thousands of people fleeing their homes in genuine terror

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beliefs differ by denomination and interpretation

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Foundationalist accounts attempt to resolve the regress by positing basic beliefs justified through experience or sensory evidence, though critics reject this as implausible

### Sample conflictingqa_e93e708d49a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, the exact energy balance depends on factors such as climate, system size usage patterns — early studies estimated a nine-to-one energy return ratio, while a more recent Royal Society of Chemistry study found daily ratios ranging from seven to 27 in different U.S. states when surplus power is fed into the grid

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: The play has been plagued with accidents ever since — including the 1937 production at the Old Vic where Laurence Olivier nearly lost his life when a 25-pound stage weight crashed near him

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, the evidence does not confirm that the Dutch were the sole or definitive discoverers of Australia, as other European powers would later challenge their claims

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some research offers a more nuanced picture: a laboratory study found that yerba mate exhibits a cytotoxic effect on cancer cells, suggesting it may possess anti-cancer properties, though these findings do not translate to a proven preventive or therapeutic role in humans

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: Additional evidence from population studies has demonstrated increased rates of esophageal, head and neck bladder cancers among habitual yerba mate consumers, though researchers note that the exact causal mechanisms remain unclear and some studies have limitations in applying animal-based findings to human health

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1
- **Claim**: No, Brontosaurus and Apatosaurus were not the same dinosaur

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: While the original promotional plan was for 50,000 attendees, tens of thousands more showed up, creating a spontaneous, inclusive moment that transcended its initial scope

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: Aryna Sabalenka (2025 US Open women's singles champion); Jessica Pegula (2024 US Open women's singles finalist)

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Maryam Mirzakhani (2014 — first and only female recipient; passed away 2017)

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: In summary, the latest major version of .NET depends on which edition (Framework, Core Mono) you are considering

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d5, d1
- **Claim**: The first atomic bomb test, called Trinity, took place in New Mexico's Jornada del Muerto desert, at the Trinity site, approximately 210 miles south of Los Alamos, on July 16, 1945

### Sample freshqa_3cd0b514193b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Employers in Tokyo must pay this rate or higher to comply with Japanese labor laws

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Season 1 premiered on November 12, 2019, Season 2 on October 30, 2020 Season 3 on March 1, 2023

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5eb89aae15f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Kantara (Rishab Shetty, 2022)

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: However, older sources may reflect Joe Biden's prior presidency (2021–2025) some contemporary sources still reference Biden's tenure as recently as 2026 , underscoring that the most current information places Trump as the incumbent

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has not won the Ballon d'Or

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: His passing dealt a significant blow to the emerging field of artificial intelligence, as Rosenblatt was the primary defender of the perceptron concept without his advocacy, funding dried up for over a decade

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Her output has steadily grown over her decade-long career, with her novels continuing to appear on the New York Times bestseller lists

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: J June 2025

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Jiangsu

### Sample freshqa_c7ac9d61059a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, this figure comes from a 2026 pricing article covering the 2027 model year, so current prices may differ

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: There is no permanent cure for cancer

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f11e8d758fc6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Some sources also point out that certain aquatic and marine slug families have evolved gill-like structures within their lungs, further illustrating the remarkable respiratory adaptations of these creatures

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: Pfizer's vaccine is approved for people ages 5 and older Novavax's vaccine is authorized for ages 12 and older

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: In addition, the Washington State Department of Health notes that pediatric vaccine studies follow patients' health up to two years after vaccination the CDC recommends that children ages 6 months to 17 years may receive the vaccine based on parent preference and clinical judgment

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: These divergent findings reflect methodological differences between a single RCT and a broader meta-analysis the weight of clinical opinion remains unsettled

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d6, d8, d7
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not support a confident answer to the query

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: The Allies went on to invade Tunisia (1943), where they surrounded and captured 250,000 German and Italian troops, effectively ending the Axis presence in North Africa

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d3, d2
- **Claim**: Sakshi Malik (Haryana); Madhuri Dixit (India); Parineeti Chopra (Haryana); Bhawna Dehariya and her daughter Siddhi Mishra (Madhya Pradesh); Avani Lekhara (Rajasthan)

### Sample qacc_15ffab2466f7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: For the latest or upcoming productions, Toronto-based ticketing platforms such as SeatGeek and Ticketmaster are useful resources to check current availability across various venues

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Some historians also point to a separate origin tied to early Christian secret signs — in which members would form an 'L' with their thumb and index finger — that may have been simplified into the familiar crossed-finger shape during the Hundred Years' War

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: As a coach: Red Auerbach (16)
As a player: Bill Russell (11)
Combined: Phil Jackson (11 as a player + 11 as a coach)

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Canada did not gain full independence on a single date; rather, it was achieved gradually through key milestones

### Sample qacc_6837d86d03ac

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d5
- **Claim**: Following Prince William are his children: Prince George of Wales (2nd), Princess Charlotte of Wales (3rd) Prince Louis of Wales (4th)

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
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: October 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Australian Shepherd

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The tower's completion made the basilica the tallest building in Barcelona and the tallest church in the world, standing at a final projected height of 172.5 meters

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that different organs hold water in varying amounts — for example, the brain and heart are about 75% water muscles and kidneys are around 79% — but the majority of the body's water is still sequestered within the cellular structure

### Sample qacc_aa94588b9477

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2, d5
- **Claim**: The bank gradually assumed central banking functions over the following decades these responsibilities were formalized under the Commonwealth Bank Act 1945

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Syndicated internationally; in the US, streamed on Paramount+

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_bfbb5f55a63f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Timothy Talbott

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
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: Notable bearer Christopher Tavarez is an American actor, illustrating the name's adoption across the Americas

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2025/26

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: The First Epistle of John was likely written between 95 and 110 C.E., though some scholars place the composition as early as 85–90 C.E

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: For those planning to move there, the community is accessible via Florida state routes FL-44, US-27 US-301, as well as an Amtrak bus stop

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Across the broader UK jurisdiction, the law strongly recommends that children maintain an alcohol-free childhood, with experts suggesting that even supervised drinking for under-18s should not occur until at least 15 years old

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Other national estimates include Poland at 5.6–5.8 million, Germany at 5.3–6.6 million Japan at 2.1–3.1 million military deaths, with additional civilian deaths in Japan estimated at 580,000–1.3 million

### Sample situatedqa_geo_4cb699778b59

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Current average Senate service stands at approximately 11.2 years as of the 119th Congress (2025–2026), though this reflects cumulative service duration rather than the official statutory term length, which remains six years

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This is further confirmed by the official U.S. government website, which states that all State governments are required to uphold a "republican form" of government, though not necessarily the three-branch structure

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Mexico has been the largest individual origin country for immigrants coming to the U.S., with Pew Research data showing Mexican immigrants made up about 22% of the total immigrant population in 2023

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5, d1
- **Supporting Docs Found**: None
- **Claim**: This landmark legislation replaced the earlier 1963 act, creating the Environmental Protection Agency and establishing national air quality standards .
- Subsequent amendments were made in 1977 and 1990 to strengthen the act's provisions as recently as June 30, 2021, President Biden signed Public Law 117-23 to address methane emissions in the oil and gas sector

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: The California state flag features a California grizzly bear (also known as the California brown bear or California golden bear), which is an extinct subspecies of the brown bear. This grizzly bear served as the basis for the original Bear Flag of 1846, which was adopted as California's state flag

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: 2018

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The switch from tea to coffee in the United States was a gradual process tied to historical events rather than a single definitive date

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The Philadelphia Eagles have won the Super Bowl multiple times. The most recent was Super Bowl LIX (59th) in 2025, when they defeated the Kansas City Chiefs. Earlier, the Eagles won Super Bowl LII (52nd) in 2018, earning their first NFL title since 1960 added another championship with a 40-22 victory over the Chiefs in Super Bowl LVII (57th) in 2023

### Sample situatedqa_temp_1c56e575f096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: Houghton Lake, Torch Lake Lake Charlevoix

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: New South Wales last won the State of Origin series in 2021

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This total reflects over two decades and 1,622 games in the NBA, making James one of the sport's all-time greats

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is a new collection of original canon stories expanding the world of Avatar, serving as a direct prequel to *Avatar: The Way of Water*

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: 1980 (established as a national park); previously designated as a national monument in 1978

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Dragon Ball Z episode 245 (Original ToP) / 246 (Blue ToP) — This is the first time Goku becomes Super Saiyan 3 in the main series

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Cleveland Browns head coach: Todd Monken (2026–present)

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d2, d1
- **Claim**: Since the battle itself is the primary focus of the query, I will highlight the most direct examples from the provided evidence:
- “Corpse-maker” (21, lines 286)
- “Shadow-stalker” (47, lines 704)
- “Shepherd of evil” (a40)
- “Whale-road” for the sea
- “Bone-house” for the human body (though not specific to the battle itself)
- “Sea-wood” for ‘ship’ (used in the journey to Grendel’s lair)

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: For context, this places Australia among the top-ranked countries globally for coastline length, alongside nations like the United States (approximately 19,924 kilometres) and New Zealand (15,134 kilometres)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2, d5, d1
- **Supporting Docs Found**: d3
- **Claim**: While d3 offers a partial view of the river's course through Nashville, the full extent of the river's headwaters and final destination spans across multiple states, as detailed by the more comprehensive sources

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The group was formed in Los Angeles in 1989 and quickly rose to fame with hits such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: These discrepancies are typical of religious organizations experiencing growth, as successive membership tallies supersede prior records without a single authoritative source for all time periods

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Tsinghua University further clarifies that the revolution was driven by intellectuals and students demanding constitutional reforms, who were inspired by the 1905 Russian Revolution

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Shay Mitchell was 23 when she portrayed 16-year-old Emily Fields in the show's pilot episode, meaning Emily's fictional age differed from her real age throughout the series

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 670–680 nm

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The first U.S. Olympics were the 1904 Summer Games in St. Louis, Missouri, which were actually part of the St. Louis World's Fair

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: 2020

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is reflected in the broader 2024 GPI dataset, which notes that India maintained its world ranking compared to the previous year

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, newer 2025 data references suggest India's standing has improved further, with the 2025 index showing only partial rankings for some categories

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: In medieval times, the surname was first recorded in the Domesday Book of 1086 as Gerardus and Girardus, reflecting the Latin form prevalent at that period

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: Over time, the name spread to other regions including England, where it is associated with the family of Gerard (died 21 May 1108), who served as Lord Chancellor of England and later became Archbishop of York

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The WTO has 166 members

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This film, produced by GMA Pictures and Star Cinema, surpassed the previous record holder, _Rewind_, just 10 days after its release

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc. and is structured as a wholly owned subsidiary of Alphabet

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore is officially called Bengaluru. Bengaluru is the official name of the city, which was previously known as Bangalore until 1 November 2014. This change was confirmed when the Government of Karnataka issued a notification renaming the city from Bangalore to Bengaluru

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The men's singles champion is **Carlos Alcaraz**, who defended his title by defeating world No. 1 Jannik Sinner in the final. This victory marked his second French Open title and fifth major

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The President of Germany is Frank-Walter Steinmeier, who has served as the Federal President of the Federal Republic of Germany since taking office on that date. This is confirmed by the official Wikipedia entries on both the President of Germany and the Federal President, which list his incumbency from 19 March 2017. Since German reunification in 1990, the President has been the head of state for all of Germany, serving as the Bundespräsident der Bundesrepublik Deutschland

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022. He is the 31st person to hold the office since its creation in 1901

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Jannik Sinner (most recent data available indicates he is the champion, though the tournament has since taken place and he may have been succeeded)

### Sample wikirevision_0150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Jannik Sinner (2025)

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The current President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022. He is the 16th President of the Philippines and serves as both head of state and head of government

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current President of India is Droupadi Murmu, who took office on 20 July 2022. She is the 15th and current President of India, serving as head of state and supreme commander of the Indian Armed Forces. While the older Wikipedia revision (2025) also confirmed her presidency, the newer 2026 revision consistently reinforces that Droupadi Murmu is indeed the current President of India

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed when the Haryana government passed a resolution changing the city's name officially to Gurugram, though the name Gurgaon is still commonly used

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 FIFA World Cup is upcoming and Argentina is the defending champion


================================================================================

*Report generated by CATS v2.0*
