# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 122 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.977 (over 736 samples)

**GR F1** *(used in CATS)*: 0.986

**Behavior Adherence**: 0.813 (over 614 applicable samples)

**Factual Grounding**: 0.878 (over 614 applicable samples)

**Single-Truth Recall**: 0.735 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.853

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.986
- **Precision**: 0.990
- **Recall**: 0.982
- **Accuracy**: 0.977
- TP=597, FP=6, FN=11, TN=122

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.917
- **Abstain Recall**: 0.953
- **Abstain F1**: 0.935
- **Specificity**: 0.982
- Abstain TP=122, FP=11, FN=6, TN=597


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (53 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.956 (n=158)
- **Grounding**: 0.942 (n=158)
- **Recall**: 0.799 (n=154)
- **CATS**: 0.921

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.968
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.921 (n=177)
- **Grounding**: 0.856 (n=177)
- **Recall**: 0.651 (n=156)
- **CATS**: 0.852

### Type 3: Conflicting Opinions

- **Samples**: 109 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.567 (n=97)
- **Grounding**: 0.833 (n=97)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.795

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.690 (n=145)
- **Grounding**: 0.902 (n=145)
- **Recall**: 0.750 (n=140)
- **CATS**: 0.835

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.946
- **GR F1** *(used in CATS)*: 0.972
- **Behavior**: 0.811 (n=37)
- **Grounding**: 0.730 (n=37)
- **Recall**: 0.770 (n=37)
- **CATS**: 0.821


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1880

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

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Therefore, while weight lifting is generally considered safe and beneficial for most people, individuals with high blood pressure should exercise caution, consult their healthcare provider consider modifications such as lifting lighter loads or avoiding the Valsalva maneuver

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Dog breeding is not universally unethical, but some practices are

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d5
- **Claim**: Overall, the evidence is mixed the widely held belief that dairy increases mucus appears to stem partly from the fact that milk can create a sticky coating in the mouth and throat, rather than from actual increases in mucus

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Not directly from chlorine — chlorine actually lightens hair, but copper (from algaecides or tap water) oxidizes and binds to hair proteins, causing the green discoloration

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d5
- **Claim**: It depends on whether the artificial tree is used for at least 20 years; otherwise, real trees are more sustainable

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: However, other sources argue that trophy hunting is morally inappropriate, that evidence of genuine conservation benefits is inconclusive given the complexity of alternative revenue models and the industry's own shortcomings that the IUCN's position itself reflects ongoing scientific debate ; researchers have also criticized the trophy hunting industry for failing to demonstrate that revenues genuinely translate to meaningful conservation outcomes in practice

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are granted in many jurisdictions and can protect core algorithms and functions, while others argue that software patents are too broad, stifle innovation fail to meet legal standards for patentability

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2
- **Claim**: These conflicting findings suggest that any protective effect of bicarbonate may be conditional on disease stage, baseline bicarbonate levels dosage the overall question remains unresolved

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d3
- **Claim**: Yes

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: It depends on jurisdiction; ISPs in the U.S. generally must obtain consent under California's CPPA and Maine's law, while federal law (S.J. Res. 34) otherwise allows selling with anonymization required

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Multiculturalism's relationship with national unity is contested in the evidence: some sources argue it acts as a barrier to civic cohesion by amplifying cultural differences , while others contend it actually facilitates political and civic integration

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5, d3
- **Supporting Docs Found**: d2
- **Claim**: On the other hand, neutering/spaying clearly provides significant benefits such as preventing certain cancers (ovarian, breast, prostate) and hormone-induced diseases like pyometra some sources note that the procedure's advantages generally outweigh the risks when performed appropriately ; the overall picture is therefore complex and depends on factors such as age, sex breed

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d1
- **Claim**: In practice, speleologists and geologists hold differing views on whether structures found underwater should be classified as stalactites if they formed in air before being submerged

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: Current CO2 levels are not unprecedented in Earth's history

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Religious and theological views differ; science offers competing models

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Religious views differ; there is no single factual answer

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: However, claims of widespread or easy manipulation are contested, with participants and observers differing on how common or impactful such practices truly are

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: It depends on the fictional universe; folklore and older traditions vary

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes — barefoot running is healthier

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence presents competing views. The瑞⼠契科夫方法要求仅使用检索到的文档证据来回答问题。

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Emoji serve as a visual supplement to written language but do not constitute a distinct written language themselves

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not confirm that the Dutch were the sole or original discoverers of Australia, as prior claims exist and the exact nature of the continent's 'discovery' remains a subject of historical debate

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Traditionally, Apatosaurus and Brontosaurus were considered the same dinosaur because fossils initially labeled as Brontosaurus were later determined to belong to the same genus as Apatosaurus under taxonomic rules, the earlier name Apatosaurus took precedence

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Prince Harry was not stripped of the Duke of Sussex title by King Charles III. The Duke of Sussex title was granted by Prince William, not King Charles has not been revoked

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: August 16, 1977

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: The site is now part of the White Sands Missile Range and is owned by the U.S. Department of Defense, with ground zero marked by a black lava rock obelisk

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: No chemical reaction between lead and any other element produces gold as a byproduct; gold cannot be created from lead via chemical reactions because chemical reactions only change electron configurations, not the nuclear proton count

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: $130

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_97f3c1fe1fd4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is corroborated by additional context showing the team missed the playoffs that season and continued to struggle in the subsequent 2024-25 season

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: Jeff Bezos did not sell Amazon. He is the founder and largest shareholder, having sold only shares worth ~$737 million in late June 2025 while remaining chairman

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This figure encompasses principal photography, reshoots, post-production on-set costs, though it excludes the global marketing campaign

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample qacc_08cf866bcb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Nomura

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The retrieved evidence indicates the Allies subsequently moved eastward into Italy, with the text stating the invasion of Sicily was the logical next step after North Africa

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: These distinct tournaments reflect different cricket formats (ODI vs. T20) and separate years of triumph, collectively forming a complete picture of India's World Cup achievements

### Sample qacc_213701765f94

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The aircraft came to rest near 130th Street in Manhattan passengers and crew waited on the wings until rescued by ferries

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

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: Canada's independence from Great Britain was a gradual process marked by several key milestones rather than a single date. The Dominion of Canada was formed on July 1, 1867, as a self-governing entity within the British Empire this is often cited as the effective date of independence. Further milestones included the Balfour Declaration of 1926, which recognized Canada as an autonomous community within the Empire the Statute of Westminster of 1931, which gave Canada full legislative independence. Finally, the 1982 Canada Act fully severed Canada's last constitutional ties to Britain by transferring authority over the Canadian constitution to Canada itself

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: October 1968

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The planned completion date for the Sagrada Familia has been updated to the early 2030s, superseding the earlier target of 2026. The basilica's construction board refused to give a precise date due to pandemic-related uncertainties, though rumors suggest the last towers could be finished by the early 2030s, potentially coinciding with the 150th anniversaries of the construction's start in 2032 and Gaudí's involvement in 2033

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Celebrity Big Brother is currently available on CBS in the USA

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Remarkably, no one was injured in the fire the following Christmas the Hoovers welcomed staff and children back to the White House once again

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_bf0bff050f03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: England

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The aircraft was named after Enola Gay Tibbets, the mother of the mission's pilot, Colonel Paul Tibbets

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: More specifically, it is often described as type SBbc, placing it between SBb and SBc, which corresponds to having moderately wound spiral arms and a moderate-sized central bar

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: ICD-10 codes have a flexible structure comprising three to seven characters

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Nassau County, NY

### Sample situatedqa_geo_362420ec2416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that at least one source also lists Mizoram as the second least populated state, further corroborating Sikkim's position at the bottom that the 2011 Census data for Sikkim aligns with broader demographic trends without contradiction

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3
- **Claim**: Historically, the bulk of U.S. immigrants came from Europe, but starting in 1965 Latin America became the dominant source

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: This grizzly bear has served as a symbol of strength and resistance since 1846, when American insurgents captured Sonoma and raised the original 'Bear Flag' during the Bear Flag Revolt

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Kiren Rijiju, who serves as the Cabinet Minister for Law and Parliamentary Affairs. He is a senior Bharatiya Janata Party (BJP) leader and has held the portfolio since 2019, overseeing the delivery of legal reforms and parliamentary affairs. This is consistent across multiple sources, including the official Punjab government website and Wikipedia

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: British troops set fire to the White House on August 24, 1814, during the War of 1812

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d1
- **Claim**: This victory opened the way for the British conquest of Philadelphia and was the largest single-day battle of the American Revolutionary War in terms of manpower

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 110

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: It was first made available to Google Pixel phones before rolling out to other manufacturers like Samsung, with Vivo also receiving the update

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: Using the 'circle of fifths' method described by some sources, you can also determine that moving clockwise from C major through five consecutive fifths lands on B major, confirming the same result

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: No astronauts have returned to the moon since then the next planned lunar landing — NASA's Artemis III mission — is currently targeted for 2028

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Cardiac biomarkers are substances that enter the bloodstream when the heart is damaged or stressed they are used to diagnose and monitor heart conditions

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Rhys Ifans

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

### Sample trust_align_079

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Clerck

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held on 19 November at the Narendra Modi Stadium in Ahmedabad. This was Australia's sixth Cricket World Cup title the 2023 tournament was the 13th edition of the ICC Men's Cricket World Cup, hosted entirely in India from 5 October to 19 November 2023. Multiple sources confirm that India was the 2023 runner-up the next scheduled tournament is the 2027 ICC Men's Cricket World Cup in South Africa, Zimbabwe Namibia

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was confirmed when the Gurgaon Municipal Corporation officially changed the city's name to Gurugram, effective from April 1, 2016. As a result, all official references to the city now use the name Gurugram, though Gurgaon is still commonly used in everyday speech and some legacy contexts

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This name change was confirmed by the Government of Haryana in 2016 the city is now officially known by this new name. As a result, Gurgaon is no longer the official name of the city


================================================================================

*Report generated by CATS v2.0*
