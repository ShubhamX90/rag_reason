# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.933 (over 736 samples)

**GR F1** *(used in CATS)*: 0.958

**Behavior Adherence**: 0.786 (over 608 applicable samples)

**Factual Grounding**: 0.792 (over 608 applicable samples)

**Single-Truth Recall**: 0.627 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.791

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.958
- **Precision**: 1.000
- **Recall**: 0.919
- **Accuracy**: 0.933
- TP=559, FP=0, FN=49, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.723
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.839
- **Specificity**: 0.919
- Abstain TP=128, FP=49, FN=0, TN=559


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.929 (n=154)
- **Grounding**: 0.869 (n=154)
- **Recall**: 0.769 (n=154)
- **CATS**: 0.887

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.914
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.903 (n=176)
- **Grounding**: 0.762 (n=176)
- **Recall**: 0.526 (n=156)
- **CATS**: 0.784

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.521 (n=96)
- **Grounding**: 0.824 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.774

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.911
- **GR F1** *(used in CATS)*: 0.949
- **Behavior**: 0.731 (n=145)
- **Grounding**: 0.766 (n=145)
- **Recall**: 0.650 (n=140)
- **CATS**: 0.774

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.541 (n=37)
- **Grounding**: 0.635 (n=37)
- **Recall**: 0.378 (n=37)
- **CATS**: 0.616


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2037

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
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Side effects are mostly mild, including dry mouth, dizziness photosensitivity, though serious side effects such as mania can occur with use of the combination of St. John's wort and SSRIs

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: High-intensity lifting can also trigger a Valsalva maneuver, which further raises blood pressure, though this is considered dangerous for those with uncontrolled hypertension

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The poem itself explores themes of madness, sacrifice rebellion against conformity Ginsberg described it as a call to freedom of emotional expression

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d1
- **Claim**: The overall picture is that peeling does reduce some nutrients but does not entirely remove all nutritional benefit

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d1, d2
- **Claim**: The answer depends on how you count them: cows have four stomachs if you combine all four compartments (rumen, reticulum, omasum abomasum), but they technically have only one stomach if you treat each compartment as a separate organ

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Yes, hair can turn green from swimming pools — but not directly from chlorine itself. The real culprit is copper, which is commonly found in algaecides used to control algae growth in pools. When copper oxidizes (exposed to air), it turns from shiny orange to a dull green when it comes into contact with hair, it causes green discoloration. Chlorine actually works to lighten hair, not turn it green — it is the presence of hard metals like copper, iron manganese in pool water that oxidize and stick to hair, causing the green color

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Some sources, including a 2022 survey of 1,000 students, even found that 82% of respondents considered audiobooks equal to or more engaging than traditional reading

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Overall, experts advise that getting omega-3 fatty acids from fish directly is far more beneficial for heart health than relying on supplements alone high-risk individuals should discuss fish oil supplementation with their doctor

### Sample conflictingqa_411445406724

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Cycads themselves were outcompeted by the flowering plants (Angiosperms) more than 100 million years ago, making them essentially obsolete as ecologically dominant species on land

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: On the other hand, critics argue that the practice is morally inappropriate, involves unnecessary animal cruelty that the revenue benefits are too often concentrated among elite hunters rather than directly protecting wildlife

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Some scientists further contend that trophy hunting disproportionately targets males or specific species, potentially threatening their long-term health and population stability , while others note that it is often justified on consequentialist grounds without sufficient evidence of net conservation benefit

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, a separate study found that a lower dose (0.5 mEq/kg/day) did not significantly reduce urinary transforming growth factor-beta (TGF-β) excretion or slow CKD progression in patients

### Sample conflictingqa_57190bca6f7a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: Yes; the 1815 eruption of Mount Tambora is widely considered the deadliest and most destructive in recorded human history

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, some states have en

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The debate reflects a methodological divergence between defining 'efficiency' as raw yield (conventional farming) versus defining it more broadly to include environmental and health impacts (organic farming), making it a contested question with no single definitive answer

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Religious truth claims are inherently contested, but the Catholic Church presents itself as the one true church by virtue of its unbroken apostolic succession and adherence to Christ's teachings

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the extent to which each individual bird's call is unique has not been thoroughly studied some research suggests that individual recognition of calls varies considerably

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: In summary, the scientific consensus is that glyphosate poses little risk when used properly

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: These soda straws can thicken and eventually develop into the familiar icicle shape associated with stalactites, as long as the conditions of dripping water and sufficient calcium carbonate concentration are met

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: On the other hand, plastic straws have their own long-term environmental costs — they take centuries to decompose and contribute to microplastic pollution — though they emit less greenhouse gas during their lifecycle

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The most current and comprehensive evidence suggests that refusing straws altogether is the most effective action, as the scientific consensus is that the choice between paper and plastic does not significantly reduce environmental harm

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Religion and cultural perspectives differ, but in modern Western society, death is still widely considered a taboo topic — a view shaped by fear of the unknown, cultural conditioning the tendency to avoid discussing uncomfortable subjects

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: Market makers can also influence price movements by controlling a large portion of an asset's liquidity derivatives exchanges can contribute to manipulation by amplifying price differences across platforms

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: However, the specific ease of manipulation varies by asset and method: wash trading and spoofing are more common on low-volume coins, while bots and social-media hype can affect even the largest exchanges

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Yes, Australia was discovered by the Dutch — specifically by Willem Janszoon, who commanded the Duyfken and reached the western coast of Cape York Peninsula in 1606

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Yes, black holes can be seen with a telescope, though not directly — they are all but invisible because their gravity is so strong that no light can escape from them. Instead, scientists observe black holes indirectly, by detecting the gravitational effects they cause — such as bending light around them (gravitational lensing) or the accretion disks of material spiraling in — rather than the black holes themselves

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: None
- **Claim**: Organizers and attendees alike emphasized sharing and mutual support, with farmer Max Yasgur opening his home to the festival and thousands of people helping each other through traffic jams, mud storms

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Religion experts and commentators hold differing views. Some sources argue that Mormons are Christians in a broad sense — believing in Jesus Christ and practicing good works — while others argue that Mormons cannot be true Christians because they reject central Christian doctrines such as the Trinity and eternal torment

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Hindi is the third largest language by total number of speakers, after English and Mandarin Chinese

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: April 1, 2026; April 2, 2026; April 21, 2027

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest public release of Android is Android 16, which became available on June 10, 2025. This version is officially supported by Google and is available for download through the Google Play store

### Sample freshqa_2d7eb41139aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 11

### Sample freshqa_3cd0b514193b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: For national context, the average minimum wage across all 47 prefectures is approximately ¥1,121 per hour, with rural prefectures like Okinawa having the lowest rates

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 4

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The retrieved evidence consistently states that gold can be produced from lead, but the process is impractical and often produces radioactive isotopes

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d1
- **Claim**: The Voice Season 29

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Two

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Laika

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Martin himself has also repeatedly confirmed his birthplace, stating that his childhood was shaped by growing up in Bayonne and the surrounding area

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

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Jiangsu Province

### Sample freshqa_c479e83e408f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Kylian Mbappé scored 15 goals in the 2025–26 UEFA Champions League, becoming the first player in history to score 10 away goals in a single Champions League season

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Star Wars: The Rise of Skywalker is the most expensive film ever made, with a net production budget of roughly $490 million

### Sample freshqa_dd85dcbc2262

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Iga Swiatek

### Sample freshqa_dd87e1e3ad3d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Nine

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: The Peruvian Ministry of Culture has designated 77 of these geoglyphs within an archaeological park researchers estimate that approximately 500 more remain undiscovered in the Nazca Pampa

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: 1864

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Instead, parents should check their child's temperature with a thermometer, offer plenty of fluids consider children's paracetamol or ibuprofen if the fever is causing distress

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: El Nuevo Cojo Ilustrado is a separate, online Spanish-language magazine published from Los Angeles as of 2020 it is owned by Imprecya S.A

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9
- **Supporting Docs Found**: None
- **Claim**: For non-pediatric care, MedStar Georgetown is often cited as one of the top private hospitals in Washington, D.C

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

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d1, d6, d4, d7
- **Claim**: Pusha T

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d8, d6, d4, d7
- **Claim**: 506

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

### Sample qacc_160a528ae07e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Oliver Stark plays Buck on the TV show 9-1-1

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d4, d1
- **Supporting Docs Found**: None
- **Claim**: Early Christians also adopted the cross gesture as a secret recognition symbol among themselves, using it to invoke God's protection , while another theory attributes the luck association to the Christian fish symbol (ichthys) used to recognize fellow believers

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: The name of the lymphatic vessels located in the small intestine is the lacteal system

### Sample qacc_54be882d5b58

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The process was not completed until 1949, when Newfoundland joined Canada and the country officially became a fully independent nation

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: These figures reflect a temporal update: the more recent the source, the higher the count, as additional countries have been added to the visa waiver programs of various nations over time

### Sample qacc_8882ab46be5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9404250d756f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2
- **Claim**: Additional contextual details place the series in and around London, as well as in a fictional American-style diner setting , contributing to the show's distinctive aesthetic of blending British and American locations

### Sample qacc_946ecfb478b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Justin Timberlake wrote the song "Can't Stop the Feeling!" for DreamWorks Animation's *Trolls* (2016), with additional lyrics by Johan Karl Schuster and Martin Karl Sandberg

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is further corroborated by reports noting that the Yankees' 2017 season was a pursuit of the Red Sox's division title

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

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: It was developed in Duluth, Minnesota by the Domestic Abuse Intervention Project (DAIP) in 1981 and has since become the most widely replicated woman abuse intervention model in the United States

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The model's core principles include recognizing domestic violence as a pattern of power and control exerted by an abuser over their intimate partner, acknowledging that men can be victims and same-sex relationships can experience abuse placing responsibility on the abuser for their actions

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: 245

### Sample qacc_aa94588b9477

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d1
- **Supporting Docs Found**: None
- **Claim**: This date of establishment is consistently confirmed across multiple authoritative sources, including the official RBA history and the Australian Parliamentary record

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Celebrity Big Brother is typically shown on CBS in the USA

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912, when President William Taft signed the New Mexico statehood bill

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Some sources cite a different date (January 4, 1912) for this admission, but this reflects timezone differences rather than factual disagreement, as the signing itself occurred on January 6, 1912

### Sample qacc_b1bd9518429b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Most recently, the UK installed a cement reef in the Bay of Gibraltar that reportedly damaged Spanish fishing nets, prompting Spain to consider legal action at the UN, though both countries have also explored diplomatic dialogue to resolve the dispute

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d2
- **Claim**: This catastrophic event occurred during a children's party being held in the East Wing, forcing the celebration to be moved to the State Dining Room

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: California's Mojave Desert

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: This joint is essential for sound transmission, as the malleus vibrates against the incus, causing the ossicular chain to move collectively

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Elton Hayes

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Some sources cite a derivation from the Mozarabic word 'tabara' meaning 'footprint,' though this theory is not widely accepted by genealogists

### Sample qacc_ce4983c8a9c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Wisconsin has the highest density of these mounds in the world many remain intact today, though hundreds were destroyed by early settlers or natural erosion

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Within three months, 25 million numbers had been issued, covering approximately 45,000 post offices

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple authoritative sources, including the official Social Security Administration website and a Federal Reserve Economic Data publication

### Sample qacc_d60bf850c4ff

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Additionally, Cadbury products are sold in other European countries, as well as in Asia, Africa the Middle East, making its international footprint broad

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: All financial statements are prepared under this accrual basis, where revenues are recognized when earned and expenses are recorded when incurred, regardless of when cash is received or paid

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: These toll roads are identified by the suffix 'D' (for 'Directo'), so a highway with the designation Fed

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4
- **Claim**: 45D is a toll road, while Fed

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: Seven

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Three to four fronts — depending on how one counts

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Coton in the Elms, Derbyshire

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: There are approximately 649,481 villages in India, of which 593,615 are inhabited

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: The California state flag features a grizzly bear, making it one of several state flags that use a bear as a symbol. The grizzly bear on the California flag is the California Grizzly Bear (Ursus arctos californicus), also known as the California Brown Bear or California Golden Bear. This species was once abundant in California but was completely extinct by the 1920s the bear on the flag is based on the last known grizzly bear specimen

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The exact composition of 'chief' crops varies by region and definition, but these broad categories capture the most common commercial tree-based industries

### Sample situatedqa_geo_d0532ee6a455

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
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: By the early 19th century, coffee had fully eclipsed tea as the most popular morning beverage, with coffeehouses becoming widespread and tea-drinking remaining largely confined to Southern sweet tea and immigrant communities

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d1
- **Claim**: However, d1 and d5 offer additional context: d1 references the Tournament of Power as the event triggering Goku's transformation d5 places it during afterlife training, meaning the specific episode number may differ depending on the source or continuity

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 km

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Jagat Prakash Nadda (2026–2027)

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4, d1, d5
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d1
- **Claim**: Earlier figures from 2020 (11,184) and 2019 (9,439) reflect older data points as the city has expanded over time

### Sample situatedqa_temp_901be1437bc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1
- **Claim**: 90 cents per gallon

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: This date is corroborated by multiple sources, with Britannica confirming that the Inca religion and culture persisted until the 16th century, directly answering the query about when the empire ended

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: The ship had previously been launched on May 4, 2014, but commissioning — the formal acceptance into Royal Navy service — occurred on December 7, 2017

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
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

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. The retrieved evidence consistently identifies Alphabet Inc. as the company that owns Google. The newer Wikipedia revision directly states that Alphabet Inc. is the parent company to Google, while the older revision and other sources further corroborate this ownership relationship

### Sample wikirevision_0046

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Argentina (defeated Germany 1-0 in extra time on December 18, 2022; the 2026 FIFA World Cup is jointly hosted by Canada, Mexico the United States)

### Sample wikirevision_0047

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Argentina (defeated France 4-2 on penalties after a 3-3 draw on December 18, 2022; this victory gave Argentina its third World Cup title)

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Alphabet Inc. The retrieved evidence consistently identifies Alphabet Inc. as the owner of Google. The newer Wikipedia revision directly states that Alphabet Inc. is the company formerly known as Google, while the older revision and additional sources further confirm this ownership by noting that Alphabet acquired Wiz in March 2026 and that Larry Page and Sergey Brin still own about 14% of publicly listed shares

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dembélé, who plays for FC Barcelona, is the first winner from Spain in seven years

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: This is consistent across multiple sources, with France Football, the magazine that organizes the award, confirming his win

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1, d2
- **Claim**: This is consistent across multiple sources, including the official Wikipedia entries on both the Leader of the Labour Party and the Deputy Leader of the Labour Party, as well as external analyses of the party's leadership history

### Sample wikirevision_0093

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The latest Wimbledon men's singles champion is Nick Kyrgios, who defeated Dan Evans in the 2026 final to win his first major title. The 2026 championships were the 139th edition of the tournament, held at the All England Lawn Tennis and Croquet Club in Wimbledon from 29 June to 12 July 2026. Multiple sources confirm that 2026 is the most recent edition, with Jannik Sinner having won the 2025 championships, making Kyrgios the latest champion

### Sample wikirevision_0150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia (defeated India by six wickets on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad)


================================================================================

*Report generated by CATS v2.0*
