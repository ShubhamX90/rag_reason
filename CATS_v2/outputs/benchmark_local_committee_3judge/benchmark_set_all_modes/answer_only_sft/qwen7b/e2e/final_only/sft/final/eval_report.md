# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.971 (over 736 samples)

**GR F1** *(used in CATS)*: 0.982

**Behavior Adherence**: 0.762 (over 609 applicable samples)

**Factual Grounding**: 0.812 (over 609 applicable samples)

**Single-Truth Recall**: 0.652 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.802

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.982
- **Precision**: 0.998
- **Recall**: 0.967
- **Accuracy**: 0.971
- TP=588, FP=1, FN=20, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.864
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.924
- **Specificity**: 0.967
- Abstain TP=127, FP=20, FN=1, TN=588


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.903 (n=155)
- **Grounding**: 0.846 (n=155)
- **Recall**: 0.792 (n=154)
- **CATS**: 0.883

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.955
- **GR F1** *(used in CATS)*: 0.971
- **Behavior**: 0.858 (n=176)
- **Grounding**: 0.784 (n=176)
- **Recall**: 0.526 (n=156)
- **CATS**: 0.785

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.458 (n=96)
- **Grounding**: 0.851 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.766

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.717 (n=145)
- **Grounding**: 0.822 (n=145)
- **Recall**: 0.686 (n=140)
- **CATS**: 0.805

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.671 (n=37)
- **Recall**: 0.473 (n=37)
- **CATS**: 0.687


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

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: In other countries, such as the European Union, dedicated design patent protection exists to fill this gap

### Sample conflictingqa_07e290a954ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2, d4
- **Claim**: Effectiveness appears to depend on depression severity—studies generally support St. John's wort for mild depression but not for severe depression or high suicidality

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: While there are other large fungal networks — such as another strain of Armillaria in the Pacific Northwest that spans about 5.5 km (roughly 2,384 acres) — Armillaria ostoyae remains the undisputed champion

### Sample conflictingqa_2395695f1604

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: The most cautious conclusion

### Sample conflictingqa_24c25ef3a801

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Research by Kahneman and Deaton found that money can buy happiness up to a point — specifically, emotional wellbeing rises logarithmically with income until about $75,000 per year, after which wellbeing plateaus — while a later challenge by Matthew Killingsworth found that experienced well-being continues to rise with income even above that threshold

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d4
- **Claim**: Additional sources note that while wrist rests can reduce muscle fatigue and pressure, they are not necessary for good ergonomics and may even cause compression of delicate nerves and tendons if used incorrectly

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: Yes, audiobooks are considered real reading by the majority of sources

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d2, d4, d5
- **Claim**: The world's largest living lizard species is classified as an Australian native, with its fossil record in Queensland dating back approximately 4 million years

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Contested — some research (IUCN, Royal Society) argues trophy hunting can provide revenue and incentives for conservation, while critics argue it selects against rare species and can drive them to extinction; a ban's true consequences are uncertain

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: It is not unconstitutional for students to pray in school; the U.S. Supreme Court has made clear that the Establishment Clause does not prohibit students from praying individually or in groups during non-instructional time

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Further high-quality, large-scale trials are needed to clarify the optimal dose, delivery method patient population most likely to benefit

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: However, one source explicitly states that while Tambora's eruption was the largest in recorded history, it was not the deadliest

### Sample conflictingqa_62b1aff6586d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Encyclopaedia Britannica similarly states that 'Earth is currently experiencing a host of environmental problems... but environmentalists and scientists have reported one bright spot: the countries of the world rallying to combat the problem of ozone depletion,' though it stops short of a definitive heal

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The conflicting results reflect the difficulty in testing the earthquake-tide relationship: the effect appears to be real for large earthquakes but weak or non-existent for smaller events it does not appear to hold consistently across all datasets

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: No, the Gutenberg Bible was not the first book printed with movable type

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: However, many products can temporarily make split ends look smoother by coating the hair shaft or creating temporary bonds, though these effects are temporary and do not constitute true repair

### Sample conflictingqa_8848765fc18a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: Yes

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: However, the retrieved evidence does not directly address whether bird calls are unique to individual birds, only to species

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2, d3
- **Claim**: While braces may provide some benefit in specific circumstances—such as functional braces aiding stability during recovery from an injury—they should not be relied upon as a sole preventive strategy individuals should consult a healthcare provider to determine the most appropriate use

### Sample conflictingqa_9b73cb6cce52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: Yes, Giant African Land Snails (GALS) can make good pets, particularly for beginners or people looking for a low-maintenance invertebrate

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This process can occur in both modern caves and ancient underground formations the resulting structures are sometimes called 'aeromorphs'

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: In summary, the current state of knowledge is that comets are likely a minor source of meteorites proving any specific meteorite's cometary origin remains an open question

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: D4 further cautions that while nutritional yeast is a good source of protein and B vitamins, the specific nutrient content can vary significantly between brands consuming fortified versions does not necessarily ensure you will meet the recommended daily levels of all vitamins

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Yes, Michael Jackson did compose songs for Sonic the Hedgeog 3; game creator Yuji Naka confirmed he wrote music for the 1994 Sonic the Hedgeog 3 soundtrack

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Religious and scholarly views differ; science cannot confirm or deny historical existence

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3
- **Claim**: While manipulation is a recognized problem, the extent and frequency of successful attempts remain actively debated

### Sample conflictingqa_e93e708d49a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, the specific ratio depends on geography and system design: for example, a typical rooftop system in Arizona might produce 27 times the manufacturing energy, while a system in Alaska might produce only 14 times as much — though the study confirms that allowing excess power to flow to the grid maximizes the net benefit in all cases

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d3
- **Claim**: The most widely supported scientific view holds that humans and apes share a common ancestor who lived millions of years ago, though the evidence is contested by creationist sources that read Genesis literally and reject the idea of evolution entirely

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d5, d2, d1
- **Supporting Docs Found**: None
- **Claim**: Research suggests that chemicals in yerba mate, including polycyclic aromatic hydrocarbons (PAHs) — known carcinogens also found in grilled meat and tobacco smoke — may contribute to this risk , as well as the synergistic effect between mate, alcohol tobacco ; however, the evidence does not establish causation conclusively some studies have shown that yerba mate exhibits a cytotoxic effect on cancer cells in laboratory settings, suggesting it may have some anti-cancer properties in vitro

### Sample freshqa_02b3ba89ebd0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d1
- **Claim**: Multiple candidates were nominated, including Jeffries (212 votes), Byron Donalds (17) Kevin Hern (3–7), but none won the speakership

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: Aryna Sabalenka and Amanda Anisimova were the 2025 US Open women's singles finalists

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: It runs through the 22nd day of the Hebrew month of Nissan, with the first seder held on the evening of April 1

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d3, d5
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani is the first and to date only female recipient of the Fields Medal

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The result ended Verstappen's four-year reign as champion and gave McLaren their first drivers' title since 1988

### Sample freshqa_2130eea851fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Earlier sources cite a lower total — 941,000 — as of a prior date, reflecting the fact that the count grows continuously

### Sample freshqa_263eca8e024e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These discrepancies reflect how rankings can shift over time as new films release and currencies fluctuate

### Sample freshqa_2877cf4bd00f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Earlier sources from 2026 reported him as being 30 years old on July 1, 2025 , but this figure refers to his age when first elected in 2016 (70 years old) rather than his current age

### Sample freshqa_28e155139ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Google has since initiated development on Android 16, with its second Developer Preview released on December 18, 2024, though Android 16 is not yet available for general use

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: The official U.S. government energy department website confirms that the Trinity test, which detonated a plutonium implosion device code-named 'Gadget' atop a 100-foot tower, occurred on July 16, 1945, releasing 18.6 kilotons of power

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

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Two million years ago

### Sample freshqa_5eb89aae15f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Kantara (Chapter 1) — ₹407.82 crore (worldwide gross)

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1, d5
- **Claim**: Multiple authoritative sources, including the White House itself, corroborate that Biden is the current president, superseding earlier terms of Trump (45th president, 2017–2021) and Barack Obama (44th president, 2009–2017)

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The Houston Astros have won 2 World Series titles in their history

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Luka Modric (2018)

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Someone You Can Build a Nest In by John Wiswell (DAW and Arcadia UK), winner of the 2025 Nebula Award for Best Novel

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Rosenblatt's death came two years after the publication of Perceptrons by Marvin Minsky and Seymour Papert, which argued that the Perceptron could only solve linearly separable functions, leading to a decline in funding and interest in the field

### Sample freshqa_ab11b5dce00e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: The FIFA World Cup 2026 will be co-hosted by the United States, Mexico Canada

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: June 2025

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Zhejiang Province borders Shanghai to the north

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Saltwater crocodile (Crocodylus porosus) — 1.2 m (4 ft) thick

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: The latest stable macOS release is macOS 26 Tahoe, as of September 2025

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 0

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: Ta-Nehisi Coates

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0031

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d10, d5
- **Claim**: Victor Mature

### Sample hotpotqa_0070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d5, d7, d6, d1
- **Claim**: Pusha T

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d7, d8, d6, d2
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: I never should set is never said in the play

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: He completed the head and the torch-bearing arm before the statue was fully designed these pieces were exhibited for publicity at international expositions

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Additionally, The Phantom of the Opera was also performed at the Ed Mirvish Theatre in Toronto, though the specific run dates are not explicitly stated in the retrieved evidence

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 15

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1
- **Claim**: The 2025 TV series adaptation is still in early development, with no final cast announced, though industry sources have begun imagining potential replacements for the original stars

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Urochordata (tunicates) — Urochordata (also known as tunicates or sea squirts) are the earliest known vertebrates, with fossil evidence dating back approximately 570 million years ago

### Sample qacc_367b09e4ed80

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Gidget is the white Pomeranian/Poodle mix who is the main character's (Max's) nemesis. She is voiced by Jenny Slate, who also voiced Mrs. Wiggins in The Lorax (2012) and Bellwether the snake in Zootopia

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: The other theory points to early Christianity: followers developed hand gestures to recognize each other in secret crossing fingers (or thumbs and index fingers) was one such sign; over time, the gesture evolved from requiring a partner to being performed solo

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The gesture's roots in Christianity help explain why it is not universally understood as a luck symbol in non-Christian cultures, such as Vietnam

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: The Statute of Westminster in 1931 further solidified Canada's legislative independence Canada's constitutional framework was fully decolonized with the 1982 Canada Act, which declared no British law would henceforth apply to Canada

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: McEwan

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 180

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The average number across eukaryotes is estimated at roughly 5,000–10,000 origins per chromosome , but this figure can be significantly higher in large genomes or under certain cellular conditions

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: 5.88

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: For the most up-to-date information on the first McDonald's in Phoenix's current status, consult local sources or official McDonald's communications

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: The song was written by Russ Ballard; Argent recorded the original version Kiss also recorded a major cover version

### Sample qacc_9c2f95b14a78

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Research comparing the Duluth Model to cognitive behavioral therapy found that participants in Duluth Model interventions were less likely to recidivate and reported fewer instances of violence with their partners

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The International Space Station (ISS) assembled in orbit in phases: Zarya, the first module, was launched November 20, 1998

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Season 10 of El Señor de los Cielos premiered on 13 February 2024

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2, d5
- **Claim**: This apparent contradiction reflects differing perspectives: some sources focus on the most visible and symbolically important milestones, while others emphasize that the project's completion is a gradual process with no single 'finished' date

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: This distribution holds true for the average adult male (about 60% body weight from water), though percentages do vary slightly by sex, age adiposity

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: 245

### Sample qacc_a78a32b7b9a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Joseph McCarthy is widely recognized as the most prominent figure who stoked the Red Scare in the 1950s, though not the sole originator

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1
- **Claim**: He then launched investigations and hearings through his Senate Investigations Subcommittee, publicly accusing government officials, military personnel others of being Communist sympathizers, using dramatic tactics that amplified national anxiety

### Sample qacc_b281f09f0959

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: While McCarthy is the dominant symbol of the 1950s Red Scare, it is important to note that his rise was enabled by a broader context of Cold War anticommunism many other organizations and individuals — including the House Un-American Activities Committee (HUAC) — were also active in fostering similar fears

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: President Herbert Hoover and his staff continued the party in another part of the house while the blaze raged

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The Christmas 1929 West Wing fire is also remembered today through a special 2016 White House Christmas Ornament that honors President Hoover's presidency

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Additional filming took place in Puerto Rico, Rio de Janeiro Atlanta, but the train scene itself is confirmed to have been filmed in the Mojave Desert region

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This joint is crucial for the transmission of sound vibrations in the middle ear

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The stapes also articulates with the incus via a ball and socket joint

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Tavarez is a Hispanic surname derived from the Portuguese surname Tavares, with its roots tracing back to medieval Portugal

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Over 50

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: September 28, 1998

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The balance sheet (statement of financial position/statement of financial condition)

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: In Mexico, toll roads are called **autopistas** or **cuota** highways. The federal agency responsible for many Mexican toll roads is Caminos y Puentes Federales de Ingresos y Servicios Conexos (CAPUFE), a division of the Secretaría de Comunicaciones y Transportes (SCT). Federal highways with the letter suffix 'D' are typically toll roads (e.g., Fed

### Sample qacc_e326d0094f42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 45D), while others are operated by state governments or private concessionaires such as Coconal, IDEAL Pinfra

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The last time Rangers were in the Champions League was the 2022–23 season, when they were drawn in Group A alongside Ajax, Napoli Liverpool

### Sample qacc_ecbc6adf8a48

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: There is no scholarly consensus on the precise date, though most scholars place it in the 90s AD

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d5
- **Claim**: The most recent ICD-10-CM (Clinical Modification) uses a seven-character format, with the first character always a letter and the remaining six positions allowing a mix of letters and numbers

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: 7

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: The meaning of red license plates in other regions — such as the United States or Japan — is not covered in the available evidence

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: September 1, 1935

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The bulk of immigrants coming to the United States have historically been from Mexico, with Mexican immigrants once accounting for roughly 60% of all U.S. immigrants during the peak period from 1965 to 2007

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: These rankings are corroborated by multiple sources, with Shanghai (China) at 29,558,908 and Guangzhou (China) at 27,563,372 following closely behind

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: A grizzly bear (Ursus arctos californicus), also known as the California brown bear or chaparral bear

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence supports Scotland winning the last Calcutta Cup, but the seed answer does not specify the exact year

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d5
- **Supporting Docs Found**: None
- **Claim**: The FOMC is composed of twelve members — seven from the Board of Governors and five rotating presidents from Federal Reserve Banks — who meet regularly to influence money supply and interest rates through open market operations

### Sample situatedqa_temp_05d714be23fd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This is confirmed across multiple sources, including the official iHeart content and the Yahoo Entertainment coverage

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: New South Wales last won the State of Origin series in 2025

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: He is the only current New Jersey Senator, as Bob Menendez resigned in August 2023 following his conviction on federal corruption charges, leaving the state with a single senator

### Sample situatedqa_temp_3026b0491e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: Williams also created the iconic 'Hedwig's Theme,' which is featured in all eight films of the series laid the foundational musical themes heard throughout the entire Harry Potter series

### Sample situatedqa_temp_32d33d503f69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3, d5
- **Claim**: January 17, 2025 (United States)

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Jordan Ridgeway

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The official FIFA website confirms Argentina's victory over France in the final, giving them their first title since 1986

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1
- **Claim**: This makes it the largest national park in the United States

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Goku first becomes Super Saiyan 3 in the Tournament of Power saga, specifically in the 14th episode ("An Astounding, Great Transformation!!

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The Pakistan Tehreek-e-Insaf (PTI) won the 2018 general election, becoming the largest single party in the National Assembly

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken is the current head coach of the Cleveland Browns

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Australia's coastline is approximately 59,681 km (37,061 miles) in total length, comprising 35,821 km (22,268 miles) of mainland coastline and 23,860 km (14,823 miles) of island coastline

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d1, d3
- **Claim**: It is specifically the absence of this enzyme that leads to the accumulation of gangliosides in nerve cells of the brain, progressively damaging those cells

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This victory came during the NBA bubble in Orlando, with LeBron James and Anthony Davis leading the team

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Their previous title had come in 2010, making the 2020 run their first championship in nearly a decade

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The population of Belgium on January 1st 2018 was approximately 11,589,069

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Earlier estimates from 2025 and 2024 reported lower figures (11,744,521 and 11,715,774 respectively), reflecting a steady decline , but these are not directly comparable to the 2018 figure

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Wilson Phillips is an American vocal trio consisting of Carnie Wilson, Chynna Phillips Wendy Wilson

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5
- **Claim**: The group was formed in Los Angeles in 1989 gained widespread fame with the release of their self-titled debut album in 1990, which produced hits such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Shay Mitchell, the actress who played Emily Fields, was 23 years old when the show premiered in 2010, making her significantly older than her in-character age of 16

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: India's overall score was 1.096, which is lower than the average score of 1.234 for all 163 countries , reflecting the need for continued improvement in societal safety and security

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: In terms of regional variation, the name is also found in Haiti and is used as a given name in many Germanic and Romance-speaking regions, with notable bearers including Archbishop of York Gerard (died 1108) and the French name scholar Jean-Pierre Gérard

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: He defeated Tyson Fury for the WBC title on May 18, 2024, becoming the first boxer in history to unify all four major belts in the heavyweight division during the four-belt era

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The award is given to the player adjudged best in English football, selected by a vote among PFA members on three occasions the same player has won both the PFA Player of the Year and Young Player of the Year awards in the same season

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: People's Sexiest Man Alive title has been awarded to multiple individuals over the years, with Brad Pitt holding the record for the most wins at 5

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: Hello, Love, Again

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The exact items are typically updated each year, so the most recent list is the best source for current eligibility

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The Philadelphia 76ers last made the playoffs in the 2021–22 NBA season, when they reached the Eastern Conference Finals

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc., having been reorganized as a wholly owned subsidiary of Alphabet in 2015

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d4
- **Claim**: This structure means that while Google operates as a separate entity, it is legally and financially integrated into the broader Alphabet Inc. ecosystem

### Sample wikirevision_0093

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Jannik Sinner is the current US Open men's singles champion, having defeated Carlos Alcaraz in the 2025 final. This result is corroborated by the Wikipedia pages on both the US Open (which identify the 2025 edition as the most recent) and the 2025 US Open specifically

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Australia are the latest Cricket World Cup champions, having won the 2023 edition in November 2023. Australia's victory in the final against India on 19 November 2023 confirmed their status as the most successful team in the history of the tournament, having won the title six times in total. The 2027 edition (14th edition) is also mentioned, but that is still future-oriented and does not supersede the 2023 result

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore is officially called Bengaluru. Bengaluru is the current official name of the city, while Bangalore was its official name until 1 November 2014, when the city government officially changed the name from Bangalore to Bengaluru. This change was confirmed by the Karnataka Government Resolution dated 1 November 2014, which stated that henceforth the city would be known officially as Bengaluru

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0150

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The latest US Open men's singles champion is Carlos Alcaraz, who won the 2025 US Open, the 145th edition of the tournament. He defeated Jannik Sinner in the final, claiming his second US Open title. The 2024 US Open (144th edition) was the previous edition, where Novak Djokovic was the champion, but that edition has since been superseded by the 2025 result

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Australia are the current Cricket World Cup champions, having won the 2023 edition in November 2023. Australia's win gave them their sixth title, making them the most successful team in the history of the competition. The 2023 World Cup was the 13th edition, held across ten venues in India, with Australia defeating India by six wickets in the final at the Narendra Modi Stadium in Ahmedabad

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The current French Open men's singles champion is Carlos Alcaraz, who defended his title in 2026 by defeating Jannik Sinner in the final

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: This victory gave Alcaraz his second French Open title and fifth major


================================================================================

*Report generated by CATS v2.0*
