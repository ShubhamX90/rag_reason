# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.929 (over 736 samples)

**GR F1** *(used in CATS)*: 0.955

**Behavior Adherence**: 0.760 (over 608 applicable samples)

**Factual Grounding**: 0.762 (over 608 applicable samples)

**Single-Truth Recall**: 0.613 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.773

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.955
- **Precision**: 1.000
- **Recall**: 0.914
- **Accuracy**: 0.929
- TP=556, FP=0, FN=52, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.711
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.831
- **Specificity**: 0.914
- Abstain TP=128, FP=52, FN=0, TN=556


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.976
- **GR F1** *(used in CATS)*: 0.983
- **Behavior**: 0.883 (n=154)
- **Grounding**: 0.835 (n=154)
- **Recall**: 0.763 (n=154)
- **CATS**: 0.866

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.910
- **GR F1** *(used in CATS)*: 0.940
- **Behavior**: 0.835 (n=176)
- **Grounding**: 0.753 (n=176)
- **Recall**: 0.484 (n=156)
- **CATS**: 0.753

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.954
- **GR F1** *(used in CATS)*: 0.973
- **Behavior**: 0.594 (n=96)
- **Grounding**: 0.748 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.772

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.911
- **GR F1** *(used in CATS)*: 0.949
- **Behavior**: 0.690 (n=145)
- **Grounding**: 0.753 (n=145)
- **Recall**: 0.636 (n=140)
- **CATS**: 0.757

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.595 (n=37)
- **Grounding**: 0.568 (n=37)
- **Recall**: 0.446 (n=37)
- **CATS**: 0.622


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2051

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
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Instead, fashion designs are primarily protected under a separate sui generis regime — the Vessel Hull Design Protection Act (VH DPA) — which offers distinct, limited protection for qualifying designs , while trademark law steps in to protect logos, labels brand elements that signify source or origin

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The poem itself explores the human cost of conformity and capitalist consumerism, using raw language to challenge dominant values — a double-edged sword that both offends some readers and reveals the very hypocrisy Ginsberg sought to expose

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The evidence is mixed and depends on the specific nutrient: fiber is significantly reduced by peeling (~50%), but vitamin C and antioxidant compounds are evenly distributed throughout the fruit, so peeling does not uniformly reduce all nutritional value

### Sample conflictingqa_21f33954c8af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The debate thus remains unresolved and continues to be actively debated by veterinary organizations, ethicists ordinary citizens

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d1, d2
- **Claim**: Yes, the Silurian period is considered the birthplace of the first land plants, specifically the first vascular plants (embryophytes), which emerged during the Middle Ordovician to early Silurian transition (~470–430 million years ago). The most famous of these early pioneers is Cooksonia, which grew on land as early as the Late Silurian, though older lycophyte fossils like Baragwanathia may represent even earlier land plants

### Sample conflictingqa_24fa0020a521

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d1, d4
- **Claim**: [[

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Overall, the evidence suggests that fluoride safety depends heavily on dosage and population, with vulnerable groups such as infants and people with certain medical conditions facing the greatest risks the CDC's own fluoridation guidelines acknowledge ongoing scientific debate

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Yes, hair can turn green from swimming pools — but not directly from chlorine itself. The real culprit is copper, which is commonly found in algaecides used to control algae growth in pools. When copper oxidizes (reacts with air), it turns from a shiny orange hue to a dull green when it comes into contact with hair, it sticks to the proteins in the hair shaft and turns it green as well

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: [[

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3
- **Claim**: Yes, audiobooks are considered real reading by many educators and researchers. The NPR-Ipsos poll cited by The New York Times found that 41 percent of adults do not believe audiobooks qualify as reading , but this is a minority view: a peer-reviewed study cited by PBS found that adults who listen to audiobooks demonstrate the same neural activity as those who read physical books the National Center for Learning Disabilities has stated that audiobooks are a valid and accessible reading format for individuals with dyslexia and other learning disabilities

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Some sources, including a 2022 survey of 5,000 students in the UK, even found that 74% of respondents considered audiobooks equal to or more engaging than traditional reading

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: Yes, real Christmas trees are generally more sustainable than artificial ones, as they have lower carbon emissions and fewer toxic impacts

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d2
- **Supporting Docs Found**: None
- **Claim**: However, other researchers and commentators argue that the current system is far from well-managed — with overhunting, indiscriminate killing revenue diversion being well-documented problems in some areas — and that the broader scientific consensus remains that blanket bans are more likely to protect species than the status quo

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2
- **Claim**: In stage 5 CKD specifically, most data suggests that bicarbonate does not prevent progression to end-stage renal disease KDIGO guidelines recommend it only when serum bicarbonate is less than 18 mEq/L

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3, d5
- **Claim**: Parents and caregivers should discuss these conflicting perspectives with their physician, as the likelihood of regrowth appears to depend heavily on age, surgical technique postoperative care

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: However, hair experts agree that certain high-quality bond-building products can repair the internal bonds broken by chemical treatments, heat styling poor combing, effectively stopping further splitting and fraying

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: These temporary repairs are not permanent fixes — the bonds will eventually break again — but they do buy time and help prevent the need for frequent trims

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: However, other research presents a more nuanced picture — a study from the University of Wisconsin and the Global Change Data Lab found that organic farming is the more sustainable method overall , while a comparison of industrial vs. organic farming practices in the UK concluded that organic farming produces the same yield with fewer environmental harms

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Religious organizations and scholars hold differing views. Some sources argue that the Catholic Church is the one true church because it traces its origins to Jesus Christ and holds an unbroken apostolic succession, while others argue that 'one true church' in the New Testament refers to a church that aligns with Scripture rather than being the historically first church

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Overall, the evidence suggests that while farmed salmon is broadly considered a healthy dietary choice, wild salmon may offer slightly more heart-health and nutritional benefits, particularly for those concerned about contaminants

### Sample conflictingqa_9b11b8e571aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Gonorrhea can be transmitted even without ejaculation, as long as there is genital-to-genital or anal-to-anal contact a multi-state study found that nearly half of reported cases in 2011 involved individuals who reported no sexual contact

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: These soda straws can later thicken and branch into the familiar icicle shape associated with stalactites, even underwater

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, scientists acknowledge ongoing debate about the precise mechanism and the role of other potential carbon reservoirs, such as methane-rich ocean sediments or organic-rich permafrost, which could have been triggered by rising temperatures

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: However, d4 notes that if global energy demand continues to grow rapidly and is met mostly with fossil fuels, human emissions could reach 75 billion tons per year or more by the end of the century, at which point atmospheric CO2 could reach 800 ppm — conditions not seen on Earth for close to 50 million years

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This discomfort is further exacerbated by the digital age, as social media platforms have increasingly become the primary outlet for discussing sensitive topics like sex, while death remains largely off-limits

### Sample conflictingqa_cd661c2c20b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Yes, Gwen Stacy's death is widely considered the end of the Silver Age of Comics, representing a symbolic turning point where mainstream superhero comics transitioned from the innocent Silver Age to the more mature and socially conscious Bronze Age

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Religious or philosophical systems may allow false beliefs to be justified, but this does not mean that false beliefs are ever truly justified in a scientific or epistemological sense

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Emojis are widely used as a supplement to written language but do not constitute a separate written language on their own

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, the evidence does not confirm that the Dutch were the sole discoverers of Australia or that they discovered the continent before other European powers

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Some low-credibility sources suggest yerba mate may have cytotoxic effects on cancer cells in vitro it is alleged to contain polycyclic aromatic hydrocarbons (PAHs) — known carcinogens also found in grilled meat and tobacco smoke — though researchers have not established definitive causal links

### Sample conflictingqa_f970957c5e52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [[

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Religion is contextually defined Mormons consider themselves Christians in the sense of believing in Jesus Christ and following His teachings; however, they are not recognized as Christians by many mainstream Christian organizations due to doctrinal differences, particularly their polytheistic concept of God and their rejection of original sin

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: English is the third largest language by total number of speakers, behind Mandarin Chinese and Spanish

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This ranking is consistent across multiple sources, with earlier reports from 2025 also confirming English as the third most spoken language, ahead of languages like Hindi, Arabic French

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence indicates that Prince Harry's Duke of Sussex title was stripped by King Charles III in the aftermath of the Sussexes' departure from their royal roles in 2020

### Sample freshqa_049cc3f14d5e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: April 1, 2026; April 2, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2130eea851fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: However, the specific citation figure is subject to change as his work continues to be cited, making it an approximate rather than definitive count

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest public release of Android is Android 16, which became available on June 10, 2025. This version is officially released and available for download through the Google Play Store, superseding the older Android 15 release from October 2024

### Sample freshqa_2d7eb41139aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: There are 11 games in the Ace Attorney main series, with the most recent being Phoenix Wright: Ace Attorney Spirit of Justice

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This explosion released approximately 18.6 kilotons of energy, instantly vaporizing a 100-foot steel tower and leaving a crater 3,600 feet in diameter

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: This escalated to a full-scale war that has displaced millions of people and resulted in thousands of deaths

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Three seasons — Season 1 premiered on November 12, 2019, Season 2 on October 30, 2020 Season 3 on March 1, 2023

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

### Sample freshqa_4e635a2542a8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d4
- **Claim**: [[

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: Previously, the oldest DNA recorded was from a million-year-old mammoth tooth, but the new find surpasses that by a factor of two

### Sample freshqa_64c12116affc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official historical records of the U.S. House of Representatives and the White House

### Sample freshqa_6927f007d7cc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d4
- **Claim**: Alexia Jayy

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8eca5bd62ae0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The 2008 Summer Olympics marked China's emergence as an equal nation on the world stage, while the 2022 Winter Olympics further cemented Beijing's status as a major international host

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

### Sample freshqa_ab11b5dce00e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Eleven U.S. cities will host the tournament, including Atlanta, Boston, Dallas, Houston, Kansas City, Los Angeles, Miami, New York, San Francisco, San Jose Washington D.C

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu Province

### Sample freshqa_c479e83e408f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Kylian Mbappé scored 15 goals in the 2025–26 UEFA Champions League season

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Star Wars: The Rise of Skywalker holds the record for the most expensive film ever made, with a net production budget of roughly $490 million

### Sample freshqa_dd85dcbc2262

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Iga Swiatek

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Twelve

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f11e8d758fc6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: This single lung functions through muscular movements that alternately compress and expand the lung cavity, much like the human lung

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d4
- **Claim**: Some sources, particularly those focusing on specific families of slugs, may describe two pairs of tentacles (as in stylommatophoran families or banana slugs), but these refer to the number of respiratory openings, not the number of lungs

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: 1864

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: A high-credibility source confirms that fever is a common symptom of scarlet fever, a bacterial infection that children with the disease should be seen by a GP for antibiotic treatment

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Overall, the available evidence suggests yoga may provide some benefit as an adjunctive or ancillary intervention, but should not be relied upon as a sole management strategy without medical supervision

### Sample hotpotqa_0056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1, d2
- **Supporting Docs Found**: d10
- **Claim**: Everton Football Club is based in Liverpool, Merseyside, England

### Sample hotpotqa_0062

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: d9
- **Claim**: While d9 explicitly confirms 'Funnybot' as the second episode of season 15, multiple other sources corroborate Trey Parker and Matt Stone's continued collaboration on the series, which has run through at least fifteen seasons

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d6, d2
- **Supporting Docs Found**: None
- **Claim**: Boston College has been located in Chestnut Hill since 1957, when Alumni Stadium was built on the main campus

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9, d5
- **Supporting Docs Found**: None
- **Claim**: The university is easily accessible from the Chestnut Hill Reservoir stop on the MBTA Green Line, making it a landmark in the area

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

### Sample hotpotqa_0079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d6, d1, d4, d7
- **Claim**: Pusha T

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d6, d8, d4, d7
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

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The retrieved evidence indicates that after landing in North Africa, the Allies (British and American forces) pushed eastward across the continent, linking up with each other and encircling Tunis by May 1943. The campaign concluded on May 7, 1944, when German and Italian troops surrendered in Tunis, ending the North Africa conflict

### Sample qacc_160a528ae07e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: His most recent MVP came in the 2017 season, when he set numerous records and guided the Patriots to a 13–3 finish

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_19ca08790764

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: Oliver Stark plays Buck on the TV show 9-1-1

### Sample qacc_2f6d2647a424

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: This alignment remained consistent throughout the 1975 regular season and playoffs, with George Foster manning left field and Dave Concepcion playing shortstop

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Early Europeans also used this cross-finger gesture as a secret handshake among themselves it was later popularized as a personal good-luck ritual

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Separately, historians note that Christians adopted the crossed-finger symbol (known as the ichthys) as a recognition symbol among themselves some scholars suggest this may have contributed to the broader popularization of the gesture as a general good-luck symbol

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Peyer's patches are organized lymphoid nodules found predominantly in the ileum, extending from the mucosa into the submucosa they play a critical role in the immune system

### Sample qacc_4387048ed24f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The tree was described as a 'goodly spectacle' and caused quite a stir at court

### Sample qacc_8882ab46be5d

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

### Sample qacc_9c2f95b14a78

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The model's core principles include recognizing domestic violence as a pattern of power and control exerted by an abuser over their intimate partner, acknowledging that men are the primary perpetrators while also recognizing that men can be victims and same-sex relationships can experience abuse

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: The International Space Station (ISS) first went into space on October 31, 2000, when Expedition 1 members Vladimir Titov and Amy Perrin aboard the Space Shuttle Endeavour docked with the station's Russian Zarya module

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3, d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The total number of elected members in the Rajya Sabha is 245, as confirmed by multiple authoritative sources

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d5, d2
- **Claim**: This figure includes 233 members elected by the Vidhan Sabha of each State and Union Territory, as well as 12 members nominated by the President for their outstanding contributions to art, literature, science social service

### Sample qacc_aa94588b9477

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2, d1, d4
- **Claim**: [[

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Celebrity Big Brother is typically covered by CBS in the USA

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912, when President William Taft signed the New Mexico statehood bill

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d4
- **Supporting Docs Found**: None
- **Claim**: Some sources cite slightly different figures due to temporal updates—one source refers to New Mexico as the 48th state another source notes it was the 47th state by the time Arizona was admitted in 1912 —but these differences reflect minor temporal variations rather than factual disagreement

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The fire originated in the Executive Offices on the main floor, which housed the President's office, the secretary's offices the press room

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: California's Mojave Desert (near Parker, Arizona and Vidal Junction)

### Sample qacc_c264cb69676e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The tensor tympani and stapedius muscles attach to the malleus and incus respectively, stabilizing these bones and protecting the joint

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Elton Hayes

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: [[

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple authoritative sources, including the official Social Security Administration website and the St. Louis Federal Reserve Economic Data

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The accounting equation (Assets = Liabilities + Equity) is the foundation of the financial statements and double-entry bookkeeping. It ensures that the balance sheet remains balanced by requiring that total debits equal total credits

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Their most recent participation came in the 2025–26 season, where they reached the group stage and finished second in their group

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Prior to that, their last Champions League appearance was in the 2022–23 season, where they lost to Ajax 4-0 in the group stage

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Initialisms are a specific type of abbreviation formed by using the first letter (or initial letters) of each word in a phrase. They are distinct from acronyms, which are pronounced as a single word from initialisms, which are pronounced as individual letters. Examples of initialisms include DNA (deoxyribonucleic acid), RT-PCR (reverse transcription-polymerase chain reaction) IT (information technology)

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Each ICD-10 code consists of letters and numbers, with three to seven characters depending on the level of detail required. The first three characters identify the body part or etiology, the next two specify the category the last one or two add further detail such as severity or procedure

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Sushma Swaraj's appointment as the first full-time woman Cabinet minister with the external affairs portfolio was confirmed by the Government of India's official gazette

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Seven

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: The conflict spanned five continents and resulted in widespread devastation, with China suffering 3–4 million military deaths and 20 million total casualties, while Germany, Italy Japan also experienced significant military and civilian losses

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

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Coton in the Elms

### Sample situatedqa_geo_779fd84224fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: The fleet, which consisted of 11 ships and carried over 1,500 people, had set sail from Portsmouth, England in May 1787

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Beyond this core federal structure, the U.S. also leverages checks and balances across branches to prevent any single body from accumulating too much power

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Federal funding and oversight roles also vary by whether the levee is located in a floodplain or on a riverbank

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: These conflicting figures reflect methodological differences over what constitutes a 'city,' with some sources counting metro-area populations and others restricting the definition to city proper populations , leading to divergent rankings depending on the metric used

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a grizzly bear, which is also the official state animal of California. The grizzly bear on the flag is a real animal called the California grizzly bear (Ursus arctos californicus) the flag itself is based on the Bear Flag Republic flag, which was used by a group of U.S. settlers who attempted to break away from Mexico in 1846

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d3, d1, d4
- **Claim**: TRUNCATE

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: After several years of experimenting with this form of government, delegates from every state except Georgia met in Philadelphia in 1787 to draft a new constitution, which replaced the Articles of Confederation in 1788

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: By the 20th century, coffee had become the dominant beverage in the United States, with approximately 75% of adults drinking it daily , a trend further reinforced by immigration and industrialization

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: The FOMC consists of twelve members—seven from the Board of Governors and five from the Federal Reserve Banks—and meets regularly to adjust interest rates and the money supply, with the Board of Governors serving as the more permanent body that appoints the presidents of the regional Federal Reserve Banks

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2
- **Claim**: At the federal level, key policies include NEPA, which requires environmental impact statements for major federal actions various air and water quality standards designed to protect public health and the natural environment

### Sample situatedqa_temp_1987d35f994b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d2
- **Claim**: [[

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida Gators (defeated LSU 6-1 on June 27)

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2
- **Claim**: Mort is a Goodman's mouse lemur, a small primate native to Madagascar, though he is technically classified as a bear due to having bear DNA (goodmanbear.com)

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

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d5
- **Claim**: This victory gave Argentina their third World Cup title, making them the reigning champions for the 2026 tournament

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3
- **Claim**: It is worth noting that some sources cite 1979 as the year of establishment, reflecting an earlier UNESCO designation as a World Heritage Site, but the primary establishment date remains December 1, 1978

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The fourteenth episode of the Majin Buu Saga and the two hundred forty-fifth overall episode in the series is titled "An Astounding, Great Transformation!!

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 68.7 billion

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3, d4
- **Claim**: This figure is derived from the most recent and comprehensive data available, with older sources reporting slightly different figures due to methodological and measurement-scale differences

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d1, d4
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: 90 cents per gallon

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_ae0882e48812

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Together, these deserts cover approximately 700,000 square kilometers, accounting for about 13% of China's total land area

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d3
- **Claim**: Some sources cite a slightly different timeline, noting that the empire's expansion under Pachacuti began around 1438 and that the Spanish conquest occurred roughly around 1532, with the Neo-Inca State surviving until 1572

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: In addition, the U.S. has contributed athletes to every single Olympics, making the country one of the most prominent participants in Olympic history

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The surname was first recorded in the Domesday Book of 1086 as Gerardus or Girardus is also found in Haiti

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [[

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Additional context from multiple sources further corroborates this figure, with a 2024 estimate of 167 adults and a 2026 median age of 68.3 years

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

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: [[

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
- **Claim**: Alphabet Inc. The retrieved evidence consistently states that Google is a subsidiary of Alphabet Inc., with the newer Wikipedia revision explicitly confirming this ownership relationship

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: The latest FIFA World Cup champion is Argentina, who won the 2026 tournament, defeating Italy 3–2 in a penalty shootout at the Rose Bowl in Pasadena, California

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. The retrieved evidence consistently identifies Alphabet Inc. as the owner of Google, with d2 and d4 explicitly confirming this corroborating via partial facts. The stronger support comes from d2, which directly states that Alphabet Inc. is the company formerly known as Google d4, which explicitly names Alphabet Inc. as Google's parent company

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 Ballon d'Or was the 70th annual ceremony, presented by French magazine France Football, recognizing the best footballer in the world for the 2025–26 season

### Sample wikirevision_0082

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: This is confirmed by the official France Football website, which lists Ousmane Dembélé as the holder of the 2026 Ballon d'Or

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
- **Claim**: The latest Wimbledon men's singles champion is Nick Kyrgios, who defeated Dan Evans in the 2026 final to win his first major title. The 2026 championships were the 139th edition of the tournament, held at the All England Lawn Tennis and Croquet Club in Wimbledon from 29 June to 12 July 2026. This result is corroborated by the Wikipedia page on the 2026 Wimbledon Championships, which confirms Kyrgios's victory as part of a comprehensive tournament summary

### Sample wikirevision_0150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of India is Droupadi Murmu, who became the country's first female president on 24 July 2022. This is confirmed by the official Wikipedia revision that superseded the older version in May 2026, which explicitly names her as the current holder of the office. As of 2026, she is serving her second term as president, having been reelected in 2024


================================================================================

*Report generated by CATS v2.0*
