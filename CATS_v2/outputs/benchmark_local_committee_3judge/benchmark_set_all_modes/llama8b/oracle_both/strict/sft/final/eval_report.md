# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 120 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.974 (over 736 samples)

**GR F1** *(used in CATS)*: 0.984

**Behavior Adherence**: 0.808 (over 616 applicable samples)

**Factual Grounding**: 0.893 (over 616 applicable samples)

**Single-Truth Recall**: 0.748 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.858

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.984
- **Precision**: 0.987
- **Recall**: 0.982
- **Accuracy**: 0.974
- TP=597, FP=8, FN=11, TN=120

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.916
- **Abstain Recall**: 0.938
- **Abstain F1**: 0.927
- **Specificity**: 0.982
- Abstain TP=120, FP=11, FN=8, TN=597


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (53 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.962 (n=158)
- **Grounding**: 0.944 (n=158)
- **Recall**: 0.847 (n=154)
- **CATS**: 0.935

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.977
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.882 (n=178)
- **Grounding**: 0.865 (n=178)
- **Recall**: 0.667 (n=156)
- **CATS**: 0.850

### Type 3: Conflicting Opinions

- **Samples**: 109 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.643 (n=98)
- **Grounding**: 0.878 (n=98)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.833

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.690 (n=145)
- **Grounding**: 0.905 (n=145)
- **Recall**: 0.743 (n=140)
- **CATS**: 0.833

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.892
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.802 (n=37)
- **Recall**: 0.703 (n=37)
- **CATS**: 0.788


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2147

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2
- **Claim**: Weight lifting causes a temporary increase in blood pressure during the actual lifting activity — up to 200/100 mmHg in some cases — but the long-term effects are generally positive: regular strength training can lower resting blood pressure and reduce cardiovascular risk

### Sample conflictingqa_114c06976f62

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The movement's own tenets reject dogma and do not require belief in the FSM to be considered a member, further complicating its classification as a 'legitimate' religion in the eyes of the law

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: All three views are represented in the evidence, reflecting a genuine and contested debate with no single definitive answer

### Sample conflictingqa_1d741779a31e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d3, d1
- **Claim**: Overall, the evidence is mixed: artificial sweeteners are permitted for diabetics by regulatory authorities and may help reduce sugar intake, but some studies raise concerns about their long-term effects on blood sugar control and cardiovascular health

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: The evidence is mixed. Some studies and experts claim wrist rests can reduce wrist extension angles by 10–15 degrees and lower reported discomfort by up to 30% when used correctly, while others argue that wrist rests are not necessary for good ergonomics and can be harmful if used improperly or for extended periods

### Sample conflictingqa_311fca0928d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The two protocols are also not mutually exclusive — dual-stack environments are common any security measures applied to IPv4 should equally be applied to IPv6

### Sample conflictingqa_3601f7480501

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d3, d1
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because report benefits of unlimited PTO (reduced stress, increased productivity, recruitment/retention benefits), while d2 and d3 argue it backfires (employees take less time off, burns employees)

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: In practice, the relationship between data volume and model performance follows a law of diminishing returns, where initial increases in data can lead to significant gains but eventually plateau, meaning that too little data can result in underfitting and too much data can lead to overfitting

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Moon has long been considered geologically inactive, with volcanism ceasing about 3 billion years ago and a magnetic field disappearing between 2.5 and 1 billion years ago

### Sample conflictingqa_3c835387fe6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5, d2
- **Claim**: Komodo dragons became extinct in Australia around 300,000 years ago, meaning they are no longer native to the continent today

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: On the other hand, other scientists argue that trophy hunting can harm population growth and perpetuate poaching, particularly when hunting is poorly managed or occurs at high levels that revenue from photo-tourism and other sources can be better for conservation in many cases

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: The conflict is further complicated by the fact that trophy hunting is not uniformly practiced—some areas see benefits while others see harms—meaning that any general answer is conditional on context

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d3, d1
- **Claim**: The conflict type is 'Conflicting opinions or research outcomes' because these documents present opposing interpretations of the same issue — one side citing empirical studies and the other side denying the gap's reality or attributing it to personal choice — producing a methodological and interpretive conflict that cannot be definitively resolved from the available evidence alone

### Sample conflictingqa_5233eab573e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d2, d3, d1
- **Claim**: These figures are consistent across multiple sources and confirm that captive tiger numbers significantly outnumber those remaining in the wild

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: CLS Bank decision and its progeny have placed too many limits on patent eligibility that international approaches vary significantly

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2
- **Claim**: The evidence is mixed and the answer depends on the stage of CKD and the dose used

### Sample conflictingqa_676188bf8139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The Chinese Lantern Festival is related to honoring ancestors but not exclusively about celebrating deceased ancestors; it is also about marking the first full moon of the new lunar year and promoting reconciliation

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The methodological divergence across these studies—differing fault models, earthquake magnitude thresholds statistical approaches—produces directly opposing research outcomes, making it difficult to reach a definitive conclusion

### Sample conflictingqa_962d8f5d5574

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict_type is 'Complementary information' because each document covers a distinct facet — definitive claim, scope of study, expert opinion surface ability — without providing a complete, authoritative answer

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Some plants can survive for extended periods without direct sunlight, but complete darkness is toxic to most plants

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Overall, the evidence is conflicting, with no consensus on whether human brain size is decreasing over time

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Nutritional yeast is not a complete protein source for vegans on its own. It is high in B12 and protein, but it is missing the amino acid methionine, making it incomplete for protein needs. The FAO/WHO recommend that all essential amino acids be present in protein sources nutritional yeast does not meet this standard on its own

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d3
- **Supporting Docs Found**: None
- **Claim**: While Sega itself has historically denied Jackson's involvement , this has been contradicted by multiple eyewitness accounts from developers and composers who worked directly with Jackson , making the evidence overwhelmingly in favor of his composition role

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, it is worth noting that manipulation is not unique to cryptocurrencies — traditional financial markets also experience similar issues — and that some manipulation attempts can be detected and mitigated through tools like on-chain analysis and vigilance

### Sample conflictingqa_e93e708d49a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: However, other sources note that if excess energy is stored in batteries before being sent to the grid, the overall energy return on investment for the solar-plus-battery system is 21% less than solar panels alone that the energy payoff period is also reduced if excess energy is wasted rather than returned to the grid

### Sample conflictingqa_f4693bea2c31

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d3
- **Supporting Docs Found**: None
- **Claim**: The debate reflects methodological divergence over how to define 'written language,' with some sources emphasizing the role of syntax and morphology, others the history of logographic systems still others the practical function of emoji in augmenting digital communication

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence supports both Dutch and British claims to have first discovered Australia, with the Dutch being the first Europeans to land on the continent in 1606

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: The evidence is mixed and the answer depends on the temperature at which yerba mate is consumed. Some studies suggest that drinking yerba mate at very hot temperatures may increase the risk of esophageal cancer, while other research indicates that yerba mate contains compounds that are cytotoxic to cancer cells. Epidemiological studies have found associations between hot mate consumption and increased risks of oral, esophageal laryngeal cancer, though methodological differences and varying risk estimates complicate a definitive conclusion

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1
- **Claim**: The conflict arises because d2's claim is limited to a specific nearby black hole and does not represent a general answer, whereas d1 and d3 provide a more general explanation of why black holes are invisible

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: The question of whether Mormons are Christian is genuinely contested. Some sources argue that Mormons can be considered Christian because they affirm belief in Jesus Christ and seek to follow him, while others argue that Mormon theology diverges from historic Christianity on matters such as the nature of God and the role of Scripture, leading to a rejection of the Mormon claim to be Christian

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The conflict is not a simple factual disagreement, but rather a methodological divergence over what criteria define 'life' and its place in the tree

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This ranking is consistent across multiple sources, including the most recent data from Ethnologue and Visual Capitalist, which place Hindi at #3 with 600 million+ total speakers , surpassing both Spanish (#4 with 560 million+) and Arabic (#5 with 450 million+)

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: She is described as the 'genius child' who overcame significant obstacles to achieve this honor her work focused on the dynamics and geometry of Riemann surfaces

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_2b9ba7e192e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Her presidency is confirmed to extend through July 2026, making her the most recent woman to become President of Peru

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: The latest major.NET version depends on which branch of.NET is meant:.NET Framework 4.8.1 is the latest for the Framework branch, while.NET 6.0 and.NET 7.0 are the latest for.NET Core/Standard

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The two are distinct and incompatible frameworks; this answer reflects the conflict between the two lines of development

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5d6e5db69928

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This finding superseded the previous record of ~1 million-year-old DNA from a mammoth tooth was confirmed through DNA analysis of 102 different genera of plants and nine animals, including mastodons, hares geese

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: This is consistent with the 2026 timeframe provided in the query

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This film, which follows a multi-generational American saga of political resistance, won six Oscars including Best Director and Best Adapted Screenplay, marking Anderson's first Academy trophy

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
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: He grew up in Bayonne, where his world was five blocks long this limited world fueled his imagination and love of reading

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: Martin's birthdate is September 20, 1948, making him 75 years old as of 2024

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: This is directly confirmed by a Congressional Record entry from July 28, 1971, which references his death in a boating accident on his 43rd birthday , a fact corroborated by the Cornell Chronicle's obituary of Rosenblatt, which notes that he died in July 1971

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Their most recent winning season was 2019–20, when they won 53 games

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d4, d2
- **Claim**: It is worth noting that while the saltwater crocodile (Crocodylus porosus) is often cited as the largest reptile by mass due to its broader body size, the Komodo dragon is actually the heaviest, as confirmed by the record of a 365-pound individual

### Sample freshqa_cf331ed7d09f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is consistent with Apple's current naming scheme, in which even-numbered versions (e.g., 26) reflect the year following the release year is further corroborated by the fact that earlier versions like Sonoma 14 are no longer the latest

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Ramadan 2026 began at sundown on Tuesday, February 17, 2026 ended at sundown on Thursday, March 19, 2026. The dates vary by country and year because many Muslims use a pre-determined date based on astronomical calculations rather than the local moon sighting the moon does not appear at the same time globally

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The statue's overall design and construction were commissioned by French historian Édouard de Laboulaye, who proposed the monument to commemorate the upcoming centennial of U.S. independence in 1876

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Madhuri Dixit was chosen as the brand ambassador for the Beti Bachao Beti Padhao campaign

### Sample qacc_2ed872eb1114

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The film's director, Benh Zeitlin, aimed to keep the production entirely authentic, using real water and live animals whenever possible, further confirming the Louisiana setting

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

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These differing counts reflect temporal updates: newer sources (Henley Index, 2026 guides) supersede older estimates (VWP count, Wikipedia's ~160), making the most recent data the most authoritative

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: These origins are not fixed in number and fall into three main classes — constitutive, flexible dormant — with their selection and recognition involving a combination of DNA sequence, chromatin structure epigenetic marks

### Sample qacc_883303a2d535

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The American Humane Society's behind-the-scenes report confirms that Nana was played by a dog of that breed, noting that trainers used verbal cues and hand signals to get her to perform specific actions on camera

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A second McDonald's location in Phoenix, which opened in 1954, still stands today , though its exact address is not specified in the available evidence

### Sample qacc_a1c73439eca9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: This season will be the last installment of the franchise, centering on Aurelio Casillas' return to reclaim his criminal empire and seek revenge

### Sample qacc_a4dde83c35da

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: The founder Zhu Yuanzhang, who proclaimed himself the Hongwu emperor, also established a bureaucracy based on literary examinations, though from the Yongle emperor onward, emperors increasingly relied on trusted eunuchs to contain the literati and maintain personal control

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: The incudomalleolar joint is a critical structure in the middle ear ossicular chain understanding its specific type is important for appreciating how hearing occurs in humans

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane (Carter Pewterschmidt/Lois's dad)

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: The Pokémon Company's first trading card game release is widely considered to have been the 1996 Japanese launch of the Pokémon Card Game, which was later followed by the 1999 English Base Set in North America

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: As a barred spiral, the Milky Way has a prominent central bulge surrounded by a disk with loose spiral arms, making it one of the most well-studied examples of this galaxy type

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is consistent with the broader understanding that the world's landmasses are generally arranged with Asia and Europe on one side of the Urals and Africa on the other, creating a natural barrier between the Pacific and Indian Oceans

### Sample situatedqa_geo_85af31651715

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: At least some pubs took steps to accommodate smokers before the ban came into effect the ban was described as a landmark occasion for the industry

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: The 1970 Act shifted the focus of air quality regulation from states to the federal government and empowered the EPA to set safe limits for six major pollutants, expanding to 189 pollutants today

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Nixon

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: In other regions, crops such as coffee, tea palm oil are also significant commercial agriculture more broadly encompasses a wide range of crops including grains, legumes root crops in addition to trees

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: For example, the Inflation Reduction Act of 2022 provided funding and tax incentives for states to invest in clean energy and environmental programs, reflecting a collaborative federal-state approach to environmental policy

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: At the local level, cities and municipalities also play an important role in environmental protection, as demonstrated by programs such as the EPA's Brownfields federal funding, which supports local revitalization efforts and environmental remediation

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: British General Sir William Howe led an army of about 16,000 British and Hessian troops against General George Washington's Continental Army of about 15,000 near Chadds Ford on Brandywine Creek in southeastern Pennsylvania, about 25 miles southwest of Philadelphia

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d1
- **Claim**: The British victory at Brandywine opened the way for them to occupy Philadelphia, the American capital, two weeks later

### Sample situatedqa_temp_3026b0491e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: He also created Hedwig's Theme, which is used in every film in the series and became the most recognizable song from the entire franchise

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Multiple election observers noted that the post-election environment was marred by allegations of widespread rigging, which PTI's incoming Prime Minister Khan promised to investigate

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: The USS prefix, which is also used on some U.S. Navy ships, stands for "United States Ship" other prefixes like USNS (United States Naval Ship) and USNV (United States Naval Vessel) are also used on different types of U.S. Navy vessels

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: All values are expressed in kilometres consistently across sources, with no contradictions in underlying data; the methodological divergence is the primary driver of these differences

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d3, d1
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Wilson Phillips is a vocal trio consisting of Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Their self-titled debut album, released in 1990, earned them a smash hit with the single "Hold On" and established them as a staple of 1990s pop music

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: This result, noting that Florida is the 10th franchise to win consecutive championships and the first since Tampa Bay in 2020-21

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is the lead ship of the Queen Elizabeth class, with her sister HMS Prince of Wales following in 2019

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: 164

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Rhys Ifans

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

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The Philadelphia 76ers last made the playoffs in 2021

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Their most recent playoff appearance was in the 2020-21 NBA season, where they played a six-game series against the Atlanta Hawks

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not justify a reliable answer

### Sample wikirevision_0046

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 tournament is the most recent edition Argentina's victory in 2022 makes it the current champion

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This was their third World Cup title, their first since 1986 their first won outside of Europe since 2002

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 season is currently underway, but as of the most recently available data from May 2026, Royal Challengers Bengaluru remain the champions from the previous season

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Earlier winners, such as Lionel Messi with a record eight titles, are also recognized for their contributions

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the most recent Cricket World Cup, the 2023 ICC Men's Cricket World Cup, defeating India in the final on 19 November 2023 at the Narendra Modi Stadium in Ahmedabad. This was Australia's sixth Cricket World Cup title, making them the most successful team in the history of the tournament

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. The official name change from Gurgaon to Gurugram was confirmed by the Haryana government this is consistently reflected across multiple sources including the older and newer Wikipedia revisions of the Gurgaon article as well as the Gurgaon (disambiguation) article

### Sample wikirevision_0111

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Meta Platforms was originally established in 2004 as TheFacebook, Inc. and was renamed Facebook, Inc. in 2005, but in 2021 it rebranded itself as Meta Platforms, Inc. to reflect its strategic shift toward developing the metaverse

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The 2026 tournament took place from 29 June to 12 July 2026, with Jannik Sinner defeating his opponent in the final to claim the title

### Sample wikirevision_0120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This victory was his second French Open title and fifth major overall

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: He defeated his opponent in the final to claim the title, which is confirmed across multiple sources

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This victory was his second French Open title and fifth major overall


================================================================================

*Report generated by CATS v2.0*
