# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 98 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.950 (over 736 samples)

**GR F1** *(used in CATS)*: 0.970

**Behavior Adherence**: 0.790 (over 638 applicable samples)

**Factual Grounding**: 0.895 (over 638 applicable samples)

**Single-Truth Recall**: 0.739 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.849

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.970
- **Precision**: 0.952
- **Recall**: 0.988
- **Accuracy**: 0.950
- TP=601, FP=30, FN=7, TN=98

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.933
- **Abstain Recall**: 0.766
- **Abstain F1**: 0.841
- **Specificity**: 0.988
- Abstain TP=98, FP=7, FN=30, TN=601


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.938
- **GR F1** *(used in CATS)*: 0.960
- **Behavior**: 0.892 (n=167)
- **Grounding**: 0.937 (n=167)
- **Recall**: 0.841 (n=154)
- **CATS**: 0.907

### Type 2: Complementary Info

- **Samples**: 221 (34 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.941
- **GR F1** *(used in CATS)*: 0.964
- **Behavior**: 0.904 (n=187)
- **Grounding**: 0.860 (n=187)
- **Recall**: 0.676 (n=156)
- **CATS**: 0.851

### Type 3: Conflicting Opinions

- **Samples**: 109 (9 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.954
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.620 (n=100)
- **Grounding**: 0.868 (n=100)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.821

### Type 4: Outdated Info

- **Samples**: 158 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.667 (n=147)
- **Grounding**: 0.939 (n=147)
- **Recall**: 0.711 (n=140)
- **CATS**: 0.827

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.892
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.788 (n=37)
- **Recall**: 0.689 (n=37)
- **CATS**: 0.781


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2248

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
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Outside the U.S., some countries offer distinct sui generis protection regimes for fashion designs, such as the European Union's Creative Designs Directive, which grants protection to new designs for three or five years

### Sample conflictingqa_0a05aabca56a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Unlike cartoons, which are typically aimed at younger children, anime spans a broader age range from childhood through adulthood across various genres

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d2, d4, d5
- **Claim**: Judaism is generally categorized as a religion, but it also functions as an ethnicity or ancestral identity — a view explicitly endorsed by Chabad Lubavitch scholarship

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Yes — cows have one stomach with four distinct compartments (the rumen, reticulum, omasum abomasum), not four separate stomachs

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: Some sources say wrist rests can reduce wrist pain by 30%, but experts say they don't always help and can have serious risks

### Sample conflictingqa_311fca0928d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, IPv6's larger address space facilitates more granular security policies, such as allocating random addresses within subnets to reduce scanning attacks, though this comes with its own trade-offs

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, primarily because they absorb carbon dioxide while growing and can be recycled as mulch or wood chips, whereas artificial trees are made from plastic and metal and release pollutants during manufacturing

### Sample conflictingqa_42d60ecaee9f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: A key debate centers on whether emoji's multi-modal capabilities sufficiently transcend the limitations of text whether they remain constrained as a form of 'pictographic decoration' , underscoring that the question of emoji as a new language remains contested ground between emerging practice and traditional linguistic criteria

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, critics counter that even ethically conducted hunts can have unintended negative consequences — such as the 2015 case of Cecil the lion, where a hunter's actions sparked widespread condemnation despite the operator claiming compliance with regulations — and that blanket bans may risk increasing poaching pressures when communities lack alternative income sources

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Progressives and feminist advocates, on the other hand, contend that underlying sexism and discriminatory workplace practices also contribute significantly to the disparity, making the debate one of competing interpretations rather than a settled empirical conclusion

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2
- **Claim**: KDIGO guidelines recommend sodium bicarbonate orally to normalize blood bicarbonate levels when serum bicarbonate is less than 18 mEq/L, but the supplementation remains uncertain in patients with normal serum bicarbonate levels

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2, d5
- **Supporting Docs Found**: None
- **Claim**: While it is not explicitly confirmed as the absolutely deadliest eruption in all of recorded history across every source, it is consistently ranked among the deadliest no other eruption is cited in the evidence as surpassing it

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: That said, some treatments can temporarily mask the appearance of split ends — such as protein-based conditioners that leave positive ions to reduce the hair's negative charge — though these effects typically last only until the next shampoo

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Major American ISPs such as AT&T, Comcast Verizon have stated that customers can opt out of data collection, though they do not always honor this choice consistently

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3
- **Claim**: When rain is light, bees can still forage and move about some species like bumblebees appear more tolerant of poor weather conditions than others

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Additionally, some researchers note that the effects depend on the type of unsaturated fat substituted for saturated fat: replacing SFAs with primarily n-6 PUFAs has no clear effect, while replacing them with n-3 PUFAs (found in fish oil) may be particularly protective

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: However, organic farming practices have significant advantages in terms of environmental sustainability — they generate fewer greenhouse gas emissions, require less fossil fuel input build soil health over time

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: Some critics argue that the path to global food security requires balancing these trade-offs by adopting more organic-style practices while also improving conventional farming efficiency, particularly through reduced chemical use and better waste management

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A peer-reviewed study found that prophylactic knee braces reduced the risk of non-contact ACL tears by 80% in female high school athletes a clinical trial confirmed that functional knee braces provide statistically significant protection against re-injury in individuals with a prior ACL tear , though researchers note that the evidence base remains incomplete

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d2
- **Claim**: Some sources note that neutering can decrease the risk of ovarian and breast cancers, prostate disease pyometras, while also reducing socially unacceptable behaviors in dogs that the procedure can help pets maintain healthier weights and prevent hormonal-driven diseases

### Sample conflictingqa_962d8f5d5574

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Some notable exceptions include burrowing snakes, which lack the physical adaptations required for aquatic locomotion

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Michael Socolow, a scholar cited by PBS's American Experience documentary, further challenges the claim that the press exaggerated the public reaction, noting that memory and media reporting can distort how events are remembered

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Mercury proxies from North Sea sedimentary cores show pulsed volcanism preceded the early PETM, while carbon isotope data confirm a comparatively heavy carbon source consistent with organic matter deposition linked to volcanic perturbation

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, the PETM onset also coincides with a mercury low, suggesting at least one other carbon reservoir was released in response to initial warming, indicating complex feedback mechanisms rather than a single definitive trigger

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: Harvard urologist Dr. Anthony L. Komaroff notes that while tea contains oxalate, increased fluid intake—including from tea—decreases the risk of calcium oxalate stones by making urine more dilute

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4, d5
- **Supporting Docs Found**: d3
- **Claim**: However, experts differ: Dr. John Milner warns that iced tea is among the worst drinks for people prone to forming the most common type of kidney stones a Reddit thread notes that research on green tea's effects on stone risk is inconclusive

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This reduction is further corroborated by a study of 276 fossil estimates, which shows a parallel decrease in body size alongside the cognitive shift to more abstract 'symbolic information processing'

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: However, comets are recognized as another potential source of primitive material some researchers suggest that approximately 5–10% of observed meteors and 38% of observed fireballs may have cometary origins

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, cultural and historical factors vary significantly — traditional American attitudes toward death have long been described as un-American, with death considered an affront to individual rights a 1991 Gallup poll found that Americans rarely thought about death

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2
- **Claim**: Protestant Christians generally distinguish between biblical inerrancy (no factual error) and infallibility (no theological error), though some denominations treat them as equivalent. The Roman Catholic Church defines the Bible as true and without error in matters of faith and practice, though it acknowledges human imperfection and historical discrepancies in the details

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: This interpretation places humans on a branch of the primate family tree alongside other apes, rather than viewing them as descended from any single modern ape species

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Over the following decades, other Dutch navigators such as Dirk Hartog, Frederik de Houtman Abel Tasman charted significant portions of Australia's western, southern northern coasts, respectively

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: While the Dutch were not the very first to encounter Australia — that distinction belongs to the Macassan fishermen who traded along the northern coast for centuries before the Europeans arrived — their systematic exploration and mapping of the continent laid the groundwork for future colonization

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: It began at sundown on the first night of the Hebrew month of Nissan, as observed across the global Jewish community

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_25b286cb2af1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: The idea that Venus could have had a moon in the past is also discussed, as the planet may have once captured a moon that later collided with Venus

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Based on the available evidence, Harry Maguire has not won the Ballon d'Or. The document notes that any claim suggesting he won the award is likely a misleading video title, as his Wikipedia page lists his EFL Cup as his first trophy with Manchester United in 2023–24, implying no Ballon d'Or was awarded to him

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This film took home six Oscars, including top honors for Best Director and Best Adapted Screenplay, marking a long-awaited coronation for Anderson

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: The 98th Oscars were held on March 5, 2026, making this the latest official Best Picture winner

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: He was 43 years old at the time, tragically losing his life just two years after the publication of Minsky and Papert's influential book that critiqued his work

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1, d4
- **Claim**: Queen Elizabeth II of England died on 8 September 2022

### Sample freshqa_c7ac9d61059a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 51,630

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: However, historical data from other sources indicates that her tenure at the top has been relatively brief, with Iga Świątek previously holding the position for 125 weeks before Sabalenka surpassed her

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: No permanent cure for cancer has been developed; ongoing research with immunotherapy and gene editing is exploring potential future cures

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d7
- **Supporting Docs Found**: None
- **Claim**: This figure represents the culmination of multiple AI-assisted phases of discovery: an initial 303 new geoglyphs identified in a 2024 PNAS study , followed by an additional 248 found in 2023–2024 further augmented by over 300 new ones revealed by AI over a six-month period in 2024

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d8
- **Claim**: Lucas di Grassi (born 11 August 1984)

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d7, d8, d2, d6, d4, d5
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Princess of Wales Theatre

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: As a coach: Phil Jackson (11) — see d2 and d3

### Sample qacc_44b315f6f4bb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: During the Queen's lying-in-state following her death, the Imperial State Crown, Sovereign's sceptre orb were temporarily placed in Westminster Hall , while the majority of the Crown Jewels remain permanently housed in the Tower of London

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: October 1, 1968

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: 5.88

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Full assembly was completed by the end of 2011, with the final pressurized module, the Russian Nauka, arriving in July 2021

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The official ceremony took place on January 6, 1912, when President William Taft signed the New Mexico statehood bill

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d1
- **Claim**: The blaze required 130 firefighters from 19 engine companies and four truck companies to contain no one was injured

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_d60bf850c4ff

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Specific product pages further illustrate its market presence: Dairy Milk is sold in at least 100 countries the brand's range includes popular items such as Dairy Milk, Cadbury Eggs, Roses, Bournville Wispa, all of which are available across its extensive global network

### Sample qacc_d60bf850c4ff

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: While the exact total number of countries is not specified in the documents, the combination of specific regional data and the 'over 50 countries' claim provides a clear picture of Cadbury's international footprint

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d4, d2
- **Claim**: Based on the available evidence, Teddy Altman married Henry Burton on Grey's Anatomy

### Sample qacc_e7318f6f3bbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: As a bonus fact, the longest word in English with only one repeated vowel is **strengthlessnesses**, further illustrating the complexity of English phonetics

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: Vernon Wells

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Specific industries also have their own requirements — for example, drivers applying for a CDL in New Hampshire must be at least 21 those under 20 are restricted to certain conditions

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: Globally, minimum ages can differ significantly by country, with some requiring drivers to be 17 or older

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: The retrieved evidence indicates that World War II was fought across multiple fronts, with the Eastern Front being the most significant

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, most states do not impose general sales tax on gasoline, though some may add other fees or excise taxes

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Recent data shows immigrants are coming from a wide range of countries, with no single source dominating

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: This legislation established the Environmental Protection Agency (EPA) and granted the federal government enforcement authority to regulate air pollution, replacing earlier laws that were largely managed at the local and state levels

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: President Kennedy was the first U.S. president to send military advisers to South Vietnam, authorizing the deployment of 16,000 American troops to help stop the north from invading the south

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: This shift was further reinforced through the late 19th and 20th centuries by immigration patterns and coffee's dominant supply chain infrastructure, though the exact administrative 'switching' date is contested

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The FOMC consists of twelve members — including seven members of the Board of Governors and four rotating Reserve Bank presidents — and meets approximately every six weeks to set interest rate and balance sheet policy, with a primary goal of achieving stable prices and maximum employment

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: In the United States, environmental policy is primarily set at the federal level, where the U.S. Environmental Protection Agency (EPA) develops pollution control standards and the White House Council on Environmental Quality coordinates policy across agencies

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: However, the U.S. system also allows for significant state-level variation, as individual states can adopt their own air and water quality rules that differ from federal standards local governments can implement their own regulations within their jurisdictions

### Sample situatedqa_temp_35156c8be377

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Seychelles (by GDP per capita / PPP); Nigeria (by nominal GDP, 2021); South Africa (by 2024 IMF GDP)

### Sample situatedqa_temp_40e6764f611f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: He appears as a side character in the main Madagascar films and as a major character in The Penguins of Madagascar

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: The latest stable major version is Android 15 (released September 3, 2024), which was first released to a device by Vivo on September 30, 2024

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, the show is often confused with the similarly titled 'Six' series, which Season 2 actually refers to

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: 1980 (established as a national park)

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d4
- **Claim**: While the Dragon Ball Fandom page mentions that the form is playable in Ultimate Battle 22 one source speculates Goku achieved the form during meditation in the afterlife , the most definitive answer places the transformation within the context of the 245th episode of the anime

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken is the head coach of the Cleveland Browns

### Sample situatedqa_temp_6f777dda5314

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: However, the Browns' official coaches roster page still lists Kevin Stefanski (2020–2025) through the 2025 season, reflecting that Stefanski's tenure has concluded and Monken has taken over as the current head coach

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The prefix is used differently across ship types: S/S or S.S. refers to a **sailing ship** USS is the official title given to ships that belong to the **United States Navy**

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: For Beowulf himself, the poet employs kennings such as "sure-footed fighter" to describe his agility, "shieldings' hero" for his martial prowess "prince of goodness" to highlight his leadership and virtuous qualities

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1
- **Claim**: Older estimates vary further, reflecting the city's consistent growth over time

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: September 1967 (released in the UK on Epic Records in September 1967; charted in the US in 1967)

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: His batting performance in the third Test at the Wanderers was also noted, as he recorded his top two longest innings in that series

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Wilson Phillips is an American vocal trio consisting of Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The group was formed in Los Angeles in 1989 and quickly rose to fame with hits such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Their music is characterized by smooth, melodic vocal harmonies they are renowned for blending pop, pop rock soft rock genres

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: This figure was confirmed when Erton Köhler was elected the new president of the General Conference in 2025

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: A broader timeline of Inca civilization places its origins in the early 13th century (circa 1200), when Manco Cápac founded the Kingdom of Cusco — the precursor to the full Inca Empire — before it grew to become the largest empire in Pre-Columbian America

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The United States has hosted the Olympics nine times: four Summer Games and five Winter Games

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d3
- **Supporting Docs Found**: d2
- **Claim**: However, the ship was formally declared operational in 2020, having conducted its first sea trials in 2017 and completing formal operational acceptance that year

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6, d5
- **Supporting Docs Found**: d2
- **Claim**: As a result, the carrier's service entry is commonly cited as 2020, reflecting its readiness for active duty rather than its initial commissioning

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d1, d5
- **Claim**: It is traditionally interpreted as meaning 'spear-brave,' reflecting its Germanic linguistic roots

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The name was first recorded in the Domesday Book of 1086, originating from the Anglo-Saxon tribes of Britain is also found in Haiti as a result of colonial and post-colonial migration

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The WTO has 166 members

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_eed796e2fbed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d5
- **Claim**: While other sources such as Golf Ranking Stats and the YouTube channel Golf Channel Podcast also identify Scottie Scheffler as the current top-ranked golfer in the world , it is worth noting that d5 references Russell Henley as number one for a specific tournament rather than the overall season rankings

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Participants can collect up to ten physical game pieces per day, which are scanned in the McDonald's app to reveal prize amounts ranging from free food to millions of dollars

### Sample trust_align_008

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Jessica Lange is a member of the cast of *American Horror Story* (Season 2)

### Sample trust_align_068

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Here is the synthesized answer using only the ground truth evidence from the retrieved documents:
[

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The key biological difference lies in the underlying mechanisms: liver donation triggers a robust repair response that replaces damaged tissue, whereas alcohol abuse induces chronic inflammation and fibrosis that outpaces the liver's natural healing processes

### Sample trust_align_143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Cyril Ramaphosa (as of 2018 following Jacob Zuma's resignation); no current president named in the evidence

### Sample trust_align_175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Saskatoon, Canada experienced temperatures colder than both the North and South Poles on occasion , suggesting local conditions and variability can complicate general comparisons

### Sample wikirevision_0002

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d13, d12
- **Supporting Docs Found**: d4, d3
- **Claim**: This is corroborated by additional sources noting that following Elon Musk's acquisition, Twitter was renamed X

### Sample wikirevision_0003

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: In April 2023, Twitter merged with X Holdings and ceased to be an independent company, becoming a part of X Corp. This follows a prior rebrand in which Twitter was renamed X in late 2022, superseding its former name that had been in use since 2006–2023

### Sample wikirevision_0049

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Earlier editions of the Wikipedia article on the IPL documented Mumbai Indians and Chennai Super Kings as most successful overall, but these references reflect historical data superseded by the 2026 season

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia (won the 2023 Cricket World Cup). The 2023 ICC Men's Cricket World Cup was the 13th edition of the tournament, hosted in India from 5 October to 19 November 2023, with Australia defeating India by six wickets in the final to claim their sixth World Cup title. As the 2027 edition is scheduled for 2027, Australia remains the latest champion


================================================================================

*Report generated by CATS v2.0*
