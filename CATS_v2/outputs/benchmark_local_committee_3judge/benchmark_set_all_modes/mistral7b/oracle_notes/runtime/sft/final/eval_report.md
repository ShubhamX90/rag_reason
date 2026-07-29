# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 125 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.944 (over 736 samples)

**GR F1** *(used in CATS)*: 0.965

**Behavior Adherence**: 0.777 (over 611 applicable samples)

**Factual Grounding**: 0.828 (over 611 applicable samples)

**Single-Truth Recall**: 0.670 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.810

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.965
- **Precision**: 0.995
- **Recall**: 0.938
- **Accuracy**: 0.944
- TP=570, FP=3, FN=38, TN=125

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.767
- **Abstain Recall**: 0.977
- **Abstain F1**: 0.859
- **Specificity**: 0.938
- Abstain TP=125, FP=38, FN=3, TN=570


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (55 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.976
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.910 (n=156)
- **Grounding**: 0.901 (n=156)
- **Recall**: 0.805 (n=154)
- **CATS**: 0.900

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.934
- **Behavior**: 0.893 (n=177)
- **Grounding**: 0.742 (n=177)
- **Recall**: 0.558 (n=156)
- **CATS**: 0.781

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.646 (n=96)
- **Grounding**: 0.880 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.839

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.634 (n=145)
- **Grounding**: 0.864 (n=145)
- **Recall**: 0.689 (n=140)
- **CATS**: 0.794

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.568 (n=37)
- **Grounding**: 0.658 (n=37)
- **Recall**: 0.514 (n=37)
- **CATS**: 0.654


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2126

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
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: Parents objected to it being read in schools

### Sample conflictingqa_0c3c7b487766

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: However, iodine deficiency remains a global health concern iodine is still recommended for endemic goiter and iodine-deficiency disorders, meaning the appropriate dose is crucial

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3, d4, d5
- **Claim**: Overall, the evidence points to risks at both extremes—too little fluoride may not provide sufficient dental protection, while too much can be harmful—and calls for further research to better understand the full range of effects

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The overall picture is that IPv6 may offer a slightly better security foundation, but only when properly configured — and that the vast majority of security risks stem from human error rather than protocol vulnerabilities common to both versions

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d4, d5
- **Claim**: The retrieved evidence presents a genuine conflict. Some sources argue that real Christmas trees are more sustainable because they are biodegradable and can sequester carbon, while others argue that artificial trees are more sustainable if used for many years due to lower per-use emissions and waste reduction

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2
- **Claim**: The answer to whether prayer in schools is constitutionally permissible depends on the specific factual context — such as whether the prayer is organized by school officials, required participation disrupts the neutrality of the educational environment — rather than being a single definitive yes or no

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The question of whether patents should apply to software is a deeply contested normative debate with significant legal implications. On one hand, recent US Supreme Court rulings suggest a higher standard is applied to software patents — particularly when they merely implement known business methods — meaning they are not automatically granted simply because they are coded

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: However, experts caution that taking too much vitamin C — more than 2,000 mg per day — can increase the risk of kidney stones in people with certain medical conditions and may not be safe for those with hemochromatosis or other conditions , while some sources note that most people get enough vitamin C from their diet alone and that taking high-dose supplements is not necessarily beneficial for everyone

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d1, d2
- **Claim**: Yes, bees can fly in the rain — though their ability depends on species, genetics conditions. Most bees generally avoid flying in heavy rain due to the physical strain of wet wings (which can reduce lift and navigation stability), though they may fly in light rain or during emergencies like defending their hive or finding food

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d4
- **Claim**: Overall, the evidence suggests saturated fats are associated with increased heart disease risk through mechanisms such as raised LDL cholesterol, but the strength of this association is actively debated in the scientific community

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Organic farming is often claimed to be more environmentally sustainable than conventional farming, but a 2018 study in Nature Sustainability found that organic farms are approximately 20% less efficient in terms of crop yields than conventional farms, while a peer-reviewed analysis of global data similarly found that organic farming systems cannot compete with conventional systems in producing reliable yields

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Religious claims about the 'one true church' are rooted in differing interpretations of Scripture and historical precedence, not in empirical proof

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Overall, the extent to which multiculturalism is a hindrance to unity depends heavily on the specific context, the types of unity being considered the methodological perspectives of the researchers involved

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The debate remains unresolved, with some researchers calling for more direct neurophysiological evidence before concluding that fish truly feel pain in a human-equivalent sense

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: In short, while sexual contact is the primary route of transmission, gonorrhea is not exclusively transmitted sexually safe sex practices remain essential even among partners who are not having penetrative sex

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: Although volcanism is the leading hypothesis, some researchers also point to methane release from organic-rich sediments or permafrost as a co-contributor, particularly in the recovery phase the exact timing and mechanism of the trigger remain subjects of ongoing scientific investigation

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The overall consensus is that cold water is not a miracle solution for hair health that other hair care practices — such as using conditioners or oils — are more effective at creating shine than temperature rinsing alone

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: [[

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Sega officially uses Jackson's music in Sonic Origins, further corroborating his composition role

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: For example, the closest black hole to Earth, located about 1,560 light-years away, can only be seen indirectly via gravitational lensing, not by direct telescopic imaging

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Organizers and attendees alike emphasized these values, with Max Yasgur, the farmer who leased his land for the festival, coining the phrase 'half a million kids can come together for three days of fun and music and have nothing but fun and music'

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: This ranking is consistent across multiple sources that compare total speaker counts, including the Ethnologue database and various language rankings

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence indicates that Prince Harry's Duke of Sussex title was not officially stripped until early 2020, when he and Meghan Markle agreed to relinquish their HRH styles

### Sample freshqa_049cc3f14d5e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d4
- **Claim**: Earlier data from 2012 and 2007 is superseded by this more recent result

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: April 2, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d4, d5
- **Claim**: The first atomic bomb test in the United States took place on July 16, 1945, in New Mexico. The test, code-named Trinity, was conducted at a site 210 miles south of Los Alamos on the Alamogordo Bombing Range, in the New Mexican desert

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: Portugal won the 2017 Eurovision Song Contest, marking the country's first victory since 1964. The contest took place in Kyiv, Ukraine Portugal's winner was Salvador Sobral, who performed the song "Amar Pelos Dois"

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1, d4
- **Supporting Docs Found**: d2
- **Claim**: You can save up to $65 in cash back rewards on a $1,250 annual spend , making the net cost as low as $65 for some members , though full pricing details vary by region and membership level

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d2
- **Supporting Docs Found**: None
- **Claim**: This result, superseding earlier reports that listed 'Anora' (2024) or 'CODA' (2022) as the most recent winners, as those awards have since been surpassed by the 2025 ceremony

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
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: He has confirmed this birthplace himself, stating that his world "was five blocks long" growing up in that city

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3, d4
- **Claim**: Beijing is the first city ever to have hosted both the Summer and Winter Olympic Games

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel was won by *When We Were Real* by Daryl Gregory, published by Saga Press

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: A boating accident

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by Britannica, which identifies 26 books and notes that Hoover self-published many of them

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 1864

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Parents should instead focus on giving their child children's paracetamol or ibuprofen, which have been shown to effectively reduce fever

### Sample hotpotqa_0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d1
- **Supporting Docs Found**: d10
- **Claim**: This casting is consistently confirmed across multiple sources, with Kimberly Ann Hart being one of the most iconic characters in the franchise's early seasons

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d10, d6
- **Supporting Docs Found**: None
- **Claim**: This is consistently confirmed across multiple sources, including alumni records, sports announcements official Boston College documentation

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

### Sample hotpotqa_0073

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: However, the query also mentions El Nuevo Cojo, a different special interest publication; the evidence does not directly address whether El Nuevo Cojo is also owned by Time Inc. what its ownership status is

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

### Sample hotpotqa_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d10
- **Supporting Docs Found**: d6
- **Claim**: St James Street appears as a segment of Whitecross Street on the 1610 map of Monmouth by English historian and cartographer John Speed

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d4
- **Supporting Docs Found**: d7
- **Claim**: This appearance on the 1610 map is corroborated by later maps and historical records, which further place St James Street in the Wye Bridge Ward of Monmouthshire

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0d85f1089c4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: These are distinct, non-contradictory appointments covering different geographic scopes, with each state free to choose its own brand ambassador provided they align with the national campaign's objectives

### Sample qacc_2e1b5edb5e0d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This layer is identified by its translucent appearance and is functionally significant, as it provides an additional barrier against friction and shear forces in high-stress areas like the palms and soles

### Sample qacc_2f6d2647a424

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d5
- **Supporting Docs Found**: None
- **Claim**: This assignment was confirmed by multiple sources, with Pete Rose Jr. and Johnny Bench appearing as the primary Reds starters at that position for the 1975 season

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1, d5
- **Claim**: Phil Jackson (11 titles as a coach); Bill Russell (11 titles as a player)

### Sample qacc_4387048ed24f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Anne Bancroft won the Best Actress Oscar for 'What Ever Happened to Baby Jane?'

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: This formal union of New Brunswick, Nova Scotia the Province of Canada marked the completion of Canada's evolution from a self-governing British colony into a fully independent nation

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This location is corroborated by additional sources confirming the same address the building itself has been recognized as a historic landmark

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: While d1 and d4 provide the regional context d2 offers Uruguay-specific data, the query asks about the 'dominant' ethnic group across all Southern Cone nations regrettably the evidence does not sustain a definitive answer for Argentina

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: This victory gave the Red Sox their first AL East title since 2013 and set the stage for their historic postseason run

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Russ Ballard

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d4
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912, when President William Taft signed the New Mexico statehood bill into law

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: This admission order is consistently confirmed across multiple sources, with New Mexico becoming the 47th state alongside Arizona the official New Mexico government website also confirms this date

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The event is commemorated by the official 2016 White House Christmas ornament, which depicts a vintage fire truck battling the blaze

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Some low-quality sources incorrectly claim the first cards were released in 1995 , but this conflates the release of the Pokémon video games with the TCG, which did not launch until 1996

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: The Milky Way's classification as a barred spiral galaxy is unanimously supported by astronomers, superseding earlier debates about its exact Hubble type

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d4, d2
- **Claim**: Henry Burton

### Sample qacc_ec5b0067c29a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d2
- **Supporting Docs Found**: None
- **Claim**: All available sources consistently confirm this location, with no contradictions across official records or historical accounts

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d2
- **Claim**: An initialism is an abbreviation formed from initial letters, pronounced as a series of letters (e.g., DNA, RT-PCR), while an acronym is pronounced as a word (e.g., NATO, UNESCO)

### Sample situatedqa_geo_082db791e263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3, d4, d5
- **Claim**: [[

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: 407,000

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

### Sample situatedqa_geo_7222d6123c27

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The formal shift to Delhi was completed in 1931, when the new capital was inaugurated there

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d5
- **Supporting Docs Found**: None
- **Claim**: The program was originally limited to workers in commerce and industry, with benefits based on average covered earnings the first monthly check was issued to Ida M. Fuller of Vermont in January 1940

### Sample situatedqa_geo_779fd84224fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: The fleet, which had set sail from Portsmouth, England in May 1787, carried approximately 750–780 convicts, along with crew members, soldiers families

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: In addition to this core federal structure, the U.S. also follows a 'republican form' of government at the State level, where State governments are modeled after the Federal Government and share its three-branch structure

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2
- **Claim**: It is worth noting that some reports citing lower figures (around 600,000) reflect different definitions of 'village' and data sources that predate Census 2011 , making the 2011 data the most current and comprehensive available

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: This landmark legislation shifted air quality regulation from a state-level focus to a federal government purview, empowering the newly created Environmental Protection Agency to determine safe limits and regulate six major air pollutants

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: The California state flag features a grizzly bear, which is also the official state animal of California. The bear on the flag is the California grizzly bear (Ursus arctos californicus), an extinct population of the brown bear its inclusion on the flag dates back to 1846 when California was part of Mexico

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Elsewhere, the transition was shaped by different factors: in France, tea drinking persisted and only began to decline in the late 19th century , while in Italy the shift to coffee was driven by immigration and industrialization in the 19th century

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4, d5, d2
- **Supporting Docs Found**: d3
- **Claim**: The FOMC is a body within the Federal Reserve System, composed of seven members of the Board of Governors and four rotating presidents of the twelve regional Federal Reserve Banks, who meet approximately every six weeks to discuss economic conditions and make policy decisions

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: At the local level, cities and communities can further influence environmental outcomes through their own policies and programs, making the federal-state-local nexus a critical framework for understanding how environmental policy is set and enacted today

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This victory gave LSU their eighth national title, their first since 2023

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Mort is a Goodman's mouse lemur, a small primate native to Madagascar, though a spin-off series reveals he is also part bear, spider starfish

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2
- **Claim**: [[

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Opposition parties also claimed the military intervened to ensure PTI's victory, an allegation denied by both the PTI and the army

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: In modern naval classification, 'SS' is used as a prefix for submarines specifically, distinguishing them from other warship types like battleships (BB) or auxiliary vessels

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2, d3, d4, d5
- **Claim**: Example of kennings from the Battle with Grendel

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 kilometres (approximately 37,604 miles)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This figure is further confirmed by UC Berkeley energy economist Severin Borenstein, who notes that about $1.80 of the Bay Area's average gas price of $6 per gallon is attributable to higher taxes and environmental fees

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

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [[

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d1, d5
- **Claim**: It was originally formed from the Old German name Gerhard, which itself derived from the Proto-Germanic elements 'gēr' (spear) and 'hard' (hardy, brave strong)

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d5, d2
- **Supporting Docs Found**: None
- **Claim**: This origin is corroborated by its first appearance in the Domesday Book in 1086 as the Latin form Gerardus by its use as a forename among French, English Scottish families

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d4, d5, d2
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin and Perry Go Large

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: This is consistently confirmed across multiple authoritative sources, including the official Charlotte city website and a Smithsonian magazine article on the city's history

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

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: 2021

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. The snippet directly states that Alphabet Inc. is the parent company of Google, which is supported by additional context that Alphabet acquired Wiz in March 2026, making it clear that Alphabet owns Google as a wholly owned subsidiary. The second snippet further confirms this, explicitly stating that Google is reorganized as a wholly owned subsidiary of Alphabet Inc

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Argentina (defeated Italy 3-2 in a penalty shootout at the Rose Bowl, July 1994)

### Sample wikirevision_0088

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1, d2
- **Claim**: This is consistent across multiple sources, including the official Wikipedia entries on both the Leader of the Labour Party and the Deputy Leader of the Labour Party, as well as external analyses

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the latest ICC Men's Cricket World Cup, the 2023 edition, by defeating India by six wickets in the final held at the Narendra Modi Stadium in Ahmedabad. This victory gave Australia their sixth World Cup title, with India as the runner-up. Multiple sources confirm that the 2023 tournament was the most recent edition as of the time of writing Australia's win there makes them the latest champion

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was confirmed by the Haryana government in 2022 the older name of Gurgaon is now used primarily as a historical or regional term

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore is officially called Bengaluru. The city officially changed its name from Bangalore to Bengaluru on 1 November 2014, as confirmed by the Bangalore City Municipal Corporation resolution of that date. As a result, Bengaluru is the current official name of the city, superseding the former name of Bangalore

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The current Wimbledon men's singles champion is Jannik Sinner, who defeated Alex de Minaur in the 2026 final to win his first major title

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 Wimbledon Championships is the 139th edition of the tournament, taking place at the All England Lawn Tennis and Croquet Club in Wimbledon, London from 29 June to 12 July 2026


================================================================================

*Report generated by CATS v2.0*
