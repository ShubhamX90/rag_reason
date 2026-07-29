# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 93 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.951 (over 736 samples)

**GR F1** *(used in CATS)*: 0.971

**Behavior Adherence**: 0.792 (over 643 applicable samples)

**Factual Grounding**: 0.908 (over 643 applicable samples)

**Single-Truth Recall**: 0.752 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.856

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.971
- **Precision**: 0.945
- **Recall**: 0.998
- **Accuracy**: 0.951
- TP=607, FP=35, FN=1, TN=93

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.989
- **Abstain Recall**: 0.727
- **Abstain F1**: 0.838
- **Specificity**: 0.998
- Abstain TP=93, FP=1, FN=35, TN=607


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.934
- **GR F1** *(used in CATS)*: 0.957
- **Behavior**: 0.899 (n=168)
- **Grounding**: 0.930 (n=168)
- **Recall**: 0.799 (n=154)
- **CATS**: 0.896

### Type 2: Complementary Info

- **Samples**: 221 (33 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.941
- **GR F1** *(used in CATS)*: 0.964
- **Behavior**: 0.947 (n=188)
- **Grounding**: 0.900 (n=188)
- **Recall**: 0.724 (n=156)
- **CATS**: 0.884

### Type 3: Conflicting Opinions

- **Samples**: 109 (5 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.927
- **GR F1** *(used in CATS)*: 0.960
- **Behavior**: 0.635 (n=104)
- **Grounding**: 0.889 (n=104)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.828

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.589 (n=146)
- **Grounding**: 0.910 (n=146)
- **Recall**: 0.739 (n=140)
- **CATS**: 0.809

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.757 (n=37)
- **Grounding**: 0.896 (n=37)
- **Recall**: 0.716 (n=37)
- **CATS**: 0.842


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2264

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
- **Claim**: Some studies further note that different types of fertilizer can influence nematode communities: organic matter favors bacteria that increase free-living nematodes, while inorganic fertilizers can negatively affect the nematode community by reducing the abundance of fungivores, omnivores predators

### Sample conflictingqa_114c06976f62

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d2, d3, d5
- **Claim**: Overall, the status of Pastafarianism as a legitimate religion remains contested, depending on legal jurisdiction and interpretation.
[

### Sample conflictingqa_311fca0928d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Additionally, IPv6's larger address space facilitates more granular security measures, such as allocating random addresses within subnets to reduce scanning attacks its header design supports a cleaner separation between encryption metadata and payloads, which some analysts view as an improvement over IPv4's IPsec implementation

### Sample conflictingqa_34610226ee3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: However, proponents argue that future advances in cloning technology, such as those demonstrated by the creation of a gene-edited pig with a human cell nucleus , could one day overcome this barrier the park would also require a sustainable ecosystem of prehistoric flora — including ferns, cycads, gingkos Wollemi pines — to support the herbivores and carnivores

### Sample conflictingqa_411445406724

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2
- **Claim**: These conflicting perspectives reflect both the well-documented fossil abundance of cycads and the emerging research clarifying that other plant lineages also played central roles in Mesozoic ecosystems

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Opponents counter that poorly regulated hunting perpetuates harmful cultural narratives, causes real harm to individual animals including orphaning cubs and spreading disease that revenue generated is often siphoned off by corrupt officials rather than funding actual conservation efforts

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Research further complicates the debate: a Royal Society study found that trophy hunting can shape public perceptions of wildlife management in pragmatic rather than purely sentimental terms , while the IUCN acknowledges both benefits and harms, calling for context-specific decision-making rather than a universal ban

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Other candidates include Mount Mount Pelee (1902) and Krakatoa (1883) , which also caused substantial fatalities, making the question of which was the single deadliest a matter of ongoing scholarly debate

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Internationally, regulations vary: in Europe, the General Data Protection Regulation (GDPR) typically requires explicit user consent before data can be sold, though some countries like the UK have different frameworks in Australia, the Australian Privacy Principles generally permit data sharing without consent if it is in accordance with the organization's privacy collection statement

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: However, experts note that high-dose vitamin C supplementation is generally unnecessary for the average person and may carry risks such as kidney stone formation in susceptible groups

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A systematic review published in the Journal of the American Heart Association found that replacing saturated fats with polyunsaturated fats is associated with a 2–3% lower risk of cardiovascular death, but the evidence remains inconclusive for the general population

### Sample conflictingqa_a3980a2921cf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: For example, lightweight oils like grape seed or jojoba are ideal for fine hair without weighing it down, while richer oils like coconut or castor are better suited for coarse or curly hair

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, some researchers present competing hypotheses: a 2019 study found that mercury anomalies coincide with the PETM onset but do not clearly point to a single source, suggesting multiple reservoirs were involved a 2021 review of the PETM's causes notes that volcanic activity may have acted as a trigger but cannot explain the full magnitude of the carbon release observed

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Additionally, the PETM's carbon source remains under active scientific debate, with some studies favoring the release of methane from ocean sediments or organic-rich permafrost as a complementary or alternative mechanism the exact sequence of events — whether volcanism preceded or followed the initial carbon release — continues to be contested

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: The prevailing scientific consensus

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d1
- **Claim**: However, some researchers argue that cometary meteoroids may make up approximately 95% of observed meteors and 38% of observed fireballs a notable minority of scientists still consider comets a plausible source for some meteorites, such as the Leonids associated with Comet Tempel-Tuttle

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A 1991 Gallup poll found that Americans almost never think about death Arnold Toynbee observed in 1973 that death is considered 'un-American,' with the culture inferring that 'you can accomplish anything — even beating out death'

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d1
- **Claim**: However, scientists have not consistently recorded instances of animals acting strangely days before an earthquake the prevailing consensus holds that reliable, replicable evidence for animal earthquake prediction remains unproven

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3, d5
- **Claim**: However, some scholars argue that emoji function as paralinguistic cues or punctuation rather than replacing written language entirely they are increasingly being treated as word-like units in digital contexts — raising ongoing debate about whether they represent a distinct form of written expression

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: PAHs (polycyclic aromatic hydrocarbons), known carcinogens also found in grilled meat and tobacco smoke, are present in yerba mate and can contribute to cancer development , though the extent of their impact varies across studies

### Sample conflictingqa_f97fef94decc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: A practical balance is therefore recommended: moderate VR use and following the 20-20-20 rule (looking away from the screen every 20 minutes to something 20 feet away for 20 seconds) can help minimize temporary strain, while users reporting persistent or worsening vision symptoms should consult an eye doctor

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: It ran from April 1 to April 9, with the first seder observed on the evening of April 1

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris won the 2020 Formula 1 World Drivers' Championship

### Sample freshqa_263eca8e024e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Dangal holds this record by a wide margin over its nearest competitors, including Baahubali 2: The Conclusion ( ₹1,810 Crore) and Jawan ( ₹1,148.32 Crore) , which were both released after Dangal in 2016

### Sample freshqa_2877cf4bd00f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: Britannica's biography notes he was 78 upon his first term's inauguration , while older sources capture his age at various points throughout his life

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Android 16 — released December 2, 2025

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Ukraine has consistently defended itself against Russia's attacks, with the two sides suffering tens of thousands of casualties and causing widespread destruction

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: The first season premiered on November 12, 2019, the second on October 30, 2020 the third on March 1, 2023

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: By December 2022, the target range stood at 3.50%–3.75%, having risen sharply from a range of 1.50%–1.75% at the end of 2021

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: JPMorgan Global Research further corroborates that the Fed kept rates steady at its March 2026 meeting, confirming no rate cuts occurred during 2022

### Sample freshqa_5eb89aae15f3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This ranking is corroborated by other sources, which note that Kantara has continued to earn money through word of mouth and crossed the ₹250-crore mark in global earnings

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has not won the Ballon d'Or

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Older sources referencing the 2024 winner () reflect information that has been updated by the official 2025 results

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Their playoff record that same year was also poor, as they were swept in the first round by the Miami Heat

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d1, d4
- **Claim**: Her reign of 70 years and 214 days is the longest of any British monarch she was succeeded by her eldest son, Charles III

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Goodreads also lists 172 editions of her novels, reflecting her prolific output

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: Jeff Bezos did not sell Amazon — he sold Amazon shares

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: There is no permanent cure for cancer

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Some vaccine series require multiple doses for children as young as 6 months the CDC recommends that all children ages 6 months and older be fully vaccinated

### Sample hotpotqa_0031

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d10
- **Supporting Docs Found**: None
- **Claim**: Kimberly Ann Hart is remembered as the first Pink Ranger and had the longest tenure of any female ranger in the series' history

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6, d10
- **Supporting Docs Found**: d3
- **Claim**: While the query's specific claim about St James Street is supported by multiple sources , the additional requirement to identify the mapmaker's period of expertise is met primarily through the broader biographical context provided by d3 rather than any direct attribution in the referenced documents

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d7, d5, d6, d8
- **Claim**: 506

### Sample qacc_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d6, d4, d2
- **Claim**: Sheldon Collins (whose real name is Sheldon Golomb) played Arnold Bailey, Opie's friend, on The Andy Griffith Show

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d1
- **Claim**: This milestone marked the USSR's dominance in the ongoing competition to achieve space firsts, a fact corroborated by additional reporting on the mission

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë (as the High Lord of the Valar)

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Jessica Biel (season 1); Alice Kremelberg (season 4) — IMDb full credits confirm these two actresses appear alongside Bill Pullman in different seasons, though neither is explicitly identified as his wife in the provided evidence

### Sample qacc_6af6e8cb8f34

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d5
- **Claim**: However, other sources complicate this view: one document attributes the first tree to Martin Luther's compatriot, a German royal who brought the custom to Britain before Prince Albert popularized it in 1841 a more recent analysis places the introduction firmly at Windsor Castle in 1800, with Prince Albert's 1841 effort being a later expansion of the tradition

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: October 1, 1968

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: As the AL East champions, the Red Sox advanced to the American League Division Series (ALDS) where they defeated the Baltimore Orioles 3–0

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the franchise has continued with related projects since then

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A 20th anniversary miniseries is set to begin in Weekly Shōnen Magazine on July 29, 2026, though this appears to be a separate event rather than a traditional final season

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The International Space Station (ISS) began its journey in 1993, when the United States and Russia formally announced plans to build the station

### Sample qacc_a3c882e062c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: This brought the basilica to its full height of 172.5 meters and made it the tallest church in the world

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2
- **Claim**: **Celebrity Big Brother** is not currently broadcast on a U.S. network; the original UK show airs on ITV in the United Kingdom, while older seasons were streamed on Paramount+ in the U.S. more recent content has been available on 9NOW in Australia. [

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane (Carter Pewterschmidt)

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: July 4, 1776

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: This is confirmed by the LCO's Hubble Tuning Fork diagram, which places the Milky Way among the SBc class , with updated classifications reflecting that it exhibits characteristics of both barred spiral (SB) and spiral (S) galaxies

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: XXXTENTENTACION

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Internationally, the UK sets a similar minimum age of 18 to buy a shotgun, though individuals under 16 are prohibited from handling or discharging them

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Pew Research data further confirms that Asians have since surpassed Hispanics as the largest immigrant group, with Mexico remaining the top individual country despite slowing immigration from there .
[

### Sample situatedqa_geo_d982055a66d8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: These events represent the seminal electoral moments for two of the most prominent democratic nations, each occurring at distinct points in history and reflecting different political contexts

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: This framework established a weak central government serving as a "league of friendship" between the states, though it lacked a strong executive or judicial branch and proved insufficient for effective governance

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Still, a more nuanced view suggests that the shift unfolded gradually over the 19th and early 20th centuries, reinforced by immigration patterns and industrial infrastructure developments

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: The FOMC consists of twelve members — including seven from the Board of Governors and five presidents from Federal Reserve Banks — and is scheduled to meet several times each year to discuss whether to maintain or change current economic policy

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: However, policy-making also occurs at lower levels: individual states can adopt their own environmental rules that are stricter than federal standards local governments may enact regulations covering specific community issues

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The document lists Australia, England India as the most prominent winners, with Australia having won the tournament four times (1987, 1999, 2003, 2015) and England winning the most recent edition in 2019

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Mort is a Goodman's mouse lemur

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4, d2
- **Claim**: 1980

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Cleveland Browns head coach: Todd Monken (2026–present)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: India's Health Minister in 2013 was Shri Ghulam Nabi Azad, who served in that capacity until January 2014 when he was succeeded by Mr. Harsh Vardhan, who continues to hold the position

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: David Harbour plays Hopper in Orange is the New Black

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The group is known for their pop, pop rock soft rock hits including "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This figure was confirmed when Erton Köhler was elected the new president of the General Conference in 2025 , representing the most recent data available

### Sample situatedqa_temp_ae0882e48812

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The Gobi Desert is located in northern China and southern Mongolia, while the Taklimakan Desert is found in the Xinjiang region

### Sample situatedqa_temp_ae0882e48812

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Together, these two deserts account for the majority of China's total desert area, which is approximately 700,000 square kilometers

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d2
- **Supporting Docs Found**: None
- **Claim**: The name was first recorded in the Domesday Book of 1086, associated with the family's early roots in Lancashire is also found in Haiti as a result of colonial expansion

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Oleksandr Usyk (WBA Super, WBO, IBF, IBO)

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Additional contextual data from other sources further confirms the community's modest size and aging demographic .
[

### Sample trust_align_009

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Black Death is generally dated to 1348–1350 in England , though some sources cite 1346–1351

### Sample trust_align_009

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: It is worth noting that the most severe plague outbreak in London (the Great Plague) occurred in 1665–1666, which is discussed in , but this refers to a later epidemic, not the original Black Death

### Sample trust_align_067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Madcon — *It's All A Madcon* (2004)

### Sample trust_align_068

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Now, I will synthesize only the grounded snippets to answer the query

### Sample trust_align_084

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: In J.R.R. Tolkien's fantasy novel *The Fellowship of the Ring*, the ship is commanded by the half-phantom Gol洞中突然出现了一个巨大的石门，门上刻着复杂的符文。突然，一个声音在洞穴中回响：“只有真正的勇者才能打开这扇门。”
]

### Sample trust_align_149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: However, iodine has limitations — it does not protect organs and areas beyond the thyroid, such as the lungs, liver bone marrow, which is where radioactive cesium and other isotopes can cause severe damage

### Sample trust_align_154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: India has not hosted the Commonwealth Games yet; the first edition was held in 1930 in Hamilton, Canada

### Sample trust_align_169

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d4
- **Claim**: Smartphone photography of the sun is generally safe during totality (the brief central phase of a total solar eclipse) because the sun appears as a small round disk, not as harmful concentrated rays the duration is so short that permanent damage is highly unlikely

### Sample wikirevision_0035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in May 2026, which also records her incumbency from that date

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Royal Challengers Bengaluru are the current champions, having won their first IPL title. This result is corroborated by the Wikipedia IPL page, which lists them as champions with their 1st title. While the 2026 season is ongoing, it is worth noting that the most recent officially recorded IPL champion prior to this was Chennai Super Kings, who won the 2023 season. The current tournament features 10 teams competing across 74 matches, with the final match scheduled for 31 May 2026

### Sample wikirevision_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Jannik Sinner defeated Novak Djokovic in the 2025 US Open men's singles final, claiming the title

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President further corroborates his tenure, noting that the institution is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in April 2025, which lists her as the current incumbent with a sexenio (six-year) term

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: His election victory gave him a nonconsecutive term following his previous presidency from 2017 to 2021, making him the only U.S. president to serve more than two full terms


================================================================================

*Report generated by CATS v2.0*
