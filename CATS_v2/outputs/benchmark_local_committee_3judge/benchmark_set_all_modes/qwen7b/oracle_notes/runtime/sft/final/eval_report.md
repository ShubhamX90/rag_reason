# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 98 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.957 (over 736 samples)

**GR F1** *(used in CATS)*: 0.974

**Behavior Adherence**: 0.779 (over 638 applicable samples)

**Factual Grounding**: 0.905 (over 638 applicable samples)

**Single-Truth Recall**: 0.755 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.853

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.974
- **Precision**: 0.953
- **Recall**: 0.997
- **Accuracy**: 0.957
- TP=606, FP=30, FN=2, TN=98

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.980
- **Abstain Recall**: 0.766
- **Abstain F1**: 0.860
- **Specificity**: 0.997
- Abstain TP=98, FP=2, FN=30, TN=606


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.938
- **GR F1** *(used in CATS)*: 0.960
- **Behavior**: 0.868 (n=167)
- **Grounding**: 0.912 (n=167)
- **Recall**: 0.825 (n=154)
- **CATS**: 0.891

### Type 2: Complementary Info

- **Samples**: 221 (35 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.950
- **GR F1** *(used in CATS)*: 0.970
- **Behavior**: 0.930 (n=186)
- **Grounding**: 0.891 (n=186)
- **Recall**: 0.718 (n=156)
- **CATS**: 0.877

### Type 3: Conflicting Opinions

- **Samples**: 109 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.945
- **GR F1** *(used in CATS)*: 0.970
- **Behavior**: 0.598 (n=102)
- **Grounding**: 0.899 (n=102)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.822

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.630 (n=146)
- **Grounding**: 0.928 (n=146)
- **Recall**: 0.739 (n=140)
- **CATS**: 0.823

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.973
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.874 (n=37)
- **Recall**: 0.676 (n=37)
- **CATS**: 0.810


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2240

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

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: No, hair cannot really turn green solely from chlorine in swimming pools

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: However, critics counter that the practice is 'morally inappropriate,' that it perpetuates colonial and anthropocentric narratives that well-regulated alternatives such as ecotourism can achieve similar conservation goals without the ethical costs , with the overall debate reflecting genuine disagreement on whether the benefits of trophy hunting outweigh its harms

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Research cited by the Brookings Institution further breaks down the gap into distinct components such as occupation, industry hours worked, showing that while these factors explain much of the disparity, genuine wage discrimination also persists

### Sample conflictingqa_5233eab573e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d1, d5, d4
- **Claim**: While d4 cites slightly different numbers (captive: 5,000+, wild: ~2,500), the underlying conclusion remains consistent: the number of tigers kept as pets significantly outpaces those in the wild

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d2
- **Claim**: Experts agree that bicarbonate supplementation may help reduce urine fibrogenic biomarkers and preserve eGFR in earlier stages, but its role in more advanced CKD remains uncertain

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Experts agree that most healthy individuals do not need to take vitamin C supplements routinely , but those at increased risk of deficiency—such as smokers, people with restricted diets those with certain chronic diseases—may benefit from appropriate supplementation

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The effects depend heavily on the type of unsaturated fat substituted for saturated fat: replacing saturated fats with primarily n-6 PUFAs has no consistent effect on heart disease risk, while n-3 PUFAs (found in fish oil) may be particularly protective

### Sample conflictingqa_9251c7a33ec5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Cats also face risks including urinary tract disorders, obesity behavior problems , while evidence for rabbits is inconclusive

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Researchers are actively investigating the full scope of post-sterilization hormonal effects on health a nuanced, individualized approach is recommended given the complexity of the evidence

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: A YouTube myth-busting video further notes that cold water rinses may also constrict scalp blood capillaries, potentially harming hair growth a salon professional suggests that cold water is a pointless step since it tightens the cuticle but is then reopened by a hair dryer

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d3, d2
- **Claim**: However, some sources acknowledge that foods with very low net calorie content, such as celery, are often cited as examples of negative-calorie foods the concept remains popular in dieting contexts despite lacking empirical validation

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: Research attributes this shrinkage partly to metabolic efficiency gains from symbolic information processing , the decline in average body size the reduced need for large brains as external information storage and processing have become prevalent

### Sample conflictingqa_bac0f4d62f96

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: The prevailing scientific consensus

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5
- **Claim**: That said, both d3 and d5 note that manual toothbrushes can work well with excellent technique, meaning proper brushing habits can mitigate some of the advantages electric toothbrushes offer

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: Some sources further challenge the narrative by noting that the public was already on edge due to ongoing geopolitical tensions, such as the Munich crisis that newspapers sensationalized the rare cases of actual fear to discredit radio

### Sample conflictingqa_c34991d9897e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d2
- **Claim**: However, copyright primarily protects the artistic design itself, not the brand identity as a whole

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Broader claims suggest the entire market is rigged, with market makers and high-volume traders exploiting weaknesses in technology and regulation for personal gain

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2
- **Claim**: Furthermore, while the initial material extraction and production phase requires energy, the panels themselves generate far more electricity than was needed to create them, making them a net positive contributor to the energy balance

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: However, modern medicine does not endorse this practice studies are predominantly focused on investigating the effects of bee venom on arthritis rather than confirming its efficacy

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: The conventional scientific view

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: PAHs (polycyclic aromatic hydrocarbons), known carcinogens also found in grilled meat and tobacco smoke, are present in yerba mate and can contribute to cancer development , though the extent of their contribution varies across studies

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d5
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d5
- **Supporting Docs Found**: None
- **Claim**: Former U.S. President Bill Clinton, on the other hand, signed a total of 372 Executive Orders during his two terms in office

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris won the 2020 Formula 1 World Drivers' Championship

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

### Sample freshqa_4590bdd9e269

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Bismuth

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The federal funds rate was 4.25%–4.50% as of December 2022, reflecting a series of 25-basis-point increases throughout the year

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: JPMorgan Global Research further corroborates that the Fed kept its policy rate unchanged at 3.50%–3.75% at its March 2026 meeting, consistent with a 2022 rate-hike cycle

### Sample freshqa_4e635a2542a8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d3
- **Claim**: This group recorded the classic album Relaxing the lineup was documented in the Miles Davis archives as his 'first important band' — though not explicitly labeled as the 'first quintet' in the snippet

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This victory marked Anderson's first Oscar win and earned the film six Oscars total, including best director and best adapted screenplay

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: When We Were Real

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2, d5
- **Supporting Docs Found**: d1
- **Claim**: While older sources reference their 2019 NBA Championship or earlier season records , these are superseded by the newer 2023–24 data, confirming the Raptors do not currently hold a winning record in the latest NBA season

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: d4, d1
- **Claim**: Queen Elizabeth II died on 8 September 2022, at Balmoral Castle in Aberdeenshire, Scotland

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: Jeff Bezos did not sell Amazon — he sold Amazon shares

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c7ac9d61059a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3, d2
- **Claim**: These discrepancies reflect regional differences, pricing changes over time variations between standard and launch-series models

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2
- **Claim**: His family includes six children with his first wife Justine Wilson (including Nevada), three with musician Claire Boucher (Grimes) four with his Neuralink executive Shivon Zilis

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: There is currently no permanent cure for cancer; however, significant milestones have been achieved in managing the disease

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: Pfizer's vaccine is approved for people ages 5 and older Novavax's is authorized for those 12 and older

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that the FDA's narrowest eligibility rules apply to unboosted children, with fully vaccinated children having a lower risk of severe disease

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d4
- **Claim**: These dates are corroborated by additional sources reporting the start date as February 17, 2026 the end date as March 19, 2026

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d8
- **Claim**: The 2016 Marrakesh ePrix was won by Lucas di Grassi

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d6, d2, d8, d5, d7, d4
- **Claim**: 506

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Norma Koch

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: However, none of the documents address the specific line or context referenced in the query about 'when is my mother said i never should set,' making it impossible to provide a complete answer under the available evidence

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d1, d2, d5
- **Claim**: While Operation Torch (the landing in Algeria and Morocco) was a significant first step, the broader answer also includes the subsequent campaigns in Sicily and Italy, as well as the eventual invasion of France (D-Day)

### Sample qacc_2243f17ccc38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: They would go on to defend the trophy the following season in 1971–72, beating Barcelona in the European Inter-Cities Fairs Cup play-off to claim the Fairs Cup title permanently

### Sample qacc_44b315f6f4bb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that the Imperial State Crown, St Edward's Crown the Queen's personal Diamond Diadem are currently on display in Westminster Hall during Her Majesty's lying-in-state

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë (as the High Lord of the Valar)

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: Canada became officially independent from Great Britain in 1867 when the Dominion of Canada was formed, though full legislative independence was not achieved until 1931 with the Statute of Westminster Canada's final colonial ties were not formally dissolved until 1982 with the Canada Act

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Jessica Biel (season 1); Jessica Hecht (season 3); Alice Kremelberg (season 4)

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d5
- **Claim**: October 1968

### Sample qacc_8daf80e943fa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: 5.88

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The remaining one-third is found in the extracellular space, which consists of the interstitial fluid and blood plasma

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: **Celebrity Big Brother** is not currently broadcast on a U.S. television channel

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The fire was extinguished before causing serious structural damage

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_bf0bff050f03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: New Zealand (have India never lost to Afghanistan or Ireland either)

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane (Carter Pewterschmidt) / Peter Serafinowicz (Peter Griffin) — depending on which character you mean

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: July 4, 1776

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: This is confirmed by the LCO's Hubble Tuning Fork diagram, which places the Milky Way among the SBc class

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: Autopistas

### Sample qacc_f69c37496013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: September 25, 1987

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: For those interested in moving there, the official The Villages website provides comprehensive information on community amenities, home ownership relocation resources

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Other major figures include Germany's military dead estimated at 5.3 million , China's civilian and military deaths estimated at 20–25 million Japan's casualties estimated at 2.12–3.1 million

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The retrieved evidence indicates that World War II was fought across multiple fronts, with the Eastern Front being the largest, but no single authoritative source states the total number of fronts

### Sample situatedqa_geo_d982055a66d8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d2
- **Claim**: These two events represent the first national elections in their respective countries, marking significant milestones in democratic governance

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A final phase came with the advent of the Boston Coffee House in 1773, which further cemented coffee's status as a preferred national beverage

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: The FOMC consists of twelve members — including seven from the Board of Governors and five presidents from Federal Reserve Banks — and is scheduled to meet several times each year to discuss and adjust monetary policy as needed

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, states also play a significant role, as they can adopt their own stricter environmental standards in areas such as air quality, water management renewable energy — a power known as "preemptive authority" — and some states have surpassed federal standards in practice

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: At the local level, cities and counties can enact their own ordinances and zoning rules to further protect the environment, though these typically build on rather than contradict state and federal frameworks

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The document lists Australia, England India as the most prominent winners, with Australia having won the tournament four times (1999, 2003, 2007, 2015) and England winning the most recent edition in 2019

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3, d1
- **Supporting Docs Found**: d2, d5
- **Claim**: While d2 references an all-time scoring list and d5 shows partial rank data, these do not explicitly name the top scorer — making the high-credibility sources the definitive answer

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Novak Djokovic (men) / Margaret Court (women)

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2, d5
- **Claim**: 1980

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken is the head coach of the Cleveland Browns

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: September 1967

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The group is known for their pop, pop rock soft rock hits including "Hold On," "Release Me," and "You're in Love," and is renowned for the rich harmonies produced by these three sisters

### Sample situatedqa_temp_ae0882e48812

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The Gobi Desert is located in northern China and southern Mongolia, while the Taklimakan Desert is found in the Xinjiang region

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances released into the blood when the heart is damaged or stressed are used to help diagnose acute coronary syndrome (ACS), myocardial infarction (heart attack) heart failure

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Other traditional cardiac biomarkers include creatine kinase (CK), creatine kinase–MB isoenzyme (CK-MB), myoglobin lactate dehydrogenase (LDH); however, these are less specific for heart disease and can also be elevated in other conditions

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: It is worth noting that the ship underwent extensive testing and sea trials before its official commissioning ceremony in Portsmouth it was formally declared operational in 2020

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: The name first appeared in the Domesday Book of 1086 as Gerardus and Girardus was originally borne by the son of Gerard

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: While Wikipedia's entry notes the name is of Proto-Germanic origin meaning 'strong spear,' it is important to recognize that the surname as adopted by Anglo-Saxon tribes of Britain dates back to the 12th and 13th centuries

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Oleksandr Usyk (WBA Super, WBO, IBF, IBO)

### Sample trust_align_008

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Her filmography further includes notable roles such as Jessica Van Helsing in *The Satanic Rites of Dracula* (1973) and various other productions

### Sample trust_align_028

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: This date is corroborated by additional context showing that the changeover period extended through 1988, with the paper note fully withdrawn by the end of that year

### Sample trust_align_081

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: Boiled water makes clear ice because the boiling process releases dissolved gases from the water, a phenomenon known as degassing. When water is heated to its boiling point, gases like oxygen and nitrogen come out of solution and bubble to the surface, leaving the water free of these impurities. In contrast, tap water contains dissolved gases that remain even after freezing, causing the resulting ice cubes to appear cloudy or hazy

### Sample trust_align_085

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Earwax fluctuates because it is normally pushed toward the ear opening where older wax falls out or is washed away the body can overproduce earwax under stress or due to fear, sometimes preventing natural expulsion

### Sample trust_align_168

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, Mike Tramp's official site confirms he released a solo album called *Nomad* in 2015

### Sample wikirevision_0003

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The official Twitter account further clarified that the rebrand reflects a shift from "Tweet" to "X," with the platform's original name, Twitter, no longer used

### Sample wikirevision_0035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Her tenure has continued through May 2026

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The current Prime Minister of Japan is Sanae Takaichi, who assumed office on 21 October 2025. She is also the first female Prime Minister of Japan, making her historic appointment particularly noteworthy

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President further corroborates his tenure, noting that the institution is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official Wikipedia revision that superseded the older version in April 2025, which explicitly names her as incumbent with a detailed biography


================================================================================

*Report generated by CATS v2.0*
