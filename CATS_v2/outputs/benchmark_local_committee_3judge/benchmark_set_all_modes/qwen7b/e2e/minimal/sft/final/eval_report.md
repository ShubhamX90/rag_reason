# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 115 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.958 (over 736 samples)

**GR F1** *(used in CATS)*: 0.974

**Behavior Adherence**: 0.768 (over 621 applicable samples)

**Factual Grounding**: 0.837 (over 621 applicable samples)

**Single-Truth Recall**: 0.686 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.816

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.974
- **Precision**: 0.978
- **Recall**: 0.970
- **Accuracy**: 0.958
- TP=590, FP=13, FN=18, TN=115

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.865
- **Abstain Recall**: 0.898
- **Abstain F1**: 0.881
- **Specificity**: 0.970
- Abstain TP=115, FP=18, FN=13, TN=590


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (51 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.948
- **GR F1** *(used in CATS)*: 0.964
- **Behavior**: 0.925 (n=160)
- **Grounding**: 0.865 (n=160)
- **Recall**: 0.747 (n=154)
- **CATS**: 0.875

### Type 2: Complementary Info

- **Samples**: 221 (42 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.964
- **GR F1** *(used in CATS)*: 0.977
- **Behavior**: 0.877 (n=179)
- **Grounding**: 0.819 (n=179)
- **Recall**: 0.625 (n=156)
- **CATS**: 0.825

### Type 3: Conflicting Opinions

- **Samples**: 109 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.520 (n=98)
- **Grounding**: 0.861 (n=98)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.789

### Type 4: Outdated Info

- **Samples**: 158 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.653 (n=147)
- **Grounding**: 0.831 (n=147)
- **Recall**: 0.718 (n=140)
- **CATS**: 0.797

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.770 (n=37)
- **Recall**: 0.568 (n=37)
- **CATS**: 0.735


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2062

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
- **Claim**: Internationally, protection also varies significantly: in the European Union, designs are protected under the Creative Designs Directive for up to five years, while in the US the Vessel Hull Design Protection Act (a sui generis regime) applies only to vessel hulls and not general fashion designs

### Sample conflictingqa_07e290a954ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d3
- **Supporting Docs Found**: None
- **Claim**: Experts generally agree the herb appears safe when used alone but cautions are warranted given interactions with other medications and risks such as serotonin syndrome when combined with SSRIs

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Some legal scholars frame the ruling less optimistically, noting that Judge Clayton Horn explicitly found the poem lacked "even the slightest redeeming social importance" before concluding it was not obscene — a nuanced distinction preserved in historical records

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: Anime is generally considered a specific style of cartoon, originating in Japan with distinct artistic, narrative cultural characteristics

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d4, d1, d3
- **Claim**: Judaism is generally categorized as a religion, but it also functions as an ethnicity or nation (chabad.org)

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d1, d3
- **Claim**: Importantly, the risk is context-dependent—iodine supplementation is safe and beneficial in iodine-deficient populations, but must be carefully managed in those already replete, where excessive intake can increase the risk of autoimmune thyroiditis and hypothyroidism

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Applying the same standard to all fruits and vegetables is misleading

### Sample conflictingqa_220ec09fbb2c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: This unique digestive organ allows cows to efficiently break down and extract nutrients from the tough plant materials they consume

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: We cannot know anything beyond our minds

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: Its fossil also reveals that it had a range of motion in its hand's third digit, suggesting it could climb trees as well as fly

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d3
- **Claim**: Data is generally considered essential for Machine Learning, as both d3 [databricks] and d5 [datarobot] confirm that quality data is necessary for models to operate efficiently, with more data enabling faster learning and improved performance

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, the exact amount required is context-dependent: d4 [unidata] notes that the relationship between data volume and performance follows the law of diminishing returns d1 [postindustria] cites a 10-times rule where data should be ten times the number of model parameters, though this applies only to smaller models

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Educators and literary organizations increasingly recognize audiobooks as a valid form of reading the International Dyslexia Association explicitly includes them in its definition of reading

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4, d1
- **Supporting Docs Found**: d3
- **Claim**: Unlike artificial trees, which are made from plastic and metal and release up to 88 lbs. of greenhouse gas emissions per tree, real trees absorb CO₂ while growing and can be composted or chipped for mulch, providing no net increase in atmospheric carbon

### Sample conflictingqa_3f3c3399259a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d3
- **Supporting Docs Found**: None
- **Claim**: Experts advise that a healthy lifestyle, including regular exercise and a diet low in saturated fats, sugars processed foods, will lower heart disease risk far more than fish oil supplements that some evidence-based medications such as statins are among the most effective tools to prevent cardiovascular events

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, critics argue that trophy hunting perpetuates unethical consumerism, often involves illegal activities such as poisoning may undermine conservation efforts when reforms are insufficient — a view supported by research indicating that blanket bans could lead to increased poaching pressures in some regions

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Overall, the evidence suggests that trophy hunting's value to conservation depends heavily on context, regulation specific practices rather than a universal benefit or harm

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The question of whether patents should apply to software is complex and contested, with no single authoritative resolution

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d2
- **Claim**: Importantly, the effectiveness appears to depend on disease stage—the KDIGO guidelines recommend bicarbonate for patients with serum bicarbonate below 18 mEq/L —and some research suggests it may work best in early to mid-stage CKD rather than in advanced stages

### Sample conflictingqa_6988dd820a61

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d4
- **Claim**: A broader review further notes that while the moon's gravitational pull can trigger small tremors, the evidence for a consistent, measurable increase in earthquake frequency during full moons remains inconclusive, with the community divided between those who see a correlation and those who view the belief as persistence of a coincidental pattern

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, critics and some clinical reviews argue that the evidence is inconclusive, with the common cold caused by multiple viruses making generalization difficult that the benefits reported in some studies may be attributable to placebo effects or methodological limitations

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: However, the debate extends beyond pure production efficiency: conventional agriculture requires synthetic inputs that can contribute more to climate change per unit of food produced, while organic farming is more sustainable in terms of emissions during production processes, even if it cannot match the same output

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5, d3
- **Claim**: Religious organizations differ in their answers

### Sample conflictingqa_894f4a4b9552

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: While no one claims to fully understand what dark matter is, observational evidence indicates it makes up approximately 85–86% of the total matter in the universe, serving as the dominant form of mass-energy at large scales

### Sample conflictingqa_a3980a2921cf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: However, some sources add nuance: benefits depend on choosing the right oil for one's hair type — lightweight oils work best for fine hair, while richer oils are更适合的选项是d1，因为它直接且明确地回答了查询，声称发油适合所有发质。]

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Clinical research has further confirmed an inverse relationship between tea consumption and kidney stone risk, with a large 2013 analysis of over 194,000 participants finding that daily tea drinkers had an 8–14% lower risk of developing new stones compared to non-drinkers

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: However, sources differ: some diet websites and anecdotal claims treat celery as an archetype of negative-calorie foods despite the academic consensus that it is unlikely to be negative calorie in practice

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: While the original soundtrack was omitted from the Sonic Origins remaster, the game's 1993 prototype — which features Jackson's music — was adapted for the release, further linking the star to the title

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Research has shown that a caffeine concentration of more than 0.1% is sufficient to deter snails, with stronger solutions (up to 1%) even killing them in some cases, though the practical challenge lies in achieving and maintaining that specific threshold in soil

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Yes

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Research conducted by McGill University and the University of Minnesota found that organic yields are approximately 25 percent lower than conventional yields overall, though this figure can rise to as much as 44 percent in warmer temperate regions

### Sample conflictingqa_e93e708d49a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, a specific solar choice consultant noted that early lifecycle energy recapture assumptions (such as a two-year payback) are contested, as the calculation typically assumes all generated energy is immediately consumed, which may not always be the case

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: While these events are attributed to the very first performance, the curse is said to have continued plaguing subsequent productions, including notable accidents such as the Astor Place Riot in 1849, a 1937 fire at the Old Vic that narrowly missed Laurence Olivier a 1980 production at the Old Vic that led to the disbandment of the theatre company

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, the evidence does not confirm that the Dutch were the sole discoverers of Australia, as other European powers—such as the Portuguese, English, French, Spanish Danes—also explored the region

### Sample conflictingqa_f777a43ba278

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d4
- **Claim**: Yerba mate is traditionally prepared by steeping dried leaves of the Ilex paraguariensis plant in hot water it is widely consumed for its stimulatory effects, antioxidants claimed anti-inflammatory properties

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Animal studies have also shown that yerba mate contains chemicals capable of inducing cytotoxicity in cancer cells, suggesting potential anti-cancer properties, though human clinical confirmation remains outstanding

### Sample freshqa_0293a11bd364

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Hindi

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Aryna Sabalenka (2025 US Open women's singles champion); Jessica Pegula (2024 US Open women's singles finalist); Ons Jabeur (2022 US Open women's singles finalist); Sabine Zellner (2021 US Open women's singles finalist)

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_049cc3f14d5e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: St. Petersburg State University (also accepted: ITMO University, St. Pete State, SPbSU, St. Pete)

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d5, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani (2014 — the first and, to date, only woman)

### Sample freshqa_3cd0b514193b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: Employers in Tokyo must pay this hourly rate or higher to comply with Japanese labor laws

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: While d3 lists several children who died in the disaster, the query specifically asks about the youngest passenger, which has been clearly answered by the evidence above

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The most recent winner overall is The Dragonfly Gambit, also in the Best Novel category, though it appears on the 2025 nominations page rather than the confirmed winners

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: He was 43 years old at the time, tragically losing his life just two years after Marvin Minsky and Seymour Papert published their influential book "Perceptrons," which highlighted the limitations of his groundbreaking work

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Jiangsu

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Crocodiles

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: Earlier sources referenced macOS Sonoma (released 2023) as the latest version , but this information has been superseded by the newer Tahoe release

### Sample freshqa_d4d59d75b206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: 2015–2016–2018

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Experts generally agree that yoga appears safe for most asthma patients, though individual circumstances vary some recommend it as a supplementary practice alongside established treatments such as bronchodilators and inhalers

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7, d1
- **Supporting Docs Found**: None
- **Claim**: This broader historical context is further corroborated by sources noting that Korea's path to independence was shaped by the 1945 Potsdam Declaration, which required Japan's unconditional surrender the subsequent establishment of separate governments in the North and South

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, Thal's casting in the 1949 film is the most direct and specific answer to the query

### Sample hotpotqa_0071

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d10
- **Claim**: American singer/songwriter, record producer, businesswoman television personality

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7
- **Supporting Docs Found**: None
- **Claim**: This 1610 map is one of several cartographic sources depicting Monmouth in the 17th century, including the 1610 Speed map, the 1646 Blaeu map the 1672 Hondius map

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d8, d6, d5, d4, d7
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The retrieved evidence indicates that following the liberation of North Africa, the Allies advanced across the Mediterranean into Italy, where they launched the invasion of Sicily in July 1943, followed by the broader campaign in Italy during 1943–1945

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d4, d1, d3
- **Claim**: Sakshi Malik (Haryana); Madhuri Dixit (India); Parineeti Chopra (Haryana); Avani Lekhara (Rajasthan); Bhawna Dehariya and her daughter Siddhi Mishra (Madhya Pradesh)

### Sample qacc_1b95727cc286

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3
- **Claim**: Wood Harris (Ace), Mekhi Phifer (Mitch), Cam'ron (Rico)

### Sample qacc_367b09e4ed80

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Canada did not acquire independence abruptly but through a gradual process. Ongoing constitutional evolution between 1919 and 1931 — including the Balfour Declaration (1926) and the Statute of Westminster (1931) — progressively reduced the residual authority of the British Parliament over Canadian legislation the formal legal bonds were not fully severed until 1982 when the Canada Act was enacted. Earlier, in 1867, the Dominion of Canada was founded as a self-governing entity from New Brunswick, Nova Scotia the Province of Canada, though full sovereign equality with the United Kingdom was not fully realized until the mid-20th century

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Steve McEwan

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 180

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: These origins are initiated concurrently in clusters and are epigenetically determined rather than based on a specific consensus sequence

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: It is important to note that the precise count can depend heavily on the organism and experimental context, as the number falls within a broad range due to the stochastic and developmentally regulated nature of origin selection

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: October 1968

### Sample qacc_950881e7c998

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d1
- **Claim**: Boston Red Sox

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, the anime series technically concluded after that, with the manga continuing into 2019, leading to some confusion about what constitutes the 'final' season

### Sample qacc_9b16fd6882f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: Russ Ballard

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a1c73439eca9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Season 9 premiered on 13 February 2024

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5, d2
- **Supporting Docs Found**: None
- **Claim**: This is consistent across the general population, with both adult males and females averaging around 55–65% body water by weight , though the exact percentage can vary by individual factors such as age, sex adiposity

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: These signs typically appear in a yellow diamond shape with a black 'S' or arrow symbol, indicating the recommended safe speed for the specific curve ahead

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Celebrity Big Brother is broadcast on CBS in the USA

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d2, d5
- **Claim**: President Hoover was hosting a young people's party for the children of his staff at the time, though the fire itself originated in the Executive Offices rather than the party itself

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: No one was injured in the blaze the following Christmas White House staff gathered again to celebrate, receiving toy fire trucks as gifts

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5
- **Claim**: Japan: 1996; in the US, the first base set was released on January 9, 1999

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2024/25

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d5
- **Claim**: Between three and seven

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5
- **Claim**: The welfare state was introduced at different times across regions, with no single universal date

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: At least three major fronts — the Eastern Front (between Nazi Germany and the Soviet Union), the Western Front (between Germany and the Allies in Europe) the Mediterranean/Italian Front (including North Africa and the Italian campaign) — formed the core battlespaces of World War II, though the conflict spanned even more theatres

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: Wales followed suit on 2 April 2007 Northern Ireland enacted a total ban in 2007 as well

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: December 31, 1970

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: The California state flag features a California grizzly bear (Ursus arctos californicus), which is a subspecies of the brown bear

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, the Eagles claimed their third Super Bowl victory in 2023 (Super Bowl LVII), though they would go on to lose Super Bowl LVIII in 2024

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: New South Wales last won the State of Origin series in 2021

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Cory Booker (current, elected 2012, serves 2013–present)

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_301378915064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Merritt Wever (Nurse Jackie)

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Goodman's mouse lemur

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d3
- **Claim**: This total is confirmed by multiple sources, with the official NBA stats page corroborating LeBron James as the all-time career scoring leader

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: Android 16 (Baklava).

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
- **Supporting Docs Found**: d2, d4, d5, d3
- **Claim**: 1980

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken (2026–present)

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: An example of a kenning from Beowulf describing Grendel is “twilight-spoiler” (lines 21, 286) for Beowulf himself, the text uses “sure-footed fighter” (lines 107, 1543)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967 (released in the UK on Epic Records in September 1967; charted in the US in 1967)

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: California's state gasoline excise tax is $0.60/gallon, the highest in the United States combined with federal taxes of $0.18/gallon as of March 2025, totals $0.78/gallon in state and federal taxes

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d1
- **Claim**: Local government contributions add an additional $0.075/gallon in sales tax, bringing the total tax burden to roughly $0.85–$0.90/gallon depending on the period

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Formed in Los Angeles in 1989, the group achieved fame with hits such as "Hold On," "Release Me," and "You're in Love," and their self-titled debut album sold over ten million copies worldwide

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This figure was confirmed when Erton Köhler was elected the new president of the General Conference in 2025 , providing the most up-to-date count available from official sources

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d4
- **Claim**: Shay Mitchell was 23 when she portrayed 16-year-old Emily Fields in the show's pilot the character is described as being in her mid-20s in real life

### Sample situatedqa_temp_ae0882e48812

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The Gobi Desert is located in northern China and southern Mongolia, while the Taklimakan Desert is found in the Xinjiang region

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: However, she was not formally declared operational until 2020 — the first ship of her class to enter service

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The WTO has 166 members

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1, d5
- **Supporting Docs Found**: None
- **Claim**: This figure reflects the most recent count available, superseding older reports that cited 164 members

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d5, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: July 11, 1987 (for the original series premiere) — Seed answer "July 11, 1987" is correct but incomplete, as the snippet references Season 4 and the original first episode aired on July 11, 1987

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This record was confirmed when the film surpassed its predecessor, **Rewind**, just 10 days after its release

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_f971e49123a1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: 13 episodes — Season 5 of The Originals premiered on April 18, 2018 and concluded on August 1, 2018, making it the final season of the series

### Sample trust_align_039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: By contrast, human euthanasia is subject to strict legal and ethical criteria because it involves a sentient being capable of autonomous consent because alternatives such as palliative care and treatment exist, which are not available to animals

### Sample trust_align_064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: Human eyes do not have a reflective layer called the tapetum lucidum, which is present in many animals including cats, dogs moths

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Additionally, high concentrations can displace oxygen in the lungs and central nervous system, leading to suffocation as breathing ceases

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0057

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc. as of 2015, when Google was reorganized as a wholly owned subsidiary of Alphabet. Larry Page and Sergey Brin, the company's founders, together own approximately 14% of Google's publicly listed shares and control about 56% of its stockholder voting power through super-voting stock. While the broader entity Alphabet Inc. is the owner, Google's day-to-day operations are led by CEO Sundar Pichai, who replaced Larry Page in 2015

### Sample wikirevision_0067

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: The 70th Ballon d'Or was presented by France Football to honour the best performing football player of the 2024–25 season, with Dembélé securing his first win

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed when the Haryana government passed a resolution changing the city's name officially to Gurugram. As of the newer Wikipedia revision, the infobox explicitly states the official name is Gurugram this is further corroborated by the city's own official name change policy, which notes that changing the city name to "Gurugram" is permitted only with prior consent of the Haryana State Government

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The President of Germany is Frank-Walter Steinmeier, who has served as Herr Bundespräsident and holds Bellevue Palace as his official residence. This is confirmed by the current Wikipedia revision, which also notes that his term is 5 years and renewable once consecutively. While the article provides historical context tracing the office back through various eras, including the Weimar Republic and post-reunification, the most recent information consistently places Steinmeier in the role

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, the official Wikipedia page for the 2026 Wimbledon Championships notes that this information may be incomplete, suggesting the tournament is still ongoing or results are not yet final

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 edition is the 139th running of the prestigious event, scheduled from 29 June to 12 July 2026, making it the most recent Wimbledon Championships held

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: 2022, Bongbong Marcos is the President of the Philippines, serving as both head of state and head of government. He assumed office following the death of his father, Ferdinand Marcos Sr. was confirmed as the country's 16th president. This is consistent across multiple up-to-date sources, including the Wikipedia article on the President of the Philippines, which also notes that he serves as the commander-in-chief of the armed forces

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed when the Haryana government passed a resolution changing the city's name officially to Gurugram. As a result, while 'Gurgaon' is still commonly used, 'Gurugram' is the correct official name

### Sample wikirevision_0171

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Jannik Sinner was the defending men's singles champion at the 2026 Australian Open, but he lost in the semifinals


================================================================================

*Report generated by CATS v2.0*
