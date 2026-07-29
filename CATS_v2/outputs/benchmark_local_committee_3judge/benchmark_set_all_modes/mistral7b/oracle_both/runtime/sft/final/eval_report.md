# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 126 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.947 (over 736 samples)

**GR F1** *(used in CATS)*: 0.967

**Behavior Adherence**: 0.785 (over 610 applicable samples)

**Factual Grounding**: 0.831 (over 610 applicable samples)

**Single-Truth Recall**: 0.662 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.811

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.967
- **Precision**: 0.997
- **Recall**: 0.939
- **Accuracy**: 0.947
- TP=571, FP=2, FN=37, TN=126

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.773
- **Abstain Recall**: 0.984
- **Abstain F1**: 0.866
- **Specificity**: 0.939
- Abstain TP=126, FP=37, FN=2, TN=571


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.980
- **Behavior**: 0.923 (n=155)
- **Grounding**: 0.888 (n=155)
- **Recall**: 0.776 (n=154)
- **CATS**: 0.892

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.946
- **Behavior**: 0.898 (n=177)
- **Grounding**: 0.790 (n=177)
- **Recall**: 0.564 (n=156)
- **CATS**: 0.800

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.625 (n=96)
- **Grounding**: 0.839 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.814

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.655 (n=145)
- **Grounding**: 0.867 (n=145)
- **Recall**: 0.693 (n=140)
- **CATS**: 0.801

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.784
- **GR F1** *(used in CATS)*: 0.879
- **Behavior**: 0.595 (n=37)
- **Grounding**: 0.633 (n=37)
- **Recall**: 0.486 (n=37)
- **CATS**: 0.648


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2125

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
- **Cited Docs**: d1, d5, d4
- **Supporting Docs Found**: None
- **Claim**: Additional sources indicate that St. John's wort may interact with other medications and is not recommended for individuals with bipolar disorder or phototoxicity risks

### Sample conflictingqa_0a05aabca56a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: At the same time, it is worth noting that within Japan, anime is often referred to simply as 'animation' and is not universally distinguished from cartoons by all audiences or scholars

### Sample conflictingqa_0c3c7b487766

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, iodine deficiency remains a global health concern iodine is still recommended for populations with confirmed deficiency, meaning the appropriate dose is key

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d3, d5
- **Claim**: The real culprit is copper, which is commonly found in algaecides used to control algae growth in swimming pools

### Sample conflictingqa_35491baf4f4b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This very thin atmosphere is created through a combination of space weathering (vaporization of the lunar surface due to meteorite impacts) and ion-sputtering from the solar wind

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: For example, simple linear models may perform adequately with a few dozen training examples, while complex deep neural networks require large datasets to train effectively

### Sample conflictingqa_39fe5c441657

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These conflicting perspectives reflect methodological disagreement over what constitutes genuine metaphysical reality versus altered consciousness, making definitive scientific proof elusive

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d3, d4
- **Claim**: Related evidence

### Sample conflictingqa_3dba586dca0f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: A real Christmas tree's carbon footprint is negligible compared to the average artificial tree's 6.5 ft. stem, which can emit up to 40 kilos of greenhouse gases

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: The question of whether patents should apply to software is deeply contested, with strong arguments on both sides. On one hand, software patents can provide valuable protection for inventions involving algorithms, core functions innovative solutions to technical problems, though they are limited by legal standards such as Alice v

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, a study published in the same Nature journal found that the 2018 Hatihan earthquake occurred during a full moon but was not preceded by high tidal stress the USGS also notes that tides alone cannot explain all earthquake occurrences , calling into question the degree of reliance placed on lunar phases as a sole predictor

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d4
- **Claim**: Overall, the evidence points to saturated fats raising LDL cholesterol and associated heart disease markers, but the extent to which this translates to increased mortality risk remains an open scientific debate with conflicting findings

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: However, other research presents a more nuanced picture — a peer-reviewed analysis in Nature Sustainability found that high-yield corporate farms are better for the environment than organic ones a review in Scientific American argues that organic farming's environmental benefits are real but its yields are not reliably higher

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: The debate thus reflects ongoing scientific controversy over what constitutes 'efficiency' in agriculture: maximizing yields using synthetic inputs minimizing environmental harms through sustainable practices

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5, d4
- **Claim**: Religious truth claims are inherently theological opinions rather than historically verifiable facts, so the evidence presents conflicting perspectives: assert the Catholic Church is the one true church established by Jesus Christ, while d2 argues that Scripture alone determines truth and lists core doctrines without explicitly confirming the Catholic Church meets them

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Overall, the evidence points to multiculturalism being neither an automatic hindrance nor a guaranteed path to unity — rather, its effects depend heavily on the specific context, the types of unity being pursued the extent to which cultural differences are managed effectively

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: On the other hand, other high-credibility sources report that neutering can prevent overpopulation, reduce the risk of ovarian and prostate cancers help with behavioral issues like aggression and roaming

### Sample conflictingqa_9251c7a33ec5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The overall picture is complex and contested, with no single definitive conclusion on whether the negative health impacts outweigh the benefits

### Sample conflictingqa_9b11b8e571aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: In short, while sexual contact is the primary route of transmission, gonorrhea is not exclusively transmitted sexually safe sex practices remain essential even among partners who are not having penetrative sex

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4
- **Claim**: The outcome depends heavily on the specific plant species, the duration of darkness whether alternative light sources are available

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence presents competing views. Some sources argue that the broadcast did cause widespread panic, with historians noting that the program was so realistic that many listeners lost control, fled their homes suffered heart attacks

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Although volcanism is the leading hypothesis, some researchers also point to methane release from organic-rich sediments or permafrost as a co-contributor, particularly in the recovery phase the exact timing and sequence of events remain subjects of ongoing scientific investigation

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d4
- **Supporting Docs Found**: None
- **Claim**: The overall consensus is that cold water does not significantly enhance shine on its own, with the best results coming from conditioners and styling products that contain silicones and oils

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: If you do choose to use coffee grounds as a deterrent, it is generally recommended to test their effectiveness first, as results can vary depending on caffeine content and slug appetite

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5, d4
- **Claim**: Religion and cultural values shape how people address death, but overall death remains a taboo topic in modern society

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: Market makers can also influence price movements by controlling liquidity through wash trading and spoofing derivatives trading can amplify these effects

### Sample conflictingqa_f22b389be1d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Overall, the evidence is inconclusive and contested, with experts noting that running shoes have not significantly changed injury rates over the years, even as barefoot running enthusiasts claim it offers a natural advantage

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3, d4
- **Claim**: The retrieved evidence presents a genuine curse belief: folklore holds that Macbeth was cursed from the beginning, with the first performance (around 1606) marred by the sudden death of the actor playing Lady Macbeth and other mishaps. Some sources further assert that the play carries a persistent curse, causing accidents and injuries across productions, though others argue that such incidents are statistically no more common than for other Shakespearean works

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: However, the evidence does not confirm whether the Dutch were the absolute first discoverers of Australia, leaving room for prior unrecorded history the specific question of whether Australia was discovered by the Dutch remains partially resolved

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, other research suggests yerba mate may also have anti-cancer properties one study found it killed colon cancer cells in vitro , while another noted it is not clearly established whether these benefits apply to humans

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_0436c0b3a9d7

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
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: April 2, 2026

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_28e155139ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: For more information on Android 16 and how to update your device, visit the official Google support pages

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: This is confirmed by the official Microsoft download page, which lists '10.0.8' as the latest release of the .NET Core 10.0 Long-Term Support (LTS) channel

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: Multiple additional sources corroborate that .NET 10.0 is the current LTS release, with .NET 9.0 and .NET 8.0 being older maintenance releases

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d4
- **Supporting Docs Found**: None
- **Claim**: This site is now part of the White Sands Missile Range the event remains a well-documented part of US nuclear history

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: Portugal won the 2017 Eurovision Song Contest, marking the country's first victory since 1964. The contest took place in Kyiv, Ukraine Portugal's winner was Salvador Sobral, who performed the song "Amar Pelos Dois"

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d4
- **Supporting Docs Found**: None
- **Claim**: This result, superseding earlier reports that listed 'Anora' (2025) or 'CODA' (2022) as the most recent winners, as those awards have since been surpassed by the 2026 ceremony

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

### Sample freshqa_8eca5bd62ae0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: The 2020 Summer Olympics in Tokyo were postponed to summer 2021, allowing Beijing to become the first city to host both the Summer and Winter Games in the same year

### Sample freshqa_8f302f0bfe82

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The latest Nebula Award for Best Novel was won by *When We Were Real* by Daryl Gregory, which took the prize in 2025

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Frank Rosenblatt died in a boating accident on July 28, 1971

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2, d3
- **Claim**: Older sources reference earlier playoff appearances and records from the 2015–16 through 2019–20 seasons, but these are superseded by the 2023–24 data from Britannica and the NBA

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5, d4
- **Claim**: Jeff Bezos did not sell the entire company; he only sold shares

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_cf331ed7d09f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: It is important to note that each reflect different temporal snapshots of macOS versions, with d1 citing Sonoma (2023), d2 noting Monterey (2025), d4 identifying Monterey as the highest officially supported version for a specific Mac model d5 superseding all with Tahoe (2026)

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
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

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: In short, the answer depends on the specific species of slug under consideration

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Parents should instead monitor their child's temperature and seek medical attention if the fever persists or worsens

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

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d6, d8, d2, d7, d4, d5
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
- **Supporting Docs Found**: d2, d4
- **Claim**: The retrieved evidence indicates that after the North Africa campaign, Allied forces moved eastward across North Africa and into Europe, with Italy being the primary destination

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: All modern vertebrates, including amphibians, reptiles, birds mammals, are descended from these ancient fish lineages

### Sample qacc_2f6d2647a424

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d4
- **Supporting Docs Found**: None
- **Claim**: This assignment was confirmed by multiple sources, with Pete Rose Jr. and Johnny Bench appearing on the team's opening day roster for the 1975 season

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: Regardless of its exact roots, the gesture became firmly associated with Christianity — and specifically with the Christian fish symbol (ichthys) — as Christians adopted it as a shorthand way to recognize and support each other

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: [[

### Sample qacc_44b315f6f4bb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d1
- **Claim**: The vault is located within the Tower itself the jewels are transferred there from Buckingham Palace when they are not in use at royal ceremonies

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: October 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nana is an Australian Shepherd

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This victory gave the Red Sox their first division title since 2013 and set them on a path to the AL Championship Series, where they ultimately fell to the Astros

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Their 2017 success was fueled by a potent lineup that included Mookie Betts, Xander Bogaerts Hanley Ramirez, as well as ace pitcher Chris Sale

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

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Older documents citing higher figures (up to 250) reflect pre-2023 data and have since been superseded by the more recent 2026 information

### Sample qacc_a6b48b7accc4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d4
- **Claim**: 2003 was the birth year of T20 cricket, with the format rapidly gaining popularity before its official launch in 2004

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d4
- **Supporting Docs Found**: None
- **Claim**: New Mexico was admitted to the Union as the 47th state on January 6, 1912, when President William Taft signed the New Mexico statehood bill into law

### Sample qacc_b1bd9518429b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Both sides have invoked international law and historical precedents to support their competing claims, with Spain arguing that the isthmus is not ceded and that British occupation is illegal , while the UK contends that its presence there is consistent with international law and that Spain's insistence on control violates Gibraltar's right to self-determination

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: California's Mojave Desert along railroad tracks between Parker, Arizona and Vidal Junction; also filmed in Rice, California and Puerto Rico

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

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: [[

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: These are typically built as bypasses, bridge crossings direct intercity connections federal toll roads often use the suffix 'D' (e.g., Fed

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d1, d4
- **Claim**: Henry Burton

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells

### Sample qacc_f10c7ad4bb81

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d5, d4
- **Claim**: An initialism is an abbreviation formed from initial letters, pronounced as a series of letters (e.g., DNA, RT-PCR), while an acronym is pronounced as a word (e.g., NATO, UNESCO)

### Sample qacc_f2218f8c979e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5
- **Claim**: This primal cut is used primarily for support and is also the source of other popular beef cuts, such as ribeye steak

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This is established by the National Minimum Drinking Age Act of 1984, which sets a uniform age limit across all states, with no exceptions

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

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Looking further back, the 19th-century immigrant wave consisted mostly of Europeans from Italy, Austria-Hungary, Russia Poland , while more recent flows have shifted dramatically in favor of South and Central America, reflecting changing global migration patterns over time

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: There are approximately 649,481 villages in India, of which 593,615 are inhabited

### Sample situatedqa_geo_897e47478bbc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This figure is derived from the most authoritative source of administrative boundary data in the country and is further corroborated by additional sources reporting similar counts

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3, d4
- **Supporting Docs Found**: d5
- **Claim**: However, if the query specifically asks about sending the first U.S. military advisers to Vietnam, a distinct group from combat troops, the answer is President Kennedy, who sent approximately 16,000 military advisers to South Vietnam in 1961, expanding the U.S. presence there significantly

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features the California grizzly bear, which is an extinct population of the brown bear. The flag's design originated in 1846 when California was part of Mexico the bear became a symbol of the Bear Flag Republic, which was a short-lived attempt by U.S. settlers to break away from Mexico. The bear on the flag is also the official state animal of California the state is the only one in the union that carries the image of an extinct animal on its flag

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
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Elsewhere, the transition was shaped by different factors: in France, tea drinking persisted and only began to decline in the late 19th century , while in Italy the shift to coffee was driven by immigration and industrialization in the 19th century

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d3, d5
- **Supporting Docs Found**: None
- **Claim**: Its members include seven governors from the Board of Governors and four rotating presidents of the twelve Federal Reserve Banks, making it a collaborative body that represents both the public sector and regional economic conditions

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: d2
- **Claim**: At the federal level, environmental policy is also shaped by various statutes including the National Environmental Policy Act (NEPA) and the Clean Air Act extended to cover topics such as climate change, hazardous waste water quality

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: He was re-elected in 2024 and sworn in on January 3, 2024, replacing former Senator Jeff Chiesa

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Prior to his current service, Booker was the Mayor of Newark (2013–2021) he previously served as a Member of the House of Representatives (2019–2021)

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This victory gave LSU their eighth national title, their first since 2023

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Mort is a Goodman's mouse lemur, a small primate native to Madagascar, though a spin-off series reveals he is also part bear, spider starfish

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d1
- **Claim**: Their most recent championship came in 2019, giving them a total of 12 titles, which is the highest number of any program in history

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
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

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7cd18101326e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Indiana QB Fernando Mendoza (Jan

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1
- **Claim**: Earlier estimates placed the population at around 10,767 and 11,803 , but these have been superseded by the more recent 2026 projection

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For context, a recent analysis by UC Berkeley found that about $1.80 of the Bay Area's average gas price of $6 per gallon in May 2026 was attributable to higher taxes and environmental fees in California the state is expected to collect nearly $8 billion from gas taxes in the current fiscal year

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2, d1, d4, d5
- **Supporting Docs Found**: None
- **Claim**: Some sources cite slightly different figures, such as Islamic Relief's figure of 1,000 casualties or Quora's description of 3,000 participants , but these discrepancies reflect methodological differences in sourcing rather than factual contradictions

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: The empire reached its height during the reign of Huayna Capac (1492–1529), who expanded Inca territory along the Pacific coast and into the Andean highlands

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Some sources cite 1533 as the year the empire effectively ended with the Spanish conquest of the Neo-Inca State , while others place the dissolution a bit later, around 1572, when the Spanish conquered the last remnants in the Vilcabamba region

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d3
- **Claim**: In the United States, the Olympics have been hosted in eight different cities across four Summer and four Winter Games

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1, d5
- **Claim**: It was originally formed from the Old German name Gerhard, which itself was derived from the Proto-Germanic elements 'gēr' (spear) and 'hard' (brave/strong)

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3, d5, d4
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin and Perry Go Large

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

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: 2021

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Argentina (defeated Italy 3-2 in a penalty shootout at the Rose Bowl, July 1994)

### Sample wikirevision_0089

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Calcutta is officially called Kolkata. The city officially changed its name from Calcutta to Kolkata in 2001 this is confirmed by the Government of Kolkata official website as well as external sources. As a result, Calcutta is no longer the official name of the city, which is now officially known as Kolkata

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was confirmed by the Haryana government in 2022 the older name of Gurgaon is now used primarily as a historical or cultural reference

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore is officially called Bengaluru. The city officially changed its name from Bangalore to Bengaluru on 1 November 2014, as confirmed by the Bangalore City Municipal Corporation resolution of that date. As a result, Bangalore is no longer the official name of the city, which is now officially called Bengaluru

### Sample wikirevision_0120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: As a result, the 2026 French Open will feature a new men's singles champion, with the tournament serving as a Grand Slam comeback for players like Rafael Nadal and Novak Djokovic

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The current Wimbledon men's singles champion is Jannik Sinner, who defeated Alex de Minaur in the 2025 final to win his first major title

### Sample wikirevision_0135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d1
- **Claim**: His current status, with the 2026 Wimbledon Championships being held as an upcoming future event

### Sample wikirevision_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The rebranding also led to the spin-off of certain product offerings, such as the Oculus division, which operates independently under the Meta Reality brand

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: Older information referring to Gurgaon as a city in Haryana is therefore outdated, as the official name is now Gurugram

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1, d3
- **Supporting Docs Found**: d4
- **Claim**: The current FIFA World Cup champion is Argentina, who won the 2026 tournament, defeating France 4–2 on penalties after a 3–3 draw in the final

### Sample wikirevision_0162

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d3
- **Claim**: This victory marked Argentina's third World Cup title, making them the first team to win the tournament three times


================================================================================

*Report generated by CATS v2.0*
