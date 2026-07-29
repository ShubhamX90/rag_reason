# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 110 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.947 (over 736 samples)

**GR F1** *(used in CATS)*: 0.968

**Behavior Adherence**: 0.778 (over 626 applicable samples)

**Factual Grounding**: 0.874 (over 626 applicable samples)

**Single-Truth Recall**: 0.703 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.831

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.968
- **Precision**: 0.970
- **Recall**: 0.965
- **Accuracy**: 0.947
- TP=587, FP=18, FN=21, TN=110

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.840
- **Abstain Recall**: 0.859
- **Abstain F1**: 0.849
- **Specificity**: 0.965
- Abstain TP=110, FP=21, FN=18, TN=587


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (54 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.976
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.936 (n=157)
- **Grounding**: 0.928 (n=157)
- **Recall**: 0.815 (n=154)
- **CATS**: 0.916

### Type 2: Complementary Info

- **Samples**: 221 (36 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.928
- **GR F1** *(used in CATS)*: 0.955
- **Behavior**: 0.849 (n=185)
- **Grounding**: 0.824 (n=185)
- **Recall**: 0.603 (n=156)
- **CATS**: 0.807

### Type 3: Conflicting Opinions

- **Samples**: 109 (8 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.936
- **GR F1** *(used in CATS)*: 0.964
- **Behavior**: 0.554 (n=101)
- **Grounding**: 0.873 (n=101)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.797

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.692 (n=146)
- **Grounding**: 0.918 (n=146)
- **Recall**: 0.714 (n=140)
- **CATS**: 0.827

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.811
- **GR F1** *(used in CATS)*: 0.896
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.730 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.737


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2015

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
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: Weight lifting causes a temporary increase in blood pressure during the actual lifting action, but the long-term effects are generally positive and include blood pressure reduction

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Peeling an apple reduces some of its nutritional value by removing dietary fiber and certain vitamins, but not all nutrients are lost

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The conflict type is conflicting opinions or research outcomes, as documents present opposing interpretations of the same question without a definitive, authoritative resolution

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Yes, palm oil is bad for the environment; it causes deforestation, biodiversity loss, habitat destruction, pollution greenhouse gas emissions

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d1
- **Claim**: A 2004 survey cited in the review found that 58.5% of parents believed milk increases mucus , suggesting the perceived link is widely held, though the scientific consensus from the reviewed studies is mixed

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Fluoride in drinking water is considered largely safe at concentrations of 0.7 mg/L or lower, but high levels are linked to risks including fluorosis, skeletal damage potential neurotoxicity

### Sample conflictingqa_29f69e16a0c3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4
- **Claim**: These interactions form a mutually beneficial relationship where flowers attract pollinators to facilitate reproduction bees collect nectar and pollen in return

### Sample conflictingqa_311fca0928d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The two protocols are also not mutually exclusive — most modern systems run both IPv4 and IPv6 simultaneously, so security best practices must be applied to both stacks equally

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Data is not always strictly required for Machine Learning (ML) in the sense that ML models can be designed to function with small datasets or even with no data at all, though this is increasingly rare in practice

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The relationship between data and ML performance follows a law of diminishing returns, where initial increases in data volume lead to significant gains but eventually plateau insufficient data can cause underfitting while excessive data can lead to overfitting

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The debate is ongoing

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, other sources caution that much of the Moon's geological movement is thought to have occurred 2.5–3 billion years ago, with some activity as recently as 200 million years ago that the Moon's interior core crystallized about 4 billion years ago causing volcanic activity to cease

### Sample conflictingqa_3dba586dca0f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Ultimately, the most sustainable option is a potted tree that can be reused yearly, as it continues to absorb CO2 and avoids the cultivation of new trees

### Sample conflictingqa_411445406724

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: By the end of the Mesozoic, flowering plants had replaced cycads as the ecologically dominant land species, leaving only about 200 surviving species today

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: The evidence is mixed. Some sources argue that trophy hunting can benefit conservation by generating revenue, controlling wildlife populations funding anti-poaching efforts, while others argue it is morally inappropriate and that bans are not harmful but rather beneficial to conservation

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A peer-reviewed, authoritative source directly addressing the gap's reality and causes is not present in the retrieved set, making the evidence insufficient to fully resolve the question

### Sample conflictingqa_5233eab573e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d1, d2
- **Claim**: These figures are consistent across sources and collectively demonstrate that the pet population significantly surpasses the wild one

### Sample conflictingqa_56fd6bf22253

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5, d3
- **Claim**: Factors such as the age at surgery, the surgical technique used postoperative infection or inflammation can influence the likelihood of regrowth, though in most cases it does not recur to a clinically significant degree

### Sample conflictingqa_676188bf8139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The Chinese Lantern Festival is related to honoring ancestors but not exclusively about celebrating deceased ancestors; it is also about marking the first full moon of the new lunar year and promoting reconciliation

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: The Catholic Church claims to be the "One True Church" founded by Jesus Christ, but this claim is not explicitly supported by Scripture and is contested by Protestant denominations

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological: studies using different metrics (civic cohesion, socioeconomic integration, ethnocentrism) reach opposing conclusions no single, universally applicable finding resolves the debate

### Sample conflictingqa_a7ff288bc615

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d1, d2
- **Claim**: AI has passed the Turing test in some form, but differ on how meaningful this achievement is

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Overall, the evidence suggests death remains a sensitive and rarely discussed topic in modern society, with no single definitive consensus on whether it is still taboo

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: Evidence also suggests that large price movements can be triggered by relatively small initial manipulations that some market makers use their liquidity control for their own benefit

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: The retrieved evidence is mixed. Some sources (RSC, JCOPERAHOUSE) confirm the curse legend and cite specific incidents from the first performance, while others (Scribd/Statistical analysis, Wikipedia) challenge the curse as a superstition with no factual basis in misfortune statistics

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence supports both Dutch and British claims of first discovery, with the Dutch being the first Europeans to land (at Cape York in 1606) and the British being the first to establish a colony

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The NIH-reviewed literature found that several studies linked hot mate infusion temperature to increased risks of oral, esophageal laryngeal cancer, with the highest risk estimate (odds ratio = 34.6) found in women drinking 1 L or more daily

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, other research has also shown that yerba mate contains anti-cancer compounds that are cytotoxic to cancer cells in lab studies some studies suggest it may lower the risk of certain cancers—though this evidence is not as strong as the risk links noted above all findings are hedged by methodological limitations

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d4, d3
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because these methodological and interpretive differences in defining 'life' and analyzing evolutionary history produce opposing conclusions about whether viruses belong in the tree of life

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

### Sample freshqa_047057d22309

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Prince Harry's HRH title was removed from the official Royal Family website there were calls for him to be stripped of his dukedom, but the snippet does not state that King Charles III formally stripped him of the title of Duke of Sussex

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: The latest major.NET version depends on which branch of.NET is meant:.NET Framework 4.8.1 is the latest for the Framework branch, while.NET 6.0 and.NET 7.0 represent the latest for.NET 5+

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: This, noting that the war is of equal length to the Soviet Union's 1941–1945 war against Nazi Germany and has resulted in the loss of over 10 million people

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This film won six Oscars, including Best Director and Best Adapted Screenplay, marking a long-in-coming coronation for Anderson

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The 98th Academy Awards also saw other notable wins, including Michael B. Jordan taking home Best Actor and Autumn Durald Arkapaw becoming the first female cinematographer to win the award

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

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Jeff Bezos did not sell Amazon; he sold Amazon shares

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: The Komodo dragon as the largest lizard species, but note it is smaller than the heaviest crocodile, creating a direct mass-based conflict

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample misinformation_0023

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0d85f1089c4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Madhuri Dixit

### Sample qacc_37fdedfe4478

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Joanna Cotten

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d4, d3
- **Supporting Docs Found**: None
- **Claim**: One theory holds that this practice predates Christianity and was later adopted by early Christians as a secret sign of recognition and a way to invoke divine protection when they were persecuted , while another theory suggests the gesture originated from the early Christian practice of forming the ichthys symbol (the fish symbol) by touching thumbs and crossing index fingers, which was used to recognize one another and seek protection

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: These structures are distinct from Peyer's patches in both function and anatomy, though both are found in the small intestine and play important roles in immune response and nutrient absorption

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This date is consistent across multiple sources, including the official Parliament of Canada website and the Encyclopedia Britannica entry on Canada

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d1, d3
- **Claim**: The conflict_type is 'Conflict due to outdated information': d1 reports 179 destinations, d2 reports 160, d3 reports 42 under VWP d5 reports 180, while d4 details conditional passport rules — each reflecting different data sources and timeframes, creating conflict due to outdated information

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: October 1968

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 63rd and final volume of the Fairy Tail manga was published on January 23, 2018 a miniseries to celebrate the 20th anniversary of the series is set to start in Weekly Shōnen Magazine on July 29, 2026

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The basilica is described as 'nearly complete' but not yet fully finished , indicating that the 2026 date applies to a major structural milestone rather than the definitive completion

### Sample qacc_a44267c115d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: The remaining one third is located in the extracellular space, which includes fluids outside cells and in blood plasma

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2
- **Claim**: In ideal driving conditions, the advisory speed of 35 mph is suggested, but drivers can be ticketed if their actual speed is unsafe for the current conditions as determined by an officer

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 9-1-1

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d4
- **Supporting Docs Found**: None
- **Claim**: This date, including the official U.S. Treasury Department's list of all U.S. states, which places New Mexico as the 47th state the Congressional Record, which references New Mexico's statehood on January 6, 1912

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: President Hoover and his wife continued the party in the East Wing the following Christmas the Hoovers gave children at the White House toy fire trucks as gifts

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 1996 (Japan)

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 18

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 6

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a grizzly bear, making it the Grizzly Bear flag

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: These crops are subject to market fluctuations and are grown on a large scale for export and local consumption

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: At the state level, California has taken a leadership role in setting its own environmental policies, including a net-zero carbon goal by 2045, though other states also have their own environmental agencies and laws

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Additionally, the U.S. federal government plays a crucial role in setting national standards and providing funding for state and local environmental initiatives the Inflation Reduction Act of 2022 allocated billions of dollars for climate and conservation programs across all levels of government

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: British General Sir William Howe's army of about 16,000 troops defeated the Continental Army of about 15,000 in the vicinity of Chadds Ford, Pennsylvania, near Philadelphia

### Sample situatedqa_temp_1baff64de20e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: While her initial contract was for a single episode, there is a possibility of her recurring later in the season

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: 112

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Multiple election observers noted that the post-election environment was marred by allegations of widespread rigging, which PTI and the army both denied

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Wilson Phillips is a vocal trio consisting of Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The ship is operational and has conducted maiden deployments, including sailing to the Indo-Pacific in 2021

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: It originated among the Anglo-Saxon tribes of Britain and was first recorded in the Domesday Book of 1086, with the Latin forms Gerardus and Girardus listed there and also in Norfolk and Lincolnshire records from 1134–1162

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: This battle is notable for being one of the earliest military engagements recorded in detail and for giving rise to the world's first peace treaty, signed in 1258 BCE between Ramesses II and Hattusili III after the death of Muwatalli II

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The WBO title is held by Daniel Dubois, making this a situation where one boxer holds four belts and another holds the fifth, a common arrangement in professional boxing

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The 76ers last made the playoffs in 2021

### Sample trust_align_035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is insufficient to answer the query

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is mixed and does not support a single definitive answer

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict_type is 'Complementary information' because the documents cover distinct but compatible facets: alcohol damage causes scarring (liver pathology) versus liver regeneration after donation (liver physiology)

### Sample trust_align_119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Cowardly Lion was played by Bert Lahr in the 1939 film

### Sample trust_align_160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is outdated and does not support a current head coach answer

### Sample trust_align_173

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is mixed and does not support a clear, authoritative answer

### Sample trust_align_181

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Shinichiro Watanabe

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the Wikipedia article on the 2022 FIFA World Cup, which states that Argentina defeated France 4–2 in a penalty shootout after the match ended 3–3 after extra time, claiming their third World Cup title and their first since 1986

### Sample wikirevision_0057

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the 2023 ICC Men's Cricket World Cup, their sixth Cricket World Cup title


================================================================================

*Report generated by CATS v2.0*
