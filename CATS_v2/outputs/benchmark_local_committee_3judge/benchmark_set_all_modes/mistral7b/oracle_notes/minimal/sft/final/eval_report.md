# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 125 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.940 (over 736 samples)

**GR F1** *(used in CATS)*: 0.963

**Behavior Adherence**: 0.787 (over 611 applicable samples)

**Factual Grounding**: 0.823 (over 611 applicable samples)

**Single-Truth Recall**: 0.666 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.810

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.963
- **Precision**: 0.995
- **Recall**: 0.933
- **Accuracy**: 0.940
- TP=567, FP=3, FN=41, TN=125

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.753
- **Abstain Recall**: 0.977
- **Abstain F1**: 0.850
- **Specificity**: 0.933
- Abstain TP=125, FP=41, FN=3, TN=567


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (55 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.994
- **Behavior**: 0.878 (n=156)
- **Grounding**: 0.903 (n=156)
- **Recall**: 0.808 (n=154)
- **CATS**: 0.896

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.896
- **GR F1** *(used in CATS)*: 0.931
- **Behavior**: 0.904 (n=177)
- **Grounding**: 0.764 (n=177)
- **Recall**: 0.545 (n=156)
- **CATS**: 0.786

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.625 (n=96)
- **Grounding**: 0.823 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.812

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.949
- **GR F1** *(used in CATS)*: 0.972
- **Behavior**: 0.683 (n=145)
- **Grounding**: 0.859 (n=145)
- **Recall**: 0.686 (n=140)
- **CATS**: 0.800

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.757
- **GR F1** *(used in CATS)*: 0.862
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.628 (n=37)
- **Recall**: 0.514 (n=37)
- **CATS**: 0.670


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2105

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

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: If you have high blood pressure or other cardiovascular risks, you should exercise caution with heavy lifting and consider modifications such as lighter weights or alternative exercises

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d3
- **Claim**: The retrieved evidence presents conflicting findings. Some sources indicate that peeling apples can remove significant amounts of dietary fiber (up to 50%) and antioxidant compounds (such as vitamin E and vitamin K), while other research suggests that peeling does not significantly reduce vitamin C content and may even offer health benefits due to antioxidant activity in the peel

### Sample conflictingqa_1d741779a31e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: If you have diabetes, discussing the specific types and amounts of artificial sweeteners you plan to use with your healthcare provider is strongly advised

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1, d2, d4
- **Claim**: The answer depends on how you count them: cows do have four stomachs if you treat each of the four compartments as a separate stomach, but they technically have only one stomach if you merge the compartments into a single organ

### Sample conflictingqa_29f69e16a0c3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Research is ongoing to fully understand these communication pathways, but the weight of current evidence clearly supports that flowers do engage in meaningful interaction with their pollinators

### Sample conflictingqa_34610226ee3f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d2, d3
- **Claim**: [[

### Sample conflictingqa_3601f7480501

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Overall, the research indicates that the impact of unlimited PTO depends heavily on cultural context, managerial support the specific workforce involved

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Additionally, the American Library Association recognizes audiobooks as legitimate formats that count toward reading goals research with adult learners found that audiobooks helped them retain information as effectively as reading textbooks

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d5, d3, d4
- **Claim**: The question of whether real Christmas trees are more sustainable than artificial ones is genuinely contested in the evidence. Some sources argue that real trees are the definitive sustainable choice because they are grown in sustainable farms, harvested sustainably can be recycled or planted again, while others argue that artificial trees are more sustainable if used for many years (more than 20), as their manufacturing and transport emissions are higher but their lifespan is longer

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: On the other hand, critics and some scientists contend that the revenue generated is too often pocketed by a small number of wealthy individuals, that the practice normalizes animal cruelty that the long-term impacts on species populations are insufficiently monitored or understood

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: On the other hand, the legal standards for patentability are frequently cited as too narrow and uncertain to provide meaningful protection for software inventions the Supreme Court's 2014 decision in Alice Corp. v

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d4
- **Supporting Docs Found**: None
- **Claim**: CLS Bank Int'l has led to significant limits on the issuance of software patents , while some critics argue that software should not be patentable subject matter at all because it is abstract, rapidly obsolete easily worked around

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d3
- **Supporting Docs Found**: None
- **Claim**: These conflicting findings reflect methodological differences in how researchers interpret the same phenomena, as probabilistic links to full moons do not constitute definitive causation

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Bees will only fly in heavy rain if absolutely necessary, such as when defending their hive or foraging in an emergency

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d4
- **Claim**: Overall, the weight of evidence points to saturated fats being associated with an increased risk of heart disease, but the evidence is not uniform — high-risk individuals should exercise caution with saturated fat intake, while those at low risk may derive minimal benefit from reducing it

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, other research presents a more nuanced picture — a peer-reviewed comparison of organic and conventional farming systems found that organic farming has lower per-unit energy use and produces fewer greenhouse gas emissions a review of multiple studies found that organic farming can match conventional yields when soil health is prioritized over maximizing production

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: The overall consensus is that organic farming is not inherently more efficient than conventional farming across all metrics, though it may offer environmental and health benefits that are not captured in conventional yield measurements

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Religious truth claims are inherently theological debates, not historical or philosophical proofs

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: Overall, the evidence suggests that while farmed salmon is broadly considered a healthy food, wild salmon may offer slightly more nutritional benefit, particularly for those seeking to minimize contaminant exposure

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: However, the specific combination and context of calls within a species can vary considerably some research suggests that individual birds may develop distinct vocalizations — such as song variations or alarm signals — that set them apart from their peers

### Sample conflictingqa_8cf8ebb94554

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d2, d3, d4
- **Claim**: Research continues to address these conflicting findings, with some evidence suggesting braces may reduce the risk of specific injuries like ACL tears in certain populations , while others report that knee braces do not reduce the incidence of meniscus tears or that their use does not protect against knee injury in youth sports

### Sample conflictingqa_962d8f5d5574

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: All snakes are not equally adept at swimming; experts and research consistently show that only about 50% of snake species are truly aquatic, while the remaining species are terrestrial or arboreal and may only swim occasionally

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d3, d4
- **Claim**: In summary, while glyphosate is widely used and considered safe when following label instructions, a growing body of scientific criticism challenges this view some regulatory agencies (such as the WHO) have called for additional research to better assess long-term health impacts [d3.5]

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d2
- **Claim**: Contrary to a common misconception, stalactites do not necessarily require dry air to grow — they can also form under water through a process known as dripstone formation, where water drips from the ceiling of an underwater cave, leaving a trail of calcium carbonate deposits behind

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Others note that water temperature has negligible effect on the cuticle that any shine from cold rinsing is largely due to the water evaporating, not the temperature itself

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: Taken together, the evidence suggests that human brain size has not uniformly decreased; rather, changes have been linked to shifts in body size, diet cognitive processing demands across different periods of hominid evolution

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3, d4
- **Supporting Docs Found**: None
- **Claim**: However, while one source notes that yeast biomass offers a complete protein profile comparable to traditional sources like meat and soybean , other evidence suggests that fortified nutritional yeast can contain very high levels of added vitamins — including vitamin B12 — which may push consumers over the recommended daily limit if used excessively

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: The underlying mechanism appears to be fear: a 1991 Gallup poll found that 88% of Americans feel uncomfortable discussing death the Red Cross reports that 60% of people avoid discussing it entirely , suggesting that death's status as a taboo is rooted in anxiety about the subject rather than a deliberate cultural choice

### Sample conflictingqa_d295f9ea94b3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1, d5
- **Claim**: While dermatologists and plastic surgeons alike can administer Botox injections, the procedure itself is not a surgical reshaping or reconstruction of body parts

### Sample conflictingqa_d9a36fe4c135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Religious and philosophical views differ; science has not established the Bible as historically or scientifically infallible

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: One specific example is wash trading, where a manipulator places large buy and sell orders to artificially inflate trading volume and price, a practice used to accumulate assets cheaply or exploit stop-loss orders

### Sample conflictingqa_dc6f972e8447

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: Additionally, derivatives exchanges can contribute to price movements by amplifying the effects of leverage and margin calls rumors or negative public comments can also induce panic selling and volatility

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d5, d2, d3
- **Claim**: The retrieved evidence is conflicting. Some sources say the Phoenix Lights were explained as military flares dropped during a training exercise, while others say witnesses believed the lights were UFOs rather than flares

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, a specific black hole located approximately 1,560 light-years away from Earth — known as GW150915 — has been observed directly, appearing as a distorted image of the galaxy it lies within

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: Religion is contextually defined Mormons consider themselves Christians — but their core doctrines are historically alien to orthodox Christianity, making their self-identification a matter of religious interpretation rather than doctrinal alignment

### Sample conflictingqa_fcdb9e210683

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: In practice, the broader phylogenetic tree of life may include viruses in a distinct branch or supergroup, while the more conventional cellular tree does not

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5, d3, d4
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2d7eb41139aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: This version supersedes earlier releases like .NET 5.0 (released May 10, 2022) and .NET 4.8.1 (released August 9, 2022), which are no longer the most current versions

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: You can download .NET 6.0 from the official Microsoft website it is explicitly identified as the latest stable release

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The test, code-named 'Trinity,' involved the detonation of a plutonium implosion device atop a 100-foot steel tower, releasing approximately 18.6 kilotons of energy

### Sample freshqa_39dcd7b38c39

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5, d3
- **Claim**: This is consistently confirmed across multiple sources, with the coin program featuring twenty women over four years Angelou being selected as the first honoree

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This escalated to a full-scale war that has displaced millions of people and resulted in thousands of deaths

### Sample freshqa_3dc3cf00bce6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1, d3, d4
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that one low-quality source incorrectly states she was presented with a Pembroke Corgi on her 18th birthday, but this is inconsistent with the well-documented facts from high-credibility sources

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Three seasons — seasons 1 through 3 — of The Mandalorian have been released, premiering on November 12, 2019, October 30, 2020 March 1, 2023 respectively

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4e635a2542a8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: This lineup was assembled in 1955 and remained together through 1956, with Garland as the pianist throughout

### Sample freshqa_5ecee1c55713

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: Portugal won the 2017 Eurovision Song Contest with the song "Amar pelos dois" performed by Salvador Sobral

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: This figure is corroborated by Costco's own website, which states that Executive Members receive an annual 2% reward of up to $1,250

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: The film, directed by Paul Thomas Anderson, also won Best Director and Best Adapted Screenplay, making it a major sweep of multiple categories

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d3, d4
- **Supporting Docs Found**: None
- **Claim**: This result, superseding earlier reports that listed *CODA* (2022) or *Sinners* (2024) as the most recent winners, as those ceremonies have since been surpassed by the 2026 event

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

### Sample freshqa_80642f637dc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: His achievement is further corroborated by FIFA's own records showing he is the first player to win a Golden Ball and Golden Boot double

### Sample freshqa_8ab63ffc9a7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: He has consistently confirmed this birthplace across multiple sources, including his own website and biographical information about him

### Sample freshqa_8eca5bd62ae0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d5
- **Claim**: The 2022 Winter Olympics were the first to be held separately from the Summer Games Beijing's selection as the host city was confirmed in July 2015

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The award was announced at the 2025 Nebula Conference, making this the most recent recognition for a novel published in 2025

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Frank Rosenblatt died in a boating accident on July 28, 1971, in Chesapeake Bay

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: No, the Toronto Raptors do not have a winning record in the latest NBA season. The most recent season in the provided evidence is 2023–24, during which the Raptors finished with a 25–57 record — not a winning record

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4
- **Claim**: Queen Elizabeth II of England died on 8 September 2022

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by Britannica, which identifies 26 books as her total output by a New York Times article noting her bestselling success

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: Bezos owned approximately 905 million Amazon shares, valued at close to $234 billion

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_cf331ed7d09f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This update introduced new features such as improved performance, enhanced security optimizations for Apple's latest hardware

### Sample freshqa_d510972df578

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d5
- **Claim**: This figure is corroborated by recent studio disclosures, superseding earlier reports that named Pirates of the Caribbean: On Stranger Tides as the top budget film is further supported by adjustments for inflation

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

### Sample freshqa_e502143179d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3
- **Claim**: Following the acquisition, Musk laid off about half of Twitter's workforce and reinstated the account of former President Donald Trump, confirming his control of the platform

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by Yamagata University researchers, who announced 248 additional geoglyphs in July 2025, noting that AI technology has enabled discoveries 20 times faster than traditional methods

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample healthcontradict_0060

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: A multi-institutional review of 43 studies found that tepid sponge baths did not lower fever any more than cooling towels or no treatment at all a separate study of 150 children confirmed that tepid sponge baths did not reduce fever duration or fever peak

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0063

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d10
- **Supporting Docs Found**: d5
- **Claim**: This directly answers the query, as no other universities are located in Chestnut Hill, making Boston College the only private research university in the area

### Sample hotpotqa_0070

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

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d8
- **Claim**: Lucas di Grassi

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d10
- **Supporting Docs Found**: d6
- **Claim**: St James Street appears as a segment of Whitecross Street on the 1610 map of Monmouth by English historian and cartographer John Speed

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d1, d5, d7, d6
- **Claim**: Pusha T

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

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d4
- **Claim**: A third theory holds that the practice developed among early Christians as a secret sign used to recognize each other and invoke the power of the Christian cross for protection, particularly against evil

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The Dominion of Canada became Canada's national identity, replacing the earlier colonial designations of Upper Canada and Lower Canada

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Steve McEwan

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: October 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nana in Snow Dogs is an Australian Shepherd

### Sample qacc_8dd2323c077d

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

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d5
- **Claim**: It was characterized by absolute monarchy, as the emperor held supreme authority and could exercise severe punishments against perceived threats to his power

### Sample qacc_a6a2f8b1f0b8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by multiple sources reporting on the same timeframe, with a Quora-based answer and a 2023 data from a different source both confirming the same total count

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: This joint allows for movement and sound transmission between the two bones, which is essential for hearing

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_c88807a22775

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that some sources also mention .223 caliber rifles as an alternative, but these are typically used in military and hunting contexts rather than Olympic biathlon competition

### Sample qacc_ce4983c8a9c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5, d2
- **Supporting Docs Found**: None
- **Claim**: Some sources suggest the practice continued as recently as 1,500 years ago, with evidence indicating mounds were built as late as A.D. 1600 in some regions

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The first number ever issued was assigned to John David Sweeney, Jr. of New Rochelle, New York within three months, 25 million numbers had been issued nationwide

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: This is a well-documented, established fact from multiple authoritative sources, including the official Social Security Administration website and the St. Louis Federal Reserve Economic Data

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: 45D is a toll (cuota) highway, while Fed

### Sample qacc_e326d0094f42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: 45 is the free (libre) version

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: In that campaign, Rangers finished second in their group and were eliminated in the round of 16, marking their best performance since the 1990s

### Sample qacc_ec5b0067c29a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that the Vice President's residence is separate from the Naval Observatory's main facility the building itself was constructed in 1893 for the Superintendent of the U.S. Naval Observatory

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells played the character Wez in Mad Max 2: The Road Warrior

### Sample qacc_f1776add7672

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For example, a code like M05.0111 uses the first three characters to specify the organ system (M), the next two to define the category (05) the last two to add etiology or site details (111)

### Sample qacc_f2218f8c979e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while 'prime rib' is often used as a general term for high-quality beef from the rib section, it does not exclusively mean USDA Prime-graded beef

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: For those in the Americas, the farthest point from the sea is a location in central Canada called the 'Northwest Bay of Canada' for Australians, the furthest point from the sea is a place called 'The Gap' in the Great Sandy Desert of Western Australia

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d1, d3
- **Claim**: While the program's origins trace back to earlier proposals and the 1934 Committee on Economic Security, the modern Social Security system was established by the 1935 Act

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: In short, the answer depends on which time period is examined — historical European origins versus more recent Latin American and Asian sources

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a grizzly bear, which is also the official state animal of California. The bear on the flag is an extinct population of the North American brown bear, known as the California grizzly bear (Ursus arctos californicus) its inclusion on the flag dates back to 1846, when it was used as the flag of the Bear Flag Republic before becoming the official state flag in 1953

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

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Shri Kiren Rijiju, who serves as a Cabinet Minister in the Ministry of Law and Justice

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: He is a senior BJP leader and has served in various legal and political roles, including as Minister of State for Home Affairs and as a Member of Parliament

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: As a result, the answer depends on the specific jurisdiction in question

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: By the early 1780s, influential groups began pushing for a stronger national government the resulting Constitutional Convention produced the current U.S. Constitution, which replaced the Articles of Confederation in 1788

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: By the early 20th century, coffee had become the dominant beverage in American culture, with approximately 75% of adults drinking it daily , though tea remained popular in the South and with immigrant communities

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: d3
- **Claim**: The FOMC is a part of the Federal Reserve System and consists of twelve members, including seven from the Board of Governors and five presidents from Federal Reserve Banks, who meet regularly to influence money supply and interest rates

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: England are the current champions after winning the 2019 World Cup, with New Zealand as the runner-up

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_301378915064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This was her first Emmy win and second consecutive nomination for the show, making it a significant career milestone

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_50748f92be3a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: Argentina defeated France 4-2 in the final held at the Lusail Stadium in Qatar, claiming their third title

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This version is available for Pixel devices and Samsung Galaxy devices, with other manufacturers like OnePlus, Xiaomi Nokia following shortly after

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: It is written by Brandon Hoáng and illustrated by BellBessa, with covers by Danielle Weires

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: PTI's victory gave Imran Khan his first term as Prime Minister, after which he promised to investigate the allegations of electoral misconduct

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d5, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: 59,681 kilometres (approximately 37,179 miles)

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The Muslim army, which numbered around 313 soldiers, emerged victorious from the battle, killing approximately 150 of the opposing Quraysh army

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This victory was considered a turning point in the early days of Islam, as it established Muhammad as a formidable military leader and set the stage for further conquests

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d1
- **Claim**: It was originally formed from the Old German name Gerhard, which itself consisted of the elements gēr ('spear') and hard ('hardy, brave strong')

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3, d4
- **Claim**: Rhys Ifans plays Eyeball Paul in Kevin and Perry Go Large

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

### Sample situatedqa_temp_f971e49123a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that one source suggests the risk of death is particularly elevated for individuals with a history of drug abuse or heart conditions, as their systems may be more susceptible to the cardiac effects of inhalants

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. The retrieved evidence consistently identifies Alphabet Inc. as the company that owns Google. The newer Wikipedia revision directly states that Alphabet Inc. is the parent company to Google, while the older revision and additional sources further confirm this ownership through consistent contextual and subsidiary relationship evidence

### Sample wikirevision_0010

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Microsoft Corporation

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: This victory marked their third title, making them the first nation from outside of Europe to win the tournament since 2002

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Argentina defeated France 4–2 on penalties after extra time in the final, securing their first title since 1986

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed by the Haryana government in 2022 the name change is consistently reflected across all sources including the newer Wikipedia revision of Gurgaon and the Rapid Metro Gurgaon website. While 'Gurgaon' is still commonly used, especially outside of Haryana, the official civic body of the city is the Gurugram Municipal Corporation

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore is officially called Bengaluru. The city officially changed its name from Bangalore to Bengaluru on 1 November 2014 this is confirmed by the Bangalore City Municipal Corporation resolution of 2014-11-01. As a result, Bangalore is no longer the official name of the city, which is now officially called Bengaluru

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This victory marked his second French Open title and fifth major, cementing his status as one of the top players on the tour


================================================================================

*Report generated by CATS v2.0*
