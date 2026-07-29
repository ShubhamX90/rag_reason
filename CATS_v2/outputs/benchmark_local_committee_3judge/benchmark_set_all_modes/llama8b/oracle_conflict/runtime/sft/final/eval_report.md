# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 127 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.973 (over 736 samples)

**GR F1** *(used in CATS)*: 0.983

**Behavior Adherence**: 0.808 (over 609 applicable samples)

**Factual Grounding**: 0.853 (over 609 applicable samples)

**Single-Truth Recall**: 0.704 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.837

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.983
- **Precision**: 0.998
- **Recall**: 0.969
- **Accuracy**: 0.973
- TP=589, FP=1, FN=19, TN=127

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.870
- **Abstain Recall**: 0.992
- **Abstain F1**: 0.927
- **Specificity**: 0.969
- Abstain TP=127, FP=19, FN=1, TN=589


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.909 (n=154)
- **Grounding**: 0.906 (n=154)
- **Recall**: 0.818 (n=154)
- **CATS**: 0.908

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.977
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.881 (n=177)
- **Grounding**: 0.870 (n=177)
- **Recall**: 0.631 (n=156)
- **CATS**: 0.842

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.698 (n=96)
- **Grounding**: 0.836 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.841

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.949
- **GR F1** *(used in CATS)*: 0.972
- **Behavior**: 0.717 (n=145)
- **Grounding**: 0.823 (n=145)
- **Recall**: 0.682 (n=140)
- **CATS**: 0.799

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.712 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.734


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2090

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

### Sample conflictingqa_05b33f4ca156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: It depends on the species: some salamanders (like fire salamanders) have toxic skin secretions and should be handled carefully or not at all, while others (like tiger salamanders) are generally not considered poisonous or harmful to humans

### Sample conflictingqa_07e290a954ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Additionally, St. John's Wort can cause serious side effects such as serotonin syndrome when combined with other medications its long-term safety remains unclear , so it should be used with caution and under medical supervision

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that while this fungus is recognized as the largest single living organism, a giant fungus discovered in China has been reported to have the largest fruiting body among all fungi, representing a different metric

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: The retrieved evidence is mixed. Some sources argue that chlorine is not the direct cause of green hair and that copper from algaecides is responsible, while others argue that chlorine can contribute to hair lightening and damage

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This methodological divergence between cognitive limitation and factual assertion creates direct contradictory research outcomes on the same question

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: The conflicting expert opinions and conditional language used by multiple sources mean the answer depends heavily on how the device is used rather than being a universal yes or no

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: d2, d1
- **Claim**: One source argues that IPv6's larger address space does not fundamentally improve security and that IPv4 can also use IPSec , while another notes that IPv6's performance advantages over IPv4 are still debated and not yet conclusive

### Sample conflictingqa_311fca0928d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because these documents present different interpretations and measurements of IPv6's security relative to IPv4

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: One Tree Planted similarly concludes that real trees are more eco-friendly because they can have negligible or even negative emissions when recycled or kept growing, whereas artificial trees release up to 40 kg of greenhouse gases per tree

### Sample conflictingqa_411445406724

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: All sources agree, however, that cycads are no longer ecologically dominant, having been replaced by flowering plants more than 100 million years ago

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: However, other sources argue that trophy hunting is not universally beneficial—phototourism has a larger carbon footprint and can cause animal mortality, while hunting itself can perpetuate a culture of killing for trophies that may not align with conservation goals

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The IUCN report cited by pro-hunting sources is from 2016 and represents a single, advocacy-oriented study, while peer-reviewed research from the University of Oxford's WildCRU found that areas managed for trophy hunting saw higher carnivore mortality than unmanaged areas , highlighting the ongoing debate over whether trophy hunting truly supports long-term conservation goals

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: The evidence is mixed and the answer depends on the stage of CKD and the dose used

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The retrieved evidence is mixed. Some randomized controlled trials have shown that vitamin C can shorten the duration of common colds and reduce their severity by about 15%, while other research concludes that taking extra vitamin C to prevent colds has not been proven and that any benefit is largely limited to slightly shortening the duration of existing infections

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Some sources note that these yield differences are not solely due to organic practices, but also to factors like soil quality and management that high-yield conventional farming can also be improved to reduce environmental impact

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d4
- **Claim**: The Catholic Church claims to be the 'One True Church' founded by Jesus Christ, but other churches also claim to be the true church, making the question a matter of conflicting opinions

### Sample conflictingqa_7cf85109a70d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2
- **Claim**: However, it is worth noting that brass offers its own unique advantages — being easier to machine, more resistant to corrosion in some environments cheaper to produce — making the choice between the two largely dependent on the specific application

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The conflict is methodological: sources differ in how they analyze the relationship between cultural identity, political cohesion social structures

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: They require a secure, well-ventilated tank with a heat source and humidity control, as well as a varied diet of leafy greens and fresh vegetables

### Sample conflictingqa_9ceca2645833

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The conflict arises because the legal standard for affirmative action is that it must be narrowly tailored to address a compelling government interest courts have become more skeptical of such programs over time, making the answer context-dependent and contested

### Sample conflictingqa_a1e36a8db854

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: Overall, while some agencies conclude glyphosate is safe within established exposure limits, others and multiple independent studies report significant harms — creating a methodological and interpretive conflict that the EPA itself has acknowledged by stating that glyphosate is 'unlikely' to cause cancer rather than definitively safe

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Stalactites do form through calcite crystals growing downward from water drips one source notes that they can form in underwater caves through a process it does not fully explain

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d5, d1, d3
- **Claim**: The conflict_type is 'Conflicting opinions or research outcomes' because expert opinions and interpretations of hair physics differ directly on the same question

### Sample conflictingqa_b9854bd5a19e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: A third perspective suggests that any apparent shrinkage is an artifact of changes in body size rather than brain size per se

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The most authoritative source in the set, the BBC Future article, notes that 'the scientific evidence is mixed' and that experts recommend avoiding straws altogether if possible, rather than switching to a single alternative

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: On the other hand, other philosophers argue that certain beliefs are justified by basic perceptual experience or entailment rules that Gettier's counterexamples do not establish that any justified belief can be false

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Overall, the evidence does not yet conclusively resolve the question of whether barefoot running is healthier than running with shoes; further research is needed to fully answer the question

### Sample conflictingqa_f7fec8c0688b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2
- **Claim**: The incident remains unexplained, with the military's official explanation never fully resolving the widespread civilian reports

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: A comprehensive answer must synthesize these opposing views, noting that viral genomes are widely accepted as part of the tree by the scientific consensus cited in , while the dissenting views represent a minority position that is still actively debated

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: This result is corroborated by ESPN's coverage of the 2025 US Open, which confirms Sabalenka defeated Anisimova to win the women's title

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: It begins before sundown on Wednesday, April 1 ends after nightfall on Thursday, April 9, according to the Hebrew calendar

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3
- **Claim**: April 1 as the start date for 2026, with the first seder taking place on the evening of April 1

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Her citation recognized her outstanding contributions to the dynamics and geometry of Riemann surfaces and their moduli spaces her work involved calculating the number of simple closed geodesics on hyperbolic surfaces

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This result ended Max Verstappen's four-year reign at the top of Formula 1, with Norris becoming the 11th British driver to win the championship

### Sample freshqa_28e155139ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The latest stable Android version is Android 16

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: It is a part of the.NET Core family and supersedes the older.NET Framework 4.8.x releases

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: It began with a Russian drone strike in the Sumy region of eastern Ukraine and has caused over 1 million casualties and a 25% decline in Ukraine's population

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This full-scale war resulted in hundreds of thousands of deaths and millions of people fleeing to nearby countries

### Sample freshqa_4a98eba95e97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Biden did not visit Russia as president-elect, as his first foreign trip of his presidency did not occur until June 2024, well after his inauguration

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5574b1447bdb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Multiple independent studies consistently place Wuhan at the center of the outbreak, with the WHO-China joint study finding most molecular evidence pointing to mid-November 2019 as the most recent common ancestor of all variants

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: This finding superseded earlier records of the world's oldest DNA, which had been held by a million-year-old mammoth tooth and environmental DNA from Antarctic sediments

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Academy Award for Best Picture was won by *Anora* (2025), directed by Sean Baker, at the 98th Academy Awards

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by Deadline's gallery of every Best Picture winner, which lists Anora as the 2025 winner is corroborated by the Chicago ABC affiliate's report of the 98th Oscars ceremony

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Andrés Iniesta (2012)

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
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

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The latest Nebula Award for Best Novel is 'The Dragonfly Gambit' (2025), as confirmed by the official Nebula Awards page

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

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5
- **Supporting Docs Found**: None
- **Claim**: All available evidence is fully consistent, with no conflicting dates reported across any source

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c7ac9d61059a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The Tesla Model Y Premium All-Wheel Drive is priced at $51,630

### Sample freshqa_d4d59d75b206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: His ranking in these years without contradiction, though his consecutive run was interrupted by Ed Sheeran's dominance in 2017

### Sample freshqa_dd85dcbc2262

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: She has held the top position for a total of 82 weeks throughout her career, with the most recent stint beginning after her 2023 US Open final appearance

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: 9 minutes

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d5, d1, d4
- **Claim**: 6 months

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10
- **Supporting Docs Found**: d5, d7
- **Claim**: Japanese colonial rule of Korea ended in 1945, when Korea declared independence after Japan's defeat in World War II

### Sample hotpotqa_0073

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: El Nuevo Cojo Ilustrado is a separate publication that was referenced in connection with Vicente Ulive-Schnell's novel and is described as an American online Spanish-language magazine published from Los Angeles, but it is not owned by Time Inc

### Sample hotpotqa_0083

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d7
- **Claim**: Sébastien Buemi (born 31 October 1988)

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Bartholdi's design was commissioned by French historian Édouard de Laboulaye, who proposed the monument to commemorate the upcoming centennial of U.S. independence and the liberation of slaves

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: The statue's internal framework was designed by Gustave Eiffel, but Bartholdi's design is the primary focus of the statue's identity

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: They are organized lymphoid nodules embedded in the mucosa of the ileum, extending into the submucosa play a critical role in immune function by filtering foreign particles and antigens from the gut

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Canada's full independence was further confirmed by the Canada Act of 1982, which declared that no future British laws would apply to Canada without Canadian consent

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: These differences reflect the fact that origin specification in eukaryotes is not strictly sequence-based but also involves epigenetic and chromatin-organization components , meaning the number of origins can vary significantly between species and even within the same organism under different conditions

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: The Airdrome

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The International Space Station (ISS) did not have a single launch or deployment date into space; it was assembled in orbit over several years

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The brain and heart are particularly water-rich at approximately 73% water each, further supporting the idea that most of the body's water is intracellular

### Sample qacc_a6a2f8b1f0b8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: 245

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Carter Pewterschmidt (voiced by Seth MacFarlane)

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The music for Disney's Robin Hood was created by a combination of composers across two different films. The 1952 live-action Robin Hood featured music composed by Elton Hayes — who also wrote and performed original songs for the film — drawing on medieval English melodies for the framing ballads. The 1973 animated Robin Hood, on the other hand, featured music and lyrics by Roger Miller for songs like 'Whistle-Stop' and 'Oo-de-lally', as well as music by Floyd Huddleston for the song 'Love' the majority of the score was composed by George Bruns

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The Tavarez name has been carried by notable figures across a range of fields and has significant connections to the British peerage, tracing its presence in England since the medieval period

### Sample qacc_d03e85bdc95a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: George Bernard Shaw

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Teddy Altman married Owen Hunt on Grey's Anatomy

### Sample qacc_ea469c846404

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Rangers last participated in the Champions League during the 2022–23 season

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells played the character Wez in The Road Warrior Wez is the character with the mohawk

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Sushma Swaraj also held the record for being the youngest Cabinet minister in Haryana government in 1977 and the first woman Chief Minister of Delhi in 1998

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: 7

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 6 fronts

### Sample situatedqa_geo_864fc7e36714

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2
- **Claim**: Today, Mexico remains the largest origin country among U.S. immigrants, accounting for about 22% of the U.S. immigrant population, though immigration from Mexico has slowed since 2007

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: California grizzly bear (Ursus arctos californicus)

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The Ministry of Law and Justice is responsible for the administration of justice, legal affairs the rule of law in India Shri Kiren Rijiju is the Minister heading this important portfolio

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: It consists of twelve members — seven from the Board of Governors and five presidents from Federal Reserve Banks — and meets regularly to decide on interest rates and the money supply through open market operations

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d5, d4
- **Supporting Docs Found**: None
- **Claim**: While the FOMC is the main policy body, the Board of Governors oversees the Reserve Banks and sets the framework for their actions the Reserve Banks themselves participate in all primary activities of the Federal Reserve System

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The British victory opened the way for them to occupy Philadelphia, the American capital, just two weeks later

### Sample situatedqa_temp_180f238d8296

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: Australia, West Indies, India, Pakistan, Sri Lanka, England

### Sample situatedqa_temp_19badef7553b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Their most recent Super Bowl win came in 2025 (Super Bowl LIX), when they defeated the Kansas City Chiefs 40-22

### Sample situatedqa_temp_32d33d503f69

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
- **Claim**: Florida Gators

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: 1939 (for the original 1939 film version by Judy Garland); 1993 (for Bruddah Iz's cover by Israel Kamakawiwo'ole)

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: The latest Android version is Android 16, released on June 10, 2025. It was first released on Google Pixel phones and has since rolled out to Samsung Galaxy and other devices. A newer version, Android 16, has superseded Android 15 as the latest version

### Sample situatedqa_temp_657c130afab6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: It spans 13.2 million acres in Alaska's Southcentral region and borders Canada's Kluane National Park, representing one of the largest wilderness areas remaining in the world

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beowulf contains kennings such as "whale-road" for the sea and "twilight-spoiler" for Grendel

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5, d1, d4
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Wilson Phillips is a vocal trio consisting of Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: Their self-titled debut album in 1990 launched the group to fame with hits like "Hold On," "Release Me," and "You're in Love," and they have since reunited multiple times to record and perform together

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Emily Fields is a fictional character and her age is not directly asked in the snippet

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3, d1
- **Claim**: The longest wavelengths in the visible spectrum are approximately 700 nanometers, which is the range of red light

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The ship then underwent sea trials and was accepted by the Royal Navy before being declared operational in 2020, with her maiden deployment — Carrier Strike Group 21 — following in 2021

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: The island's population is predominantly White (Non-Hispanic), with a median age of 68.3 years and a median household income of $189,109

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Their most recent playoff appearance was in the 2020-21 NBA playoffs, where they were defeated by the Atlanta Hawks in the first round

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The 76ers have not appeared in the playoffs in the 2022-23 or 2023-24 seasons

### Sample wikirevision_0001

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Twitter is currently known as X

### Sample wikirevision_0007

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Alphabet Inc. is a public company listed on the Nasdaq under the ticker symbols GOOGL (Class A) and GOOG (Class C) as of 2025, it reported revenue of $402.836 billion

### Sample wikirevision_0035

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The latest Prime Minister of Japan is Sanae Takaichi, who became Japan's first female Prime Minister on 21 October 2025. She is the 32nd Prime Minister of Japan and the incumbent, serving in office since 21 October 2025

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This was their third World Cup title, their first since 1986 their first from outside of Europe since 2002

### Sample wikirevision_0049

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The 2026 Indian Premier League (IPL) is the 19th edition of the tournament the most recent season

### Sample wikirevision_0049

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Chennai Super Kings are the most successful franchise in IPL history with five titles, but they did not win in 2026

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0082

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d4
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources, including the high-credibility Wikipedia articles on both the President of Indonesia and Prabowo Subianto himself

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Australia

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving a five-year term that is renewable once consecutively resides at Bellevue Palace in Berlin

### Sample wikirevision_0132

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The current Prime Minister of Japan is Sanae Takaichi, who became Japan's first female Prime Minister after assuming office on 21 October 2025. She is the 32nd Prime Minister of Japan and the incumbent, serving in office from 21 October 2025

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Australia


================================================================================

*Report generated by CATS v2.0*
