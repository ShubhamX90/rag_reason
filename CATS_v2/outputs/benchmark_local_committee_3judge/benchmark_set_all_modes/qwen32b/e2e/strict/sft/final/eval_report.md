# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.963 (over 736 samples)

**GR F1** *(used in CATS)*: 0.977

**Behavior Adherence**: 0.821 (over 608 applicable samples)

**Factual Grounding**: 0.812 (over 608 applicable samples)

**Single-Truth Recall**: 0.749 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.840

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.977
- **Precision**: 1.000
- **Recall**: 0.956
- **Accuracy**: 0.963
- TP=581, FP=0, FN=27, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.826
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.905
- **Specificity**: 0.956
- Abstain TP=128, FP=27, FN=0, TN=581


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.995
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.974 (n=154)
- **Grounding**: 0.935 (n=154)
- **Recall**: 0.844 (n=154)
- **CATS**: 0.937

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.946
- **GR F1** *(used in CATS)*: 0.965
- **Behavior**: 0.949 (n=176)
- **Grounding**: 0.770 (n=176)
- **Recall**: 0.692 (n=156)
- **CATS**: 0.844

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.573 (n=96)
- **Grounding**: 0.767 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.775

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.697 (n=145)
- **Grounding**: 0.820 (n=145)
- **Recall**: 0.743 (n=140)
- **CATS**: 0.811

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.811
- **GR F1** *(used in CATS)*: 0.896
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.590 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.702


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1836

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
- **Supporting Docs Found**: d4, d3
- **Claim**: Yes — many salamanders are poisonous to touch, as nearly all species possess toxins in their skin that can cause serious illness if ingested some (like the fire salamander) are well-documented as toxic. However, the degree of danger varies by species: tiger salamanders and yellow-spotted salamanders are generally considered safe to handle gently, as long as hands are washed afterward to remove bacteria

### Sample conflictingqa_0c3c7b487766

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3
- **Claim**: High iodine intakes are generally well tolerated by healthy individuals, but excess supplementation can disrupt thyroid homeostasis and alter hormone production research has shown that daily intakes of 200 μg or more in pregnancy were associated with a 2.5-fold increased risk of elevated TSH compared to intakes below 100 μg

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Yes, peeling an apple removes a significant portion of its nutritional value, particularly fiber and antioxidants, though it does not reduce vitamin C content

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2, d3
- **Claim**: Yes — palm oil production causes serious environmental harm through deforestation, loss of biodiversity pollution; however, sustainably certified palm oil may mitigate some of these impacts

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Dog breeding is not universally considered unethical, but neither is it unconditionally accepted as ethical. The practice is ethically contested along several dimensions: some argue that breeding is inherently exploitative and contributes to overpopulation, while others contend that breeding can be conducted responsibly with proper regulations and health screenings. The ethical debate thus centers on whether any breeding is acceptable or whether all commercial breeding should be discouraged in favor of adoption

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Cows technically have one stomach that is divided into four distinct compartments — the rumen, reticulum, omasum abomasum — rather than four separate stomachs

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1, d2, d3
- **Claim**: The Silurian period marks the first documented appearance of land plants, though the evidence suggests a gradual transition rather than a sudden birth. Several sources confirm that the Silurian saw the emergence of simple vascular plants—such as Cooksonia—on land for the first time, with some researchers placing the earliest radiation of land plants (embryophytes) even earlier, in the Middle Ordovician. This makes the Silurian a critical period in the evolution of terrestrial life, but not necessarily the moment of the very first land plants

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Not directly from chlorine; chlorine lightens hair and oxidizes copper, but copper is the actual green-staining agent

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: There is no single authoritative answer — proponents argue that audiobooks are genuinely reading (or at least count equally toward reading goals), while a notable minority of adults and some commentators dissent

### Sample conflictingqa_3c835387fe6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: This evolutionary origin in Australia has been confirmed by multiple studies, including research from The Australian National University, which found that Komodo dragons interbred with Australian lizard ancestors before crossing over to Indonesia

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Yes — real Christmas trees are generally considered more sustainable than artificial ones, as real trees absorb CO2 while growing, can be recycled after use are sourced from renewable farm-grown stocks, whereas artificial trees are made from non-biodegradable plastics and metals, cannot be recycled emit significant greenhouse gases during manufacturing and transport

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: The query's claim is false. The trash island is not as large as Texas; it is approximately 2–3 times the size of Texas

### Sample conflictingqa_5233eab573e5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: Yes — in the US

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are valuable for protecting investments and encouraging innovation, while others argue that software is too abstract or mathematical to qualify for patent protection that granting patents hampers rather than helps innovation

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, it is not universally recognized as the single deadliest eruption of all time, as some sources distinguish between eruption power and total mortality: the Huaynaputina eruption of 1600 Peru and the Laki eruption of 1783 Iceland caused comparable or greater death tolls through famine and environmental disruption, illustrating that 'deadliest' can be defined by different metrics than sheer eruptive power

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: Methodological differences across studies — including variability in dietary assessment, population characteristics the specific fats used for replacement — contribute to these divergent conclusions, making the evidence as a whole inconclusive and warranting continued investigation

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d5, d3
- **Claim**: Overall, the consensus across large-scale nutritional analyses is that the differences are moderate and neither type consistently outperforms the other across all metrics

### Sample conflictingqa_9251c7a33ec5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Individual factors such as age, breed sex appear to moderate the balance of benefits and harms the optimal decision is often debated

### Sample conflictingqa_9261438d6ee2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Yes — the scientific consensus, based on multiple peer-reviewed studies, is that fish do possess pain receptors and respond to noxious stimuli in ways similar to mammals . Research has shown that fish exhibit behaviors indicative of distress when exposed to painful stimuli, such as rapid breathing, rocking motion altered blood chemistry that these responses are reduced by analgesic drugs

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d4, d3
- **Claim**: Stalactites do not form underwater

### Sample conflictingqa_a3980a2921cf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Not all hair types and concerns benefit from hair oil in the same way; some people with fine hair or oily scalps may find it weighs their hair down or causes breakouts

### Sample conflictingqa_a507c5b61631

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: However, other evidence suggests that the PETM onset may have involved multiple carbon reservoirs—such as methane release from ocean sediments or permafrost—potentially triggered by tectonic shifts or rising temperatures, indicating that volcanic activity may have been a contributor rather than the sole cause

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence is divided on this question. Some sources argue that cold water can seal the cuticle, reduce frizz improve shine, while others argue that the effect is negligible, temporary easily reversed by subsequent heat styling

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Some sources go further, stating that coffee grounds actually attract slugs and snails rather than repelling them that the 0.06–0.1% caffeine found in typical brewed coffee is well below the ~0.1% threshold shown in University of Nebraska research to reliably deter snails

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Religious and theological views differ; science offers no settled answer

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Heads differ on whether barefoot running is healthier; the predominant scientific evidence suggests it can strengthen foot muscles and reduce certain injuries, but risks of road debris and stress fractures persist

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Emoji are widely used as written signs, but experts differ on whether they constitute a form of written language

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence indicates that the Dutch were among the earliest Europeans to explore and map Australia, with Willem Janszoon's 1606 voyage to Cape York Peninsula representing one of the first recorded European sightings of the continent. However, the snippet does not explicitly state that the Dutch 'discovered' Australia in the absolute sense, nor does it address competing claims by other nations or indigenous precedent

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1
- **Claim**: Once considered the same dinosaur, Apatosaurus and Brontosaurus were reclassified as distinct genera in a 2015 study

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: For example, the Event Horizon Telescope produced the first direct image of the black hole at the center of M87 in 2019, showing a dark core surrounded by a ring of emission subsequent observations have continued to refine these images

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d2
- **Claim**: Religious scholars and commentators differ on whether to classify Mormons as Christians; the answer depends significantly on the definition of Christianity being applied

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: Aryna Sabalenka and Amanda Anisimova

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

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: This year's Passover (Pesach) began on April 2, 2026

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

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: The latest major version of .NET is 7.0

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

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7bc92b47dc43

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: 2

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is the most recent full season record available in the evidence

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The heaviest reptile in the world is the saltwater crocodile (Crocodylus porosus), which lives in Southeast Asia, Northern Australia New Guinea

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: 13

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: A permanent cure for cancer has not been developed

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The Bills vs. Bengals game on January 2, 2023, resumed play approximately 21 minutes after Damar Hamlin suffered cardiac arrest on the field

### Sample freshqa_e1a40e76ba15

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d3
- **Claim**: The game was first suspended pending further notice while medical personnel attended to Hamlin, who received CPR for approximately nine minutes before being placed in an ambulance

### Sample freshqa_f11e8d758fc6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The apparent single-lung consensus applies to the vast majority of slug species, making the answer 'one' a broadly valid but incomplete summary of the full taxonomic diversity

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2, d3
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: A total of 893 Nazca geoglyphs had been discovered, comprising 645 nonfigurative lines and geometric forms along with 248 known figurative geoglyphs

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: February 18–March 19

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d10
- **Supporting Docs Found**: d5, d7
- **Claim**: Japanese colonial rule in Korea ended with the conclusion of World War II

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6
- **Supporting Docs Found**: d3
- **Claim**: Stuart

### Sample qacc_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d7, d4
- **Claim**: Pusha T wrote the "I'm Lovin' It" jingle for McDonald's, according to multiple reports including The Washington Post and BBC

### Sample qacc_08cf866bcb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Norma Koch

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_132167e66120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: India won the Cricket World Cup in 1983, 2007, 2024 2026

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: These early vertebrates belonged to the broader group of craniates, characterized by the presence of a skull and, later, a vertebral column surrounding the neural tube

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2, d3
- **Claim**: As a player, Bill Russell holds the record with 11 NBA championships; as a coach, Phil Jackson holds the record with 11 championships. Combined across both roles, Red Auerbach has the most with 16 championships (nine as a coach and seven as an executive)

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: While Peyer's patches are also important lymphoid structures in the ileum of the small intestine, they are not the lymphatic vessels themselves but rather clusters of lymphoid tissue associated with the villi

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6f8df54650a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: A U.S. passport provides visa-free or visa-on-arrival access to 180 countries and territories, making it among the most powerful passports in the world for travel freedom

### Sample qacc_720d8b210357

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3
- **Claim**: Multiple — eukaryotic DNA replication initiates at multiple origins of replication the number varies by organism

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: October 1, 1968

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d1
- **Claim**: Nana in Snow Dogs is a collie

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

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

### Sample qacc_a3c882e062c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: The planned completion date for the Sagrada Familia has been updated to the early 2030s, superseding the earlier target of 2026

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2, d3
- **Claim**: Hosanna is a Hebrew expression meaning 'save us' or 'save us now,' derived from the phrase 'hoshi'a na.' It is used as a cry for salvation or deliverance, particularly in Jewish and Christian religious contexts, including the Feast of Tabernacles and the biblical account of Jesus' entry into Jerusalem

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: George Bruns composed the music for Disney's 1973 animated film Robin Hood

### Sample qacc_d39801b5de65

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2, d3
- **Claim**: July 4, 1776

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2
- **Supporting Docs Found**: None
- **Claim**: The aircraft was named after Enola Gay Tibbets, the mother of the pilot, Colonel Paul Tibbets participated in the second nuclear attack as the weather reconnaissance aircraft for the primary target of Kokura

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2025–26

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: For example, a basic code like K26.1 (acute duodenal ulcer with perforation) is only five characters long, whereas a more detailed CM code such as S32.010A (wedge compression fracture of the first lumbar vertebra, initial encounter) extends to seven characters

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: 7

### Sample situatedqa_geo_082db791e263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4, d3
- **Claim**: It depends on jurisdiction; federal law (U.S.) generally requires at least 18 to buy a shotgun, though many states have raised the minimum to 21 for all firearms

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: American casualties in World War II stood at approximately 416,800 military deaths and 1,700 civilian deaths, according to the National WWII Museum's compilation of country-level data

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: In the US, President Roosevelt's 1935 Social Security Act is considered the foundational legislation , while Canada's welfare state similarly took shape via the 1930s–1950s

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: The Clean Air Act was first passed in 1970, signed by President Richard Nixon on December 31, 1970, although earlier federal air pollution laws were enacted as early as 1955

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Kennedy

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California grizzly bear (Ursus arctos californicus)

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_d982055a66d8

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
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: None
- **Claim**: In modern times, the switch is often framed as a personal health choice rather than a political one, with green tea and herbal alternatives gaining popularity as lower-caffeine, antioxidant-rich substitutes

### Sample situatedqa_temp_14a587def215

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: This appearance marked their first trip to the Stanley Cup Final in 20 years, as they previously reached the Final only in 2006

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Darren Criss

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: The latest stable version of Android is Android 16, which was released on June 10, 2025. This supersedes earlier information from May 2025 identifying Android 15 as the current version confirms that Android 16 is now the most recent stable release

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2, d3
- **Claim**: 1980

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Todd Monken

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2, d3
- **Claim**: From the battle with Grendel in Beowulf, Grendel is described through kennings such as 'twilight-spoiler' and 'shepherd of evil,' while the narrator also uses 'sea-wood' for ship and 'battle-sweat' for blood in that context

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: This figure is confirmed by high-accuracy geographic data from the Australian Government's Geoscience agency, which records 35,821 km of mainland coastline and 23,860 km of island coastline, summing to the total

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4, d2
- **Claim**: The Cumberland River begins in the mountains of eastern Kentucky, formed by the confluence of its headwater forks — Poor Fork, Clover Fork Martin's Fork — near Harlan County (Wikipedia, Tennessee Encyclopedia)

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: It flows generally westward through Kentucky before turning south into Tennessee, traveling almost 700 miles and draining a watershed of approximately 18,000 square miles (Tennessee Encyclopedia)

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4, d2
- **Claim**: The river ultimately ends by merging with the Ohio River at Smithland, Kentucky, northeast of Paducah, effectively terminating its journey (Wikipedia, Tennessee Encyclopedia, Nashville MLS)

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: Cardiac biomarkers are substances that appear in the blood when the heart is stressed or damaged they are used to diagnose and monitor heart disease

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: For the 2025-26 NBA season, LeBron James is the highest-paid player at $132.6 million in total earnings, though Stephen Curry holds the top playing salary at $59.6 million — marking the ninth straight year Curry has claimed that distinction. This represents a temporal update from earlier reports identifying Curry or LeBron as the highest-paid, reflecting the most current salary data available

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d2, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 133

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Hello, Love, Again

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2026

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held in Ahmedabad on 19 November 2023. This was Australia's sixth Cricket World Cup title the tournament was the 13th edition organized by the ICC. The 2027 Cricket World Cup is next scheduled to be held in South Africa, Zimbabwe Namibia, making Australia the most recent champion

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Prime Minister of Canada is Mark Carney, who assumed office on 14 March 2025. He is the 24th person to serve as Canada's head of government and holds the highest office of the Canadian federal government. This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the Prime Minister of Canada page, as well as the list of prime ministers of Canada

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022. He is the 31st person to hold the office since its creation in 1901

### Sample wikirevision_0156

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d2
- **Claim**: The current President of the Philippines is Bongbong Marcos, who assumed office on June 30, 2022. He is the 17th President of the Philippines and serves as both head of state and head of government. This is consistent across multiple high-credibility sources, including the older and newer Wikipedia revisions of the President of the Philippines page, as well as the list of presidents of the Philippines


================================================================================

*Report generated by CATS v2.0*
