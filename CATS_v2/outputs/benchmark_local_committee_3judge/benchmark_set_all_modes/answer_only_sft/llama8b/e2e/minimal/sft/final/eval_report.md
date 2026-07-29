# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 128 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.955 (over 736 samples)

**GR F1** *(used in CATS)*: 0.972

**Behavior Adherence**: 0.791 (over 608 applicable samples)

**Factual Grounding**: 0.828 (over 608 applicable samples)

**Single-Truth Recall**: 0.706 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.824

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.972
- **Precision**: 1.000
- **Recall**: 0.946
- **Accuracy**: 0.955
- TP=575, FP=0, FN=33, TN=128

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.795
- **Abstain Recall**: 1.000
- **Abstain F1**: 0.886
- **Specificity**: 0.946
- Abstain TP=128, FP=33, FN=0, TN=575


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (57 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.987
- **Behavior**: 0.942 (n=154)
- **Grounding**: 0.877 (n=154)
- **Recall**: 0.841 (n=154)
- **CATS**: 0.911

### Type 2: Complementary Info

- **Samples**: 221 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.923
- **GR F1** *(used in CATS)*: 0.949
- **Behavior**: 0.926 (n=176)
- **Grounding**: 0.782 (n=176)
- **Recall**: 0.603 (n=156)
- **CATS**: 0.815

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.552 (n=96)
- **Grounding**: 0.852 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.798

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.981
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.662 (n=145)
- **Grounding**: 0.888 (n=145)
- **Recall**: 0.718 (n=140)
- **CATS**: 0.814

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.811
- **GR F1** *(used in CATS)*: 0.896
- **Behavior**: 0.649 (n=37)
- **Grounding**: 0.550 (n=37)
- **Recall**: 0.541 (n=37)
- **CATS**: 0.659


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1998

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
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Trademark law provides additional protection for logos, labels brand names, but not for the look of the garment itself

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The retrieved evidence consistently confirms that chlorine does not directly turn hair green; the green coloration is caused by copper in pool water

### Sample conflictingqa_37ab7146eb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2, d1
- **Supporting Docs Found**: d4
- **Claim**: Others argue that true pain requires biological processes and consciousness, which robots currently lack that any appearance of pain is at most a clever program

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2
- **Supporting Docs Found**: d3
- **Claim**: Real trees also absorb CO2 while they grow and can have negligible or even negative emissions when recycled, whereas artificial trees carry a significant carbon footprint from manufacturing, transportation waste disposal

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The debate thus remains unresolved, with the best answer depending heavily on context and the specific type of species and hunting regime involved

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3, d1
- **Claim**: The U.S. Supreme Court has consistently held that official or school-led prayer in public schools is unconstitutional, while personal, voluntary prayer by students is broadly protected under the First Amendment

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The Great Pacific Garbage Patch is approximately 1.5–1.6 million km^2 in size, which is roughly 2–3 times the area of Texas (about 700,000 km^2), not just twice as large

### Sample conflictingqa_56fd6bf22253

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: A peer-reviewed study found that adenoidal regrowth was correlated with the age of the patient and postoperative antibiotic treatment that most children do not experience a recurrence

### Sample conflictingqa_676188bf8139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: The festival originated as a Buddhist tradition of lighting lanterns on the 15th day of the first lunar month to commemorate the Buddha, while another theory attributes its roots to a myth about the Jade Emperor and his crane

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Additional variation comes from the fact that some Spanish varieties, such as those spoken in Mexico, tend to roll the R more freely than others, so what is considered 'necessary' can depend on the region and individual dialect

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Bees can fly in light rain, but not in heavy rain

### Sample conflictingqa_80857a692531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: The overall scientific consensus leans toward farmed salmon being safe and healthy, but with some important caveats — the type of salmon species, the time of year it was harvested the diet it was fed can all affect the nutrient content wild salmon remains the higher-credibility choice for vitamin and mineral content

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The debate is further complicated by the recognition that multiculturalism has multiple forms — demographic, philosophical governmental — and that its relationship to civic unity is therefore not uniform

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The decision to wear a knee brace should always be based on individual factors such as the type and severity of the injury, the sport or activity being participated in the extent of rehabilitation completed

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: The conflicting assessments reflect fundamentally different weightings of the evidence, with the EPA relying on a larger dataset that includes studies submitted to support registration research groups pointing to a more limited subset of studies that show stronger links to adverse health outcomes

### Sample conflictingqa_a25014a5c5b5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: Some plants can survive without light for extended periods, while others cannot. The Royal Horticultural Society notes that plants need light to grow and survive, but that some species can tolerate low light or artificial light

### Sample conflictingqa_a994724a28e7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The retrieved evidence is divided on whether cold water makes hair shinier. Some sources argue that cold water does make hair appear shinier by sealing the hair cuticle, while others argue that the effect is negligible or that cold water can even be harmful to hair growth

### Sample conflictingqa_a9bed39d234d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The retrieved evidence is divided on this question. Some sources argue that certain foods can provide fewer calories than it takes to digest them, while others argue that no such foods exist

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Yes, Michael Jackson did compose music for Sonic the Hedgehog 3; the game's original soundtrack was largely written by Jackson, with the music later adapted from a 1993 prototype for the Sonic Origins re-release

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: The evidence suggests that coffee grounds may provide some benefit, but likely only at higher concentrations of caffeine than most garden applications would achieve that other methods (such as copper barriers or diatomaceous earth) may be more effective

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3, d4
- **Supporting Docs Found**: None
- **Claim**: In the United States, a 1991 Gallup poll cited by the Oral Cancer Foundation found that Americans almost never think about death except occasionally Arnold Toynbee described it as an 'un-American' topic , while the broader Western world has historically shied away from discussing it openly

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: The retrieved evidence is divided on whether werewolves can be created by a full moon. Some sources argue that the full moon is a common trigger for transformations, while others argue that the idea is a modern invention of cinematic storytelling with no basis in folklore or scientific fact

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2, d4
- **Claim**: The evidence is divided on whether barefoot running is healthier than running with shoes. Some research suggests that barefoot running is associated with fewer injuries and improved running efficiency, while other studies found that shoes may actually reduce the load on foot muscles and provide a beneficial spring-like effect

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: The two views are thus in direct conflict, with the former grounded in the scientific consensus and the latter in religious and philosophical opposition to evolutionary theory

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: The USGS notes that while there have been reports of animals behaving strangely before earthquakes since ancient times, there is no consistent, reliable evidence that animals can predict earthquakes

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, the retrieved evidence does not exclude the possibility that other Europeans, such as the Portuguese, British Spanish, may have encountered the continent independently the Dutch themselves abandoned the idea of colonization after the failure of Abel Tasman's 1644 voyage

### Sample conflictingqa_f97fef94decc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d2, d1
- **Supporting Docs Found**: d3
- **Claim**: However, some individual users have reported vision problems after extended VR use manufacturers typically recommend taking regular breaks and following the 20-20-20 rule to mitigate eye strain

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, the more recent and widely supported view

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_0436c0b3a9d7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: Aryna Sabalenka and Amanda Anisimova were the finalists in the US Open women's singles last year

### Sample freshqa_047057d22309

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
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: August 16, 1977

### Sample freshqa_122352ad92e3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: This year's Passover (2026) began at sundown on April 1 and ends at nightfall on April 9

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy won the most recent Grammy Award for Best Jazz Performance, taking the award for "Twinkle Twinkle Little Me" at the 67th Annual Grammy Awards in 2025. This is confirmed by the official Grammy website, which lists her as the 2025 winner alongside Sullivan Fortner. Earlier winners include Chick Corea, Christian McBride Brian Blade, who won the 2026 award for "Windows - Live," but that result is from the most recent ceremony held, making it the latest available information

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that the conflict is ongoing, so these figures are subject to continued escalation

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

### Sample freshqa_4e635a2542a8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or, making the first year he won it "None"

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This was confirmed by multiple sources, with the film beating out notable nominees including *Sinners* (16 nominations) and *Oppenheimer* (2024)

### Sample freshqa_7bc7bb2dde20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d4
- **Claim**: Earlier winners include *Parasite* (2020), *Nomadland* (2021), *CODA* (2022) *Everything Everywhere All at Once* (2023), reflecting a consistent rotation of recent winners

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
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

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This is corroborated by the fact that they missed the playoffs in the 2023–24 season, which is consistent with a losing record

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Queen Elizabeth II of England died on 8 September 2022, at Balmoral Castle in Scotland, at the age of 96

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Jeff Bezos sold Amazon shares in June and July 2025, with the most recent reported sale valued at approximately $737 million

### Sample freshqa_c479e83e408f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Kylian Mbappé scored 15 goals in the UEFA Champions League last season

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d2, d1
- **Supporting Docs Found**: d5
- **Claim**: The Komodo dragon (Varanus komodoensis) is also frequently cited as the largest reptile, but it is actually the largest lizard species, not the overall heaviest

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: No permanent cure for cancer has been developed; cancer remains an ongoing area of active research

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: Ta-Nehisi Coates.

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Earlier vaccine versions had been authorized for all children 6 months to 17 years, but the FDA narrowed the indication for the updated vaccines to only those with at least one high-risk health condition, such as asthma or obesity, effectively making 6 months the youngest eligible age group

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Ramadan 2026 is expected to begin at sundown on Tuesday, February 17 end at sundown on Wednesday, March 18

### Sample freshqa_fd00b29e848c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: This is based on the first sighting of the crescent Moon over Mecca, Saudi Arabia, which is the standard reference point for beginning the month

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: 1864

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8
- **Supporting Docs Found**: None
- **Claim**: Stanford University is a private research university located in Stanford, California, adjacent to Palo Alto and between San Jose and San Francisco

### Sample hotpotqa_0070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0073

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d9, d2, d6
- **Supporting Docs Found**: d4
- **Claim**: Multiple other publications with 'El Nuevo' in their name are unrelated to Time Inc., including El Nuevo Cojo Ilustrado , El Nuevo Heraldo , El Nuevo Herald El Nuevo Diario

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

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d7, d8, d6, d4, d5, d2
- **Claim**: 506

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_2f6d2647a424

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Cesar Geronimo played third base for the 1975 Cincinnati Reds

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: The exact origin of crossing your fingers for good luck is uncertain, but the most widely supported theory is that it comes from pre-Christian European practices involving hand gestures and finger positioning as magical symbols, where the intersection of fingers was believed to anchor wishes and protect against evil. A second theory holds that the gesture originated in early Christianity, when believers would cross their index and middle fingers to form an X as a secret symbol of recognition and protection, a practice also linked to the Christian cross and the Ichthys (fish) symbol

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Peyer's patches

### Sample qacc_51b23ea15977

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The U.S. would not launch its first human astronaut, Alan Shepard, until May 5, 1961, when he made a suborbital flight aboard Mercury-Redstone 3

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Canada's transition to full independence from Great Britain was a gradual process spanning several centuries rather than a single date. The country was first established as the Dominion of Canada on July 1, 1867, when the British North America Act united the provinces of Nova Scotia, New Brunswick the Province of Canada into a single entity

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9404250d756f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a4dde83c35da

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3, d1
- **Claim**: The Ming dynasty was an imperial, autocratic government with a centralized bureaucracy

### Sample qacc_a6b48b7accc4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_a6df0af8c2ba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: Hosanna is a Hebrew expression meaning'save us' or 'help us,' used as an urgent cry for rescue or salvation is frequently recorded in the Bible as a shout of praise and recognition of divine power

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: This is consistent across multiple sources, with the Manual on Uniform Traffic Control Devices (MUTCD) confirming that yellow horizontal alignment signs with speed advisories are not regulatory

### Sample qacc_ac5341df9a83

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Security Council also recognizes the importance of involving troop-contributing countries in the planning and mandate review phases of missions, further formalizing this arrangement

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: In the United States, *Celebrity Big Brother* is broadcast on CBS

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: New Mexico was admitted to the Union as the 47th state

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The blaze was brought under control by approximately 10:30 pm the Christmas party continued in another area of the house

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: The music for Disney's 1973 animated *Robin Hood* was composed by George Bruns, with songs by Roger Miller and Floyd Huddleston. Elton Hayes also composed the music for the 1952 live-action version, *The Story of Robin Hood and His Merrie Men*

### Sample qacc_cb5bcdb1ef9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The character was also portrayed by Pernell Roberts in the M*A*S*H spin-off TV series Trapper John, M.D., but that is a separate entity from the original film and TV show

### Sample qacc_d8b24beb2f90

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The balance sheet (also known as the statement of financial position or statement of financial condition) is the primary financial statement that involves all aspects of the accounting equation

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_ecd3d9c0ca11

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Vernon Wells

### Sample qacc_ff2cb00f4c03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: 7

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: In Ontario, red license plates are primarily used by motor vehicle dealers and diplomats, with dealer plates having a white background and red lettering and diplomatic plates having a red background and white lettering. In Spain, red license plates are for vehicles in circulation during registration processing, temporarily out of service used for research and tests. In general, red license plates indicate that a vehicle is part of a fleet or is being used for a specific purpose, rather than being a standard registration plate

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3, d4
- **Supporting Docs Found**: d2
- **Claim**: For non-commercial vehicles, most states allow 16- or 17-year-olds to obtain a restricted license with supervised driving requirements, while a standard, full license is typically issued at 18

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The answer depends on how'sea' is defined the record according to Guinness World Records is the Eurasian pole of inaccessibility in northwestern China, while the furthest point from the nearest tidal water in the UK is Coton in the Elms, Derbyshire

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag, also known as the Bear Flag, features a grizzly bear

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: In response, a Constitutional Convention was held in Philadelphia in 1787, where the United States Constitution was drafted and signed, eventually replacing the Articles of Confederation as the framework of government

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d1
- **Supporting Docs Found**: None
- **Claim**: The FOMC consists of twelve members — seven from the Board of Governors and five presidents from the Federal Reserve Banks — and it meets regularly to decide on interest rates and open market operations

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d1
- **Supporting Docs Found**: None
- **Claim**: Historically, the federal government has been the primary driver of U.S. environmental policy, with landmark legislation such as NEPA and the EPA established under the Nixon administration , but more recent developments suggest a growing role for state and local governments in shaping the country's environmental agenda

### Sample situatedqa_temp_05d714be23fd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Additionally, Ludacris will receive the iHeartRadio Landmark Award Miley Cyrus will receive the iHeartRadio Innovator Award

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d5, d2, d4
- **Claim**: Wrangell-St. Elias National Park was established in 1980

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Todd Monken

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Springfield (41), Franklin (35), Greenville (31), Bristol (29) Clinton (29) round out the top five most common city/town names, further corroborating that Washington leads by a wide margin

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2
- **Claim**: These figures are all consistent with each other, with the most up-to-date reading coming from the Bureau of Economic Analysis for the first quarter of 2026

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Jagat Prakash Nadda

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d20
- **Supporting Docs Found**: d5, d4, d3, d1
- **Claim**: Mohamed Salah

### Sample situatedqa_temp_8808c106a115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: Tay-Sachs is an autosomal recessive genetic disorder

### Sample situatedqa_temp_901be1437bc7

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
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Shay Mitchell, the actress who plays Emily Fields, was 23 years old when the show first aired in 2010 and is now 36 years old, making Emily approximately 31–32 years old in the character's timeline

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: He also holds the WBC title, making him the undisputed heavyweight champion

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The WBO title was previously held by Daniel Dubois, who defeated Fabio Wardley on May 9, 2026, but Usyk's overall record across all four belts confirms his status as the current champion

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: He was the first openly LGBTQ+ person to receive the honor, following a rich history of 40 years that previously included icons like Brad Pitt, George Clooney Johnny Depp

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Stephen Curry

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: This supersedes older information from 2021 that referenced him as the Director of National Intelligence, as the CIA role is the most recent appointment

### Sample wikirevision_0007

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Alphabet Inc. owns Google, the search engine and technology company. Google is a wholly owned subsidiary of Alphabet Inc., which was confirmed when Google reorganized under Alphabet's umbrella in 2015. This relationship is further corroborated by additional context showing that Alphabet acquired Wiz, Inc. in March 2026, solidifying its ownership of Google

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The 2027 ICC Men's Cricket World Cup is scheduled to be the next edition, the 14th will be hosted in South Africa, Zimbabwe Namibia in October and November 2027

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of Germany is Frank-Walter Steinmeier, who has been in office since 19 March 2017. He is the Federal President of the Federal Republic of Germany, serving as the country's head of state, with Bellevue Palace in Berlin as his official residence. This is consistent across multiple sources, including the current Wikipedia revision of the President of Germany article, which confirms his incumbency from 19 March 2017

### Sample wikirevision_0134

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current Prime Minister of Australia is Anthony Albanese, who has served in office since 23 May 2022. He is the 31st person to hold the position since the office was created in 1901

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 Wimbledon Championships are the next edition, scheduled to take place from 29 June to 12 July 2026, but the results of that edition have not been documented in the retrieved evidence

### Sample wikirevision_0151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: The 2027 ICC Men's Cricket World Cup is scheduled for South Africa, Zimbabwe Namibia in October and November 2027, but that edition has not yet been played

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of India is Droupadi Murmu, who is the 15th President of India and took office in 2022. She is the first person from the Adivasi community to hold the office and the second woman to be elected as President


================================================================================

*Report generated by CATS v2.0*
