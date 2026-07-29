# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 101 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.955 (over 736 samples)

**GR F1** *(used in CATS)*: 0.973

**Behavior Adherence**: 0.791 (over 635 applicable samples)

**Factual Grounding**: 0.885 (over 635 applicable samples)

**Single-Truth Recall**: 0.737 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.847

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.973
- **Precision**: 0.957
- **Recall**: 0.990
- **Accuracy**: 0.955
- TP=602, FP=27, FN=6, TN=101

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.944
- **Abstain Recall**: 0.789
- **Abstain F1**: 0.860
- **Specificity**: 0.990
- Abstain TP=101, FP=6, FN=27, TN=602


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.938
- **GR F1** *(used in CATS)*: 0.960
- **Behavior**: 0.898 (n=167)
- **Grounding**: 0.924 (n=167)
- **Recall**: 0.792 (n=154)
- **CATS**: 0.894

### Type 2: Complementary Info

- **Samples**: 221 (37 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.955
- **GR F1** *(used in CATS)*: 0.972
- **Behavior**: 0.935 (n=184)
- **Grounding**: 0.854 (n=184)
- **Recall**: 0.692 (n=156)
- **CATS**: 0.863

### Type 3: Conflicting Opinions

- **Samples**: 109 (8 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.945
- **GR F1** *(used in CATS)*: 0.969
- **Behavior**: 0.653 (n=101)
- **Grounding**: 0.846 (n=101)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.823

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.616 (n=146)
- **Grounding**: 0.922 (n=146)
- **Recall**: 0.736 (n=140)
- **CATS**: 0.818

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.958
- **Behavior**: 0.649 (n=37)
- **Grounding**: 0.829 (n=37)
- **Recall**: 0.703 (n=37)
- **CATS**: 0.784


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2186

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
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The temporary spikes are attributed to factors such as the Valsalva maneuver or muscle effort, rather than the exercise itself are generally not dangerous for healthy individuals with normal blood pressure

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: However, the poem remained a subject of controversy a group of parents even objected to it being read in a Colorado high school class in 2019–2020

### Sample conflictingqa_0a05aabca56a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1, d4
- **Supporting Docs Found**: d5, d3
- **Claim**: The two terms are frequently treated as synonyms — for example, some sources describe anime as a 'Japanese cartoon genre' — but others emphasize that anime represents a deliberate departure from traditional cartoon standards, featuring limited animation techniques and a distinct artistic style

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Unlike animals or trees, fungi reproduce through genetic cloning, allowing individual organisms to grow to unprecedented sizes

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Research on blood pressure management further suggests that apple peel supplementation may elevate nitric oxide levels and potentially lower ACE activity, though daily consumption of two unpeeled apples did not significantly alter 24-hour blood pressure in humans

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Proponents of responsible breeding argue that it is necessary to maintain valuable working and service dog breeds that unethical practices such as backyard breeding and puppy mills are the true causes of dog suffering — not breeding itself. Opponents counter that even responsible breeding exploits dogs for profit, contributes to overpopulation perpetuates inherited health conditions; some argue it is unnecessary given that dogs are naturally social animals and many working roles can be performed by mixed-breed dogs. The debate thus centers on whether the benefits of maintaining specific breeds justify the harms caused by substandard practices what regulatory frameworks are needed to protect dogs

### Sample conflictingqa_220ec09fbb2c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some sources simply say "four stomachs" as a shorthand for all four compartments functioning together this is widely accepted in popular understanding

### Sample conflictingqa_237adb87065f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d2, d3
- **Claim**: This temporal conflict — with d1 and d5 suggesting plants first appeared in the Silurian and d2 noting both Silurian and Ordovician evidence, while d3 dates the earliest radiation to the Ordovician — reflects ongoing research and differing interpretations of the fossil record, making the answer to the query contested

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: Chlorine itself does not turn blonde hair green; rather, it swells the hair shaft and acts as a catalyst for the reaction between copper (from algaecides or tap water) and oxygen, which is what causes the green discoloration

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: We cannot know anything for certain beyond our minds

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d4
- **Claim**: However, some experts argue that wrist rests are not universally effective, as they can promote harmful 'planting' behavior where users press their wrists into the rest, potentially increasing pressure on the median nerve rather than reducing it

### Sample conflictingqa_37ab7146eb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: A Japanese research team called Affetto can detect pressure changes and react with facial expressions, but the authors emphasize that 'only life forms can actually suffer' and that the robot's responses are 'merely obeying commands from the central nervous system'

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beliefs differ depending on who you ask

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: However, the debate remains contested: some readers maintain that true reading requires active visual engagement with the text the question of whether audiobooks count as 'real' reading continues to be a matter of personal and scholarly disagreement

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, primarily because they absorb carbon dioxide while growing and can be recycled as mulch or wood chips, whereas artificial trees are made from plastic and metal and end up in landfills where they release stored carbon

### Sample conflictingqa_52181cd092aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, it is important to note that the patch resembles more of a 'plastic soup' — a dispersed concentration of particles — rather than a single solid island

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The question of whether patents should apply to software is genuinely contested, with strong arguments on both sides. Opponents argue that software is too abstract, that patents create harmful litigation that the Supreme Court's Alice v

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: CLS Bank decision introduced unworkable uncertainty

### Sample conflictingqa_62b1aff6586d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: While the overall trend is positive, scientists still cannot say definitively that the ozone layer has been fully healed

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d1, d3
- **Claim**: Bees can fly in the rain, but their ability to do so depends on the intensity of the rain, the current situation within the hive the individual bee's genetics. When rain is light, bees may continue to forage some species like bumblebees appear more tolerant of poor weather conditions than others. However, wet wings reduce a bee's ability to generate lift, making flight more challenging large raindrops can cause serious damage — even potentially breaking a bee's wing — which is why bees generally prefer to return to their hive during heavy downpours

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: However, organic farming practices have their own distinct advantages: they contribute far less emissions during production and are better for the environment in multiple ways, such as conserving land and reducing pollution

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d3
- **Claim**: Religious authorities differ; see also competing claims and analyses

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, some scientists argue that dark matter is not a confirmed necessity — that unpalatable hypothesis that 85% of the gravitational matter in the universe remains unexplained is not exclusive, as alternatives such as modified gravity are pursued actively

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: At the same time, researchers have documented that birds can distinguish between individual conspecifics through unique vocalizations: some species like White-crowned Sparrows vary the fundamental frequency of their songs others like Bengalese Finches alter spectral features, suggesting a capacity to encode individual identity in calls

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, these species represent exceptions rather than a general rule the dominant evidence across studies points to calls being more broadly species-specific or context-dependent than uniquely individual

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The governing body for speleology, the National Speleological Society, further clarifies that the typical formation mechanism of dripping water applies universally, whether above or below sea level, suggesting that underwater stalactites share the same fundamental process as their above-water counterparts

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The traditional view holds that Orson Welles's 1938 War of the Worlds radio broadcast caused widespread panic across the United States, with newspapers at the time reporting suicides, heart attacks mass hysteria

### Sample conflictingqa_a3264277980a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: Surveys conducted immediately after the program illustrate that very few people heard the broadcast virtually no one thought it was real — with the C.E. Hooper ratings service reporting only 2 percent of national respondents were tuned into Welles's broadcast on the evening of 30 October 1938

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: The PETM onset coincides with a mercury low, indicating at least one other carbon reservoir was released in response to initial warming, reflecting the ongoing scientific debate about the event's full causal mechanism

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Wikipedia's overview of brain size evolution further complicates the picture by highlighting that the relationship between brain size and intelligence is contested, with scientists from Stony Brook University and the Max Planck Institute demonstrating that the brain size-to-body size ratio has changed over time in response to various conditions — sometimes driven by factors unrelated to intelligence

### Sample conflictingqa_bd2e652cd64d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d2
- **Claim**: A 2019 study following nearly 3,000 patients found that those using sonic toothbrushes experienced 22% less gum recession and 18% less tooth decay over an 11-year period research confirms they are more effective at reducing signs of periodontal disease and tooth loss

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: A University of Nebraska study found that snails are deterred when substrate or plant leaves are sprayed with caffeine solutions of 0.1% or greater are even killed at concentrations exceeding 1%, while commercially brewed coffee solutions can achieve similar results

### Sample conflictingqa_c574530da7a3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Religious and scholarly views differ. Some Christian interpreters treat the Genesis account as a literal historical record, while others acknowledge it as an ancient literary narrative without contradicting the spiritual truths it conveys

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: However, some JTB advocates grant that justified belief can be false while denying that a premise must entail its conclusion, meaning non-deductive justification does not require the premise to be true

### Sample conflictingqa_e2e2361dc28b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d2, d4
- **Claim**: This creates a direct tension: if justification can occur for false beliefs (as in Gettier's examples), then a belief can be justified even if it is false, but if justification requires the belief to be true or refutable, false beliefs cannot be justified

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: PAHs (polycyclic aromatic hydrocarbons) are known carcinogens found in yerba mate studies suggest that combining mate consumption with tobacco or alcohol significantly increases cancer risk through synergistic metabolic reactions

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some research offers a contrasting perspective: lab experiments have shown yerba mate possesses a cytotoxic effect capable of killing cancer cells outright, though this does not establish it as a proven therapeutic agent

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: However, in contexts where clarity is critical — particularly with overlapping or ambiguous list items — the Oxford comma becomes indispensable

### Sample conflictingqa_f970957c5e52

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d2
- **Claim**: For example, consider the sentence "I applied to Cambridge, Oxford London schools"; the comma before 'and' ensures that these are three distinct institutions rather than a single entity

### Sample conflictingqa_f97fef94decc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, some research has also found that VR can benefit vision — under an eye doctor's guidance it can improve eye coordination, hand-eye coordination depth perception — and no serious deterioration of vision has been established among users of certain headsets

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: In rare cases, a black hole can even be directly imaged — such as the first-ever direct image of a black hole released in 2019 — showing a black hole surrounded by a orange donut-shaped ring of material

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Concert-goers stayed despite mud, rain scarce resources, proving they were there primarily for peace, love music

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: No, King Charles III has not yet stripped Prince Harry of his Duke of Sussex title as of October 2025

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris won the 2020 Formula 1 World Drivers' Championship, securing a clean sweep for his McLaren team. The season concluded at the Abu Dhabi Grand Prix, with Norris accumulating 423 points — just two points shy of Max Verstappen's four consecutive titles (2021–2024). This victory marked the first British driver to claim the championship since Lewis Hamilton in 2020, though Hamilton had actually won his seventh Drivers' title that same year, not the 2020 season

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
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_64c12116affc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Multiple congressional sources corroborate Trump's ongoing presidency, noting his tenure extends through the 117th and 118th Congresses

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_7bc92b47dc43

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Their total of two World Series wins makes them the only team to have won a postseason series in seven consecutive seasons they are the fifth expansion team to win two World Series championships

### Sample freshqa_7e63fcff2dea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The first animal to land on the moon was **two Russian tortoises** (Zond 5 mission in September 1968), according to Wikipedia's overview of animals in space

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These tortoises circled the Moon and returned safely to Earth, predating the Apollo 11 mission that carried humans

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The award ceremony took place on May 20, 2025, presenting the 58th annual Nebula Awards

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: No, the Toronto Raptors do not appear to have a winning record in the latest NBA season. The most recent data available — the 2023–24 season — shows the Raptors finished with a 25–57 record, which is not a winning record

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This is further contextualized by older records showing the team has struggled to achieve consistent success in recent years, with the 2022–23 season marking a period of transition following the trade of franchise icon DeMar DeRozan

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_c7ac9d61059a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Older sources may report different figures because they reference the pre-adjustment pricing, making them outdated relative to the current official 2026 Model Y pricing

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: The Allies went on to invade Italy (the boot and the surrounding islands), then France, after liberating North Africa

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Princess of Wales Theatre

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d4
- **Claim**: While the exact origin is not definitively known, historians propose these two competing theories as the most plausible explanations for the widespread adoption of the gesture

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: As a coach: Phil Jackson (11)
As a player: Bill Russell (11)

### Sample qacc_51b23ea15977

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The USSR's dominance in 1961 was further underscored by the fact that just five days after Gagarin's flight, President Kennedy convened a critical review of the U.S. space program, signaling the urgent need to close the growing gap

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d4, d5
- **Claim**: Jessica Hecht (Season 1); Alice Kremelberg (Season 2); Cora Tannetti (Season 3); Meg Muldoon (Season 4)

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: Steve McEwan

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: October 1, 1968

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The new season (actually the tenth and final season) premiered on 13 February 2024

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This distribution is consistent across the general population, with the USGS further noting that fat tissue retains less water than lean tissue, causing slight sex-based variations in body water percentage

### Sample qacc_aaf0f638e99b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2
- **Claim**: These signs are treated as suggested speeds in ideal driving conditions drivers can be ticketed for unsafe driving even if they are traveling below the advisory speed

### Sample qacc_bc34664caee4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1, d3
- **Claim**: Rice, California

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: XXXTENTENTACION

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d1, d4
- **Claim**: Teddy Altman married Henry Burton (Season 10–11; they divorced after his death)

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: 3–7

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: For those interested in moving there, the document also provides county-level details, including that Marion County is zoned to Lake Weir High School, Sumter County to Wildwood Middle High School Lake County to Leesburg High School

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: This rule applies nationwide anyone under 21 is prohibited from purchasing, possessing consuming alcohol

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: These figures include both military deaths and missing personnel, representing the number of Americans who died or went missing in action during the war

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_4cb699778b59

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Wikipedia further clarifies that approximately one-third of the Senate faces election or reelection every two years, reflecting the constitutional requirement for periodic turnover

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: The retrieved evidence indicates that World War II was fought across multiple fronts, with the Eastern Front being the largest and most lethal single theater

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: More broadly, Pew Research reports that Asians are projected to become the largest immigrant group in the U.S. by 2055, surpassing Hispanics, who currently make up about 31% of all immigrants

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: President Kennedy was the first U.S. president to send military advisers to South Vietnam, authorizing the deployment of 16,000 American troops to help stem a communist military invasion of the south

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: A grizzly bear

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, current active player Jalen Green is topping the current season's scoring leaderboard with 19.3 points per game through the 2025–26 season

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Mort is a Goodman's mouse lemur (also called a pygmy lemur), but in the spin-off series *All Hail King Julien*, it is revealed that his genetic makeup is approximately 40% bear, 20% spider, 20% starfish 20% other non-lemur elements — making him technically a bear. This is further corroborated by the fact that Mort is listed among animals found in jungles and forests of Madagascar, though the franchise also includes zoo animals and other creatures

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: While the 2017 NCAA records list only through 1986 data, the Wikipedia tournament overview and USA Today article reflect updated information showing UCLA's dominance through the 2019 season

### Sample situatedqa_temp_61d9095f5b70

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: For additional context, the song's title was inspired by a breakup between Jon Bon Jovi and Diane Lane, with the band writing it in just 90 minutes

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: These discrepancies reflect differences in measurement methodology, with the scientific study accounting for the fractal nature of coastlines and the dataset citing 2004 data, while the more commonly cited figures (e.g., 25,760 km) appear to represent simpler national estimates

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is further corroborated by the Ministry of Health and Family Welfare's official page, which lists his tenure alongside other ministers

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: David Harbour plays Hopper in Orange is the New Black

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The group, formed in Los Angeles in 1989, rose to fame with hits such as "Hold On," "Release Me," and "You're in Love," and is renowned for their rich harmonies and blend of pop, pop rock soft rock genres

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Shay Mitchell has been consistently in her 30s throughout the show's run, with the character aging through a five-year time jump in Season 6B

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2, d5
- **Claim**: The most commonly used and specific biomarker is cardiac troponin, which has the highest sensitivity and stays elevated for days after a heart attack, though it is not present in all heart conditions and can be raised in other situations such as skeletal muscle injury

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The United States has hosted the Olympics nine times: four Summer Games and five Winter Games

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: However, the ship was initially declared operational in 2020, following the completion of its sea trials and formal acceptance

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The most up-to-date information places the Queen Elizabeth's service entry at 2020, with ongoing operational deployments in the Indo-Pacific region

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d4
- **Supporting Docs Found**: None
- **Claim**: The surname was first recorded in the Domesday Book of 1086, originating with the Anglo-Saxon tribes of Britain is also found in Haiti

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: This naming tradition reflects the strong ties between North Carolina's early settlement and the British monarchy of the 18th century

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: Additional demographic details indicate that Pawleys Island is 100% composed of U.S. citizens, with 84 times more White (Non-Hispanic) residents than any other racial group

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Earlier in their history, the 76ers qualified for the playoffs in each of Julius Erving's 11 seasons in Philadelphia, including a memorable 1982–83 postseason where they lost only one game before being eliminated by the Los Angeles Lakers

### Sample trust_align_008

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Jessica Lange is a member of the cast of *American Horror Story* (Season 2), where she portrays Sister Jude

### Sample trust_align_026

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Meanwhile, Blue Origin's New Shepard program is also developing capabilities for Mars missions, with an estimated launch window of 2024–2028

### Sample trust_align_038

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: It is also worth noting that certain beverages like coffee and alcohol act as diuretics and can increase fluid loss, further complicating the relationship between natural sensation and adequate hydration

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Additionally, high concentrations can displace oxygen in the lungs and central nervous system, leading to suffocation as respiration is suppressed

### Sample trust_align_086

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: Complementary Information — Documents cover distinct but valid aspects (location, competition, services, taxes) that together form a complete picture of why gas prices differ between stations

### Sample trust_align_149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: However, iodine does not protect the rest of the body from other harmful effects of nuclear radiation, such as those caused by cesium or strontium isotopes, which is why complementary protective measures like potassium iodide are recommended alongside iodine supplementation

### Sample trust_align_193

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4, d3
- **Claim**: Blue cheese is generally safe to eat with mould because it is made from aged, often heavily salted sometimes unpasteurized milk, which creates an environment inhospitable to harmful bacteria like Listeria. In contrast, soft blue-veined cheeses such as Roquefort, Gorgonzola Danish blue are made with moulds that can harbor listeria, a bacterium capable of causing serious illness in pregnant women and immune-compromised individuals, which is why they are advised against. During pregnancy, blue cheese is considered riskier than other types of cheese because it is typically made from unpasteurized milk, increasing the likelihood of listeria contamination

### Sample wikirevision_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This ownership is further corroborated by Microsoft's 2025 annual report, which cites LinkedIn's revenue of US$17.8 billion

### Sample wikirevision_0028

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: This is confirmed by the newer Wikipedia revision, which supersedes an older 2026-02-16 revision that also named him but is marked as an older version

### Sample wikirevision_0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the newer Wikipedia revision, which superseded the older 2025-08-29 entry and consistently uses 'Gurugram' as the infobox official_name

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed when the city officially changed its name in 2016 the Wikipedia infobox reflects the updated official designation. As a result, Gurugram is the current official name of the city, while 'Gurgaon' is still commonly used

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: His election victory gave him the first Republican presidential win in what would become a record-setting eight years, though his tenure has been marked by ongoing legal challenges and political controversy

### Sample wikirevision_0170

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The men's singles champion is **Carlos Alcaraz**, who defended his title by defeating world No. 1 Jannik Sinner in the final. This victory marked his second French Open title and fifth major. However, it is worth noting that the 2026 French Open also saw Coco Gauff claiming the women's singles title, marking her first major at the tournament


================================================================================

*Report generated by CATS v2.0*
