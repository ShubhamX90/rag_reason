# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 99 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.948 (over 736 samples)

**GR F1** *(used in CATS)*: 0.969

**Behavior Adherence**: 0.801 (over 637 applicable samples)

**Factual Grounding**: 0.885 (over 637 applicable samples)

**Single-Truth Recall**: 0.739 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.849

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.969
- **Precision**: 0.954
- **Recall**: 0.985
- **Accuracy**: 0.948
- TP=599, FP=29, FN=9, TN=99

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.917
- **Abstain Recall**: 0.773
- **Abstain F1**: 0.839
- **Specificity**: 0.985
- Abstain TP=99, FP=9, FN=29, TN=599


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (45 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.943
- **GR F1** *(used in CATS)*: 0.963
- **Behavior**: 0.898 (n=166)
- **Grounding**: 0.910 (n=166)
- **Recall**: 0.821 (n=154)
- **CATS**: 0.898

### Type 2: Complementary Info

- **Samples**: 221 (33 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.937
- **GR F1** *(used in CATS)*: 0.961
- **Behavior**: 0.936 (n=188)
- **Grounding**: 0.850 (n=188)
- **Recall**: 0.692 (n=156)
- **CATS**: 0.860

### Type 3: Conflicting Opinions

- **Samples**: 109 (9 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.954
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.620 (n=100)
- **Grounding**: 0.870 (n=100)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.821

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.664 (n=146)
- **Grounding**: 0.939 (n=146)
- **Recall**: 0.732 (n=140)
- **CATS**: 0.833

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.838
- **GR F1** *(used in CATS)*: 0.912
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.784 (n=37)
- **Recall**: 0.622 (n=37)
- **CATS**: 0.755


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2165

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
- **Cited Docs**: d3, d2, d4
- **Supporting Docs Found**: None
- **Claim**: The mechanism involves improvements in vascular function, lower resting heart rate, reduced arterial stiffness healthier body composition — all of which contribute to better blood pressure control over time, even though blood pressure may rise temporarily during individual exercises

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While some legal scholars and commentators argue that the ruling itself was a narrow exception rather than a broad categorical declaration , the poem's eventual acceptance into mainstream literary canon is directly attributable to this historic legal vindication

### Sample conflictingqa_0dba017da71c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3, d2
- **Claim**: Multiple authoritative sources confirm that while there are other large fungi and tree colonies that could qualify as runners-up, Armillaria ostoyae holds the top title as the largest single living organism on Earth

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The question of whether dog breeding is unethical is genuinely contested, with strong arguments on both sides. Proponents of the status quo note that ethical breeding regulated through standards bodies can produce healthy, well-socialized dogs that significantly enhance human life, while opponents argue that even responsible breeding exploits dogs for profit, perpetuates overpopulation causes avoidable health problems

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d1
- **Claim**: However, research has raised serious concerns about neurodevelopmental effects — a landmark 2018 study found that exposure to fluoridated water during pregnancy was associated with reduced IQ in children by approximately 0.5 points per part per million of fluoride NIH's toxicology program similarly reported that higher fluoride levels are linked to lowered IQ in children — with some scientists calling for better dose targeting

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d1
- **Claim**: Wrist rests are commonly recommended to reduce wrist pain during typing by encouraging a neutral position and minimizing strain on muscles and tendons

### Sample conflictingqa_35491baf4f4b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d4
- **Claim**: Over time, the Moon's weak gravity and continuous solar wind erosion have gradually stripped away these early atmospheric layers, leaving behind only the faint exosphere observed today

### Sample conflictingqa_39fe5c441657

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beliefs differ depending on who you ask

### Sample conflictingqa_411445406724

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Therefore, while cycads were widespread during the Mesozoic and are frequently depicted as the dominant plants in museum dioramas, they were not ecologically the most prevalent group, with flowering plants eventually replacing them as the dominant terrestrial vegetation more than 100 million years ago

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Opposing views contend that even ethically conducted hunting causes severe psychological harm to participants and the animals that a ban would not necessarily lead to widespread habitat destruction — especially if funding gaps are addressed through alternative conservation models

### Sample conflictingqa_52181cd092aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: However, it is important to note that the patch resembles more of a 'plastic soup' than a solid island, with plastic density varying widely from hotspots containing hundreds of kilograms per square kilometer to much less dense areas

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The question of whether patents should apply to software is actively debated, with strong arguments on both sides. The debate is far from resolved: recent US Supreme Court rulings have raised the standard for patenting software that implements known business methods the Alice v

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: CLS Bank decision has created significant uncertainty around patent eligibility

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The core debate hinges on whether the observed correlation between moon phases and large earthquake magnitudes represents a causal mechanism or merely coincidental pattern recognition

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: False — the Gutenberg Bible was not the first book printed with movable type; the oldest surviving example is the Jikji printed in Korea in 1377, predating Gutenberg's Bible by 78 years

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: High-dose supplements (up to 2,000 mg per day) are generally considered safe for most adults, but individuals with certain underlying conditions such as hemochromatosis or kidney stones should exercise caution

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Bees can fly in the rain, but their ability to do so is limited by the intensity of the rain, the wetness of their wings reducing lift and making navigation more difficult. Some sources indicate that bees are capable of flying in light to moderate rain when driven by strong hive needs, such as defending the colony or collecting nectar, while heavier downpours pose a more serious challenge with large raindrops capable of damaging a bee's wings

### Sample conflictingqa_a2f06d54b240

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Yes — a stalactite was found in the Blue Hole of Lighthouse Reef Atoll that formed ca

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The traditional narrative holds that Orson Welles's 1938 War of the Worlds radio broadcast caused widespread panic across the United States, with newspapers at the time describing readers as "saved only by the timely intervention of friends or neighbors"

### Sample conflictingqa_a3980a2921cf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: However, experts caution that the effectiveness depends heavily on the type of oil chosen: coconut oil is ideal for dry or damaged hair, while grapeseed or jojoba oil is better for fine hair heavier oils like coconut or castor are更适合的说是“视发质而定”而非普遍适用于所有发质。

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, the PETM onset also coincided with a mercury low, suggesting at least one additional carbon reservoir was released in response to initial warming multiple studies propose competing mechanisms such as methane release from ocean sediments or organic carbon feedbacks

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2
- **Claim**: In summary, while nutritional yeast can contribute meaningfully to a vegan protein intake, it should be part of a broader dietary mix to ensure completeness

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: Some gardeners have reported success using 1–2 tablespoons of used grounds per square foot, scattered around affected plants, though commercial slug pellets are generally regarded as a more effective and established control measure

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d4
- **Claim**: Belief in the full moon creating werewolves is largely a modern cinematic invention rather than a factual historical claim

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, experts note that yields are only one dimension of farming performance, as organic systems can offer environmental benefits such as improved soil health and reduced chemical runoff that conventional farming may lack

### Sample conflictingqa_ece626a6cba9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: Further complicating the picture, some scholars point to the sudden, almost instantaneous deaths described by contemporary observers and the low mortality rates among carriers of familial Mediterranean fever (FMF), suggesting the disease may have included components of pneumonic plague or other viral infections like Ebola-like viruses

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d1
- **Claim**: Folklore states that Macbeth was cursed from the beginning because witches objected to Shakespeare using real incantations its first performance around 1606 was reportedly riddled with disaster — including the death of the actor playing Lady Macbeth — though the RSC notes that the play has been performed successfully many times since

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The evidence is mixed

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: Yerba Mate is traditionally prepared by steeping dried leaves of the Ilex paraguariensis plant in hot water it is commonly consumed in South America and beyond for its stimulating and antioxidant properties

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: None
- **Claim**: PAHs (polycyclic aromatic hydrocarbons), known carcinogens also found in grilled meat and tobacco smoke, are present in yerba mate some research suggests that combining yerba mate with tobacco or alcohol may further amplify the risk of cancer development

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Importantly, though, some studies have also shown that yerba mate exhibits a cytotoxic effect on cancer cells in laboratory settings, suggesting it may possess inherent anti-cancer properties that have not yet been confirmed through clinical research

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Concert-goers demonstrated an unwavering spirit of community and mutual support despite mud, rain scarce resources, with Max Yasgur noting that 'half a million kids can come together for three days of fun and music and have nothing but fun and music'

### Sample conflictingqa_fad0d30903d2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: d2
- **Claim**: The event has since become a powerful symbol of peace, love unity the original lineup featured legends like The Who, Jefferson Airplane Jimi Hendrix who closed the festival at 8am on Monday

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_4d9a80505e01

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The Federal Reserve did not cut interest rates from August to December 2022; it raised the federal funds rate significantly during that period

### Sample freshqa_64c12116affc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: His election was validated by a landslide win in the 2024 Electoral College, making him the oldest person ever elected to the presidency

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has not won the Ballon d'Or

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: An earlier unconfirmed claim (also referenced by Wikipedia) suggests that two monkeys (Able and Baker) may have been the first animals in space, but no definitive evidence supports this

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d4
- **Claim**: Bayonne, New Jersey

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: This untimely death dealt a significant blow to the fledgling field of neural networks, as Rosenblatt was actively defending the potential of perceptrons against critics like Minsky and Papert

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Their playoff performance that season was also noted, as they missed the postseason

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: While older data exists showing the Raptors had winning records in the 2019–20 (53–19) and 2020–21 (27–45) seasons a 56–26 record in the 2015–16 season , the most recent available evidence from the 2023–24 season indicates they did not achieve a winning record in the latest NBA season

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The discrepancy reflects evolving data: the original record holder *Pirates of the Caribbean: On Stranger Tides* was once credited with a $378–379 million budget, but as more detailed UK tax documents surfaced, analysts revised that figure upward, pushing *Rise of Skywalker* past it

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: No permanent cure for cancer has been developed; however, significant milestones in achieving complete remission have been documented

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_ef3ad40c6540

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: LeBron James plays for the Los Angeles Clippers

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: These successive discoveries reflect the ongoing use of advanced technology to reveal the Nazca Pampa's hidden treasures mainstream archaeologists continue to propose scientifically grounded theories about their origins and meanings

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample healthcontradict_0145

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Overall, while yoga may be helpful as an adjunct to established treatments, it remains an area of ongoing research patients should consult their healthcare providers before making any changes to their asthma management regimen

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d9, d1
- **Claim**: Their self-titled debut album, simply called Lit, was released in 1995, though their most enduring hit came later with "My Own Worst Enemy" reaching number 51 on the Billboard Hot 100 and winning Modern Rock Track of the Year at the 1999 Billboard Music Awards

### Sample hotpotqa_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d1
- **Claim**: A Place in the Sun was indeed recorded in 1999, not 1995 as the query suggests

### Sample hotpotqa_0196

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d2, d4, d7
- **Supporting Docs Found**: d3
- **Claim**: This map is a valuable historical record of the town's layout in the early 17th century Speed's work as a whole is particularly associated with the Stuart period

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d8, d6, d2, d4, d7
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The Allies went on to invade Sicily in July–August 1943, followed by Italy in September 1943

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Princess of Wales Theatre

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: Their victory came in the 1971–72 season, when they defeated Arsenal 1–0 at Wembley Stadium, with Allan "Sniffer" Clarke scoring the winning goal

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: As a coach: Red Auerbach (16)

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: The USSR's dominance was further underscored by the fact that Gagarin's flight was confirmed before the end of April, leaving little doubt about who held the record at that time

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6edf1477bd7e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Steve McEwan

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: October 1968

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This distribution is consistent across the general population, with the USGS further noting that fat tissue retains less water than lean tissue, causing slight sex-based variations in body water percentage

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: **Celebrity Big Brother** is streamed in the USA on **Paramount+**, with older seasons also available on **Netflix**

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: The West Wing, which housed the President's office and several staff offices, sustained extensive damage, with the fire cutting off the main floor offices from the attic storage above

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

### Sample qacc_c675e6cd8ad6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d4
- **Claim**: George Bruns (for the 1973 animated version — see Disney+ soundtrack); Roger Miller (for the 1952 live-action version — see Legacy Collection soundtrack); Floyd Huddleston (for the 1972 musical — see IMDB soundtrack listing)

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Japan: 1996; Japan: 1996

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: XXXTENTENTACION

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: 3–7

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: 16

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2, d4
- **Claim**: The retrieved evidence indicates that World War II was fought on multiple fronts, with the Eastern Front being the most significant

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: This system ensures that no single branch accrues too much power through the mechanism of checks and balances is further defined by the Bill of Rights, which enumerates fundamental citizen freedoms

### Sample situatedqa_geo_85af31651715

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Other UK nations followed suit in subsequent years: Wales in April 2007 , Northern Ireland in 2007 Ireland enacted a similar ban in 2004

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: This legislation established the Environmental Protection Agency (EPA) and gave the federal government authority to set and enforce standards to control air pollution, replacing earlier state-level regulations

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d2
- **Claim**: Both Kennedy and his predecessor Lyndon B. Johnson sent advisers to South Vietnam, though Kennedy's program was the most significant early effort, eventually growing to 23,000 by 1964

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4, d1
- **Supporting Docs Found**: d5
- **Claim**: This framework created a weak central government serving as a war-time confederation of states, but it was superseded by the adoption of the U.S. Constitution in 1787, which established the current federal republican system

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3, d2, d1
- **Supporting Docs Found**: None
- **Claim**: While the exact date of 1865 is commonly cited , some sources suggest the shift was more gradual, driven by both health awareness and personal preference rather than any single defining moment

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_0ad0871fcc33

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Kobe Bryant's 81-point game (January 2006) ranks second Wilt Chamberlain's previous mark of 78 points (1961) rounds out the top-five list

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Earlier in the same calendar year, the Eagles also claimed their first-ever NFL Championship in the 1981 season, defeating the Oakland Raiders in Super Bowl XV

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d2
- **Supporting Docs Found**: None
- **Claim**: Earlier records from 2017 and 2018 further contextualize the series history, showing a more recent stretch of dominance for Queensland

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2
- **Claim**: Novak Djokovic (men) / Margaret Court (women)

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Mort is a Goodman's mouse lemur (also called a pygmy lemur or toyotama), though a spin-off series reveals his ancestry also includes components of a bear, spider starfish — making him technically a bear — and his DNA further incorporates elements of spiders, starfish sawdust. This makes Mort one of the most genetically diverse characters in the Madagascar franchise, alongside the film's other iconic animals like Alex the lion, Melman the giraffe King Julien the lemur

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: He had previously served as a senior puisne judge of the SHC and was confirmed as a permanent judge on 27 August 2015

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: An example of a kenning from Beowulf describing Grendel is “twilight-spoiler,” which is used in the battle with Grendel. This kenning highlights the creature’s evil nature and its emergence at night, reflecting the battle’s dark and violent context

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: David Harbour plays Hopper in Orange is the New Black

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: September 1967

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While NASA's Artemis II successfully circled the moon in 2023, carrying astronauts around but not landing them the Artemis III mission is scheduled to land astronauts there in 2025 , none of these have yet matched the record set by Apollo 17

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Wilson Phillips is an American vocal trio consisting of Carnie Wilson, Wendy Wilson Chynna Phillips, the daughters of Beach Boy Brian Wilson

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: Formed in Los Angeles in 1989, the group initially gained fame with their self-titled debut album released in 1990, which featured hit singles such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This figure was confirmed when Erton Köhler was elected the new president of the General Conference in 2025 , representing the denomination's continued growth over previous estimates

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: While the broader movement involved numerous factions and leaders such as Huang Xing and Zhang Taiyan, Sun Yat-sen remains the preeminent figure in historical accounts

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This ranking is corroborated by the fact that the 2018 GPI report covers 163 independent states and territories , indicating India was among the less peaceful nations that year

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The surname is commonly found in regions where Germanic and Romance languages are spoken it was first recorded in the Domesday Book of 1086, tracing its lineage to the son of Gerard

### Sample situatedqa_temp_d15849658c20

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Oleksandr Usyk (WBA Super, WBO, IBF, IBO)

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: August 20, 1989

### Sample trust_align_008

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Jessica Lange is a member of the cast of *American Horror Story* (Season 2)

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: In contrast, the liver possesses remarkable regenerative capacity: if up to half of a healthy liver is donated, the remaining portion can fully recover and grow back within a year, a process facilitated by the proliferation of surviving hepatocytes

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: The key biological difference lies in the mechanism: alcohol-induced scarring progresses through inflammatory cycles that eventually overwhelm the liver's repair systems, whereas surgical donation triggers a one-time massive growth response that the liver's cells can manage when isolated from ongoing harmful inputs

### Sample trust_align_154

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: India has not hosted the Commonwealth Games; India participated in the 2002 Commonwealth Games in Manchester

### Sample trust_align_193

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Blue cheese is generally safe to eat with mould because it is made from unpasteurized milk and contains active moulds that inhibit the growth of harmful bacteria like listeria

### Sample trust_align_193

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Unlike other soft mould-ripened cheeses such as brie and camembert, blue cheese's distinctive curing process creates an environment where the beneficial moulds dominate, making it less susceptible to contamination

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed when the city changed its name officially in 2016 the Wikipedia infobox reflects this current status with "official_name = Gurugram." Earlier revisions of the Wikipedia article referenced 'Gurgaon' as the city's name, but these have been superseded by the newer 2026 revision which makes 'Gurugram' the explicit official name

### Sample wikirevision_0123

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Carlos Alcaraz (2025 US Open) — see the 2025 US Open page for match results and complete coverage

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This was confirmed when the city officially changed its name in 2016 the Wikipedia infobox reflects this current designation. As a metropolis in Haryana, India, Gurugram is recognized as the official name moving forward


================================================================================

*Report generated by CATS v2.0*
