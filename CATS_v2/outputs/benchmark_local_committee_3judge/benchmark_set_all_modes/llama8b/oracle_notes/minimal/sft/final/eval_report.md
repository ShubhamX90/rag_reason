# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 123 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.961 (over 736 samples)

**GR F1** *(used in CATS)*: 0.976

**Behavior Adherence**: 0.788 (over 613 applicable samples)

**Factual Grounding**: 0.869 (over 613 applicable samples)

**Single-Truth Recall**: 0.718 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.837

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.976
- **Precision**: 0.992
- **Recall**: 0.961
- **Accuracy**: 0.961
- TP=584, FP=5, FN=24, TN=123

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.837
- **Abstain Recall**: 0.961
- **Abstain F1**: 0.895
- **Specificity**: 0.961
- Abstain TP=123, FP=24, FN=5, TN=584


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (56 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.991
- **GR F1** *(used in CATS)*: 0.994
- **Behavior**: 0.948 (n=155)
- **Grounding**: 0.946 (n=155)
- **Recall**: 0.844 (n=154)
- **CATS**: 0.933

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.923
- **GR F1** *(used in CATS)*: 0.950
- **Behavior**: 0.893 (n=178)
- **Grounding**: 0.792 (n=178)
- **Recall**: 0.587 (n=156)
- **CATS**: 0.805

### Type 3: Conflicting Opinions

- **Samples**: 109 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.963
- **GR F1** *(used in CATS)*: 0.979
- **Behavior**: 0.510 (n=98)
- **Grounding**: 0.835 (n=98)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.775

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.994
- **GR F1** *(used in CATS)*: 0.997
- **Behavior**: 0.703 (n=145)
- **Grounding**: 0.931 (n=145)
- **Recall**: 0.757 (n=140)
- **CATS**: 0.847

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.865
- **GR F1** *(used in CATS)*: 0.928
- **Behavior**: 0.676 (n=37)
- **Grounding**: 0.757 (n=37)
- **Recall**: 0.595 (n=37)
- **CATS**: 0.739


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2013

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
- **Supporting Docs Found**: d2, d5, d4
- **Claim**: Weight lifting causes a temporary increase in blood pressure during the actual lifting action — a response that is considered normal for most healthy individuals — but the long-term effects of regular strength training are generally positive the rise can be mitigated by factors such as proper technique, avoiding Valsalva maneuvers combining with other lifestyle changes

### Sample conflictingqa_111bef268aa6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Peeling an apple reduces some of its nutritional value by removing dietary fiber and certain vitamins, but not all nutrients are lost

### Sample conflictingqa_11c5ef7c4545

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The evidence does not support a universal yes or no answer

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: Yes, palm oil is bad for the environment; it causes deforestation, biodiversity loss, habitat destruction, pollution greenhouse gas emissions

### Sample conflictingqa_2395695f1604

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: Overall, the weight of high-credibility evidence suggests dairy does not increase mucus production, but methodological complexity and conflicting findings preclude a definitive, general answer

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: Fluoride in drinking water is considered largely safe at concentrations of 0.7 mg/L or lower, but high levels are linked to risks including fluorosis, skeletal damage potential neurotoxicity

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: These methodological and philosophical differences create conflicting research outcomes regarding what can be known beyond the mind

### Sample conflictingqa_288cd1b45aab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4
- **Claim**: The evidence does not support a clear general conclusion that wrist rests minimize wrist pain during typing for everyone

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: The conflict is methodological: d4 uses experimental research with C. elegans to demonstrate inheritance, while d5 cites evolutionary reasoning and DNA methylation biology to argue that inheritance has not been proven, with d2 offering partial support from reprogramming mechanisms

### Sample conflictingqa_2c0ea18839df

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The answer is therefore conditional rather than definitive

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Additionally, the Moon's interior is not thought to host a liquid outer core capable of driving a dynamo or volcanic activity like Earth's, which some consider a defining criterion for geological activity

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Third, real trees absorb CO2 while growing and emit it only when burned, resulting in negligible net emissions; artificial trees, on the other hand, release embedded carbon emissions over their full lifecycle

### Sample conflictingqa_411445406724

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: No, cycads did not dominate the Mesozoic era plant kingdom. According to paleobotanists, the Mesozoic is more accurately described as the 'age of dinosaurs' because flowering plants eventually replaced cycads as ecologically dominant species on land more than 100 million years ago

### Sample conflictingqa_4317242e485c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3, d4
- **Claim**: The evidence is mixed. Some sources argue that trophy hunting can benefit conservation by generating revenue, controlling wildlife populations funding anti-poaching efforts, while others argue it is morally inappropriate and that bans are not harmful but actually beneficial to conservation

### Sample conflictingqa_4786f87b62be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d4, d1
- **Claim**: The evidence does not support a definitive, universal answer; rather, it presents multiple plausible but conflicting perspectives on the same issue

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: CLS Bank decision and its progeny have placed significant limits on the patentability of software-based inventions that excluding software from patent protection may hamper technical development

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The Kidney Disease Improving Global Outcomes (KDIGO) guidelines recommend bicarbonate supplementation only when serum bicarbonate is below 18 mEq/L, further complicating a general answer

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4
- **Claim**: Overall, the evidence does not establish bicarbonate supplementation as a routine preventive measure for all stages of CKD

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: They make up roughly ten percent of the colony's population and have no corbiculae or scopae on their legs, abdomen thorax, meaning they do not deliberately collect pollen like female worker bees do

### Sample conflictingqa_676188bf8139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The Chinese Lantern Festival is related to honoring ancestors but not exclusively about celebrating deceased ancestors; it is also about marking the first full moon of the new lunar year and promoting reconciliation, peace forgiveness

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: The Catholic Church claims to be the "One True Church" founded by Jesus Christ, but this claim is not explicitly supported by Scripture and is contested by Protestant denominations

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d4
- **Claim**: Overall, while some level of individuality is likely present in bird calls, the evidence suggests that calls are not universally unique to each individual bird are shaped by a complex interplay of learning, anatomy species-level factors

### Sample conflictingqa_9261438d6ee2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: A 2010 book by Victoria Braithwaite and a 2014 paper by Rose and colleagues provide some of the key points on both sides of this debate, with Braithwaite arguing that fish brain structures are virtually identical to those of humans for pain detection, while Rose argues that fish perception of pain is very different from that of humans and must be distinguished from mere nociception

### Sample conflictingqa_9ceca2645833

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Affirmative action can be argued to constitute reverse discrimination in two main ways: first, by favoring one group (blacks, Hispanics, women) over another (whites, men) on the basis of race or sex, which is the core definitional criterion of reverse discrimination; second, by creating a system in which the very same practices that were once used to exclude minorities are now used to include them, thereby perpetuating the prior discrimination in a new form

### Sample conflictingqa_a864ff85e648

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The evidence is mixed. Some clinical studies reported benefits such as improved body composition, skin thickness cognitive function, while others found no clear evidence of long-term benefits and noted serious risks

### Sample conflictingqa_bd2e652cd64d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: The Cleveland Clinic notes that electric toothbrushes produce thousands of strokes per minute versus your own hand, making them more reliable for effective plaque removal , while also noting they are more expensive and require charging

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d2, d3
- **Supporting Docs Found**: None
- **Claim**: One source notes that 100 grams of yeast biomass contains nearly 47 grams of protein, which is almost 100% of the recommended daily intake for adults nutritional yeast is also recognized for being high in B12 and other B vitamins, making it particularly valuable for vegans who cannot obtain these nutrients from animal products

### Sample conflictingqa_cc71318e5853

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d3, d4
- **Claim**: Overall, the evidence suggests that death remains a sensitive and often avoided subject in modern society, with no single definitive consensus on whether it is still a taboo

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: The retrieved evidence is mixed. Some sources (RSC, JCOPERAHOUSE) affirm the curse originated from a witch coven objecting to Shakespeare's use of real spells in the first performance, while others (Scribd/Statistical analysis, Regina Jeffers' blog) challenge the curse's validity by noting it is folklore and that Macbeth does not statistically experience more mishaps than other Shakespearean plays

### Sample conflictingqa_f4811561af0c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The retrieved evidence supports both Dutch and British claims of first discovery, with the Dutch being the first Europeans to sight the Australian coast and the British being the first to establish a colony

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: This is consistent across multiple sources: one ranks it third with 600 million+ total speakers , another places it third with 542 million native speakers (implying a much larger total when including second-language speakers) a third confirms it as third with 380 million native speakers but notes the total could be much higher

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: On its ranking; the divergence is only in the absolute count, with higher figures reflecting inclusion of second-language speakers

### Sample freshqa_02b3ba89ebd0

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
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_3227ea6c6056

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: The latest major.NET version depends on which branch of.NET is meant:.NET Framework 4.8.1 is the latest for the Framework branch, while.NET 6.0 and.NET 7.0 represent the latest major versions for.NET 5+

### Sample freshqa_3847b5cb9b42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The conflict began in 2022 and is ongoing, with Russia struggling to achieve victory on the battlefield

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_50f8f03fd30f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: She traveled in third-class with her parents and older brother her father died in the disaster

### Sample freshqa_5d6e5db69928

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: This finding superseded the previous record of ~1 million-year-old DNA from a mammoth tooth, making it the oldest DNA ever recovered from an environmental sample rather than a fossil specimen

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This film won six Oscars total, including Best Director and Best Adapted Screenplay, marking Anderson's first Academy trophy

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a47283064972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Their most recent winning season was 2019–20, when they won 53 games

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The 2023–24 season was a losing season the Raptors missed the playoffs

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Jeff Bezos sold Amazon shares in late June and July 2025, but the documents do not state he sold the entire company

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

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: 12

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_fd00b29e848c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0155

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

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: This victory is also confirmed by the club's broader history, which notes that the 1971-72 season was the year they won the FA Cup under the management of Don Revie

### Sample qacc_3c1297608017

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: Historians note that the practice was originally a two-person gesture where one person placed their index finger over the other's it wasn't until the Hundred Years' War that the modern solo version of crossing the index and middle fingers became popularized

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This date is consistent across multiple sources, including the official Parliament of Canada website and the Royal Gazette

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: October 1968

### Sample qacc_8dd2323c077d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: The Airdrome

### Sample qacc_a3c882e062c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: None
- **Claim**: The basilica is considered nearly complete as of 2026, with only the last two towers of the Glory Façade remaining to be finished

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: The high water content in cells is also reflected in specific organ water percentages: the brain and heart are composed of 73% water, muscles and kidneys are 79% even bones are 31% water, further illustrating the widespread distribution of water throughout the body's cellular structures

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_b198a514fff8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: New Mexico was admitted to the Union as the 47th state

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
- **Supporting Docs Found**: d2, d4
- **Claim**: 1996 (Japan), 1999 (USA)

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: The Milky Way Galaxy is classified as a barred spiral (SBc) galaxy according to the Hubble classification system

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_e064a7a717ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Many of the interior scenes were built as sets or constructed by the production team

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 18

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is mixed and does not support a definitive answer

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: 1963 (for the original 1963 Clean Air Act); 1970 (for the Clean Air Act of 1970); 1990 (for the 1990 amendments); 2011 (for EPA regulation of greenhouse gases under the Clean Air Act)

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: The California state flag features a grizzly bear, making it the Grizzly Bear flag. The grizzly bear is a subspecies of the brown bear (Ursus arctos californicus) California is the only U.S. state to have an extinct animal on its official flag

### Sample situatedqa_geo_c76a815bb275

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
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: By the 20th century, coffee had become the dominant beverage in the United States , though tea continued to thrive in specific regional cultures like the American South

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, federal policy is not the only level — state governments also play a significant role in environmental protection, as demonstrated by the Inflation Reduction Act of 2022, which provided tax incentives for individuals to adopt energy-efficient measures the U.S. EPA itself, which works closely with state governments to implement federal policies and develop additional state-level measures

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: Additionally, local governments and tribes are increasingly involved in environmental decision-making through programs such as the Environmental Quality Incentives Program (EQIP), which provides financial and technical support to farmers to reduce emissions

### Sample situatedqa_temp_1baff64de20e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: While her initial contract was for a single episode, there is a possibility of her recurring later in the season

### Sample situatedqa_temp_3df0e6082901

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Florida Gators (2016)

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

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Beowulf contains kennings for both Grendel and the sea

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3
- **Claim**: 23,000,000

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: HMS Queen Elizabeth (R08) was commissioned in 2017 and formally declared operational in 2020

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The ship is the lead vessel of the Queen Elizabeth class and conducted her maiden operational deployment in 2021

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3, d4
- **Claim**: Rhys Ifans

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

### Sample trust_align_035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is partially supportive and does not justify a confident answer

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence is insufficient to support a complete answer

### Sample wikirevision_0032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: He is the 25th and current holder of the office, succeeding François Hollande and being reelected after a presidential campaign in which he defeated his far-right rival Marine Le Pen

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: This was their third World Cup title, their first since 1986 their first won outside of Europe since 2002

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia won the 2023 ICC Men's Cricket World Cup, their sixth Cricket World Cup title

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This is explicitly recognized in the infobox metadata of the Wikipedia article on Gurgaon, which states its official name as Gurugram. The city's official name was changed from Gurgaon to Gurugram this change is consistently reflected across multiple sources including the official Wikipedia infobox and the disambiguation page for Gurgaon

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: This victory is confirmed across multiple sources, with the 2026 French Open representing the current year's edition of the tournament


================================================================================

*Report generated by CATS v2.0*
