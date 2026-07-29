# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 120 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.961 (over 736 samples)

**GR F1** *(used in CATS)*: 0.976

**Behavior Adherence**: 0.802 (over 616 applicable samples)

**Factual Grounding**: 0.830 (over 616 applicable samples)

**Single-Truth Recall**: 0.686 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.823

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.976
- **Precision**: 0.987
- **Recall**: 0.965
- **Accuracy**: 0.961
- TP=587, FP=8, FN=21, TN=120

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.851
- **Abstain Recall**: 0.938
- **Abstain F1**: 0.892
- **Specificity**: 0.965
- Abstain TP=120, FP=21, FN=8, TN=587


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (52 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.957
- **GR F1** *(used in CATS)*: 0.971
- **Behavior**: 0.899 (n=159)
- **Grounding**: 0.867 (n=159)
- **Recall**: 0.756 (n=154)
- **CATS**: 0.873

### Type 2: Complementary Info

- **Samples**: 221 (44 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.959
- **GR F1** *(used in CATS)*: 0.974
- **Behavior**: 0.949 (n=177)
- **Grounding**: 0.805 (n=177)
- **Recall**: 0.593 (n=156)
- **CATS**: 0.830

### Type 3: Conflicting Opinions

- **Samples**: 109 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.972
- **GR F1** *(used in CATS)*: 0.984
- **Behavior**: 0.619 (n=97)
- **Grounding**: 0.887 (n=97)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.830

### Type 4: Outdated Info

- **Samples**: 158 (12 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.975
- **GR F1** *(used in CATS)*: 0.986
- **Behavior**: 0.678 (n=146)
- **Grounding**: 0.806 (n=146)
- **Recall**: 0.743 (n=140)
- **CATS**: 0.803

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.892
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.649 (n=37)
- **Grounding**: 0.730 (n=37)
- **Recall**: 0.568 (n=37)
- **CATS**: 0.722


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2078

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
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: It is generally considered safe and well-tolerated for mild conditions, but can interact with other medications and may carry risks for individuals with bipolar disorder or certain other conditions

### Sample conflictingqa_114c06976f62

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: The religion's own definition further complicates the question, as it rejects dogma and does not require belief in the Flying Spaghetti Monster, making it distinct from traditional faiths

### Sample conflictingqa_220ec09fbb2c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Technically, no — cows have one stomach, but it is divided into four distinct compartments: the rumen, the reticulum, the omasum the abomasum

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: We cannot know anything beyond our minds

### Sample conflictingqa_37ab7146eb9b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: Robots cannot literally feel pain because pain is an internal, subjective human experience tied to consciousness current robotics does not replicate that internal experience

### Sample conflictingqa_37ebad668bb7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d1
- **Claim**: However, experts note that the relationship between data volume and performance is not one-size-fits-all — it depends heavily on the specific problem, model architecture desired application

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Some researchers argue that better algorithms can sometimes compensate for limited data that the most critical factor is often the relevance and representativeness of the features themselves rather than pure data volume

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: The retrieved evidence is conflicting. Some sources argue that audiobooks are every bit as legitimate as physical books, with studies showing the brain processes written and auditory narratives similarly. Others argue that audiobooks are fundamentally different and do not count as real reading

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, primarily because they are grown like agricultural crops and absorb carbon dioxide during their lifetime; one hectare can absorb approximately 2 tonnes of CO2 annually

### Sample conflictingqa_3f3c3399259a

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, large clinical trials such as VITAL and JELLO have found that fish oil does not significantly lower the risk of major cardiovascular events the American Heart Association concludes that fish oil supplements are not proven to prevent heart attack or stroke

### Sample conflictingqa_517b918aa677

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: However, some constitutional scholars and advocates argue that the First Amendment protects students' right to pray voluntarily at school the courts have also permitted Bible studies and devotional meetings by school personnel during appropriate times

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Internationally, approaches diverge — the European Patent Convention excludes 'computer programs as such' from patentability, while the U.S. Supreme Court's Alice Corp. v

### Sample conflictingqa_544ebeeccda5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2
- **Claim**: KDIGO guidelines recommend sodium bicarbonate orally to normalize blood bicarbonate levels when serum bicarbonate is less than 18 mEq/L, reflecting a nuanced position between definitive prevention and ongoing uncertainty

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: False — the Gutenberg Bible was not the first book printed with movable type; the oldest surviving example is the Jikji printed in Korea in 1377, predating Gutenberg's Bible by 78 years

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: However, many products can temporarily make split ends appear smoother by coating the hair shaft, adding weight to frayed ends creating temporary glue-like bonds between split fibers — though these effects usually disappear after one shampoo

### Sample conflictingqa_73560dfab1ae

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d1
- **Claim**: Importantly, the effectiveness of vitamin C at high doses remains uncertain, as the evidence is mixed and the optimal daily intake for cold management is debated

### Sample conflictingqa_894f4a4b9552

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Indirectly: scientists know dark matter exists because the observed gravitational effects of galaxies and cosmic structure are 85–90% unexplained by visible matter alone no alternative explanation has been found, though researchers actively pursue alternatives such as modified gravity models

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Prospective owners should be prepared to provide a warm, moist, ventilated tank with a secure lid to monitor temperature and humidity carefully

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: On the other side, some accounts hold that the broadcast did create a genuine public furor — described as a 'national event' by Orson Welles himself — with early reports citing suicides, heart attacks mass evacuations from cities like New York and Philadelphia a PBS documentary on the subject notes that the original press coverage was so intense that it became 'burned into our national consciousness'

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: However, the PETM onset also coincided with a mercury low, suggesting at least one other carbon reservoir was simultaneously activated , making the trigger mechanism a subject of ongoing scientific debate

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d1, d4
- **Claim**: Clinical guidelines similarly note that moderate green or low-oxalate herbal tea consumption is safe and potentially beneficial for individuals at risk of kidney stones, as increased fluid intake from tea can help reduce urinary saturation and lower stone risk

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: However, experts caution that the evidence is nuanced: some studies indicate an inverse relationship between tea consumption and kidney stone risk, while others highlight that patients with a personal history of stones should remain vigilant and opt for lower-oxalate varieties in moderation

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2, d4, d3
- **Claim**: Overall, while the risk is real and not entirely negligible, it remains far below the level of threat commonly associated with asteroids or comets

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: In informal contexts, 'alright' is widely accepted as a valid spelling of 'all right,' used by major dictionaries including the New Oxford American Dictionary

### Sample conflictingqa_b7fd50f9f980

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5, d1
- **Supporting Docs Found**: None
- **Claim**: It is particularly common in British English, where it has gained increasing acceptance over time is treated as a frequent spelling variant by the British Shorter Oxford Dictionary

### Sample conflictingqa_bac0f4d62f96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Some evidence suggests cometary meteoroids make up approximately 95% of observed meteors and 38% of observed fireballs, though no conclusive evidence links any specific meteorite to a particular comet scientists actively debate the topic

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The BBC's analysis further notes that even a well-made reusable straw must be used hundreds of times to outweigh its initial environmental costs, suggesting that neither material is unambiguously green

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: However, some nuance is warranted: while nutritional yeast is fortified with B vitamins and protein, it is worth noting that unfortified varieties contain only about 180% of the Daily Value for riboflavin per serving vegans should still strive for a varied plant-based diet to ensure all nutrient needs are met

### Sample conflictingqa_c3e66630928c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: A middle-ground perspective acknowledges that dry coffee grounds can act as a temporary deterrent, though their effectiveness is limited by factors such as moisture retention and caffeine concentration, making them less reliable than stronger alternatives like copper strips or commercial slug pellets

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This view is further complicated by foundationalist and coherentist epistemologies, which hold that beliefs can be justified without being true, as long as they cohere with other beliefs in a broader system

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Over the course of a year, solar panels typically do not consume more energy than they produce — contrary to a common myth — but they do not always produce a net surplus either

### Sample conflictingqa_ece626a6cba9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Ancient DNA evidence from 14th-century victims has strongly rebutted the Ebola virus theory , while research on familial Mediterranean fever carriers suggests plague immunity could explain why some groups died at lower rates than expected , adding nuance to the traditional view of bubonic plague as the sole cause

### Sample conflictingqa_f43b2c51deea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1, d4
- **Claim**: Experts remain divided: some point to coren's observations of dogs reacting to high-frequency vibrations as potentially promising, while others note that dogs also show increased anxiety without clear earthquake correlation experts broadly doubt whether any animal behavior can reliably predict quakes days or weeks in advance

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: However, some research offers a nuanced perspective: while large amounts of hot yerba mate may increase cancer risk, the active compound itself has also been shown to possess cytotoxic effects on cancer cells in laboratory settings, suggesting potential anti-cancer properties

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Religious affiliation is a matter of self-identification and belief Mormons do self-identify as Christian: the Church of Jesus Christ of Latter-day Saints' official website states that its members "unequivocally affirm[ ] themselves to be Christians," and Mormons participate in Christian ministerial alliances and sing songs about Jesus

### Sample freshqa_0436c0b3a9d7

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
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: August 16, 1977

### Sample freshqa_150b9ed24a07

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d4, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_1ef881d26e2e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d4
- **Claim**: Maryam Mirzakhani (2014 — first and only female recipient; passed away 2017)

### Sample freshqa_28e155139ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Android 17

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The awards were announced on November 17, 2021, with nominations announced on November 1, 2022 the ceremony consisted of two separate events — one for creative and technical arts and another for performances and programming — hosted by JoJo Siwa and Jack McBrayer respectively

### Sample freshqa_354e7097602d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4, d3
- **Claim**: The first atomic bomb test, called Trinity, took place at the Trinity Site in New Mexico on July 16, 1945

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: *The Mandalorian* has 3 released seasons

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: While filming for Season 4 has begun as of late 2024, no official confirmation exists regarding when it will be released, making the current total three seasons

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The season featured a unique format with three "all-star coaches" — Adam Levine, Kelly Clarkson John Legend — and Jayy's victory marked Adam Levine's fourth overall win as coach

### Sample freshqa_7bc92b47dc43

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by the official MLB timeline, which lists their 2022 World Series appearance as their eighth overall, with the Astros winning that series 4-2

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: Bayonne, New Jersey

### Sample freshqa_97f3c1fe1fd4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Eminem

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: This untimely death occurred when Rosenblatt was operating a boat near his home in Maryland the incident is well-documented across multiple sources including the Cornell Chronicle and academic publications

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d1, d4
- **Claim**: Queen Elizabeth II of England died on 8 September 2022 at Balmoral Castle in Aberdeenshire, Scotland

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: ALL EVIDENCE SUPPORTS THE SAME CONCLUSION: Jeff Bezos sold Amazon shares in late June/July 2025, with specific amounts reported across multiple documents

### Sample freshqa_c3f10dc1632d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Jiangsu

### Sample freshqa_c7315f8b3029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: OpenAI never released a GPT-5.5 model; the entire sequence is a misinformation / non-existent entity

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These discrepancies reflect the methodological ambiguity around what constitutes 'production cost' — whether original budget figures, inflation-adjusted totals final net expenditures — and the evolving nature of record-keeping in the film industry

### Sample freshqa_dd87e1e3ad3d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: 12

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: There is no permanent cure for cancer; however, researchers have developed several life-saving treatments such as chemotherapy, immunotherapy targeted therapies that can induce complete remissions

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_f6ac249bdf53

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: Ta-Nehisi Coates

### Sample freshqa_f6cc6071caa5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4
- **Claim**: The ongoing use of AI and drone technology continues to reveal more of these ancient desert drawings, making the total count a figure that has grown steadily over time

### Sample hotpotqa_0025

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d9
- **Supporting Docs Found**: d7, d5, d10
- **Claim**: At the time of his birth, Korea remained a Japanese colony the war's conclusion marked the formal dissolution of that era

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample hotpotqa_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d9
- **Supporting Docs Found**: None
- **Claim**: This is the most specific and directly relevant information we can provide based on the evidence

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d8, d7, d6, d5, d4
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not support a confident answer

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: The Allies went on to invade Sicily in July 1943, then turned to the Italian mainland — the boot and the heel — where they advanced northward, fighting the Winter Campaign of 1943–1944

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Toronto; at the Princess of of Wales Theatre (1989–2018)

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_3d4ebfa8b6dd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: As a coach: Red Auerbach (16)
As a player: Bill Russell (11)
Combined: Phil Jackson (11)

### Sample qacc_41c44ecfd0f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: Lacteals (also spelled lacteal lymphatics or lacteal vessels)

### Sample qacc_51b23ea15977

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d5
- **Claim**: While the USA had previously tested suborbital flights, including Alan Shepard's brief suborbital mission in early 1961, these did not match the USSR's orbital achievement the US would continue to feel pressure to surpass Gagarin's feat throughout the remainder of 1961

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d5
- **Claim**: Canada did not achieve full political independence from Great Britain on a single date; rather, it was a gradual process spanning several key milestones. The formal recognition of Canada's autonomy came with the Balfour Declaration of 1926, which acknowledged Canada as an autonomous member of the British Commonwealth — a significant step toward independence

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: October 1, 1968

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

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_aa94588b9477

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The original bank began operating in mid-1912, with the note issue administered by the Treasury

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Celebrity Big Brother is streamed in the USA on Paramount+, with older seasons also available on Pluto TV

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: For the most current USA broadcast information, check your local listings or streaming services

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: No one was injured in the blaze the following Christmas, White House staff and their children gathered again to celebrate the holidays, receiving toy fire trucks as gifts from the Hoovers

### Sample qacc_bc7e9a7b4a83

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Nico Rosberg

### Sample qacc_bf0bff050f03

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: New Zealand (as of the 2026 data in the retrieved evidence)

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: Notable surname bearers include Christopher Tavarez, an American actor Elisa Tavárez, a Puerto Rican pianist, reflecting the name's widespread use across the Spanish-speaking world

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: Genealogical research further traces the Tavarez surname to 13th century Portugal, when hereditary surnames were becoming standardized following the Norman Conquest

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2, d5
- **Claim**: Japan: 1996 (released in Japan); in the US, the first Base Set was released on January 9, 1999

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence does not support a confident answer

### Sample qacc_d96b47272030

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: October 11, 1887

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_e6d89fce1b8e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d1, d4
- **Claim**: Teddy Altman married Henry Burton on Grey's Anatomy

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: 2024/25

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Between three and seven

### Sample qacc_f69c37496013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: September 25, 1987 (limited); October 9, 1987 (wide release)

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: However, scholars and sources differ significantly on the exact numbers, with some estimates ranging as high as 80 million regional figures reflect the varying impacts of the conflict

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: For instance, the Soviet Union reported approximately 26.6 million fatalities — including about 8 to 9 million due to famine and disease — while Germany reported around 5.3 million military deaths Japan estimated its casualties at 3.1 million

### Sample situatedqa_geo_5f5c20228969

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: The retrieved evidence indicates that World War II featured multiple fronts across different regions, with the Eastern Front being the largest and most deadly

### Sample situatedqa_geo_66684169f016

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The participation of even women and children in the civil disobedience movement

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This is further confirmed by the official U.S. government website, which notes that all State governments are modeled after the Federal Government and must uphold a "republican form" of government, though the three-branch structure is not required

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: The California state flag features a grizzly bear (Ursus arctos californicus), which is also California's official state animal

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: New South Wales last won the State of Origin series in 2025 (Source: )

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Earlier records from 2021 and 2019 (Source: ) are superseded by the 2025 result, which is further confirmed by the BBC Sport report detailing the dramatic game-three decider in Sydney (Source: )

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_301378915064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Merritt Wever (Nurse Jackie)

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Florida, Arizona Oklahoma State have also won multiple titles, making the all-time leaderboard competitive across several programs

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d4
- **Claim**: Android 16 (released June 10, 2025)

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: 1980 (established as a national park); earlier designated as a national monument (1978)

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: It is worth noting that other prefixes such as USS (United States Ship), USNS (United States Naval Shipping) USRC (United States Revenue Cutter) were made obsolete in 1901, with USRC being replaced by USCGC when the Revenue Cutter Service merged with the U.S. Coast Guard in 1915

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Australia's mainland coastline is approximately 23,860 kilometres , which converts to about 14,800 miles

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3
- **Claim**: This figure is composed of $0.60 in state excise tax, $0.10 in state sales tax $0.18 in federal taxes, with environmental compliance costs adding an additional $0.54 per gallon as of March 2025

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: India's highest individual score in the 2018 South Africa Test series was 129 by Virat Kohli in the 6th ODI at Centurion Park, Johannesburg. The series as a whole saw India win by 113 runs in the 1st Test at St George's Park, Port Elizabeth

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: The group, formed in Los Angeles in 1989, quickly rose to fame with hits such as "Hold On," "Release Me," and "You're in Love," and is renowned for its rich harmonies and blend of pop, pop rock soft rock genres

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: This conflict, the first major military engagement between the early Islamic prophet Muhammad's followers and the pagan Quraysh tribe, is considered one of the most significant events in early Islamic history

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4
- **Claim**: Shay Mitchell was 23 when she portrayed 16-year-old Emily Fields in the show's pilot episode, which aired in 2010. Since then, Mitchell has aged naturally Emily's real-life age would reflect her character's progression through the series

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d3
- **Claim**: A Reddit comment estimated Emily at her mid-20s the IBTimes article noted that Ashley Benson (Hanna) is the only character close to her actual age, suggesting Emily is also in her early-to-mid-20s in real life

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: 670

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The 2024 data further shows that India's overall GPI score increased to 2.32, up from 2.31 in 2023, indicating sustained improvement in national security and stability

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Early records of the surname in England further contextualize its adoption following the Norman Conquest of 1066, where it first appeared in the Domesday Book of 1086 under the Latin form *Gerardus*

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Stephen Curry has been the highest-paid player for nine consecutive seasons, earning $156 million last year has also logged the second-most career minutes in NBA history behind only James

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d4, d3
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: This figure is corroborated by additional demographic details showing 100% US citizenship and 84 times more White (Non-Hispanic) residents than any other ethnic group

### Sample trust_align_038

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: Research further indicates that caffeinated and alcoholic beverages can act as diuretics, increasing water loss beyond what is replaced, making pure water especially important for maintaining balance

### Sample trust_align_064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Humans do not have a tapetum lucidum, the reflective layer found in many animals' eyes that causes them to glow in the dark

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
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: Google is owned by Alphabet Inc. — the company was reorganized as a wholly owned subsidiary of Alphabet in 2015 Page and Brin together hold approximately 14% of Google's publicly listed shares, exercising 56% of its stockholder voting power through super-voting stock. Google operates as a holding company for Alphabet's internet properties and interests, with Sundar Pichai serving as CEO

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The presidency has been directly elected since 2004

### Sample wikirevision_0096

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current Chief Justice of India is Surya Kant, who became the incumbent on 24 November 2025. He is the 53rd Chief Justice to serve since the Supreme Court of India was established in 1950

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia (won the 2023 Cricket World Cup, their sixth title)

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called **Gurugram** now. This is confirmed by the newer Wikipedia revision, which reflects a consensus to change the city name from Gurgaon to Gurugram — a change that was pending as of the older revision but is now fully implemented. As a metropolis in Haryana, India, Gurugram is recognized as the official name of the city this is corroborated by additional sources including the Rapid Metro Gurgaon entry, which continues to identify the city by its current name

### Sample wikirevision_0105

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Bangalore's official name was changed to Bengaluru. This is confirmed by the official Bangalore Municipal Corporation resolution, which stated that henceforth the city would be known officially as Bengaluru. As a result, Bengaluru is now the current official name of the city, while Bangalore is used as an alternative name

### Sample wikirevision_0109

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Mark Carney is the current Prime Minister of Canada, having assumed office on 14 March 2025. This is confirmed by the official Wikipedia revision that superseded the older version in March 2026, which also notes that he is the 24th Prime Minister of Canada. His inauguration took place at the Office of the Prime Minister and Privy Council building he is serving as the Right Honourable the Prime Minister of Canada

### Sample wikirevision_0120

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The men's singles champion is **Carlos Alcaraz**, who defended his title by defeating world No. 1 Jannik Sinner in the final. This result is corroborated by additional context indicating that Alcaraz was the two-time defending champion at the 2026 tournament but withdrew before the start of the men's draw, suggesting the 2025 title referenced in some sources has been superseded

### Sample wikirevision_0124

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The President of Germany is Frank-Walter Steinmeier, who has served as Herr Bundespräsident and holds Bellevue Palace as his official residence. This is confirmed by the current Wikipedia revision, which also notes that his term is 5 years and renewable once consecutively. While the article "List of presidents of Germany" provides additional historical context about the evolution of the German presidency, it does not alter the current fact that Frank-Walter Steinmeier is the incumbent holder of the office

### Sample wikirevision_0125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: He leads the Labor Party and has been a consistent figure in Australian politics for decades

### Sample wikirevision_0125

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2, d1
- **Claim**: Wikipedia's list of Prime Ministers of Australia confirms that his tenure spans from 23 May 2022 to the present, making him the latest incumbent

### Sample wikirevision_0135

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Jannik Sinner (most recent data available indicates he is the current champion; the 2026 Wimbledon Championships are scheduled but results are not yet confirmed)

### Sample wikirevision_0141

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1
- **Claim**: Jannik Sinner (most recent data available: 2026 Wikipedia revision)

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The Office of the Vice President further corroborates his tenure, noting that it is headed by the chief of staff to Vice President JD Vance

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: The current President of India is Droupadi Murmu, who became sworn in on 25 July 2022. She is the 15th and current President of India, serving as head of state with a term that extends until March 2027

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: Wikipedia's page on the President of the United States confirms his incumbency, noting that both the older and newer revisions agree on Trump's tenure

### Sample wikirevision_0172

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4
- **Claim**: The men's singles champion is **Carlos Alcaraz**, who successfully defended his title by defeating world No. 1 Jannik Sinner in the final. This result is corroborated by Wikipedia's page on the 2026 French Open, which notes that two-time defending champion Carlos Alcaraz withdrew before the tournament began due to a wrist injury, making his defense all the more notable


================================================================================

*Report generated by CATS v2.0*
