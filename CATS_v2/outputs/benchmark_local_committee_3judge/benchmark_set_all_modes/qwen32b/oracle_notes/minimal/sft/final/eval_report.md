# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 123 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.971 (over 736 samples)

**GR F1** *(used in CATS)*: 0.983

**Behavior Adherence**: 0.811 (over 613 applicable samples)

**Factual Grounding**: 0.878 (over 613 applicable samples)

**Single-Truth Recall**: 0.761 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.858

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.983
- **Precision**: 0.992
- **Recall**: 0.974
- **Accuracy**: 0.971
- TP=592, FP=5, FN=16, TN=123

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.885
- **Abstain Recall**: 0.961
- **Abstain F1**: 0.921
- **Specificity**: 0.974
- Abstain TP=123, FP=16, FN=5, TN=592


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (54 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.986
- **GR F1** *(used in CATS)*: 0.990
- **Behavior**: 0.943 (n=157)
- **Grounding**: 0.947 (n=157)
- **Recall**: 0.841 (n=154)
- **CATS**: 0.930

### Type 2: Complementary Info

- **Samples**: 221 (43 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.955
- **GR F1** *(used in CATS)*: 0.971
- **Behavior**: 0.921 (n=178)
- **Grounding**: 0.818 (n=178)
- **Recall**: 0.699 (n=156)
- **CATS**: 0.852

### Type 3: Conflicting Opinions

- **Samples**: 109 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.982
- **GR F1** *(used in CATS)*: 0.989
- **Behavior**: 0.521 (n=96)
- **Grounding**: 0.870 (n=96)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.793

### Type 4: Outdated Info

- **Samples**: 158 (13 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.717 (n=145)
- **Grounding**: 0.922 (n=145)
- **Recall**: 0.764 (n=140)
- **CATS**: 0.849

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 0.892
- **GR F1** *(used in CATS)*: 0.943
- **Behavior**: 0.838 (n=37)
- **Grounding**: 0.730 (n=37)
- **Recall**: 0.676 (n=37)
- **CATS**: 0.797


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 1885

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

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: Yes — palm oil production causes deforestation, habitat destruction, biodiversity loss substantial greenhouse gas emissions; Wikipedia confirms these harmful environmental effects coincide with palm oil cultivation on formerly forested and biodiverse lands

### Sample conflictingqa_21f33954c8af

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Dog breeding is not universally considered unethical, but some practices are

### Sample conflictingqa_2500bfd10608

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d2, d1
- **Claim**: Because risks appear to rise with dose and some studies capture effects at or near typical public water levels, the ongoing scientific debate challenges the universal safety conclusion

### Sample conflictingqa_264e97231363

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Not directly from chlorine — chlorine actually lightens hair, but copper (from algaecide or tap water) oxidizes and binds to hair proteins, causing the green discoloration

### Sample conflictingqa_3afd7f725cb4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: Ultimately, the consensus leans toward equality—especially in educational and literary contexts where relevant authorities broadly recognize audiobooks as valid reading —but individual perspectives vary significantly based on personal definition and experience

### Sample conflictingqa_3bd13d25098b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: d3
- **Claim**: While much of this activity is slow and subtle compared to Earth's dynamic geology, it is real and continues to be investigated by planetary scientists

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: It depends on whether the artificial tree is used for at least 20 years; otherwise, real trees are more sustainable

### Sample conflictingqa_52181cd092aa

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d4, d3
- **Claim**: No, the trash island (aka the Great Pacific Garbage Patch) is much larger than Texas — nearly three times as big according to one source

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: The retrieved evidence presents competing views. Some sources argue that software patents are legally challenging, practically limited to recordable media or technical processes that automating known methods is generally not patentable. Others argue that software patents do apply in many jurisdictions, that 62% of U.S. patents involve software that patents remain strategically valuable for protecting core algorithms and functions despite legal limits. The answer depends on jurisdiction, the specific nature of the software invention evolving legal standards

### Sample conflictingqa_734859937e46

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: It depends on jurisdiction; in the U.S. federal law generally allows ISPs to sell anonymized browsing history without consent, but some states like Maine and California require opt-in or opt-out consent other states are considering similar laws

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: Bees generally avoid flying in the rain because wet wings reduce lift and maneuverability, but they are capable of flying in light rain and will do so when circumstances require

### Sample conflictingqa_76956c2fba7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d2
- **Claim**: On the other hand, a systematic review and meta-analysis of observational studies, published in the British Journal of Sports Medicine, found no association between saturated fat consumption and all-cause mortality a review on Examine.com notes that RCTs and observational studies have not consistently reported strong associations between saturated fat intake and heart disease outcomes — highlighting that the evidence is not unanimous and that the degree of risk remains contested

### Sample conflictingqa_80baf25496cd

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Multiculturalism is not inherently a hindrance to unity; rather, the evidence suggests it can facilitate political and civic integration while preserving cultural diversity

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Bird calls are not entirely unique to each individual bird — rather, birds of the same species typically share calls while still producing recognizable individual variations

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, as the BBC and Wired notes, some historians acknowledge that a limited degree of genuine, localized panic did occur, particularly among listeners who had tuned in late or were already on edge due to recent news events the overall phenomenon was sufficiently real to prompt academic study and debate

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: The retrieved evidence presents competing views. Some sources conclude that meteor showers pose no significant threat to Earth's surface or human life, as the vast majority of meteors burn up harmlessly in the atmosphere even larger chunks historically have not caused widespread destruction. Other sources argue that certain meteor streams may contain boulder-sized objects that could theoretically survive atmospheric entry and cause localized damage, while meteor showers also represent a serious operational threat to spacecraft in orbit, requiring active risk mitigation strategies

### Sample conflictingqa_b323dd4b5820

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3, d2, d1
- **Claim**: Current CO2 levels are not unprecedented in Earth's history

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: Experts across several studies consistently note that the most environmentally friendly choice is to refuse straws altogether when possible

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: It depends on the interpretation or version of the werewolf myth being used; there is no universally accepted scientific mechanism for werewolf creation by a full moon

### Sample conflictingqa_e93e708d49a4

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: While early-stage panels may require a payback period of one to three years before turning a net energy profit, the overall lifecycle return on energy invested is decidedly positive

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Yes — barefoot running is healthier

### Sample conflictingqa_f39c966c2ec1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence presents competing views. Some sources claim the curse dates to the first performance, while others argue the evidence does not support a unique trend of accidents specific to Macbeth

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: Emoji serve as a visual supplement to written language but do not constitute a distinct written language themselves

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: However, the evidence does not confirm that the Dutch were the sole or original discoverers of Australia, as prior indigenous occupation and potentially earlier unrecorded European contacts remain possible

### Sample conflictingqa_f8da23d84ecc

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1
- **Claim**: Once considered the same dinosaur, Apatosaurus and Brontosaurus were reclassified as distinct genera in a 2015 study

### Sample conflictingqa_fa98c00bd697

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: In recent years, instruments like the Event Horizon Telescope have captured direct images of black holes by observing the shadow they cast against the bright emission from surrounding plasma, making it possible to 'see' the outline of a black hole in a practical sense

### Sample conflictingqa_fbedb688b1d9

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3, d2
- **Claim**: Religious scholars and commentators present competing views on whether Mormons are Christian. Some argue that Mormons are Christian because they believe in Jesus Christ and claim to worship Him as God, while others argue that key Mormon doctrines contradict core Christian tenets — for example, that the Mormon concept of deification conflicts with the traditional Christian view of God's uncreated nature. Because the question is genuinely contested, there is no single authoritative answer that resolves the debate

### Sample freshqa_2130eea851fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: None
- **Claim**: These citations reflect his foundational contributions to deep learning, including the popularization of backpropagation, the development of the AlexNet architecture his receipt of the Turing Award and 2024 Nobel Prize in Physics

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

### Sample freshqa_6a45fadeb16b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: $130

### Sample freshqa_6f42c128eb6c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Harry Maguire has never won the Ballon d'Or

### Sample freshqa_8ab63ffc9a7d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d3
- **Claim**: George R. R. Martin, the author of A Game of Thrones, was born in Bayonne, New Jersey

### Sample freshqa_97f3c1fe1fd4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Boating accident

### Sample freshqa_b99c189f2222

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3, d2
- **Claim**: Jeff Bezos did not sell Amazon. He is the founder and executive chairman, having stepped down as CEO in 2021 has since sold millions of shares worth billions of dollars while retaining over 900 million shares

### Sample freshqa_c4976f8629cb

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: The saltwater crocodile (Crocodylus porosus) is generally considered the heaviest living reptile, though some sources differ

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

### Sample qacc_0bd7153f19ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d2
- **Claim**: The retrieved evidence indicates the Allies subsequently moved eastward into Tunisia

### Sample qacc_2243f17ccc38

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: 1968

### Sample qacc_5eeb46719843

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: Canada's independence from Great Britain was a gradual process rather than a single date, but the most commonly cited milestone is July 1, 1867 — when the Dominion of Canada was formed by the British North America Act, uniting Ontario, Quebec, Nova Scotia New Brunswick

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: This is why Canada Day is celebrated on July 1st, commemorating the country's formation as a self-governing dominion within the British Empire

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_7df263780268

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3
- **Claim**: October 1968

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_9fbf28f5786f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: Earlier documents reference related milestones such as the 1993 announcement of the ISS project and the station's design phase between 1984 and 1993 , but these predate the actual launch of hardware into space

### Sample qacc_b0346f60b6ab

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5
- **Claim**: Celebrity Big Brother is currently available on CBS in the USA

### Sample qacc_bf0bff050f03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_c2975d69d57c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: Seth MacFarlane

### Sample qacc_ce4983c8a9c5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d1
- **Claim**: Most effigy mounds were built between approximately 700 and 1200 A.D., with the majority constructed within the Late Woodland period (roughly A.D. 750–1050)

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: The aircraft was named after Enola Gay Tibbets, the mother of the mission's pilot, Colonel Paul Tibbets

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: More specifically, it is often described as type SBbc, indicating a intermediate-to-late barred spiral with a moderate-sized central bar and open spiral arms

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d3
- **Claim**: XXXTENTACION

### Sample qacc_f1776add7672

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d1
- **Claim**: ICD-10 codes generally consist of three to seven characters

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Nassau County, NY

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
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: This shift was formally completed in 1931 with the inauguration of New Delhi as the new capital

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Additionally, cashew, coffee mango are among other commercially important tree crops recognized across broader contexts

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_geo_f2031e426cee

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d1
- **Claim**: The present Law Minister of India is Kiren Rijiju, who serves as the Cabinet Minister for Law and Parliamentary Affairs at the central government level

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d5
- **Claim**: The FOMC meets regularly to decide on policies regarding the money supply and interest rates, with its decisions having significant implications for inflation, employment overall economic stability

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: In the United States, environmental policy is set at multiple levels of government: federal, state often locally

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3, d2
- **Claim**: 1980

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The five sharps are F#, C#, G#, D# A# the key can be identified using the circle of fifths or the rule that the major key is a half step above the last sharp in the signature

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d4, d5
- **Claim**: Todd Monken

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: On naval ships, SS most commonly stands for "submersible ship," used in hull classification symbols like SSN for nuclear submarines

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: September 1967

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d2
- **Claim**: Carnie Wilson, Wendy Wilson Chynna Phillips

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Cardiac biomarkers are substances that enter the bloodstream when the heart is damaged or stressed they are used to diagnose and monitor heart conditions

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: HMS Queen Elizabeth was commissioned on December 7, 2017 formally declared operational in 2020

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4, d5, d3, d2
- **Claim**: Rhys Ifans

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1
- **Supporting Docs Found**: None
- **Claim**: This naming decision was rooted in the city founders' desire to curry favor with the British crown, reflecting a broader colonial practice of the era

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

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: The latest Cricket World Cup champion is Australia, who defeated India by six wickets in the 2023 final held on 19 November at the Narendra Modi Stadium in Ahmedabad. This was Australia's sixth Cricket World Cup title the 2023 tournament was the 13th edition of the ICC Men's Cricket World Cup, hosted entirely in India from 5 October to 19 November 2023. Wikipedia's main Cricket World Cup page also lists Australia (AUS) as the champions with a 6th title, though an older revision of that page had incorrectly identified India as the 2023 winner

### Sample wikirevision_0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This change was confirmed when the Gurgaon Municipal Corporation officially changed the city's name to Gurugram the change is consistently reflected across all standard references. As a result, while the city was once known as Gurgaon, the official name going forward is Gurugram

### Sample wikirevision_0157

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3, d2
- **Claim**: The current President of India is Droupadi Murmu, who has held office since July 2022. This is confirmed by the official Wikipedia revision that superseded the older version in February 2025, which explicitly names her as President with a 2022 official portrait. Her presidency is further contextualized by the broader list of all Indian presidents, which notes she is the 15th to hold the office since its establishment in 1950

### Sample wikirevision_0161

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Gurgaon is officially called Gurugram. This name change was confirmed by the Government of Haryana in 2016 the city is now officially known by this new name. As a result, Gurgaon is no longer the official name of the city


================================================================================

*Report generated by CATS v2.0*
