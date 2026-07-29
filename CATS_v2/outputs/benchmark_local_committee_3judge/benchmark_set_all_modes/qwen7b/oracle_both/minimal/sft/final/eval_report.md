# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**Correct Refusals**: 87 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.940 (over 736 samples)

**GR F1** *(used in CATS)*: 0.965

**Behavior Adherence**: 0.790 (over 649 applicable samples)

**Factual Grounding**: 0.902 (over 649 applicable samples)

**Single-Truth Recall**: 0.742 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.850

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.965
- **Precision**: 0.937
- **Recall**: 0.995
- **Accuracy**: 0.940
- TP=605, FP=41, FN=3, TN=87

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.967
- **Abstain Recall**: 0.680
- **Abstain F1**: 0.798
- **Specificity**: 0.995
- Abstain TP=87, FP=3, FN=41, TN=605


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211 (42 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.929
- **GR F1** *(used in CATS)*: 0.954
- **Behavior**: 0.858 (n=169)
- **Grounding**: 0.929 (n=169)
- **Recall**: 0.821 (n=154)
- **CATS**: 0.890

### Type 2: Complementary Info

- **Samples**: 221 (28 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.919
- **GR F1** *(used in CATS)*: 0.951
- **Behavior**: 0.922 (n=193)
- **Grounding**: 0.883 (n=193)
- **Recall**: 0.699 (n=156)
- **CATS**: 0.864

### Type 3: Conflicting Opinions

- **Samples**: 109 (6 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.917
- **GR F1** *(used in CATS)*: 0.954
- **Behavior**: 0.650 (n=103)
- **Grounding**: 0.891 (n=103)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.832

### Type 4: Outdated Info

- **Samples**: 158 (11 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.987
- **GR F1** *(used in CATS)*: 0.993
- **Behavior**: 0.660 (n=147)
- **Grounding**: 0.905 (n=147)
- **Recall**: 0.714 (n=140)
- **CATS**: 0.818

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.703 (n=37)
- **Grounding**: 0.910 (n=37)
- **Recall**: 0.703 (n=37)
- **CATS**: 0.829


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 2299

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
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d4
- **Claim**: Handling a salamander is therefore not detrimental to the animal itself, but it can be detrimental to humans, as rigorous handwashing is strongly advised after contact to prevent numbness, dizziness, muscle weakness other symptoms

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: At the international level, the Berne Convention offers broad protection for literary and artistic works, which would include fashion designs, though member states interpret eligibility differently

### Sample conflictingqa_0ad05303220b

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d2, d5, d1, d4
- **Claim**: Judaism is generally categorized as a religion, but it also functions as an ethnicity or ancestral identity — a view explicitly endorsed by Chabad Lubavitch scholarship

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: However, a nuanced perspective is also important: palm oil is highly profitable and provides livelihoods for millions of small farmers in tropical regions, making the issue more complex than simply banning or replacing the product

### Sample conflictingqa_288cd1b45aab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Importantly, experts agree that wrist rests are not a substitute for proper posture and desk alignment that continuous contact with the rest can compress carpal tunnel nerves, making periodic pauses essential

### Sample conflictingqa_3dba586dca0f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: Real Christmas trees are generally considered more sustainable than artificial ones, primarily because they absorb carbon dioxide while growing and can be recycled as mulch or wood chips, whereas artificial trees are made from plastic and metal and release pollutants during manufacturing

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: None
- **Claim**: Opposing perspectives highlight serious ethical concerns about the practice itself — including the suffering inflicted on animals and the risk that blanket bans could lead to unregulated poaching or displacement of communities, particularly among vulnerable species like lions in Africa

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: Research remains divided, with scientists and conservation organizations offering conflicting studies on whether financial benefits outweigh harms, underscoring that the answer depends heavily on contextual factors such as regulation quality and species involved

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: However, patent eligibility remains contested — recent Supreme Court rulings such as Alice Corp. v

### Sample conflictingqa_52e01830d2fe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2, d5
- **Claim**: Rather than a blanket prohibition, the debate centers on defining clear boundaries for when software implementations constitute novel, protectable inventions

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1, d3
- **Supporting Docs Found**: d4
- **Claim**: However, many commercially available products claim to temporarily smooth the cuticle, add weight to frayed ends bond split fibers together, effectively masking the damage rather than reversing it

### Sample conflictingqa_747727772a30

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: When rain is light, bees can still forage and move about some species like bumblebees appear more tolerant of poor weather conditions than others

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Randomized controlled trials further complicate the picture, with some showing that replacing saturated fats with primarily n-6 polyunsaturated fats has no effect on heart disease risk, while others suggest a 10% reduction in risk is achievable through such dietary swaps

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: A notable exception is the Harvard PREDIMED trial, which found that replacing saturated fats with monounsaturated fats significantly reduced the risk of major cardiovascular events by approximately 30% , underscoring that the type of unsaturated fat may also play a critical role in determining cardiovascular outcomes

### Sample conflictingqa_7998e59d9b97

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d4
- **Claim**: However, organic farming offers significant non-yield benefits such as reduced pesticide use, lower greenhouse gas emissions during production preservation of natural biodiversity

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: Researchers note that the trade-off between higher yields and environmental footprint creates a complex challenge that a balanced combination of both methods may be necessary to meet future global food demand sustainably

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, some scientists argue that dark matter is not a confirmed necessity: observational clues are plentiful, but no single theory has achieved widespread acceptance researchers continue to pursue alternatives such as modified gravity models

### Sample conflictingqa_8efa53ba7c60

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d1
- **Claim**: No, birds are not descendants of T-Rex

### Sample conflictingqa_a25014a5c5b5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: A notable example is the snake plant, which can grow well with overhead office lights for at least eight hours a day, as long as the lighting is consistent and the plant is not subjected to complete darkness for too long

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: On the other hand, some sources acknowledge that a minority of listeners did experience genuine fear — Frank Brady's biography of Welles describes the public as 'on edge' that night — and the incident remains firmly embedded in popular culture as a prime example of radio's early power

### Sample conflictingqa_b2524e4883ad

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: In summary, while the average meteor shower is harmless, researchers monitor specific streams like the Taurids with a cautious eye toward understanding the full range of potential impacts

### Sample conflictingqa_b7fd50f9f980

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: d3
- **Claim**: Major dictionaries such as Merriam-Webster further clarify that 'alright' is a valid variant while it is widely accepted, 'all right' remains the preferred form in formal writing

### Sample conflictingqa_dd426f7706e8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3, d4
- **Claim**: A separate mythological claim holds that a full moon can create a werewolf monster , though this is presented as a cinematic trope rather than a factual historical account

### Sample conflictingqa_f1932b75ace7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d3
- **Claim**: However, studies are predominantly focused on investigating the effects of bee venom on arthritis rather than confirming therapeutic efficacy research has not definitively established that bee stings treat or cure the condition

### Sample conflictingqa_f22b389be1d6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d1
- **Supporting Docs Found**: d3
- **Claim**: Barefoot running is widely reported to shift gait to a more mid-foot strike, potentially reducing impact forces on the knees and hips

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: None
- **Claim**: Other early disasters included the accidental use of real daggers during the murder of King Duncan, which resulted in the death of the actor playing Duncan a stage weight nearly crushing Laurence Olivier during a 1937 production at the Old Vic

### Sample conflictingqa_f3b163170581

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: The conventional scientific view

### Sample conflictingqa_f4693bea2c31

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Emojipedia defines emoji as "pictographic icons used to express an idea or emotion," and most linguists agree that emoji are used to augment, enhance add complexity to text rather than replacing it. While some scholars argue that emoji are developing into something more linguistically significant — capable of conveying nuanced emotional and paralinguistic information — they are generally not considered a separate written language, but rather a complex system of pictographs that expand communication. As such, the broader consensus across linguistic research is that emoji complement traditional written language rather than serving as a standalone form of expression

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: Over the following decades, other Dutch navigators such as Dirk Hartog, Frederik de Houtman Abel Tasman charted extensive sections of Australia's western, southern northern coasts, respectively

### Sample freshqa_114b9082bc42

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3
- **Claim**: August 16, 1977

### Sample freshqa_1fe3ba1b1cba

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Lando Norris

### Sample freshqa_25b286cb2af1

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d2
- **Claim**: The idea that Venus could have had a moon in the past is also discussed, as collisions with other objects could have ejected material that later formed moons, but no current natural satellite is confirmed around Venus

### Sample freshqa_31ad09b9cd22

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Samara Joy

### Sample freshqa_42796b35e143

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: *The Mandalorian* has three full seasons released

### Sample freshqa_42796b35e143

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Season 3 premiered on March 1, 2023 was the last season released at the time of the Wikipedia article's update

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample freshqa_5eb89aae15f3

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d4
- **Claim**: Kantara

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: The first animal to orbit the Moon was a dog named Laika on the Soviet Union's Sputnik 2 mission in 1957

### Sample freshqa_a41257e9d6f6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: He was 43 years old at the time, tragically losing his life just two years after the publication of Minsky and Papert's influential book that critiqued his work

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: Queen Elizabeth II of England died on 8 September 2022

### Sample freshqa_a50d0f1f3cdf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3, d4
- **Supporting Docs Found**: d2
- **Claim**: This date is corroborated across multiple authoritative sources, including Wikipedia, Britannica History.com

### Sample freshqa_c7315f8b3029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: No, OpenAI did not release a GPT-5.5 model

### Sample freshqa_ddd643091cbe

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3
- **Claim**: No permanent cure for cancer has been developed; ongoing research is continuously exploring new treatments such as vaccines and gene editing that could one day eliminate all traces of the disease

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, d3 states play resumed by 9:16 p.m., creating a conflict in reported resumption times

### Sample freshqa_fc07d4525dbf

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: d3
- **Claim**: However, eligibility varies by vaccine type: Moderna's vaccine is approved for individuals 6 months and older, Pfizer's is for those 5 and older Novavax's is for people 12 and older

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Additional guidance notes that children under 5 years old are no longer eligible for Pfizer's vaccine the overall framework is being closely monitored by the CDC and IDSA

### Sample qacc_0091

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d8, d7, d6, d2, d5, d4
- **Claim**: 506

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_15ffab2466f7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5
- **Claim**: Princess of Wales Theatre

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A fourth character, Calvin, is based on Kevin Carroll

### Sample qacc_51c89636151e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d3
- **Claim**: Manwë (as represented in Quora and Nerdist content)

### Sample qacc_67b35f41ba84

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample qacc_6837d86d03ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d6, d8, d3
- **Supporting Docs Found**: None
- **Claim**: Other sources reflect a superseded view — such as Quora and Reddit discussions referencing Prince Harry — but these predate King Charles III's coronation following Queen Elizabeth II's death

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

### Sample qacc_883303a2d535

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d4
- **Claim**: Australian Shepherd

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, the official 2017 AL West page on Wikipedia is the most authoritative source cited, confirming the Red Sox as the AL East champions for that season

### Sample qacc_ba7aaa9b36c8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d1
- **Claim**: The blaze required 130 firefighters from 19 engine companies and four truck companies to contain no one was injured

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: President Hoover, clad in a heavy blue overcoat, watched the fire from the West Terrace the following Christmas, White House staff and their children gathered again to celebrate the holidays

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

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Notable individuals with the surname include American actor Christopher Tavarez and Dominican-American professional baseball player Jesús Tavárez, further illustrating its presence in both the United States and the Caribbean

### Sample qacc_d78d45c0e30f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d4
- **Supporting Docs Found**: None
- **Claim**: Senegal's CAF Third Round Group D win

### Sample qacc_d7c6682b5335

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Japan: 1996

### Sample qacc_d7df0a1856b7

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d7, d1, d6
- **Supporting Docs Found**: d2, d4
- **Claim**: While d4 provides a direct and current classification, d2 offers essential context on how the Hubble classification system operates multiple other sources confirm the barred spiral designation through educational and historical references

### Sample qacc_d9b756cb0eea

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d3, d4
- **Claim**: XXXTENTACION

### Sample qacc_e7318f6f3bbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: A related variant, 'strengthlessnesses,' is sometimes cited as the longest word with only one vowel repeated, further illustrating the complexity of English phonetic structure

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3, d4
- **Claim**: The welfare state was introduced at different times across regions: Germany began introducing social insurance in the late 19th century (1883 legislation), Britain's first modern state welfare measures were enacted by Liberal governments between 1906 and 1914, the U.S. established its foundational programs (Social Security) in 1935 the modern British welfare state was consolidated following World War II under the Beveridge Report of 1942

### Sample situatedqa_geo_4cb699778b59

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: To clarify further, the original Senate design called for seven-year terms, but this was amended to six years via the Sixteenth Amendment ratified in 1913

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Recent data shows immigrants are coming from a broad range of countries, with no single origin dominating

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: President Kennedy was the first U.S. president to send military advisers to South Vietnam, authorizing the deployment of 16,000 American troops to help stem a communist military invasion of the south

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Contextual scope mechanism: documents cover distinct but valid contextual scopes — specific historical elections versus broader suffrage milestones — providing complementary information rather than contradictions

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5
- **Claim**: This framework initially formed a war-time confederation of states, creating a weak central government that largely preserved state power and independence

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d3
- **Claim**: August 24, 1814

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The FOMC consists of twelve members — including seven members of the Board of Governors and four rotating Reserve Bank presidents — who meet regularly to adjust interest rates and manage the money supply, with the primary goals of promoting stable prices and maximum employment

### Sample situatedqa_temp_051502801f9c

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: July 13, 1972

### Sample situatedqa_temp_40e6764f611f

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: Mort is a Goodman's mouse lemur (also called a pygmy lemur or toyotama monkey), which is a small primate native to Madagascar — though the spin-off series All Hail King Julien reveals that he is also 40% bear, 20% spider, 10% starfish 30% other things, making him technically a bear. The Madagascar franchise is known for blending reality and fantasy, so while Mort's core species is the mouse lemur, his full genetic makeup includes non-lemur elements, making him a unique fictional hybrid

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d1
- **Claim**: While d5 lists only through 1986, the most current and comprehensive sources confirm UCLA's record stands at 12 titles, reflecting the conflict due to outdated information presented in the older document

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d3, d4
- **Supporting Docs Found**: None
- **Claim**: Older data once had Michael Jordan at the top with 59,870 points, but that table was superseded by the updated 2025–26 season statistics the most current rankings consistently place LeBron James at the absolute top

### Sample situatedqa_temp_5a59faf24972

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5
- **Supporting Docs Found**: d3
- **Claim**: The latest stable major version is Android 15 (released September 3, 2024), which was first released to a device by Vivo on September 30, 2024

### Sample situatedqa_temp_657c130afab6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d5, d3, d4
- **Claim**: 1980 (established as a national park)

### Sample situatedqa_temp_6f777dda5314

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5, d4
- **Claim**: Todd Monken is the head coach of the Cleveland Browns

### Sample situatedqa_temp_9049c197a579

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d3
- **Claim**: California's state gasoline excise tax is the highest in the United States, at $0.60 per gallon as of July 2025

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4
- **Supporting Docs Found**: d2, d3
- **Claim**: Wilson Phillips is an American vocal trio consisting of Carnie Wilson, Wendy Wilson Chynna Phillips, the daughters of Beach Boy Brian Wilson

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: d5
- **Claim**: Formed in Los Angeles in 1989, the group initially gained fame with their self-titled debut album released in 1990, which featured hit singles such as "Hold On," "Release Me," and "You're in Love"

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: All three members contributed to the group's distinctive three-part harmonies, with the Wilson sisters being the daughters of John and Michelle Phillips (of the Mamas and the Papas) Carnie being the daughter of Brian Wilson

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: The actress has remained in her 30s throughout the show's run, with the character aging through a five-year time jump in Season 6B

### Sample situatedqa_temp_b797de4c6610

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: d1
- **Claim**: Other traditional biomarkers include creatine kinase (CK), creatine kinase–MB isoenzyme (CK-MB) lactate dehydrogenase (LDH), though these are less specific to the heart and can rise in response to other conditions

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The United States has hosted the Olympics nine times: four Summer Games and five Winter Games

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: It was formally declared operational in 2020, when it deployed on its maiden operational tour to the Indo-Pacific region

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d5
- **Supporting Docs Found**: d1
- **Claim**: Older sources may cite slightly different sub-threshold rankings , but the authoritative 2018 data consistently places India at 136th, reflecting a progressively improving peace score over time

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The surname is also found in Haiti, further confirming its widespread adoption across different regions

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: The WTO has 166 members

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1, d4
- **Claim**: Earlier records reflect a lower count — as reported by citing 164 members — but these figures are superseded by the more recent and authoritative data from 2024

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

### Sample trust_align_008

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2
- **Claim**: Jessica Lange is a member of the cast of *American Horror Story* (Season 2)

### Sample trust_align_068

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d2, d3
- **Claim**: The Monty Hall Problem: You pick Door 1 the host opens Door 3 to reveal a goat. The probability that the car is behind Door 1 remains 1/3, while the probability it is behind Door 2 increases to 2/3 — meaning switching is to your advantage. This is because the host's action of opening Door 3 provides new information that updates your initial 1/3 chance, making the remaining unopened door (Door 2) the better choice

### Sample trust_align_081

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: When water is rapidly frozen without prior boiling, these gases become incorporated into the ice structure, creating the familiar cloudy appearance of most commercial ice cubes

### Sample trust_align_081

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Commercial clear ice products, therefore, always begin with boiled (and degassed) water to ensure the final product is optically transparent

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4
- **Supporting Docs Found**: None
- **Claim**: The key biological difference lies in the mechanism of injury — alcohol-induced inflammation progresses to irreversible fibrosis, while surgical removal of up to half the organ triggers compensatory growth, reflecting the contrasting pathways of acute versus chronic liver damage

### Sample trust_align_115

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: d2
- **Claim**: The Boston Celtics last won the NBA championship in 1986, when they defeated the Houston Rockets in the 1986 NBA Finals

### Sample trust_align_135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the first officially documented race meeting took place much later — in 1651, when the Epsom Derby was first run

### Sample trust_align_151

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d3, d4
- **Claim**: The retrieved evidence indicates that Brown v

### Sample trust_align_169

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d4
- **Supporting Docs Found**: None
- **Claim**: During the partial phases of the eclipse, however, it is generally safe to look directly at the sun without special filters, though caution is advised when totality approaches

### Sample trust_align_189

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: While magnesium's flammability is well-known, its specific applications in consumer electronics like computer casings are not directly addressed by the available evidence — though its reactive properties make it relevant in certain specialized electronic manufacturing processes

### Sample trust_align_192

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d2
- **Claim**: Pat Metheny Group

### Sample wikirevision_0013

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This ownership is further corroborated by Microsoft's 2025 annual report, which references LinkedIn as a 17.8 billion dollar revenue generating subsidiary

### Sample wikirevision_0099

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3
- **Claim**: Australia (won the 2023 Cricket World Cup). The 2023 ICC Men's Cricket World Cup was the 13th edition of the tournament, hosted in India from 5 October to 19 November 2023, with Australia defeating India by six wickets in the final to claim their sixth World Cup title. As the 2027 edition is scheduled for 2027, Australia remains the latest champion

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This result is corroborated by the official Wikipedia page for the 2026 Wimbledon Championships, which confirms Sinner's victory over Matteo Berrettini in the final

### Sample wikirevision_0125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: He leads the Labor Party and has held the role continuously since taking office following the 2022 federal election

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: This result is corroborated by the Wikipedia page for the 2026 Wimbledon Championships, which confirms that the most recent edition of the tournament took place as scheduled

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1
- **Supporting Docs Found**: None
- **Claim**: Earlier information from a Wikipedia revision from October 2025 also listed Sinner as the current champion while the 2026 revision supersedes this, the underlying data showing his victory remains the authoritative source for the latest outcome

### Sample wikirevision_0151

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: As the 2027 Cricket World Cup is still scheduled for South Africa, Zimbabwe Namibia, Australia remains the most recent champion

### Sample wikirevision_0152

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: d4
- **Claim**: For the women's award, Rodri and Aitana Bonmatí were recognized, making them the latest female Ballon d'Or recipients

### Sample wikirevision_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Since his election, Steinmeier has represented Germany as head of state, presiding over national affairs and conducting official duties including state visits

### Sample wikirevision_0162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3
- **Supporting Docs Found**: None
- **Claim**: The 2026 FIFA World Cup is the most recent edition Argentina won that tournament — their third overall — beating France 4–2 in the final

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d1, d3
- **Supporting Docs Found**: None
- **Claim**: His election victory gave him the first Republican presidential win since 2016 he is the only U.S. president to have served more than two terms under the Twenty-second Amendment


================================================================================

*Report generated by CATS v2.0*
