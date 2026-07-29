# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 4 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.755 (over 49 samples)

**GR F1** *(used in CATS)*: 0.846

**Behavior Adherence**: 0.622 (over 45 applicable samples)

**Factual Grounding**: 0.281 (over 45 applicable samples)

**Single-Truth Recall**: 0.941 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.673

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.846
- **Precision**: 0.750
- **Recall**: 0.971
- **Accuracy**: 0.755
- TP=33, FP=11, FN=1, TN=4


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.737
- **GR F1** *(used in CATS)*: 0.828
- **Behavior**: 0.765 (n=17)
- **Grounding**: 0.333 (n=17)
- **Recall**: 1.000 (n=12)
- **CATS**: 0.731

### Type 2: Complementary Info

- **Samples**: 15 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.667
- **GR F1** *(used in CATS)*: 0.762
- **Behavior**: 0.538 (n=13)
- **Grounding**: 0.231 (n=13)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.510

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.600 (n=10)
- **Grounding**: 0.200 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.582

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.400 (n=5)
- **Grounding**: 0.400 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.622


================================================================================

## Cost Summary

**Total Cost**: $0.0701

**Decisions Made**: 150

**Average Cost per Decision**: $0.000467


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 150
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0702
- **Total Requests**: 150
- **Average Cost per Request**: $0.000468


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d7, d3, d9, d2
- **Claim**: Abi Carter won season 22 of American Idol

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d10, d1, d12, d7, d3, d8, d11, d9, d5, d2
- **Claim**: While some studies suggest that educational television can be beneficial for language skills in children, others indicate that it may not be as effective as previously thought

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d2
- **Claim**: The quality of the programming and the presence of adult interaction are crucial factors in determining the impact of television on language development

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d10, d1, d7, d3, d2
- **Claim**: Therefore, parents and caregivers should be cautious and consider a balanced approach, combining television with other forms of language stimulation, such as conversation and play, to promote optimal language development in children

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d9, d5, d2
- **Claim**: The PIN code system was introduced in India on August 15, 1972

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d10, d1, d7, d3, d8, d9, d5, d2
- **Claim**: Dina Boluarte is the most recent woman to become President of Peru, becoming the first female president in 2022

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d9, d8, d5
- **Claim**: Jannik Sinner is the current ATP top-ranked men's singles tennis player, according to some retrieved documents (), but the information may be outdated

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d7, d3, d8, d9, d5, d2
- **Claim**: <think>
Stage 1 - Evidence assessment:
- d1: partially supports - The snippet mentions the East India Company was founded in 1600, but it does not provide a clear answer to the query about when the East India Company came

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Verdict_reason: The snippet mentions the founding year but does not directly answer the query

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d7, d3, d8, d2
- **Claim**: Key fact: The East India Company was founded in 1600

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d7, d3, d6
- **Claim**: Source quality: high.
- d2: supports - The snippet directly answers the query by stating that the East India Company was incorporated by royal charter on December 31, 1600

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d6
- **Claim**: Verdict_reason: The snippet directly answers the query with a specific date

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d7, d3, d8, d9, d2
- **Claim**: Key fact: The East India Company was incorporated by royal charter on December 31, 1600

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d3, d9, d2
- **Claim**: Source quality: high.
- d3: supports - The snippet also directly answers the query by stating that the English East India Company was incorporated by royal charter on December 31, 1600

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d6
- **Claim**: Verdict_reason: The snippet directly answers the query with a specific date

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d7, d3, d5
- **Claim**: While stress may exacerbate ulcer symptoms, the majority of the evidence suggests that it is not a direct cause of stomach ulcers

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d8, d9, d5, d2
- **Claim**: At least one person from Princeton University has won the Fields Medal, specifically June Huh in 2022

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d5, d2
- **Claim**: Heated gemstones can be less valuable than their unheated counterparts, but the extent of the value difference varies depending on the type of gemstone and the treatment process

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d11, d9, d2
- **Claim**: Unheated gemstones, particularly sapphires and rubies, are often rarer and more valuable than heated ones

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d10, d11, d9, d5, d2
- **Claim**: However, heat treatment can improve the appearance and value of gemstones some gemstones, like Tanzanite, can be heat-treated to change their color

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d7, d3, d8, d9, d5, d2
- **Claim**: The existence and nature of innate knowledge is a topic of ongoing debate among philosophers and researchers, with some arguing that it is present from birth and others claiming that it is acquired through experience

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d3, d8, d9, d5, d2
- **Claim**: While some studies suggest that humans may have innate mathematical perception or be born with certain ideas, others argue that knowledge is acquired through sensory experience and that humans are born with no knowledge

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: Further research is needed to resolve this conflict

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d8, d5
- **Claim**: The final answer is: The Mercedes-Benz Group has approximately 166,056 to 167,397 employees, based on the most recent and specific information provided by d3 and d8

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d3, d8, d9, d5
- **Claim**: This is consistent with the information provided in other documents, which indicate that some states allow marriage at 16 or 17 with parental consent, but that others have raised the age to 18 with no exceptions

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d10, d1, d7, d3, d11, d9, d5
- **Claim**: Whether a startup needs a business plan depends on its specific circumstances, but a business plan can be beneficial for determining direction and sustainability, securing funding overcoming potential difficulties

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d10, d11
- **Claim**: However, alternative approaches like pitch decks and the lean start-up methodology may be more effective for fundraising and innovation

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d11, d10, d1
- **Claim**: Sitting is not definitively the new smoking, as some studies suggest that the risks associated with sitting are not as severe as those associated with smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d10, d1, d7, d3, d8, d9, d5
- **Claim**: However, sitting can still be bad for your health it is essential to be aware of the potential risks and take steps to mitigate them

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d9, d5, d2
- **Claim**: While public transportation is often slower than driving, there are potential solutions and benefits to making public transportation more convenient and efficient

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: Cities like Mexico City have successfully implemented bus rapid transit systems that have reduced travel times and emissions innovations like ridesharing and carpooling can also make public transportation more attractive

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d3, d8, d9
- **Claim**: By addressing the challenges and inefficiencies of public transportation, it is possible to make it a more viable alternative to driving

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d5, d2
- **Claim**: While dogs can learn to associate specific words with specific actions or objects can detect speech and distinguish languages without explicit training, the extent to which they understand human language remains unclear

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8
- **Claim**: All octopuses are venomous, with some species being more deadly than others

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d14, d6, d10, d1, d12, d3, d8, d11, d13, d9, d5, d2
- **Claim**: Melting land ice contributes to sea level rise, but the impact of melting sea ice is still a topic of debate

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: <think>
Stage 1 - Evidence assessment:
- d1: partially supports - The snippet mentions two books, but does not provide an exact count

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d7, d3, d8, d9, d5, d2
- **Claim**: Key fact: Shoshana Zuboff is the author of multiple books

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d7, d9, d1
- **Claim**: Source quality: high.
- d2: partially supports - The snippet mentions three books, but does not provide an exact count

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d7, d9, d1
- **Claim**: Key fact: Shoshana Zuboff is the author of three major books

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Source quality: high.
- d3: partially supports - The snippet lists two books, but does not provide an exact count

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d9, d5, d2
- **Claim**: Key fact: Shoshana Zuboff has written multiple books

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: Source quality: high.
- d4: partially supports - The snippet mentions 16 books, but this is likely an aggregation of ratings rather than an exact count

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d7, d3, d8, d9, d5, d2
- **Claim**: Key fact: Shoshana Zuboff has written many books

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d2
- **Claim**: Additionally, the web-shooters are not that special, just something an average person couldn't build

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The webbing dissolves in Raimi's Spider-Man trilogy, acting exactly the same way as normal spider webs

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d9, d5, d2
- **Claim**: The current world population is estimated to be around 8 billion, but the number of people who have ever lived is estimated to be between 45 billion and 125 billion, with some documents providing more recent estimates, such as 106 billion and 117 billion

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1, d7, d3, d8, d9, d5, d2
- **Claim**: Champagne comes solely from the Champagne region of France

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d7, d3, d5
- **Claim**: However, the exact date when it became mandatory for all members is unclear, with some sources suggesting 1851 () and others not specifying ()

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: The final answer is: Club soda can be used to remove certain types of stains, including beer and red wine stains, but its effectiveness may vary

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The final answer is: Washing Turkish towels increases their absorbency, making them even more effective at drying the body

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1, d3, d5, d2
- **Claim**: Unfortunately, the evidence does not provide a clear answer to the query "When does American's Got Talent come on?" The documents provide information about the show's premiere dates and broadcast schedules, but the information is not directly relevant to the query

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Metals appear coloured because of the selective reflection, absorption re-emission of light, which is dependent on the wavelength of the light

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: This is due to the unique properties of each metal, such as the reflectivity and absorption of different wavelengths

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: For example, gold and copper have low reflectivity at short wavelengths, resulting in the appearance of yellow and red colours

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Silver, on the other hand, has a high reflectivity of 95%-99% even into the far infrared, but suffers from decreasing reflectivity in the blue and ultraviolet spectral regions

### Sample #0517

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: This is why our brains register reflective surfaces as silver, as they are reflecting a wide range of colours, including those in the visible spectrum

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Night vision is often green, but the reasons behind this are not clearly stated

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The human eye is more sensitive to yellow-green light, which is why some LED flashlights emit this color to preserve night vision

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the exact reason why night vision is green is not explicitly stated in the provided documents

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Based on the evidence from d1, it appears that Kenya and Jamaica are tied for the most gold medals won at the 2014 Commonwealth Games, with 10 gold medals each

### Sample #0592

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the evidence from d2 suggests that Canada won only one gold medal, which is not the most

### Sample #0592

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Therefore, the answer is not definitive, but based on the available evidence, Kenya and Jamaica are the top contenders for the most gold medals won at the 2014 Commonwealth Games

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1
- **Claim**: Bookmakers calculate and set odds based on probability, but the exact process of deciding odds in play is not explicitly stated in the retrieved documents

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3
- **Supporting Docs Found**: None
- **Claim**: Dorothy lives in the Land of Oz, as stated in

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Mordecai Richler is a well-known author, specifically mentioned in his essays and book "Oh Canada!

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: The final answer is: Salt-based water softeners use salt to remove minerals from water, but the exact process of how salt softens water is not explicitly stated in the retrieved documents

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Gravity is a natural phenomenon that can be explained in various ways, including through motion, as an analogy with data through modified theories

### Sample #0654

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: However, a clear and universally accepted definition remains elusive


================================================================================

*Report generated by CATS v2.0*
