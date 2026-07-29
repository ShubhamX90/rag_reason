# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 2 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.735 (over 49 samples)

**GR F1** *(used in CATS)*: 0.840

**Behavior Adherence**: 0.660 (over 47 applicable samples)

**Factual Grounding**: 0.735 (over 47 applicable samples)

**Single-Truth Recall**: 0.765 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.750

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.840
- **Precision**: 0.723
- **Recall**: 1.000
- **Accuracy**: 0.735
- TP=34, FP=13, FN=0, TN=2


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.737
- **GR F1** *(used in CATS)*: 0.828
- **Behavior**: 0.647 (n=17)
- **Grounding**: 0.745 (n=17)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.763

### Type 2: Complementary Info

- **Samples**: 15
- **GR Accuracy**: 0.533
- **GR F1** *(used in CATS)*: 0.696
- **Behavior**: 0.867 (n=15)
- **Grounding**: 0.692 (n=15)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.752

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.400 (n=10)
- **Grounding**: 0.750 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.699

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.800 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.750


================================================================================

## Cost Summary

**Total Cost**: $0.0803

**Decisions Made**: 176

**Average Cost per Decision**: $0.000456


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 176
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0803
- **Total Requests**: 176
- **Average Cost per Request**: $0.000456


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d4, d7, d5, d2, d3, d1
- **Claim**: The evidence suggests that children can potentially learn language skills from television, but the effectiveness depends on various factors

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d4, d7, d5, d2, d3, d9, d10, d1
- **Claim**: Dina Boluarte is the most recent woman to become President of Peru, having been sworn in on Dec

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d1
- **Claim**: Due to the outdated nature of some sources, the exact current status cannot be definitively determined from the provided evidence [d1-d9]

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d4, d7, d2, d3, d1
- **Claim**: The retrieved evidence consistently indicates that heated gemstones are generally less valuable than unheated ones

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d2
- **Claim**: The concept of innate knowledge is debated among philosophers and researchers

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d3, d5
- **Claim**: These figures are consistent and reflect the company's global workforce. []

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: The retrieved documents present conflicting opinions on whether every startup needs a business plan

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9, d11, d1
- **Claim**: The evidence from the retrieved documents indicates that while prolonged sitting can lead to various health issues, it is not as harmful as smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d11, d9
- **Claim**: Studies and expert opinions suggest that the risks associated with smoking are significantly higher than those associated with sitting

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d7, d1
- **Claim**: Public transportation is generally slower than driving in most cities

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d7, d2, d3, d9, d1
- **Claim**: While there are exceptions and specific circumstances where public transportation can be faster, the overall trend is that driving is faster [d6, d8]

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d4, d7, d5, d2, d3, d1
- **Claim**: Dogs can understand human language to a certain extent, particularly familiar words and associated actions

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d1
- **Claim**: However, their understanding is significantly enhanced by tone and body language

### Sample #0320

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d4, d7, d3, d1
- **Claim**: All octopuses are venomous, though the level of toxicity varies among species

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d8, d4
- **Supporting Docs Found**: d6
- **Claim**: The exact count cannot be determined from the retrieved evidence

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d4, d7, d2, d9
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Continental Congress

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d4, d7, d5, d2, d3, d9, d1
- **Claim**: The current world population is approximately 8 billion people

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d3, d4
- **Claim**: Other sources confirm that the world population reached 8 billion in November 2022 []

### Sample #0399

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The evidence presents conflicting opinions on whether pregnant women should follow a vegan diet

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d4, d7, d5, d2
- **Claim**: However, other sources indicate that a well-planned vegan diet can be safe and even beneficial during pregnancy if proper nutritional needs are met through careful planning and supplementation []

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d4, d7, d5, d2
- **Claim**: Therefore, while a vegan diet can be healthy during pregnancy, it is crucial to ensure that all necessary nutrients are obtained through a balanced diet and supplements []. []

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d8, d4, d7, d5, d2, d3, d9, d1
- **Claim**: The evidence from the retrieved documents clearly supports the fact that Champagne can only come from the Champagne region of France

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d5, d1
- **Claim**: An AUV stands for Asian Utility Vehicle, a term used to describe vehicles primarily sold in Asia and designed for the Asian market

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d3
- **Claim**: Additionally, the fabric retains its softness even after many washes, which contributes to its overall quality and effectiveness

### Sample #0509

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d5, d2, d3, d1
- **Supporting Docs Found**: None
- **Claim**: However, the current scheduling details are not provided in the retrieved documents

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: Reflective surfaces, such as metal, can appear silver because silver has a very high reflectivity rate, typically between 95%-99% across the visible spectrum

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the specific reason why reflective surfaces are perceived as silver rather than a mix of colors is not directly addressed in the provided evidence

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d5, d1
- **Claim**: The perception of light and reflections plays a crucial role in how we perceive colors

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: Despite the high reflectivity of silver, the exact mechanism behind why reflective surfaces are perceived as silver remains unclear based on the given information

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved documents do not directly state who wrote the jazz classic "I Got Rhythm"

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, they provide context suggesting it was likely composed during the jazz era

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d4, d5
- **Supporting Docs Found**: None
- **Claim**: One document mentions "I Got Rhythm" being recorded in October 1930 , while another discusses George Gershwin's work on songs for Bessie Smith and Fats Waller

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Given the timeframe and the artists involved, it is reasonable to infer that the composer might be associated with the jazz scene of that period, though the exact identity is not specified in the provided evidence

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d1
- **Supporting Docs Found**: None
- **Claim**: However, the exact reason for the green color in night vision remains partially explained based on the provided evidence

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2, d3, d4, d1
- **Supporting Docs Found**: None
- **Claim**: However, the exact number of championships cannot be determined from the provided evidence

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d4, d1
- **Supporting Docs Found**: None
- **Claim**: In-play odds are adjusted quickly based on live events and updated probabilities, though the exact methods for rapid in-play adjustments are not detailed in the provided snippets

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Mordecai Richler wrote "Oh Canada!

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the retrieved documents do not provide a comprehensive list of all his books [d1-d5]


================================================================================

*Report generated by CATS v2.0*
