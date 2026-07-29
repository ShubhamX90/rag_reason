# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 6 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.755 (over 49 samples)

**GR F1** *(used in CATS)*: 0.838

**Behavior Adherence**: 0.651 (over 43 applicable samples)

**Factual Grounding**: 0.151 (over 43 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.616

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.838
- **Precision**: 0.775
- **Recall**: 0.912
- **Accuracy**: 0.755
- TP=31, FP=9, FN=3, TN=6


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.737
- **GR F1** *(used in CATS)*: 0.828
- **Behavior**: 0.824 (n=17)
- **Grounding**: 0.294 (n=17)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.715

### Type 2: Complementary Info

- **Samples**: 15 (3 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.733
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.583 (n=12)
- **Grounding**: 0.083 (n=12)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.489

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.875
- **Behavior**: 0.444 (n=9)
- **Grounding**: 0.056 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.458

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.600 (n=5)
- **Grounding**: 0.000 (n=5)
- **Recall**: 0.600 (n=5)
- **CATS**: 0.522


================================================================================

## Cost Summary

**Total Cost**: $0.0508

**Decisions Made**: 119

**Average Cost per Decision**: $0.000427


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 119
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0508
- **Total Requests**: 119
- **Average Cost per Request**: $0.000427


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d5, d8, d1, d6
- **Claim**: Children learn language skills from television, but the effectiveness of educational television in improving language skills is a matter of conflicting opinions and research outcomes

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d11, d7, d3, d5, d9, d12, d8, d2, d1, d10, d6
- **Claim**: Some studies suggest that educational television can be beneficial for language development, while others suggest that it has no effect or can be detrimental

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d3, d12, d2, d1, d6
- **Claim**: Parents should carefully consider the content of the educational television shows their children are watching and engage in back-and-forth conversation with their children to promote language development

### Sample #0042

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d9, d8, d2, d1, d10, d6
- **Claim**: The song "My Girl (Gone, Gone, Gone)" was performed by the Canadian group Chilliwack

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d5, d9, d8, d2, d1, d10
- **Claim**: Dina Boluarte became the first female president of Peru on December 7, 2022

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d3, d8, d2, d1, d6
- **Claim**: The East India Company was founded in 1600

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d8, d2, d1, d6
- **Claim**: Are heated gemstones less valuable than natural gemstones?

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d3, d5, d8, d2, d1
- **Claim**: It depends on the specific gemstone and its qualities

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d9, d8, d2, d1, d10, d6
- **Claim**: While some heated gemstones may be less valuable than their natural counterparts, others may be more valuable due to their rarity or improved color and clarity

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d5, d9, d8, d2, d1, d6
- **Claim**: Do we have innate knowledge?

### Sample #0175

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The documents present conflicting opinions on the existence of innate knowledge, with some arguing for its existence and others arguing against it

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2
- **Claim**: The question of whether we have innate knowledge remains a topic of debate among philosophers

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d2, d3
- **Claim**: 166,000 employees (d2, d3)

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: The minimum legal age for marriage in the United States is 18 without exceptions, as of June 1, 2020

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d11, d7, d3, d5, d9, d8, d1, d10, d6
- **Claim**: Does every startup need a business plan?

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d8, d1, d10
- **Claim**: The evidence suggests that while some sources argue that every startup needs a business plan, others suggest that a pitch deck may be more effective for fundraising

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d7, d3
- **Claim**: However, a business plan can still serve as a useful tool for startups, as it allows them to think through their strategy at a deeper level and gather customer feedback

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d10, d6, d9
- **Claim**: Ultimately, the decision to create a business plan or a pitch deck may depend on the specific needs and goals of the startup

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d9, d10, d2, d1, d6
- **Claim**: Channel 5 started on March 30, 1997

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10, d9
- **Claim**: While some research suggests that sitting for prolonged periods can increase the risk of premature death and some chronic diseases, the risks associated with smoking are substantially higher

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d8, d9
- **Claim**: However, it is important to engage in regular physical activity to counteract the effects of prolonged sitting

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d2, d1, d6
- **Claim**: Can smoking cause Rheumatoid Arthritis?

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d2, d1, d6
- **Claim**: Yes, smoking can increase the risk of developing Rheumatoid Arthritis and can make the disease worse, according to multiple studies ()

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d9, d8, d2, d1, d6
- **Claim**: Public transportation is generally slower than driving in many cases, but it can be made faster through ridesharing, bus rapid transit systems improving the quality of public transportation service. ()

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d5, d9, d8, d2, d1, d6
- **Claim**: Can dogs understand human language?

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d5, d8, d2, d1, d6
- **Claim**: The evidence suggests that dogs can understand some words and intonation, but it is unclear whether they understand human language as a whole

### Sample #0301

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d7, d8, d1
- **Claim**: Dorfromantik - The Board Game won the 2023 Spiel des Jahres award

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d11, d3, d5, d9, d12, d13, d10, d2, d1, d14, d6
- **Claim**: Does melting sea ice contribute to sea level rise?

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d9, d12, d13, d8, d1, d11
- **Claim**: The documents agree that melting land ice contributes to sea level rise

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d11, d3, d5, d9, d12, d13, d10, d2, d1, d14, d6
- **Claim**: However, they disagree on the role of melting sea ice in sea level rise

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d11, d3, d5, d9, d12, d13, d10, d2, d1, d14, d6
- **Claim**: Some documents state that melting sea ice does not contribute to sea level rise, while others state that it does contribute to sea level rise

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d5, d8, d2, d6
- **Claim**: The Nobel Prize in Literature 2024 was awarded to Han Kang

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Ketanji Brown Jackson was appointed to the U.S. Supreme Court on April 7, 2022

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3
- **Claim**: Shoshana Zuboff has authored at least two books: "The Age of Surveillance Capitalism" and "In the Age of the Smart Machine" ()

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Spider-Man's original web shooters are mechanical, not organic. ()

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d5, d9, d8, d2, d1, d6
- **Claim**: Who signed the Declaration of Independence?

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d2
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Continental Congress, including future Presidents, Vice Presidents members of the United States Congress

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: The exact date of signing is disputed, with some sources stating it was on July 4, 1776 others stating it was on August 2, 1776

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1, d9
- **Claim**: The USA is currently at the top rank at the FIBA Men's World Ranking

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d3, d8, d1, d6
- **Claim**: Super Bowl LXI will be held at the Caesars Superdome in New Orleans in 2025

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d3, d5, d2, d1, d6
- **Claim**: Should pregnant women follow a vegan diet?

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The meaning of AUV in cars is not directly answered in the provided documents

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These documents discuss the differences between AUV, MPV SUV, but do not provide a clear definition of AUV in the context of cars

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d1
- **Claim**: Club soda is often used to remove stains, although its effectiveness may be debated. ()

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Bath towels made of Turkish cotton and Turkish towels become more absorbent over time due to their unique cotton fibers. (d1, d3)

### Sample #0509

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5, d2, d1
- **Claim**: America's Got Talent season 11 premiered on May 31, 2016

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1
- **Claim**: Our brains perceive reflective surfaces as silver because they selectively absorb and re-emit certain wavelengths of light, with gold and copper having low reflectivity at short wavelengths and yellow and red being preferentially reflected

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Silver, on the other hand, has a higher reflectivity across the visible spectrum, which is why it appears white or silver

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: George Gershwin wrote "I Got Rhythm"

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: The color of night vision is green because it is more accurately recognized by the cones in the human eye, which are more sensitive to yellow-green light (d3, d4)

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Dorothy lived in the Emerald City, the capital of Oz. ()

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Salt softens water by replacing minerals in the water with sodium ions, which reduces the hardness of the water

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: This process is facilitated by a resin inside the water softener that collects the minerals and is periodically regenerated with salt


================================================================================

*Report generated by CATS v2.0*
