# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 4 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.735 (over 49 samples)

**GR F1** *(used in CATS)*: 0.831

**Behavior Adherence**: 0.778 (over 45 applicable samples)

**Factual Grounding**: 0.745 (over 45 applicable samples)

**Single-Truth Recall**: 0.853 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.802

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.831
- **Precision**: 0.744
- **Recall**: 0.941
- **Accuracy**: 0.735
- TP=32, FP=11, FN=2, TN=4


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.737
- **GR F1** *(used in CATS)*: 0.828
- **Behavior**: 0.647 (n=17)
- **Grounding**: 0.775 (n=17)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.791

### Type 2: Complementary Info

- **Samples**: 15 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.667
- **GR F1** *(used in CATS)*: 0.762
- **Behavior**: 0.923 (n=13)
- **Grounding**: 0.795 (n=13)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.827

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.700
- **GR F1** *(used in CATS)*: 0.824
- **Behavior**: 0.800 (n=10)
- **Grounding**: 0.552 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.725

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.900 (n=5)
- **Recall**: 0.700 (n=5)
- **CATS**: 0.850


================================================================================

## Cost Summary

**Total Cost**: $0.0843

**Decisions Made**: 178

**Average Cost per Decision**: $0.000474


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 178
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0843
- **Total Requests**: 178
- **Average Cost per Request**: $0.000474


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d3, d5, d8
- **Claim**: The value of heated gemstones can vary depending on the specific type and context

### Sample #0175

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence presents conflicting views on whether humans have innate knowledge

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d7, d6
- **Claim**: The legal age for marriage varies significantly by jurisdiction

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: The retrieved evidence presents conflicting opinions on whether every startup needs a business plan

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d3, d7, d11
- **Supporting Docs Found**: d9, d6, d10
- **Claim**: Therefore, the necessity of a business plan depends on the specific circumstances and goals of the startup

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d3, d10
- **Claim**: Citing both sides of the argument, the evidence reveals conflicting opinions on whether sitting is the new smoking

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d9, d10
- **Supporting Docs Found**: d3
- **Claim**: Some experts and studies support the notion that sitting poses health risks comparable to smoking, while others argue that the risks are significantly lower

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: The evidence presents conflicting opinions on this issue

### Sample #0324

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d6, d2, d5
- **Supporting Docs Found**: None
- **Claim**: This is confirmed by multiple high-quality sources

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d6
- **Supporting Docs Found**: d3, d4, d5
- **Claim**: While Amy Coney Barrett was previously considered the most recent appointee, this information is outdated

### Sample #0334

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Shoshana Zuboff has published a varying number of books according to different sources

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d2, d5
- **Claim**: Spider-Man originally did not have organic web shooters; in the comics, he had mechanical web shooters that he designed himself

### Sample #0399

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The retrieved evidence presents conflicting opinions on whether pregnant women should follow a vegan diet

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6
- **Claim**: The Word of Wisdom became mandatory at different times according to varying interpretations

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: AUV stands for Asian Utility Vehicle in the context of cars

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: It refers to vehicles designed for use in the Asian market, capable of seating 8-10 people, hauling goods serving commercial purposes

### Sample #0470

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d5
- **Supporting Docs Found**: None
- **Claim**: However, the specific scientific reasons behind its effectiveness are not detailed in the retrieved evidence

### Sample #0499

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the retrieved evidence does not provide a detailed explanation for why this occurs across all types of bath towels

### Sample #0517

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d5
- **Claim**: Our brains perceive reflective surfaces like metal as silver because of how light interacts with these materials and how our eyes process the reflected light

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: While white light contains all colors, metals like silver reflect light in a way that our retinal neurons interpret as a single color, despite reflecting all colors

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d5
- **Supporting Docs Found**: None
- **Claim**: However, the exact mechanism of why reflective surfaces appear silver rather than a mix of reflected colors is not fully explained by the provided evidence

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: However, the exact major differences between socialism and communism cannot be fully determined from the provided evidence

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d3, d4
- **Supporting Docs Found**: None
- **Claim**: However, the exact technical reason for the green color in night vision devices is not fully explained by the retrieved documents

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the specific methods and speed at which bookmakers adjust odds in real-time during live events are not detailed in the provided evidence [d1-d5]

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The retrieved evidence only partially supports the query, naming one book by Mordecai Richler

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Mordecai Richler wrote a 1992 book titled "Oh Canada!

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: However, the exact list of all books written by Mordecai Richler cannot be determined from the retrieved evidence

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Salt-based water softeners use a resin to collect minerals found in hard water

### Sample #0654

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: These differing perspectives highlight the complexity of defining gravity in a simple manner


================================================================================

*Report generated by CATS v2.0*
