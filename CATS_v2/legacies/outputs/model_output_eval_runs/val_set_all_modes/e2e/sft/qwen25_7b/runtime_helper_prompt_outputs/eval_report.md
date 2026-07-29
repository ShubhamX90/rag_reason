# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.735 (over 34 applicable samples)

**Factual Grounding**: 0.907 (over 34 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.881

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 1.000
- **Precision**: 1.000
- **Recall**: 1.000
- **Accuracy**: 1.000
- TP=34, FP=0, FN=0, TN=15


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.583 (n=12)
- **Grounding**: 0.917 (n=12)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.854

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.833 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.903

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.778 (n=9)
- **Grounding**: 0.907 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.895

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.900


================================================================================

## Cost Summary

**Total Cost**: $0.0928

**Decisions Made**: 151

**Average Cost per Decision**: $0.000615


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 151
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0928
- **Total Requests**: 151
- **Average Cost per Request**: $0.000615


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9
- **Supporting Docs Found**: d6, d8, d7, d2, d4, d1, d3
- **Claim**: The overall rule is that heating is viewed as a corrective or enhancing treatment that reduces value, except for particularly rare, naturally heated stones that command a premium

### Sample #0203

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6, d1
- **Supporting Docs Found**: d2
- **Claim**: Following the initial terrestrial launch, approximately 65% of UK households could receive the signal a month later Channel 5 also began broadcasting on the Astra satellite to reach the remaining areas

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d6, d2, d4, d1, d3
- **Claim**: Overall, the weight of evidence across multiple high-credibility sources consistently supports that smoking plays a causal role in the development of RA

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d1
- **Supporting Docs Found**: None
- **Claim**: The idea of organic web shooters originated in James Cameron's movie concept as a metaphor for puberty while comic fans were initially resistant to this major departure from the source material, director Sam Raimi handled the concept tastefully by making it a natural evolution of Peter Parker's character

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d5, d3
- **Supporting Docs Found**: d2, d4
- **Claim**: In the Raimi film trilogy, Peter Parker did not have organic web shooters from the beginning; instead, the 2002 film featured him with mechanical web shooters it was only later in the comics that he developed the ability to secrete organic web fluid himself

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: On the other hand, Belgium's Royal Academy of Medicine recommended that pregnant women avoid vegan diets due to unavoidable nutritional shortcomings and risks of irreversible harm a 1987 study noted that vegans must use supplements such as prenatal vitamins, iron calcium—indicating that未经格式化处理的完整答案在此，来自文档的直接支持引文已按规则提供。

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d5, d6, d8, d7, d2, d9, d1
- **Claim**: While the revelation itself was received on 27 February 1833 and was originally presented as "not by commandment," it gradually became a binding commandment for all Church members, including a requirement for temple recommends

### Sample #0427

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: d5, d1
- **Claim**: In the automotive context, AUV stands for Asian Utility Vehicle, which is a class of passenger vehicle primarily sold in Asia, designed to seat 8–10 people and be used for commercial purposes

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7, d4, d3
- **Supporting Docs Found**: None
- **Claim**: In the broader context of robotics and marine science, AUV actually means Autonomous Underwater Vehicle — a free-swimming, untethered robot controlled by an onboard computer, used for underwater research and survey missions


================================================================================

*Report generated by CATS v2.0*
