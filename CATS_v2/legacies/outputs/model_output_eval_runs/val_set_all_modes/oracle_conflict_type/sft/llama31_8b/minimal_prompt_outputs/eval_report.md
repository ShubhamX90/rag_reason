# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.980 (over 49 samples)

**GR F1** *(used in CATS)*: 0.985

**Behavior Adherence**: 0.647 (over 34 applicable samples)

**Factual Grounding**: 0.912 (over 34 applicable samples)

**Single-Truth Recall**: 0.824 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.842

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.985
- **Precision**: 1.000
- **Recall**: 0.971
- **Accuracy**: 0.980
- TP=33, FP=0, FN=1, TN=15


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.947
- **GR F1** *(used in CATS)*: 0.957
- **Behavior**: 0.583 (n=12)
- **Grounding**: 0.896 (n=12)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.817

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.500 (n=8)
- **Grounding**: 0.823 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.774

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.667 (n=9)
- **Grounding**: 0.963 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.877

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 1.000 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.950


================================================================================

## Cost Summary

**Total Cost**: $0.0963

**Decisions Made**: 155

**Average Cost per Decision**: $0.000622


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 155
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0963
- **Total Requests**: 155
- **Average Cost per Request**: $0.000622


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d1, d9
- **Claim**: This result is consistent across multiple sources, with Entertainment Tonight, ABC News E!

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d7, d4, d6, d9, d8
- **Claim**: On the other hand, d6 (John Locke's Empiricism) and d9 (SparkNotes on Locke) present John Locke's influential empiricist view that the mind is a tabula rasa (blank slate) at birth, with all knowledge acquired solely through sensory experience, a view also reflected in d8 (Locke's Essay). d7 (Locke's Essay) further challenges the idea of innate knowledge, arguing that anything requiring reason to discover it cannot be innate. d3 (Northeastern University) adds a psychological dimension, showing that people are biased against accepting that knowledge can be inborn, while d4 (Apologetics Central) defines innate knowledge as a priori knowledge not derived from experience

### Sample #0190

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5, d9
- **Supporting Docs Found**: None
- **Claim**: 2020, all U.S. states have raised the minimum marriage age to 18, with no exceptions allowed

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d8
- **Claim**: In contrast, d4 (Goodreads) and d8 (Amazon author page) report 16 and 11 titles respectively d3 lists only two books, creating a significant count discrepancy that cannot be resolved from the available evidence

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d7, d8
- **Supporting Docs Found**: None
- **Claim**: In contrast, in oceanography and marine research, AUV stands for Autonomous Underwater Vehicle — a unmanned, untethered robot operated by an onboard computer that conducts underwater surveys and research

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The two definitions are mutually exclusive and reflect a contextual scope difference: the AUV body type is a car design category, while AUV in oceanography refers to a type of underwater robotic vehicle


================================================================================

*Report generated by CATS v2.0*
