# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 15 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 1.000 (over 49 samples)

**GR F1** *(used in CATS)*: 1.000

**Behavior Adherence**: 0.824 (over 34 applicable samples)

**Factual Grounding**: 0.904 (over 34 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.903

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
- **Behavior**: 0.833 (n=12)
- **Grounding**: 0.951 (n=12)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.925

### Type 2: Complementary Info

- **Samples**: 15 (7 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.875 (n=8)
- **Grounding**: 0.885 (n=8)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.920

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.778 (n=9)
- **Grounding**: 0.935 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.904

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.767 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.842


================================================================================

## Cost Summary

**Total Cost**: $0.0932

**Decisions Made**: 151

**Average Cost per Decision**: $0.000617


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 151
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0932
- **Total Requests**: 151
- **Average Cost per Request**: $0.000617


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6, d1, d4, d2, d7, d10, d11, d12, d9, d8, d3
- **Claim**: Overall, the evidence suggests that the impact of television on language learning is highly conditional—depending on the program's quality, content, duration whether adult interaction accompanies viewing—rather than a universal benefit

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d6, d3
- **Supporting Docs Found**: d1, d4, d2, d5, d9
- **Claim**: The six-digit PIN code was designed to streamline mail sorting and delivery in a country with over 1.5 lakh post offices and diverse regional names and languages

### Sample #0104

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8, d4, d5
- **Claim**: As the conflict is due to temporal updates — with d7 reflecting mid-2023 data and d4 through d9 showing Sinner as #1 as of January 2025 — the most current information suggests Jannik Sinner is the current ATP top-ranked men's singles tennis player

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d6, d1, d4, d3
- **Claim**: It is well-established that other factors—such as NSAID use, H. pylori

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d7
- **Supporting Docs Found**: d10, d5
- **Claim**: No other documents contradict this count of 15 Princeton winners, though some sources list additional non-Princeton-related figures in truncated contexts

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d11
- **Supporting Docs Found**: d6, d1, d4, d2, d7, d8, d3
- **Claim**: Heated gemstones are generally considered less valuable than their unheated counterparts, though the extent depends on the specific stone and treatment quality

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d7, d2
- **Claim**: These figures are consistent with the company's annual reporting and reflect the difference between the global parent company workforce and its U.S. subsidiary specifically

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4, d5
- **Claim**: Her nomination was made by President Biden she was confirmed by the Senate on April 7, 2022, following the retirement of Justice Stephen Breyer

### Sample #0394

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: SoFi Stadium previously hosted Super Bowl 50 in 2015 will mark its second time hosting the NFL's premier championship game

### Sample #0427

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d8, d7, d3
- **Supporting Docs Found**: None
- **Claim**: In a completely different context, AUV can also mean **Autonomous Underwater Vehicle**, which is an unmanned vehicle designed to operate underwater without guidance according to preprogrammed instructions ; this usage is entirely unrelated to automobiles and applies to marine robotics and underwater research


================================================================================

*Report generated by CATS v2.0*
