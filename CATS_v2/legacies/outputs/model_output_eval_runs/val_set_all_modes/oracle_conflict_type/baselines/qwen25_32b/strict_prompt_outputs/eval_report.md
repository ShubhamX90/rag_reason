# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 13 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.469 (over 49 samples)

**GR F1** *(used in CATS)*: 0.435

**Behavior Adherence**: 0.806 (over 36 applicable samples)

**Factual Grounding**: 0.819 (over 36 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.736

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.435
- **Precision**: 0.833
- **Recall**: 0.294
- **Accuracy**: 0.469
- TP=10, FP=2, FN=24, TN=13


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (6 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.368
- **GR F1** *(used in CATS)*: 0.143
- **Behavior**: 0.769 (n=13)
- **Grounding**: 0.897 (n=13)
- **Recall**: 0.917 (n=12)
- **CATS**: 0.682

### Type 2: Complementary Info

- **Samples**: 15 (6 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.667
- **GR F1** *(used in CATS)*: 0.615
- **Behavior**: 0.778 (n=9)
- **Grounding**: 0.798 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.730

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.500
- **GR F1** *(used in CATS)*: 0.615
- **Behavior**: 0.889 (n=9)
- **Grounding**: 0.628 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.711

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 0.200
- **GR F1** *(used in CATS)*: 0.333
- **Behavior**: 0.800 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.733


================================================================================

## Cost Summary

**Total Cost**: $0.0801

**Decisions Made**: 155

**Average Cost per Decision**: $0.000517


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 155
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0801
- **Total Requests**: 155
- **Average Cost per Request**: $0.000517


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d1, d9, d6, d4, d7, d3, d5, d10
- **Claim**: <final answer with proper citations>
The most recent woman to become President of Peru is Dina Boluarte, who was sworn in on Dec

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d7, d3, d5
- **Claim**: However, there is conflicting opinion regarding the role of stress in causing ulcers

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d9, d4, d3
- **Claim**: Therefore, while it is clear that Princeton University has a notable presence among Fields Medalists, the precise count remains unspecified

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d1, d6, d4, d7, d3, d5
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
Heated gemstones can be less valuable than unheated ones, but the context and type of gemstone play a role

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1
- **Supporting Docs Found**: d5, d4
- **Claim**: Prior to July 29, 2022, adolescents could be married with judicial consent, but this has been changed to set the minimum age at 18

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d10
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
The retrieved documents present conflicting opinions on whether every startup needs a business plan

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d10, d6
- **Claim**: Therefore, the necessity of a business plan for startups depends on the specific circumstances and goals of the startup

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d1, d11, d10
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
The comparison of sitting to smoking in terms of health risks is contentious

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d1, d9, d11, d10
- **Claim**: Therefore, while sitting can pose health risks, it is not necessarily equivalent to smoking in terms of overall health impact

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d7
- **Claim**: Public transportation is generally slower than driving in most cities

### Sample #0263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d7
- **Supporting Docs Found**: None
- **Claim**: In the Grand Rapids-Wyoming metro area, the average commute is 21.4 minutes for those who drive themselves and 40.4 minutes for public transport riders

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d9, d6, d7
- **Claim**: However, there are potential improvements and exceptions

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d8, d2
- **Supporting Docs Found**: d7
- **Claim**: Overall, while public transportation is generally slower than driving, there are efforts to improve its speed and efficiency

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d6, d4, d7, d5
- **Claim**: <final answer with proper citations>
The evidence presents conflicting views on whether dogs fully understand human language

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d1, d6, d4, d7, d3, d5
- **Claim**: Therefore, while dogs can understand some aspects of human language, the extent of their comprehension remains unclear

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: <final answer with proper citations>
The question of whether melting sea ice contributes to sea level rise is subject to conflicting opinions and research outcomes

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6
- **Claim**: The debate centers around the physical properties of water and ice, with some studies indicating that the displacement theory does not fully account for the effects of density differences

### Sample #0322

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d6
- **Claim**: Given the conflicting evidence, it is important to consider both perspectives when evaluating the impact of melting sea ice on sea level rise

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d1, d9, d4, d7, d3, d5
- **Claim**: <final answer with proper citations>
Shoshana Zuboff has published several books

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8, d2, d1, d9, d6, d4, d7, d3, d5
- **Claim**: The exact number of books varies across sources, but it is clear that she has published multiple works

### Sample #0399

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: <final answer with proper citations exactly CANNOT ANSWER, INSUFFICIENT EVIDENCE>
The safety of a vegan diet during pregnancy is a topic of debate

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1
- **Claim**: Some sources discourage vegan diets for pregnant women due to potential nutritional deficiencies and developmental issues (d1, d3)

### Sample #0399

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d6, d4, d7, d5
- **Claim**: However, other sources support well-planned vegan diets during pregnancy, emphasizing the importance of proper nutrient intake and supplementation to ensure the health of both the mother and the baby ()

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6
- **Claim**: <final answer with proper citations>
The exact date when the Word of Wisdom became mandatory is subject to differing interpretations

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d2, d3, d1, d4
- **Claim**: <final answer with proper citations>
Bookmakers calculate odds based on various factors, including probability and balancing their books

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Salt softens water by using a resin inside the water softener device

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: The resin collects minerals from the water


================================================================================

*Report generated by CATS v2.0*
