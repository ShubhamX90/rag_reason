# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 1 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.714 (over 49 samples)

**GR F1** *(used in CATS)*: 0.829

**Behavior Adherence**: 0.646 (over 48 applicable samples)

**Factual Grounding**: 0.847 (over 48 applicable samples)

**Single-Truth Recall**: 0.882 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.801

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.829
- **Precision**: 0.708
- **Recall**: 1.000
- **Accuracy**: 0.714
- TP=34, FP=14, FN=0, TN=1


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.632
- **GR F1** *(used in CATS)*: 0.774
- **Behavior**: 0.789 (n=19)
- **Grounding**: 0.746 (n=19)
- **Recall**: 0.833 (n=12)
- **CATS**: 0.786

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.857 (n=14)
- **Grounding**: 1.000 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.861

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.900
- **GR F1** *(used in CATS)*: 0.947
- **Behavior**: 0.000 (n=10)
- **Grounding**: 0.750 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.566

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 1.000 (n=5)
- **Recall**: 1.000 (n=5)
- **CATS**: 0.950


================================================================================

## Cost Summary

**Total Cost**: $0.0501

**Decisions Made**: 132

**Average Cost per Decision**: $0.000379


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 132
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0501
- **Total Requests**: 132
- **Average Cost per Request**: $0.000379


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d8, d2, d3, d1, d6, d5
- **Claim**: Children can learn language skills from television, but it is essential to use it in moderation and with parental guidance

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d2, d3, d1, d6, d5
- **Claim**: Well-designed television programs, such as Sesame Street and Dora the Explorer, can be beneficial for child language skills, especially when they are accompanied by adult interaction and co-viewing

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8
- **Claim**: Additionally, watching TV with subtitles can improve children's reading skills and literacy

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d11, d10, d1, d6
- **Claim**: However, excessive screen time can displace other forms of language development, such as conversation and play can have negative effects on children's language skills

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d12, d10, d2, d1
- **Claim**: Therefore, parents should engage in back-and-forth conversation with their children and encourage them to participate in real-time social interactions to promote language development

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d8, d3, d1, d5
- **Claim**: Every startup needs a business plan, as it serves as a roadmap for achieving goals, helps with fundraising provides a clear direction for the company

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d11
- **Claim**: However, some startups may benefit from a lean start-up approach, which involves testing and revising hypotheses, gathering customer feedback rapidly iterating on and reengineering products

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9, d7, d4, d8, d2, d3, d1, d5
- **Supporting Docs Found**: d10
- **Claim**: Additionally, a pitch deck may be more effective in fundraising than a traditional business plan

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d9
- **Supporting Docs Found**: d7, d4, d8, d3, d1, d6, d5
- **Claim**: While dogs may not understand human language in the same way as humans, they are capable of learning to recognize and respond to certain words and phrases

### Sample #0343

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4
- **Supporting Docs Found**: d5
- **Claim**: The web shooters were later replaced with mechanical ones in some versions

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The major differences between socialism and communism are not explicitly stated in the provided documents

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d3
- **Supporting Docs Found**: d4
- **Claim**: However, based on the information in d3, which discusses the differences between Marxism, Socialism Communism, we can infer that socialism is often seen as a step towards communism, but the exact differences between the two ideologies are not clearly defined

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3
- **Supporting Docs Found**: None
- **Claim**: Socialism typically involves a mixed economy with a strong public sector, while communism aims for a classless, stateless society

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3
- **Supporting Docs Found**: None
- **Claim**: However, the exact nature of these differences is not explicitly stated in the provided documents

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d4
- **Supporting Docs Found**: None
- **Claim**: The composer of the jazz classic "I Got Rhythm" is George Gershwin

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d2, d3
- **Supporting Docs Found**: None
- **Claim**: The Emerald City in the Land of Oz

### Sample #0644

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Mordecai Richler wrote essays and a book, including "Oh Canada!

### Sample #0644

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: He was also the stepfather of Daniel Richler

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d4, d2
- **Supporting Docs Found**: d3, d1
- **Claim**: This process is used in salt-based water softeners


================================================================================

*Report generated by CATS v2.0*
