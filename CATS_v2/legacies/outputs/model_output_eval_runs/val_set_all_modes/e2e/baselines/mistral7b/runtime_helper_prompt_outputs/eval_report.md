# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 1 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.673 (over 49 samples)

**GR F1** *(used in CATS)*: 0.800

**Behavior Adherence**: 0.479 (over 48 applicable samples)

**Factual Grounding**: 0.632 (over 48 applicable samples)

**Single-Truth Recall**: 0.676 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.647

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.800
- **Precision**: 0.696
- **Recall**: 0.941
- **Accuracy**: 0.673
- TP=32, FP=14, FN=2, TN=1


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19
- **GR Accuracy**: 0.579
- **GR F1** *(used in CATS)*: 0.733
- **Behavior**: 0.737 (n=19)
- **Grounding**: 0.671 (n=19)
- **Recall**: 0.875 (n=12)
- **CATS**: 0.754

### Type 2: Complementary Info

- **Samples**: 15 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.600
- **GR F1** *(used in CATS)*: 0.727
- **Behavior**: 0.429 (n=14)
- **Grounding**: 0.542 (n=14)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.566

### Type 3: Conflicting Opinions

- **Samples**: 10
- **GR Accuracy**: 0.800
- **GR F1** *(used in CATS)*: 0.889
- **Behavior**: 0.100 (n=10)
- **Grounding**: 0.600 (n=10)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.530

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.400 (n=5)
- **Grounding**: 0.800 (n=5)
- **Recall**: 0.200 (n=5)
- **CATS**: 0.600


================================================================================

## Cost Summary

**Total Cost**: $0.0514

**Decisions Made**: 130

**Average Cost per Decision**: $0.000395


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 130
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0514
- **Total Requests**: 130
- **Average Cost per Request**: $0.000395


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d9, d7, d3, d2
- **Claim**: Abi Carter won American Idol this year ()

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d8, d7, d3, d6, d2
- **Claim**: Children can learn language skills from television, but the effectiveness of this learning depends on the quality of the educational programs, the presence of social interactions the appropriateness of the content

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d12, d2, d7, d6
- **Claim**: Co-viewing and parent-child conversations are particularly important for promoting language acquisition

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d2, d8, d7, d3, d6
- **Claim**: Are heated gemstones less valuable?

### Sample #0175

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d9, d8, d7, d3, d6, d2
- **Claim**: Some documents support the existence of innate knowledge, while others argue against it

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9
- **Claim**: The documents suggest that as of June 1, 2020, the minimum marriageable age in all states, including Massachusetts, is 18 without exceptions

### Sample #0190

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d4, d3, d2
- **Claim**: However, the documents do not all specify the current legal age for marriage in Massachusetts there is some inconsistency and lack of clarity about the matter

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d7, d3, d8
- **Claim**: Every startup needs a business plan, as it provides guidance, attracts investors serves as a roadmap for success

### Sample #0201

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4, d5, d9, d11, d8, d7, d3, d2
- **Supporting Docs Found**: d10
- **Claim**: However, a pitch deck may be more effective for fundraising

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d3, d1, d2
- **Claim**: In cities, driving is generally faster than public transportation

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d4, d5, d8, d7, d3, d6, d2
- **Claim**: Dogs can understand certain words and associate them with specific actions or objects they can distinguish between human words they've previously heard and words they haven't

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d1, d4, d5, d9, d10, d8, d7, d3, d2
- **Supporting Docs Found**: d6
- **Claim**: The exact number of books Shoshana Zuboff has published cannot be determined with the provided documents

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: CANNOT ANSWER, INSUFFICIENT EVIDENCE

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d9
- **Claim**: The current world population is 8,198,260,420 as of 2025

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d4, d1
- **Claim**: Club soda can be effective for removing stains, as supported by personal anecdotes and a study conducted by the Dry-cleaning & Laundry Institute (DLI)

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Socialism and Communism share some similarities, such as a focus on collective ownership and a concern for the welfare of the community

### Sample #0531

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: However, there are significant differences between the two ideologies

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Communism, as proposed by Marx, is the final stage of societal evolution, where all property is owned in common the state has withered away

### Sample #0542

- **Reason**: no_eligible_docs
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: George Gershwin wrote the jazz classic "I Got Rhythm"

### Sample #0561

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1, d4, d5, d3, d2
- **Supporting Docs Found**: None
- **Claim**: The evidence is inconclusive as to whether Jamie Oliver is a member of a television series cast

### Sample #0588

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3
- **Claim**: Night vision is green because of the technology used in night vision devices, which amplifies a specific range of light that is more visible to the human eye in low light conditions

### Sample #0603

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: UCLA has won 4 NCAA basketball championships

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d2
- **Claim**: Bookmakers calculate odds in play by using probability and trying to achieve a balanced book

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d1, d2
- **Claim**: They offer odds based on how likely they believe an event is to happen they adjust the odds as more bets are placed to try to achieve a balanced book

### Sample #0609

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: This means that they will offer different odds for the same event depending on the distribution of bets among different outcomes

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Dorothy does not live in the Emerald City in the Wizard of Oz

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: Further research is needed to determine where she lives

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Salt softens water by replacing the minerals in hard water with sodium ions

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: This process is known as ion exchange

### Sample #0650

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d3, d1, d2
- **Supporting Docs Found**: None
- **Claim**: When hard water passes through a resin bed containing sodium ions, the calcium and magnesium ions in the hard water are exchanged for sodium ions, resulting in soft water

### Sample #0654

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d5
- **Supporting Docs Found**: None
- **Claim**: However, the documents do not provide a simple, easy-to-understand definition of gravity


================================================================================

*Report generated by CATS v2.0*
