# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 49

**Correct Refusals**: 4 (GR=1.0 only; excluded from behavior/grounding/recall averages)

**GR Accuracy**: 0.776 (over 49 samples)

**GR F1** *(used in CATS)*: 0.861

**Behavior Adherence**: 0.378 (over 45 applicable samples)

**Factual Grounding**: 0.078 (over 45 applicable samples)

**Single-Truth Recall**: 0.647 (over 17 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.491

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.861
- **Precision**: 0.756
- **Recall**: 1.000
- **Accuracy**: 0.776
- TP=34, FP=11, FN=0, TN=4


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 19 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.684
- **GR F1** *(used in CATS)*: 0.800
- **Behavior**: 0.389 (n=18)
- **Grounding**: 0.000 (n=18)
- **Recall**: 0.583 (n=12)
- **CATS**: 0.443

### Type 2: Complementary Info

- **Samples**: 15 (2 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 0.667
- **GR F1** *(used in CATS)*: 0.762
- **Behavior**: 0.385 (n=13)
- **Grounding**: 0.115 (n=13)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.421

### Type 3: Conflicting Opinions

- **Samples**: 10 (1 correct refusals excluded from sub-metrics)
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.111 (n=9)
- **Grounding**: 0.000 (n=9)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.370

### Type 4: Outdated Info

- **Samples**: 5
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.800 (n=5)
- **Grounding**: 0.400 (n=5)
- **Recall**: 0.800 (n=5)
- **CATS**: 0.750


================================================================================

## Cost Summary

**Total Cost**: $0.0438

**Decisions Made**: 116

**Average Cost per Decision**: $0.000378


### Per-Model Cost Breakdown

#### codex-cli/default

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 116
- **Average Cost per Request**: $0.000000

#### deepseek-v4-flash

- **Total Cost**: $0.0438
- **Total Requests**: 116
- **Average Cost per Request**: $0.000378


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample #0015

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d2, d9, d3
- **Claim**: The winner of American Idol this year is Abi Carter

### Sample #0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d10, d11, d7, d1, d2, d8, d9, d6, d4, d3, d12
- **Claim**: Overall, the evidence is mixed and conditional the answer to the query is partially supported

### Sample #0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict label is "No conflict", which means that there is no conflicting information in the documents that would contradict the answer

### Sample #0102

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1, d2, d8, d9, d6, d4, d3
- **Claim**: The PIN code system was introduced in India on August 15, 1972

### Sample #0103

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d10, d7, d1, d2, d8, d9, d6, d4, d3
- **Claim**: The most recent woman to become President of Peru is Dina Boluarte

### Sample #0127

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1, d2, d8, d6, d3
- **Claim**: The East India Company was founded in 1600

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Given the conflicting opinions and research outcomes, I would say that the answer to the query is partially supported by some sources, but not definitively proven by others

### Sample #0133

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d6, d1
- **Claim**: The most consistent answer is that stress is not a direct cause of stomach ulcers, but may contribute to their development in certain circumstances

### Sample #0139

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: Therefore, the answer to the query is 15

### Sample #0159

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1, d2, d8, d6, d4, d3
- **Claim**: Therefore, the final answer is that heated gemstones are less valuable, but the extent of the value reduction varies depending on the type of gemstone and the level of treatment

### Sample #0175

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict label is relevant because the documents present conflicting opinions and research outcomes on the topic of innate knowledge, with some documents arguing for its existence and others arguing against it

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d5, d2
- **Claim**: The most consistent and reliable answer is 166,000 employees, as it is mentioned in three different documents () with high source quality

### Sample #0187

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d8
- **Claim**: However, d8 also provides a similar answer with a slightly different number (167,397 employees)

### Sample #0201

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer to the query is partially supported, with some documents providing more direct and categorical statements than others

### Sample #0203

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d4, d6, d2, d9
- **Supporting Docs Found**: None
- **Claim**: has a high source quality, while documents have low source quality

### Sample #0206

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d11, d10, d1, d2, d8, d3
- **Claim**: The conflict label "CONFLICTING OPINIONS OR RESEARCH OUTCOMES" accurately reflects the mixed opinions and evidence presented in the retrieved documents

### Sample #0229

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d2, d6, d4, d3
- **Claim**: Smoking can cause Rheumatoid Arthritis

### Sample #0263

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3, d1, d2, d7
- **Claim**: Therefore, the overall verdict is that public transportation is generally slower than driving in cities, with some exceptions and variations depending on the specific context

### Sample #0300

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1, d2, d8, d6, d4, d3
- **Claim**: Therefore, the final answer to the query is that dogs can understand human language to a certain extent, but the extent of their understanding is not fully clear and may vary depending on the individual dog and the context

### Sample #0322

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The conflict label "Conflicting opinions or research outcomes" is appropriate for this query, as there is a clear disagreement between the majority of the documents and one document on the question of whether melting sea ice contributes to sea level rise

### Sample #0324

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d2, d8, d6
- **Claim**: The latest Nobel Prize in Literature was awarded to Han Kang in 2024

### Sample #0333

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d3, d5, d1
- **Claim**: The last person appointed to the U.S. Supreme Court is Ketanji Brown Jackson

### Sample #0334

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1, d2, d8, d9, d4
- **Claim**: Therefore, the total number of books Shoshana Zuboff has published is at least 3, but possibly more

### Sample #0343

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the final answer is based on the majority of the supporting evidence from the documents

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d7, d1, d2, d9, d6, d4
- **Claim**: The Declaration of Independence was signed by 56 delegates to the Continental Congress

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4
- **Claim**: The signers included notable individuals such as John Hancock, Thomas Jefferson, John Adams, Benjamin Franklin Edward Rutledge

### Sample #0373

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d7, d1, d9
- **Claim**: The signing of the Declaration took place on August 2, 1776, with John Hancock being the first to sign

### Sample #0381

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d7, d1
- **Claim**: There are approximately 8.2 billion people in the world as of January 8, 2025, according to the most recent United Nations estimates

### Sample #0392

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d1, d9, d4, d3
- **Claim**: These documents have a high source quality and directly answer the query

### Sample #0394

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d5
- **Claim**: The host stadium of this year's Super Bowl is Allegiant Stadium in Las Vegas, NV

### Sample #0399

- **Reason**: cross_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Overall, the evidence is not sufficient to provide a clear, definitive answer to the query

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: * d1: "The rules agreed upon by the CIVC are submitted for the final approval of the Institut national de l'origine et de la qualité (formerly the Institut National des Appellations d'Origine, INAO)

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: In 2007, the INAO, the government organization that controls wine appellations in France, was preparing to make the largest revision of the region's legal boundaries since 1927, in response to economic pressures

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: With soaring demand and limited production of grapes, Champagne houses say the rising price could produce a consumer backlash that would harm the industry for years into the future

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: That, along with political pressure from villages that want to be included in the expanded boundaries, led to the move

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5, d7, d1, d2, d8, d9, d6, d4, d3
- **Claim**: Changes are subject to significant scientific review and are said not to impact Champagne-produced grapes until 2020. A final decision is not expected until 2023 or 2024. 1915 English magazine illustration of a lady riding a Champagne cork (Lordprice Collection) Sparkling wines are produced worldwide, but many legal structures reserve the word Champagne exclusively for sparkling wines from the Champagne region, made in accordance with Comité Interprofessionnel du vin de Champagne regulations

### Sample #0408

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1, d8
- **Claim**: In the European Union and many other countries, the name Champagne is legally protected by the Madrid system under an 1891 treaty, which reserved it for the sparkling wine produced in the eponymous region and adhering to the standards defined for it as an appellation d'origine contrôlée; the protection was reaffirmed in the Treaty of Versailles after World War I. Similar legal protection has been adopted by over 70 countries

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Most recently Australia,Chile, Brazil, Canada and China passed laws or signed agreements with Europe that limit the use of the term \"Champagne\" to only those products produced in the Champagne region

### Sample #0408

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: The United States bans the use from all new U.S.-produced wine brands."
* d2: "Champagne is more than sparkling wine; it’s an experience that captures our hopes and happy memories

### Sample #0416

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d9, d8
- **Claim**: Therefore, based on the retrieved documents and the provided per-document notes, the most accurate answer to the query is that the Word of Wisdom became mandatory in 1919

### Sample #0470

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d4, d1
- **Claim**: Therefore, the answer to the query is that the documents partially support the idea that club soda works well for stain removal, but they do not provide a clear explanation of why it is effective

### Sample #0499

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Therefore, the answer is partially supported, but the explanation is limited to Turkish cotton towels

### Sample #0517

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer to the question is that the documents do not provide a clear answer the conflict label "Complementary information" is not applicable in this case

### Sample #0531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, I must conclude that the answer to the question is not fully supported by the provided documents

### Sample #0588

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: However, the scientific reason for this is not explicitly stated in any of the documents

### Sample #0603

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d3
- **Claim**: Therefore, the answer to the query is: UCLA has won at least 10 NCAA basketball championships

### Sample #0609

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: If you would like to know more about in-play odds calculation, I would recommend searching for additional sources that specifically address this topic

### Sample #0638

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d5
- **Claim**: However, the gold per-document notes for d5 mention that the query likely refers to the original "Wizard of Oz" where Dorothy lives in Kansas, which is not mentioned in d5

### Sample #0638

- **Reason**: no_supporting_doc_found
- **Cited Docs**: None
- **Supporting Docs Found**: None
- **Claim**: Therefore, the answer to the query cannot be definitively determined from the provided documents

### Sample #0650

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: None
- **Supporting Docs Found**: d1
- **Claim**: Therefore, the answer to the query is that salt softens water through an ion-exchange mechanism, but the exact process is not fully explained in any of the provided documents


================================================================================

*Report generated by CATS v2.0*
