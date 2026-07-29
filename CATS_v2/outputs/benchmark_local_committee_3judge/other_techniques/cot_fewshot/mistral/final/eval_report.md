# CATS v2.0 Evaluation Report

================================================================================

## Overall Conflict-Aware Metrics

**Total Samples**: 736

**GR Accuracy**: 0.826 (over 736 samples)

**GR F1** *(used in CATS)*: 0.905

**Behavior Adherence**: 0.000 (over 736 applicable samples)

**Factual Grounding**: 0.000 (over 736 applicable samples)

**Single-Truth Recall**: 0.000 (over 487 applicable samples)

--------------------------------------------------------------------------------

### CATS Score: 0.226

*(average of 4 applicable sub-metrics)*

--------------------------------------------------------------------------------


### Dataset-level GR Metrics

- **F1** *(CATS component)*: 0.905
- **Precision**: 0.826
- **Recall**: 1.000
- **Accuracy**: 0.826
- TP=608, FP=128, FN=0, TN=0

### Abstain-Oriented GR Diagnostics

- **Abstain Precision**: 0.000
- **Abstain Recall**: 0.000
- **Abstain F1**: 0.000
- **Specificity**: 1.000
- Abstain TP=0, FP=0, FN=128, TN=608


## Per Conflict Type Breakdown

### Type 1: No Conflict

- **Samples**: 211
- **GR Accuracy**: 0.730
- **GR F1** *(used in CATS)*: 0.844
- **Behavior**: 0.000 (n=211)
- **Grounding**: 0.000 (n=211)
- **Recall**: 0.000 (n=154)
- **CATS**: 0.211

### Type 2: Complementary Info

- **Samples**: 221
- **GR Accuracy**: 0.796
- **GR F1** *(used in CATS)*: 0.887
- **Behavior**: 0.000 (n=221)
- **Grounding**: 0.002 (n=221)
- **Recall**: 0.000 (n=156)
- **CATS**: 0.222

### Type 3: Conflicting Opinions

- **Samples**: 109
- **GR Accuracy**: 0.881
- **GR F1** *(used in CATS)*: 0.937
- **Behavior**: 0.000 (n=109)
- **Grounding**: 0.000 (n=109)
- **Recall**: 0.000 (n=0)
- **CATS**: 0.312

### Type 4: Outdated Info

- **Samples**: 158
- **GR Accuracy**: 0.918
- **GR F1** *(used in CATS)*: 0.957
- **Behavior**: 0.000 (n=158)
- **Grounding**: 0.000 (n=158)
- **Recall**: 0.000 (n=140)
- **CATS**: 0.239

### Type 5: Misinformation

- **Samples**: 37
- **GR Accuracy**: 1.000
- **GR F1** *(used in CATS)*: 1.000
- **Behavior**: 0.000 (n=37)
- **Grounding**: 0.000 (n=37)
- **Recall**: 0.000 (n=37)
- **CATS**: 0.250


================================================================================

## Cost Summary

**Total Cost**: $0.0000

**Decisions Made**: 3428

**Average Cost per Decision**: $0.000000


### Per-Model Cost Breakdown

#### local/qwen3.5-397b-a17b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000

#### local/mistral-small-4

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000

#### local/deepseek-r1-distill-32b

- **Billing Note**: Cost is **unmetered in this report**. This judge ran via local CLI, and token usage was not available to the evaluator.
- **Total Cost**: $0.0000
- **Total Requests**: 0
- **Average Cost per Request**: $0.000000


## Unsupported Grounding Claims

These claims reduced the factual-grounding score and are useful for ceiling debugging.

### Sample conflictingqa_04e1627e9fc5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_04e1627e9fc5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_04e1627e9fc5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_05b33f4ca156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_05b33f4ca156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_05b33f4ca156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_060e5f26c453

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_060e5f26c453

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_060e5f26c453

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_0717d0e62f3b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_07e290a954ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_07e290a954ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_07e290a954ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_08757808c000

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_0875b5f3262a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_0a05aabca56a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_0a05aabca56a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_0a05aabca56a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_0ad05303220b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_0ad05303220b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_0ad05303220b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_0c3c7b487766

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_0c3c7b487766

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_0c3c7b487766

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_0dba017da71c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_111bef268aa6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_114c06976f62

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_114c06976f62

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_114c06976f62

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_11c5ef7c4545

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_11c5ef7c4545

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_11c5ef7c4545

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_151865dc414b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_151865dc414b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_151865dc414b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_1d741779a31e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_1d741779a31e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_1d741779a31e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_1e9c6cd8e30b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_21f33954c8af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_21f33954c8af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_21f33954c8af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_220ec09fbb2c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_220ec09fbb2c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_220ec09fbb2c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_237adb87065f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_237adb87065f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_237adb87065f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_2395695f1604

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_2395695f1604

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_2395695f1604

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_24c25ef3a801

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_24c25ef3a801

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_24c25ef3a801

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_24fa0020a521

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_24fa0020a521

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_24fa0020a521

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_2500bfd10608

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_2500bfd10608

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_2500bfd10608

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_264e97231363

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_264e97231363

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_264e97231363

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_287debe5bb90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_288cd1b45aab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_288cd1b45aab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_288cd1b45aab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_29f69e16a0c3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_29f69e16a0c3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_29f69e16a0c3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_2c0ea18839df

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_2c0ea18839df

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_2c0ea18839df

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_311fca0928d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_311fca0928d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_311fca0928d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_34610226ee3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_34610226ee3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_34610226ee3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_34fef928d452

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_35491baf4f4b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_35491baf4f4b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_35491baf4f4b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_3601f7480501

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_3601f7480501

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_3601f7480501

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_37ab7146eb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_37ab7146eb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_37ab7146eb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_37ebad668bb7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_39fe5c441657

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_39fe5c441657

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_39fe5c441657

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_3afd7f725cb4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_3bd13d25098b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_3c835387fe6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_3c835387fe6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_3c835387fe6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_3dba586dca0f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_3dba586dca0f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_3dba586dca0f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_3f3c3399259a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_3f3c3399259a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_3f3c3399259a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_411445406724

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_411445406724

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_411445406724

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_42d60ecaee9f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_42d60ecaee9f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_42d60ecaee9f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_4317242e485c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_4786f87b62be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_517b918aa677

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_517b918aa677

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_517b918aa677

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_52181cd092aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_52181cd092aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_52181cd092aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_5233eab573e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_5233eab573e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_5233eab573e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_52e01830d2fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_544ebeeccda5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_56fd6bf22253

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_56fd6bf22253

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_56fd6bf22253

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_57190bca6f7a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_5e7a6a2debfd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_613a0093714b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_613a0093714b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_613a0093714b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_62b1aff6586d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_62b1aff6586d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_62b1aff6586d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_63fde268aa8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_63fde268aa8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_63fde268aa8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_676188bf8139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_676188bf8139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_676188bf8139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_6988dd820a61

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_6d7279b0a8ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_6ea6bbcb8743

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_6fe31cd2ef65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_734859937e46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_73560dfab1ae

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_747727772a30

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_76956c2fba7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_7998e59d9b97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_7ba822a2f2fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_7cf85109a70d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_7cf85109a70d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_7cf85109a70d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_80857a692531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_80857a692531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_80857a692531

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_80baf25496cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_8848765fc18a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_8848765fc18a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_8848765fc18a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_894f4a4b9552

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_8bedefe7ac1a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_8cf8ebb94554

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_8efa53ba7c60

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_8efa53ba7c60

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_8efa53ba7c60

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_9251c7a33ec5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_9251c7a33ec5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_9251c7a33ec5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_9261438d6ee2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_9261438d6ee2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_9261438d6ee2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_9275911a2961

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_9275911a2961

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_9275911a2961

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_962d8f5d5574

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_962d8f5d5574

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_962d8f5d5574

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_9b11b8e571aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_9b11b8e571aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_9b11b8e571aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_9b73cb6cce52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_9ceca2645833

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_9ceca2645833

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_9ceca2645833

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a1e36a8db854

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a25014a5c5b5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a25014a5c5b5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a25014a5c5b5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a2f06d54b240

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a3264277980a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a3980a2921cf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a3980a2921cf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a3980a2921cf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a507c5b61631

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a7ff288bc615

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a7ff288bc615

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a7ff288bc615

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a864ff85e648

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a864ff85e648

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a864ff85e648

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a8ec1c0d2b92

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a994724a28e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_a9bed39d234d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_a9bed39d234d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_a9bed39d234d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_b2524e4883ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_b2524e4883ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_b2524e4883ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_b323dd4b5820

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_b323dd4b5820

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_b323dd4b5820

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_b7fd50f9f980

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_b7fd50f9f980

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_b7fd50f9f980

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_b9854bd5a19e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_bac0f4d62f96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_bac0f4d62f96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_bac0f4d62f96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_bd2e652cd64d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_bd2e652cd64d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_bd2e652cd64d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_bdee100fa8e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_be17259fe5c0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_be17259fe5c0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_be17259fe5c0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_bea0b68b5d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_bf6e9339f2c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_bfbbc2c7a1af

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_c1119b945459

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_c1119b945459

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_c1119b945459

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_c34991d9897e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_c34991d9897e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_c34991d9897e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_c3e66630928c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_c418fecfc1e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_c418fecfc1e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_c418fecfc1e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_c574530da7a3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_c574530da7a3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_c574530da7a3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_cc71318e5853

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_cd661c2c20b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_cd661c2c20b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_cd661c2c20b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_d295f9ea94b3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_d295f9ea94b3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_d295f9ea94b3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_d9a36fe4c135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_d9a36fe4c135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_d9a36fe4c135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_dc6f972e8447

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_dd426f7706e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_dd426f7706e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_dd426f7706e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_e2e2361dc28b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_e60bdcf75ba7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_e93e708d49a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_e93e708d49a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_e93e708d49a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_ece626a6cba9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_ece626a6cba9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_ece626a6cba9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f1932b75ace7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f1932b75ace7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f1932b75ace7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f22b389be1d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f22b389be1d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f22b389be1d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f39c966c2ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f3b163170581

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f3b163170581

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f3b163170581

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f3ba8599e370

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f3ba8599e370

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f3ba8599e370

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f43b2c51deea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f43b2c51deea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f43b2c51deea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f4693bea2c31

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f4693bea2c31

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f4693bea2c31

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f4811561af0c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f777a43ba278

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f7fec8c0688b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f7fec8c0688b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f7fec8c0688b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f8da23d84ecc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f8da23d84ecc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f8da23d84ecc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f970957c5e52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f970957c5e52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f970957c5e52

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_f97fef94decc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_f97fef94decc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_f97fef94decc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_fa98c00bd697

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_fad0d30903d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_fbedb688b1d9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_fbedb688b1d9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_fbedb688b1d9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample conflictingqa_fcdb9e210683

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_0293a11bd364

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_02b3ba89ebd0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_0436c0b3a9d7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_0436c0b3a9d7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_0436c0b3a9d7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_047057d22309

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_049cc3f14d5e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_049cc3f14d5e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_049cc3f14d5e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_1009f5c49e12

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_1009f5c49e12

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_1009f5c49e12

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_114b9082bc42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_114b9082bc42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_114b9082bc42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_122352ad92e3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_150b9ed24a07

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_1ef881d26e2e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_1fe3ba1b1cba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_2130eea851fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_2130eea851fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_2130eea851fe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_25b286cb2af1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_25b286cb2af1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_25b286cb2af1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_263eca8e024e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_263eca8e024e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_263eca8e024e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_2877cf4bd00f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_2877cf4bd00f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_2877cf4bd00f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_28e155139ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_28e155139ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_28e155139ec1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_2b9ba7e192e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_2b9ba7e192e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_2b9ba7e192e2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_2d7eb41139aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_2d7eb41139aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_2d7eb41139aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_2e51f51132ee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_31ad09b9cd22

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_31ad09b9cd22

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_31ad09b9cd22

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_3227ea6c6056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_354e7097602d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_35bf342002aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_35bf342002aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_35bf342002aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_3847b5cb9b42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_39dcd7b38c39

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_39dcd7b38c39

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_39dcd7b38c39

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_3ad16f379533

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_3cd0b514193b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_3cd0b514193b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_3cd0b514193b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_3dc3cf00bce6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_3dc3cf00bce6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_3dc3cf00bce6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_42796b35e143

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_42796b35e143

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_42796b35e143

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_4590bdd9e269

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_4a98eba95e97

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_4d9a80505e01

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_4e635a2542a8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_4e635a2542a8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_4e635a2542a8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_50f8f03fd30f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_50f8f03fd30f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_50f8f03fd30f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_5574b1447bdb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_5574b1447bdb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_5574b1447bdb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_5d6e5db69928

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_5d6e5db69928

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_5d6e5db69928

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_5eb89aae15f3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_5eb89aae15f3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_5eb89aae15f3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_5ecee1c55713

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_5ecee1c55713

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_5ecee1c55713

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_64c12116affc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_64c12116affc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_64c12116affc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_6927f007d7cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_6a45fadeb16b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_6a45fadeb16b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_6a45fadeb16b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_6f42c128eb6c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_7bc7bb2dde20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_7bc92b47dc43

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_7bc92b47dc43

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_7bc92b47dc43

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_7dce5d575302

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_7e63fcff2dea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_7f1c3aae61a5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_80642f637dc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_80642f637dc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_80642f637dc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_8ab63ffc9a7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_8ab63ffc9a7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_8ab63ffc9a7d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_8eca5bd62ae0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_8eca5bd62ae0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_8eca5bd62ae0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_8f302f0bfe82

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_97f3c1fe1fd4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_97f3c1fe1fd4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_97f3c1fe1fd4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_a41257e9d6f6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_a47283064972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_a50d0f1f3cdf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_a5492f36ca23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_a5492f36ca23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_a5492f36ca23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_a8b908895e11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_a8b908895e11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_a8b908895e11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_ab11b5dce00e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_ab11b5dce00e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_ab11b5dce00e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_b0ffe73c1789

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_b3264b37f54b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_b3264b37f54b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_b3264b37f54b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_b99c189f2222

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_c3f10dc1632d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_c3f10dc1632d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_c3f10dc1632d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_c479e83e408f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_c4976f8629cb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_c7315f8b3029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_c7315f8b3029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_c7315f8b3029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_c7ac9d61059a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_c7ac9d61059a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_c7ac9d61059a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_cbfca321cce4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_cbfca321cce4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_cbfca321cce4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_cf331ed7d09f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_cf331ed7d09f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_cf331ed7d09f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_d4d59d75b206

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_d510972df578

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_dd85dcbc2262

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_dd85dcbc2262

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_dd85dcbc2262

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_dd87e1e3ad3d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_dd87e1e3ad3d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_dd87e1e3ad3d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_ddd643091cbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_ddd643091cbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_ddd643091cbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_e1a40e76ba15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_e502143179d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_e502143179d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_e502143179d6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_edf4ae4f32e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_edf4ae4f32e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_edf4ae4f32e5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_ef3ad40c6540

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_ef3ad40c6540

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_ef3ad40c6540

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_f11e8d758fc6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_f11e8d758fc6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_f11e8d758fc6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_f5d8e53958c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_f5d8e53958c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_f5d8e53958c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_f5eef5c415e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_f5eef5c415e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_f5eef5c415e8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_f6ac249bdf53

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_f6ac249bdf53

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_f6ac249bdf53

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_f6cc6071caa5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_fc07d4525dbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_fd00b29e848c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_fd00b29e848c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_fd00b29e848c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample freshqa_ff04793833cc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample healthcontradict_0060

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample healthcontradict_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample healthcontradict_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample healthcontradict_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0031

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d10
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0031

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0031

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0062

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0062

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0062

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0064

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0073

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0083

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample hotpotqa_0196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample hotpotqa_0196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample hotpotqa_0196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample misinformation_0023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample misinformation_0023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample misinformation_0023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0091

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0091

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0091

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_08cf866bcb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_08cf866bcb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_08cf866bcb9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0a580da7f2cd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0ac549afb037

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0ac549afb037

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0ac549afb037

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0b3b6452e224

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0b75ed799d46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0b75ed799d46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0b75ed799d46

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0bd7153f19ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0bd7153f19ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0bd7153f19ad

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_0d85f1089c4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_0d85f1089c4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_0d85f1089c4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_1025b0681710

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_1025b0681710

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_1025b0681710

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_132167e66120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_132167e66120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_132167e66120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_15ffab2466f7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_15ffab2466f7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_15ffab2466f7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_160a528ae07e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_160a528ae07e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_160a528ae07e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_17dc360eea55

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_19ca08790764

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_19ca08790764

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_19ca08790764

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_1a764b8b6cf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_1a764b8b6cf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_1a764b8b6cf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_1b95727cc286

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_213701765f94

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_213701765f94

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_213701765f94

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_2243f17ccc38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_2243f17ccc38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_2243f17ccc38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_252987b8054c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_252987b8054c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_252987b8054c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_287da9f37864

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_287da9f37864

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_287da9f37864

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_290c939ed6e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_290c939ed6e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_290c939ed6e7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_292033e4b039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_292033e4b039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_292033e4b039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_2a7f7e06e365

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_2cbc9a53426f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_2cbc9a53426f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_2cbc9a53426f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_2e1b5edb5e0d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_2e1b5edb5e0d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_2e1b5edb5e0d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_2ed872eb1114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_2f6d2647a424

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_2f6d2647a424

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_2f6d2647a424

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_34cba3c71e06

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_34cba3c71e06

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_34cba3c71e06

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_367b09e4ed80

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_367b09e4ed80

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_367b09e4ed80

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_37fdedfe4478

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_3c1297608017

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_3d4ebfa8b6dd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_403a59870dc2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_403a59870dc2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_403a59870dc2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_41c44ecfd0f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_4387048ed24f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_4387048ed24f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_4387048ed24f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_44b315f6f4bb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_44b315f6f4bb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_44b315f6f4bb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_4fb90d57c274

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_4fb90d57c274

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_4fb90d57c274

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_51b23ea15977

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_51b23ea15977

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_51b23ea15977

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_51c89636151e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_51c89636151e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_51c89636151e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_531aff489b71

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_531aff489b71

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_531aff489b71

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_54be882d5b58

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_54be882d5b58

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_54be882d5b58

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_5a9576fc5d8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_5a9576fc5d8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_5a9576fc5d8c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_5eeb46719843

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_5fb5c311d373

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_5fb5c311d373

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_5fb5c311d373

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_6485f021b694

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_6485f021b694

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_6485f021b694

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_66ba2af9c3b9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_66ba2af9c3b9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_66ba2af9c3b9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_67b35f41ba84

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_6837d86d03ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_6837d86d03ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_6837d86d03ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_6969589d80c1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_6969589d80c1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_6969589d80c1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_6af6e8cb8f34

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_6b3b372cf27d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_6b3b372cf27d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_6b3b372cf27d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_6edf1477bd7e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_6edf1477bd7e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_6edf1477bd7e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_6f8df54650a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_720d8b210357

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_7916ffefdb99

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_7916ffefdb99

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_7916ffefdb99

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_798b6853d20f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_798b6853d20f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_798b6853d20f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_7bf02a7deb69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_7bf02a7deb69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_7bf02a7deb69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_7df263780268

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_7df263780268

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_7df263780268

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_7f5e5a4a4391

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_7f5e5a4a4391

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_7f5e5a4a4391

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_883303a2d535

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_883303a2d535

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_883303a2d535

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_8882ab46be5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_8882ab46be5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_8882ab46be5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_899648874637

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_899648874637

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_899648874637

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_8d7c14ed548f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_8d7c14ed548f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_8d7c14ed548f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_8daf80e943fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_8daf80e943fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_8daf80e943fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_8dd2323c077d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_8ef7b3cf5c3f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_9404250d756f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_9404250d756f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_9404250d756f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_940e6d9275f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_940e6d9275f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_940e6d9275f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_946ecfb478b8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_946ecfb478b8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_946ecfb478b8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_950881e7c998

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_9a9a28d7e159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_9b16fd6882f3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_9b16fd6882f3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_9b16fd6882f3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_9c2f95b14a78

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_9c2f95b14a78

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_9c2f95b14a78

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_9fbf28f5786f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a1c73439eca9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a3c882e062c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a3c882e062c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a3c882e062c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a44267c115d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a4dde83c35da

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a4dde83c35da

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a4dde83c35da

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a635c2fd4869

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a635c2fd4869

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a635c2fd4869

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a6a2f8b1f0b8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a6a2f8b1f0b8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a6a2f8b1f0b8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a6b48b7accc4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a6b48b7accc4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a6b48b7accc4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a6df0af8c2ba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a6df0af8c2ba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a6df0af8c2ba

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a78a32b7b9a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a78a32b7b9a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a78a32b7b9a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a91ae87c969d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a91ae87c969d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a91ae87c969d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_a927c4cccc6a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_a927c4cccc6a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_a927c4cccc6a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_aa94588b9477

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_aa94588b9477

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_aa94588b9477

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_aaf0f638e99b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_aaf0f638e99b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_aaf0f638e99b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ac5341df9a83

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ac5341df9a83

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ac5341df9a83

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_b0346f60b6ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_b0ee06f2950d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_b0ee06f2950d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_b0ee06f2950d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_b198a514fff8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_b1bd9518429b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_b1bd9518429b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_b1bd9518429b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_b281f09f0959

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_b281f09f0959

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_b281f09f0959

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ba7aaa9b36c8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_bc34664caee4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_bc34664caee4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_bc34664caee4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_bc7e9a7b4a83

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_bc7e9a7b4a83

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_bc7e9a7b4a83

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_bf0bff050f03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_bf0bff050f03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_bf0bff050f03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_bfbb5f55a63f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c264cb69676e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c27400199055

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c27400199055

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c27400199055

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c2975d69d57c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c675e6cd8ad6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c69855566c76

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d1, d3, d5
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c69855566c76

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c69855566c76

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c731579bb51c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c731579bb51c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c731579bb51c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c88807a22775

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c88807a22775

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c88807a22775

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_c9b95dd57e73

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_c9b95dd57e73

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_c9b95dd57e73

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_cb5bcdb1ef9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_cb5bcdb1ef9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_cb5bcdb1ef9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_cbddef47777e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_cbddef47777e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_cbddef47777e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_cbf25273f973

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ce4983c8a9c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ce4983c8a9c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ce4983c8a9c5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d00b0063e747

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d00b0063e747

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d00b0063e747

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d03e85bdc95a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d03e85bdc95a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d03e85bdc95a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d39801b5de65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d39801b5de65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d39801b5de65

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d3b85d857358

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d44802dc3c96

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d60bf850c4ff

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d60bf850c4ff

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d60bf850c4ff

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d78d45c0e30f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d78d45c0e30f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d78d45c0e30f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d7c6682b5335

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d7df0a1856b7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d8b24beb2f90

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d96b47272030

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d96b47272030

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d96b47272030

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_d9b756cb0eea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_d9b756cb0eea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_d9b756cb0eea

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_e064a7a717ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_e064a7a717ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_e064a7a717ed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_e06ada156e0e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_e06ada156e0e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_e06ada156e0e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_e326d0094f42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_e326d0094f42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_e326d0094f42

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_e6d89fce1b8e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_e6d89fce1b8e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_e6d89fce1b8e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_e7318f6f3bbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_e7318f6f3bbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_e7318f6f3bbe

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_e87ffc07efd1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ea469c846404

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_eb6f14795c45

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_eb6f14795c45

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_eb6f14795c45

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_eb7c676e133e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_eb7c676e133e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_eb7c676e133e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ec5b0067c29a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ec5b0067c29a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ec5b0067c29a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ecbc6adf8a48

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ecbc6adf8a48

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ecbc6adf8a48

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ecd3d9c0ca11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ecd3d9c0ca11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ecd3d9c0ca11

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_f10c7ad4bb81

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_f10c7ad4bb81

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_f10c7ad4bb81

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_f1776add7672

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_f1776add7672

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_f1776add7672

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_f2218f8c979e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_f2218f8c979e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_f2218f8c979e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_f69c37496013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_f69c37496013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_f69c37496013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_fbdae168fc6f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_fbe562911999

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_fbe562911999

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_fbe562911999

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample qacc_ff2cb00f4c03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample qacc_ff2cb00f4c03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample qacc_ff2cb00f4c03

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_00f6ee9a6705

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_082db791e263

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_09a6a048cfbf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_0c9730c6cd25

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_2150ef9a29c6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_32318bdc3cf9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_362420ec2416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_362420ec2416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_362420ec2416

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_3815eef77a3c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_48e19f6b37bc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_4cb699778b59

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_4cb699778b59

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_4cb699778b59

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_5f5c20228969

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_66684169f016

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_66684169f016

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_66684169f016

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_6cca7fdb9b41

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_7222d6123c27

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_7222d6123c27

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_7222d6123c27

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_74e7e677cae5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_779fd84224fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_779fd84224fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_779fd84224fa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_7f4d90dffb38

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_80c72c2797bf

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_85af31651715

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_85af31651715

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_85af31651715

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_864fc7e36714

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_897e47478bbc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_897e47478bbc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_897e47478bbc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_8c889f8ce07a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_932ca9fa3259

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_9f6c5f849b40

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_a1cde1787dab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_adcb94e5d70e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_c68eb66efad5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_c76a815bb275

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_d0532ee6a455

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_d982055a66d8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_eec0ebc0dd88

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_f2031e426cee

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_f26078ec6467

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_f26078ec6467

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_f26078ec6467

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_f2a7d027353d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_f53ff7fb024c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_f75bc1a9cad8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_f7c719d9b0be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_geo_fd48e977c2d0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_051502801f9c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_051502801f9c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_051502801f9c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_05d714be23fd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_05d714be23fd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_05d714be23fd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_0ad0871fcc33

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_0ad0871fcc33

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_0ad0871fcc33

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_0c2289f57504

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_0c2289f57504

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_0c2289f57504

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_14a587def215

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_14a587def215

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_14a587def215

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_14f29fe7ff15

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_14f70522567e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_14f70522567e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_14f70522567e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_180f238d8296

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_1987d35f994b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_1987d35f994b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_1987d35f994b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_19badef7553b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_1baff64de20e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_1baff64de20e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_1baff64de20e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_1c56e575f096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_1d8ddaf99c95

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_23d05dc2d7dc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_2cafa66bbe5d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_2f45ec399f17

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_2f5dc02a5ce1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_2f73dc6c2df1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_301378915064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_301378915064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_301378915064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_3026b0491e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_3026b0491e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_3026b0491e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_32d33d503f69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_32d33d503f69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_32d33d503f69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_35156c8be377

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_35156c8be377

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_35156c8be377

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_3521e5dc831c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_3781ac7b3ead

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_3df0e6082901

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_40e6764f611f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_40e6764f611f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_40e6764f611f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_43c0aaae9828

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_43c0aaae9828

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_43c0aaae9828

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_44d30ce08bf5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_45cbc4dc3e28

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d3, d5
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_4ead2f4cd2d5

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_4fbe76f974ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_50748f92be3a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_50748f92be3a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_50748f92be3a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_557d04a28d7f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_587e89bbcbe1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_5a59faf24972

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_5e6b388e88aa

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_60095fdf39a4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_603f0dc417ac

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_61a79d74d827

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_61d9095f5b70

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_61d9095f5b70

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_61d9095f5b70

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_657c130afab6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_657c130afab6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_657c130afab6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_6a683d0d04f0

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_6a8aed3e3d5f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_6c424fc78a69

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_6f777dda5314

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_6f777dda5314

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_6f777dda5314

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_78c89bf81e8b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_7a7a7f43e575

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_7bb7fe6c9287

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_7cd18101326e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_7d5f559b313d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_7dd0bea41e4a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_7ee5807518be

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_7f07f7a6c607

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_8808c106a115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_8808c106a115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_8808c106a115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_88ef1cfab62e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_8b66ca78c3d1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_8b8427cd7b6d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_8f57822ed7d2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_8fbdda192a13

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_901be1437bc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_901be1437bc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_901be1437bc7

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_9049c197a579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_956e1f6e518e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_9a91e43d8136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_a2624f6c031f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_a3a59c3f4b9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_a3a59c3f4b9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_a3a59c3f4b9b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_a403222d0ab8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_a5e5db28902b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_a8024d95c6e1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_a8024d95c6e1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_a8024d95c6e1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_a8c74bced99a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_ac0e6a4a7e32

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_ac7a73d20fdd

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_ae0882e48812

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_ae0882e48812

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_ae0882e48812

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_b3ad3248b7d4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_b6c75b99a2f4

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_b797de4c6610

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_b797de4c6610

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_b797de4c6610

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_bb7a678b9ec9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_bb85e25a8159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_bc0542b3c97d

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_bdc0853f9a23

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_c02b89bc9dc1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_c92bc2a37a57

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_ce10e6f883fb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_ce10e6f883fb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_ce10e6f883fb

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_cfe45b0c90b6

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_d0579ca3907c

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_d15849658c20

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_d1bfc673775e

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_d59edcfab87f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_d63c04b198fc

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_da5762edf2c2

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_dc4d0c4e24ab

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_dcdb77c2f3d9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_dcdb77c2f3d9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_dcdb77c2f3d9

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_df5975a9678a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_df5975a9678a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_df5975a9678a

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_e3caa7edab1f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_e3caa7edab1f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_e3caa7edab1f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_ebd3c33e8be8

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_eed796e2fbed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_eed796e2fbed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_eed796e2fbed

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_eeebc5ea5c4f

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_efa79b1e7fb1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_f099d9d1452b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_f196a847a496

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_f196a847a496

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_f196a847a496

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_f1e9ce4608b3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_f1e9ce4608b3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_f1e9ce4608b3

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_f3b3face1b6b

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_f7d6e484b579

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample situatedqa_temp_f971e49123a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample situatedqa_temp_f971e49123a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample situatedqa_temp_f971e49123a1

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_002

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_002

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_002

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_003

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_003

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_003

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_008

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_008

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_008

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_009

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_009

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_009

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_012

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_012

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_012

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_015

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_016

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_016

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_016

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_018

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_018

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_018

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_019

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_019

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_019

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_022

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_022

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_022

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_023

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_026

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_026

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_026

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_029

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_033

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_033

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_033

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_034

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_034

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_034

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_038

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_038

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_038

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_039

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_040

- **Reason**: no_eligible_docs
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_040

- **Reason**: no_eligible_docs
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_040

- **Reason**: no_eligible_docs
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_043

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_043

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_043

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_045

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_045

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_045

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_048

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_048

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_048

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_050

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_050

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_050

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_052

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_052

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_052

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_056

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_058

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_058

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_058

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_059

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_059

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_059

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_062

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_062

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_062

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_063

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_068

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_068

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_068

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_070

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_078

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_078

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_078

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_079

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_080

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_080

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_080

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_081

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_081

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_081

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_084

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_084

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_084

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_086

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_086

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_086

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_087

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_087

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_087

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_089

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_089

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_089

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_091

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_091

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_091

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_095

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_101

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_101

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_101

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_105

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_105

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_105

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_106

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_106

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_106

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_107

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_107

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_107

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_110

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_110

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_110

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_113

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_113

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_113

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_114

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_116

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_116

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_116

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_117

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_117

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_117

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_118

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_118

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_118

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_122

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_122

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_122

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_124

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_124

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_124

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_129

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_129

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_129

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_130

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_130

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_130

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_132

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_132

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_132

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_136

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_137

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_137

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_137

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_139

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_140

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_140

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_140

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_143

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_143

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_143

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_144

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_144

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_144

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_146

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_146

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_146

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_148

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_148

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_148

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_151

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_151

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_151

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_158

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_158

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_158

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_159

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_163

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_163

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_163

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_166

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_166

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_166

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_167

- **Reason**: supporting_doc_not_cited
- **Cited Docs**: d2
- **Supporting Docs Found**: d5
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_168

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_168

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_168

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_169

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_169

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_169

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_171

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_171

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_171

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_173

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_173

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_173

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_174

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_174

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_174

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_175

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_178

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_178

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_178

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_180

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_180

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_180

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_181

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_181

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_181

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_183

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_183

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_183

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_185

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_185

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_185

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_186

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_187

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_187

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_187

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_188

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_188

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_188

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_189

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_189

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_189

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_191

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_191

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_191

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_192

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_193

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_193

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_193

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_194

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_194

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_194

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample trust_align_196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample trust_align_196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample trust_align_196

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0001

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0001

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0001

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0002

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0002

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0002

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0003

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0003

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0003

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0004

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0007

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0007

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0007

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0010

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0010

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0010

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0013

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0025

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0028

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0032

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0033

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0033

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0033

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0035

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0040

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0040

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0040

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0041

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0042

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0046

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0047

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0049

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0049

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0049

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0057

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0057

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0057

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0061

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0061

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0061

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0064

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0065

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0065

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0065

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0066

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0066

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0066

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0067

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0071

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0072

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0074

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0076

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0082

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0082

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0082

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0085

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0086

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0086

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0086

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0088

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0088

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0088

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0089

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0089

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0089

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0090

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0093

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0093

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0093

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0096

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0097

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0097

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0097

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0099

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0100

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0103

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0104

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0105

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0105

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0105

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0109

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0111

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0111

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0111

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0112

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0115

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0119

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0120

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0121

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0123

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0124

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0124

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0124

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0125

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0129

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0129

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0129

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0132

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0132

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0132

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0134

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0135

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0137

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0137

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0137

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0141

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0142

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0145

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0149

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0150

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0151

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0151

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0151

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0152

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0153

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0154

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0155

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0156

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0157

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0160

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0161

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0161

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0161

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0162

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0165

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0166

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0166

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0166

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0167

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0170

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0171

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0171

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0171

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d2
- **Supporting Docs Found**: None
- **Claim**: Dr. Rochelle Walensky served as the Director of the United States Centers for Disease Control and Prevention (CDC) from January 2021 to June 2023

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: Dr. Robert Redfield was the CDC Director from March 2018 to January 2021 , but his tenure ended before Dr. Walensky's began

### Sample wikirevision_0172

- **Reason**: no_supporting_doc_found
- **Cited Docs**: d1
- **Supporting Docs Found**: None
- **Claim**: The CDC website's information about Dr. Redfield's tenure is outdated


================================================================================

*Report generated by CATS v2.0*
