# 7B Run F Boundary-Guarded Manual Inspection

Date: 2026-05-27

Scope:

- `sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_strict_val_stagewise`
- `sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_trace_text_val_stagewise`
- `sft_qwen25_stagewise_main_trace_text_f_boundary_guarded_csis_e2e_minimal_val_stagewise`

Joined audit artifact:

- `outputs/analysis/def_7b_prompt_robust_audit.csv`
- `outputs/analysis/def_7b_prompt_robust_audit.jsonl`

## Executive Read

7B F is much healthier than 7B E. E's broad conflict calibration collapsed 7B conflict accuracy, while F mostly preserves D-like minimal behavior and substantially improves strict conflict/doc behavior.

However, 7B F minimal is not a clean win over D minimal because it has one malformed/misaligned trace row and one abstain miss. This appears isolated rather than systemic, but it matters because minimal-prompt trace reliability is one of our core research gates.

## Aggregate D/E/F Comparison

Strict:

```text
D strict: conflict=66.67, doc_micro=74.22, contract_adj=75.5, think=49/49
E strict: conflict=61.22, doc_micro=76.98, contract_adj=71.4, think=49/49
F strict: conflict=77.55, doc_micro=79.43, contract_adj=71.4, think=49/49
```

Runtime:

```text
D runtime: conflict=68.75, doc_micro=75.79, contract_adj=73.5, think=48/49
E runtime: conflict=57.14, doc_micro=76.98, contract_adj=81.6, think=49/49
F runtime: conflict=71.43, doc_micro=77.12, contract_adj=71.4, think=49/49
```

Minimal:

```text
D minimal: conflict=73.47, doc_micro=76.98, contract_adj=79.6, think=49/49
E minimal: conflict=58.33, doc_micro=78.52, contract_adj=85.7, think=49/49
F minimal: conflict=72.92, doc_micro=76.94, contract_adj=77.6, think=48/49
```

Interpretation:

- F is a major recovery from E for 7B.
- F strict is the best 7B strict run so far.
- F runtime is better than D/E on conflict and doc verdicts, though contract-adjusted is below E.
- F minimal is close to D minimal, but D minimal remains cleaner because F has one malformed row.

## F Conflict Behavior

F strict conflict confusion:

```text
No conflict:
  correct 17/19
  -> Complementary information: 1
  -> Conflict due to outdated information: 1

Complementary information:
  correct 8/15
  -> No conflict: 5
  -> Conflict due to outdated information: 2

Conflicting opinions or research outcomes:
  correct 8/10
  -> No conflict: 1
  -> Complementary information: 1

Conflict due to outdated information:
  correct 5/5
```

F minimal conflict confusion:

```text
No conflict:
  correct 16/19
  -> Complementary information: 1
  -> Conflict due to outdated information: 1

Complementary information:
  correct 7/15
  -> No conflict: 7
  -> Conflict due to outdated information: 1

Conflicting opinions or research outcomes:
  correct 8/10
  -> No conflict: 1
  -> Complementary information: 1

Conflict due to outdated information:
  correct 4/5
  -> No conflict: 1
```

The remaining 7B F weakness is now mostly complementary recall. F predicts `No conflict` too often for some gold-complementary abstain/partial-evidence rows.

## F vs D/E Row-Level Deltas

F strict improves over D on conflict for:

```text
#0015
#0263
#0373
#0638
#0542
#0531
```

F strict has no conflict regressions relative to D.

F minimal improves over D on conflict for:

```text
#0015
#0159
#0373
```

F minimal regresses relative to D on:

```text
#0333
#0427
#0588
#0603
```

F minimal improves over E on conflict for:

```text
#0300
#0203
#0015
#0373
#0517
#0561
#0470
```

F minimal has no conflict regressions relative to E.

## Manual Row Notes

### #0531 Socialism vs Communism

Gold conflict: `No conflict`; gold abstain: true.

F strict and runtime are good: both classify `No conflict` and abstain. F minimal is malformed. The output opens a `<think>` block but never cleanly closes it; the row includes an instruction-like Chinese phrase from a source snippet and then produces a Chinese refusal. Evaluators mark `think_block_missing_or_misaligned`, `PRED_MISSING`, and abstain mismatch.

This is likely a prompt-injection/source-contamination robustness failure rather than a broad F strategy problem, because strict/runtime handle the same row correctly.

### #0015 American Idol Winner

Gold conflict: `No conflict`.

F fixes D/E's over-outdated behavior. D and E treat previous-season winners as outdated conflict. F correctly says relevant documents consistently identify Abi Carter as the current/recent winner, while older winners are contextual past-season references.

### #0159 Heated Gemstones

Gold conflict: `Complementary information`.

F fixes D's over-conflict error. It frames the evidence as different gemstone/treatment/value scopes rather than direct contradiction.

### #0373 Declaration of Independence Signers

Gold conflict: `No conflict`.

F fixes D/E minimal by recognizing that counts, dates, and signer subsets all support one unified answer rather than requiring `Complementary information`.

### #0333 Supreme Court Appointment

Gold conflict: `Conflict due to outdated information`.

F minimal regresses relative to D. It smooths Barrett-era and Jackson-era evidence as contextual rather than treating older Barrett snapshots as outdated. This remains a temporal-boundary weakness.

### #0427 AUV Meaning in Cars

Gold conflict: `Complementary information`.

F minimal predicts `No conflict` and has poor doc verdict accuracy. It likely over-focuses on the automotive answer and ignores the cross-domain acronym ambiguity that gold treats as complementary.

### #0588 / #0603 Partial-Evidence Abstain Rows

F minimal tends to collapse some gold-complementary abstain rows to `No conflict`. The final abstain behavior is usually correct, but Stage 2 taxonomy loses the complementary partial-evidence distinction.

### #0392 FIBA Ranking

Gold conflict: `No conflict`, but F predicts outdated information, same broad failure family as earlier ranking/current-year rows. It treats ranking snapshots as temporal conflict even when gold says the current answer is not materially conflicted.

## Structural Notes

Good:

- F strict and runtime both have `think=49/49`, `sentinel=49/49`, and final abstain accuracy `100.0`.
- F strict is clearly the best 7B strict run by conflict and doc verdict.
- F runtime is a good recovery from E and better than D on parsed conflict count and doc verdict.
- F did not repeat E's broad 7B conflict collapse.

Concerns:

- F minimal has `think=48/49`, caused by `#0531`.
- F minimal final abstain accuracy is `97.96`, also because of `#0531`.
- F minimal is not clearly better than D minimal.
- Complementary recall remains fragile.
- Temporal conflict boundary is still inconsistent: sometimes over-detects outdated conflict, sometimes misses it.

## Judgment

For 7B, D minimal remains the cleanest minimal-prompt internalization run because it has perfect trace structure and slightly higher conflict accuracy. F is still valuable because it gives the best strict/runtime 7B behavior and confirms that the boundary-drill idea is much safer than E's broad oversampling.

Current 7B preference:

```text
Minimal internalization proof: D
Strict/runtime conflict and doc quality: F
Avoid for 7B: E
```

Run G is still justified for 32B because F's 32B weakness was doc-verdict over-partialization. But if we later run G on 7B, the acceptance gate must be strict about minimal `think=49/49` and no source-instruction leakage like `#0531`.

