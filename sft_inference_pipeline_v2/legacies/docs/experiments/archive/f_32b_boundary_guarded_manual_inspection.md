# 32B Run F Boundary-Guarded Manual Inspection

Date: 2026-05-27

Scope:

- `sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_strict_val_stagewise`
- `sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_trace_text_val_stagewise`
- `sft_qwen25_32b_stagewise_main_trace_text_f_boundary_guarded_e2e_minimal_val_stagewise`

Joined audit artifact:

- `outputs/analysis/def_32b_prompt_robust_audit.csv`
- `outputs/analysis/def_32b_prompt_robust_audit.jsonl`

## Executive Read

F on 32B did what it was designed to do: it improved conflict taxonomy and contract/citation behavior while preserving minimal-prompt trace emergence. The tradeoff is real: doc-verdict accuracy dropped, mostly because the model became more conservative and over-used `partially supports`.

F is therefore a successful conflict/format experiment, but not an unqualified replacement for D/E unless we accept the doc-verdict cost.

## Aggregate D/E/F Comparison

Strict:

```text
D strict: conflict=31/49, doc_micro=85.42, contract_adj=69.4
E strict: conflict=36/49, doc_micro=85.42, contract_adj=75.5
F strict: conflict=38/49, doc_micro=80.56, contract_adj=85.7
```

Runtime:

```text
D runtime: conflict=34/49, doc_micro=85.68, contract_adj=77.6
E runtime: conflict=36/49, doc_micro=84.65, contract_adj=69.4
F runtime: conflict=36/48 parsed, doc_micro=79.54, contract_adj=77.6
```

Minimal:

```text
D minimal: conflict=31/49, doc_micro=86.45, contract_adj=69.4
E minimal: conflict=35/49, doc_micro=82.35, contract_adj=77.6
F minimal: conflict=36/49, doc_micro=81.07, contract_adj=81.6
```

All F profiles have:

```text
think=49/49
sentinel=49/49
final abstain accuracy=100.0
invalid citations=0
```

## F Conflict Behavior

F minimal confusion matrix:

```text
No conflict:
  correct 11/19
  -> Complementary information: 6
  -> Conflict due to outdated information: 2

Complementary information:
  correct 13/15
  -> No conflict: 2

Conflicting opinions or research outcomes:
  correct 8/10
  -> Complementary information: 2

Conflict due to outdated information:
  correct 4/5
  -> No conflict: 1
```

Compared with D minimal, F minimal fixed these conflict rows:

```text
#0187
#0159
#0263
#0381
#0592
```

F minimal did not introduce any new conflict regression relative to D minimal.

Compared with E minimal, F minimal improved:

```text
#0187
#0517
#0644
#0603
```

Compared with E minimal, F minimal regressed:

```text
#0333
#0470
#0531
```

## F Doc Verdict Behavior

The main regression is doc labels. F often moves gold `supports` or `irrelevant` into `partially supports`.

F minimal doc confusion:

```text
gold supports:
  supports=140
  partially supports=37
  irrelevant=0

gold partially supports:
  supports=16
  partially supports=138
  irrelevant=4

gold irrelevant:
  supports=0
  partially supports=17
  irrelevant=39
```

Worst F minimal doc regressions relative to D minimal:

```text
#0427: D doc_acc=0.8750 -> F doc_acc=0.1250
#0206: D doc_acc=0.8182 -> F doc_acc=0.4545
#0399: D doc_acc=0.7143 -> F doc_acc=0.4286
#0104: D doc_acc=0.8889 -> F doc_acc=0.6667
#0301: D doc_acc=0.7778 -> F doc_acc=0.5556
#0190: D doc_acc=1.0000 -> F doc_acc=0.7778
#0654: D doc_acc=0.8000 -> F doc_acc=0.6000
#0542: D doc_acc=0.8000 -> F doc_acc=0.6000
```

Best F minimal doc improvements relative to D minimal:

```text
#0517: D doc_acc=0.8000 -> F doc_acc=1.0000
#0592: D doc_acc=0.8000 -> F doc_acc=1.0000
#0650: D doc_acc=0.8000 -> F doc_acc=1.0000
#0373: D doc_acc=0.6667 -> F doc_acc=0.7778
#0127: D doc_acc=0.8889 -> F doc_acc=1.0000
#0203: D doc_acc=0.7000 -> F doc_acc=0.8000
```

Interpretation: boundary drilling helped Stage 2 labels but appears to have made Stage 1 more nuanced/hedged, especially marking relevant-but-direct docs as partial.

## Manual Row Notes

### #0159 Heated Gemstones

Gold conflict: `Complementary information`.

F fixes D's over-conflict behavior. D minimal called this `Conflicting opinions or research outcomes`; F calls it `Complementary information`, because the docs are mostly scoped by gemstone type, treatment type, and value context. This is exactly the desired boundary-drill effect.

### #0263 Public Transportation vs Driving

Gold conflict: `Complementary information`.

F fixes D's `Conflicting opinions` prediction. F correctly frames the evidence as broad U.S. driving-is-faster trends plus city/infrastructure exceptions, not direct contradiction.

### #0381 World Population

Gold conflict: `Conflict due to outdated information`.

F fixes D's `No conflict` prediction. F recognizes that 2022 ~8B population figures are superseded by a 2025 ~8.2B figure. This is a good sign for temporal-boundary behavior.

### #0592 2014 Commonwealth Games Gold Medals

Gold conflict: `Complementary information`; gold abstain: true.

F correctly abstains and classifies as complementary. D had `No conflict`. The F explanation recognizes that country/team gold counts and hockey gold context are partial, non-overlapping evidence that still cannot answer the individual/entity query.

### #0333 Last Person Appointed to U.S. Supreme Court

Gold conflict: `Conflict due to outdated information`.

F regresses relative to E. F calls this `No conflict`, saying Barrett and Jackson are different contextual scopes. E correctly recognized Barrett-era documents as outdated relative to Jackson. This is a remaining temporal-boundary weakness.

### #0394 This Year's Super Bowl Host Stadium

Gold conflict: `Conflict due to outdated information`.

F minimal is correct, but F strict/runtime call it `No conflict`. The strict/runtime generations smooth over year-specific differences rather than treating annual "this year" updates as temporal conflict. This is another remaining temporal-boundary weakness.

### #0343 Spider-Man Organic Web Shooters

Gold conflict: `No conflict`.

F strict is correct. F minimal predicts `Complementary information`. F runtime is malformed for Stage 2: the generation stops inside Stage 1 after a source snippet containing an instruction-like Chinese phrase, then jumps to the final answer. This is a data/prompt-injection-style robustness issue, not a general trace failure. The final answer itself is semantically good.

### #0373 Declaration of Independence Signers

Gold conflict: `No conflict`.

F continues the D/E error: it predicts `Complementary information`, treating count/date/name details as separate facets. This is the classic over-complementary failure: distinct facets that all support one unified answer should remain `No conflict`.

### #0416 Word of Wisdom Mandatory

Gold conflict: `Conflicting opinions or research outcomes`.

F continues the D/E error: it predicts `Complementary information`, treating 1851 commandment/covenant evidence and 1915-1919 temple-recommend evidence as contextual scopes. The gold label considers these competing interpretations of "mandatory." This remains a hard taxonomy boundary.

### #0654 Definition of Gravity

Gold conflict: `Conflicting opinions or research outcomes`; gold abstain: true.

F correctly abstains but predicts `Complementary information`. It treats speculative gravity explanations and basic definitions as facets rather than conflict. Also doc verdicts drift: F marks too many speculative/irrelevant snippets as partially supporting.

## Quality Notes

Good:

- Minimal trace emergence remains fully intact.
- F has the best 32B strict conflict accuracy so far.
- F has the best 32B minimal conflict accuracy so far.
- F has improved adjusted contract rates and citation coverage.
- Final abstain behavior is clean across all F profiles.

Concerns:

- Doc verdict accuracy drops materially versus D and E.
- `partially supports` is over-used.
- Runtime has one parse failure: `#0343` missing Stage 2 conflict line after an instruction-like source snippet.
- Several remaining conflict errors are not random; they are stable boundary problems:
  - `#0127`: gold complementary, predicted no conflict.
  - `#0333`: gold outdated, predicted no conflict.
  - `#0373`: gold no conflict, predicted complementary.
  - `#0416`: gold conflicting opinions, predicted complementary.
  - `#0654`: gold conflicting opinions, predicted complementary.

## Current Judgment

For 32B alone, F is promising but not obviously final.

If we prioritize conflict taxonomy and contract robustness, F is the best 32B variant so far. If we prioritize doc-verdict accuracy, D or E remains safer. The final SFT decision should wait for CSIS 7B F because E was model-size sensitive and F could still behave differently on 7B.

Likely next lesson for a possible G run:

- Keep F's boundary idea, but add a doc-verdict stabilizer to reduce `partially supports` overuse.
- Specifically train sharper Stage 1 boundaries:
  - `supports` when the snippet directly answers even if wording is not exhaustive.
  - `irrelevant` when a snippet shares vocabulary but is in the wrong domain.
  - `partially supports` only when it genuinely addresses the query but misses a necessary fact/scope.

