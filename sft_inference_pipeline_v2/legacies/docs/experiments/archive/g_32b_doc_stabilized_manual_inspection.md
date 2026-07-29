# Manual Inspection: 32B Run G Doc-Stabilized

Run prefix:

```text
sft_qwen25_32b_stagewise_main_trace_text_g_doc_stabilized_e2e_{strict,trace_text,minimal}_val_stagewise
```

Inspection date: 2026-05-28

## Summary

Run G successfully restored much of the doc-verdict quality that F lost, but it did not preserve F's conflict/contract gains. The doc-stabilizer helped Stage 1, especially strict/minimal, but softened Stage 2 conflict boundaries too much. Overall, G is a useful doc-stabilized ablation, not the best 32B final checkpoint if conflict/contract/citation are primary.

Best current 32B checkpoint candidate remains F boundary-guarded. G should be retained as the doc-quality fallback.

## Headline Metrics

| Profile | Contract adj | Doc micro | Doc macro | Conflict acc | Final abstain acc | Token F1 | Citation cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G strict | 77.6 | 85.17 | 0.8590 | 69.39 | 97.96 | 0.5530 | 0.6207 |
| G runtime | 75.5 | 81.59 | 0.8159 | 73.47 | 100.00 | 0.5605 | 0.6095 |
| G minimal | 71.4 | 83.63 | 0.8469 | 69.39 | 97.96 | 0.5730 | 0.5898 |

Structural checks:

```text
strict:  sentinel=49/49, think=49/49
runtime: sentinel=49/49, think=49/49
minimal: sentinel=49/49, think=49/49
```

## Compared With F

F remains stronger on conflict and contract:

| Profile | F conflict | G conflict | F contract adj | G contract adj | F doc micro | G doc micro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 77.55 | 69.39 | 85.7 | 77.6 | 80.56 | 85.17 |
| runtime | 75.00 | 73.47 | 77.6 | 75.5 | 79.54 | 81.59 |
| minimal | 73.47 | 69.39 | 81.6 | 71.4 | 81.07 | 83.63 |

Interpretation:

- G strict gives the strongest doc macro/micro among recent non-D 32B runs, but drops conflict by 8.16 points and contract-adjusted by 8.1 points versus F strict.
- G runtime is closest to F overall, but still trails F on conflict/contract while improving doc only modestly.
- G minimal improves doc, but loses too much contract and conflict relative to F minimal.

## Conflict Error Pattern

G tends to over-smooth conflicts into `No conflict`, especially when sources differ in nuance, scope, or interpretive stance rather than giving directly contradictory facts.

Top minimal confusions:

```text
Complementary information -> No conflict: 6
Conflicting opinions or research outcomes -> No conflict: 3
No conflict -> Complementary information: 3
No conflict -> Conflict due to outdated information: 1
Conflict due to outdated information -> No conflict: 1
Conflicting opinions or research outcomes -> Complementary information: 1
```

Repeated conflict misses across profiles:

```text
#0127  Complementary information -> No conflict
#0203  Complementary information -> No conflict
#0015  No conflict -> Conflict due to outdated information
#0187  Complementary information -> No conflict
#0381  Conflict due to outdated information -> No conflict
#0416  Conflicting opinions or research outcomes -> Complementary information
#0399  Conflicting opinions or research outcomes -> No conflict
#0654  Conflicting opinions or research outcomes -> No conflict
#0650  No conflict -> Complementary information
#0638  No conflict -> Complementary information
```

## Manual Row Notes

`#0333` Supreme Court appointment:

- G correctly predicts `Conflict due to outdated information` in all profiles.
- Final answer correctly identifies Ketanji Brown Jackson and treats Amy Coney Barrett as superseded older information.
- This is a true G win over some earlier over-smoothing behavior.

`#0373` Declaration signers:

- Runtime and minimal correctly settle on `No conflict`.
- Strict still over-labels as `Complementary information`.
- Final answers are grounded and structurally clean.

`#0381` world population:

- G predicts `No conflict` in all profiles, but gold is outdated conflict.
- The model treats 8.0B and 8.2B style estimates as compatible trend estimates, missing that the task expects temporal supersession.
- This is a real Stage-2 regression.

`#0399` vegan pregnancy:

- G predicts `No conflict` in all profiles.
- The final answer actually mentions opposing/conditional views, but Stage 2 labels them as non-conflicting.
- This is the clearest example of G's conflict-softening failure.

`#0654` gravity:

- G predicts `No conflict` and abstains.
- The final abstain is correct, but Stage 2 should identify conflicting opinions/research outcomes.
- This suggests answer safety survived even when conflict taxonomy regressed.

`#0015` American Idol:

- G predicts outdated conflict, while gold is no conflict.
- The model over-triggers temporal conflict because older season results are present, even though they do not contradict the latest winner answer.

`#0392` FIBA ranking:

- G correctly predicts `No conflict` in all profiles.
- This fixes F's tendency to over-call outdated conflict on this example.

`#0531` socialism/communism:

- G predicts `No conflict` and abstains cleanly in all profiles.
- The malformed/source-contaminated minimal behavior seen in the 7B F line is not present here.

## Doc Verdict Inspection

Minimal doc report:

```text
total_doc_pairs=391
correct_doc_pairs=327
doc_micro=83.63
doc_macro=0.8469
supports_f1=0.8459
partial_f1=0.8118
irrelevant_f1=0.8829
```

Main doc confusions:

```text
supports -> partially supports: 37
partially supports -> supports: 14
partially supports -> irrelevant: 6
irrelevant -> partially supports: 7
```

G did reduce F's Stage-1 weakness, but did not fully eliminate the model's bias toward `partially supports`.

## Final Judgment

Run G is not a better final 32B SFT checkpoint than F. It is valuable because it proves the doc-verdict stabilizer works, but the tradeoff is not favorable:

- Use F when selecting one 32B checkpoint for downstream DPO/ORPO.
- Keep G as the doc-stabilized ablation and as evidence for future targeted Stage-1 tuning.
- Do not port G as the default large-model strategy to Qwen3-32B, Mistral 24B, or Gemma 27B unless doc-verdict accuracy becomes more important than conflict/contract behavior.
