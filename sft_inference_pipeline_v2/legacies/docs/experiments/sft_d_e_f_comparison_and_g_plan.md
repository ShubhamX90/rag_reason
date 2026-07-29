# SFT D/E/F Comparison and Final G Plan

Status: updated after syncing D/E/F, 7B G, and 32B G outputs.

This memo is the compact decision log for the SFT line of work. It compares runs D, E, and F on both model sizes, then records the exact G strategy we should use for each model.

## Executive Summary

- `D` is the safest minimal-internalization baseline.
- `E` is only worthwhile for 32B, where it boosts conflict but weakens some doc behavior.
- `F` is the best strict/runtime conflict run for both models, but the final preference splits by size:
  - 7B: `D` remains the cleanest minimal run; `F` is best for strict/runtime.
  - 32B: `F` remains the strongest overall conflict/contract run after G; G recovers doc verdicts but gives up too much conflict/contract.

## Qwen 7B

| Run | Strict conflict | Strict doc | Runtime conflict | Runtime doc | Minimal conflict | Minimal doc | Trace reliability | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| D | 66.67 | 74.22 | 68.75 | 75.79 | 73.47 | 76.98 | Strict `49/49`, Runtime `48/49`, Minimal `49/49` | Safest minimal run |
| E | 61.22 | 76.98 | 57.14 | 76.98 | 58.33 | 78.52 | Strict/Runtime/Minimal `49/49` | Contract improved, conflict collapsed |
| F | 77.55 | 79.43 | 71.43 | 77.12 | 72.92 | 76.94 | Strict/Runtime `49/49`, Minimal `48/49` | Best strict/runtime, one minimal leak |

7B row-level conclusion:

- `D` remains the cleanest minimal trace internalization run.
- `E` is not a final candidate because conflict accuracy drops sharply.
- `F` is the best strict/runtime run, but minimal has one malformed/source-contaminated row: `#0531`.

## Qwen 32B

| Run | Strict conflict | Strict doc | Runtime conflict | Runtime doc | Minimal conflict | Minimal doc | Trace reliability | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| D | 63.27 | 85.42 | 69.39 | 85.68 | 63.27 | 86.45 | Strict/Runtime/Minimal `49/49` | Best doc-verdict baseline |
| E | 73.47 | 85.42 | 73.47 | 84.65 | 71.43 | 82.35 | Strict/Runtime/Minimal `49/49` | Conflict improves, doc weakens modestly |
| F | 77.55 | 80.56 | 71.43 | 79.54 | 72.92 | 81.07 | Strict/Runtime `49/49`, Minimal `48/49` | Best conflict/contract, doc-regressed |

32B row-level conclusion:

- `D` is the strongest doc-verdict baseline.
- `E` is a useful conflict boost.
- `F` is the strongest conflict/contract/citation run, but it over-predicts `partially supports` and loses some Stage-1 doc quality.

## Cross-Model Takeaways

- `E` is model-size sensitive:
  - It hurts 7B too much.
  - It is acceptable for 32B, but not the end state.
- `F` is the most generally successful conflict-boundary idea.
- The final SFT direction must diverge by model size:
  - 7B should optimize for minimal robustness and source-hygiene.
  - 32B should optimize for doc-verdict recovery without losing F's conflict gains.

## Final G Plan for 32B

Goal: preserve F's conflict gains and recover doc-verdict quality.

Key strategy:

- Keep F's backbone.
- Add `runtime_doc_boundary_trace_text` copies for runtime `doc_verdict` rows.
- Increase runtime `doc_verdict` weight.
- Softly reduce conflict pressure relative to F.
- Raise dev doc-verdict weight so selection favors better Stage-1 labels.

Reference:

- [SFT Run G: Doc-Stabilized Boundary-Guarded Strategy](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/docs/experiments/sft_run_g_doc_stabilized_strategy.md)

Observed 32B G result:

| Run | Strict conflict | Strict doc | Runtime conflict | Runtime doc | Minimal conflict | Minimal doc | Trace reliability | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| G | 69.39 | 85.17 | 73.47 | 81.59 | 69.39 | 83.63 | Strict/Runtime/Minimal `49/49` | Doc recovered, conflict/contract softened too much |

32B G judgment:

- G succeeded at doc recovery, especially strict and minimal.
- G did not preserve F's conflict/contract behavior.
- G introduced a visible tendency to smooth real conflicts into `No conflict`.
- G also over-triggered temporal conflict on rows where older facts were contextual rather than contradictory.

Updated 32B preference:

```text
Best overall final 32B checkpoint: F
Best doc-stabilized fallback: G
Best pure doc-verdict baseline: D
Avoid as final 32B checkpoint: E unless conflict-only calibration is being studied
```

Manual inspection:

- [Manual Inspection: 32B Run G Doc-Stabilized](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/analysis/g_32b_doc_stabilized_manual_inspection.md)

## Final G Plan for 7B

Goal: keep F's strict/runtime gains while restoring D-like minimal robustness and preventing source-instruction leakage.

Key strategy:

- Keep F's backbone.
- Increase minimal E2E weight from `4` to `5`.
- Increase only `Complementary information` boundary pressure.
- Add source-hygiene E2E drills so the model learns to ignore prompt-like text inside retrieved snippets.
- Do not add the 32B doc-verdict stabilizer.

Reference:

- [SFT Run G for 7B: Source-Guarded Boundary Strategy](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/docs/experiments/sft_run_g_7b_source_guarded_strategy.md)

Observed 7B G result:

| Run | Strict conflict | Strict doc | Runtime conflict | Runtime doc | Minimal conflict | Minimal doc | Trace reliability | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| G | 73.47 | 77.75 | 73.47 | 77.49 | 69.39 | 76.21 | Strict/Runtime/Minimal `49/49` | Fixed minimal structure, but overcorrected complementary boundary |

7B G judgment:

- G successfully restored minimal `think=49/49`.
- G fixed the malformed `#0531` structure.
- G lost F's strict conflict edge and D/F's minimal conflict edge.
- G lowered contract-adjusted scores substantially, especially runtime and minimal.
- G should not be selected as the primary 7B SFT checkpoint.

Updated 7B preference:

```text
Minimal internalization proof: D
Strict/runtime conflict and doc quality: F
Avoid as final 7B checkpoint: E and G
Useful lesson from G: source-hygiene drills help structure, but complementary pressure at 2x is too strong for 7B.
```

## Practical Replication Rule

If we later port this work to Llama or Mistral:

- Start from `D` if the priority is minimal-prompt trace internalization.
- Start from `F` if the priority is strict/runtime conflict quality.
- Use `32B G` only when doc-verdict quality matters more than a medium conflict/contract tradeoff.
- Do not directly reuse `7B G`; reuse only the source-hygiene idea, and keep complementary pressure closer to F.

For current next-model SFT runs:

```text
CSIS 7B/8B family default: F boundary-guarded
Sharanga 24B/27B/32B family default: F boundary-guarded
Keep G doc-stabilized as a targeted ablation, not the default launch strategy.
```
