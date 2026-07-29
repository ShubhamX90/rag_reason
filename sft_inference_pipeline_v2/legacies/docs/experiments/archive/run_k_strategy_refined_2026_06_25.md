# Run K Strategy Refined - 2026-06-25

## Why Run K exists

Run J fixed the catastrophic over-abstention problem, especially relative to the older runs, but the benchmark audit still showed a narrow residual weakness:

- answerable 5-doc rows are still the main external stress case
- the hardest slice is `partial_only`, especially `Conflict due to misinformation`
- the main stage-2 confusion boundary is still `Complementary information <-> No conflict`
- there are also some `support_present` 5-doc answerable false abstains, so the issue is not only partial-only

This means Run K should not be a broad retraining shake-up. It should be a targeted short-context answerable calibration pass.

## Evidence behind the design

Benchmark holdout (`benchmark_final_v2_holdout_clean_736.jsonl`) is heavily 5-doc:

- answerable 5-doc support-present:
  - `No conflict`: 151
  - `Complementary information`: 122
  - `Conflict due to outdated information`: 67
  - `Conflicting opinions or research outcomes`: 61
  - `Conflict due to misinformation`: 25
- answerable 5-doc partial-only:
  - `Complementary information`: 35
  - `Conflicting opinions or research outcomes`: 31
  - `Conflict due to misinformation`: 11

Run J train already improved this geometry, but the most fragile slice is still tiny in-train:

- 5-doc partial-only answerable in Run J train:
  - `Complementary information`: 11
  - `Conflicting opinions or research outcomes`: 11
  - `Conflict due to misinformation`: 3

That explains why simple global weighting is not enough by itself.

## What Run K changes

### 1. Keep Run J as the base

Run K starts from Run J train/val, not from Run I. This preserves the better refusal calibration and the current 81-row validation setup.

### 2. Add targeted short-context answerable variants

Use [scripts/prepare_run_k_splits.py](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/scripts/prepare_run_k_splits.py:1) to derive 5-doc answerable variants from existing Run J training rows only.

Design rules:

- derive only from answerable rows with more than 5 docs
- keep only conflict-bearing target labels:
  - `Complementary information`
  - `Conflicting opinions or research outcomes`
  - `Conflict due to outdated information`
  - `Conflict due to misinformation`
- exclude `No conflict` derived rows, because 32B already tends to over-predict `No conflict`
- cap the derived set conservatively:
  - `Complementary information`: 10
  - `Conflicting opinions or research outcomes`: 10
  - `Conflict due to outdated information`: 6
  - `Conflict due to misinformation`: 1

Result:

- derived rows added: 27
- final Run K train rows: 889
- val rows unchanged: 81

### 3. Strengthen only the boundaries that still break

Use [slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh:1).

Key changes versus Run J:

- conflict boundary drills:
  - `Complementary information=2`
  - `Conflicting opinions or research outcomes=2`
  - `Conflict due to outdated information=2`
  - `Conflict due to misinformation=2`
  - `No conflict=1`
- doc-verdict boundary drill enabled with weight `1`
- partial-synthesis drills strengthened:
  - runtime e2e `2`
  - runtime answer_only `2`
  - minimal `3`

### 4. Keep weighting conservative

Run K does **not** repeat the aggressive Run I weighting pattern.

Important weights:

- `answerable_exact_weight=1.35`
- `answerable_short_weight=1.4`
- `decision_answerable_short_extra_weight=1.15`
- `answerable_partial_only_weight=1.2`
- `benchmark_like_aug_weight=1.05`
- `run_k_short5_support` origin weight = `1.0`
- partial-only label extras:
  - `Complementary information=1.15`
  - `Conflicting opinions or research outcomes=1.15`
  - `Conflict due to misinformation=1.6`
- refusal weights stay at Run J values

## Mixture sanity check

Run I weighted 5-doc answerable vs 5-doc refusal:

- `1113.84` vs `1232.64`

Run J weighted 5-doc answerable vs 5-doc refusal:

- `4179.50` vs `1430.55`

Refined Run K weighted 5-doc answerable vs 5-doc refusal:

- `6235.53` vs `1653.69`

This is stronger than Run J, but still far more controlled than the first over-aggressive Run K draft. It also lands much closer to the benchmark’s held-out 5-doc answerable/refusal geometry than Run I did.

## Current recommendation

Run J 32B is already the best current model and the benchmark is not broken anymore. So Run K should be treated as a **targeted improvement run**, not an emergency reset.

I would only launch Run K if we explicitly decide that the remaining benchmark gap is worth another training cycle. The data and message rebuild path is now ready and sanity-checked.
