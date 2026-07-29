# Run K: Short-Context Targeted Strategy And Results

## Purpose

Run K was designed as a narrow follow-up to Run J, not a reset.

Run J had already solved the catastrophic version of over-abstention. Run K therefore targeted the leftover boundary cases:

- 5-doc answerable rows
- `partial_only` answerable rows
- especially misinformation
- the `No conflict <-> Complementary information` confusion boundary

The philosophy was: do not shake the whole training recipe; add only the minimum extra signal needed for the remaining weak spots.

## What changed relative to Run J

Run K kept the Run J train/val backbone and made targeted additions on top.

| area | Run J | Run K |
| --- | --- | --- |
| starting split | benchmark-augmented `862/81` | same Run J `862/81` |
| new derived rows | none beyond J | `27` short-context answerable derived rows |
| added `No conflict` derived rows | no | no |
| boundary conflict reweighting | flat `1` each | stronger weighting for all non-`No conflict` boundary labels |
| doc-verdict boundary drill | off | on with weight `1` |
| partial-synthesis drills | moderate | stronger |

## Derived-data engineering

Run K used [scripts/prepare_run_k_splits.py](../../scripts/prepare_run_k_splits.py) to derive benchmark-like 5-doc answerable variants from existing Run J training rows.

### Run K derived rows

| item | value |
| --- | --- |
| base train rows | 862 |
| base val rows | 81 |
| derived rows added | 27 |
| final train rows | 889 |
| final val rows | 81 |

### Derived rows by conflict type

| conflict type | count |
| --- | --- |
| Complementary information | 10 |
| Conflicting opinions or research outcomes | 10 |
| Conflict due to outdated information | 6 |
| Conflict due to misinformation | 1 |

### Derived rows by origin

| origin | count |
| --- | --- |
| `run_k_short5_support` | 27 |

### Derived rows by parent doc count

| parent docs | count |
| --- | --- |
| 8 | 1 |
| 10 | 13 |
| 11 | 4 |
| 12 | 2 |
| 13 | 4 |
| 14 | 2 |
| 15 | 1 |

Source: [data/splits/run_k/run_k_split_summary.json](../../data/splits/run_k/run_k_split_summary.json)

## Message-mixture engineering

Run K’s rebuild script was [slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh](../../slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh).

The key difference from Run J was stronger local pressure on the short answerable boundary.

### Important message-builder changes vs Run J

| knob | Run J | Run K |
| --- | --- | --- |
| boundary `No conflict` | 1 | 1 |
| boundary `Complementary information` | 1 | 2 |
| boundary `Conflicting opinions...` | 1 | 2 |
| boundary `Outdated` | 1 | 2 |
| boundary `Misinformation` | 1 | 2 |
| doc-verdict boundary drill | 0 | 1 |
| runtime partial-synthesis `e2e` | 1 | 2 |
| runtime partial-synthesis `answer_only` | 1 | 2 |
| minimal partial-synthesis | 2 | 3 |

### Important sample-weight settings

| knob | value |
| --- | --- |
| `answerable_exact_weight` | `1.35` |
| `answerable_short_weight` | `1.4` |
| `decision_answerable_short_extra_weight` | `1.15` |
| `answerable_partial_only_weight` | `1.2` |
| `benchmark_like_aug_weight` | `1.05` |
| `run_k_short5_support` origin weight | `1.0` |
| partial-only `Complementary information` extra | `1.15` |
| partial-only `Conflicting opinions...` extra | `1.15` |
| partial-only `Misinformation` extra | `1.6` |
| refusal weights | unchanged from Run J |

### Weighted geometry

| slice | Run J weighted | Run K weighted |
| --- | --- | --- |
| total refusal | 1438.95 | 1662.785 |
| total answerable | 10884.6592 | 14204.3825 |
| 5-doc refusal | 1430.55 | 1653.685 |
| 5-doc answerable | 4179.4987 | 6235.5271 |

Interpretation:

- Run K deliberately increased short answerable exposure further.
- It still did not overshoot back into the extreme refusal imbalance seen in the older failed direction.

Source: [data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_weight_summary.json](../../data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_weight_summary.json)

## Results summary

The detailed result numbers below were preserved from the audited Run K findings discussed during the experiment cycle. The full raw Run K report tree is not currently synced in this repo, so this markdown is the main archived record here.

## Run K vs Run J headline comparison

### 7B benchmark

| metric | Run J | Run K | delta |
| --- | --- | --- | --- |
| final abstain accuracy | 94.29 | 95.79 | +1.50 |
| predicted abstains | 164 | 133 | -31 |
| false abstains | 39 | 18 | -21 |
| missed refusals | 3 | 13 | +10 |
| token F1 | 0.3555 | 0.3296 | -0.0259 |
| Rouge-L F1 | 0.3215 | 0.2935 | -0.0280 |
| conflict type accuracy | 62.07 | 60.14 | -1.93 |
| doc micro accuracy | 77.96 | 80.44 | +2.48 |
| doc macro F1 | 0.7332 | 0.7636 | +0.0304 |
| contract ok | 92.5 | 93.6 | +1.1 |
| citation pass | 66.3 | 73.3 | +7.0 |
| citation avg coverage | 0.8299 | 0.8433 | +0.0134 |

### 32B benchmark

| metric | Run J | Run K | delta |
| --- | --- | --- | --- |
| final abstain accuracy | 94.97 | 96.60 | +1.63 |
| predicted abstains | 149 | 153 | +4 |
| false abstains | 29 | 25 | -4 |
| missed refusals | 8 | 0 | -8 |
| token F1 | 0.3945 | 0.4068 | +0.0123 |
| Rouge-L F1 | 0.3620 | 0.3697 | +0.0077 |
| conflict type accuracy | 62.55 | 67.30 | +4.75 |
| doc micro accuracy | 82.05 | 83.27 | +1.22 |
| doc macro F1 | 0.7808 | 0.7929 | +0.0121 |
| contract ok | 92.8 | 93.5 | +0.7 |
| citation pass | 55.1 | 75.3 | +20.2 |
| citation avg coverage | 0.7741 | 0.8426 | +0.0685 |

### 32B val

| metric | Run J | Run K | delta |
| --- | --- | --- | --- |
| final abstain accuracy | 100.0 | 100.0 | 0.0 |
| token F1 | 0.4578 | 0.4822 | +0.0244 |
| Rouge-L F1 | 0.3362 | 0.3515 | +0.0153 |
| conflict type accuracy | 67.09 | 70.37 | +3.28 |
| doc micro accuracy | 83.27 | 84.03 | +0.76 |
| doc macro F1 | 0.8310 | 0.8424 | +0.0114 |
| contract ok | 95.1 | 95.1 | 0.0 |
| citation pass | 79.3 | 79.7 | +0.4 |
| citation avg coverage | 0.8664 | 0.8667 | +0.0003 |

## Interpretation

Run K behaved differently by model size.

### 7B reading

Run K 7B became more conservative in a useful way:

- false abstains fell sharply
- doc-verdict behavior improved
- citation discipline improved
- contract compliance improved

But there was also a cost:

- more missed refusals
- weaker answer overlap
- slightly worse conflict-type classification

So for 7B, Run K was a trade-off run rather than a clean universal win.

### 32B reading

Run K 32B was much closer to the ideal result:

- abstention improved
- missed refusals disappeared
- conflict-type accuracy improved strongly
- final-answer overlap improved
- doc-verdict and citation metrics also improved

For 32B, Run K was the strongest overall result among J and K.

## What Run K taught us

Run K produced one important insight:

- short-context targeted augmentation works
- but if that signal is too one-sided, smaller models can trade answer richness for caution
- larger models can absorb the same signal more cleanly

That lesson directly motivated Run L, which kept the short-context idea but reintroduced `No conflict` coverage to avoid boundary skew.

## Stored artifacts

- Split summary: [data/splits/run_k/run_k_split_summary.json](../../data/splits/run_k/run_k_split_summary.json)
- Weight summary: [data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_weight_summary.json](../../data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_weight_summary.json)
- Rebuild script: [slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh](../../slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh)

The completed Qwen matrices are retained in `final_model_outputs/qwen7b/` and `final_model_outputs/qwen32b/`.
