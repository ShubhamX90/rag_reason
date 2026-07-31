# CATS v2 Reproduction and Artifact Provenance

**Status:** Current end-to-end artifact lineage for dataset, model-output,
evaluation, orchestration, auditing, post-hoc analysis, and paper-facing result
export.

This document answers a practical reproducibility question: given a row in the
master workbook, where did it come from, how can it be recomputed, which files
are authoritative, which are diagnostics, and which apparently similar files
must not be mixed with it?

## 1. Reproducibility principle

The repository is a chain of versioned transformations. The lineage is:

```text
gold dataset and model-output exports -> prepared evaluator input JSONL
prepared input -> validation and taxonomy normalization
validated input -> local judge collection or all-at-once evaluation
run -> detailed_results.json, eval_report.md, and run_config.yaml
source results -> authoritative 108-row inventory and master artifacts
master artifacts -> audits, post-hoc diagnostics, and paper tables
```

The safest reproducibility unit is a result directory containing detailed JSON
and run config, together with the exact prepared input and committee/cache
provenance. A scalar copied into Excel without its source path is not a
reproducible result.
## 2. Dataset and split artifacts

### 2.1 Training/validation split snapshot

The current split manifest is
[`../data/splits/92p5_7p5/split_manifest.json`](../data/splits/92p5_7p5/split_manifest.json).
It records seed 21 and the current 658-row source population:

```text
train 609, val 49, test 0, total 658
```

The split stratification key is refusal status plus normalized conflict type.
The normalization treats the type-3 'and' and 'or' spellings as one stratum.
`train_ids.json` and `val_ids.json` are the row-id membership artifacts and
must be treated as a matched set with the manifest.

This split is distinct from the 736-row benchmark holdout. The holdout is not
formed by appending the 49 validation rows and should not be described as the
same population.

### 2.2 Benchmark holdout

The canonical benchmark is `data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl`.
It contains 736 examples and is the gold base merged onto model outputs for
benchmark evaluation. It supplies queries, retrieved documents, per-document
gold notes, conflict metadata, expected response/answerability information,
and any gold target required by STR.

### 2.3 Ceiling pilot

The gold-ceiling pilot is `data/ceiling_pilots/val_stage3_gold_expected_as_model_output.jsonl`.
It contains 49 validation examples with the expected answer placed in the
model-output position. Its purpose is to estimate evaluator/judge ceiling
behavior and debug the scoring path, not to claim model-generation performance.

## 3. Model-output preparation

[`../scripts/prep_model_outputs_for_eval.py`](../scripts/prep_model_outputs_for_eval.py)
merges exported model answers onto a canonical gold base and writes
evaluator-ready JSONL under `inputs/prepped_model_eval_inputs/`.

The preparation script discovers exports, indexes them by stable sample id,
extracts the final answer from raw/think-block formats, removes end markers and
dangling scaffolding, normalizes conflict metadata, preserves raw output and
source metadata, writes a clean `model_output`, and can fail on empty answers.

The evaluator must not silently score `expected_response.answer` as the model
answer unless preparation explicitly enables that fallback for a diagnostic
use. Otherwise, a missing model output could become a gold-against-gold score.

Canonical prepared-input roots are:

```text
inputs/prepped_model_eval_inputs/benchmark_set_all_modes/
inputs/prepped_model_eval_inputs/other_techniques/
inputs/prepped_model_eval_inputs/other_techniques_fixed/
```

Standard leaves use `<model>/<eval_family>/<prompt_mode>/<run_type>/input.jsonl`.
Comparison leaves use their technique/model layout. A renamed directory can
change workbook identity even if JSON content is unchanged.

[`../scripts/normalize_benchmark_conflict_categories.py`](../scripts/normalize_benchmark_conflict_categories.py)
is the canonical taxonomy repair/check for the benchmark gold file and every
prepared benchmark input. Run it in dry-run mode first on an unfamiliar snapshot.

## 4. Input validation gate

The benchmark-prepared validator checks strict JSONL parsing, expected row
count, unique and ordered ids, exact canonical-gold alignment, required
model-output metadata, non-empty cleaned output, absence of raw reasoning or
scaffolding markers, valid conflict ids, required document/gold fields, and
existence of the declared model-output source path.

Example:

```bash
python scripts/validate_eval_input_jsonl.py \
  --input inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen7b/e2e/minimal/sft/input.jsonl \
  --mode benchmark_prepped \
  --gold data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl \
  --expected-rows 736
```

An input that fails this gate is not ready for a final committee run, even if
the evaluator could technically load it.

## 5. Experiment matrix and scope

The paper-facing master scope is exactly 108 rows. Its authoritative row list
is the source-path column in
`outputs/benchmark_local_committee_3judge/master_results/cats_master_results_20260708.csv`.

The current composition is 96 standard benchmark rows, 6 answer-only SFT rows,
2 Llama comparison rows, and 4 latest fixed Mistral/Qwen comparison rows. The
older unfixed comparison finals, staged collection files, and historical
verifier paths are not additional paper experiments.

Some historical scripts enumerate 114 result-like files. The six extra paths
are outside the current 108-row scope. Use the authoritative scope audit for
paper-facing counts; never infer the experiment count from a recursive count of
`detailed_results.json`.

The four fixed comparison rows are the latest results for fixed CoT Mistral,
fixed CoT Qwen, fixed CoN Mistral, and fixed CoN Qwen. Their latest detailed
results and applicability counts supersede older unfixed siblings within the
current matrix.

## 6. Local committee execution modes

The active benchmark config uses Qwen3.5-397B-A17B at priority 6, Mistral
Small 4 at priority 3, and DeepSeek-R1-Distill-32B at priority 2. The all-at-
once mode requires all three endpoints. Responses may be cached under
`outputs/benchmark_local_committee_3judge/response_cache/`.

The staged configs are:

```text
configs/local_staged/benchmark_local_stage_qwen397_collect.yaml
configs/local_staged/benchmark_local_stage_mistral4_collect.yaml
configs/local_staged/benchmark_local_stage_deepseek32_collect.yaml
configs/local_staged/benchmark_local_stage_final_readonly.yaml
```

The three collection passes each run one judge and write a shared cache. The
final read-only pass loads all three cached responses and computes committee
decisions without requiring model servers. Cache directory, prompt-affecting
settings, model ids, and input file must remain consistent across stages.

The canonical output home is `outputs/benchmark_local_committee_3judge/`, with
standard, comparison, fixed-comparison, master-results, and citation-quality
subtrees. A normal final leaf contains `final/detailed_results.json`,
`final/eval_report.md`, and `final/run_config.yaml`. Staged leaves are not
automatically master rows.

## 7. Sharanga orchestration

Operational scripts are under
[`../slurm/sharanga/local_committee/`](../slurm/sharanga/local_committee/).
Start servers with the validated hardware-specific Slurm scripts, record each
job's actual endpoint, and probe it before collection:

```bash
python slurm/sharanga/local_committee/probe_openai_endpoint.py \
  --base-url http://<node>:<port>/v1 \
  --model <served-model-name> \
  --timeout 180 \
  --disable-thinking
```

The probe checks both `/models` and `/chat/completions` with a JSON-only task.
An endpoint that answers `/models` but fails the completion probe is not ready.

The one-file watcher validates input, chooses controller placement, submits
collection jobs, watches Slurm state, verifies stage artifacts, and submits the
final read-only merge. It persists input path and run label so a resume against
a different file is rejected. The relevant wrappers are
`benchmark_collect_eval.sbatch`, `benchmark_final_merge.sbatch`,
`benchmark_endpoint_health_gate.sbatch`, and `benchmark_watch_pipeline.sbatch`.

Failure rules are conservative: wait and probe if an endpoint is not ready;
complete missing collection if final read-only has a cache miss; repair and
revalidate failed inputs; preserve malformed judge error records; and never
mix watcher state or caches across input files.
## 8. Master result generation

[`../scripts/audit_cats_master_results.py`](../scripts/audit_cats_master_results.py)
loads the 108 source paths, recomputes aggregate summaries from each
`per_sample` payload, and compares generated values to master CSV, JSON, and
Markdown artifacts. It also checks inventory, uniqueness, source metadata, and
result sibling artifacts.

The source detailed JSON is authoritative for numeric recomputation. The
workbook is a presentation/export artifact.

[`../scripts/update_master_results_workbook.py`](../scripts/update_master_results_workbook.py)
updates the hierarchical workbook from detailed source JSON while preserving
existing non-CATS component cells from the supplied older workbook. It matches
rows using a legacy numeric signature, recomputes answer quality, CATS
prevalence, CATS balanced, and count fields, then verifies every updated cell
and header. This prevents a CATS-logic update from silently rewriting unrelated
historical component values.

Current workbook columns are:

```text
J: GR answer precision     K: GR answer recall      L: GR answer F1
M: GR refusal precision   N: GR refusal recall    O: GR refusal F1
P: GR accuracy             Q: STR                  R: FG
S: behavior                T: answer quality       U: final CATS prevalence
V: final CATS balanced     W: n                     X: behavior_n
Y: fg_n                    Z: str_n                 AA: answer_quality_n
```

## 9. Audit hierarchy

Use audits from strongest source to weakest presentation layer:

1. input validation against the canonical gold file;
2. per-run `run_config.yaml` and detailed JSON inspection;
3. source inventory audit over the 108 paths;
4. recomputation from every source `per_sample` list;
5. master CSV/JSON/Markdown comparison;
6. workbook cell/header audit; and
7. visual/manual inspection of formatting and paper-facing labels.

The older Excel audit scripts remain useful for historical workbook checks but
are not substitutes for the current hierarchical source audit.

## 10. Post-hoc citation-quality analysis

The active CATS FG score answers whether a claim is supported and citation-linked
under the committee support rule. The separate post-hoc tools examine cited
document quality without rerunning judges:

```text
scripts/compute_posthoc_citation_quality.py
scripts/run_posthoc_citation_quality_all.py
```

They combine existing claim-level details from `detailed_results.json` with gold
per-document notes from the corresponding prepared input. The analysis
separates committee-alignment precision, gold-positive citation precision,
hard-negative citation rate, soft-extra citation rate, and committee-support
citation recall.

Gold-positive precision and hard-negative rate are the primary deterministic
citation-cleanliness pair. Committee alignment is a stricter diagnostic, not a
universal truth label. The output is supplementary and does not alter primary
CATS components or the 108-row workbook.

## 11. Validation and ceiling lanes

The validation family is separate from the benchmark family:

```text
configs/val_tier2*.yaml
configs/local_staged/val_local_stage_*.yaml
configs/local_staged_gold_ceiling/val_gold_stage_*.yaml
outputs/val_gold_ceiling_local_staged/
```

The ordinary validation lane tests evaluator and committee behavior on the
49-row validation input. The gold-ceiling lane places a gold expected answer in
the model-output position to isolate evaluator/judge ceiling behavior. Neither
lane belongs in the benchmark 108-row matrix.

## 12. Artifact retention rules

Retain together the canonical gold input, prepared evaluator input, model-output
source metadata, final `run_config.yaml`, final `detailed_results.json`, final
`eval_report.md`, response-cache manifest or staged cache, source inventory row,
audit JSON/Markdown, and any post-hoc analysis output.

Keep staged collection results and earlier runs for provenance, but label them
as staged, supplementary, or historical. Do not place their scalar values in a
paper table without stating their status.

## 13. Paper-facing reproduction recipe

For one standard benchmark row:

1. use the exact source path from the 108-row inventory;
2. confirm its prepared `input.jsonl` passes the benchmark validator;
3. use the matching current local committee config and prompt bundle;
4. run all-at-once or complete all three staged collection passes;
5. run the final read-only aggregation;
6. verify the final detailed JSON has 736 ordered per-sample rows;
7. recompute the row from `per_sample` using `aggregate_sample_results`;
8. compare the recomputed row to the master CSV/JSON; and
9. only then inspect or cite the corresponding workbook row.

For the full paper matrix, repeat the process over the authoritative 108 source
paths. Do not use recursive file counts as a substitute for the inventory.

## 14. Final provenance checklist

- The input is the intended benchmark, validation, or ceiling family.
- The input row count and id order pass validation.
- The model-output source exists and is recorded.
- The conflict taxonomy is canonical.
- Committee model ids, priorities, prompt version, and cache mode are recorded.
- The result directory contains detailed JSON, report, and config.
- The source path is in the 108-row inventory if paper-facing.
- Result `n` and applicability counts are recomputed from per-sample data.
- Master artifacts pass their source audit.
- The workbook passes its header and cell audit.
- Post-hoc citation analysis is labeled supplementary.
- Historical, staged, and fixed-result distinctions are preserved in tables.
