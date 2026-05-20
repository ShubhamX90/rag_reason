# CATS v2.0 - Pipeline Issues v2

Generated: 2026-05-21

Scope: current `CATS_v2` source tree, including the v3 code changes, CLI
runners, scripts, config, tests, and the unused batch processor. This is a
second-pass audit after `ISSUES.md` and `FIXES-v3.md`.

Severity legend:

- P0: corrupts outputs, silently skips evaluation, or can make headline metrics wrong.
- P1: likely wrong behavior, resource leak, reproducibility issue, or serious UX trap.
- P2: stale docs/scripts, dead config, maintainability, or edge-case risk.

Validation performed:

- `python3 -m py_compile` over all pipeline Python files: passed.
- AST parse over all `CATS_v2/**/*.py`: passed.
- No live API run was performed.

---

## 1. P0 Issues

### 1.1 `cost_summary` is still written after reports

Files:

- `rag_eval/evaluator.py:103-118`
- `rag_eval/evaluator.py:425-434`

`evaluate_async()` writes `eval_report.md` and `detailed_results.json` before
adding `self.results["cost_summary"]`. The returned in-memory result has cost,
but both persisted output files miss it.

Impact:

- Cost summary is absent from `detailed_results.json`.
- Markdown reports do not include cost even though `_write_markdown_report()`
  supports a cost section.
- Any downstream tool reading files, not the live return value, sees incomplete
  results.

Fix:

Move the cost-summary block before `_write_markdown_report()` and
`_write_detailed_results()`.

### 1.2 `--committee none` still skips evaluation instead of using a single judge

Files:

- `run_evaluation.py:218-220`
- `run_evaluation_batch.py:147-149`
- `rag_eval/evaluator.py:81-101`

The CLI help says `none` means "single judge", but the config sets
`use_judge_committee=False` and no replacement single judge is installed.
`EnhancedEvaluator.evaluate_async()` then logs "Conflict evaluation skipped" and
writes empty metrics.

Impact:

- `--committee none` produces an empty report, not a fast/single-judge run.
- Interactive mode advertises a working "single judge" option that is not real.

Fix:

Either remove `none` or create a one-judge `JudgeCommitteeConfig` containing only
the Anthropic judge.

### 1.3 Concurrency limits still do not bound the full pipeline

Files:

- `rag_eval/evaluator.py:134-144`
- `rag_eval/conflict_eval.py:146-163`
- `rag_eval/judge_committee.py:340-355`
- `rag_eval/config.py:66-67`

The committee semaphore bounds only the per-judge behavior/recall calls inside
`JudgeCommittee.judge_behavior()`. The evaluator still schedules every sample at
once with `asyncio.as_completed(tasks)`, and the dedicated NLI judge is not
protected by that committee semaphore.

Impact:

- For large datasets, every sample can start NLI work concurrently.
- `batch_size` and `PipelineConfig.use_async_evaluation` do not limit anything.
- Anthropic NLI calls can still stampede the API even if
  `max_concurrent_requests` is low.

Fix:

Add an evaluator-level semaphore or process dataset chunks by `batch_size`.
Apply a separate semaphore to `nli_judge.judge_nli()` or share a global request
limiter across all judge clients.

### 1.4 `batch_size` is configured but unused

Files:

- `run_evaluation.py:190`
- `run_evaluation_batch.py:121`
- `rag_eval/evaluator.py:134-144`
- `rag_eval/config.py:117-121`

The runners store `args.batch_size`, but the evaluator never chunks by it. This
creates a false safety knob: users think they can reduce parallelism, but the
whole dataset is still scheduled immediately.

Fix:

Chunk `dataset` in `_evaluate_conflicts_async()` or replace the all-sample task
list with bounded worker tasks.

### 1.5 Boolean JSON parsing still accepts `"false"` as true

File:

- `rag_eval/judge_committee.py:267-288`

`_parse_judge_response()` still does:

```python
"adherent": bool(obj.get("adherent", False))
```

If a model returns `"adherent": "false"` as a string, Python converts it to
`True` because all non-empty strings are truthy.

Impact:

- A common malformed-but-plausible LLM output can invert a vote.
- This can flip committee outcomes when the vote is close.

Fix:

Parse booleans explicitly:

```python
raw = obj.get("adherent", False)
if isinstance(raw, bool):
    adherent = raw
elif isinstance(raw, str):
    adherent = raw.strip().lower() == "true"
else:
    adherent = bool(raw)
```

### 1.6 Correct-refusal full-credit can hide hallucinated answers

Files:

- `rag_eval/metrics.py:41-54`
- `rag_eval/evaluator.py:171-213`

The refusal detector searches for refusal phrases anywhere in the answer. If an
answer says "I cannot answer from the evidence, but X is definitely true", it is
classified as a refusal. If `gold_answerable=False`, the carve-out gives all
metrics `1.0` and skips behavior, grounding, and recall.

Impact:

- A mixed refusal plus hallucinated answer can receive perfect score.
- Claim extraction and NLI never inspect the hallucinated part.

Fix:

Use a stricter refusal classifier that requires the whole response to be an
abstention, or run claim extraction on mixed refusal text and only grant the
carve-out when no substantive claims remain.

### 1.7 Behavior judge is asked to evaluate evidence sufficiency without seeing evidence

Files:

- `rag_eval/judge_prompts.py:73-80`
- `rag_eval/evaluator.py:215-217`
- `rag_eval/conflict_eval.py:49-91`

The behavior prompt tells the judge to treat a refusal as adherent when evidence
is insufficient, but the prompt only includes query, answer, and conflict type.
It does not include retrieved docs, `per_doc_notes`, dates, or source quality.

Impact:

- For Type 4, judges cannot verify whether newer information was prioritized.
- For Type 5, judges cannot verify whether misinformation was rejected.
- For refusals not caught by the correct-refusal carve-out, the judge must guess
  evidence sufficiency from answer wording alone.

Fix:

Pass a compact evidence summary to the behavior prompt: doc IDs, verdicts,
dates, key facts, source quality, and conflict reason.

### 1.8 YAML output paths can overwrite multi-file outputs

Files:

- `run_evaluation.py:71-75`
- `run_evaluation.py:222-225`
- `configs/default.yaml:4-7`

When `--config configs/default.yaml` is used, the YAML contains fixed
`report_md` and `detailed_results_json` paths. These override the per-file
timestamped paths created in `process_single_file()`.

Impact:

- Multi-input runs with a config can write all reports to `outputs/eval_report.md`
  and `outputs/detailed_results.json`.
- Parallel files can race and overwrite each other.

Fix:

Do not let fixed report/detail paths from YAML override per-file output paths in
multi-input mode. Prefer treating `outputs_dir` as a base and deriving report
paths per input.

### 1.9 Sample failures are not isolated despite `skip_on_error`

Files:

- `rag_eval/config.py:117-121`
- `rag_eval/evaluator.py:134-144`
- `run_evaluation.py:331-333`

`PipelineConfig.skip_on_error` exists and YAML can set it, but
`_evaluate_conflicts_async()` does not catch per-sample exceptions. A single
unexpected sample-level crash aborts the evaluation after some API calls have
already been spent.

Fix:

Wrap each sample evaluation in a try/except when `skip_on_error=True`, emit a
per-sample error record, and continue aggregation over successful samples.

---

## 2. P1 Issues

### 2.1 Batch runner accepts `--config` but never loads YAML

Files:

- `run_evaluation_batch.py:74-79`
- `run_evaluation_batch.py:108-150`

Unlike `run_evaluation.py`, the batch runner parses `--config` and ignores it.
Priority overrides, concurrency, metric flags, and custom report paths have no
effect in batch mode.

Fix:

Share `_load_yaml_config()` and `_apply_yaml_to_config()` between the two
runners, or move config loading into the package.

### 2.2 Batch runner leaks judge clients on evaluation errors

Files:

- `run_evaluation_batch.py:198-206`
- `run_evaluation_batch.py:256-263`

`evaluator.close()` is called only after `evaluate_async()` succeeds. If loading,
evaluation, report writing, or summary formatting raises after clients are
created, async HTTP clients are not closed.

Fix:

Use `try/finally` around evaluator creation and evaluation, matching
`run_evaluation.py`.

### 2.3 CLI failure status is unreliable

Files:

- `run_evaluation.py:331-333`
- `run_evaluation_batch.py:329-359`

`process_single_file()` returns `None` on failure. `main()` does not check it for
single-file runs and does not `sys.exit(1)` when any parallel file fails.

Impact:

- CI or shell automation can see exit code 0 even when evaluation failed.

Fix:

Aggregate success/failure and exit non-zero if any requested input failed.

### 2.4 `cost_summary.avg_cost_per_decision` is inconsistent after NLI cost is added

Files:

- `rag_eval/judge_committee.py:487-500`
- `rag_eval/evaluator.py:109-118`

`get_cost_summary()` computes `avg_cost_per_decision` from committee-only cost.
`evaluate_async()` later adds NLI cost to `total_cost_usd` but does not recompute
the average or clarify that NLI calls are excluded from `decisions_made`.

Impact:

- `total_cost_usd / decisions_made` does not equal `avg_cost_per_decision`.
- Cost accounting is confusing for budget estimates.

Fix:

Track separate `committee_cost`, `nli_cost`, `total_cost`, `committee_decisions`,
`nli_requests`, and recompute any total average explicitly.

### 2.5 `read_jsonl()` silently drops malformed records

File:

- `rag_eval/data.py:59-69`

Invalid JSON lines are skipped without line number or warning. This can reduce
the dataset size and skew metrics without any visible signal.

Fix:

Log a warning with path and line number, or raise unless a `skip_bad_lines` flag
is enabled.

### 2.6 Missing `per_doc_notes` makes samples look unanswerable

Files:

- `rag_eval/data.py:108-109`
- `rag_eval/evaluator.py:164-184`

`gold_answerable` is derived only from `per_doc_notes`. If notes are missing but
retrieved docs are present, the sample becomes `gold_answerable=False`. A model
refusal can then receive the correct-refusal full-credit carve-out.

Impact:

- Minimal JSONL records can be scored as perfect refusals even when the docs
  actually answer the question.

Fix:

Validate required annotation fields before scoring, or support an explicit
record-level `gold_answerable` field.

### 2.7 `load_dotenv()` depends on current working directory

Files:

- `run_evaluation.py:20-21`
- `run_evaluation_batch.py:28-29`
- `test_api_keys.py:12-13`

`load_dotenv()` searches from the process working directory. Running
`python CATS_v2/run_evaluation.py ...` from the repository root can load the
wrong `.env` or none at all.

Fix:

Load from `Path(__file__).parent / ".env"` in CATS entry points.

### 2.8 `setup.sh` still generates stale v2 config and old model wording

File:

- `scripts/setup.sh:80-127`

The generated `.env` says "Claude Haiku" and the generated config includes old
fields that v3 removed or ignores (`confidence_threshold`, `use_async`,
`max_workers`). If `configs/default.yaml` does not already exist, setup creates
a config that does not match the current source.

Fix:

Make setup copy the checked-in `.env.example` and `configs/default.yaml`, or keep
the embedded template synchronized.

### 2.9 `run_eval.sh` still advertises Haiku and broken single-judge mode

File:

- `scripts/run_eval.sh:121-141`

Interactive mode says default/conservative use Haiku and says `None` is
"Single judge". Current code uses Sonnet, and `none` skips evaluation.

Fix:

Update labels and remove/repair option 3.

### 2.10 `run_eval.sh` loads `.env` unsafely

File:

- `scripts/run_eval.sh:36-37`

`export $(cat .env | grep -v '^#' | xargs)` breaks on quoted values, spaces,
comments after values, and blank assignment edge cases.

Fix:

Use `set -a; . ./.env; set +a` or let Python load `.env`.

### 2.11 Batch input order is nondeterministic

File:

- `run_evaluation_batch.py:292-293`

`input_files = list(set(input_files))` removes duplicates but randomizes order.
This changes logging order and output directory numbering between runs.

Fix:

Deduplicate while preserving order:

```python
input_files = list(dict.fromkeys(input_files))
```

### 2.12 Timestamped output directories can collide

Files:

- `run_evaluation.py:235-238`
- `run_evaluation_batch.py:163-167`

Directories use `file_stem + YYYYMMDD_HHMMSS`. Two files with the same stem
starting within the same second collide.

Fix:

Include `file_index`, a monotonic counter, or microseconds in the directory name.

### 2.13 `JudgeCommittee` logs may not enter CATS file logs

File:

- `rag_eval/judge_committee.py:29`

This module uses `logging.getLogger(__name__)`, while the rest of the package
uses the configured `CATS_v2` logger. Because no handler is attached to
`rag_eval.judge_committee`, warnings/errors can be routed differently or lost
depending on root logging configuration.

Fix:

Use `from .logging_config import logger` consistently.

### 2.14 `allow_paraphrases` is accepted but ignored

Files:

- `rag_eval/config.py:95-96`
- `rag_eval/conflict_eval.py:191-196`
- `rag_eval/judge_prompts.py:160-161`

`allow_paraphrases` is passed into `enhanced_single_truth_recall()` but not used.
The prompt always allows paraphrases and spelling differences.

Fix:

Branch the prompt based on `allow_paraphrases`, or remove the config field.

### 2.15 YAML fields are partially ignored

Files:

- `run_evaluation.py:82-111`
- `configs/default.yaml:14-39`

The loader does not apply `conflict_eval.enable`,
`conflict_eval.use_judge_committee`, or the commented `nli_judge` section.
`PipelineConfig.use_async_evaluation` and `show_progress` are also not used by
the evaluator.

Fix:

Either implement full schema loading or make the YAML contain only supported
fields.

### 2.16 Claim-extraction meta filter can drop real claims

File:

- `rag_eval/metrics.py:71-81`

The second branch of `_META_REFERENCE` matches sentences that start with verbs
such as `report`, `reports`, `show`, `indicate`, or `support`. Real claims like
"Reports indicate the policy changed in 2020" can be dropped before NLI.

Fix:

Limit the broad verb-starting branch to sentences that were produced after a
leading citation list was stripped, or require explicit anaphora such as "these
sources" / "the documents".

### 2.17 Factual grounding is sequential within each sample

File:

- `rag_eval/conflict_eval.py:146-163`

For each claim, each support doc is judged one at a time. This reduces the
benefit of async execution and creates uneven sample runtimes. If an evaluator-
level semaphore is added, the internal loop can still be slow on many-doc
records.

Fix:

Use bounded gather over claim-doc NLI pairs, or short-circuit after threshold is
met when `require_cross_doc=False`.

### 2.18 `batch_processor.py` is still unused and internally incomplete

Files:

- `rag_eval/batch_processor.py:47-230`
- `CHANGES.md:84-118`

The batch processor is not integrated with the evaluator. Inside the module,
`max_batch_size` is stored but never used to split large request lists, cost is
not accumulated into `self.total_cost`, and sync Anthropic client methods are
called from async functions.

Fix:

Either remove the module from active docs or integrate it with chunking, async
safe calls, and cost reporting.

### 2.19 Tests use stale placeholder detection

File:

- `test_installation.py:74-89`

The test checks for the lowercase string `your-key-here`, but `.env.example`
uses `YOUR_KEY_HERE` and `setup.sh` uses `your_anthropic_key_here`. A copied
placeholder can be reported as configured.

Fix:

Check for all known placeholders, empty values, and key prefixes.

---

## 3. P2 Issues

### 3.1 `__version__` still says `2.0.0`

File:

- `rag_eval/__init__.py:23`

The code and docs refer to v3 fixes, but package version remains `2.0.0`.

Fix:

Bump the version or remove version claims from docs.

### 3.2 `get_haiku_judge` alias now returns Sonnet

File:

- `rag_eval/config.py:188-190`

This keeps old imports working, but the function name is now misleading. Any
caller expecting the cheaper Haiku model silently gets Sonnet.

Fix:

Keep a real `get_haiku_judge()` for backward compatibility and add
`get_sonnet_judge()` separately, or emit a deprecation warning.

### 3.3 CLI docs and README are stale relative to current code

Files:

- `README.md`
- `QUICKSTART.md`
- `BATCH_PROCESSING_GUIDE.md`
- `scripts/run_eval.sh`

Several docs still mention Haiku, automatic batching, or single-judge mode.
These do not match current source behavior.

Fix:

Regenerate docs after deciding whether `none` and batch processing are supported.

### 3.4 `test_api_keys.py` always requires OpenRouter

File:

- `test_api_keys.py:147-154`

The script marks OpenRouter missing as a failure even for a user who intends to
run a truly Anthropic-only/single-judge mode. This is less serious while
`--committee none` is broken, but it should be aligned with supported modes.

### 3.5 `test_api_keys.py` does not close the Anthropic async client

File:

- `test_api_keys.py:87-94`

The test creates `anthropic.AsyncAnthropic` without using an async context
manager or explicit close.

### 3.6 `read_jsonl()` strips leading/trailing whitespace before JSON parse

File:

- `rag_eval/data.py:61-67`

This is usually fine, but it changes exact line content for diagnostics and
prevents reporting precise bad input. It matters once malformed-line logging is
added.

### 3.7 Report includes a warning marker in generated markdown

File:

- `rag_eval/evaluator.py:387-388`

The v3 notes say console/report emojis were removed in an earlier pass, but the
current report uses a warning marker. This is cosmetic, not a metric bug, but it
is inconsistent with earlier formatting goals.

---

## 4. Recommended Fix Order

1. Move cost summary creation before report/detail writes.
2. Repair or remove `--committee none`.
3. Add evaluator-level bounded concurrency and actually use `batch_size`.
4. Fix `_parse_judge_response()` boolean parsing.
5. Make behavior prompts evidence-aware for refusals, Type 4, and Type 5.
6. Make YAML safe in multi-input runs; avoid fixed report paths overriding
   per-file output directories.
7. Share config loading with `run_evaluation_batch.py`.
8. Add per-sample error isolation for `skip_on_error`.
9. Fix runner exit codes and batch cleanup.
10. Synchronize `setup.sh`, `run_eval.sh`, tests, and docs with current Sonnet
    and config behavior.

---

## 5. Notes On Already-Fixed Items

These earlier issues appear fixed in current source:

- Behavior prompt now interpolates the selected rubric, not the whole rubric
  dict (`rag_eval/judge_prompts.py:52-71`).
- `conflict_category_id=0` no longer maps to 1 (`rag_eval/evaluator.py:53-68`).
- Per-type aggregation no longer crashes on unknown types
  (`rag_eval/evaluator.py:285-324`).
- `model_output` no longer falls back to `final_grounded_answer.answer`
  (`rag_eval/data.py:117-139`).
- Dataset-level GR F1 now exists separately from per-sample GR accuracy
  (`rag_eval/metrics.py:247-275`).
- `single_truth_recall` no longer averages structural Type 3 zeros when
  `single_truth_applicable=False` (`rag_eval/evaluator.py:304-309`).
