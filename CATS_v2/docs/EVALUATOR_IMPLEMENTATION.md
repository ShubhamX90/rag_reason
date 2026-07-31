# CATS v2 Evaluator Implementation Contract

**Status:** Current implementation-level guide for the executable CATS evaluator.

**Scope:** This document explains the code path from a prepared JSONL input to
per-sample metric records, dataset summaries, Markdown reports, JSON artifacts,
committee calls, cache behavior, and failure/missingness semantics. It is the
implementation companion to [`CATS_METRICS_METHODOLOGY.md`](CATS_METRICS_METHODOLOGY.md)
and [`LOCAL_COMMITTEE_GUIDE.md`](LOCAL_COMMITTEE_GUIDE.md).

The current paper-facing benchmark path is the local OpenAI-compatible
three-judge committee. The evaluator still contains compatibility code for
remote providers, Codex CLI judges, and an optional standalone NLI path, but
those are not part of the active benchmark prompt bundle unless explicitly
enabled by configuration.

## 1. Executable entry point

The command-line entry point is [`../run_evaluation.py`](../run_evaluation.py).
The normal benchmark invocation is:

```bash
python run_evaluation.py \
  --input inputs/prepped_model_eval_inputs/benchmark_set_all_modes/<model>/<family>/<prompt>/<train_type>/input.jsonl \
  --config configs/benchmark_local_openai_3judge_qwen397.yaml \
  --committee local
```

The CLI accepts one or more `--input` paths. With one input, the supplied
`--output-dir` is used directly. With multiple inputs, a timestamped directory
is made for each input under the output root. The paper-facing orchestrator
normally supplies a unique output directory so timestamps do not become the
experiment identity.

### 1.1 CLI controls

| Argument | Meaning |
| --- | --- |
| `--input` | One or more evaluator-ready JSONL files. Required. |
| `--output-dir` | Output directory for the report and detailed JSON. |
| `--committee default` | OpenRouter default committee; requires the relevant API key. |
| `--committee cli` | Codex CLI judge committee. |
| `--committee local` | Local OpenAI-compatible committee defined by YAML. |
| `--committee none` | Disables the committee and is intended only for alternate or diagnostic paths. |
| `--config` | YAML file whose recognized fields override base settings. |
| `--batch-size` | Number of examples scheduled in each evaluator batch. |
| `--max-samples` | Optional prefix limit for smoke tests; never use for a final result. |
| `--process-sequentially` | Sequential handling for multiple input files. |
| `--verbose` | Enables verbose logging. |

### 1.2 Configuration precedence

`setup_config` creates an `EvaluationConfig`, applies CLI paths and pipeline
flags, selects a committee preset, then applies recognized YAML fields. YAML
values win for fields that it specifies; CLI values remain authoritative for
controls that are not represented by YAML.

The local path has an additional guard: `--committee local` requires a YAML
committee definition with local OpenAI-compatible judges. Provider-specific API
keys are checked after YAML overrides are applied. A YAML file can therefore
switch a nominal default invocation from remote to local behavior.

## 2. Module map

| Module | Contract |
| --- | --- |
| `rag_eval/config.py` | Dataclasses, provider definitions, committee factories, default priorities, and YAML-compatible settings. |
| `rag_eval/data.py` | JSONL loading, model-output access, gold-answer access, answerability precedence, and data helpers. |
| `rag_eval/metrics.py` | Refusal detection, think-trace cleanup, claim/citation extraction, grounded-refusal metrics, and text normalization. |
| `rag_eval/conflict_eval.py` | BA, active committee FG-v2, optional legacy NLI FG, and STR task functions. |
| `rag_eval/judge_committee.py` | Provider transport, JSON parsing, error records, cache access, parallel judge calls, weighted aggregation, and cost accounting. |
| `rag_eval/evaluator.py` | Per-sample orchestration, applicability, hierarchical CATS, summary/report generation, and deterministic output ordering. |
| `rag_eval/batch_processor.py` | Optional provider-specific batch support for Anthropic-style requests. |
| `rag_eval/logging_config.py` | Shared logging configuration. |
| `run_evaluation.py` | CLI parsing, config loading, input loading, evaluator lifecycle, and top-level logging. |

The evaluator does not import the human-evaluation CLI. Human evaluation is a
separate validation package with its own normalization and receipt contract.

## 3. Input contract

The evaluator reads JSON Lines. Each row is one model-output example and keeps
the stable `id` used throughout prepared inputs, per-sample results, committee
cache keys, audits, and human-study alignment.

### 3.1 Expected fields

| Field | Role |
| --- | --- |
| `id` | Stable sample identifier. A positional fallback is created only when absent. |
| `query` | User question supplied to judgment prompts. |
| `retrieved_docs` | Retrieved evidence, normally containing `doc_id` and `snippet`. |
| `per_doc_notes` | Gold document annotations, including verdict, key fact, quote, and source/temporal metadata where available. |
| `conflict_category_id` | Canonical numeric type 1 through 5 when available. |
| `conflict_type` | Human-readable taxonomy fallback. |
| `expected_response` | Gold answerability/abstention and sometimes the gold answer. |
| `gold_answer` or `expected_response.answer` | Target used by STR when applicable. |
| `model_output` | Clean model answer used by the evaluator. |
| `model_output_raw` | Optional preserved raw output, including reasoning traces. |
| `model_output_field` | Preparation provenance identifying the source field. |
| `model_output_source` | Preparation provenance identifying the source export. |

`validate_eval_input_jsonl.py` is the pre-launch validator for the prepared
benchmark profile. It checks row count, id order, metadata, non-empty output,
absence of scaffolding markers in the cleaned output, canonical gold-field
alignment, and source-path existence.

### 3.2 Gold answerability precedence

`gold_answerable_from_record` follows this precedence:

1. If `expected_response` is a mapping with a Boolean `abstain`, use
   `not abstain`.
2. Otherwise, if `answerable_under_evidence` is present, use its Boolean value.
3. Otherwise, infer answerability from gold document notes with a positive or
   accepted partial-support verdict.

This prevents fallback inference from overriding explicit gold supervision.

### 3.3 Conflict-type normalization

The evaluator accepts numeric or textual taxonomy values. Numeric values 1
through 5 are used directly. Invalid numeric sentinels are mapped through the
text field when possible. Type-3 spelling variants with 'and' and 'or' are
treated as the same type; type-4 temporal wording is also recognized.

If neither field is parseable, the implementation logs a warning and falls
back to type 1. This is defensive compatibility behavior, not a substitute for
running the taxonomy normalizer and strict input validator before final runs.

## 4. Per-sample execution

`EnhancedEvaluator._evaluate_single_sample` executes the following sequence for
each record.

### 4.1 Stable identifiers and context

The evaluator obtains `sample_id`, query, normalized conflict type, notes, and
gold answerability. It strips visible think traces from the model output and
determines `pred_answered` using the shared refusal detector.

### 4.2 Document context construction

Retrieved documents are merged with matching `per_doc_notes` by `doc_id`. The
committee FG prompt receives a compact record containing:

```text
doc_id
snippet
verdict
key_fact
quote
```

Gold note verdicts define the eligible support pool. The committee cannot
promote a gold-negative document into an eligible support document merely
because its text looks plausible.

### 4.3 Grounded-refusal decision

The binary per-example decision is:

```text
gr_accuracy_i = 1[pred_answered_i == gold_answerable_i]
```

The evaluator also stores predicted and gold answer/refusal labels. Dataset-
level GR precision, recall, F1, answer-positive, and refusal-positive
diagnostics are calculated from the resulting confusion counts.

### 4.4 Correct-refusal branch

When the model refuses and the evidence is genuinely unanswerable, the example
is a correct refusal. The evaluator stores decision correctness but marks BA,
FG, and STR as inapplicable for their component means. Placeholder values are
retained in the per-sample schema only to keep output shape stable.

The aggregate represents the correct refusal through its decision gate. It does
not silently treat a skipped behavior or answer-content score as an observed
zero.

### 4.5 Answer-content branch

For non-correct-refusal examples, the evaluator:

1. extracts up to the configured maximum number of claims and citations;
2. calls the BA committee with query, answer, conflict type, and retrieved docs;
3. calls committee FG-v2 with claims, eligible annotated docs, query, and full
   cleaned answer;
4. calls STR only when a gold answer exists and the type is configured as
   single-truth applicable; and
5. stores both binary judgments and continuous committee consensus where
   available.

The active benchmark configuration uses up to eight claims per answer. The
human package uses a separate cap of twelve; this difference must be disclosed
in cross-system comparisons.

## 5. Active judgment tasks

### 5.1 Behavior Adherence

The committee returns binary `adherent` plus confidence, rationale, vote
counts, weighted totals, minority confidence, failure state, and committee
details. The evaluator stores:

```text
behavior_score           = 1 if adherent else 0
behavior_consensus_score = weighted_for / (weighted_for + weighted_against)
                            when weighted totals exist
```

The continuous consensus preserves distinctions such as a narrow two-of-three
decision instead of turning every majority into one. The aggregate uses this
continuous consensus when it is available.

### 5.2 Committee Factual Grounding v2

The active FG path is `committee_factual_grounding_v2`. It uses deterministic
claims and gold-eligible documents, then asks the committee whether each claim
is semantically supported by a single document or an allowed cross-document
combination and whether the model cited a supporting document.

For claim `k`:

```text
y_k = 1 if supported and citation-linked
      0 otherwise
FG = sum(y_k) / evaluable_claim_count
```

The result stores claim details, supporting documents, cross-document
combination, cited documents, support reason, and committee error state.

### 5.3 Single-Truth Recall

STR is a semantic judgment that the model asserts a gold target as its own
conclusion. An exact match is one; a sufficiently uncertain negative judgment
can receive a half-credit partial match under the committee rule. If there are
multiple gold answers, an exact match to any one is sufficient for full recall;
partial matches are normalized by the number of gold candidates.

Type 3 is normally not STR-applicable because it lacks a single truth target.
The exact configured type set is part of each run's `run_config.yaml` and must
be read rather than inferred from a report heading.

### 5.4 Grounded Refusal is deterministic

The active benchmark does not use the committee to decide whether a response
should have refused. Answer/refusal prediction is derived from the response and
gold answerability; GR metrics are computed deterministically. The optional
standalone NLI path is not the active benchmark FG path.

## 6. Committee transport and cache mechanics

`JudgeClient` supports Anthropic, OpenRouter, direct DeepSeek, local OpenAI-
compatible, and Codex CLI transport modes. The local benchmark uses the local
OpenAI-compatible transport and sends JSON-only judge requests.

### 6.1 Parallelism

`JudgeCommittee` schedules all configured judges for one judgment concurrently,
then applies a semaphore to enforce `max_concurrent_requests`. The evaluator
processes examples in batches. Batch size and outbound concurrency are
separate:

- batch size controls how many example coroutines are in flight;
- max concurrency controls committee calls to judge endpoints.

Increasing both without checking GPU capacity can produce timeouts, queue
saturation, or incomplete cache writes.

### 6.2 Timeout and failure records

Each judge call can be wrapped in an evaluator timeout. A timeout or transport
exception becomes an explicit judge response with an error field; it is not
silently converted into a successful vote. If all judges fail, the committee
returns an `all_failed` decision and the downstream result retains that state.

Final results must be checked for all-failure samples, invalid judge counts, and
unexpected zero denominators.

### 6.3 Persistent response cache

The cache key is derived from task mode, model identity, and a hash of the full
prompt. Supported modes are:

| Mode | Read | Write | Intended use |
| --- | --- | --- | --- |
| `off` | no | no | Fresh all-at-once run. |
| `read_write` | yes | yes | Collection or resumable run. |
| `read_only` | yes | no | Final aggregation without model servers. |
| `write_only` | no | yes | Explicit collection-only behavior. |

The three staged benchmark collection configs must share the same cache
directory with the final read-only config. Changing prompt text, model id,
judge task, or relevant input context changes the prompt hash and creates a
cache miss.

### 6.4 Deterministic output order

Examples are evaluated asynchronously, but `sample_results` are sorted by
`sample_id` before writing. Repeated runs over the same input are therefore
diffable and completion order cannot become a hidden output difference.
## 7. Aggregation and output schema

The evaluator returns three main objects:

```text
conflict_overall
conflict_per_type
gr_dataset_metrics
```

The JSON artifact wraps these under `summary` and stores the complete ordered
`per_sample` list.

### 7.1 Per-sample schema

Important fields include:

```text
sample_id
conflict_type
pred_answered
gold_answerable
correct_refusal
gr_accuracy
behavior_score
behavior_applicable
behavior_consensus_score
behavior_details
factual_grounding_score
factual_grounding_applicable
factual_grounding_details
single_truth_recall_score
single_truth_applicable
single_truth_recall_details
```

The detail objects are needed for audit, post-hoc citation analysis,
disagreement investigation, and independent recomputation.

### 7.2 Aggregate bucket fields

Overall and per-type buckets retain counts and score means separately:

```text
n
correct_refusals
behavior_n
factual_grounding_n
single_truth_recall_n
answer_quality_n
cats_answerable_n
cats_refusal_required_n
cats_unscorable_n
cats_complete
```

Applicability counts are first-class result fields, not metadata that can be
discarded after copying a scalar score into a workbook.

### 7.3 Current CATS aggregate fields

The active aggregate version is `cats_h_gated_harmonic_v1`. The evaluator stores
`cats_prevalence_score`, `cats_balanced_score`, the older equal-type diagnostic
`cats_type_balanced_score`, `cats_flat_legacy_score`, completeness, and
unscorable counts. Exact formulas and paper status are specified in
[`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md).

## 8. Reports and cost artifacts

Each run writes:

```text
eval_report.md
detailed_results.json
run_config.yaml             # for orchestrated/config-rendered runs
```

The Markdown report includes overall and per-type metrics, applicability counts,
CATS summaries, GR confusion diagnostics, cost information, and unsupported FG
claim details when available. The JSON artifact is authoritative for numeric
recomputation.

Cost summaries contain total cost, decisions made, average cost per decision,
and per-judge request/token cost when a provider exposes usage. Local and Codex
CLI runs are marked unmetered where billing tokens are unavailable; zero cost in
such a report does not mean zero compute cost.

## 9. Active versus legacy paths

The source still contains compatibility code for standalone NLI judging, older
single-judge and alternate FG logic, remote providers, Codex CLI mode, legacy
flat CATS fields, and historical configs/output trees. These paths are not
interchangeable with the active benchmark path.

The active benchmark identity is defined by the current config,
`run_config.yaml`, prompt bundle, committee model ids, priorities, cache mode,
and aggregate version. Seeing a function or config field in source does not
make it part of the paper-facing result.

## 10. Implementation audit procedure

For any final run, verify:

1. the input passes `validate_eval_input_jsonl.py` in `benchmark_prepped` mode;
2. `run_config.yaml` names the intended input, output, committee, priorities,
   prompt-affecting options, and cache mode;
3. every expected row has a `per_sample` result;
4. sample ids are unique and sorted in the detailed JSON;
5. `n` equals the validated input row count;
6. GR counts and F1 recompute from `pred_answered` and `gold_answerable`;
7. applicability counts match the per-sample flags;
8. no unexpected all-failed committee judgments are present;
9. CATS completeness is true before a scalar is used in the master matrix; and
10. the result is included only if its source path belongs to the authoritative
    108-row scope.

## 11. Minimal smoke test

For a local code-path smoke test without running the full benchmark, validate a
small prepared prefix and use `--max-samples`. Treat the output as a test
artifact only. A smoke test cannot establish final benchmark correctness,
committee reliability, or workbook membership.
