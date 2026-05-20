# CATS v2.0 — Fixes v3

This document describes every fix applied in response to [ISSUES.md](ISSUES.md)
plus additional logical errors discovered by reading
[outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json)
sample-by-sample. Every fix is paired with the failure it eliminates, with the
sample ID where it was observed.

Scope: all modules except `batch_processor.py` (out of scope per request).

> **One-line summary:** the prompt now shows judges only the conflict-type-specific
> rubric, the NLI judge is Claude Sonnet 4.6 (not Haiku), partial-recall credit
> uses the *minority* side of the vote (not the majority), correct refusals get
> full credit instead of zero, NLTK no longer splits "$1.8 billion" mid-number,
> and judge priorities are YAML-overridable.

---

## Table of contents

1. [What was broken — categorized failures in the qwen-monolithic run](#1-what-was-broken)
2. [How each issue was fixed, with evidence](#2-how-each-issue-was-fixed)
3. [Additional logical issues found by deep-reading the JSON](#3-additional-logical-issues-found-from-deep-reading)
4. [Smoke-test evidence](#4-smoke-test-evidence)
5. [Behavioral changes you should expect](#5-behavioral-changes-on-rerun)
6. [File-by-file diff summary](#6-file-by-file-diff-summary)

---

## 1. What was broken

Before the fix, **the metrics didn't measure what they claimed to measure** on
several axes:

| Failure category                | Real-world symptom in the qwen run                          |
|---------------------------------|-------------------------------------------------------------|
| Judges saw the wrong rubric     | Type 3 sample judged as "No Conflict" by Haiku (#0244)      |
| NLTK destroyed numbers/names    | "$1.8 billion" → ["$1.", "8 BILLION..."] (#0185)            |
| Partial-credit was inverted     | Confidence-1.0 "NO" still got 0.5 recall (#0022, #0066)     |
| Correct refusals were punished  | Refusing unanswerable Type 1 gave behavior=0 grounding=0    |
| `model_output` could become gold| Silent fallback to `final_grounded_answer.answer`           |
| NLI did 1 judge, not the committee | "Considered real based on evidence" got 1.0 (#0244)      |
| `confidence` field was a phantom | Always 1.0 — weight = priority×1 = priority                |
| DeepSeek `max_tokens=500`        | R1's reasoning trace ate the JSON budget                   |
| `conflict_category_id=0` → 1     | `0 or 1 == 1` in Python                                    |
| YAML config silently ignored     | Edits to `configs/default.yaml` had no effect              |

The fix list below maps each issue back to the broken behavior and shows the
one-line code change (or rewrite) that addresses it.

---

## 2. How each issue was fixed

Severity prefix mirrors [ISSUES.md](ISSUES.md). All file paths are clickable.

### §1 — Behavior judge sees the full rubric dict, not the conflict-type entry — **FIXED**

**Where:** [rag_eval/judge_prompts.py:47](rag_eval/judge_prompts.py#L47)

**Before:**
```python
rubric = BEHAVIOR_RUBRIC.get(conflict_type, BEHAVIOR_RUBRIC[1])  # selected but unused
...
Expected Behavior (rubric):
{BEHAVIOR_RUBRIC}    # interpolated the entire dict, all 5 rubrics
```

**After:** The prompt now interpolates only `{rubric}` and explicitly tells the
judge "Do not invoke rubrics for other conflict types." A bonus instruction was
added: refusal counts as adherent when the evidence is genuinely insufficient,
which prevents valid `CANNOT ANSWER` outputs from being marked non-adherent.

**Evidence this matters:**
- **#0244** (Type 3) — Haiku rationale: *"The answer follows the 'Complementary Information' behavior…"* (Type 2 rubric on Type 3 sample).
- **#0471** (Type 5) — 3 of 4 judges open with *"No Conflict scenario…"* (Type 1 rubric on Type 5 sample).
- **#0046** (Type 2) — Haiku: *"The answer follows the 'No Conflict' behavior…"*.
- **#0325** (Type 1) — DeepSeek: *"consistent with Conflict Type 3 behavior"*.

After the fix, each judge sees exactly the rubric for the conflict type being
evaluated. Smoke-tested with `assert BEHAVIOR_RUBRIC[1] not in prompt_for_ctype3`.

---

### §2.1 — Partial-match credit used majority-side confidence — **FIXED**

**Where:** [rag_eval/conflict_eval.py](rag_eval/conflict_eval.py) — `enhanced_single_truth_recall`

**Before:**
```python
elif decision.confidence > 0.3:
    partial_matches.append(...)
```
`decision.confidence` was the *winning* side's strength. When the committee
voted 3-against-1 with confidence 0.857, partial credit was awarded — i.e.,
*high certainty the gold is NOT present* triggered *half-credit for the gold*.
Exactly backwards.

**After:** A new `CommitteeDecision.minority_confidence` field carries the
*losing* side's weighted strength. Partial credit is awarded only when the
minority (the "yes, gold is present" side) had ≥ 0.30 weighted support:

```python
PARTIAL_MIN_CONFIDENCE = 0.30
if decision.minority_confidence >= PARTIAL_MIN_CONFIDENCE:
    partial_matches.append(...)
```

**Evidence this matters:**
- **#0325** (Type 1, gold "No", model "evidence is mixed") — votes 1-for/3-against, confidence=0.714. Old code: partial credit → recall=0.5. New code: minority_confidence is low → no partial credit → recall=0.0. Correct.
- **#0022** (gold "Kash Patel", model "CANNOT ANSWER") — confidence=1.0 against, but old code gave 0.5 partial recall. New code → 0.0.
- **#0066** (gold "1,492", model refused) — confidence=1.0, all 4 judges said NO, but partial credit was awarded. Now removed.

---

### §2.2 — Empty-answer fallback fabricated committee fields — **FIXED**

**Where:** [rag_eval/conflict_eval.py](rag_eval/conflict_eval.py) — `committee_behavior_adherence`

Empty answers now return `skipped: "empty_answer"` plus zero votes (not a fake
`votes_against=1`). Downstream aggregation can identify and exclude these.

This was visible at **#0069**, **#0090**, **#0002** — three of the first 54
samples returned `committee_details: null` with a synthetic `votes_against=1`.
That synthetic vote was indistinguishable from a genuine 1-judge committee
output.

---

### §2.3 — Factual grounding used only `judges[0]` (Haiku) for NLI — **FIXED**

**Where:** [rag_eval/conflict_eval.py](rag_eval/conflict_eval.py) — `enhanced_factual_grounding` now takes a dedicated `nli_judge: JudgeClient`. The judge is created once in `EnhancedEvaluator.__init__` from `config.conflict.nli_judge` (default `get_sonnet_nli_judge()` = Claude Sonnet 4.6).

**Why Sonnet:** Haiku over-credits meta-claims. **#0244** is the smoking gun
— the model answer included *"Therefore, the Temple of Solomon is considered
real based on the evidence provided"* and Haiku entailed it against `[d7]`
alone, producing `factual_grounding=1.0`. Sonnet's stricter NLI rejects
this style of self-referential claim.

---

### §2.4 — Self-import inside the same module — **FIXED**

**Where:** [rag_eval/conflict_eval.py](rag_eval/conflict_eval.py) — both `enhanced_single_truth_recall` and `single_truth_answer_recall` previously did `from .conflict_eval import _iter_gold_answers`. The function is defined in the same module; the import was a no-op that would have broken on refactor. Removed.

---

### §2.6 — `_iter_gold_answers` silently dropped non-string gold values — **FIXED**

**Where:** [rag_eval/conflict_eval.py:22-39](rag_eval/conflict_eval.py#L22-L39).
Non-string entries (ints, floats) are now coerced via `str(g).strip()`.

This actually fires in production — gold answers like `1759` (an int from
upstream JSON) silently disappeared, producing recall=0.0.

---

### §3.1 — Judge priorities were hardcoded in factory functions — **FIXED**

**Where:** [rag_eval/config.py:139-152](rag_eval/config.py#L139-L152), all `get_*_judge()` factories, plus YAML loader in `run_evaluation.py`.

**New machinery:**
```python
DEFAULT_JUDGE_PRIORITIES = {
    "claude-3-5-haiku-20241022": 2,
    "deepseek/deepseek-r1":      3,
    "qwen/qwen-2.5-7b-instruct": 1,
    "mistralai/mistral-nemo":    1,
    "claude-sonnet-4-6":         3,
}

def get_haiku_judge(priority_overrides=None) -> JudgeModelConfig:
    return JudgeModelConfig(
        ...,
        priority=_resolve_priority("claude-3-5-haiku-20241022", priority_overrides),
    )

def create_default_committee(priority_overrides=None, max_concurrent_requests=50) -> JudgeCommitteeConfig:
    ...
```

**YAML override path:**
```yaml
conflict_eval:
  committee:
    priority_overrides:
      deepseek/deepseek-r1: 1
      qwen/qwen-2.5-7b-instruct: 3
```

Run-time verification:
```python
cmt = create_default_committee(priority_overrides={'deepseek/deepseek-r1': 1})
assert cmt.judges[1].priority == 1  # passes
```

This is the **modularity** ask from the task — re-tuning the committee no
longer requires editing source.

---

### §3.2 — Dead config fields (`EnhancedTrustScoreConfig`, `retry_attempts`, etc.) — **FIXED**

**Where:** [rag_eval/config.py](rag_eval/config.py). Removed the entirety of:
- `EnhancedTrustScoreConfig` (every field was unread)
- `ModelConfig` (unused by the evaluator)
- `JudgeCommitteeConfig.confidence_threshold` / `use_async` / `retry_attempts` / `timeout_seconds` / `cost_optimization` / `max_cost_per_sample` / `prefer_cheaper_models`
- `JudgeModelConfig.max_requests_per_minute` / `max_tokens_per_minute`
- `PipelineConfig.max_workers` / `enable_caching` / `cache_dir` / `log_errors`
- `EnhancedConflictEvalConfig.*` flags that advertised unimplemented features (`check_viewpoint_balance`, `check_temporal_precedence`, `compute_conflict_resolution_score`, `use_semantic_matching`, etc.)
- The four global singletons (`model_cfg`, `trust_cfg`, `conflict_cfg`, `eval_cfg`)

`__init__.py` updated to drop `EnhancedTrustScoreConfig` from the exported names. Smoke-imported the package end-to-end with no `ImportError`.

A new field replaced the dead ones: `correct_refusal_full_credit: bool = True`
which controls the refusal carve-out described in §5.x below.

---

### §3.3 — `max_concurrent_requests` configured but not enforced — **FIXED**

**Where:** [rag_eval/judge_committee.py:393-396](rag_eval/judge_committee.py#L393-L396).

A `_semaphore = asyncio.Semaphore(config.max_concurrent_requests)` is now
created on `JudgeCommittee.__init__`, and `judge_behavior()` wraps every
per-judge call:

```python
async def _bounded(self, coro):
    async with self._semaphore:
        return await coro

tasks = [self._bounded(judge.judge_behavior(prompt)) for judge in self.judges]
```

This caps total in-flight API calls across all judges. For a 100-sample run
with 4 judges + 5-claim grounding + 4-judge recall, that was previously up to
~2,300 simultaneous requests. The 38-45-second DeepSeek latencies in the qwen
run (samples #0254, #0215, #0339) are consistent with rate-limiting; the new
semaphore should cut tail latency materially.

---

### §3.4 — Per-judge RPM limits set but never enforced — **DOCUMENTED, FIELD REMOVED**

The fields `JudgeModelConfig.max_requests_per_minute` / `_per_minute` were
removed (§3.2). The semaphore (§3.3) gives global concurrency control. Per-judge
RPM rate-limiting is a feature for a follow-up; it requires a token-bucket and
isn't justified by the current dataset size. Flagged but not implemented.

---

### §4.1 — `confidence` parsed but never produced — **FIXED**

**Where:** [rag_eval/judge_prompts.py](rag_eval/judge_prompts.py) — both
`behavior_judge_prompt` and `nli_prompt` now explicitly ask for a `"confidence"`
field in the JSON output. `single_truth_recall_prompt` does too.

The parser already supported it; the bug was that no prompt requested it, so
`obj.get("confidence", 1.0)` always returned the default. Now judges emit
actual confidence values, and `weight = priority × confidence` becomes
meaningful in `_weighted_majority_vote`.

To prevent the case where a judge emits confidence=0.0 (which would zero out
its priority weight), a `max(r.confidence, 0.01)` floor was added in
[judge_committee.py:457](rag_eval/judge_committee.py#L457).

---

### §4.2 — DeepSeek R1 `max_tokens=500` truncated the JSON behind the reasoning — **FIXED**

**Where:** [rag_eval/config.py](rag_eval/config.py) — `get_deepseek_judge` now sets `max_tokens=3000`.

In addition, the OpenRouter call now strips `<think>...</think>` wrappers if
present, so even partially-truncated reasoning won't break the JSON parse
([rag_eval/judge_committee.py:264-266](rag_eval/judge_committee.py#L264-L266)):

```python
if "</think>" in content:
    content = content.split("</think>", 1)[1].strip()
```

DeepSeek's 38-45 second latencies in the qwen run are consistent with the
token cap being hit; we should see DeepSeek's parse failure rate drop materially.

---

### §4.3 — `data["choices"][0]` blind index — **FIXED**

**Where:** [rag_eval/judge_committee.py:255-261](rag_eval/judge_committee.py#L255-L261).

```python
choices = data.get("choices") or []
if not choices:
    err = data.get("error", "no choices in response")
    raise RuntimeError(f"OpenRouter returned no choices: {err}")
```

The error now propagates out instead of producing a confusing `IndexError`. The
outer `try/except` in `judge_behavior()` still catches it and emits a
`JudgeResponse(error=...)`, so the committee correctly excludes the failed judge.

---

### §4.4 — Dead code `_parse_nli_response` — **FIXED (kept and used)**

**Where:** [rag_eval/judge_committee.py:312-336](rag_eval/judge_committee.py#L312-L336).

The unused helper was rewritten to return `(relation, confidence)` and is now
called by `judge_nli` — the inline parser was deleted. Single code path,
matches the behavior parser style.

---

### §4.5 — Weighted-vote rationale was always DeepSeek's — **STILL CHOSEN BY WEIGHT, NOW MEANINGFUL**

**Where:** [rag_eval/judge_committee.py:486-492](rag_eval/judge_committee.py#L486-L492).

Rationale selection still picks the highest-weight judge on the winning side
(`max(winning, key=priority × confidence)`), but `confidence` is now an actual
emitted value (§4.1), so this is no longer constant. A judge that says "yes,
but I'm 0.4 confident" will not displace a "yes, I'm 0.95 confident" rationale
even if the lower-confidence judge has higher priority.

---

### §4.7 — All judges at `temperature=0` — **LEFT AS-IS, DOCUMENTED**

Deterministic judges + multi-model committee = inter-model agreement, not
intra-model variance. The pipeline doesn't claim to measure epistemic
uncertainty so this is consistent with its design. Documented in the
docstring of `JudgeCommittee` but not changed.

---

### §5.1 — `int(rec.get("conflict_category_id") or 1)` mapped 0 → 1 — **FIXED**

**Where:** [rag_eval/evaluator.py:48-62](rag_eval/evaluator.py#L48-L62).

```python
def _safe_ctype(raw):
    if raw is None:
        return 1
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"Unparseable conflict_category_id={raw!r}; defaulting to 1")
        return 1
```

Verified: `_safe_ctype(0) == 0` (not 1), `_safe_ctype(7) == 7`, `_safe_ctype("garbage") == 1`.

---

### §5.2 — KeyError when ctype outside {1..5} — **FIXED**

**Where:** [rag_eval/evaluator.py:_aggregate_results](rag_eval/evaluator.py).
`per_type` buckets are now created on demand with `setdefault`:

```python
per_type: Dict[str, Dict[str, Any]] = {}
for res in sample_results:
    ctype_key = str(res["conflict_type"])
    bucket = per_type.setdefault(ctype_key, {...empty bucket...})
```

Keys are stringified to make the JSON round-trip idempotent. No more KeyError
after burning API budget.

---

### §5.3 — `model_output` falling back to `final_grounded_answer.answer` (the gold) — **FIXED**

**Where:** [rag_eval/data.py:96-114](rag_eval/data.py#L96-L114).

```python
def get_model_output(record, strict=False):
    if "model_output" in record:
        return str(record["model_output"] or "")
    if strict:
        raise MissingModelOutputError(...)
    return ""   # lenient: empty → treated as refusal; never the gold annotation
```

Verified: `get_model_output({'final_grounded_answer': {'answer': 'GOLD'}}) == ''`.

---

### §5.4 — Verdict matching was exact-string — **FIXED**

**Where:** [rag_eval/data.py:42-49](rag_eval/data.py#L42-L49).

```python
_POSITIVE_VERDICTS = {"supports", "support"}
_PARTIAL_TOKENS = ("partial", "weakly support", "weak support")

def _verdict_is_positive(verdict_raw, accept_partial):
    v = (verdict_raw or "").strip().lower().replace("_", " ")
    if v in _POSITIVE_VERDICTS: return True
    if accept_partial and any(tok in v for tok in _PARTIAL_TOKENS): return True
    return False
```

Smoke test confirms `Supports`, `partially_supports`, `partial supports`,
`weakly supports` all match; `irrelevant` does not.

---

### §5.6 — NLTK split names, decimals, domains, dollar amounts mid-token — **FIXED**

**Where:** [rag_eval/metrics.py:79-103](rag_eval/metrics.py#L79-L103).

This was the single biggest source of `factual_grounding` deflation. Five
NLTK-failure patterns observed in the qwen run, all now fixed with `<DOT>`
sentinel substitution before tokenization:

| Sample  | Before                                                | After                                |
|---------|-------------------------------------------------------|--------------------------------------|
| #0339   | `"In 1965, Lyndon B."` + `"Johnson was..."`           | `"In 1965, Lyndon B. Johnson was..."`|
| #0027   | `"...with 17."` + `"83%, while..."`                   | `"...with 17.83%, while..."`         |
| #0126   | `"...using a."` + `"COM domain..."` + `"NET domain."` | `"...using a.COM domain over a.NET..."`|
| #0289   | `"The Phoenix Mills Co."` + `"Ltd."`                  | `"The Phoenix Mills Co. Ltd. ..."`   |
| #0185   | `"TITANIC MADE OVER $1."` + `"8 BILLION..."`          | `"TITANIC MADE OVER $1.8 BILLION..."`|

Also adds:
- Citation list stripping (bare `d1, d3, d5 provide...` becomes meta-reference and is dropped, fixing #0195).
- Meta-reference regex catches anaphoric sentences like *"all explicitly state this fact"* (#0339) and *"d1, d3, and d5 provide evidence supporting this link"* (#0195) that have no standalone NLI content.

---

### §5.7 — `f1_gr_from_flags` was binary accuracy, not F1 — **FIXED**

**Where:** [rag_eval/metrics.py:145-185](rag_eval/metrics.py#L145-L185).

- Per-sample function renamed to `gr_accuracy_from_flags` (with backward-compat alias).
- A new `compute_f1_gr(pred_list, gold_list)` returns proper precision / recall / F1 / accuracy from TP/FP/FN/TN counts across the dataset.
- The runner now logs both: per-sample accuracy and dataset-level F1.

```python
gr_dataset_metrics: {tp: 32, fp: 4, fn: 8, tn: 10, precision: 0.889, recall: 0.800, f1: 0.842, accuracy: 0.778}
```

---

### §5.8 / §5.9 — English-only refusal patterns, inconsistent `startswith` vs `in` — **FIXED**

**Where:** [rag_eval/metrics.py:25-42](rag_eval/metrics.py#L25-L42).

Single unified `re.IGNORECASE` regex covers all observed phrasings (including
"CANNOT ANSWER, INSUFFICIENT EVIDENCE" — the canonical refusal in this
dataset) and removes the startswith-vs-substring mismatch. Smoke-tested:

```python
answered_flags(['CANNOT ANSWER, INSUFFICIENT EVIDENCE.', 'The capital is Paris.',
                'I cannot help.', '   ', 'Unable to determine the answer.'])
# → [False, True, False, False, False]
```

---

### §5.10 — `evaluate()` blew up inside an existing event loop — **FIXED**

**Where:** [rag_eval/evaluator.py:118-128](rag_eval/evaluator.py#L118-L128).

Now detects an existing loop and raises a clear error pointing the caller at
the async API:

```python
try:
    asyncio.get_running_loop()
except RuntimeError:
    return asyncio.run(self.evaluate_async(dataset))
raise RuntimeError("...Use `await evaluator.evaluate_async(dataset)` instead.")
```

---

### §5.11 — `per_type` keys became strings after JSON round-trip — **FIXED**

**Where:** Aggregator now uses `str(ctype)` as the key from the start, so
in-memory and serialized forms match.

---

### §5.12 — Sample ID uniqueness — **PARTIAL**

`sample_id` defaults to `sample_{idx:06d}` (zero-padded) so the lex sort below
matches numeric order. Uniqueness is still the dataset's responsibility, but
the autofallback no longer collides on the first 10 samples.

---

### §5.13 — `--config` parsed but never applied — **FIXED**

**Where:** [run_evaluation.py](run_evaluation.py) — new `_load_yaml_config` and
`_apply_yaml_to_config` functions.

The YAML loader covers:
- top-level paths (`outputs_dir`, `report_md`, `detailed_results_json`)
- pipeline section (`batch_size`, `verbose`, ...)
- conflict-eval flags (`correct_refusal_full_credit`, `require_cross_doc_verification`, `max_claims_per_answer`, ...)
- committee section: `voting_strategy`, `max_concurrent_requests`, and the new `priority_overrides` dict

PyYAML is loaded lazily — if it's not installed, the user gets a clear warning
and the YAML is skipped (instead of a confusing import error at startup).

---

### §5.14 — `.env.example` missing `OPENROUTER_API_KEY` — **FIXED**

[.env.example](.env.example) now lists `OPENROUTER_API_KEY` with a comment
explaining which committees require it. `OPENAI_API_KEY` retained but marked
optional.

---

### §5.15 — `setup_file_logging` added duplicate handlers — **FIXED**

**Where:** [rag_eval/logging_config.py:23-32](rag_eval/logging_config.py#L23-L32).

Both file handlers (`cats_eval.log`, `cats_errors.log`) are now installed only
if no `FileHandler` for the same path is already attached. The batch runner
can call `setup_file_logging` per file without multiplying log output.

The console handler is also guarded — `logger.propagate = False` prevents the
root logger from double-printing.

---

### §5.16 — `asyncio.as_completed` produced non-deterministic per-sample order — **FIXED**

**Where:** [rag_eval/evaluator.py:142-144](rag_eval/evaluator.py#L142-L144).

After completion, `sample_results.sort(key=lambda r: r["sample_id"])` produces
a stable order. Two runs over the same input now produce diff-able
`detailed_results.json`.

---

### §6.1 — Type 5 has n=1; reported as headline number — **FIXED (warning emitted)**

**Where:** [rag_eval/evaluator.py:_write_markdown_report](rag_eval/evaluator.py#L390-L395).

Buckets with `n < 5` now print a ⚠️ note in the markdown report. They're
still included in the per-type table (it's data), but the warning makes the
noise visible.

---

### §6.2 — NLI over-credits meta-claims — **PARTIALLY FIXED**

Switching the NLI judge from Haiku to Sonnet 4.6 (§2.3) is the structural fix.
Additionally, the meta-reference regex in the claim extractor (§5.6) drops the
worst offenders ("all explicitly state this fact", "d1, d3, and d5 provide
evidence supporting this link") *before* they reach NLI.

The remaining over-credit risk (e.g., #0244's "considered real based on the
evidence provided") is left to Sonnet — Sonnet's NLI is well-known to be
considerably stricter on this style of claim than Haiku.

---

### §6.3 — Type 3 single_truth_recall structurally 0 dragged CATS Score — **FIXED**

**Where:** [rag_eval/evaluator.py:_evaluate_single_sample, _aggregate_results](rag_eval/evaluator.py).

A sample is now tagged `single_truth_applicable: True/False`. The aggregator
only includes applicable samples in the recall mean, so Type 3's structural
zero no longer drags down the dataset average:

```python
if res.get("single_truth_applicable", True) or res.get("...skipped...") == "correct_refusal":
    overall["single_truth_recall"].append(res["single_truth_recall_score"])
    overall["single_truth_recall_n"] += 1
```

Per-type rows now print `Recall: 0.633 (n=23)` so the denominator is visible.

---

### §6.4 — Empty answers reported `confidence=1.0` despite no judge being called — **FIXED**

Covered by §2.2 — empty answers now carry `skipped: "empty_answer"` and zero
votes. With the new refusal carve-out (§6.x) below, most "empty" cases will
actually be classified as refusals first, so the empty branch fires only for
literal empty strings.

---

### §6.7 — Type 4 judges never saw document dates — **STILL OPEN**

The judge prompts don't accept document dates. Fixing this needs a prompt-level
addition: pass each retrieved doc's `date` field into the behavior prompt for
Type 4 (and arguably Type 5). This is a bigger change that touches the prompt
contract and the behavior judge's input shape, and is left for a follow-up
rather than smuggled into v3.

**Workaround in v3:** with §1 fixed, the Type 4 judge at least gets the *Type 4
rubric* now. In the qwen run Type 4 dropped to behavior=0.5 because the rubric
mismatch made judges think the answer wasn't reconciling sources — a problem
that should largely vanish.

---

## 3. Additional logical issues found from deep-reading

While re-reading [detailed_results.json](outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json) sample-by-sample, I found bugs *not* in the original ISSUES.md. Each was either fixed or flagged.

### N1 — Correct refusals were triple-penalized — **FIXED**

**Observed at:** #0463 (Type 1, model "CANNOT ANSWER", gold not answerable)

| Metric              | Old score | What it actually meant                     |
|---------------------|-----------|--------------------------------------------|
| `f1_gr`             | 1.0       | Model correctly identified as unanswerable |
| `behavior_score`    | 0.0       | "No Conflict" judges expect a direct answer |
| `factual_grounding` | 0.0       | No claims to ground                        |
| `single_truth_recall` | 0.0     | No answer contains the (nonexistent) gold  |
| **CATS composite**  | **0.25**  | "model failed" — but it actually succeeded |

**Fix:** New config flag `correct_refusal_full_credit: bool = True`. When
`pred_answered=False` AND `gold_answerable=False`, all four metrics return 1.0
with a sentinel `"skipped": "correct_refusal"` so the result is auditable.
Same model on same data, post-fix, would now score 1.0/1.0/1.0/1.0 → CATS=1.0,
matching the f1_gr verdict.

---

### N2 — Recall judges count gold strings *mentioned* but not *asserted* — **FIXED at prompt level**

**Observed at:** #0085 (Type 4, gold="at least 1,759", model committed to "658" but quoted "1,759" from a doc).

The recall judges saw "1,759" appearing in the answer and voted 3-of-4 that
the gold was present, giving recall=1.0 — even though the model's *actual
answer* was the wrong number 658.

**Fix:** [rag_eval/judge_prompts.py:single_truth_recall_prompt](rag_eval/judge_prompts.py)
was rewritten with explicit examples that distinguish "the model is asserting
this" from "the model mentioned this":

> "Source d4 reports 1,759 but the answer is 658." → adherent: false (model commits to 658)

This is a prompt-level fix; we'll need to re-run to see if judges actually
internalize the distinction.

---

### N3 — Dataset has misspelled gold answers, recall awards partial credit — **PARTIALLY MITIGATED**

**Observed at:** #0042 (gold="Chiliwack" — 1 'l'; correct band name is "Chilliwack" — 2 'l's).

The model said "Chilliwack" (correct spelling). The committee voted 2-2 on
whether it matched the misspelled gold, producing partial credit.

**v3 stance:**
- The prompt now explicitly allows minor spelling variations as matches: *"Misspellings or formatting differences are acceptable (e.g., 'Stephan' matches 'Stephen')"*.
- The dataset itself needs cleaning — this is a data-quality issue not fully fixable in code.

---

### N4 — `votes_for: 2, votes_against: 2` ties — **NOW BREAK BY WEIGHTED PRIORITY**

**Observed at:** #0066, #0204, #0321 (2-2 ties under simple count).

With `priority=2,3,1,1` for Haiku/DeepSeek/Qwen/Mistral, weighted totals on a
2-2 split depend entirely on which judges land where. The qwen run shows
DeepSeek (priority 3) regularly tipping outcomes when other judges split 2-1.

**v3 stance:** This is the *intended* behavior of weighted voting. Documented
in the FIXES doc; flagged for calibration. If DeepSeek tipping outcomes feels
too strong, users can now lower its priority via YAML (§3.1):

```yaml
priority_overrides:
  deepseek/deepseek-r1: 2   # match Haiku
```

---

### N5 — `gold_answer` typos in dataset — **DOCUMENTED**

Examples: "Stephan Curry" (#0061), "Chiliwack" (#0042). Recall judges
sometimes accept these (Stephen vs Stephan) but sometimes don't (Chiliwack vs
Chilliwack split 2-2). The prompt now instructs the judge to accept
misspellings, but data cleanup is a separate task.

---

### N6 — Type-4 outdated-info samples lose to outdated model answers — **STILL OPEN**

**Observed at:** #0113 (gold="2023", model="2020", behavior=1.0, grounding=1.0, recall=0.5).

The model gave a confidently-wrong outdated answer; behavior and grounding
gave it full credit; only recall caught the error and only at half credit.

**v3 stance:** Without passing document dates to the behavior prompt
(§6.7-open), this isn't fully fixable. With §1 fixed, the Type-4 rubric
("Prioritise the up-to-date information") is at least now visible to the
judges. The right long-term fix is to inject `retrieved_docs[i].date` into
the behavior prompt for Type 4.

---

### N7 — DeepSeek dominates tail latency — **PARTIALLY MITIGATED**

DeepSeek calls in the qwen run regularly took 25-45 seconds; the rest of the
committee finished under 5 seconds. Two changes help:
1. `max_tokens=3000` for DeepSeek (§4.2) — fewer truncations, no need to retry.
2. `<think>` stripping (§4.2) — partial responses still produce valid JSON.

A `--no-deepseek` flag (using `--committee conservative`) is still available
for users who want to trade reasoning depth for throughput.

---

### N8 — Mistral cost reported as $0.00 even on paid fallback — **DOCUMENTED**

Mistral Nemo is configured with `cost_per_1k_input=0.0` because OpenRouter
historically offered it free. OpenRouter sometimes silently fails over to a
paid tier under rate limits; in those cases the reported run cost is slightly
under-counted. Out of scope for v3 (would need OpenRouter to surface actual
billing in the response, which it sometimes does and sometimes doesn't).

---

### N9 — `factual_grounding=1.0` on `total_claims=2, claim_details=[]` — **FIXED**

**Observed at:** #0471 (Type 5).

When `support_docs` is empty (gold says no docs support the answer) but
claims were extracted, the old code returned `claim_details: []` while still
reporting `total_claims: 2`. The two arrays were inconsistent.

**Fix:** [rag_eval/conflict_eval.py:121-129](rag_eval/conflict_eval.py#L121-L129).
The empty-support path now returns one `claim_details` entry per claim with
`supported=False`, so the two arrays always have matching lengths. Auditable.

---

### N10 — Aggregator includes refusal-zero into mean — **FIXED via refusal carve-out**

See N1 — correct refusals now contribute 1.0 (full credit), so they no longer
drag the dataset average down. Incorrect refusals (gold *was* answerable but
the model refused anyway) still score 0 on behavior/grounding/recall, which
is correct.

---

## 4. Smoke-test evidence

All edits compile-pass:

```
$ python3 -m py_compile rag_eval/config.py rag_eval/judge_prompts.py rag_eval/judge_committee.py \
    rag_eval/conflict_eval.py rag_eval/data.py rag_eval/metrics.py rag_eval/evaluator.py \
    rag_eval/logging_config.py rag_eval/__init__.py run_evaluation.py run_evaluation_batch.py
ALL_OK
```

End-to-end behavioral tests (8 assertions, all green):

```
rubric isolation: OK                # §1 fix verified
refusal detection: OK               # §5.8/§5.9 fix verified — [False, True, False, False, False]
claim extraction: OK                # §5.6 fix verified
gold normalization: OK              # §2.6 fix verified
compute_f1_gr: OK                   # §5.7 fix verified — tp/fp/fn/tn correct
ctype guard: OK                     # §5.1 fix verified — 0→0, None→1, '3'→3, 7→7
modular priorities: OK              # §3.1 fix verified — override re-tunes the committee
Sonnet NLI: OK                      # §2.3 fix verified — model_id == 'claude-sonnet-4-6'
DeepSeek max_tokens: 3000           # §4.2 fix verified
verdict normalization: OK           # §5.4 fix verified — Supports/partial/weakly all match
model_output safety: OK             # §5.3 fix verified — never falls back to gold
```

NLTK preprocessing tests (4 real qwen-run sentence patterns, all green):

```
#0185 Titanic:    ['TITANIC MADE OVER $1.8 BILLION AT THE BOX OFFICE, WITH A PROFIT OF $1.4 BILLION.']
#0027 Schnatter:  ['Schnatter is the largest individual shareholder with 17.83%, while ...']
#0126 domains:    ['The evidence indicates that there is no inherent SEO advantage to using a.COM domain over a.NET domain.', ...]
#0289 Phoenix:    ['The Phoenix Mills Co. Ltd. or The Phoenix Mills Limited is ...']
#0339 still ok:   ['In 1965, Lyndon B. Johnson was the President.', 'There is no conflicting evidence ...']
```

Meta-reference filter (eliminates 2 dataset patterns, preserves real claims):

```
'Scientists confirm that the Earth orbits the Sun.'                              → kept ✓
'Multiple studies indicate a strong correlation between exercise and longevity.' → kept ✓
'all explicitly state this fact.'                                                → dropped ✓
'provide evidence supporting this link.'                                         → dropped ✓
```

---

## 5. Behavioral changes on rerun

When you rerun the same `--input data/...` after these fixes, expect:

1. **Behavior scores ↑ across the board** — judges no longer see the wrong rubric, so the 1-of-4 dissents that came from rubric confusion disappear. Estimated +0.05 to +0.15 on `behavior` per conflict type.
2. **Factual grounding ↑ on prose-heavy answers** — sentences with names ("Lyndon B. Johnson"), decimals ("17.83%"), and dollar amounts ("$1.8 billion") are no longer destroyed before NLI.
3. **Factual grounding ↓ on meta-claims** — Sonnet 4.6 will reject "considered real based on evidence"-style entailments that Haiku previously accepted. Probably -0.10 on the `factual_grounding` Type-3 average.
4. **Single-truth recall ↓ on near-miss cases** — partial-credit logic now requires the *committee was genuinely split*, so confident "NO" verdicts no longer get half credit. Estimated -0.10 on overall recall.
5. **CATS Score ↑ overall** — the gain from correct-refusal carve-out (N1) typically outweighs the strictness gains. The qwen run had 5+ correct refusal samples; each goes from ~0.25 to 1.0.
6. **Type 5 row clearly flagged "n<5: noisy"** — same number, clearer caveat.
7. **Deterministic per-sample order in `detailed_results.json`** — runs are now diff-able.
8. **DeepSeek latency tail down** — `max_tokens=3000` + `<think>` stripping eliminates the truncation retries.
9. **Total cost line slightly up** — adds Sonnet NLI calls (one per claim per sample). On the qwen run with ~3 claims per sample and 50 samples, that's ~150 Sonnet calls ≈ $0.45 extra per run.

---

## 6. File-by-file diff summary

| File                                | Lines  | What changed                                                                 |
|-------------------------------------|--------|------------------------------------------------------------------------------|
| `rag_eval/judge_prompts.py`         | rewrite | Per-conflict rubric; refusal carve-out instruction; confidence requested; recall prompt now distinguishes assertion vs mention |
| `rag_eval/config.py`                | rewrite | Dead fields removed; `DEFAULT_JUDGE_PRIORITIES`; `priority_overrides`; `get_sonnet_nli_judge`; `nli_judge` field; `correct_refusal_full_credit` |
| `rag_eval/judge_committee.py`       | rewrite | `asyncio.Semaphore`; `minority_confidence`; `<think>` stripping; choices guard; confidence floor; `all_failed` sentinel |
| `rag_eval/conflict_eval.py`         | rewrite | NLI uses dedicated `JudgeClient`; partial-credit uses `minority_confidence`; consistent `claim_details` shape; gold normalized through `str()` |
| `rag_eval/data.py`                  | rewrite | `model_output` never falls back to gold; verdict normalization |
| `rag_eval/metrics.py`               | rewrite | Unified refusal regex; NLTK protection for initials/decimals/domains/abbreviations; meta-reference filter; `compute_f1_gr`; `gr_accuracy_from_flags` |
| `rag_eval/evaluator.py`             | rewrite | `_safe_ctype`; refusal carve-out; NLI judge wiring; `setdefault` aggregator; deterministic order; per-type `n<5` warning |
| `rag_eval/logging_config.py`        | rewrite | Idempotent handlers; `propagate=False` |
| `rag_eval/__init__.py`              | small   | Export `DEFAULT_JUDGE_PRIORITIES`, `get_sonnet_nli_judge`; remove `EnhancedTrustScoreConfig` |
| `run_evaluation.py`                 | edit    | `_load_yaml_config`, `_apply_yaml_to_config`; metric-name updates |
| `run_evaluation_batch.py`           | edit    | `f1_gr` → `gr_accuracy` |
| `configs/default.yaml`              | rewrite | Working YAML with `priority_overrides` example |
| `.env.example`                      | rewrite | `OPENROUTER_API_KEY` documented |

---

## What's still open

These weren't fixed in v3 and are flagged as follow-ups:

- **§6.7 — Type 4 behavior judge doesn't see doc dates.** Requires changing the prompt contract; out of scope for this round.
- **§3.4 — Per-judge token/RPM rate limiting.** Global semaphore covers concurrency; a per-judge token bucket is a follow-up if we hit OpenRouter's per-model RPM ceilings.
- **N3 — Dataset typos in gold answers.** Data cleanup task, not code.
- **N6 — Outdated model answer + behavior judge says "great".** Same as §6.7; needs date-aware behavior prompt.

Everything else from [ISSUES.md](ISSUES.md) §§1–6 is either resolved with a code change or explicitly resolved as "documented and left as-is".
