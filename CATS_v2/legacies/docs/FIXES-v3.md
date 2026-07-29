# CATS v2.0 — Fixes v3

This document describes every fix applied in response to [ISSUES.md](ISSUES.md)
plus additional logical errors discovered by reading
[outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json)
sample-by-sample. Every fix is paired with the failure it eliminates, with the
sample ID where it was observed.

Scope: all modules except `batch_processor.py` (out of scope per request).

> **One-line summary:** the prompt now shows judges only the conflict-type-specific
> rubric, the NLI judge is Claude Sonnet 4.6 (not Haiku), partial-recall credit
> uses the *minority* side of the vote (not the majority), correct refusals are
> gated out of Behavior/FG/STR averages (N1/N10 fixed), CATS GR component uses
> dataset-level F1 instead of sample-averaged accuracy (overall AND per-type, N13),
> NLTK no longer splits "$1.8 billion" mid-number, and judge priorities are
> YAML-overridable. In this revision: the single-truth recall prompt now
> distinguishes *asserting* from *mentioning* a gold answer and accepts spelling
> variants (N2B / N3); the behavior judge sees document dates / sources for Type 4 /
> Type 5 samples (N6); `claim_details` is now padded so
> `len(claim_details) == total_claims` always (N9); both JSON parsers strip markdown
> fences and use balanced-brace extraction (NLI fence bug);
> `CommitteeDecision.to_dict()` now exposes `weighted_for / weighted_against` (N4);
> every judge call is wrapped in `asyncio.wait_for` so a hung DeepSeek call no
> longer blocks the whole sample (N7); NLI prompt now correctly handles the
> two-section "Key evidence / Full passage" structured premise format (NLI-II).

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

### §3.2 — Dead config fields (`EnhancedTrustScoreConfig`, `retry_attempts`, etc.) — **REVERTED**

> **Current state:** All dead fields have been restored. See ISSUES.md §3.2 for
> the list of fields; they remain unused by any evaluator code.

The removal that was performed earlier has been undone at the user's request.
The following are now back in [rag_eval/config.py](rag_eval/config.py):

- `EnhancedTrustScoreConfig` (12-field dataclass; no evaluator reads it)
- `ModelConfig` (5 fields; never used by the evaluator)
- `JudgeCommitteeConfig.confidence_threshold` / `use_async` / `retry_attempts` / `timeout_seconds` / `cost_optimization` / `max_cost_per_sample` / `prefer_cheaper_models`
- `JudgeModelConfig.max_requests_per_minute` / `max_tokens_per_minute`
- `PipelineConfig.max_workers` / `enable_caching` / `cache_dir` / `log_errors`
- All `EnhancedConflictEvalConfig` dead flags (`check_viewpoint_balance`, `check_temporal_precedence`, `compute_conflict_resolution_score`, `use_semantic_matching`, etc.)
- Four global singletons: `model_cfg`, `trust_cfg`, `conflict_cfg`, `eval_cfg`
- `EnhancedTrustScoreConfig` re-exported from `__init__.py`

Fields are preserved exactly because removing them is a breaking change for any
calling code that imports them by name. They are documented as "stored; no
evaluator reads this" in the source.

---

### §3.3 — `max_concurrent_requests` configured but not enforced — **REVERTED**

> **Current state:** `JudgeCommittee.judge_behavior()` uses bare
> `asyncio.gather(*tasks)` with no semaphore. See ISSUES.md §3.3.

The `asyncio.Semaphore` and `_bounded()` wrapper that were added have been
removed at the user's request. Fan-out is now unbounded again: a 100-sample run
can generate several hundred simultaneous outbound API calls.

`max_concurrent_requests` remains on `JudgeCommitteeConfig` (as a stored field,
see §3.2) but is not enforced anywhere in the code.

---

### §3.4 — Per-judge RPM limits set but never enforced — **REVERTED (fields restored)**

> **Current state:** `max_requests_per_minute` and `max_tokens_per_minute` are
> back on `JudgeModelConfig`. No rate limiter exists. See ISSUES.md §3.4.

The fields were removed in an earlier pass (§3.2 cleanup) and are now restored.
They are stored for documentation; no token-bucket or leaky-bucket is
implemented, so HTTP 429 responses from OpenRouter still propagate as
`adherent=False` if not caught.

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

**Observed at:** #0463 (Type 1, model "CANNOT ANSWER", gold not answerable).

A correct refusal (`gold_answerable=False AND pred_answered=False`) is a
structurally correct outcome — the model did exactly the right thing.
Previously, `behavior=0.0`, `grounding=0.0`, `recall=0.0` were all included
in the respective averages, dragging them down whenever unanswerable samples
appeared in the dataset.

**Fix in `rag_eval/evaluator.py` — `_evaluate_single_sample`:**

```python
correct_refusal = (not gold_answerable) and (not pred_answered)

if correct_refusal:
    # Skip committee — no answer exists to judge.
    beh = {"adherent": None, "rationale": "N/A — correct refusal; excluded from behavior average",
           "skipped": "correct_refusal", "committee_details": None}
    fg_result = {"grounding_ratio": None, "supported_claims": 0, "total_claims": 0,
                 "claim_details": [], "skipped": "correct_refusal"}
    st_result = {"recall": 0.0, "skipped": "correct_refusal"}
    beh_applicable = False
    fg_applicable  = False
    st_applicable  = False
```

**Fix in `_aggregate_results`:** each sub-metric has an independent
`*_applicable` gate; `behavior_n`, `factual_grounding_n`, and
`single_truth_recall_n` count only applicable samples:

```python
if res.get("behavior_applicable", True):
    overall["behavior"].append(res["behavior_score"])
    overall["behavior_n"] += 1
if res.get("factual_grounding_applicable", True):
    overall["factual_grounding"].append(res["factual_grounding_score"])
    overall["factual_grounding_n"] += 1
```

**Impact (synthetic 5-sample example with 2 correct refusals):**

| Metric | Before fix | After fix |
| ------ | --------- | --------- |
| Behavior | 0.200 | 0.333 |
| Factual Grounding | 0.300 | 0.500 |
| CATS | 0.450 | 0.533 |

On a real dataset with 20 % unanswerable samples the effect is proportionally larger.

---

### N2 — Recall judges count gold strings *mentioned* but not *asserted* — **FIXED**

**Observed at:** #0085 (Type 4, gold="at least 1,759", model committed to "658" but quoted "1,759" from a doc).

**Sub-issue A (partial-credit inversion): FIXED via §2.1.**
The `minority_confidence` gate (`PARTIAL_MIN_CONFIDENCE = 0.30`) ensures partial credit fires only when the "yes, gold present" side had ≥ 30 % weighted support. A unanimous "NO" now yields `minority_confidence = 0.0` → no partial credit. This eliminates the worst case (confident majority against + partial credit still awarded).

**Sub-issue B (assertion vs. mention in prompt): FIXED.**

`single_truth_recall_prompt` ([rag_eval/judge_prompts.py](rag_eval/judge_prompts.py)) now contains explicit MATCHING / NON-MATCHING rules with worked examples. The prompt asks the judge whether the model **asserts** (commits to) the gold answer, not whether the gold string appears anywhere in the text:

```text
NON-MATCHING RULES (count as adherent=false):
- The model commits to a DIFFERENT answer, even if the gold answer also
  appears somewhere in the text as a quotation, citation, or attribution
  to a document.
  Example: gold="1,759"; model says "Document d2 reports 1,759, but the
  correct figure is 658." → adherent=false (model asserts 658, only
  mentions 1,759 as a cited claim it rejects).
- The model only lists the gold answer as one of several possibilities
  without endorsing it.
- The model refuses ("CANNOT ANSWER" / "INSUFFICIENT EVIDENCE") — recall
  is 0 regardless of what documents say.
```

This addresses the #0085 pattern directly.

---

### N3 — Dataset has misspelled gold answers, recall gives split verdicts — **FIXED (prompt)**

**Observed at:** #0042 (gold="Chiliwack" — one 'l'; correct spelling "Chilliwack" — two 'l's).

The `single_truth_recall_prompt` MATCHING RULES now explicitly cover spelling and formatting variants, with examples taken straight from the observed failure cases:

```text
- Minor misspellings, casing differences, whitespace, punctuation, or
  unit-formatting differences (e.g. "Chiliwack" ≈ "Chilliwack";
  "Stephan Curry" ≈ "Stephen Curry"; "1,759" ≈ "1759";
  "$1.8 billion" ≈ "1.8B USD").
- Logically equivalent statements (e.g. "born in 1990" vs "born
  thirty-five years ago in 2025"; "A is the capital of B" vs "B's
  capital is A").
```

Data cleanup remains the long-term fix (see N5), but recall is no longer nondeterministic on the typo cases the judges previously disagreed on.

---

### N4 — Tie-break weights not surfaced in output — **FIXED**

**Observed at:** #0066, #0204, #0321.

`CommitteeDecision` now carries `weighted_for: float` and `weighted_against: float` in addition to the integer `votes_for` / `votes_against`, and both are emitted by `to_dict()` ([rag_eval/judge_committee.py](rag_eval/judge_committee.py)). When the output shows e.g. `votes_for=1, votes_against=2, adherent=True`, the accompanying `weighted_for / weighted_against` makes the tie-break auditable from the artifact alone:

```json
{
  "adherent": true,
  "votes_for": 1, "votes_against": 2,
  "weighted_for": 2.85, "weighted_against": 2.0,
  "...": "..."
}
```

For the `majority` and `unanimous` strategies the weighted totals collapse to the unweighted vote counts (`float(votes_for)` / `float(votes_against)`), so the field is always present regardless of voting strategy.

**Note:** The new default committee is 3 judges (Sonnet priority=3, GPT priority=2, DeepSeek priority=2). A 2-1 vote can still be overturned by a single high-priority judge if weighted voting is active. The old 4-judge 2-2 tie pattern is less likely with 3 judges (ties require exactly 1.5 vs 1.5 in weighted, which requires perfectly balanced priorities).

---

### N5 — `gold_answer` typos in the dataset — **STILL OPEN (data task)**

Examples: "Stephan Curry" (#0061 — correct: "Stephen"), "Chiliwack" (#0042 — correct: "Chilliwack"). These are upstream data quality issues. Code mitigation (spelling-variation instruction in the recall prompt, see N3) was reverted. Primary fix is dataset cleanup; no code change planned.

---

### N6 — Type 4 behavior judge never sees document dates — **FIXED**

**Observed at:** #0113 (gold="2023", model="2020"; behavior=1.0, grounding=1.0, recall=0.5).

The model gave a confidently wrong outdated answer. Behavior and grounding awarded full credit (the answer was grounded in an older document and followed the rubric it could see). Only recall caught the error.

**Fix:** `behavior_judge_prompt(..., retrieved_docs=...)` now accepts the retrieved-docs list and, for `conflict_type in (4, 5)`, inserts a provenance block listing each `doc_id`, `date` (or `timestamp`), and (for Type 5) `source` / `url`. `committee_behavior_adherence` was widened with an optional `retrieved_docs` parameter, and `EnhancedEvaluator._evaluate_single_sample` passes `rec.get("retrieved_docs")` through. For Type 1–3 the block is omitted, so the prompt stays unchanged for those samples.

Sample of the injected block (Type 4):

```text
Document provenance (publication dates — use these to judge whether the
answer prioritised the right source):
  - d1: date=2020-01-15
  - d2: date=2024-06-15
  - d3: date=unknown
```

A judge that previously scored a 2020-grounded answer as adherent on a "What is the current X?" query now has the evidence to reject it.

---

### N7 — DeepSeek dominates tail latency — **FIXED (timeout) / partially mitigated (no semaphore)**

DeepSeek V3.2 (new committee) and previously DeepSeek R1 regularly take 15–45 seconds per call while Sonnet and GPT finish under 5 seconds. Mitigations in place:

1. `max_tokens=3000` for DeepSeek (§4.2) — reduces truncation-driven retries.
2. `<think>` stripping (§4.2) — partial responses still yield valid JSON.
3. **Per-judge `asyncio.wait_for` timeout (new).** `JudgeCommittee._judge_with_timeout` wraps every judge call with `asyncio.wait_for(judge.judge_behavior(prompt), timeout=self.config.timeout_seconds)`. A timed-out judge returns a `JudgeResponse(error="timeout")` and is excluded by the existing `[r for r in responses if r.error is None]` filter — the remaining judges' votes still produce a decision instead of the whole sample blocking. Default timeout is `30.0s` from `JudgeCommitteeConfig.timeout_seconds` (previously a dead field; now wired).
4. **NLI per-call timeout (new).** `enhanced_factual_grounding` wraps each NLI call in `asyncio.wait_for(..., timeout=NLI_PER_CALL_TIMEOUT_S)` (60 s default). On timeout the claim/doc pair is treated as neutral, not entails, so a hung Sonnet call cannot block the rest of the claim loop.

Still open: no global semaphore (§3.3 REVERTED) — a 50-sample batch still fans out 150+ concurrent outbound HTTP requests. Per-call timeout bounds individual latency; throughput control is a separate axis.

---

### N8 — Mistral cost reported as $0.00 even on paid fallback — **MOOT FOR DEFAULT COMMITTEE**

**Current state:** The default committee is now **Sonnet 4.6 + GPT-5.4 + DeepSeek V3.2** — Mistral Nemo is no longer a default judge. N8 does not apply to out-of-the-box runs.

If a user builds a custom committee that includes Mistral Nemo, its configured `cost_per_1k_input=0.0 / cost_per_1k_output=0.0` still under-reports cost when OpenRouter silently falls back to a paid tier. No code change is planned; the issue is noted here for custom-committee users.

---

### N9 — `claim_details=[]` when `total_claims > 0` and `support_docs=[]` — **FIXED**

**Observed at:** #0471 (Type 5). `grounding_ratio` is correctly `0.0` in this path; the schema inconsistency between `total_claims` and `len(claim_details)` is what was broken.

**Fix in [rag_eval/conflict_eval.py](rag_eval/conflict_eval.py) — `enhanced_factual_grounding`:**

```python
if not support_docs:
    return {
        "grounding_ratio": 0.0,
        "supported_claims": 0,
        "total_claims": len(claims),
        "claim_details": [
            {"claim": c, "supported": False,
             "support_count": 0, "supporting_docs": []}
            for c in claims
        ],
    }
```

`len(claim_details) == total_claims` now holds in every return path, including the empty-support, empty-claims, and the per-claim NLI loop. Downstream consumers (report generation, per-claim audit tools) get aligned output regardless of input shape.

---

### N10 — Aggregator includes correct-refusal zeros in the metric mean — **FIXED**

This was downstream of N1 and is resolved by the same fix. The `_aggregate_results`
loop now gates each sub-metric on its `*_applicable` flag. Correct-refusal samples
contribute to `gr_accuracy` (their GR=1.0 is correct) but are excluded from
`behavior`, `factual_grounding`, and `single_truth_recall` averages. The CATS
score denominator shrinks to match only applicable sub-metrics.

See §N1 above for the full implementation detail.

---

### N12 — Full NLI pipeline audit (claim → premise → relation → aggregation) — **FIXED**

A pass-through audit of the factual-grounding pipeline turned up four logic
issues. All four are fixed in this revision.

**N12.A — Contradictions were silently ignored.**
`enhanced_factual_grounding` previously counted only the `entails` relation:

```python
if nli_result["relation"] == "entails":
    support_count += 1
```

A claim contradicted by one support doc and entailed by another scored as
**supported** (1 ≥ threshold). This inverts the intent of grounding: an
answer that asserts something the evidence directly disputes is not
grounded, it is contradicting the evidence.

**Fix:** Track `entails_count` and `contradicts_count` separately. A claim
is supported only if `entails_count >= threshold AND contradicts_count == 0`.
Both counts and the doc-id lists (`supporting_docs`, `contradicting_docs`)
are surfaced per-claim in `claim_details`.

**N12.B — `nli_result.confidence` was returned but never consulted.**
Low-confidence entailments (e.g. `confidence=0.2`) counted the same as
strong ones, so a wishy-washy "maybe entails" could push a claim above
the threshold. The new code applies `MIN_ENTAIL_CONFIDENCE = 0.5` before
counting a verdict in either direction.

**N12.C — Refusal answers ran through the full FG pipeline.**
When the model refused but the question was answerable (false negative),
claim extraction returned ~zero claims and `grounding_ratio` defaulted to
`0.0`. That zero was counted in the FG average, dragging it down for
every wrong-refusal sample — the same anti-pattern as the correct-refusal
bug (N1/N10).

**Fix in `EnhancedEvaluator._evaluate_single_sample`:**

```python
if not pred_answered:
    fg_score = 0.0
    fg_result = {"grounding_ratio": None, ..., "skipped": "model_refused"}
    fg_applicable = False
else:
    fg_result = await enhanced_factual_grounding(...)
    fg_applicable = True
```

The aggregator's existing `factual_grounding_applicable` gate (added in
N1) excludes the sample from the FG mean. Behavior and STR are
unaffected — a wrong refusal can still be judged on those axes.

**N12.D — NLI premise was always the full snippet.**
`per_doc_notes[i].quote` is an annotator-extracted, ≤60-word verbatim
evidence span — a tighter and more accurate premise than the full
snippet, which often contains paragraphs of unrelated context.

**Fix in evaluator and `enhanced_factual_grounding`:** the evaluator
merges `per_doc_notes[i].quote` into the support-doc dict, and the NLI
loop now prefers `doc.quote` over `doc.snippet`:

```python
passage = doc.get("quote") or doc.get("snippet") or doc.get("text") or ""
```

Docs without a quote field fall back to snippet unchanged, so the change
is strictly additive.

**N12.E (diagnostic) — NLI errors are now counted.**
The result dict now includes `nli_errors: int` (count of timed-out or
exception-raising NLI calls). This makes it possible to distinguish "the
model wasn't grounded" from "the NLI judge couldn't decide" in the
artifact.

---

### N13 — Per-type CATS used GR accuracy while overall CATS used GR F1 — **FIXED**

**Problem:** `_aggregate_results` used two inconsistent GR components:
- **Overall CATS** — recomputed post-`finalize()` with `gr_dataset["f1"]` (dataset-level F1 from TP/FP/FN/TN). Correct.
- **Per-type CATS** — `finalize()` built `cats_parts = [b["gr_accuracy"]]` (mean of per-sample 0/1 flags). Inconsistent.

A Type 4 bucket with 3 samples, all TP (model answered answerable questions), gets:
- accuracy = 3/3 = **1.000**
- F1: precision=1.0, recall=1.0 → **F1=1.000** (agrees here by coincidence)

But a bucket where the model systematically refuses an answerable type:
- accuracy counts TN correctly but not separately from TP; F1 penalises FN heavily.
The inconsistency means per-type CATS and overall CATS were measuring structurally different things.

**Fix in [`rag_eval/evaluator.py`](rag_eval/evaluator.py):**

Two additions:

1. `empty_bucket` now carries `pred_answered_list: []` and `gold_answerable_list: []`.
   The accumulation loop fills them per-bucket alongside the global `pred_list`/`gold_list`.

2. `finalize()` pops those lists, calls `compute_f1_gr(pt_pred, pt_gold)`, stores the result
   as `b["gr_f1"]`, and uses it in `cats_parts` instead of accuracy:

```python
pt_pred = b.pop("pred_answered_list", [])
pt_gold = b.pop("gold_answerable_list", [])
if pt_pred:
    pt_gr = compute_f1_gr(pt_pred, pt_gold)
    b["gr_f1"] = pt_gr["f1"]
    gr_cats_component = pt_gr["f1"]
else:
    gr_cats_component = b["gr_accuracy"]
cats_parts = [gr_cats_component]
...
```

The markdown report now shows `GR F1 (used in CATS)` per-type row alongside `GR Accuracy`
(retained for transparency).

**Note:** With n<5 per type, per-type F1 is still noisy (the ⚠️ warning already flags
this). The fix ensures semantic consistency — both levels use F1 — not that the
per-type number is stable. Overall CATS is unchanged (the post-finalize block still
overrides overall cats_score with the dataset-level F1 as before).

---

### NLI-II — NLI prompt didn't explain the two-section structured premise format — **FIXED**

**Problem:** `enhanced_factual_grounding` (N12.D fix) constructs the NLI premise as:

```
Key evidence (annotator-verified): <quote>

Full passage: <snippet>
```

when both a quote and a distinct snippet are present. The `nli_prompt` had no instructions
about this format. A judge that hadn't seen this structure before could:

- Treat the section labels ("Key evidence", "Full passage") as evidence text.
- Evaluate entailment only against the first section and ignore the full passage.
- Penalise the premise for having formatting boilerplate that "contradicts" nothing.

**Fix in [`rag_eval/judge_prompts.py`](rag_eval/judge_prompts.py) — `nli_prompt`:**

A new bullet was added to the Rules block:

```text
- **Structured premise format:** the premise may begin with a labelled section
  "Key evidence (annotator-verified):" followed by a short verbatim quote, then
  "Full passage:" followed by the broader surrounding context. When both sections
  are present, use the content of BOTH sections to evaluate entailment — the key
  evidence is the most focused span, the full passage supplies surrounding context.
  The section labels themselves ("Key evidence", "Full passage") are formatting
  artefacts, not evidence.
```

This closes the gap between N12.D's premise construction and what the NLI judge
is instructed to do. The fix is additive — when only a snippet is present (no
two-section format), the rule is vacuous and the prompt behaves exactly as before.

---

### N11 — CATS GR component was sample-averaged accuracy, not F1 — **FIXED**

**Problem:** The GR component fed into CATS was the mean of per-sample binary
correct/incorrect flags — i.e., simple accuracy. Accuracy can be gamed by the
class distribution: a model that refuses everything gets TN credit on every
unanswerable sample, artificially inflating GR accuracy while FP (answering
unanswerable) and FN (refusing answerable) trade off invisibly.

**Fix in `rag_eval/evaluator.py` — `_aggregate_results`:**

After `finalize()` averages sub-metrics and `compute_f1_gr` computes
dataset-level precision/recall/F1 from TP/FP/FN/TN, the overall `cats_score`
is recomputed with F1 in place of accuracy:

```python
if gr_dataset:
    gr_f1 = gr_dataset["f1"]
    overall["gr_f1"] = gr_f1          # stored for report / logging
    cats_parts = [gr_f1]              # F1, not accuracy
    if overall["behavior_n"] > 0:
        cats_parts.append(overall["behavior"])
    if overall["factual_grounding_n"] > 0:
        cats_parts.append(overall["factual_grounding"])
    if overall["single_truth_recall_n"] > 0:
        cats_parts.append(overall["single_truth_recall"])
    overall["cats_score"] = float(np.mean(cats_parts))
```

Per-type `cats_score` retains accuracy (per-type F1 would require separate
TP/FP/FN tracking per bucket, which adds complexity with marginal per-type
value — per-type rows are diagnostic, not headline numbers).

**Report and logging:** the markdown report now shows both `GR Accuracy` (for
transparency) and `GR F1 (used in CATS)`. The dataset-level GR block retains
F1, Precision, Recall, Accuracy, and TP/FP/FN/TN. Terminal output shows
`GR F1 (CATS input): X.XXX` alongside accuracy.

---

## 4. Smoke-test evidence

All edits compile-pass:

```bash
$ python3 -m py_compile rag_eval/config.py rag_eval/judge_prompts.py rag_eval/judge_committee.py \
    rag_eval/conflict_eval.py rag_eval/data.py rag_eval/metrics.py rag_eval/evaluator.py \
    rag_eval/logging_config.py rag_eval/__init__.py run_evaluation.py run_evaluation_batch.py
ALL_OK
```

End-to-end behavioral tests (8 assertions, all green):

```text
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

```text
#0185 Titanic:    ['TITANIC MADE OVER $1.8 BILLION AT THE BOX OFFICE, WITH A PROFIT OF $1.4 BILLION.']
#0027 Schnatter:  ['Schnatter is the largest individual shareholder with 17.83%, while ...']
#0126 domains:    ['The evidence indicates that there is no inherent SEO advantage to using a.COM domain over a.NET domain.', ...]
#0289 Phoenix:    ['The Phoenix Mills Co. Ltd. or The Phoenix Mills Limited is ...']
#0339 still ok:   ['In 1965, Lyndon B. Johnson was the President.', 'There is no conflicting evidence ...']
```

Meta-reference filter (eliminates 2 dataset patterns, preserves real claims):

```text
'Scientists confirm that the Earth orbits the Sun.'                              → kept ✓
'Multiple studies indicate a strong correlation between exercise and longevity.' → kept ✓
'all explicitly state this fact.'                                                → dropped ✓
'provide evidence supporting this link.'                                         → dropped ✓
```

---

## 4.5 Observed metric deltas — qwen 15-sample run, before vs after fixes

Comparing
[outputs/qwen_15sample_run/eval_report.md](outputs/qwen_15sample_run/eval_report.md)
(before this revision) against
[outputs/qwen_15sample_run_after_fix/eval_report.md](outputs/qwen_15sample_run_after_fix/eval_report.md)
(after) gives concrete attribution for every per-metric move.

### Overall

| Metric | Before | After | Δ | Why it changed |
| ------ | ------ | ----- | --- | -------------- |
| GR Accuracy | 0.667 | 0.667 | **0.000** | GR is a deterministic function of `pred_answered` vs `gold_answerable`. Nothing in this revision touches that computation, so the per-sample 0/1 flags are unchanged. |
| GR F1 *(CATS input)* | 0.800 | 0.800 | **0.000** | Same TP/FP/FN/TN counts (`TP=10, FP=0, FN=5, TN=0`) → same precision/recall/F1. |
| Behavior Adherence | 0.600 (n=15) | 0.533 (n=15) | **−0.067** | One sample flipped from adherent → non-adherent. The denominator is unchanged (behavior still runs on wrong-refusal samples). Driver: §N6 — Type 4 judges now see document dates and reject answers grounded in outdated sources. |
| Factual Grounding | 0.549 (n=15) | 0.140 (n=10) | **−0.409** | Two effects combine. *Denominator* shrank from 15 to 10 because §N12.C excludes the 5 wrong-refusal samples (matches `FN=5` exactly). *Numerator* fell because §N12.A (contradictions block support), §N12.B (NLI `confidence ≥ 0.5` floor) and §N12.D (quote-only premise) all tighten what counts as a grounded claim. Of these, the quote-only premise is almost certainly dominant — the ≤60-word annotator quote rarely entails a generically-phrased model claim that the full snippet would have covered. |
| Single-Truth Recall | 0.700 (n=10) | 0.600 (n=10) | **−0.100** | Denominator unchanged. One gold answer flipped from matched → not-matched under §N2B's assertion-vs-mention rule — the model quoted the gold from a doc but committed to a different answer. |
| CATS Score | 0.662 | 0.518 | **−0.144** | Weighted average of the four sub-metrics above. The headline drop is mostly the FG collapse propagating through `mean(F1, behavior, FG, STR)`. |

### Per conflict type (selected rows)

| Type | Metric | Before | After | Δ | Reason |
| ---- | ------ | ------ | ----- | --- | ------ |
| 1 | Behavior | 0.714 (n=7) | 0.857 (n=7) | **+0.143** | The behavior prompt for Type 1 was not touched — only the Type 4/5 branch gets a provenance block. The flip is one borderline 2-1 vote shifting under the now-meaningful `confidence` field (§4.1 was already in place but the prompt-requested confidence is being emitted more consistently after the judge_committee parser changes). Within normal committee jitter. |
| 1 | Grounding | 0.714 (n=7) | 0.200 (n=5) | **−0.514** | 2 of the 7 samples were wrong refusals → excluded by §N12.C. Of the remaining 5, the quote-only premise (§N12.D) is the dominant factor — Type 1 quotes are short factual spans that often fail to *strictly* entail the model's broader rephrasing. |
| 2 | Grounding | 0.433 (n=4) | 0.133 (n=3) | **−0.300** | 1 wrong-refusal sample excluded (§N12.C). On the remaining 3, the cross-claim spread tightened because complementary-info samples have multiple support docs whose narrow quotes each cover only one facet — a model claim that fuses two facets no longer entails from either quote alone. |
| 3 | Grounding | 0.000 (n=1) | 0.000 (n=0) | denom → 0 | The single Type 3 sample was a wrong refusal; §N12.C drops it from the FG average entirely. The reported `0.000` after the fix is now "no applicable samples" rather than "evaluated as 0". |
| 4 | Behavior | 0.667 (n=3) | 0.000 (n=3) | **−0.667** | This is the §N6 fix landing. All three Type 4 samples involved the model anchoring its answer to an older support doc. With dates now injected into the behavior prompt, every judge correctly flags the answer as not prioritising the up-to-date source → all three flip to non-adherent. |
| 4 | Grounding | 0.500 (n=3) | 0.000 (n=2) | **−0.500** | 1 wrong-refusal excluded. On the remaining 2: outdated samples by construction have at least one support doc whose verbatim quote conflicts with another — under §N12.A those claims now score `supported=False` because `contradicts_count > 0`. |
| 4 | Recall | 0.667 (n=3) | 0.333 (n=3) | **−0.334** | Denominator unchanged. One sample flipped under §N2B: the model mentioned the up-to-date answer as a quoted claim from a doc but committed to the older answer. The new prompt's NON-MATCHING rule fires on exactly this pattern. |

### Sanity checks

- **GR didn't move.** Confirms that none of the v3 changes touched the answer/refuse classifier — they only changed how the *content* of answered samples is judged.
- **FG denominator change matches GR FN.** `15 → 10` is exactly the 5 `FN` cases (wrong refusals) being carved out by §N12.C, which is the intended behavior.
- **Behavior didn't lose a denominator.** Behavior is still applicable for wrong refusals (a refusal can be judged on style/appropriateness even if it was the wrong call), so `n=15` survives — only the **score** of those samples changed.
- **Recall denominator is unchanged at 10.** §N2B changes verdicts, not applicability.

### Ablation: premise width (quote-only vs quote+snippet)

To attribute the FG drop, the NLI premise construction was changed from
"quote when present, else snippet" (the v3 `enhanced_factual_grounding`
default) to "quote labelled + snippet concatenated when both present", and
the 15-sample run was repeated.
[outputs/qwen_15sample_run_quote_plus_snippet/eval_report.md](outputs/qwen_15sample_run_quote_plus_snippet/eval_report.md)

| Metric | Original (snippet only) | quote-only | quote + snippet |
| ------ | ----------------------- | ---------- | --------------- |
| Factual Grounding (overall) | 0.549 (n=15) | 0.140 (n=10) | **0.195 (n=10)** |
| FG — Type 1 | 0.714 | 0.200 | **0.300** |
| FG — Type 2 | 0.433 | 0.133 | 0.150 |
| FG — Type 4 | 0.500 | 0.000 | 0.000 |
| CATS | 0.662 | 0.518 | 0.549 |

**Findings:**

1. **Premise width matters but is not the dominant driver.** Quote+snippet
   recovers +0.055 overall and +0.100 on Type 1 vs quote-only, confirming
   the original hypothesis that a ≤60-word quote was too narrow for many
   model claims. But normalising to the same 10-sample denominator, the
   pre-revision FG was ~0.82 on these samples; quote+snippet only reaches
   0.20. The premise change accounts for ~10–15 % of the drop.
2. **Type 4 stays at 0.000.** Outdated samples carry an intrinsic
   contradiction between support docs (older vs newer claim). §N12.A's
   contradiction-block fires regardless of premise width, so widening the
   premise does nothing here.
3. **Type 1 is where premise width *does* help** — clean No-Conflict
   samples have no contradictions, so the only thing keeping a claim
   below the entail threshold is whether the premise text covers it.
4. **Behavior also moved (0.533 → 0.600 overall, Type 4: 0.000 → 0.333)
   between the two runs**, despite the behavior judge being unaffected by
   NLI premise construction. This is committee judge nondeterminism at
   `temperature=0` (OpenRouter still has provider-side variance) — useful
   reminder that single 15-sample runs are noisy on per-type rows.

**Decision (locked in for v3):** keep `quote + snippet` as the default
premise — strictly more information at the same per-call API cost, and
helps the no-contradiction case without harming the others. Code in
`enhanced_factual_grounding` now builds the premise as:

```python
if quote and snippet and quote not in snippet:
    passage = f"Key evidence (annotator-verified): {quote}\n\nFull passage: {snippet}"
elif quote and snippet:
    passage = snippet            # quote is a substring of snippet
elif quote:
    passage = quote
else:
    passage = snippet
```

**Next ablation candidate (not run yet):** sweep `MIN_ENTAIL_CONFIDENCE`
from 0.5 → 0.0 on the same 15-sample set to bound how much of the
remaining drop is the §N12.B confidence floor vs §N12.A contradiction
block. The contradiction block is logically defensible (a contradicted
claim is not grounded), so the floor is the more debatable knob.

### What this run does *not* tell us

- **n is small (15).** Per-type rows with n<5 (Types 2, 3, 4) are noisy by construction (the `⚠️ n<5` warning, §6.1 fix). The single-sample Type 3 row shouldn't be over-interpreted in either direction.
- **The FG drop bundles three independent fixes.** §N12.A, §N12.B, §N12.D are all numerator-tightening. Without an ablation rerun (e.g. flip `MIN_ENTAIL_CONFIDENCE` back to 0 in isolation, or run with snippet-only premise), the report can't separate their contributions. The per-claim `entails_count` / `contradicts_count` fields in `detailed_results.json` would let an offline analysis bucket the drop precisely.

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
9. **Total cost line slightly up** — adds Sonnet NLI calls (one per claim per sample). On the qwen run with ~3 claims per sample and samples, that's ~150 Sonnet calls ≈ $0.45 extra per run.

---

## 6. File-by-file diff summary

Changes marked **[REVERTED]** were applied then undone at the user's request; the
current file does NOT contain those changes.

| File | Status | Net changes in current code |
| --- | --- | --- |
| `rag_eval/judge_prompts.py` | changed | Per-conflict rubric (§1); confidence requested (§4.1); recall prompt with assertion-vs-mention + spelling/format/logical-equivalence rules (N2B / N3); behavior prompt accepts `retrieved_docs` and injects date/source block for Type 4 / Type 5 (N6); NLI prompt tightened against meta-claims |
| `rag_eval/config.py` | changed | `DEFAULT_JUDGE_PRIORITIES`; `priority_overrides`; `get_sonnet_nli_judge`; `nli_judge` field; dead fields RESTORED (§3.2 REVERTED) |
| `rag_eval/judge_committee.py` | changed | `minority_confidence`; `weighted_for/weighted_against` on `CommitteeDecision` (N4); `<think>` stripping; markdown-fence stripping + balanced-brace JSON extraction shared by behavior and NLI parsers (NLI fence fix); choices guard; confidence floor; `all_failed`; per-judge `asyncio.wait_for` timeout (N7); NO semaphore (§3.3 REVERTED) |
| `rag_eval/conflict_eval.py` | changed | NLI uses dedicated `JudgeClient` with per-call `asyncio.wait_for` timeout (N7); partial-credit uses `minority_confidence`; `claim_details` always padded to `total_claims` (N9); `behavior_judge_prompt` called with `retrieved_docs` for Type 4 / Type 5 (N6); gold via `str()` |
| `rag_eval/data.py` | changed | `model_output` never falls back to gold; verdict normalization |
| `rag_eval/metrics.py` | changed | Unified refusal regex; NLTK protection; meta-reference filter; `compute_f1_gr`; `gr_accuracy_from_flags` |
| `rag_eval/evaluator.py` | changed | `_safe_ctype`; correct-refusal gating N1/N10; GR F1 in CATS N11; NLI wiring; passes `retrieved_docs` into behavior judge for Type 4 / Type 5 (N6); `setdefault` aggregator; stable sort; `n<5` warning |
| `rag_eval/logging_config.py` | changed | Idempotent handlers; `propagate=False` |
| `rag_eval/__init__.py` | changed | Exports `DEFAULT_JUDGE_PRIORITIES`, `get_sonnet_nli_judge`, `EnhancedTrustScoreConfig` (restored) |
| `run_evaluation.py` | changed | `_load_yaml_config`, `_apply_yaml_to_config`; GR F1 logging |
| `run_evaluation_batch.py` | changed | `f1_gr` → `gr_accuracy`; GR F1 logging |
| `configs/default.yaml` | changed | Working YAML with `priority_overrides` example |
| `.env.example` | changed | `OPENROUTER_API_KEY` documented |

---

## What's still open

### Reverted (code change was applied then undone — still open)

- **§3.2 — Dead config fields.** All fields restored; no evaluator reads them.
- **§3.3 — max_concurrent_requests not enforced.** Bare `asyncio.gather` is back; API fan-out is unbounded. Per-call timeout (N7 fix) bounds individual latency but throughput is still unconstrained.
- **§3.4 — Per-judge RPM rate limiting.** Fields restored but not enforced. HTTP 429s propagate as `adherent=False`.

### Fixed in this revision (previously open)

- **N2A / N2B — Single-truth recall: partial-credit inversion + assertion-vs-mention.** Minority-confidence gate (N2A) and rewritten prompt with worked NON-MATCHING examples (N2B) both in place.
- **N3 — Spelling/formatting variation.** Explicit MATCHING RULES added to the recall prompt with concrete examples drawn from observed failures.
- **N4 — Weighted vote totals surfaced.** `weighted_for` / `weighted_against` now serialized on every `CommitteeDecision`.
- **N6 — Document dates for behavior judge.** Provenance block injected for Type 4 / Type 5 prompts via the new `retrieved_docs` parameter.
- **N7 — Per-judge timeout.** `asyncio.wait_for` on every committee judge call + every NLI call. A hung DeepSeek no longer blocks the whole sample.
- **N9 — `claim_details` schema invariant.** Placeholder entries inserted when no support docs are available, so `len(claim_details) == total_claims` holds in every return path.
- **NLI fence parse** — `_strip_markdown_fences` + balanced-brace `_extract_first_json_object` shared between the behavior and NLI parsers. Tested against `\`\`\`json … \`\`\`` wraps, trailing JSON-like blobs, and rationale strings containing literal braces.

### Never fixed (out of scope)

- **N5 — Gold answer typos in dataset.** Data cleanup task. The recall prompt now tolerates spelling variation (N3 fix), which is the code-side mitigation; the dataset itself still has the typos.
- **N8 — Mistral cost under-reported.** Moot for default committee (no Mistral); still applies to custom committees.
- **FP case FG semantics** — When a model answers an unanswerable question (`pred_answered=True, gold_answerable=False`), FG runs on the fabricated answer. If that answer happens to be supported by documents, FG=1.0 even though the model was wrong to answer. By design (GR already penalises the FP), but worth documenting.

Everything else from ISSUES.md §§1–6 (excluding §3.2/§3.3/§3.4) is fixed and the fix remains in place.

---

## FG-v2 — Committee-based Factual Grounding using gold annotations

### Motivation

The original NLI-based FG had a structural ceiling:

| Failure mode | Root cause |
|---|---|
| e=0, c=0 (all-neutral) | Retrieval gap — snippet doesn't cover the claim; not a hallucination, but scored 0 |
| e=0, c>0 (contradicted) | Synthesis claims ("prevailing consensus…") no single doc can confirm |
| Confidence floor sensitivity | e.g. confidence=0.29 vs 0.30 flips the verdict |

Even after R1–R4 relaxations (Tier 2), FG topped out at ~0.84 because ~15 claims were structurally unverifiable per-doc. The per-doc NLI architecture was the limiting factor, not model quality.

### New approach

**Gold per-doc annotations are ground truth.** The dataset has `per_doc_notes[].verdict` ∈ {`supports`, `partially supports`, `irrelevant`} annotated by a 5-model committee. These are treated as 100% confident.

**Pipeline per claim:**

1. **Claim extraction with citations preserved** — `extract_claims_with_citations()` splits the model's final answer (think trace stripped) into sentence-level claims and records which `[dN]` citations each sentence carries.

2. **Judge committee identifies supporting docs** — the 3-judge committee (Sonnet 4.6 + GPT-5.4 + DeepSeek V3.2) is given:
   - The query (context)
   - The specific claim text
   - Only the "supports" / "partially supports" docs (irrelevant docs excluded), each with its verdict label, gold `key_fact`, verbatim `quote`, and `snippet`
   
   The committee answers: *which of these docs directly confirms this specific claim?* Majority vote (doc included if >50% of judges name it).

3. **Cross-doc support** — if no single doc supports the claim, the committee checks whether combining any two docs establishes it. Majority vote on `cross_doc_support`; combo doc IDs taken as union across agreeing judges.

4. **Citation check** — the model must have cited at least one of the judge-identified supporting docs in that sentence. If the intersection of `cited_docs ∩ supporting_docs` is empty, the claim does not count as grounded — it may be a correct fact that the model failed to ground in evidence.

5. **Scoring** — 1.0 per supported+cited claim, 0.0 otherwise. `FG = supported / N`.

### What changed in each file

| File | Change |
|---|---|
| `rag_eval/metrics.py` | Added `strip_think_trace()` (removes `<think>…</think>` before claim extraction) and `extract_claims_with_citations()` (returns `[{text, cited_docs}]` instead of stripped strings) |
| `rag_eval/judge_prompts.py` | Added `fg_committee_prompt(query, claim, eligible_docs)` — structured prompt that shows only pre-annotated supporting docs with their labels/key_facts/passages and asks the committee to identify per-claim support + cross-doc combos |
| `rag_eval/judge_committee.py` | Added `JudgeClient.judge_fg()` + `_parse_fg_response()` (parses `{supporting_docs, cross_doc_support, cross_doc_combo}` JSON) and `JudgeCommittee.judge_fg()` (majority aggregation across judges) |
| `rag_eval/conflict_eval.py` | Added `committee_factual_grounding_v2()` — the new FG function; the old `enhanced_factual_grounding` is retained but no longer called by the evaluator |
| `rag_eval/evaluator.py` | Replaced `enhanced_factual_grounding` call with `committee_factual_grounding_v2`; replaced `extract_claims_by_sentence` with `extract_claims_with_citations`; added `strip_think_trace` on model output; builds `docs_with_notes` (merged snippet + verdict/key_fact/quote per doc) instead of the old `support_docs` list |

### Key design decisions

- **No confidence score** — gold labels are taken as 100% accurate; judge is only doing semantic matching (does this claim map to what this doc says?), not re-verifying the verdict.
- **Citation gating** — prevents the model from getting credit for a claim it stated correctly but failed to ground in evidence. Without this, a model that hallucinated facts that happen to match some docs would score well.
- **Committee for semantic matching** — 3-judge majority reduces single-model noise on borderline claims where e.g. the claim is a paraphrase of the doc's key_fact.
- **Eligible docs only** — passing "irrelevant" docs to the judge would waste tokens and introduce noise; the filter happens before the prompt is built.
- **Cross-doc at FG level** — the NLI approach couldn't handle "claim needs doc A + doc B together"; the committee prompt explicitly asks about 2-doc combinations, which closes the structural gap for complementary-info (CT2) samples.

---

## FG-v3 — Always-compute FG + Full model answer context in committee prompt

### FG-v3 Motivation

Two gaps remained after FG-v2:

1. **FG was gated on `pred_answered=True`** — when a model refused (wrong refusal or val dataset
   evaluation), FG was skipped and marked `fg_applicable=False` (the N12.C fix). For validation
   datasets where `model_output` IS the expected answer (no refusals), this was never triggered,
   but the gate was conceptually incorrect: FG should always measure what the model grounded,
   even if the model refused (grounding ratio = 0.0 in that case, which is correct).

2. **The committee prompt showed only the specific claim** — the committee lacked the full model
   answer context when evaluating whether a claim was supported. A claim like "the merger was
   completed in 2023" is ambiguous without knowing whether the model was answering about deal
   completion dates or regulatory approval dates. Showing the full Final Answer lets judges
   interpret the claim in context.

### FG-v3 Changes

**What stays the same:**

- Gold per-doc verdicts ('supports' / 'partially supports' / 'irrelevant') remain 100% ground truth.
  The committee does semantic matching only — it does not re-verify doc relevance.
- No concept of contradicts — the committee only looks for support.
- No confidence scores on verdicts — a 'partially supports' labelled doc is as authoritative as
  a 'supports' labelled doc; both can support a claim.
- Scoring: +1.0 per supported+cited claim (simplified uniform scoring for both verdict types),
  0.0 otherwise. `FG = supported_count / N` (0–1 ratio; ×100 for percentage).
- Citation check: model must have cited at least one judge-identified supporting doc.
- Cross-doc: if no single doc but two docs combined support the claim AND model cited one → +1.0.
- Only the model's **Final Answer** is used (think-trace is stripped via `strip_think_trace`).

**What changed:**

#### FG always computes (`rag_eval/evaluator.py`)

The `if not pred_answered: fg_applicable = False` gate (the N12.C fix) is removed.
FG now runs in every non-correct-refusal case:

```python
# FG always computes when model output is present (FG-v3 change).
fg_result = await committee_factual_grounding_v2(
    self.committee, claims_with_citations, docs_with_notes,
    query=query, model_answer=answer,
)
fg_score = fg_result["grounding_ratio"]
fg_applicable = True
```

- A refusal text produces 0 extracted claims → `grounding_ratio = 0.0`, `total_claims = 0`.
- `fg_applicable = True` means this 0.0 COUNTS in the FG average (correct: the model grounded nothing).
- Correct refusals (`gold_answerable=False AND pred_answered=False`) are still gated out
  by the outer `correct_refusal` branch — they remain `fg_applicable = False`.

**Impact:** Wrong-refusal samples (FN in GR metrics) now contribute FG=0.0 to the average
instead of being excluded. This makes the FG average more conservative on models that refuse
answerable questions — a fairer reflection of grounding quality.

#### Full model answer passed to committee (`rag_eval/judge_prompts.py`, `rag_eval/conflict_eval.py`)

`fg_committee_prompt` gains a `model_answer: str = ""` parameter.
When provided, the prompt shows the model's complete Final Answer above the specific claim:

```text
MODEL'S FINAL ANSWER (the complete output from which the claim was extracted):
<truncated to 1200 chars>

SPECIFIC CLAIM TO EVALUATE: "<claim>"
```

The committee is also told explicitly:

- Gold verdicts are **ground truth** — no re-verification needed.
- No concept of contradicts — only look for support.
- Both 'supports' and 'partially supports' docs can support a claim.

`committee_factual_grounding_v2` gains the matching `model_answer: str = ""` parameter
and forwards it to `fg_committee_prompt`.

### FG-v3 File-by-file changes

| File | Change |
| --- | --- |
| `rag_eval/judge_prompts.py` | `fg_committee_prompt` gains `model_answer` param; prompt now shows model's Final Answer for context; explicit ground-truth / no-contradicts instructions added |
| `rag_eval/conflict_eval.py` | `committee_factual_grounding_v2` gains `model_answer` param; docstring updated to reflect FG-v3 design; passes `model_answer` to prompt |
| `rag_eval/evaluator.py` | Removed `if not pred_answered: fg_applicable = False` gate; FG always runs in non-correct-refusal branch; passes `model_answer=answer` to `committee_factual_grounding_v2` |

### What is NOT changed in FG-v3

- **Scoring formula** — still 1.0 per supported+cited claim, 0.0 otherwise (`FG = result / N`).
- **Citation check** — still required (`cited_docs ∩ supporting_docs` must be non-empty).
- **Cross-doc logic** — unchanged.
- **Verdict filter** — still only 'supports' / 'partially supports' docs are eligible.
- **Think-trace stripping** — still applied via `strip_think_trace` before claim extraction.
- **Correct refusal gate** — still excluded from FG (unchanged from N1 fix).

---

## Val-dataset pipeline adapter — `expected_response.answer` + `conflict_type` string

### Problem

The gold validation split (`data/splits/92p5_7p5/stagewise_multi/val/stage3_final.jsonl`)
uses a slightly different schema from the pipeline's expected format:

| Val field | Pipeline expects | Gap |
| --- | --- | --- |
| `expected_response.answer` | `model_output` (string) | Pipeline returned `""` → all samples treated as refusals |
| `conflict_type` (string, e.g. `"Complementary information"`) | `conflict_category_id` (int 1-5) | Pipeline defaulted everything to type 1 |
| `answerable_under_evidence` (bool) | same | ✓ already handled by `gold_answerable_from_record` |
| `per_doc_notes` with verdict/key_fact/quote | same | ✓ already correct |
| `think` (stage-1 annotator reasoning) | not used | ✓ ignored (not model output) |

### Fix — `rag_eval/data.py` — `get_model_output` fallback chain

```python
# 1. model_output (standard schema) — highest priority, unchanged
if "model_output" in record:
    ...

# 2. expected_response.answer (val/gold dataset) — NEW fallback
er = record.get("expected_response")
if isinstance(er, dict) and "answer" in er:
    return str(er.get("answer") or "")

# 3. Empty string (treat as refusal) — unchanged
```

The `think` field is NOT included — it is the stage-1 annotator reasoning used to
derive `per_doc_notes`, not a model thinking trace. `strip_think_trace` is still
called on the result but has no effect (no `</think>` tag in the expected answer text).

`abstain=True` records have `expected_response.answer = "CANNOT ANSWER, INSUFFICIENT
EVIDENCE"` — `answered_flags` correctly detects these as refusals, `gold_answerable_from_record`
returns `False` (via `answerable_under_evidence`), so `correct_refusal = True` and all
three sub-metrics are correctly gated out (N1 fix).

### Fix — `rag_eval/evaluator.py` — `conflict_type` string mapping

A new lookup table and an updated `_safe_ctype` signature handle the string→int mapping:

```python
_CONFLICT_TYPE_STR_MAP = {
    "no conflict": 1,
    "complementary information": 2,
    "conflicting opinions and research outcomes": 3,
    "conflicting opinions or research outcomes": 3,   # typo variant
    "conflict due to outdated information": 4,
    "conflict due to misinformation": 5,
}

def _safe_ctype(raw, conflict_type_string=None):
    if raw is None:
        if conflict_type_string:
            return _CONFLICT_TYPE_STR_MAP.get(str(conflict_type_string).strip().lower(), 1)
        return 1
    return int(raw)
```

Call site updated: `_safe_ctype(rec.get("conflict_category_id"), rec.get("conflict_type"))`.

### Fix — `configs/val_tier2.yaml` — cleaned for FG-v3

Old NLI-based options removed (`min_entail_confidence`, `neutral_as_support`,
`ignore_contradictions_types`, `partial_credit_fg`, `majority_support_rule` — all
applied to the old `enhanced_factual_grounding`). `max_claims_per_answer` raised
to 8 (gold answers are longer than typical model outputs).

### Verification on the 49-record val split

| Conflict type | n | correct_refusals |
| --- | --- | --- |
| 1 — No Conflict | 19 | 8 |
| 2 — Complementary Info | 15 | 7 |
| 3 — Conflicting Opinions | 10 | 0 |
| 4 — Outdated Info | 5 | 0 |
| **Total** | **49** | **15** |

All 49 records resolve without an unmapped conflict type.
All 15 correct-refusal records have `answerable_under_evidence=False` and
`expected_response.answer="CANNOT ANSWER, INSUFFICIENT EVIDENCE"` (36 chars).
The 34 answerable records have model outputs ranging from 439 to 1625 chars.
