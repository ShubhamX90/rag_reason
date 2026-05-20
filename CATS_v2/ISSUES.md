# CATS_v2 — Issues and Bug Report

**Scope:** All modules except `batch_processor.py` (ignored per request).
**Evidence:** Source code in [CATS_v2/rag_eval/](CATS_v2/rag_eval/) + observed behavior in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json) (54 samples, qwen monolithic untuned run).

Issues are graded:
- **P0** — corrupts evaluation results; metrics are not what the docstring claims.
- **P1** — silently wrong on edge cases; bias scores in non-obvious directions.
- **P2** — code smells, dead code, refactor opportunities.

---

## 1. Critical Prompt Bug — Behavior judges see the entire rubric, not the one for the conflict type

**Severity:** P0
**File:** [rag_eval/judge_prompts.py:38-85](CATS_v2/rag_eval/judge_prompts.py#L38-L85)

```python
def behavior_judge_prompt(query: str, answer: str, conflict_type: int) -> str:
    rubric = BEHAVIOR_RUBRIC.get(conflict_type, BEHAVIOR_RUBRIC[1])   # selected but unused
    return f"""
    ...
    Conflict Type: {conflict_type}
    Expected Behavior (rubric):
    {BEHAVIOR_RUBRIC}     # <-- bug: dumps the full dict literal of all 5 rubrics
    ...
    """
```

The local variable `rubric` is computed and then thrown away. The f-string interpolates the whole `BEHAVIOR_RUBRIC` dict, so every judge sees a Python-dict literal containing all 5 rubric entries on every call. The judge then has to guess which row applies, and gets only one numeric hint ("Conflict Type: 3").

**Confirmed in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json):**
- Sample `#0244` (type 3, "Conflicting Opinions"): Haiku's rationale begins **"The answer follows the 'Complementary Information' behavior…"** — i.e., it judged using the Type 2 rubric instead of Type 3.
- Sample `#0471` (type 5, "Misinformation"): Three of four judges open their rationale with **"No Conflict scenario…"** — Type 1 rubric on a Type 5 sample.
- Sample `#0046` (type 2, "Complementary"): Haiku says **"The answer follows the 'No Conflict' behavior…"**.
- Sample `#0325` (type 1): DeepSeek says **"consistent with Conflict Type 3 behavior"** — opposite conflict type entirely.

This is the single biggest source of judge disagreement in the dataset. Many `votes_for=1, votes_against=3` and `2–2` ties on simple queries appear to be one judge picking the wrong rubric.

**Fix:** replace `{BEHAVIOR_RUBRIC}` with `{rubric}` in the f-string at line 66.

---

## 2. Logical Bugs in `conflict_eval.py`

### 2.1 Partial-match credit uses the wrong confidence

**Severity:** P0
**File:** [rag_eval/conflict_eval.py:255-273](CATS_v2/rag_eval/conflict_eval.py#L255-L273)

```python
if decision.adherent:
    matches.append(...)
elif decision.confidence > 0.3:   # Partial match threshold
    partial_matches.append(...)
```

`decision.confidence` is the **winning side's** strength (computed in `_weighted_majority_vote` as `max(weighted_votes_for, weighted_votes_against) / total_weight`). When `adherent=False`, that confidence reflects how *certain the committee is that the gold answer is NOT present*. Using it as a partial-match threshold awards more partial credit precisely when the committee is most confident the answer is missing — the inverse of the intended behaviour.

**Observed in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json):**
- Sample `#0325` — gold `"No"`, model said "the evidence is mixed". Votes 1-for/3-against (committee strongly says the gold is NOT present). `confidence = 0.7143 > 0.3` → partial match recorded → `single_truth_recall_score = 0.5`. Should be 0.0.

**Fix:** Either invert the metric (use `1 - decision.confidence` if `not adherent`), capture per-judge raw votes/scores, or remove the partial-credit path entirely. As written, the formula `recall + len(partial)*0.5/len(gold_iter)` lets a single dissenting judge produce 0.5 recall on a clearly-missing answer.

### 2.2 Empty-answer fallback fabricates committee-shaped output

**Severity:** P1
**File:** [rag_eval/conflict_eval.py:37-45](CATS_v2/rag_eval/conflict_eval.py#L37-L45)

```python
if not (answer or "").strip():
    return {
        "adherent": False,
        "votes_for": 0,
        "votes_against": 1,    # synthetic single negative vote
        "confidence": 1.0,
        "committee_details": None
    }
```

For empty answers the function returns `votes_against=1` (no `total_votes` field) and `committee_details=None`. Downstream code that schemas-checks `"committee_details"` may not handle this correctly. Errors and empty answers become indistinguishable in the aggregate scoring.

**Observed:** Samples `#0069`, `#0090`, `#0002` in the qwen run — `committee_details: null`, `votes_against: 1`. Aggregation simply counts these as `behavior_score=0.0`, with no marker that the judges were never consulted.

**Fix:** Use a distinct sentinel (e.g., `"skipped": "empty_answer"`) and exclude these from behavior-adherence aggregation (or report two numbers: behavior over judged samples, and skip rate).

### 2.3 Factual grounding uses only `judges[0]` — committee is a single judge here

**Severity:** P1
**File:** [rag_eval/conflict_eval.py:136-157](CATS_v2/rag_eval/conflict_eval.py#L136-L157)

```python
if hasattr(committee, 'judges') and len(committee.judges) > 0:
    first_judge = committee.judges[0]    # always Haiku in the default committee
    for doc in support_docs:
        ...
        nli_result = await first_judge.judge_nli(prompt)
```

Despite the docstring promising "Enhanced factual grounding with cross-document verification", grounding is computed with a single judge — by index, the first one passed in (Haiku in the default committee). The other three judges in the committee never see NLI calls. If Haiku rate-limits/errors on a sample, that sample's NLI score silently goes to 0.0 for *every* claim.

**Fix:** Either run NLI through the full committee with a separate `judge_nli` aggregation, or document the single-judge behaviour and make the chosen judge configurable.

### 2.4 Hidden self-import inside both single-truth recall functions

**Severity:** P2 (works today; will break under refactor)
**Files:** [rag_eval/conflict_eval.py:218](CATS_v2/rag_eval/conflict_eval.py#L218), [rag_eval/conflict_eval.py:283](CATS_v2/rag_eval/conflict_eval.py#L283)

```python
async def enhanced_single_truth_recall(...):
    from .conflict_eval import _iter_gold_answers   # importing from own module
```

`_iter_gold_answers` is defined in the same file (line 308), so this is a no-op import. It survives only because Python tolerates circular imports of already-loaded modules. Any refactor (e.g., moving the helper out) will break silently in a confusing way.

**Fix:** Delete the import lines — `_iter_gold_answers` is already in module scope.

### 2.5 Cross-doc verification threshold isn't really cross-doc

**Severity:** P2
**File:** [rag_eval/conflict_eval.py:159-160](CATS_v2/rag_eval/conflict_eval.py#L159-L160)

```python
is_supported = support_count > 0
if require_cross_doc:
    is_supported = support_count >= 2
```

"Cross-doc" here means "two docs entailed the claim". But the NLI judge can hallucinate entailment (it has been observed to do so — see §6.2). With a single judge running NLI, two false positives count the same as two true positives. Without independent confirmation across judges, `require_cross_doc=True` provides weaker guarantees than the name suggests.

### 2.6 Gold-answer iterator silently drops non-string entries

**Severity:** P2
**File:** [rag_eval/conflict_eval.py:308-316](CATS_v2/rag_eval/conflict_eval.py#L308-L316)

If gold_answer is `["1759", 1759]` (int alongside string), the int is dropped without warning. If the upstream pipeline ever serialises a number as a Python int, recall silently zeroes.

**Fix:** `[str(g) for g in gold_answers if g not in (None, "")]`.

---

## 3. Bugs / Logical Issues in `config.py`

### 3.1 Judge priorities are hardcoded inside factory functions

**Severity:** P1 (this is the modularity ask in the task)
**File:** [rag_eval/config.py:252-355](CATS_v2/rag_eval/config.py#L252-L355)

Priorities (Haiku=2, DeepSeek=3, Qwen=1, Mistral=1) are baked into `get_haiku_judge()`, `get_deepseek_judge()`, etc. To re-tune the committee you must edit code. There is also no way for a CLI/YAML user to override them.

**Recommended refactor:**

```python
# config.py
DEFAULT_JUDGE_PRIORITIES = {
    "claude-3-5-haiku-20241022": 2,
    "deepseek/deepseek-r1":      3,
    "qwen/qwen-2.5-7b-instruct": 1,
    "mistralai/mistral-nemo":    1,
}

def get_haiku_judge(priority: Optional[int] = None) -> JudgeModelConfig:
    return JudgeModelConfig(
        model_id="claude-3-5-haiku-20241022",
        ...
        priority=priority if priority is not None else DEFAULT_JUDGE_PRIORITIES["claude-3-5-haiku-20241022"],
    )

def create_default_committee(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeCommitteeConfig:
    p = {**DEFAULT_JUDGE_PRIORITIES, **(priority_overrides or {})}
    return JudgeCommitteeConfig(
        judges=[
            get_haiku_judge(p["claude-3-5-haiku-20241022"]),
            get_deepseek_judge(p["deepseek/deepseek-r1"]),
            get_qwen_judge(p["qwen/qwen-2.5-7b-instruct"]),
            get_mistral_nemo_judge(p["mistralai/mistral-nemo"]),
        ],
        ...
    )
```

This enables a YAML override of the form:
```yaml
committee:
  priority_overrides:
    deepseek/deepseek-r1: 1
    qwen/qwen-2.5-7b-instruct: 3
```

### 3.2 Dead config fields

**Severity:** P2
**File:** [rag_eval/config.py](CATS_v2/rag_eval/config.py)

Fields declared but never read anywhere in the codebase:
- `JudgeModelConfig.max_requests_per_minute`, `max_tokens_per_minute` (no rate limiter exists).
- `JudgeCommitteeConfig.retry_attempts`, `timeout_seconds` (the per-judge `httpx.AsyncClient(timeout=30.0)` is hardcoded; no retry logic).
- `JudgeCommitteeConfig.max_cost_per_sample`, `cost_optimization`, `prefer_cheaper_models` (no cost-gating logic exists).
- `JudgeCommitteeConfig.confidence_threshold` — never compared against in any voting function.
- `JudgeCommitteeConfig.max_concurrent_requests` — see §3.3.
- All of `EnhancedTrustScoreConfig` and most of `EnhancedConflictEvalConfig` (citation accuracy, temporal consistency, conflict_resolution_score, viewpoint_balance, …) — these flags advertise features that are not implemented.
- `ModelConfig` is initialised in `EvaluationConfig` but never used by the evaluator.

These deceive readers: a user enabling `weight_by_source_quality=True` sees no change because no code reads it.

**Fix:** Remove unused fields, OR implement them, OR move them to an explicit "future work" stub with a runtime warning when set.

### 3.3 `max_concurrent_requests` is set everywhere but enforced nowhere

**Severity:** P0 for large runs (API blowup), P1 in general
**File:** [rag_eval/config.py:72,330,351](CATS_v2/rag_eval/config.py#L72) + [rag_eval/judge_committee.py:379-402](CATS_v2/rag_eval/judge_committee.py#L379-L402)

The value flows through `JudgeCommitteeConfig`, but `JudgeCommittee.judge_behavior` does `asyncio.gather(*tasks)` with no semaphore. With 54 samples × (4 judges + Haiku-NLI per claim × ~3 claims + 4 judges per gold-answer recall) the actual fan-out reaches several hundred simultaneous outbound calls. The qwen run hit DeepSeek latencies of 38s and 45s on individual calls (samples `#0254`, `#0030`) which is consistent with rate-limit throttling, not the model's normal speed (~3-5s).

**Fix:** Wrap fan-out with `asyncio.Semaphore(committee.config.max_concurrent_requests)` either at the committee level (`judge_behavior` per-judge) or at the evaluator level (per-sample).

### 3.4 OpenRouter rate limits set per judge, then ignored

**Severity:** P1
**Files:** [rag_eval/config.py:279,295,311](CATS_v2/rag_eval/config.py#L279) + [rag_eval/judge_committee.py:99-128](CATS_v2/rag_eval/judge_committee.py#L99-L128)

`max_requests_per_minute=30` on DeepSeek is set but there is no token-bucket / leaky-bucket. The first 30 requests hit the API; the next 30 also hit immediately; OpenRouter responds with HTTP 429 and `response.raise_for_status()` lets the exception propagate. The judge returns `adherent=False`, which leaks into the aggregate metric as if the model genuinely violated the rubric.

### 3.5 Global singletons re-create the default committee at import time

**Severity:** P2
**File:** [rag_eval/config.py:361-370](CATS_v2/rag_eval/config.py#L361-L370)

```python
conflict_cfg = EnhancedConflictEvalConfig(
    use_judge_committee=True,
    committee=create_default_committee()
)
```

`create_default_committee()` runs on import even when the caller is going to build its own `EvaluationConfig` (which `setup_config()` does, in [run_evaluation.py:100-145](CATS_v2/run_evaluation.py#L100-L145)). The singleton is never used by the runner, so this is dead work — but it does mean importing `rag_eval` always constructs four `JudgeModelConfig` objects for nothing.

---

## 4. Bugs / Logical Issues in `judge_committee.py`

### 4.1 Confidence field is parsed but never produced

**Severity:** P0
**File:** [rag_eval/judge_committee.py:305](CATS_v2/rag_eval/judge_committee.py#L305) + [rag_eval/judge_prompts.py](CATS_v2/rag_eval/judge_prompts.py)

`_parse_judge_response` reads `obj.get("confidence", 1.0)`. Neither `behavior_judge_prompt` nor `nli_prompt` ever asks the judge to emit a confidence value. Therefore `parsed["confidence"]` is **always 1.0** in practice.

`_weighted_majority_vote` then computes `weight = priority * r.confidence` (line 457) — so the confidence factor is a no-op and the effective weight is just `priority`. Every "weighted" vote in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json) shows `"confidence": 1.0` for every individual judge.

**Fix path A (cheap):** Drop the confidence dimension; document that voting is priority-only.
**Fix path B (do it right):** Add `"confidence": 0.0-1.0` to the JSON schema in `behavior_judge_prompt` and ask the judge to emit it. Caveat: small models don't calibrate well. Better to derive confidence from log-probs (if the provider exposes them) or from inter-judge agreement.

### 4.2 DeepSeek R1 reasoning trace likely truncates against `max_tokens=500`

**Severity:** P0 (very likely silently dropping DeepSeek votes)
**Files:** [rag_eval/config.py:273](CATS_v2/rag_eval/config.py#L273) + [rag_eval/judge_committee.py:130-183](CATS_v2/rag_eval/judge_committee.py#L130-L183)

DeepSeek R1 emits a `<think>…</think>` chain-of-thought before the final answer. Routine reasoning traces for this kind of task are 800-2000 tokens. With `max_tokens=500`, the reasoning often fills the budget and the JSON never appears. `_parse_judge_response` then logs a warning and returns `adherent=False, confidence=0.0`.

Two pieces of evidence in the qwen run support this:
- DeepSeek latencies of 38s (`#0254`), 36s (`#0215`), 39s (`#0339`), 45s on the long-tail — consistent with hitting the token cap.
- Several samples have DeepSeek's `rationale` identical to the committee's chosen rationale even when other judges disagree — because the highest-priority judge that "succeeded" is DeepSeek (priority=3), and the rationale-selector picks it whether or not its output was sensible. (See §4.5.)

**Fix:** Raise `max_tokens` for DeepSeek R1 to ~2000-3000, OR drop the reasoning model entirely, OR special-case the parser to keep only the trailing JSON after `</think>`.

### 4.3 `_call_openrouter` indexes `data["choices"][0]` blindly

**Severity:** P1
**File:** [rag_eval/judge_committee.py:286](CATS_v2/rag_eval/judge_committee.py#L286)

```python
output_tokens = usage.get("completion_tokens", len(data["choices"][0]["message"]["content"]) // 4)
return data["choices"][0]["message"]["content"], input_tokens, output_tokens
```

OpenRouter (and free-tier providers especially) sometimes return `{"error": ...}` with `status=200` and no `choices` array, or `choices=[]` when upstream content-filters fire. The resulting `IndexError`/`KeyError` is caught in `judge_behavior`'s outer `try/except`, becomes `adherent=False`, and pollutes the score.

**Fix:**
```python
choices = data.get("choices") or []
if not choices:
    raise RuntimeError(f"OpenRouter returned no choices: {data}")
```

### 4.4 `_parse_nli_response` is defined but unused

**Severity:** P2 (dead code)
**File:** [rag_eval/judge_committee.py:315-352](CATS_v2/rag_eval/judge_committee.py#L315-L352)

`judge_nli` does inline JSON extraction (lines 203-215). `_parse_nli_response` is never called. The two implementations have diverged: the inline version returns `"neutral"` on any malformed JSON; the dead helper has a substring fallback to `"entails"`/`"contradicts"`. Pick one and delete the other.

### 4.5 Weighted vote picks rationale from "winning side, highest priority" — biases reports toward DeepSeek

**Severity:** P1
**File:** [rag_eval/judge_committee.py:468-475](CATS_v2/rag_eval/judge_committee.py#L468-L475)

```python
best_response = max(winning_responses, key=lambda r: priority_map.get(r.model_id, 1) * r.confidence)
rationale = best_response.rationale
```

Because confidence is always 1.0 (§4.1) and DeepSeek has priority 3, the displayed `rationale` is **DeepSeek's** in every sample where DeepSeek sided with the majority. This is observable throughout the qwen run — most committee-level rationales match DeepSeek's verbatim. If DeepSeek truncates (§4.2) and the parser falls back, the displayed rationale becomes "Parse error" or DeepSeek doesn't appear in `winning_responses` at all — meaning the reported reasoning quality silently degrades.

### 4.6 `_unanimous_vote` confidence is meaningless binary

**Severity:** P2
**File:** [rag_eval/judge_committee.py:502](CATS_v2/rag_eval/judge_committee.py#L502)

`confidence = 1.0 if all_adherent else 0.0` — but `all_adherent=False` simply means at least one disagreed, which is not the same as "we're certain it's false". The dataclass field should be `Optional[float]` or the value should encode disagreement strength.

### 4.7 All judges get temperature=0.0 then we count their "disagreement"

**Severity:** P1 (methodological)
**File:** [rag_eval/config.py:257,272,288,304](CATS_v2/rag_eval/config.py#L257)

With `temperature=0`, each judge is deterministic given the same prompt. The same model rerun yields the same vote — so the committee is not measuring epistemic uncertainty, just static disagreement between models. That can still be useful, but the report language implies the committee captures uncertainty, which it does not.

If the intent is to measure real disagreement, run each judge at `temperature≈0.3` with N=3 samples and use majority of the per-judge majority.

---

## 5. Data-schema / Pipeline issues (`data.py`, `evaluator.py`, `metrics.py`, `run_evaluation.py`)

### 5.1 `conflict_category_id = 0` silently becomes `1`

**Severity:** P0
**File:** [rag_eval/evaluator.py:139](CATS_v2/rag_eval/evaluator.py#L139)

```python
ctype = int(rec.get("conflict_category_id") or 1)
```

`0 or 1 == 1` in Python. Any record annotated with `conflict_category_id=0` (which some upstream pipelines use as "uncategorised") is silently re-labelled as Type 1 ("No Conflict") and judged against the wrong rubric.

**Fix:**
```python
raw = rec.get("conflict_category_id")
ctype = int(raw) if raw is not None else 1
```

### 5.2 `_aggregate_results` crashes on unexpected conflict types

**Severity:** P0
**File:** [rag_eval/evaluator.py:205-228](CATS_v2/rag_eval/evaluator.py#L205-L228)

`per_type = {k: ... for k in [1, 2, 3, 4, 5]}` then `per_type[ctype][...]` — any ctype outside {1..5} (negative number, 6, "outdated_v2") raises `KeyError`. Combined with §5.1, an out-of-range ctype that isn't `0` blows up the whole evaluation **after** all API calls have been made and paid for.

**Fix:** `per_type.setdefault(ctype, {"n": 0, ...})`.

### 5.3 `model_output` fallback to `final_grounded_answer.answer` mixes gold and prediction

**Severity:** P0 if it ever fires on real data
**File:** [rag_eval/data.py:100-108](CATS_v2/rag_eval/data.py#L100-L108)

```python
def get_model_output(record):
    if "model_output" in record and record["model_output"]:
        return record["model_output"]
    return record.get("final_grounded_answer", {}).get("answer", "") or ""
```

`final_grounded_answer.answer` is the **gold annotation** (per the schema docstring in [rag_eval/data.py:25-30](CATS_v2/rag_eval/data.py#L25-L30): "evidence", "abstain" — these are annotation fields). If a record lacks `model_output` (e.g., model timed out and the JSONL was patched), every metric is scored against the gold answer instead of the model's actual output — producing artificially high scores.

**Fix:** Raise an explicit error when `model_output` is missing or empty, OR mark the sample as "no output" and exclude it from scoring. The silent fallback is unsafe.

### 5.4 Verdict matching is case-sensitive

**Severity:** P1
**File:** [rag_eval/data.py:84-85](CATS_v2/rag_eval/data.py#L84-L85)

```python
verdict = (n.get("verdict") or "").lower()
if verdict == "supports" or (accept_partial and verdict == "partially supports"):
```

A note with `verdict="Supports"` is fine (lowered first). But `verdict="partial"`, `verdict="partial support"`, `verdict="partial_supports"`, or `verdict="weakly supports"` — all of which are reasonable annotator outputs — are silently treated as **irrelevant**. That flips `gold_answerable` from `True` to `False`, which flips `f1_gr` from 1.0 to 0.0 in either direction.

**Fix:** Whitelist with normalisation:
```python
verdict_norm = verdict.replace("_", " ").strip()
positive = {"supports", "partially supports", "partial supports", "partial support", "support"}
if verdict_norm in positive or (accept_partial and "partial" in verdict_norm):
    ...
```

### 5.5 `gold_answerable = at least one note exists` ignores `retrieved_docs` without notes

**Severity:** P1
**File:** [rag_eval/data.py:93-97](CATS_v2/rag_eval/data.py#L93-L97)

A record may have `retrieved_docs=[d1..d10]` but `per_doc_notes=[]` (or `per_doc_notes` only for a subset). In that case `gold_answerable=False`, but the question may genuinely be answerable from the un-annotated docs — and the model may have answered correctly. The annotation gap silently penalises the model.

### 5.6 `extract_claims_by_sentence` mis-splits on initials and abbreviations

**Severity:** P0 for accurate grounding
**File:** [rag_eval/metrics.py:40-64](CATS_v2/rag_eval/metrics.py#L40-L64)

NLTK `sent_tokenize` splits on the period after middle initials, ordinal abbreviations, and "etc." Observed in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json):

- Sample `#0339`: `"In 1965, Lyndon B."` and `"Johnson was the President…"` are two separate "claims". The first is unsupported by any doc (no claim, just half a name). The second is supported. Reported `grounding_ratio=0.5` when the model actually wrote one correct sentence.
- Sample `#0085`: `"[d4] and  report 1,759 and 658 confirmed tornadoes respectively…"` is treated as a claim. NLI naturally returns "not entailed" for citation-format fragments.
- Across the run, the bullet-list and citation-marker outputs that Qwen produces split into 5–8 claims, of which 1–2 are pure citation noise, dragging grounding_ratio below the true value.

**Fix:** Pre-process the answer (strip citation markers with `remove_citations` — which exists in `metrics.py:126` but is unused here — and standalone bracketed lists) before sentence-splitting. Consider using `nltk.PunktSentenceTokenizer` with a custom abbreviation list, or a regex-based clause extractor for citation-heavy outputs.

### 5.7 `f1_gr_from_flags` is binary accuracy, not F1

**Severity:** P1 (methodological)
**File:** [rag_eval/metrics.py:95-107](CATS_v2/rag_eval/metrics.py#L95-L107)

```python
def f1_gr_from_flags(pred_answered, gold_answerable) -> float:
    return 1.0 if int(pred_answered) == int(gold_answerable) else 0.0
```

This is per-sample accuracy, then macro-averaged in `_aggregate_results`. Real F1 needs TP/FP/FN over the dataset. The metric is mis-named, the docstring lies, and the value will not match what a reader compares against in the literature.

**Fix:** Rename to `gr_accuracy` (or compute proper F1: TP=both True, FP=pred True & gold False, FN=pred False & gold True; F1 = 2TP/(2TP+FP+FN)).

### 5.8 `answered_flags` refusal detection is English-only and English-specific

**Severity:** P1
**File:** [rag_eval/metrics.py:14-37](CATS_v2/rag_eval/metrics.py#L14-L37)

Refusal patterns are a hardcoded list of English substrings. Models that refuse in another language, with "Unable to determine", "Not enough context provided", "No reliable evidence to answer", or with a structured refusal like `{"abstain": true}` will be classified as "answered=True". Any future multilingual evaluation needs this rewired.

### 5.9 Refusal substring vs `startswith` is inconsistent

**Severity:** P2
**File:** [rag_eval/metrics.py:24-33](CATS_v2/rag_eval/metrics.py#L24-L33)

Some patterns use `startswith` (e.g., `"i cannot"`), others use `in` (e.g., `"cannot answer"`). A response that mentions "I cannot list every case but here is…" gets flagged as a refusal because `"i cannot"` startswith fires. Mixed-policy is bug-prone. Use a single normalised regex.

### 5.10 `evaluator.evaluate()` calls `asyncio.run()` inside a possibly-running loop

**Severity:** P2
**File:** [rag_eval/evaluator.py:95-97](CATS_v2/rag_eval/evaluator.py#L95-L97)

```python
def evaluate(self, dataset):
    return asyncio.run(self.evaluate_async(dataset))
```

This raises `RuntimeError: asyncio.run() cannot be called from a running event loop` if invoked from Jupyter or any host that has its own loop. The CLI path doesn't hit this, but anyone integrating the evaluator as a library does.

### 5.11 `per_type` keys become strings after JSON round-trip

**Severity:** P2
**File:** [rag_eval/evaluator.py:205-211](CATS_v2/rag_eval/evaluator.py#L205-L211)

`per_type = {1: ..., 2: ...}` is written to JSON, where keys become `"1"`, `"2"`. Visible in `detailed_results.json` line 11-46. Any downstream tool expecting integer keys after reload needs to coerce. Cheap pre-emptive fix: stringify in Python before serialising.

### 5.12 Sample IDs preserve only `rec.get("id")` — no fallback uniqueness guard

**Severity:** P2
**File:** [rag_eval/evaluator.py:137](CATS_v2/rag_eval/evaluator.py#L137)

If two records share the same `id` or both lack one, both samples get `sample_id="sample_<idx>"` for different `idx` values — fine — but if duplicate `id` values exist they collide silently. Worth a uniqueness check on dataset load.

### 5.13 `--config` argument accepted but YAML never loaded

**Severity:** P1 (silent UX bug)
**Files:** [run_evaluation.py:68-72](CATS_v2/run_evaluation.py#L68-L72), [configs/default.yaml](CATS_v2/configs/default.yaml)

`argparse` has `--config`, but `setup_config()` never opens the file. Users editing `configs/default.yaml` see zero effect on the run. This is the second-biggest UX trap in the project (after §1).

**Fix:** Either load YAML and overlay onto the dataclasses, or remove the flag and the example file.

### 5.14 `.env.example` omits `OPENROUTER_API_KEY`

**Severity:** P2 (UX)
**File:** [CATS_v2/.env.example](CATS_v2/.env.example)

`run_evaluation.py:127` exits with error if `OPENROUTER_API_KEY` is missing for default/conservative committees, but the example file only lists `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. New users hit `sys.exit(1)` without guidance.

### 5.15 `setup_file_logging` adds handlers without deduplication

**Severity:** P2
**File:** [rag_eval/logging_config.py:31-44](CATS_v2/rag_eval/logging_config.py#L31-L44)

Each call appends a new `FileHandler`. Repeated calls (e.g., in tests, or in the batch runner when multiple files are processed) produce N× log lines. Add a guard:
```python
if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    return
```

### 5.16 `asyncio.as_completed` produces non-deterministic per-sample order

**Severity:** P2
**File:** [rag_eval/evaluator.py:113-120](CATS_v2/rag_eval/evaluator.py#L113-L120)

`per_sample_results` is appended in completion order. Two runs over the same dataset produce JSON in different orders. Comparing two `detailed_results.json` files with `diff` is impossible. After completion, sort by `sample_id` (or by original index) before writing.

---

## 6. Logical issues visible in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json)

### 6.1 Type 5 has n=1 — that bucket is statistical noise

**Severity:** P1 (dataset/aggregation)
**Evidence:** summary lines 39-45 — Type 5 (Misinformation) has only 1 sample (#0471). Reporting `f1_gr=0.000`, `behavior=1.000`, `factual_grounding=0.000`, `single_truth_recall=0.000` is meaningless — these are single-sample numbers. The Type 5 row should be flagged as "n<5: not reliable" or dropped from the headline summary.

### 6.2 Factual grounding hallucinates support

**Severity:** P0 (the most damaging silent bug in NLI)
**Evidence:** Sample `#0244` (type 3, "Conflicting Opinions") in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json:2862-2986):

The model's answer is itself a meta-statement: `"No conflict — The evidence supports the reality of the Temple of Solomon, with some uncertainty…"`. This is **not a claim about the world**; it's a claim about the documents. Yet `factual_grounding_score=1.0`, with `supported_claims=5/5`. The NLI judge (Haiku) entailed:

- *"Therefore, the Temple of Solomon is considered real based on the evidence provided."* — judged supported by `[d7]` alone.

Haiku's NLI judge is treating "X is real based on the evidence" as entailed by a document that merely *mentions* the topic. This is over-credit. The grounding ratio is high not because the answer is grounded but because the NLI judge is permissive.

**Implication:** The headline `factual_grounding=0.591` overall is inflated by similar over-credit. Recommend running a calibration test: a held-out set with known-unsupported claims, measure Haiku's false-positive rate on `entails`.

### 6.3 Single-truth recall is structurally zero for Type 3

**Severity:** P1 (design)
**Evidence:** Type 3 row: `single_truth_recall=0.000` across all 11 samples. By design, `enhanced_single_truth_recall` is only called when `ctype in {1,2,4,5}` AND `gold_answer` is present ([rag_eval/evaluator.py:171](CATS_v2/rag_eval/evaluator.py#L171)). For Type 3, `gold_answer` is intentionally null. But the **CATS Score averages all four metrics** — so every Type 3 sample contributes a 0.0 to the recall slot, dragging the composite. This is a structural penalty against any model that gets a balanced Type 3 mix.

**Fix:** Compute CATS Score per-type (excluding the structurally-zero metric for Types 3), then macro-average. Or: explicitly weight `single_truth_recall` only over Types {1,2,4,5}.

### 6.4 Empty answers (4 of the 54 samples shown) get `confidence=1.0` against — fake certainty

**Severity:** P1
**Evidence:** Samples `#0069`, `#0090`, `#0002` (and likely more) in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json:48-125) have `committee_details: null`, `votes_against: 1`, `confidence: 1.0`. The committee was never called. Treating these as "high-confidence non-adherent" inflates the apparent strength of negative votes in any per-judge analysis.

### 6.5 Same conflict-type, opposite behavior judgments on near-identical inputs

**Severity:** P1 (correlates with §1)
**Evidence:**

- `#0463`, `#0504`, `#0481`, `#0503`, `#0544` are all Type 1 with model output containing `"CANNOT ANSWER, INSUFFICIENT EVIDENCE"`. Haiku judges:
  - `#0463`: non-adherent ("inappropriate handling of a simple query")
  - `#0481`: adherent ("model directly acknowledges lack of information")
  - `#0503`: adherent ("provides a direct response indicating lack of information")
  - `#0504`, `#0544`: non-adherent

The same model, same conflict type, same response pattern, opposite judgments. This is exactly what §1 predicts: the judge sees the entire rubric and picks differently each time depending on query specifics. The committee then masks the inconsistency by voting.

### 6.6 Citation-style noise inflates claim count and tanks grounding

**Severity:** P1
**Evidence:**
- `#0339` (line 4627): `factual_grounding_score=0.5` because `"In 1965, Lyndon B."` and `"[d1], [d2], [d5], and [d6] all explicitly state this fact."` were each treated as standalone claims. Only the substantive sentence was entailed.
- `#0195` (line 4759): same pattern — `"d1, d3, and d5 provide evidence supporting this link."` is a meta-claim about citations, not a substantive claim, and the NLI judge correctly says it isn't entailed by any doc, so grounding drops from 1.0 to 0.5.

The model isn't ungrounded; the claim-extractor is over-splitting.

### 6.7 Outdated-info samples (Type 4) score 0.5 behavior — judges aren't using dates

**Severity:** P1 (per Q5 in the prior open-questions list)
**Evidence:** Type 4 row: `behavior=0.500` (3 of 6 adherent). Sample `#0085` got 4-0 against with rationale "fails to prioritize the up-to-date figure". But nowhere in the prompt is the document `date` passed to the judge — the judge is inferring "outdatedness" from how the answer hedges, not from actual dates. A model that gives a date-aware answer to a Type 4 query may still be marked non-adherent because its **answer text** doesn't say "the newer source says…".

**Fix:** Pass `retrieved_docs[].date` into the behavior judge prompt for Type 4 (and arguably Type 5).

### 6.8 DeepSeek's high latency dominates per-sample wall time

**Severity:** P2 (cost/runtime)
**Evidence:** Looking at `total_latency_ms` across samples in [outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json](CATS_v2/outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json): committee decisions routinely take 30-45 seconds, of which DeepSeek alone takes 20-40 seconds (see `#0254`, `#0215`, `#0339`). Haiku, Qwen, and Mistral together usually finish under 9 seconds. The pipeline's tail is single-model bound; cutting DeepSeek would 3-4× throughput. Worth considering as an option (`--no-deepseek`) for large runs.

### 6.9 Mistral Nemo always reports cost=0

**Severity:** P2
**Evidence:** every sample's Mistral row has `cost: 0.0`. This is consistent with `cost_per_1k_input=0.0, cost_per_1k_output=0.0` in [rag_eval/config.py:306-307](CATS_v2/rag_eval/config.py#L306-L307) — i.e., assumed free. But OpenRouter occasionally charges for nominally-free models when the free tier is rate-limited and a paid fallback engages. The reported `total_cost` per run is slightly under-estimated.

### 6.10 Composite CATS Score under-reports model quality

**Severity:** P1 (methodological)
**Evidence:** Summary row gives `f1_gr=0.852, behavior=0.648, factual_grounding=0.591, single_truth_recall=0.361`. The recall is depressed by Type 3 structural zeros (§6.3). The behavior is depressed by the rubric-mismatch bug (§1). The grounding is partly inflated (§6.2) and partly deflated (§6.6) — direction unclear. Stacking these together gives a single number that no one can interpret.

**Recommendation:** Report the four metrics separately with per-type breakdowns + n; deprecate the single CATS Score, or compute it only over Type 1 samples where every metric is well-defined.

---

## 7. Additional logical issues found by deep-reading the qwen run

These issues were discovered by reading every sample in
`outputs/untuned_generations/qwen/monolithic/e2e/detailed_results.json` and
checking whether the logged input/output/metric values made logical sense.
None of these is currently fixed in code.

### N1 — Correct refusals are triple-penalized

**Severity:** P0 (composite CATS score is wrong by design for correct refusals)
**Files:** [rag_eval/evaluator.py](CATS_v2/rag_eval/evaluator.py), [rag_eval/conflict_eval.py](CATS_v2/rag_eval/conflict_eval.py)
**Evidence:** Sample `#0463` (Type 1, model output: "CANNOT ANSWER, INSUFFICIENT EVIDENCE", gold unanswerable).

| Metric              | Actual score | Why it is wrong                                          |
|---------------------|--------------|----------------------------------------------------------|
| `gr_accuracy`       | 1.0          | Correct — pred=False AND gold=False, so accuracy = 1    |
| `behavior_score`    | 0.0          | "No Conflict" rubric expects a direct answer; refusal fails |
| `factual_grounding` | 0.0          | No claims extracted from refusal text                   |
| `single_truth_recall` | 0.0        | No answer to recall the gold from                       |
| **CATS composite**  | **0.25**     | "model failed" verdict on a correctly-handled sample    |

When `pred_answered=False` AND `gold_answerable=False`, the model made exactly
the right call. All downstream metrics (behavior, grounding, recall) zero out on
an answer that doesn't exist — they are measuring the wrong thing.

**Fix:** When `pred_answered=False AND gold_answerable=False`, skip the behavior/
grounding/recall calls and record all metrics as 1.0 with a `skipped: "correct_refusal"`
sentinel so the result is auditable.

---

### N2 — Recall judges count gold strings *mentioned* but not *asserted*

**Severity:** P1 (false-positive recall on answers that explicitly reject the gold)
**Files:** [rag_eval/judge_prompts.py](CATS_v2/rag_eval/judge_prompts.py), [rag_eval/conflict_eval.py](CATS_v2/rag_eval/conflict_eval.py)
**Evidence:** Sample `#0085` (Type 4, gold="at least 1,759", model output committed to "658" but quoted "1,759" from a document it was contrasting).

The recall prompt asks if the gold answer "appears in" the model answer. It does —
the model cited "1,759" — but the model's actual conclusion was "658". The
committee voted 3-of-4 that the gold was present, giving `single_truth_recall=1.0`
even though the model gave the wrong number.

**Fix:** Rewrite `single_truth_recall_prompt` to ask the judge whether the model
is *asserting* the gold as its answer, not whether the gold string is *mentioned*
anywhere. Provide explicit examples like:
- Gold "1,759"; Candidate "Source d4 reports 1,759 but the answer is 658" → false
(model commits to 658, not 1,759).

---

### N3 — Dataset has misspelled gold answers; recall gives split verdicts

**Severity:** P2 (data quality; manifests as recall instability)
**Evidence:** Sample `#0042` (gold="Chiliwack" — 1 'l'; correct spelling "Chilliwack" — 2 'l's).
The model said "Chilliwack" (correct). The committee voted 2-2 on whether
"Chilliwack" matches "Chiliwack", producing partial credit.

**Fix:** Either instruct the recall judge to accept minor spelling variations as
matches, or clean the gold answer strings in the dataset. Code-level fix alone
can't catch all typos; dataset cleanup is needed.

---

### N4 — `votes_for: 2, votes_against: 2` ties are broken by weighted priority, not documented

**Severity:** P2 (surprising behavior; hard to audit without explanation)
**Evidence:** Samples `#0066`, `#0204`, `#0321`. In all three, a 2-2 raw-vote
tie was broken by DeepSeek's priority=3, which was sufficient to tip the
weighted total. The committee report shows `votes_for=2, votes_against=2` but
`adherent=True` — confusing without knowing the weighting.

**Fix (documentation):** In `CommitteeDecision.to_dict()`, also emit
`weighted_for` and `weighted_against` alongside the raw vote counts. Callers
reading the JSON will then understand how the tie was resolved without
re-implementing the voting math.

---

### N5 — `gold_answer` typos in the dataset cause nondeterministic recall

**Severity:** P2 (data quality)
**Evidence:** "Stephan Curry" (#0061; correct: "Stephen Curry"). The recall
judge sometimes treats "Stephen" as matching "Stephan" (acceptable spelling
variation) and sometimes doesn't. This is consistent with N3 — the root cause
is dataset-level annotation errors that propagate into the metric.

**Fix:** Data cleanup is the primary fix. As a code mitigation, see N2/N3 above
(instruct the recall judge to accept minor spelling variations).

---

### N6 — Type 4 samples score `behavior=1.0` even when the model gives an outdated answer

**Severity:** P1 (behavior metric completely misses the Type 4 failure mode)
**Files:** [rag_eval/judge_prompts.py](CATS_v2/rag_eval/judge_prompts.py)
**Evidence:** Sample `#0113` (gold="2023", model confidently answered "2020",
`behavior_score=1.0`, `factual_grounding=1.0`, `single_truth_recall=0.5`).

The behavior judge is told to check if the model "Prioritises the up-to-date
information" (Type 4 rubric). But the judge has **no access to document dates**
— it can only read the answer text. The model's "2020" answer reads as a
confident direct answer, so the judge marks it adherent. Only recall (which
checks against the gold "2023") catches the mistake, and only at half credit.

**Fix:** Pass `retrieved_docs[i].date` into the behavior judge prompt for
Type 4 (and Type 5) samples. The judge needs to see what "up-to-date" means in
context. Without dates, the Type 4 behavior metric is measuring answer
confidence, not temporal correctness.

---

### N7 — DeepSeek dominates tail latency; the committee wall-time is single-model bound

**Severity:** P2 (cost/runtime)
**Files:** [rag_eval/config.py](CATS_v2/rag_eval/config.py), [rag_eval/judge_committee.py](CATS_v2/rag_eval/judge_committee.py)
**Evidence:** In the qwen run, per-sample `total_latency_ms` values routinely
reach 30-45 seconds, of which DeepSeek alone takes 20-40 seconds (samples
`#0254`, `#0215`, `#0339`). Haiku, Qwen, and Mistral together usually finish
under 9 seconds. The committee waits for all judges (`asyncio.gather`) so a
single slow judge controls wall time.

**Fix options:**
- Use `--committee conservative` (no DeepSeek) for large runs.
- Add a per-judge timeout: if a judge doesn't respond within N seconds, treat
  it as an error and exclude from voting.
- Implement `asyncio.wait` with `return_when=FIRST_COMPLETED` and a timeout
  threshold, voting on whoever responded in time.

---

### N8 — Mistral cost is always reported as $0.00 even when OpenRouter uses a paid fallback

**Severity:** P2 (cost under-estimation)
**Files:** [rag_eval/config.py](CATS_v2/rag_eval/config.py)
**Evidence:** Every sample in the qwen run shows `"cost": 0.0` for the Mistral
row. This matches `cost_per_1k_input=0.0, cost_per_1k_output=0.0` in
`get_mistral_nemo_judge()`. However, OpenRouter sometimes silently falls over to
a paid tier when the free tier is rate-limited. In that case the actual run cost
is slightly under-reported because the cost tracking is based on the configured
price, not on the OpenRouter invoice.

**Fix:** Either query the OpenRouter `/api/v1/generation` endpoint after each
call to get actual billed cost, or at least emit a warning when cost is 0 but
the model is known to have a paid fallback.

---

### N9 — `factual_grounding_score=1.0` when `total_claims=2` but `claim_details=[]`

**Severity:** P1 (inconsistent output schema; downstream tools break on the mismatch)
**Files:** [rag_eval/conflict_eval.py](CATS_v2/rag_eval/conflict_eval.py)
**Evidence:** Sample `#0471` (Type 5). When `support_docs` is empty (no doc
notes marked as supporting), `enhanced_factual_grounding` returns early with:

```python
return {
    "grounding_ratio": 0.0,
    "supported_claims": 0,
    "total_claims": len(claims),  # e.g. 2
    "claim_details": [],          # empty — inconsistent with total_claims=2
}
```

Any downstream code that zips `claim_details` with extracted claims gets
mis-aligned output. `len(claim_details) != total_claims` violates the implicit
contract of the return schema.

**Fix:** When returning early for empty support_docs, either:
(a) populate `claim_details` with one `{claim, supported=False, ...}` entry per
claim so the list length always equals `total_claims`, or
(b) set `total_claims=0` (but this loses the information that claims were extracted).
Option (a) is preferred.

---

### N10 — Aggregator includes correct-refusal zeros in the metric mean

**Severity:** P1 (downstream of N1)
**Files:** [rag_eval/evaluator.py](CATS_v2/rag_eval/evaluator.py)

When a model correctly refuses (N1), `behavior_score=0.0`, `factual_grounding=0.0`,
`single_truth_recall=0.0` are recorded. The `_aggregate_results` loop averages
these zeros into the overall means. This drags all three aggregate metrics down
for models that correctly refuse unanswerable questions — exactly the opposite of
what the evaluation should reward.

**Fix:** Implement the carve-out described in N1. As a secondary fix, at
minimum, track and report separately: (a) the number of correct refusals, and (b)
the metric averages both with and without correct refusals, so the degradation is
visible in the report.

---

## 8. Summary table

| ID  | Severity | Where                          | One-line                                                                 |
|-----|----------|--------------------------------|--------------------------------------------------------------------------|
| §1  | P0       | judge_prompts.py:66            | Behavior prompt shows full rubric dict, not the selected entry           |
| §2.1| P0       | conflict_eval.py:255-273       | Partial-match credit awarded by wrong-side confidence                    |
| §2.2| P1       | conflict_eval.py:37-45         | Empty-answer fallback fabricates committee fields                        |
| §2.3| P1       | conflict_eval.py:136-157       | "Cross-doc grounding" uses only judges[0]                                |
| §2.4| P2       | conflict_eval.py:218,283       | Self-import inside own module                                            |
| §2.6| P2       | conflict_eval.py:308           | Non-string gold answers silently dropped                                 |
| §3.1| P1       | config.py:252-355              | Judge priorities hardcoded — make modular                                |
| §3.2| P2       | config.py                      | Many advertised config fields are dead code                              |
| §3.3| P0       | config.py + judge_committee.py | max_concurrent_requests is configured but never enforced                 |
| §3.4| P1       | config.py:279, judge_committee | Per-judge RPM limits never enforced; HTTP 429s become "non-adherent"     |
| §4.1| P0       | judge_committee.py:305         | confidence parsed but never produced by prompt → weight = priority only  |
| §4.2| P0       | config.py:273                  | DeepSeek R1 max_tokens=500 likely truncates the JSON behind the trace    |
| §4.3| P1       | judge_committee.py:286         | `data["choices"][0]` blind index — fails on empty/error responses        |
| §4.4| P2       | judge_committee.py:315         | `_parse_nli_response` is dead code                                       |
| §4.5| P1       | judge_committee.py:468         | Displayed rationale biased to DeepSeek (priority×constant-confidence)    |
| §5.1| P0       | evaluator.py:139               | `conflict_category_id=0` silently re-labels to Type 1                    |
| §5.2| P0       | evaluator.py:205-228           | Out-of-range conflict types KeyError after all API spend                 |
| §5.3| P0       | data.py:100                    | `model_output` fallback to `final_grounded_answer.answer` (gold!)        |
| §5.4| P1       | data.py:84-85                  | Verdict matching is exact-string; near-synonyms become "irrelevant"      |
| §5.6| P0       | metrics.py:40-64               | NLTK splits "Lyndon B." mid-name; bracket-only sentences become claims   |
| §5.7| P1       | metrics.py:95                  | "F1_GR" is binary accuracy, not F1                                       |
| §5.13|P1       | run_evaluation.py + YAML       | `--config` is parsed but the YAML is never loaded                        |
| §6.1| P1       | summary aggregation            | Type 5 n=1 reported as headline number                                   |
| §6.2| P0       | NLI behaviour                  | Haiku NLI over-credits meta-claims like "considered real based on evidence" |
| §6.3| P1       | scoring                        | Type 3 always contributes 0.0 to recall — drags CATS Score down          |
| §6.5| P1       | judge behaviour                | Same model/answer/conflict-type gets opposite verdicts (downstream of §1)|
| §6.7| P1       | judge prompt                   | Type 4 judges never see document dates — they guess from answer wording  |

---

## 8. Suggested fix order (smallest blast radius first)

1. **§1** — one-line fix: `{BEHAVIOR_RUBRIC}` → `{rubric}` in [rag_eval/judge_prompts.py:66](CATS_v2/rag_eval/judge_prompts.py#L66). Rerun §6 evidence to confirm rationales now reference the correct rubric.
2. **§5.1, §5.2** — guard against `ctype=0` and out-of-range ctypes.
3. **§5.3** — make `get_model_output` fail loudly when the field is missing.
4. **§4.2** — raise DeepSeek `max_tokens` to ≥2000.
5. **§3.3** — add a semaphore around the per-sample fan-out.
6. **§2.1** — fix partial-match credit logic, or remove it.
7. **§3.1** — make judge priorities modular (YAML-overridable).
8. **§4.1** — either delete confidence from weight formula or actually request confidence in the prompt.
9. **§5.6** — preprocess answers to strip bracket noise before sentence-splitting.
10. **§5.13** — wire YAML config loading, or delete the flag.

After (1)–(4), rerun on the same 54-sample qwen dataset and compare the new `detailed_results.json` against the existing one to quantify the fix's impact on behavior and grounding scores.
