"""
src/voting.py
=============
Weighted majority voting for the multi-LLM annotation committee (v3).

Default models are accessed via OpenRouter.  Stage runners can also switch the
active committee to a local OpenAI-compatible backend by loading a local judge
config and updating MODEL_WEIGHTS in-place before any voting starts.

Committee
---------
  anthropic/claude-sonnet-4.6      weight 0.35  — primary annotator; best JSON fidelity + nuanced
                                                  verdict and conflict reasoning
  openai/gpt-5.4                   weight 0.30  — strong instruction following; diverse GPT signal
  deepseek/deepseek-v3.2           weight 0.20  — DeepSeek V3.2 (much cheaper than R1; R1’s
                                                  extended CoT is unnecessary for JSON annotation)
  mistralai/mistral-small-2603    weight 0.15  — low-cost fourth seat; diverse non-xAI signal

Weights sum to 1.0.  To change the committee or rebalance, only edit MODEL_WEIGHTS here —
all multi_async scripts read from this single source of truth.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils import source_quality_from_url


# ─── Committee definition ─────────────────────────────────────────────────────

_DEFAULT_MODEL_WEIGHTS: Dict[str, float] = {
    "anthropic/claude-haiku-4.5":   0.35,
    "openai/gpt-5.4":               0.30,
    "deepseek/deepseek-v3.2":       0.20,
    "mistralai/mistral-small-2603": 0.15,
}

def _load_model_weights() -> Dict[str, float]:
    raw = os.environ.get("V3_COMMITTEE_WEIGHTS_JSON", "").strip()
    if not raw:
        return dict(_DEFAULT_MODEL_WEIGHTS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid V3_COMMITTEE_WEIGHTS_JSON: expected JSON object of "
            "{model_slug: weight}. "
            f"Parse error: {exc}"
        ) from exc
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("V3_COMMITTEE_WEIGHTS_JSON must be a non-empty JSON object")
    weights: Dict[str, float] = {}
    for model, weight in parsed.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Committee override contains an invalid model key")
        try:
            w = float(weight)
        except Exception as exc:
            raise ValueError(f"Invalid weight for model {model!r}: {weight!r}") from exc
        if w <= 0:
            raise ValueError(f"Weight must be positive for model {model!r}, got {w}")
        weights[model.strip()] = w
    return weights


MODEL_WEIGHTS: Dict[str, float] = _load_model_weights()

COMMITTEE_MODELS: List[str] = list(MODEL_WEIGHTS.keys())

# Sanity-check
_weight_sum = round(sum(MODEL_WEIGHTS.values()), 9)
assert _weight_sum == 1.0, f"MODEL_WEIGHTS must sum to 1.0, got {_weight_sum}"


def normalize_priorities(priorities: Dict[str, float]) -> Dict[str, float]:
    """Convert positive priority/weight values to sum-normalized weights."""
    if not priorities:
        raise ValueError("Committee priorities must not be empty")
    cleaned: Dict[str, float] = {}
    for model, value in priorities.items():
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Committee priority contains an invalid model key")
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"Committee priority must be positive for {model!r}")
        cleaned[model.strip()] = weight
    total = sum(cleaned.values())
    return {model: weight / total for model, weight in cleaned.items()}


def set_model_weights(weights: Dict[str, float], *, normalize: bool = False) -> None:
    """Replace the active committee weights without breaking imported globals.

    The multi-async scripts import MODEL_WEIGHTS and COMMITTEE_MODELS directly,
    so this mutates the existing dict/list objects in-place.
    """
    new_weights = normalize_priorities(weights) if normalize else dict(weights)
    weight_sum = round(sum(float(w) for w in new_weights.values()), 9)
    if weight_sum != 1.0:
        raise ValueError(f"MODEL_WEIGHTS must sum to 1.0, got {weight_sum}")

    MODEL_WEIGHTS.clear()
    MODEL_WEIGHTS.update({model: float(weight) for model, weight in new_weights.items()})

    COMMITTEE_MODELS.clear()
    COMMITTEE_MODELS.extend(MODEL_WEIGHTS.keys())


# ─── Core vote logic ──────────────────────────────────────────────────────────

def weighted_majority_vote(
    votes: List[Tuple[str, Any, float]],
) -> Tuple[Any, Dict[Any, float]]:
    """
    Compute a weighted majority vote.

    Parameters
    ----------
    votes : list of (model_slug, voted_value, weight)
        Models that errored can be omitted from this list.

    Returns
    -------
    (winning_value, tally_dict)
        tally_dict maps each unique candidate value to its summed weight.
        winning_value is None if votes is empty.

    Tiebreak strategy: highest cumulative weight wins.
    When two values tie exactly, the lexicographically smaller str(value) wins
    (deterministic, stable across runs).
    """
    tally: Dict[Any, float] = {}
    for _model, value, weight in votes:
        tally[value] = tally.get(value, 0.0) + weight
    if not tally:
        return None, {}
    winning = max(tally, key=lambda v: (round(tally[v], 9), -len(str(v)), str(v)))
    return winning, tally


def select_winner_model(
    votes: List[Tuple[str, Any, float]],
    winning_value: Any,
) -> str:
    """
    Among models that voted for `winning_value`, return the one with the
    highest weight.  That model's associated text fields are adopted into
    the merged record (verdict_reason, key_fact, quote, conflict_reason, etc.).

    Falls back to the first entry in `votes` if no model matches
    (shouldn't happen in normal operation).
    """
    candidates = [
        (model, weight)
        for model, value, weight in votes
        if value == winning_value
    ]
    if not candidates:
        return votes[0][0] if votes else ""
    return max(candidates, key=lambda x: x[1])[0]


def _build_votes(
    model_records: Dict[str, Optional[Dict[str, Any]]],
    field: str,
    fallback: Any,
) -> List[Tuple[str, Any, float]]:
    """
    Build vote tuples from a dict of {model: record}.
    Records that are None (model errored out entirely) are excluded.
    """
    return [
        (model, (rec or {}).get(field, fallback), MODEL_WEIGHTS.get(model, 0.0))
        for model, rec in model_records.items()
        if rec is not None
    ]


# ─── Stage 1 merge ────────────────────────────────────────────────────────────

def merge_stage1_votes(
    model_notes: Dict[str, Optional[Dict[str, Any]]],
    fallback_doc_id: str = "",
    fallback_source_url: str = "",
) -> Dict[str, Any]:
    """
    Merge per-model Stage-1 per-doc notes into one consensus note.

    Votes on         : verdict
    Adopts from winner: key_fact, quote, verdict_reason, source_quality
    (i.e. all text fields come from the highest-weight model that voted
    for the winning verdict — not an average or blend)

    Parameters
    ----------
    model_notes : dict of {model_slug: parsed_stage1_note_or_None}
    fallback_doc_id : doc_id to use if winner's record has none

    Returns
    -------
    Consensus note dict with extra audit fields:
        _vote_tally     : {verdict: summed_weight, ...}
        _winner_model   : slug of the model whose text fields were adopted
        _all_verdicts   : {model: verdict} for every committee member
    """
    votes = _build_votes(model_notes, "verdict", "irrelevant")
    winning_verdict, tally = weighted_majority_vote(votes)
    if winning_verdict is None:
        return {
            "doc_id": fallback_doc_id,
            "verdict": "irrelevant",
            "key_fact": "",
            "quote": "",
            "verdict_reason": "No committee responses available.",
            "source_quality": "low",
            "_vote_tally": {},
            "_winner_model": "",
            "_all_verdicts": {m: None for m in COMMITTEE_MODELS},
            "_all_models_failed": True,
        }
    winning_model = select_winner_model(votes, winning_verdict)

    base: Dict[str, Any] = (model_notes.get(winning_model) or {}).copy()
    # Normalize to the benchmark schema even if the winning parse was partially malformed.
    key_fact = str(base.get("key_fact") or "").strip()
    quote = str(base.get("quote") or "").strip()
    if winning_verdict != "irrelevant":
        if not key_fact and quote:
            key_fact = quote
        if not quote and key_fact:
            quote = key_fact
    else:
        key_fact = ""
        quote = ""
    verdict_reason = str(base.get("verdict_reason") or "").strip()
    if not verdict_reason:
        if winning_verdict == "irrelevant":
            verdict_reason = "The snippet does not directly answer the query."
        elif key_fact:
            verdict_reason = f"Relevant evidence: {key_fact[:160]}"
        else:
            verdict_reason = "The snippet contains relevant but incomplete evidence."
    source_quality = str(base.get("source_quality") or "").strip().lower()
    if source_quality not in {"high", "low"}:
        source_quality = source_quality_from_url(fallback_source_url)

    base["doc_id"] = fallback_doc_id
    base["verdict"]         = winning_verdict
    base["key_fact"]        = key_fact
    base["quote"]           = quote
    base["verdict_reason"]  = verdict_reason
    base["source_quality"]  = source_quality
    base["_vote_tally"]     = {str(k): round(v, 4) for k, v in tally.items()}
    base["_winner_model"]   = winning_model
    base["_all_verdicts"]   = {
        m: (model_notes.get(m) or {}).get("verdict")
        for m in COMMITTEE_MODELS
    }
    return base


# ─── Stage 2 merge ────────────────────────────────────────────────────────────

def merge_stage2_votes(
    model_records: Dict[str, Optional[Dict[str, Any]]],
    is_refusal: bool,
    vote_conflict_type: bool = False,
) -> Dict[str, Any]:
    """
    Merge per-model Stage-2 outputs.

    CONFLICTS dataset  (is_refusal=False):
        Votes on         : answerable_under_evidence (bool)
        conflict_type    : taken verbatim from input record (gold human label — NOT voted on)
        conflict_reason  : adopted from the answerable-vote winner

    REFUSALS dataset  (is_refusal=True):
        Votes on         : conflict_type  AND  answerable_under_evidence independently
        conflict_reason  : adopted from the conflict_type-vote winner
        (because that model produced the most authoritative explanation of the
        conflict it independently classified)

    Parameters
    ----------
    model_records : dict of {model_slug: parsed_stage2_output_or_None}
        For conflicts, model output contains {conflict_reason, answerable_under_evidence}.
        For refusals, model output contains {conflict_type, conflict_reason, answerable_under_evidence}.

    Returns
    -------
    Merged dict with keys set ready to copy back into the record:
        answerable_under_evidence, conflict_reason,
        and (if is_refusal) conflict_type,
        plus audit fields: _ans_vote_tally, _ans_winner_model,
                           _ct_vote_tally, _ct_winner_model (refusals only)
    """
    # ── 1. answerable_under_evidence vote ────────────────────────────────────
    ans_votes = _build_votes(model_records, "answerable_under_evidence", False)
    winning_ans, ans_tally = weighted_majority_vote(ans_votes)
    if winning_ans is None:
        base: Dict[str, Any] = {
            "conflict_reason": "No committee responses available.",
            "answerable_under_evidence": False,
            "_ans_vote_tally": {},
            "_ans_winner_model": "",
            "_all_models_failed": True,
        }
        if vote_conflict_type:
            base["conflict_type"] = ""
            base["_ct_vote_tally"] = {}
            base["_ct_winner_model"] = ""
        return base
    ans_winner = select_winner_model(ans_votes, winning_ans)

    # Start with the ans-winner's full record as base
    base: Dict[str, Any] = (model_records.get(ans_winner) or {}).copy()
    base["answerable_under_evidence"] = winning_ans
    base["_ans_vote_tally"]           = {str(k): round(v, 4) for k, v in ans_tally.items()}
    base["_ans_winner_model"]         = ans_winner

    if not vote_conflict_type:
        # Conflicts: conflict_reason from the answerable-winner is fine
        # conflict_type is NOT touched here — it remains the gold label in the record
        return base

    # ── 2. conflict_type vote (refusals only) ──────────────────────────────
    ct_votes = _build_votes(model_records, "conflict_type", "")
    winning_ct, ct_tally = weighted_majority_vote(ct_votes)
    ct_winner = select_winner_model(ct_votes, winning_ct)

    base["conflict_type"]    = winning_ct
    base["_ct_vote_tally"]   = {str(k): round(v, 4) for k, v in ct_tally.items()}
    base["_ct_winner_model"] = ct_winner

    # Override conflict_reason: take from ct_winner (most authoritative for this conflict type)
    ct_winner_rec = model_records.get(ct_winner) or {}
    base["conflict_reason"] = ct_winner_rec.get(
        "conflict_reason", base.get("conflict_reason", "")
    )

    return base


# ─── Stage 3 merge ────────────────────────────────────────────────────────────

def merge_stage3_votes(
    model_records: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Merge per-model Stage-3 outputs.

    Votes on         : expected_response.abstain (bool)
    Adopts from winner: expected_response.answer, .evidence, .abstain_reason,
                        and the think trace

    All fields of the winning model's record are adopted wholesale —
    the answer text is never averaged or blended across models.

    Parameters
    ----------
    model_records : dict of {model_slug: parsed_stage3_output_or_None}
        Each value is a dict with keys: expected_response (dict), think (str)

    Returns
    -------
    Merged dict with:
        expected_response  : from the abstain-vote winner
        think              : from the abstain-vote winner
        _abstain_vote_tally, _abstain_winner_model
    """
    # Pull abstain out of nested expected_response for voting
    flat_abstain: Dict[str, Optional[Dict[str, Any]]] = {
        model: {
            "abstain": ((rec or {}).get("expected_response") or {}).get("abstain", False)
        }
        for model, rec in model_records.items()
        if rec is not None
    }

    abstain_votes = _build_votes(flat_abstain, "abstain", False)
    winning_abstain, abstain_tally = weighted_majority_vote(abstain_votes)
    if winning_abstain is None:
        return {
            "expected_response": {
                "answer": "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                "evidence": [],
                "abstain": True,
                "abstain_reason": "No committee responses available.",
            },
            "think": "",
            "_abstain_vote_tally": {},
            "_abstain_winner_model": "",
            "_all_models_failed": True,
        }
    abstain_winner = select_winner_model(abstain_votes, winning_abstain)

    base: Dict[str, Any] = (model_records.get(abstain_winner) or {}).copy()

    # Enforce the voted abstain value (in case the winner's own inner value differs)
    er = base.setdefault("expected_response", {})
    er["abstain"] = winning_abstain

    base["_abstain_vote_tally"]   = {str(k): round(v, 4) for k, v in abstain_tally.items()}
    base["_abstain_winner_model"] = abstain_winner
    return base
