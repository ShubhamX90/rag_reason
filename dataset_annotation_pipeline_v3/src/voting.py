"""
src/voting.py
=============
Weighted majority voting for the multi-LLM annotation committee (v3).

All models are accessed via OpenRouter — no direct-provider keys used.

Committee
---------
  anthropic/claude-sonnet-4.6      weight 0.30  — primary annotator; best JSON fidelity + nuanced
                                                  verdict and conflict reasoning
  openai/gpt-5.4                   weight 0.25  — strong instruction following; diverse GPT signal
  qwen/qwen3.5-27b                 weight 0.20  — 27B dense; Qwen's own docs state performance
                                                  "comparable to 122B-A10B" at ~40% lower cost
                                                  ($0.195/$1.56/M vs $0.26/$2.08/M output)
  deepseek/deepseek-v3.2           weight 0.15  — DeepSeek V3.2 (much cheaper than R1; R1’s
                                                  extended CoT is unnecessary for JSON annotation)
  x-ai/grok-4.1-fast              weight 0.10  — fast Grok 4.1; diverse signal / tiebreaker

Weights sum to 1.0.  To change the committee or rebalance, only edit MODEL_WEIGHTS here —
all multi_async scripts read from this single source of truth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ─── Committee definition ─────────────────────────────────────────────────────

MODEL_WEIGHTS: Dict[str, float] = {
    "anthropic/claude-sonnet-4.6":  0.30,
    "openai/gpt-5.4":               0.25,
    "qwen/qwen3.5-27b":             0.20,   # comparable to 122B per Qwen docs; ~40% cheaper
    "deepseek/deepseek-v3.2":       0.15,
    "x-ai/grok-4.1-fast":           0.10,
}

COMMITTEE_MODELS: List[str] = list(MODEL_WEIGHTS.keys())

# Sanity-check
_weight_sum = round(sum(MODEL_WEIGHTS.values()), 9)
assert _weight_sum == 1.0, f"MODEL_WEIGHTS must sum to 1.0, got {_weight_sum}"


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
    winning_model = select_winner_model(votes, winning_verdict)

    base: Dict[str, Any] = (model_notes.get(winning_model) or {}).copy()
    base["verdict"]         = winning_verdict
    base["_vote_tally"]     = {str(k): round(v, 4) for k, v in tally.items()}
    base["_winner_model"]   = winning_model
    base["_all_verdicts"]   = {
        m: (model_notes.get(m) or {}).get("verdict")
        for m in COMMITTEE_MODELS
    }
    if not base.get("doc_id"):
        base["doc_id"] = fallback_doc_id
    return base


# ─── Stage 2 merge ────────────────────────────────────────────────────────────

def merge_stage2_votes(
    model_records: Dict[str, Optional[Dict[str, Any]]],
    is_refusal: bool,
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
    ans_winner = select_winner_model(ans_votes, winning_ans)

    # Start with the ans-winner's full record as base
    base: Dict[str, Any] = (model_records.get(ans_winner) or {}).copy()
    base["answerable_under_evidence"] = winning_ans
    base["_ans_vote_tally"]           = {str(k): round(v, 4) for k, v in ans_tally.items()}
    base["_ans_winner_model"]         = ans_winner

    if not is_refusal:
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
    abstain_winner = select_winner_model(abstain_votes, winning_abstain)

    base: Dict[str, Any] = (model_records.get(abstain_winner) or {}).copy()

    # Enforce the voted abstain value (in case the winner's own inner value differs)
    er = base.setdefault("expected_response", {})
    er["abstain"] = winning_abstain

    base["_abstain_vote_tally"]   = {str(k): round(v, 4) for k, v in abstain_tally.items()}
    base["_abstain_winner_model"] = abstain_winner
    return base
