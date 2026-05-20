# rag_eval/conflict_eval.py
# -*- coding: utf-8 -*-
"""
Enhanced Conflict-Aware Evaluation with Multi-Judge Committee
-------------------------------------------------------------
Implements behavior, factual-grounding, and single-truth-recall metrics
for the Dragged-into-Conflicts taxonomy.

Behavior adherence uses the multi-judge committee.
Factual grounding uses a dedicated NLI judge (Claude Sonnet 4.6 by default)
rather than reusing committee.judges[0] — see get_sonnet_nli_judge().
Single-truth recall uses the committee but with a corrected partial-match
formula that uses minority-side confidence, not majority-side.

Authors: Enhanced by Claude AI
"""

import asyncio
from typing import Dict, Any, List, Optional
from .judge_prompts import behavior_judge_prompt, single_truth_recall_prompt, nli_prompt
from .judge_committee import JudgeCommittee, JudgeClient, CommitteeDecision
from .logging_config import logger


def _iter_gold_answers(gold_answers: Any) -> List[str]:
    """Normalize gold_answers to a list of stringified entries (drops None/empty only)."""
    if gold_answers is None:
        return []
    if isinstance(gold_answers, str):
        s = gold_answers.strip()
        return [s] if s else []
    if isinstance(gold_answers, (list, tuple)):
        out = []
        for g in gold_answers:
            if g is None:
                continue
            s = str(g).strip()
            if s:
                out.append(s)
        return out
    # any other type (int/float/dict) — coerce to string
    s = str(gold_answers).strip()
    return [s] if s else []


# --------------------
# Behavior Adherence (Multi-Judge)
# --------------------
async def committee_behavior_adherence(
    committee: JudgeCommittee,
    query: str,
    answer: str,
    conflict_type: int,
) -> Dict[str, Any]:
    """
    Evaluate behavior adherence using multi-judge committee.

    Returns a dict including a `skipped` flag distinguishing "empty answer
    skipped" from "committee ran and voted non-adherent". The downstream
    aggregator can use this to exclude skipped samples from the average if
    desired.
    """
    if not (answer or "").strip():
        return {
            "adherent": False,
            "rationale": "Empty answer",
            "confidence": 1.0,
            "minority_confidence": 0.0,
            "votes_for": 0,
            "votes_against": 0,
            "total_votes": 0,
            "skipped": "empty_answer",
            "committee_details": None,
        }

    prompt = behavior_judge_prompt(query, answer, conflict_type)

    try:
        decision: CommitteeDecision = await committee.judge_behavior(prompt)
        return {
            "adherent": decision.adherent,
            "rationale": decision.rationale,
            "confidence": decision.confidence,
            "minority_confidence": decision.minority_confidence,
            "votes_for": decision.votes_for,
            "votes_against": decision.votes_against,
            "total_votes": decision.total_votes,
            "skipped": None,
            "all_failed": decision.all_failed,
            "committee_details": decision.to_dict(),
        }
    except Exception as e:
        logger.error(f"Committee evaluation error: {e}")
        return {
            "adherent": False,
            "rationale": f"Committee error: {e}",
            "confidence": 0.0,
            "minority_confidence": 0.0,
            "votes_for": 0,
            "votes_against": 0,
            "total_votes": 0,
            "skipped": "committee_error",
            "committee_details": None,
        }


# --------------------
# Enhanced Factual Grounding (dedicated NLI judge)
# --------------------
async def enhanced_factual_grounding(
    nli_judge: JudgeClient,
    claims: List[str],
    support_docs: List[Dict[str, Any]],
    require_cross_doc: bool = False,
) -> Dict[str, Any]:
    """
    Factual grounding via NLI entailment, computed by a single dedicated
    NLI judge (Sonnet 4.6 by default).

    For each claim, we ask the NLI judge whether each support_doc entails it.
    A claim is "supported" if at least one doc entails it (or two, when
    require_cross_doc=True).
    """
    if not claims:
        return {
            "grounding_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "claim_details": [],
        }

    if not support_docs:
        return {
            "grounding_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": len(claims),
            "claim_details": [
                {"claim": c, "supported": False, "support_count": 0, "supporting_docs": []}
                for c in claims
            ],
        }

    claim_details = []
    supported_count = 0

    for claim in claims:
        support_count = 0
        supporting_docs: List[str] = []

        for doc in support_docs:
            passage = doc.get("snippet") or doc.get("text") or ""
            if not passage.strip():
                continue

            prompt = nli_prompt(passage, claim)
            try:
                nli_result = await nli_judge.judge_nli(prompt)
                if nli_result["relation"] == "entails":
                    support_count += 1
                    supporting_docs.append(doc.get("doc_id", "unknown"))
            except Exception as e:
                logger.warning(f"NLI error for claim '{claim[:50]}...': {e}")
                continue

        threshold = 2 if require_cross_doc else 1
        is_supported = support_count >= threshold

        if is_supported:
            supported_count += 1

        claim_details.append({
            "claim": claim,
            "supported": is_supported,
            "support_count": support_count,
            "supporting_docs": supporting_docs,
        })

    grounding_ratio = supported_count / len(claims)

    return {
        "grounding_ratio": grounding_ratio,
        "supported_claims": supported_count,
        "total_claims": len(claims),
        "claim_details": claim_details,
    }


# --------------------
# Enhanced Single-Truth Recall
# --------------------
async def enhanced_single_truth_recall(
    committee: JudgeCommittee,
    gold_answers: Any,
    answer_text: str,
    allow_paraphrases: bool = True,
) -> Dict[str, Any]:
    """
    Single-truth recall via committee voting on the single_truth_recall_prompt.

    Partial-match logic (fixed in v3): partial credit is awarded only when the
    committee is *uncertain* about its negative verdict — i.e., the minority
    side (the "yes, gold is present" side) has non-trivial weight. Previously
    we used `decision.confidence > 0.3`, but confidence is the *majority*
    side's strength; a 3-against/1-for vote with confidence=0.75 wrongly
    triggered partial credit even though the committee was confidently saying
    the gold was NOT present.
    """
    gold_iter = _iter_gold_answers(gold_answers)
    if not gold_iter:
        return {"recall": 0.0, "exact_matches": 0, "partial_matches": 0, "match_details": [], "partial_details": []}

    candidate = answer_text or ""
    if not candidate.strip():
        return {"recall": 0.0, "exact_matches": 0, "partial_matches": 0, "match_details": [], "partial_details": []}

    matches: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []

    PARTIAL_MIN_CONFIDENCE = 0.30  # minimum minority-side support to award partial credit

    for gold in gold_iter:
        prompt = single_truth_recall_prompt(gold_answer=gold, model_answer=candidate)

        try:
            decision = await committee.judge_behavior(prompt)

            if decision.adherent:
                matches.append({
                    "gold_answer": gold,
                    "confidence": decision.confidence,
                    "votes_for": decision.votes_for,
                    "votes_against": decision.votes_against,
                })
            else:
                # Partial credit only when the minority (the "yes" side) was non-trivial.
                # decision.minority_confidence reflects how much the "yes, gold present"
                # side had — high minority means the committee was genuinely split.
                if decision.minority_confidence >= PARTIAL_MIN_CONFIDENCE:
                    partial_matches.append({
                        "gold_answer": gold,
                        "minority_confidence": decision.minority_confidence,
                        "votes_for": decision.votes_for,
                        "votes_against": decision.votes_against,
                    })
        except Exception as e:
            logger.warning(f"Single-truth recall error: {e}")
            continue

    # Recall: 1.0 if any exact match; otherwise weighted by partial fraction.
    if matches:
        recall = 1.0
    elif partial_matches:
        recall = min(1.0, 0.5 * len(partial_matches) / len(gold_iter))
    else:
        recall = 0.0

    return {
        "recall": recall,
        "exact_matches": len(matches),
        "partial_matches": len(partial_matches),
        "match_details": matches,
        "partial_details": partial_matches,
    }
