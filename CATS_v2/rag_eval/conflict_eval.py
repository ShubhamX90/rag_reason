# rag_eval/conflict_eval.py
# -*- coding: utf-8 -*-
"""
Enhanced Conflict-Aware Evaluation with Multi-Judge Committee
-------------------------------------------------------------
Implements advanced evaluation metrics for the Dragged-into-Conflicts taxonomy.

New Features:
  • Multi-judge committee voting for behavior adherence
  • Cross-document verification for factual grounding
  • Semantic matching for single-truth recall
  • Conflict resolution strategy assessment

Authors: Enhanced by Claude AI
"""

import asyncio
from typing import Dict, Any, List, Optional
from .judge_prompts import behavior_judge_prompt, single_truth_recall_prompt, nli_prompt
from .judge_committee import JudgeCommittee, CommitteeDecision
from .logging_config import logger


# --------------------
# Behavior Adherence (Multi-Judge)
# --------------------
async def committee_behavior_adherence(
    committee: JudgeCommittee,
    query: str,
    answer: str,
    conflict_type: int
) -> Dict[str, Any]:
    """
    Evaluate behavior adherence using multi-judge committee.
    Returns aggregated decision with voting breakdown.
    """
    if not (answer or "").strip():
        return {
            "adherent": False,
            "rationale": "Empty answer",
            "confidence": 1.0,
            "votes_for": 0,
            "votes_against": 1,
            "committee_details": None
        }
    
    prompt = behavior_judge_prompt(query, answer, conflict_type)
    
    try:
        decision: CommitteeDecision = await committee.judge_behavior(prompt)
        
        return {
            "adherent": decision.adherent,
            "rationale": decision.rationale,
            "confidence": decision.confidence,
            "votes_for": decision.votes_for,
            "votes_against": decision.votes_against,
            "total_votes": decision.total_votes,
            "committee_details": decision.to_dict()
        }
    except Exception as e:
        logger.error(f"Committee evaluation error: {e}")
        return {
            "adherent": False,
            "rationale": f"Committee error: {e}",
            "confidence": 0.0,
            "votes_for": 0,
            "votes_against": 1,
            "committee_details": None
        }


# Backward compatibility: single judge version
def behavior_adherence(llm: Any, query: str, answer: str, conflict_type: int) -> Dict[str, Any]:
    """Single judge behavior adherence (backward compatible)."""
    if not (answer or "").strip():
        return {"adherent": False, "rationale": "Empty answer"}
    
    prompt = behavior_judge_prompt(query, answer, conflict_type)
    try:
        res = llm.judge_behavior(prompt)
        return {
            "adherent": bool(res.get("adherent")),
            "rationale": (res.get("rationale") or "").strip(),
        }
    except Exception as e:
        return {"adherent": False, "rationale": f"judge error: {e}"}


# --------------------
# Enhanced Factual Grounding
# --------------------
async def enhanced_factual_grounding(
    committee: JudgeCommittee,
    claims: List[str],
    support_docs: List[Dict[str, Any]],
    require_cross_doc: bool = False
) -> Dict[str, Any]:
    """
    Enhanced factual grounding with cross-document verification.
    
    Args:
        committee: Judge committee for NLI decisions
        claims: List of claims to verify
        support_docs: Supporting documents
        require_cross_doc: If True, require multiple docs to support each claim
    
    Returns:
        Dict with grounding ratio and detailed breakdown
    """
    if not claims:
        return {
            "grounding_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "claim_details": []
        }
    
    if not support_docs:
        return {
            "grounding_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": len(claims),
            "claim_details": []
        }
    
    claim_details = []
    supported_count = 0
    
    for claim in claims:
        # Check each document
        support_count = 0
        supporting_docs = []
        
        # Optimize: Use single judge for NLI instead of full committee (faster)
        if hasattr(committee, 'judges') and len(committee.judges) > 0:
            first_judge = committee.judges[0]
            
            for doc in support_docs:
                passage = doc.get("snippet") or doc.get("text") or ""
                if not passage.strip():
                    continue
                    
                prompt = nli_prompt(passage, claim)
                
                try:
                    # Use the dedicated NLI method
                    nli_result = await first_judge.judge_nli(prompt)
                    
                    # Check if relation is "entails"
                    if nli_result["relation"] == "entails":
                        support_count += 1
                        supporting_docs.append(doc.get("doc_id", "unknown"))
                except Exception as e:
                    logger.warning(f"NLI error for claim '{claim[:50]}...': {e}")
                    continue
        
        is_supported = support_count > 0
        if require_cross_doc:
            is_supported = support_count >= 2
        
        if is_supported:
            supported_count += 1
        
        claim_details.append({
            "claim": claim,
            "supported": is_supported,
            "support_count": support_count,
            "supporting_docs": supporting_docs
        })
    
    grounding_ratio = supported_count / len(claims)
    
    return {
        "grounding_ratio": grounding_ratio,
        "supported_claims": supported_count,
        "total_claims": len(claims),
        "claim_details": claim_details
    }


# Backward compatible version
def factual_grounding_ratio(
    nli_fn: Any,
    claims: List[str],
    support_docs: List[Dict[str, Any]],
) -> float:
    """Original factual grounding (backward compatible)."""
    if not claims:
        return 0.0
    
    supported = 0
    for claim in claims:
        for d in support_docs:
            passage = d.get("snippet") or d.get("text") or ""
            rel = nli_fn(passage, claim)
            if rel == "entails":
                supported += 1
                break
    
    return supported / len(claims)


# --------------------
# Enhanced Single-Truth Recall
# --------------------
async def enhanced_single_truth_recall(
    committee: JudgeCommittee,
    gold_answers: Any,
    answer_text: str,
    allow_paraphrases: bool = True
) -> Dict[str, Any]:
    """
    Enhanced single-truth recall with semantic matching.
    
    Returns detailed breakdown including partial matches.
    """
    from .conflict_eval import _iter_gold_answers
    
    gold_iter = list(_iter_gold_answers(gold_answers))
    if not gold_iter:
        return {
            "recall": 0.0,
            "matches": [],
            "partial_matches": []
        }
    
    candidate = answer_text
    if not candidate:
        return {
            "recall": 0.0,
            "matches": [],
            "partial_matches": []
        }
    
    matches = []
    partial_matches = []
    
    for gold in gold_iter:
        if not gold:
            continue
        
        prompt = single_truth_recall_prompt(gold_answer=gold, model_answer=candidate)
        
        try:
            decision = await committee.judge_behavior(prompt)
            
            if decision.adherent:
                matches.append({
                    "gold_answer": gold,
                    "confidence": decision.confidence,
                    "votes_for": decision.votes_for,
                    "votes_against": decision.votes_against
                })
            elif decision.confidence > 0.3:  # Partial match threshold
                partial_matches.append({
                    "gold_answer": gold,
                    "confidence": decision.confidence
                })
        except Exception as e:
            logger.warning(f"Single-truth recall error: {e}")
            continue
    
    recall = 1.0 if matches else 0.0
    partial_credit = len(partial_matches) * 0.5 / len(gold_iter) if partial_matches else 0.0
    
    return {
        "recall": min(1.0, recall + partial_credit),
        "exact_matches": len(matches),
        "partial_matches": len(partial_matches),
        "match_details": matches,
        "partial_details": partial_matches
    }


# Backward compatible version
def single_truth_answer_recall(
    llm: Any,
    gold_answers: Any,
    answer_text: str,
) -> float:
    """Original single-truth recall (backward compatible)."""
    from .conflict_eval import _iter_gold_answers
    
    gold_iter = list(_iter_gold_answers(gold_answers))
    if not gold_iter:
        return 0.0
    
    candidate = answer_text
    if not candidate:
        return 0.0
    
    for gold in gold_iter:
        if not gold:
            continue
        prompt = single_truth_recall_prompt(gold_answer=gold, model_answer=candidate)
        try:
            judge_res: Dict[str, Any] = llm.judge_behavior(prompt)
            contains = bool(judge_res.get("adherent"))
            if contains:
                return 1.0
        except Exception:
            continue
    
    return 0.0


def _iter_gold_answers(gold_answers: Any) -> List[str]:
    """Normalize gold_answers to list of strings."""
    if gold_answers is None:
        return []
    if isinstance(gold_answers, str):
        return [gold_answers]
    if isinstance(gold_answers, (list, tuple)):
        return [g for g in gold_answers if isinstance(g, str)]
    return []
