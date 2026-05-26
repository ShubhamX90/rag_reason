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
import re
from typing import Dict, Any, List, Optional
from .judge_prompts import behavior_judge_prompt, single_truth_recall_prompt, nli_prompt
from .judge_committee import JudgeCommittee, JudgeClient, CommitteeDecision
from .logging_config import logger


# ---------------------------------------------------------------------------
# Meta-claim detection (Fix 3)
# ---------------------------------------------------------------------------
# Claims that reference the aggregate of docs ("confirmed across all sources",
# "unanimously confirmed", etc.) cannot be verified per-doc because no single
# doc states "all sources agree on X". Excluding them from the FG denominator
# avoids unfairly penalising the model for making true aggregate statements.

_META_CLAIM_RE = re.compile(
    r"(?:"
    # Trailing meta phrases — claim says sources agree, not a verifiable world fact
    r"confirmed\s+across\s+(?:all\s+available|all|multiple)\s+sources?"
    r"|unanimously\s+(?:confirmed|agreed|supported|established|corroborated)"
    r"|consistent(?:ly)?\s+(?:across|between|among)\s+(?:all\s+)?sources?"
    r")",
    re.IGNORECASE,
)

# Source-reference fragments: "An older high-credibility source from November 2023 (Olympics.com)"
# — no actual claim about the world, just a source label.
_SOURCE_FRAG_RE = re.compile(
    r"^[Aa]n?\s+(?:[\w-]+\s+){0,4}"
    r"(?:source|study|report|document|article|paper|page|survey|analysis)\s+from\s+",
    re.IGNORECASE,
)

# Anaphoric meta-references: "These individual confirmations are consistent with..."
# — the subject is other citations, not a real-world entity.
_ANAPHORIC_META_RE = re.compile(
    r"^(?:These|This|Those|The)\s+(?:individual\s+)?"
    r"(?:confirmations?|sources?|documents?|findings?|results?|references?)\s+",
    re.IGNORECASE,
)


def _is_meta_claim(claim: str) -> bool:
    """Return True if claim is a meta-attribution statement not verifiable against individual docs."""
    return bool(
        _META_CLAIM_RE.search(claim)
        or _SOURCE_FRAG_RE.match(claim)
        or _ANAPHORIC_META_RE.match(claim)
    )


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
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate behavior adherence using multi-judge committee.

    `retrieved_docs` is optional. When provided for conflict types 4
    (Outdated) and 5 (Misinformation), document dates / sources are surfaced
    to the behavior judges so "prioritise the up-to-date / reliable source"
    can be evaluated against evidence rather than wording alone (N6 fix).

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

    prompt = behavior_judge_prompt(query, answer, conflict_type, retrieved_docs=retrieved_docs)

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
NLI_PER_CALL_TIMEOUT_S = 60.0  # N7 fix: cap a single NLI call so a hung Sonnet
                                # request can't block sample completion.


async def enhanced_factual_grounding(
    nli_judge: JudgeClient,
    claims: List[str],
    support_docs: List[Dict[str, Any]],
    require_cross_doc: bool = False,
    min_entail_confidence: float = 0.5,
    majority_support_rule: bool = False,
    conflict_type: int = 0,
) -> Dict[str, Any]:
    """
    Factual grounding via NLI entailment, computed by a single dedicated
    NLI judge (Sonnet 4.6 by default).

    For each claim, we ask the NLI judge whether each support_doc entails it.
    A claim is "supported" if at least one doc entails it (or two, when
    require_cross_doc=True).

    Meta-attribution claims (e.g. "unanimously confirmed across all sources")
    are excluded from the FG denominator — they cannot be verified per-doc.

    conflict_type=3 (Conflicting Opinions) uses a relaxed rule: contradictions
    from opposing docs are expected and do not block a claim that has at least
    one entailment.
    """
    if not claims:
        return {
            "grounding_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": 0,
            "evaluable_claims": 0,
            "meta_excluded_claims": 0,
            "claim_details": [],
            "nli_errors": 0,
        }

    if not support_docs:
        # N9 fix: keep the schema invariant len(claim_details) == total_claims.
        return {
            "grounding_ratio": 0.0,
            "supported_claims": 0,
            "total_claims": len(claims),
            "evaluable_claims": len(claims),
            "meta_excluded_claims": 0,
            "claim_details": [
                {
                    "claim": c,
                    "supported": False,
                    "entails_count": 0,
                    "contradicts_count": 0,
                    "support_count": 0,
                    "supporting_docs": [],
                    "contradicting_docs": [],
                }
                for c in claims
            ],
            "nli_errors": 0,
        }

    claim_details = []
    supported_count = 0
    meta_excluded_count = 0
    total_errors = 0

    # NLI confidence floor — passed in from config (default 0.5, sweep via YAML).
    MIN_ENTAIL_CONFIDENCE = min_entail_confidence

    for claim in claims:
        # Fix 3: Skip meta-attribution claims that reference the aggregate of
        # docs. They cannot be entailed by any single doc and unfairly penalise
        # the model for accurate aggregate statements.
        if _is_meta_claim(claim):
            meta_excluded_count += 1
            claim_details.append({
                "claim": claim,
                "supported": False,
                "entails_count": 0,
                "contradicts_count": 0,
                "support_count": 0,
                "supporting_docs": [],
                "contradicting_docs": [],
                "evaluation_method": "meta_excluded",
            })
            continue

        entails_count = 0
        contradicts_count = 0
        supporting_docs: List[str] = []
        contradicting_docs: List[str] = []

        for doc in support_docs:
            # NLI-I fix (v3.1): combine the annotator-verified verbatim quote
            # with the full snippet when both are present.
            #   • Quote alone is too narrow (~60 words) — generic model claims
            #     often fail to strictly entail from such a tight span.
            #   • Snippet alone misses the annotator's focus signal.
            # Concatenating gives NLI both the focused span and the surrounding
            # context. The quote is labelled so the judge can weight it.
            quote = (doc.get("quote") or "").strip()
            snippet = (doc.get("snippet") or doc.get("text") or "").strip()
            if quote and snippet and quote not in snippet:
                passage = f"Key evidence (annotator-verified): {quote}\n\nFull passage: {snippet}"
            elif quote and snippet:
                # Quote is a substring of snippet — snippet already contains it.
                passage = snippet
            elif quote:
                passage = quote
            else:
                passage = snippet
            if not passage:
                continue

            prompt = nli_prompt(passage, claim)
            try:
                nli_result = await asyncio.wait_for(
                    nli_judge.judge_nli(prompt),
                    timeout=NLI_PER_CALL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"NLI judge timeout for claim '{claim[:50]}...' on doc "
                    f"{doc.get('doc_id', '?')}; treating as neutral."
                )
                total_errors += 1
                continue
            except Exception as e:
                logger.warning(f"NLI error for claim '{claim[:50]}...': {e}")
                total_errors += 1
                continue

            relation = nli_result.get("relation", "neutral")
            conf = float(nli_result.get("confidence", 0.0) or 0.0)

            if relation == "entails" and conf >= MIN_ENTAIL_CONFIDENCE:
                entails_count += 1
                supporting_docs.append(doc.get("doc_id", "unknown"))
            elif relation == "contradicts" and conf >= MIN_ENTAIL_CONFIDENCE:
                # NLI-A fix: a confident contradiction by a support doc is a
                # strong negative signal — track it and use it to gate
                # `supported`.
                contradicts_count += 1
                contradicting_docs.append(doc.get("doc_id", "unknown"))

        threshold = 2 if require_cross_doc else 1
        if conflict_type == 3:
            # Fix FG-CT3: For conflicting-opinions queries, docs on the opposing
            # side will always contradict claims about the other side — that is
            # expected, not evidence of hallucination. Only require that at least
            # one doc entails the claim; ignore contradictions entirely.
            is_supported = entails_count >= threshold
        elif majority_support_rule:
            # Majority rule + tie-breaking: claim passes when entails >= contradicts
            # (ties go to the claim — on gold data a tie means genuine ambiguity,
            # not hallucination). Previously used strict e > c which lost tie cases.
            is_supported = (entails_count >= threshold) and (entails_count >= contradicts_count)
        else:
            # Strict rule (default): any contradiction blocks the claim.
            is_supported = (entails_count >= threshold) and (contradicts_count == 0)

        if is_supported:
            supported_count += 1

        claim_details.append({
            "claim": claim,
            "supported": is_supported,
            "entails_count": entails_count,
            "contradicts_count": contradicts_count,
            "support_count": entails_count,        # backward-compat alias
            "supporting_docs": supporting_docs,
            "contradicting_docs": contradicting_docs,
        })

    evaluable_claims = len(claims) - meta_excluded_count
    grounding_ratio = supported_count / evaluable_claims if evaluable_claims > 0 else 0.0

    return {
        "grounding_ratio": grounding_ratio,
        "supported_claims": supported_count,
        "total_claims": len(claims),
        "evaluable_claims": evaluable_claims,
        "meta_excluded_claims": meta_excluded_count,
        "claim_details": claim_details,
        "nli_errors": total_errors,
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
