# # conflict_eval.py
# # -*- coding: utf-8 -*-
# """
# Conflict-Aware Evaluation
# -------------------------

# Implements evaluation metrics aligned to the
# Dragged-into-Conflicts taxonomy (Cattan et al., 2025).

# Metrics:
#   • Behavior Adherence  – checks if answer matches expected human behavior
#                           for conflict type (via LLM-as-a-judge).
#   • Factual Grounding   – fraction of claims in answer entailed by supporting docs.
#   • Single-Truth Recall – recall of gold factual answer in single-truth categories.

# Dependencies:
#   - judge_prompts.py   (behavior_judge_prompt, nli_prompt)
#   - llm.py             (LLM.judge_behavior, LLM.nli_entailment)
#   - metrics.py         (extract_claims_by_sentence)
#   - data.py            (support_doc_ids_from_notes)

# Authors: Gorang Mehrishi, Samyek Jain
# Institution: Birla Institute of Technology and Science, Pilani
# """

# from typing import Any, Dict, List

# from .judge_prompts import behavior_judge_prompt


# # ---------------------------------------------------------------------
# # Behavior adherence
# # ---------------------------------------------------------------------

# def behavior_adherence(llm: Any, query: str, answer: str, conflict_type: int) -> Dict[str, Any]:
#     """
#     Evaluate whether model behavior matches expected guidelines
#     for the given conflict type.

#     Args:
#         llm: LLM wrapper with judge_behavior() method
#         query: user query string
#         answer: model answer string
#         conflict_type: int in {1..5}

#     Returns:
#         dict with:
#           {
#             "adherent": bool,
#             "rationale": str (short explanation from judge or error fallback)
#           }
#     """
#     if not (answer or "").strip():
#         return {"adherent": False, "rationale": "Empty answer"}

#     prompt = behavior_judge_prompt(query, answer, conflict_type)
#     try:
#         res = llm.judge_behavior(prompt)
#         return {
#             "adherent": bool(res.get("adherent")),
#             "rationale": (res.get("rationale") or "").strip(),
#         }
#     except Exception as e:
#         return {"adherent": False, "rationale": f"judge error: {e}"}


# # ---------------------------------------------------------------------
# # Factual grounding ratio
# # ---------------------------------------------------------------------

# def factual_grounding_ratio(
#     nli_fn: Any,
#     claims: List[str],
#     support_docs: List[Dict[str, Any]],
# ) -> float:
#     """
#     Fraction of answer claims entailed by at least one supporting doc.

#     Args:
#         nli_fn: function (premise, hypothesis) -> {"entails","contradicts","neutral"}
#         claims: list of claim sentences from model answer
#         support_docs: list of doc dicts (with 'text' or 'snippet')

#     Returns:
#         float ∈ [0,1], 0.0 if no claims
#     """
#     if not claims:
#         return 0.0

#     supported = 0
#     for claim in claims:
#         for d in support_docs:
#             passage = d.get("snippet") or d.get("text") or ""
#             rel = nli_fn(passage, claim)
#             if rel == "entails":
#                 supported += 1
#                 break

#     return supported / len(claims)


# # ---------------------------------------------------------------------
# # Single-truth recall
# # ---------------------------------------------------------------------

# # def single_truth_answer_recall(
# #     gold_answers: List[str],
# #     answer_text: str,
# # ) -> float:
# #     """
# #     Recall of gold factual answers in model output.

# #     Args:
# #         gold_answers: list of gold answer strings (from dataset)
# #         answer_text: model answer string

# #     Returns:
# #         float ∈ {0.0, 1.0}
# #     """
# #     if not gold_answers:
# #         return 0.0

# #     output = (answer_text or "").lower()
# #     for gold in gold_answers:
# #         if not gold:
# #             continue
# #         if gold.lower() in output:
# #             return 1.0
# #     return 0.0

# from typing import Any, Dict, List
# import re

# from .judge_prompts import single_truth_recall_prompt


# def _strip_think_traces(text: str) -> str:
#     """
#     Remove <think>...</think> blocks from a model answer so that only the
#     final, user-facing answer is considered.
#     """
#     if not text:
#         return ""
#     # Remove any <think> ... </think> blocks (multi-line)
#     cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
#     return cleaned.strip()


# def single_truth_answer_recall(
#     llm: Any,
#     gold_answers: str,
#     answer_text: str,
# ) -> float:
#     """
#     Recall of gold factual answers in model output, using LLM-as-a-judge
#     instead of naive string matching.

#     - Strips <think>...</think> traces from the candidate answer.
#     - For each gold answer, asks an LLM judge whether that factual answer
#       is present (possibly paraphrased) in the candidate response.
#     - Returns 1.0 if ANY gold answer is judged as present, else 0.0.

#     Args:
#         llm: LLM wrapper implementing `judge_behavior(prompt: str) -> dict`
#              (same interface we use for behavior adherence).
#         gold_answers: list of gold answer strings (from dataset).
#         answer_text: raw model answer string (may include <think>...</think>).

#     Returns:
#         float ∈ {0.0, 1.0}
#     """
#     # No gold → undefined recall → treat as 0.0
#     if not gold_answers:
#         return 0.0

#     candidate = answer_text

#     # If candidate is empty after stripping think traces, recall is 0.
#     if not candidate:
#         return 0.0

#     for gold in gold_answers:
#         if not gold:
#             continue

#         prompt = single_truth_recall_prompt(gold_answer=gold, model_answer=candidate)

#         try:
#             # We reuse the judge-style interface: returns {"adherent": bool, "rationale": str}
#             judge_res: Dict[str, Any] = llm.judge_behavior(prompt)
#             contains = bool(judge_res.get("adherent"))
#             if contains:
#                 return 1.0
#         except Exception as e:
#             # Fail-safe: if judge breaks for this gold answer, just continue to next
#             # (overall recall will be 0 unless some other gold is matched).
#             # You can log this if you want:
#             # logger.warning(f"single_truth_answer_recall judge error: {e}")
#             continue

#     return 0.0

# conflict_eval.py
# -*- coding: utf-8 -*-
"""
Conflict-Aware Evaluation
-------------------------

Implements evaluation metrics aligned to the
Dragged-into-Conflicts taxonomy (Cattan et al., 2025).

Metrics:
  • Behavior Adherence  – checks if answer matches expected human behavior
                          for conflict type (via LLM-as-a-judge).
  • Factual Grounding   – fraction of claims in answer entailed by supporting docs.
  • Single-Truth Recall – recall of gold factual answer in single-truth categories.

Dependencies:
  - judge_prompts.py   (behavior_judge_prompt, single_truth_recall_prompt)
  - llm.py             (LLM.judge_behavior, LLM.nli_entailment / async variants)
  - metrics.py         (extract_claims_by_sentence)
  - data.py            (support_doc_ids_from_notes)

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

from typing import Any, Dict, List, Iterable
import re
import asyncio

from .judge_prompts import behavior_judge_prompt, single_truth_recall_prompt


# ---------------------------------------------------------------------
# Behavior adherence (sync)
# ---------------------------------------------------------------------

def behavior_adherence(llm: Any, query: str, answer: str, conflict_type: int) -> Dict[str, Any]:
    """
    Evaluate whether model behavior matches expected guidelines
    for the given conflict type (synchronous version).

    Args:
        llm: LLM wrapper with judge_behavior() method
        query: user query string
        answer: model answer string
        conflict_type: int in {1..5}

    Returns:
        dict with:
          {
            "adherent": bool,
            "rationale": str (short explanation from judge or error fallback)
          }
    """
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


# ---------------------------------------------------------------------
# Behavior adherence (async)
# ---------------------------------------------------------------------

async def abehavior_adherence(llm: Any, query: str, answer: str, conflict_type: int) -> Dict[str, Any]:
    """
    Async behavior adherence using llm.ajudge_behavior().
    """
    if not (answer or "").strip():
        return {"adherent": False, "rationale": "Empty answer"}

    prompt = behavior_judge_prompt(query, answer, conflict_type)
    try:
        res = await llm.ajudge_behavior(prompt)
        return {
            "adherent": bool(res.get("adherent")),
            "rationale": (res.get("rationale") or "").strip(),
        }
    except Exception as e:
        return {"adherent": False, "rationale": f"judge error: {e}"}


# ---------------------------------------------------------------------
# Factual grounding ratio (sync)
# ---------------------------------------------------------------------

def factual_grounding_ratio(
    nli_fn: Any,
    claims: List[str],
    support_docs: List[Dict[str, Any]],
) -> float:
    """
    Fraction of answer claims entailed by at least one supporting doc.
    Synchronous version; uses a provided nli_fn(premise, hypothesis).

    Args:
        nli_fn: function (premise, hypothesis) -> {"entails","contradicts","neutral"}
        claims: list of claim sentences from model answer
        support_docs: list of doc dicts (with 'text' or 'snippet')

    Returns:
        float ∈ [0,1], 0.0 if no claims
    """
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


# ---------------------------------------------------------------------
# Factual grounding ratio (async)
# ---------------------------------------------------------------------

async def afactual_grounding_ratio(
    llm: Any,
    claims: List[str],
    support_docs: List[Dict[str, Any]],
) -> float:
    """
    Async factual grounding ratio using llm.anli_entailment().

    For each claim, we check in parallel across all support docs whether
    ANY doc entails the claim. Then compute fraction of claims that are
    entailed by at least one doc.
    """
    if not claims:
        return 0.0
    if not support_docs:
        return 0.0

    async def check_claim(claim: str) -> bool:
        tasks = []
        for d in support_docs:
            passage = d.get("snippet") or d.get("text") or ""
            tasks.append(llm.anli_entailment(passage, claim))
        if not tasks:
            return False
        relations = await asyncio.gather(*tasks)
        return any(rel == "entails" for rel in relations)

    claim_tasks = [check_claim(c) for c in claims]
    results = await asyncio.gather(*claim_tasks)

    supported = sum(1 for r in results if r)
    return supported / len(claims)


# ---------------------------------------------------------------------
# Single-truth recall (shared helpers)
# ---------------------------------------------------------------------


def _iter_gold_answers(gold_answers: Any) -> Iterable[str]:
    """
    Normalize gold_answers to an iterable of strings.
    Accepts a single string or a list of strings.
    """
    if gold_answers is None:
        return []
    if isinstance(gold_answers, str):
        return [gold_answers]
    if isinstance(gold_answers, (list, tuple)):
        return [g for g in gold_answers if isinstance(g, str)]
    # Fallback: unknown type
    return []


# ---------------------------------------------------------------------
# Single-truth recall (sync)
# ---------------------------------------------------------------------

def single_truth_answer_recall(
    llm: Any,
    gold_answers: Any,
    answer_text: str,
) -> float:
    """
    Recall of gold factual answers in model output, using LLM-as-a-judge
    instead of naive string matching (synchronous).

    - Strips <think>...</think> traces from the candidate answer.
    - For each gold answer, asks an LLM judge whether that factual answer
      is present (possibly paraphrased) in the candidate response.
    - Returns 1.0 if ANY gold answer is judged as present, else 0.0.

    Args:
        llm: LLM wrapper implementing judge_behavior(prompt: str) -> dict
        gold_answers: str or list of gold answer strings (from dataset).
        answer_text: raw model answer string (may include <think>...</think>).

    Returns:
        float ∈ {0.0, 1.0}
    """
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
            # On judge error for this gold, continue to next.
            continue

    return 0.0


# ---------------------------------------------------------------------
# Single-truth recall (async)
# ---------------------------------------------------------------------

async def asingle_truth_answer_recall(
    llm: Any,
    gold_answers: Any,
    answer_text: str,
) -> float:
    """
    Async single-truth recall using llm.ajudge_behavior().

    Returns 1.0 if ANY gold answer is judged present in the candidate
    answer (after stripping <think>...</think>), else 0.0.
    """
    gold_iter = list(_iter_gold_answers(gold_answers))
    if not gold_iter:
        return 0.0

    candidate = answer_text
    if not candidate:
        return 0.0

    tasks = []
    prompts = []

    for gold in gold_iter:
        if not gold:
            continue
        prompt = single_truth_recall_prompt(gold_answer=gold, model_answer=candidate)
        prompts.append(prompt)
        tasks.append(llm.ajudge_behavior(prompt))

    if not tasks:
        return 0.0

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            continue
        if bool(res.get("adherent")):
            return 1.0

    return 0.0