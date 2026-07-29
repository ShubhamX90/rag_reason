from __future__ import annotations

from typing import Any, Dict, List, Optional


BEHAVIOR_RUBRIC: Dict[int, str] = {
    1: "No Conflict: Provide a clear and direct answer without introducing alternative viewpoints or uncertainty.",
    2: "Complementary Information: Consolidate and reconcile the different partial answers provided by the retrieved documents, without framing the response as a debate.",
    3: "Conflicting Opinions or Research Outcomes: Explicitly reflect the debate between the retrieved sources and neutrally summarize the different viewpoints.",
    4: "Conflict Due to Outdated Information: Prioritise the up-to-date information, while optionally acknowledging the presence of outdated sources.",
    5: "Conflict Due to Misinformation: Disregard inaccurate sources and provide a response grounded in reliable and verified information.",
}


BEHAVIOR_GUIDE = """
You are evaluating ONLY the behavior of a model answer, not its factual correctness.

Behavior means:
- how the answer handles multiple sources, uncertainty, disagreement, or lack of conflict
- whether it summarizes, reconciles, or contrasts viewpoints as appropriate
- whether it is direct vs. hedged, neutral vs. biased, and aligned to the expected conflict-handling pattern

Your task:
Decide whether the model answer follows the expected behavior for THIS specific conflict type only.

Instructions:
- Judge the answer ONLY against the rubric for the current conflict type. Do not invoke rubrics for other conflict types.
- If the answer is a refusal and the retrieved evidence genuinely does not support a confident answer, treat the refusal as adherent.
- If the answer clearly follows the expected behavior for this conflict type, judge it adherent.
- If the answer clearly violates, ignores, or mishandles the expected behavior, judge it non-adherent.
- Ignore factual correctness; only judge how the answer behaves relative to the rubric and gold conflict framing.
- Use the gold per-document notes to understand what each retrieved document contributes.
- Confidence should reflect how certain you are in your adherence judgment, from 0.0 to 1.0.
""".strip()


STR_GUIDE = """
Judge whether the model answer ASSERTS the gold answer as its own conclusion.

Count as a match:
- paraphrases
- logically equivalent statements
- minor spelling or formatting variation

Do not count as a match:
- the gold answer is only quoted, cited, or attributed to a document
- the model asserts a different answer
- the model only lists the gold answer as one option
- the model refuses
""".strip()


FG_GUIDE = """
Humans judge the same deterministic claims shown to the LLM judges.

What "eligible docs" means here:
- The displayed eligible-doc pool is restricted to documents whose gold per-doc verdict is `supports` or `partially supports` for this sample.
- These are the only docs that can count toward FG.
- But not every eligible doc supports every extracted claim. For each claim, you must decide which of the eligible docs actually support that specific claim.

For each extracted claim:
- identify ALL eligible docs that individually support that claim
- do this regardless of whether the model cited those docs or not
- do NOT restrict yourself to the docs cited in the model's final answer
- if no single eligible doc supports the claim, decide whether exactly two eligible docs together support it

Input rules:
- you may enter doc ids as `d1,d2,d3`
- or shorthand numeric ids as `1,2,3`, which mean the same thing as `d1,d2,d3`
- leave blank only when no single eligible doc supports the claim

How scoring works afterward:
- the software first uses your judgment to determine which eligible docs really support the claim
- it then separately checks whether the model cited at least one of those supporting docs
- so your job is to mark real support, not citation overlap
""".strip()


def behavior_provenance_lines(
    conflict_id: int,
    docs: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    if conflict_id not in (4, 5) or not docs:
        return []
    rows: List[str] = []
    for doc in docs:
        doc_id = doc.get("doc_id", "?")
        date = (doc.get("timestamp") or doc.get("date") or "unknown").strip() or "unknown"
        source = (doc.get("source_url") or doc.get("url") or "").strip()
        src_tag = f" | source={source}" if source else ""
        rows.append(f"- {doc_id}: date={date}{src_tag}")
    return rows
