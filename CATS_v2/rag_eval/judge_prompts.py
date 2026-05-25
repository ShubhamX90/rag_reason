# judge_prompts.py
# -*- coding: utf-8 -*-
"""
Judge Prompts
-------------

Defines strict JSON prompts for LLM-as-a-Judge components in
Conflict-Aware Evaluation.

Tasks supported:
  • Behavior Adherence – check if model answers match expected behavior
                         for conflict type (Dragged-into-Conflicts taxonomy).
  • NLI Entailment – check if passage (premise) entails, contradicts, or is neutral
                     to claim (hypothesis).

These prompts are designed to be strict, concise, and enforce JSON output.

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Behavior adherence rubric (aligned to taxonomy)
# ---------------------------------------------------------------------

BEHAVIOR_RUBRIC = {
 1: "No Conflict: Provide a clear and direct answer without introducing alternative viewpoints or uncertainty.", 
 2: "Complementary Information: Consolidate and reconcile the different partial answers provided by the retrieved documents, without framing the response as a debate.", 
 3: "Conflicting Opinions or Research Outcomes: Explicitly reflect the debate between the retrieved sources and to neutrally summarize the different viewpoints.",
 4: "Conflict Due to Outdated Information: Prioritise the up-to-date information, while optionally acknowledging the presence of outdated sources.", 
 5: "Conflict Due to Misinformation: Disregard inaccurate sources and provide a response grounded in reliable and verified information.",
 }

# ---------------------------------------------------------------------
# Behavior adherence prompt
# ---------------------------------------------------------------------

def behavior_judge_prompt(
    query: str,
    answer: str,
    conflict_type: int,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build prompt for behavior adherence judgment.

    The judge should focus on whether the ANSWER'S BEHAVIOR matches the
    expected behavior for the given conflict type (how it handles multiple
    sources, uncertainty, disagreement, etc.), NOT whether the answer is
    factually correct.

    A valid refusal ("CANNOT ANSWER, INSUFFICIENT EVIDENCE") is adherent
    when the retrieved documents do not actually support a confident answer,
    regardless of conflict type — refusal is always an acceptable behavior
    when the evidence is genuinely insufficient.

    For Type 4 (Outdated Information) and Type 5 (Misinformation), the judge
    needs document dates / source provenance to assess whether the model
    prioritised the newer / more reliable source. If `retrieved_docs` is
    provided and contains a `date` or `source` field, a compact provenance
    table is inserted into the prompt for those conflict types.
    """
    rubric = BEHAVIOR_RUBRIC.get(conflict_type, BEHAVIOR_RUBRIC[1])

    # N6 fix: surface document dates/sources to behavior judges for Type 4 / Type 5,
    # so "prioritise the up-to-date information" can be judged on evidence rather
    # than wording alone.
    provenance_block = ""
    if retrieved_docs and conflict_type in (4, 5):
        rows = []
        for d in retrieved_docs:
            doc_id = d.get("doc_id", "?")
            date = (d.get("date") or d.get("timestamp") or "unknown").strip() or "unknown"
            source = (d.get("source") or d.get("url") or "").strip()
            src_tag = f"  source={source}" if source else ""
            rows.append(f"  - {doc_id}: date={date}{src_tag}")
        if rows:
            label = "publication dates" if conflict_type == 4 else "publication dates / sources"
            provenance_block = (
                f"\nDocument provenance ({label} — use these to judge whether the "
                f"answer prioritised the right source):\n"
                + "\n".join(rows)
                + "\n"
            )

    return f"""
You are evaluating ONLY the **behavior** of a model answer, not its factual correctness.

Behavior means:
- How the answer handles multiple sources, uncertainty, disagreement, or lack of conflict.
- Whether it summarizes, reconciles, or contrasts viewpoints as appropriate.
- Whether it is direct vs. hedged, neutral vs. biased, etc.

Given:
- A user query
- A model-generated answer
- A conflict type with ONE expected behavior rubric (shown below)

Your task:
Decide whether the model's answer **follows the expected behavior** for THIS specific conflict type only.

Conflict Type: {conflict_type}
Expected Behavior for this conflict type:
{rubric}
{provenance_block}
Instructions:
- Judge the answer ONLY against the rubric shown above. Do not invoke rubrics for other conflict types.
- If the answer is a refusal ("CANNOT ANSWER" / "INSUFFICIENT EVIDENCE") AND the retrieved evidence genuinely does not support an answer, treat the refusal as adherent.
- If the answer clearly follows the expected behavior for this conflict type, set "adherent": true.
- If the answer clearly violates or ignores the expected behavior, set "adherent": false.
- Ignore factual correctness; only judge how the answer behaves relative to the rubric.
- The "rationale" should briefly cite the key aspects of the answer's behavior.
- The "confidence" should reflect how certain you are in YOUR adherent decision, between 0.0 (coin flip) and 1.0 (clearly correct call). Use 0.5 if you are uncertain.

Return ONLY a JSON object with fields:
  "adherent": true or false,
  "rationale": "short explanation",
  "confidence": 0.0 to 1.0

Query:
{query}

Answer:
{answer}
""".strip()


# ---------------------------------------------------------------------
# NLI entailment prompt
# ---------------------------------------------------------------------

def nli_prompt(premise: str, hypothesis: str) -> str:
    """
    Build prompt for NLI entailment judgment tailored for grounded-citation evaluation.

    The judge determines whether the retrieved document passage (premise)
    supports, contradicts, or is unrelated to the model's claim (hypothesis).
    Only the information in the premise may be used.
    """
    return f"""
You are performing **evidence-based Natural Language Inference (NLI)** for grounded citation checking.

Your job:
Determine the logical relationship between a retrieved document passage (the *premise*) and a model-generated claim (the *hypothesis*), using ONLY the explicit content of the premise.

Definitions:
- "entails": The premise clearly supports or confirms the hypothesis. The hypothesis must logically follow from what the premise states.
- "contradicts": The premise clearly conflicts with or disproves the hypothesis.
- "neutral": The premise does not provide enough information to either support or contradict the hypothesis. The claim may be plausible, but it is not justified by the premise.

Rules:
- **Do not add knowledge**, outside interpretation, or world facts.
- **Do not guess** beyond what the premise literally says.
- Focus ONLY on whether the hypothesis is justified by the premise.
- Meta-statements about the answer itself ("the documents agree", "this is therefore considered real")
  are NOT entailed by a premise unless the premise itself states that meta-fact verbatim.
- A premise that merely mentions a topic without making the specific claim in the hypothesis is "neutral",
  not "entails".
- **Structured premise format:** the premise may begin with a labelled section
  "Key evidence (annotator-verified):" followed by a short verbatim quote, then
  "Full passage:" followed by the broader surrounding context. When both sections
  are present, use the content of BOTH sections to evaluate entailment — the key
  evidence is the most focused span, the full passage supplies surrounding context.
  The section labels themselves ("Key evidence", "Full passage") are formatting
  artefacts, not evidence.

Return ONLY a JSON object (no markdown fences, no extra text):
{{
  "relation": "entails" | "contradicts" | "neutral",
  "confidence": 0.0 to 1.0
}}

Premise (evidence from retrieved document):
{premise}

Hypothesis (claim from model answer):
{hypothesis}
""".strip()


def single_truth_recall_prompt(gold_answer: str, model_answer: str) -> str:
    """
    Prompt for LLM-as-a-judge to decide if the gold factual answer is
    *asserted* by the model's answer (not merely mentioned or attributed
    to a document).

    Key distinctions enforced by this prompt (v3 fix for N2B + N3):
      • Assertion vs mention — quoting / citing the gold answer from a
        retrieved document while the model itself commits to a different
        answer does NOT count as recall.
      • Spelling / formatting variation — minor misspellings, casing,
        whitespace, punctuation, or unit-formatting differences are
        acceptable matches.
      • Logical equivalence — semantically equivalent statements
        (paraphrases, equivalent dates, "X is Y" vs "Y is X" for
        symmetric relations) count as matching, mirroring the paper's
        single-truth recall judge.
    """
    return f"""
You are checking whether a model answer correctly **asserts** a given factual answer.

The gold answer is the correct answer to a question. Your job is to decide whether the model's answer
**commits to** (asserts) this gold answer as the model's own conclusion.

MATCHING RULES (count as adherent=true):
- Paraphrases, abbreviations, and equivalent wording that convey the same fact.
- **Logically equivalent statements** (e.g. "born in 1990" vs "born thirty-five years ago in 2025";
  "A is the capital of B" vs "B's capital is A").
- Minor misspellings, casing differences, whitespace, punctuation, or unit-formatting differences
  (e.g. "Chiliwack" ≈ "Chilliwack"; "Stephan Curry" ≈ "Stephen Curry"; "1,759" ≈ "1759";
  "$1.8 billion" ≈ "1.8B USD").

NON-MATCHING RULES (count as adherent=false):
- The model commits to a DIFFERENT answer, even if the gold answer also appears somewhere in the text
  as a **quotation, citation, or attribution** to a document.
  Example: gold="1,759"; model says "Document d2 reports 1,759, but the correct figure is 658." →
  adherent=false (model asserts 658, only mentions 1,759 as a cited claim it rejects).
- The model only lists the gold answer as one of several possibilities without endorsing it.
- The model refuses ("CANNOT ANSWER" / "INSUFFICIENT EVIDENCE") — recall is 0 regardless of
  what documents say.
- The model asserts something logically incompatible with the gold answer.

Return ONLY a JSON object with fields:
  "adherent": true or false (whether the model's answer ASSERTS the gold answer as its own conclusion),
  "rationale": short string explanation (briefly note whether the gold was asserted or merely mentioned),
  "confidence": 0.0 to 1.0 (your certainty in the adherent decision)

Gold factual answer:
{gold_answer}

Model answer:
{model_answer}
""".strip()