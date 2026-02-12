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

def behavior_judge_prompt(query: str, answer: str, conflict_type: int) -> str:
    """
    Build prompt for behavior adherence judgment.

    The judge should focus on whether the ANSWER'S BEHAVIOR matches the
    expected behavior for the given conflict type (how it handles multiple
    sources, uncertainty, disagreement, etc.), NOT whether the answer is
    factually correct.
    """
    rubric = BEHAVIOR_RUBRIC.get(conflict_type, BEHAVIOR_RUBRIC[1])
    return f"""
You are evaluating ONLY the **behavior** of a model answer, not its factual correctness.

Behavior means:
- How the answer handles multiple sources, uncertainty, disagreement, or lack of conflict.
- Whether it summarizes, reconciles, or contrasts viewpoints as appropriate.
- Whether it is direct vs. hedged, neutral vs. biased, etc.

Given:
- A user query
- A model-generated answer
- A conflict type with an expected behavior rubric

Your task:
Decide whether the model's answer **follows the expected behavior** for this conflict type.

Conflict Type: {conflict_type}
Expected Behavior (rubric):
{BEHAVIOR_RUBRIC}

Instructions:
- If the answer clearly follows the expected behavior, set "adherent": true.
- If the answer clearly violates or ignores the expected behavior, set "adherent": false.
- Ignore factual correctness; only judge how the answer behaves relative to the rubric.
- The "rationale" should briefly point to the key aspects of the answer's behavior
  (for example, whether it mentions multiple viewpoints, reconciles partial info,
  prioritizes newer evidence, corrects misinformation, etc.).

Return ONLY a JSON object with fields:
  "adherent": true or false,
  "rationale": "short explanation"

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

Return ONLY a JSON object:
{{
  "relation": "entails" | "contradicts" | "neutral"
}}

Premise (evidence from retrieved document):
{premise}

Hypothesis (claim from model answer):
{hypothesis}
""".strip()


def single_truth_recall_prompt(gold_answer: str, model_answer: str) -> str:
    """
    Prompt for LLM-as-a-judge to decide if the gold factual answer is
    present in the candidate model answer (possibly paraphrased or with
    extra context).

    Output must be ONLY a JSON object of the form:
      {
        "adherent": true/false,   // true if gold answer is present
        "rationale": "short explanation"
      }
    """
    return f"""
You are checking whether a candidate answer correctly contains a given factual answer.
Consider paraphrases, equivalent wording, and logically equivalent statements as MATCHING.

Return ONLY a JSON object with fields:
  "adherent": true or false,
  "rationale": short string explanation.

The interpretation:
  - "adherent": true  -> the candidate answer DOES clearly state the gold answer
                         (possibly paraphrased or with additional context).
  - "adherent": false -> the candidate answer does NOT contain the gold answer
                         or states something incompatible.

Gold factual answer:
{gold_answer}

Candidate model answer:
{model_answer}
""".strip()