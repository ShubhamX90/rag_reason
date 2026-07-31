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
- Do not use answerability correctness, factual entailment, citation validity, unsupported-claim detection, or single-truth recall as hidden behavior criteria. Those are separate metrics.
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
{model_answer}"""


# ---------------------------------------------------------------------
# FG committee prompt (v2 grounding approach)
# ---------------------------------------------------------------------

def fg_committee_prompt(
    query: str,
    claim_text: str,
    eligible_docs: List[Dict[str, Any]],
    model_answer: str = "",
) -> str:
    """Prompt for the judge committee to identify which pre-annotated supporting
    documents contain evidence for a specific claim extracted from the model's answer.

    The committee is given full context:
      - The query
      - The model's complete Final Answer (think-trace stripped) — for context
      - The specific claim extracted from the Final Answer to evaluate
      - Retrieved documents pre-filtered to 'supports' / 'partially supports' verdicts,
        each shown with its gold annotation (verdict, key_fact, quote, snippet).
        These verdicts are **ground truth** — the committee does NOT need to re-verify
        whether a doc is relevant; it only needs to check whether the specific claim
        is stated or clearly inferable from that doc's content.

    eligible_docs: list of dicts with keys doc_id, verdict, key_fact, quote, snippet.
    Only "supports" and "partially supports" docs are passed here — "irrelevant" docs
    are filtered out before calling this function.

    model_answer: the model's Final Answer (think-trace stripped). Shown to the
    committee for context so the claim can be understood in the full answer's frame.
    """
    docs_block_lines = []
    for d in eligible_docs:
        doc_id = d.get("doc_id", "?")
        verdict = d.get("verdict", "unknown")
        key_fact = (d.get("key_fact") or "").strip()
        quote = (d.get("quote") or "").strip()
        snippet = (d.get("snippet") or "").strip()

        if quote and snippet and quote not in snippet:
            passage = f"Key evidence (annotator-verified): {quote}\nFull passage: {snippet}"
        elif quote and snippet:
            passage = snippet
        elif quote:
            passage = quote
        else:
            passage = snippet

        docs_block_lines.append(
            f"--- Document {doc_id} [gold verdict: {verdict}] ---\n"
            f"Gold key fact: {key_fact}\n"
            f"Passage: {passage[:350]}"
        )

    docs_block = "\n\n".join(docs_block_lines) if docs_block_lines else "(no eligible documents)"
    valid_ids_str = ", ".join(d.get("doc_id", "?") for d in eligible_docs)

    model_answer_block = ""
    if model_answer:
        # Truncate long answers — the committee needs context, not the full 2000-token trace.
        truncated = model_answer[:500]
        if len(model_answer) > 500:
            truncated += "\n[... truncated for brevity ...]"
        model_answer_block = (
            f"MODEL'S FINAL ANSWER (the complete output from which the claim was extracted):\n"
            f"{truncated}\n\n"
        )

    return (
        "You are a fact-checking judge. Your task is to determine which retrieved documents "
        "support a specific claim from a model's answer.\n\n"
        "IMPORTANT GROUND RULES:\n"
        "- The document verdicts ('supports', 'partially supports') are GROUND TRUTH annotations "
        "made by human experts. You do NOT need to re-verify whether these docs are relevant to "
        "the query — they already are. Your job is ONLY to check whether the SPECIFIC CLAIM "
        "is conveyed by each doc's text.\n"
        "- Do NOT use outside knowledge. Only what the document text says.\n"
        "- No concept of 'contradicts' — ignore any conflict between docs. Only look for support.\n\n"
        "WHAT COUNTS AS SUPPORT (be generous — partial and paraphrastic support BOTH count):\n"
        "  A document supports the claim if its text establishes the claim's core assertion. "
        "The wording does NOT need to match the claim word-for-word. In particular, the doc "
        "supports the claim when ANY of the following holds:\n"
        "  (a) The doc paraphrases the claim, uses synonyms, or summarizes the same fact.\n"
        "  (b) The doc states a fact that, taken at face value, entails or implies the claim.\n"
        "  (c) The doc gives a more specific instance, example, or sub-case of the claim "
        "(specific instance → the general claim is supported).\n"
        "  (d) The doc gives a more general statement that covers the claim (the claim is a "
        "natural sub-case of what the doc says).\n"
        "  (e) The doc supplies a numeric range, qualitative range, or category that "
        "includes the value the claim asserts.\n"
        "  (f) The doc covers a meaningful portion of the claim's substance — even if "
        "secondary modifiers, qualifiers, or peripheral details in the claim aren't in the "
        "doc. Missing modifiers do NOT disqualify a doc; missing the core fact does.\n"
        "  (g) The gold verdict is 'partially supports' AND the doc's key_fact/passage "
        "addresses the claim's main assertion — treat this as supporting unless the doc is "
        "clearly about a different topic.\n\n"
        "WHAT DOES NOT COUNT AS SUPPORT:\n"
        "  - The doc is about a different entity, topic, or time period than the claim.\n"
        "  - The doc only mentions a keyword from the claim without addressing the "
        "claim's actual assertion.\n"
        "  - The doc would require external knowledge or several inferential leaps to "
        "connect to the claim.\n\n"
        "Bias toward INCLUSION when in doubt: if a reasonable reader would say 'yes, this "
        "doc backs up the gist of the claim,' mark it supporting. Reserve exclusion for "
        "docs that are clearly off-topic or only tangentially related.\n\n"
        f"QUERY: {query}\n\n"
        f"{model_answer_block}"
        f'SPECIFIC CLAIM TO EVALUATE: "{claim_text}"\n\n'
        "RETRIEVED DOCUMENTS (pre-annotated supporting docs — decide which contain "
        "evidence for the SPECIFIC CLAIM above):\n"
        f"{docs_block}\n\n"
        "YOUR TASKS:\n"
        "1. For each document above, apply the WHAT COUNTS AS SUPPORT rules (a)–(g). "
        "List every doc that meets ANY of (a)–(g) in supporting_docs.\n"
        "2. If NO single document satisfies (a)–(g), check whether the COMBINATION "
        "of any two documents together supports the claim (cross-document evidence — one "
        "doc provides part A, another provides part B, and together they establish the claim). "
        "Only set cross_doc_support=true in this case.\n\n"
        f"Valid document IDs you may reference: {valid_ids_str}\n\n"
        "Return ONLY a valid JSON object with exactly these keys and no comments, no markdown, and no extra text:\n"
        "{\n"
        '  "supporting_docs": ["d2", "d3"],\n'
        '  "cross_doc_support": false,\n'
        '  "cross_doc_combo": []\n'
        "}\n\n"
        "Rules for the JSON fields:\n"
        '- "supporting_docs": doc_ids that individually support the claim; use [] if none.\n'
        '- "cross_doc_support": true ONLY when no single doc supports but two docs together do.\n'
        '- "cross_doc_combo": exactly the two doc_ids that combine to support when cross_doc_support is true; otherwise [].'
    )
