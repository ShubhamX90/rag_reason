# rag_eval/metrics.py
# -*- coding: utf-8 -*-
"""
Utility functions for CATS v2.0 metrics.

Provides:
  • refusal detection (unified regex over both startswith and substring patterns)
  • claim extraction (filters citation noise and tiny fragments before NLI)
  • per-sample binary GR (renamed to gr_accuracy_from_flags — the old name F1 was misleading)
  • dataset-level F1 (compute_f1_gr) computed from TP/FP/FN/TN counts
"""

import re
from typing import List, Dict
from nltk import sent_tokenize


# --------------------
# Refusal detection
# --------------------
# Unified refusal patterns. The previous version mixed startswith and substring
# checks inconsistently; this single regex covers both English-language refusal
# phrasings and the "CANNOT ANSWER, INSUFFICIENT EVIDENCE" stylized outputs the
# models in this benchmark produce. Case-insensitive substring match.
_REFUSAL_REGEX = re.compile(
    r"\b("
    r"cannot\s+answer|can't\s+answer|cant\s+answer"
    r"|i\s+cannot|i\s+can't|i\s+cant"
    r"|i'?m\s+unable|i\s+am\s+unable"
    r"|unable\s+to\s+(?:answer|determine|conclude)"
    r"|insufficient\s+(?:evidence|information|context)"
    r"|no\s+(?:reliable\s+)?(?:evidence|information)\s+to\s+answer"
    r"|(?:do\s*n'?t|does\s*n'?t)\s+have\s+(?:enough|sufficient)\s+information"
    r"|not\s+enough\s+(?:evidence|information|context)"
    r"|cannot\s+(?:determine|conclude|establish)"
    r")\b",
    re.IGNORECASE,
)


def answered_flags(outputs: List[str]) -> List[bool]:
    """
    Detect if each output is a real answer (vs a refusal/abstain).
    Returns parallel list of bools: True if output looks like an answer.
    """
    flags = []
    for o in outputs:
        text = (o or "").strip()
        if not text:
            flags.append(False)
            continue
        is_refusal = bool(_REFUSAL_REGEX.search(text))
        flags.append(not is_refusal)
    return flags


# --------------------
# Claim extraction
# --------------------
# Patterns that mark "this sentence is just citation/meta text, not a substantive claim"
_CITATION_ONLY = re.compile(r"^\s*[\[\(\s,]*(?:d\d+|\[\d+\])(?:[\s,]*(?:d\d+|\[\d+\]))*[\s\.\]\)]*$", re.IGNORECASE)

# Anaphoric meta-references to citations: "all explicitly state this fact",
# "these sources confirm", "provide evidence supporting this link", etc.
# Two patterns:
#  (a) starts with an anaphoric quantifier ("all", "these", "those", "both",
#      "the documents") followed by a verb of saying.
#  (b) starts directly with a verb of saying (left behind after a leading
#      citation list was stripped, e.g., "d1, d3, and d5 [provide evidence...]"
#      → "provide evidence supporting this link").
_META_REFERENCE = re.compile(
    r"^\s*("
    r"(all|these|those|both|they|the\s+(documents|sources|references|citations))\b"
    r".{0,30}?\b(state|say|report|confirm|show|indicate|provide|mention|explicitly|note|claim|support)\b"
    r"|"
    r"(state|states|stated|say|says|report|reports|reported|confirm|confirms|confirmed"
    r"|show|shows|showed|indicate|indicates|indicated|provide|provides|provided"
    r"|mention|mentions|mentioned|note|notes|noted|claim|claims|claimed|support|supports|supported)\b"
    r")",
    re.IGNORECASE,
)


def _strip_citations_inplace(text: str) -> str:
    """Remove citation markers from a sentence before NLI.

    Handles bracketed (`[d1]`, `[1]`, `(d2)`) and bare (`d1, d3, d5`) citation
    forms. Bare doc IDs are only stripped when they appear in a comma-separated
    list at the start of a sentence — this matches the citation idiom without
    deleting legitimate uses of `d1` in mid-sentence prose.

    Also cleans up the comma/conjunction debris ", , , and" left behind so
    NLI doesn't see "all explicitly state this fact" as a stand-alone claim.
    """
    # Bracketed forms
    text = re.sub(r"\[\s*d\d+\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\(\s*d\d+\s*\)", "", text, flags=re.IGNORECASE)

    # Bare doc-id lists at sentence start: "d1, d3, and d5 ..." → " ..."
    text = re.sub(
        r"^\s*d\d+(?:\s*,\s*(?:and\s+)?d\d+)+\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Or appearing as a list elsewhere: ", d1, d3, d5 "
    text = re.sub(
        r"(?:,\s*)?d\d+(?:\s*,\s*(?:and\s+)?d\d+){2,}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse comma-and-comma debris
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
    text = re.sub(r"^\s*[,\s]+", "", text)
    text = re.sub(r"\s*,\s*(and|or)\b\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(and|or)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:")
    return text


# --------------------
# Attribution stripping (Fix 4)
# --------------------
# Many gold answers wrap verifiable facts inside source-attribution phrases:
#   "According to Wikipedia, X"             → X
#   "Source A and Source B both confirm X"  → X
#   "As of early 2025, X"                   → X
#   "X, confirmed across multiple sources"  → X
# Stripping the attribution exposes the atomic verifiable fact so NLI can
# evaluate it against individual docs without being confused by meta-language.

# Leading attribution patterns:
#   (a) "According to / As reported/stated/noted/confirmed by [Source],"
#   (b) "[Sources...] both/also confirm/show/report/state/note (that)"
#       — requires "both" or "also" to avoid stripping normal sentences
#         like "The study shows that X"
#   (c) "As of [temporal qualifier],"
_ATTR_PREFIX_RE = re.compile(
    r"^(?:"
    # (a) Standard attribution opener
    r"(?:According|As\s+(?:reported|stated|noted|confirmed))\s+(?:to|by)\s+[^,]{4,80}?,\s*"
    # (b) Named-source attribution: "[Sources] both/also VERB (that)"
    r"|[A-Z][^.]{8,120}?\s+(?:both|also)\s+(?:confirm|confirms|show|shows|report|reports|state|states|note|notes|indicate|indicates)\s+(?:that\s+)?"
    # (c) Quantity-source attribution: "Multiple/Several/Both/All sources VERB (that)"
    r"|(?:Multiple|Several|Both|All|Many)\s+sources?\s+(?:confirm|confirms|show|shows|report|reports|indicate|indicates|agree|corroborate|concur)\s+(?:that\s+)?"
    # (d) Temporal qualifier
    r"|[Aa]s\s+of\s+[^,]{4,60},\s*"
    r")",
    re.DOTALL,
)

# Trailing confirmation suffix:
#   "X, confirmed across multiple sources"  → X
#   "X, unanimously established by all studies"  → X
_TRAILING_META_RE = re.compile(
    r",\s*(?:unanimously|widely|consistently|independently)?\s*"
    r"(?:confirmed|verified|established|documented|supported|agreed)"
    r"(?:\s+(?:across|by|among|in)\s+[^.]*)?\.?$",
    re.IGNORECASE,
)


def _strip_attribution(claim: str) -> str:
    """
    Strip source-attribution wrapper from a claim, returning the core verifiable fact.

    Only strips when the residual is >= 4 words so we don't create empty fragments.
    Capitalises the first letter of the residual when the original was capitalised.
    """
    s = claim

    # Strip leading attribution prefix
    m = _ATTR_PREFIX_RE.match(s)
    if m and m.end() < len(s) - 1:
        residual = s[m.end():].strip(" ,")
        if len(residual.split()) >= 4:
            s = (residual[0].upper() + residual[1:]) if residual else residual

    # Strip trailing meta-confirmation suffix
    s_stripped = _TRAILING_META_RE.sub("", s).strip(" .")
    if len(s_stripped.split()) >= 4:
        s = s_stripped

    return s


def extract_claims_by_sentence(answer_text: str, max_claims: int = 12) -> List[str]:
    """
    Split answer into candidate claims (sentences) suitable for NLI.

    Pre-processing applied before sentence-splitting:
      • Protects "Lyndon B. Johnson"-style initials from being split mid-name
        by temporarily replacing the period after a single capital initial.
      • Strips bracket citations like [d1], [1], (d2) from each sentence
        because NLI on a sentence that's just citations gives garbage.

    Post-processing:
      • Drops sentences that are <= 4 words or pure citation noise.
    """
    if not answer_text:
        return []

    text = answer_text

    # Pre-tokenization protection — NLTK splits sentences on `.` and routinely
    # destroys text that contains periods inside tokens. Replace every "safe"
    # period with the sentinel <DOT>, restore it after sentence-splitting.
    #
    # The qwen-monolithic run shows these failures repeatedly:
    #   • initials   "Lyndon B. Johnson"        → ["Lyndon B.", "Johnson was..."]
    #   • decimals   "17.83%", "$1.8 billion"   → ["...with 17.", "83%..."]
    #   • domains    "a.COM domain"             → ["a.", "COM domain..."]
    #   • company    "Phoenix Mills Co. Ltd."   → ["...Co.", "Ltd.", "..."]
    #   • Mr./Dr./Inc./vs./etc.

    # 1. Initials before a capitalized word: "Lyndon B. Johnson"
    text = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", r"\1<DOT>", text)

    # 2. Decimal numbers (and percentages/currency): "1.8", "17.83", "$2.5"
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)

    # 3. Domain extensions and any "a.UPPERCASE" pattern: ".COM", ".NET", "a.com"
    text = re.sub(r"\b([a-zA-Z])\.([A-Za-z]{2,4})\b", r"\1<DOT>\2", text)

    # 4. Common abbreviations that end with `.` and should not split.
    _ABBREV = (
        "Inc", "Ltd", "Co", "Corp", "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr",
        "vs", "etc", "e\\.g", "i\\.e", "Fig", "Vol", "No", "Ave", "Blvd", "Mt",
    )
    for ab in _ABBREV:
        text = re.sub(rf"\b({ab})\.", rf"\1<DOT>", text)

    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = [s + "." for s in text.split(".") if s.strip()]

    out: List[str] = []
    for s in sents:
        s = s.replace("<DOT>", ".").strip()
        if not s:
            continue

        stripped = _strip_citations_inplace(s)
        if not stripped:
            continue

        # Drop pure-citation sentences and very short fragments.
        if _CITATION_ONLY.match(stripped):
            continue

        # Drop anaphoric meta-references like "all explicitly state this fact"
        # that survive citation stripping but say nothing substantive on their own.
        if _META_REFERENCE.match(stripped):
            continue

        words = stripped.split()
        if len(words) < 4:
            continue

        # Heuristic: if the sentence has no content word of >=5 chars,
        # it's almost certainly a meta-citation fragment, not a claim.
        if not any(len(w.strip(".,;:!?")) >= 5 for w in words):
            continue

        # Fix 4: Strip source-attribution wrappers ("According to X, ...",
        # "A and B both confirm ...", "As of DATE, ...") so NLI evaluates
        # the core verifiable fact rather than the meta-attribution sentence.
        stripped = _strip_attribution(stripped)
        if not stripped or len(stripped.split()) < 4:
            continue

        out.append(stripped)
        if len(out) >= max_claims:
            break

    return out


def strip_think_trace(text: str) -> str:
    """Remove <think>...</think> reasoning trace so FG only scores the final answer."""
    if text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text or ""


# _CITE_EXTRACT: capture bare doc-id inside [dN] brackets for per-sentence citation extraction
_CITE_EXTRACT = re.compile(r"\[d(\d+)\]", re.IGNORECASE)


def extract_claims_with_citations(answer_text: str, max_claims: int = 12) -> List[Dict]:
    """Split answer into claims, preserving the inline citations found in each sentence.

    Returns a list of dicts: {"text": <clean claim>, "cited_docs": [<doc_id>, ...]}.
    Applies the same sentence-splitting and filtering logic as
    extract_claims_by_sentence, but captures [dN] citations before stripping them.
    """
    if not answer_text:
        return []

    text = answer_text

    # Same pre-tokenization protection as extract_claims_by_sentence
    text = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    text = re.sub(r"\b([a-zA-Z])\.([A-Za-z]{2,4})\b", r"\1<DOT>\2", text)
    _ABBREV = (
        "Inc", "Ltd", "Co", "Corp", "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr",
        "vs", "etc", "e\\.g", "i\\.e", "Fig", "Vol", "No", "Ave", "Blvd", "Mt",
    )
    for ab in _ABBREV:
        text = re.sub(rf"\b({ab})\.", rf"\1<DOT>", text)

    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = [s + "." for s in text.split(".") if s.strip()]

    out: List[Dict] = []
    for s in sents:
        s = s.replace("<DOT>", ".").strip()
        if not s:
            continue

        # Extract citations BEFORE stripping them
        cited_docs = [f"d{m}" for m in _CITE_EXTRACT.findall(s)]
        # Deduplicate while preserving order
        seen_cites: set = set()
        unique_cites: List[str] = []
        for d in cited_docs:
            if d not in seen_cites:
                unique_cites.append(d)
                seen_cites.add(d)

        stripped = _strip_citations_inplace(s)
        if not stripped:
            continue
        if _CITATION_ONLY.match(stripped):
            continue
        if _META_REFERENCE.match(stripped):
            continue
        words = stripped.split()
        if len(words) < 4:
            continue
        if not any(len(w.strip(".,;:!?")) >= 5 for w in words):
            continue
        stripped = _strip_attribution(stripped)
        if not stripped or len(stripped.split()) < 4:
            continue

        out.append({"text": stripped, "cited_docs": unique_cites})
        if len(out) >= max_claims:
            break

    return out


# --------------------
# Citations
# --------------------

def extract_bracket_citations(answer_text: str) -> List[str]:
    """Extract [dX] style citations from answer text, preserving first-seen order."""
    if not answer_text:
        return []
    citations = re.findall(r"\[(d\d+)\]", answer_text, flags=re.IGNORECASE)
    seen: set = set()
    out: List[str] = []
    for c in citations:
        c_lower = c.lower()
        if c_lower not in seen:
            out.append(c_lower)
            seen.add(c_lower)
    return out


# --------------------
# Grounded-Refusal metrics
# --------------------

def gr_accuracy_from_flags(pred_answered: bool, gold_answerable: bool) -> float:
    """
    Per-sample binary correctness of the answer/refuse decision.

    Renamed from f1_gr_from_flags because per-sample it is not an F1 — it is
    a 0/1 indicator. The dataset-level F1 is computed by compute_f1_gr below.
    """
    return 1.0 if int(pred_answered) == int(gold_answerable) else 0.0


# Backward-compatible alias (deprecate later)
f1_gr_from_flags = gr_accuracy_from_flags


def compute_f1_gr(pred_answered_list: List[bool], gold_answerable_list: List[bool]) -> Dict[str, float]:
    """
    Compute proper precision/recall/F1 for the grounded-refusal task across
    the whole dataset.

    Treats `answered` as the positive class:
      • TP = pred answered & gold answerable
      • FP = pred answered & gold NOT answerable
      • FN = pred refused  & gold answerable
      • TN = pred refused  & gold NOT answerable
    """
    if len(pred_answered_list) != len(gold_answerable_list):
        raise ValueError("List lengths must match")

    tp = sum(1 for p, g in zip(pred_answered_list, gold_answerable_list) if p and g)
    fp = sum(1 for p, g in zip(pred_answered_list, gold_answerable_list) if p and not g)
    fn = sum(1 for p, g in zip(pred_answered_list, gold_answerable_list) if not p and g)
    tn = sum(1 for p, g in zip(pred_answered_list, gold_answerable_list) if not p and not g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(pred_answered_list))

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
    }


# --------------------
# Text normalization helpers
# --------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import string
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def remove_citations(text: str) -> str:
    """Remove [N] and [dN] citation markers and collapse whitespace."""
    text = re.sub(r"\[\d+\]", "", text or "")
    text = re.sub(r"\[d\d+\]", "", text)
    return " ".join(text.split())
