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
# Refusal detection is intentionally start-oriented rather than substring-based.
# On the benchmark outputs, true refusals overwhelmingly begin with a canonical
# abstention phrase ("CANNOT ANSWER, INSUFFICIENT EVIDENCE", "Cannot answer",
# etc.). A broad substring search would incorrectly label substantive answers
# that merely contain phrases like "insufficient evidence" in the middle of an
# otherwise grounded explanation.
_REFUSAL_AT_START_RE = re.compile(
    r"^\s*(?:"
    r"cannot\s+answer(?:\b|[,:.;-])"
    r"|can'?t\s+answer(?:\b|[,:.;-])"
    r"|cant\s+answer(?:\b|[,:.;-])"
    r"|i\s+(?:cannot|can'?t|cant)\s+answer\b"
    r"|i(?:'m|\s+am)?\s+unable\s+to\s+(?:answer|determine|conclude)\b"
    r"|unable\s+to\s+(?:answer|determine|conclude)\b"
    r"|insufficient\s+(?:evidence|information|context)\b"
    r"|not\s+enough\s+(?:evidence|information|context)\b"
    r"|no\s+(?:reliable\s+)?(?:evidence|information)\s+to\s+answer\b"
    r"|(?:do\s+not|don'?t|does\s+not|doesn'?t)\s+have\s+(?:enough|sufficient)\s+information\b"
    r"|cannot\s+(?:determine|conclude|establish)\b"
    r"|cannot\s+be\s+determined\b"
    r")",
    re.IGNORECASE,
)

_REFUSAL_WRAPPED_START_RE = re.compile(
    r"^\s*(?:based\s+on|from)\s+the\s+"
    r"(?:retrieved\s+|provided\s+)?(?:documents?|evidence|information|context)\b"
    r".{0,200}\b(?:"
    r"i\s+(?:cannot|can'?t|cant)\s+(?:answer|determine)"
    r"|i(?:'m|\s+am)?\s+unable\s+to\s+(?:answer|determine|conclude)"
    r"|unable\s+to\s+(?:answer|determine|conclude)"
    r"|not\s+enough\s+(?:evidence|information|context)"
    r"|insufficient\s+(?:evidence|information|context)"
    r"|cannot\s+be\s+determined"
    r"|there\s+is\s+no\s+information"
    r")\b",
    re.IGNORECASE,
)

_END_SENTINEL_RE = re.compile(
    r"(?:\[\[\s*END-OF-ANSWER\s*\]\]|\[\s*END-OF-ANSWER\s*\]|\bEND-OF-ANSWER\b)",
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
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        head = lines[0] if lines else text
        is_refusal = bool(_REFUSAL_AT_START_RE.match(head) or _REFUSAL_WRAPPED_START_RE.match(head))
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
    r"(all|these|those|both|they|this|the\s+(documents|sources|references|citations))\b"
    r".{0,30}?\b(state|say|report|confirm|show|indicate|provide|mention|explicitly|note|claim|support|supported)\b"
    r"|"
    r"(state|states|stated|say|says|report|reports|reported|confirm|confirms|confirmed"
    r"|show|shows|showed|indicate|indicates|indicated|provide|provides|provided"
    r"|mention|mentions|mentioned|note|notes|noted|claim|claims|claimed|support|supports|supported)\b"
    r")",
    re.IGNORECASE,
)

_TRAILING_ATTRIBUTION_STUB_RE = re.compile(
    r"(?:,\s*as\s+stated\s+in(?:\s+multiple\s+reliable\s+sources)?(?:,\s*including)?"
    r"|,\s*according\s+to"
    r"|,\s*including"
    r"|\s+including"
    r"|,\s*as\s+reported\s+by"
    r"|,\s*as\s+confirmed\s+by)\s*$",
    re.IGNORECASE,
)

_BARE_DOC_REF_RE = re.compile(
    r"\b(?:according\s+to|including|include|includes|documents?|docs?|sources?)\s+"
    r"(d\d+(?:\s*,\s*d\d+)*(?:\s*,?\s*and\s*d\d+)?)",
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
    text = re.sub(r"\[\[\s*(?:d\d+\s*(?:[,;]\s*)?)+\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*(?:d\d+\s*(?:[,;]\s*)?)+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\(\s*d\d+\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:[^)]*?\bd\d+\b[^)]*?)\)", "", text, flags=re.IGNORECASE)

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
    # Citation phrases like "according to d1, d2, and d3" or
    # "including d1, d2, and d3"
    text = re.sub(
        r"\b(?:according\s+to|including|include|includes|documents?|docs?|sources?)\s+"
        r"d\d+(?:\s*,\s*d\d+)*(?:\s*,?\s*and\s*d\d+)?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse comma-and-comma debris
    text = text.replace("[][][]", "").replace("[][]", "")
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
    text = re.sub(r"^\s*[,\s]+", "", text)
    text = re.sub(r"\s*,\s*(and|or)\b\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(and|or)\s+", "", text, flags=re.IGNORECASE)
    text = _TRAILING_ATTRIBUTION_STUB_RE.sub("", text)
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


def _protect_sentence_internal_periods(text: str) -> str:
    """Protect periods that should not trigger sentence splitting."""
    # Multi-initial abbreviations / acronyms: U.S., U.K., e.g., i.e.
    text = re.sub(
        r"\b((?:[A-Za-z]\.){2,})(?=\s|[A-Za-z])",
        lambda m: m.group(1).replace(".", "<DOT>"),
        text,
    )

    # Single initials before a following word: "Lyndon B. Johnson", "H. pylori"
    text = re.sub(r"\b([A-Z])\.(?=\s+[A-Za-z])", r"\1<DOT>", text)

    # Decimal numbers (and percentages/currency): "1.8", "17.83", "$2.5"
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)

    # Domain extensions and any "a.UPPERCASE" pattern: ".COM", ".NET", "a.com"
    text = re.sub(r"\b([a-zA-Z])\.([A-Za-z]{2,4})\b", r"\1<DOT>\2", text)

    # Common abbreviations that end with `.` and should not split.
    _ABBREV = (
        "Inc", "Ltd", "Co", "Corp", "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr", "Jr",
        "vs", "etc", "Fig", "Vol", "No", "Ave", "Blvd", "Mt",
    )
    for ab in _ABBREV:
        text = re.sub(rf"\b({ab})\.", rf"\1<DOT>", text)

    return text


def _sentence_tokenize_safe(text: str) -> List[str]:
    """Sentence-tokenize with a non-naive fallback when NLTK punkt data is unavailable."""
    try:
        return sent_tokenize(text)
    except Exception:
        # `text` is expected to already have internal periods protected as <DOT>.
        parts = re.split(r'(?<=[.!?])\s+(?=(?:["\'(\[])?[A-Z0-9])', text)
        return [p for p in parts if p.strip()]


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

    text = _protect_sentence_internal_periods(text)

    sents = _sentence_tokenize_safe(text)

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
    """Remove think-trace and answer-end sentinels before downstream scoring."""
    cleaned = text or ""
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    cleaned = _END_SENTINEL_RE.sub("", cleaned)
    return cleaned.strip()


# Capture repeated single-doc citations like [d1][d2] and bundled citations
# like [d1, d2, d3] or [d1; d2].
_CITE_EXTRACT = re.compile(r"\[\s*((?:d\d+\s*(?:[,;]\s*)?)*)\]", re.IGNORECASE)
_DOC_ID_IN_CITE = re.compile(r"d\d+", re.IGNORECASE)
_PAREN_DOC_REF_RE = re.compile(r"\((?:[^)]*?\bd\d+\b[^)]*?)\)", re.IGNORECASE)


def _should_inherit_neighbor_citations(text: str) -> bool:
    """Heuristic for short lead-summary claims that standard answer style leaves uncited."""
    words = text.split()
    if not text or len(words) > 24:
        return False
    lowered = text.lower().strip()
    starters = (
        "yes,",
        "no,",
        "research on whether",
        "whether ",
        "opinions differ",
        "the phrase ",
        "there is no single",
        "the prevailing ",
        "the dominant ",
        "the last person ",
    )
    return lowered.startswith(starters)


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
    text = _protect_sentence_internal_periods(text)

    sents = _sentence_tokenize_safe(text)

    out: List[Dict] = []
    fallback_candidates: List[Dict] = []
    for s in sents:
        s = s.replace("<DOT>", ".").strip()
        if not s:
            continue

        # Extract citations BEFORE stripping them
        cited_docs: List[str] = []
        for block in _CITE_EXTRACT.findall(s):
            cited_docs.extend(d.lower() for d in _DOC_ID_IN_CITE.findall(block))
        for block in _PAREN_DOC_REF_RE.findall(s):
            cited_docs.extend(d.lower() for d in _DOC_ID_IN_CITE.findall(block))
        for block in _BARE_DOC_REF_RE.findall(s):
            cited_docs.extend(d.lower() for d in _DOC_ID_IN_CITE.findall(block))
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
        fallback_candidates.append({"text": stripped, "cited_docs": unique_cites})
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

    # Standard answer style often states the short conclusion first and puts the
    # supporting citations in the immediately following sentence. Inherit those
    # citations for concise lead-summary claims so FG does not penalize
    # answer-first formatting when the evidence follows directly after.
    for i, item in enumerate(out[:-1]):
        if item["cited_docs"]:
            continue
        if not _should_inherit_neighbor_citations(item["text"]):
            continue
        nxt = out[i + 1]
        if nxt.get("cited_docs"):
            item["cited_docs"] = list(nxt["cited_docs"])

    # Terse-answer fallback:
    # Some benchmark answers are intentionally minimal, e.g. "Paris [d1]",
    # "78 [d3]", or "Yes [d1]". The normal extractor drops them because they
    # are shorter than four words, which silently forces FG=0.0 despite a
    # potentially well-cited correct answer. If no standard claim survived,
    # promote short cleaned candidates deterministically.
    if not out and fallback_candidates:
        for item in fallback_candidates[:max_claims]:
            text = (item.get("text") or "").strip()
            cited_docs = list(item.get("cited_docs") or [])
            if not text:
                continue
            tokens = text.split()
            if len(tokens) > 8:
                continue
            if not cited_docs and len(tokens) > 3:
                continue
            out.append({"text": text, "cited_docs": cited_docs})
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
    citations: List[str] = []
    for block in _CITE_EXTRACT.findall(answer_text):
        citations.extend(d.lower() for d in _DOC_ID_IN_CITE.findall(block))
    for block in _PAREN_DOC_REF_RE.findall(answer_text):
        citations.extend(d.lower() for d in _DOC_ID_IN_CITE.findall(block))
    for block in _BARE_DOC_REF_RE.findall(answer_text):
        citations.extend(d.lower() for d in _DOC_ID_IN_CITE.findall(block))
    seen: set = set()
    out: List[str] = []
    for c in citations:
        if c not in seen:
            out.append(c)
            seen.add(c)
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

    # Abstain-oriented diagnostics over the same confusion table.
    # This mirrors the "refusal is the positive class" framing used in the
    # older final-answer evaluator without changing the CATS GR component,
    # which still uses answered-positive F1 for continuity.
    abstain_tp = tn
    abstain_fp = fn
    abstain_fn = fp
    abstain_tn = tp
    abstain_precision = abstain_tp / (abstain_tp + abstain_fp) if (abstain_tp + abstain_fp) > 0 else 0.0
    abstain_recall = abstain_tp / (abstain_tp + abstain_fn) if (abstain_tp + abstain_fn) > 0 else 0.0
    abstain_f1 = (
        2 * abstain_precision * abstain_recall / (abstain_precision + abstain_recall)
        if (abstain_precision + abstain_recall) > 0
        else 0.0
    )
    abstain_specificity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "abstain_tp": abstain_tp,
        "abstain_fp": abstain_fp,
        "abstain_fn": abstain_fn,
        "abstain_tn": abstain_tn,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "abstain_f1": abstain_f1,
        "abstain_specificity": abstain_specificity,
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
