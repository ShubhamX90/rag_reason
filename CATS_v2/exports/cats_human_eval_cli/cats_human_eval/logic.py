from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from nltk import sent_tokenize
except Exception:  # pragma: no cover - optional runtime dependency behavior
    sent_tokenize = None


_POSITIVE_VERDICTS = {"supports", "support"}
_PARTIAL_TOKENS = ("partial", "weakly support", "weak support")

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

_CITATION_ONLY = re.compile(
    r"^\s*[\[\(\s,]*(?:d\d+|\[\d+\])(?:[\s,]*(?:d\d+|\[\d+\]))*[\s\.\]\)]*$",
    re.IGNORECASE,
)
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
_ATTR_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:According|As\s+(?:reported|stated|noted|confirmed))\s+(?:to|by)\s+[^,]{4,80}?,\s*"
    r"|[A-Z][^.]{8,120}?\s+(?:both|also)\s+(?:confirm|confirms|show|shows|report|reports|state|states|note|notes|indicate|indicates)\s+(?:that\s+)?"
    r"|(?:Multiple|Several|Both|All|Many)\s+sources?\s+(?:confirm|confirms|show|shows|report|reports|indicate|indicates|agree|corroborate|concur)\s+(?:that\s+)?"
    r"|[Aa]s\s+of\s+[^,]{4,60},\s*"
    r")",
    re.DOTALL,
)
_TRAILING_META_RE = re.compile(
    r",\s*(?:unanimously|widely|consistently|independently)?\s*"
    r"(?:confirmed|verified|established|documented|supported|agreed)"
    r"(?:\s+(?:across|by|among|in)\s+[^.]*)?\.?$",
    re.IGNORECASE,
)
_CITE_EXTRACT = re.compile(r"\[\s*((?:d\d+\s*(?:[,;]\s*)?)*)\]", re.IGNORECASE)
_DOC_ID_IN_CITE = re.compile(r"d\d+", re.IGNORECASE)
_PAREN_DOC_REF_RE = re.compile(r"\((?:[^)]*?\bd\d+\b[^)]*?)\)", re.IGNORECASE)


def _verdict_is_positive(verdict_raw: Optional[str], accept_partial: bool) -> bool:
    verdict = (verdict_raw or "").strip().lower().replace("_", " ")
    if verdict in _POSITIVE_VERDICTS:
        return True
    if accept_partial and any(tok in verdict for tok in _PARTIAL_TOKENS):
        return True
    return False


def support_doc_ids_from_notes(
    per_doc_notes: List[Dict[str, Any]],
    accept_partial: bool = True,
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for note in per_doc_notes or []:
        if not _verdict_is_positive(note.get("verdict"), accept_partial):
            continue
        doc_id = note.get("doc_id")
        if doc_id and doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)
    return out


def gold_answerable_from_record(
    record: Dict[str, Any],
    accept_partial: bool = True,
) -> bool:
    expected = record.get("expected_response")
    if isinstance(expected, dict):
        abstain = expected.get("abstain")
        if isinstance(abstain, bool):
            return not abstain
    answerable = record.get("answerable_under_evidence")
    if answerable is not None:
        return bool(answerable)
    return len(support_doc_ids_from_notes(record.get("per_doc_notes") or [], accept_partial)) > 0


def get_model_output(record: Dict[str, Any]) -> str:
    if "model_output" in record and record["model_output"] is not None:
        return str(record["model_output"])
    expected = record.get("expected_response")
    if isinstance(expected, dict) and "answer" in expected:
        return str(expected.get("answer") or "")
    return ""


def strip_think_trace(text: str) -> str:
    cleaned = text or ""
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    cleaned = _END_SENTINEL_RE.sub("", cleaned)
    return cleaned.strip()


def split_think_trace(text: str) -> Tuple[str, str]:
    raw = text or ""
    if "</think>" in raw and "<think>" in raw:
        think_block, rest = raw.split("</think>", 1)
        think = think_block.split("<think>", 1)[-1].strip()
        answer = _END_SENTINEL_RE.sub("", rest).strip()
        return think, answer
    answer = _END_SENTINEL_RE.sub("", raw).strip()
    return "", answer


def answered_flags(outputs: List[str]) -> List[bool]:
    flags: List[bool] = []
    for output in outputs:
        text = (output or "").strip()
        if not text:
            flags.append(False)
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        head = lines[0] if lines else text
        is_refusal = bool(_REFUSAL_AT_START_RE.match(head) or _REFUSAL_WRAPPED_START_RE.match(head))
        flags.append(not is_refusal)
    return flags


def _strip_citations_inplace(text: str) -> str:
    text = re.sub(r"\[\[\s*(?:d\d+\s*(?:[,;]\s*)?)+\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*(?:d\d+\s*(?:[,;]\s*)?)+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\(\s*d\d+\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:[^)]*?\bd\d+\b[^)]*?)\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*d\d+(?:\s*,\s*(?:and\s+)?d\d+)+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:,\s*)?d\d+(?:\s*,\s*(?:and\s+)?d\d+){2,}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:according\s+to|including|include|includes|documents?|docs?|sources?)\s+"
        r"d\d+(?:\s*,\s*d\d+)*(?:\s*,?\s*and\s*d\d+)?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("[][][]", "").replace("[][]", "")
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
    text = re.sub(r"^\s*[,\s]+", "", text)
    text = re.sub(r"\s*,\s*(and|or)\b\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(and|or)\s+", "", text, flags=re.IGNORECASE)
    text = _TRAILING_ATTRIBUTION_STUB_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;:")


def _strip_attribution(claim: str) -> str:
    text = claim
    match = _ATTR_PREFIX_RE.match(text)
    if match and match.end() < len(text) - 1:
        residual = text[match.end():].strip(" ,")
        if len(residual.split()) >= 4:
            text = residual[0].upper() + residual[1:] if residual else residual
    stripped = _TRAILING_META_RE.sub("", text).strip(" .")
    if len(stripped.split()) >= 4:
        text = stripped
    return text


def _protect_sentence_internal_periods(text: str) -> str:
    text = re.sub(
        r"\b((?:[A-Za-z]\.){2,})(?=\s|[A-Za-z])",
        lambda match: match.group(1).replace(".", "<DOT>"),
        text,
    )
    text = re.sub(r"\b([A-Z])\.(?=\s+[A-Za-z])", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    text = re.sub(r"\b([a-zA-Z])\.([A-Za-z]{2,4})\b", r"\1<DOT>\2", text)
    abbreviations = (
        "Inc", "Ltd", "Co", "Corp", "Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sr",
        "Jr", "vs", "etc", "Fig", "Vol", "No", "Ave", "Blvd", "Mt",
    )
    for abbr in abbreviations:
        text = re.sub(rf"\b({abbr})\.", rf"\1<DOT>", text)
    return text


def _sentence_tokenize_safe(text: str) -> List[str]:
    if sent_tokenize is not None:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    parts = re.split(r'(?<=[.!?])\s+(?=(?:["\'(\[])?[A-Z0-9])', text)
    return [part for part in parts if part.strip()]


def _should_inherit_neighbor_citations(text: str) -> bool:
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


def extract_claims_with_citations(answer_text: str, max_claims: int = 12) -> List[Dict[str, Any]]:
    if not answer_text:
        return []
    text = _protect_sentence_internal_periods(answer_text)
    sentences = _sentence_tokenize_safe(text)
    out: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    for sentence in sentences:
        sentence = sentence.replace("<DOT>", ".").strip()
        if not sentence:
            continue
        cited_docs: List[str] = []
        for block in _CITE_EXTRACT.findall(sentence):
            cited_docs.extend(doc.lower() for doc in _DOC_ID_IN_CITE.findall(block))
        for block in _PAREN_DOC_REF_RE.findall(sentence):
            cited_docs.extend(doc.lower() for doc in _DOC_ID_IN_CITE.findall(block))
        for block in _BARE_DOC_REF_RE.findall(sentence):
            cited_docs.extend(doc.lower() for doc in _DOC_ID_IN_CITE.findall(block))
        seen_cites: set[str] = set()
        unique_cites: List[str] = []
        for doc in cited_docs:
            if doc not in seen_cites:
                unique_cites.append(doc)
                seen_cites.add(doc)
        stripped = _strip_citations_inplace(sentence)
        if not stripped or _CITATION_ONLY.match(stripped) or _META_REFERENCE.match(stripped):
            continue
        fallback_candidates.append({"text": stripped, "cited_docs": unique_cites})
        words = stripped.split()
        if len(words) < 4:
            continue
        if not any(len(word.strip(".,;:!?")) >= 5 for word in words):
            continue
        stripped = _strip_attribution(stripped)
        if not stripped or len(stripped.split()) < 4:
            continue
        out.append({"text": stripped, "cited_docs": unique_cites})
        if len(out) >= max_claims:
            break
    for idx, item in enumerate(out[:-1]):
        if item["cited_docs"]:
            continue
        if not _should_inherit_neighbor_citations(item["text"]):
            continue
        nxt = out[idx + 1]
        if nxt.get("cited_docs"):
            item["cited_docs"] = list(nxt["cited_docs"])
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


def merge_docs_with_notes(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    notes = record.get("per_doc_notes") or []
    notes_by_id = {note.get("doc_id"): note for note in notes if note.get("doc_id")}
    merged: List[Dict[str, Any]] = []
    for doc in record.get("retrieved_docs", []) or []:
        doc_id = doc.get("doc_id", "")
        note = notes_by_id.get(doc_id) or {}
        merged.append({
            "doc_id": doc_id,
            "title": doc.get("title") or "",
            "snippet": (doc.get("snippet") or "").strip(),
            "source_url": doc.get("source_url") or doc.get("url") or "",
            "timestamp": doc.get("timestamp") or doc.get("date") or "",
            "verdict": (note.get("verdict") or "").strip(),
            "key_fact": (note.get("key_fact") or "").strip(),
            "quote": (note.get("quote") or "").strip(),
        })
    return merged


def eligible_fg_docs(merged_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for doc in merged_docs:
        if _verdict_is_positive(doc.get("verdict"), accept_partial=True):
            out.append(doc)
    return out


def compute_fg_claim_support(
    cited_docs: List[str],
    supporting_docs_found: List[str],
    cross_doc_support: bool,
    cross_doc_combo: List[str],
) -> Tuple[bool, str]:
    cited = set(cited_docs or [])
    supporting = set(supporting_docs_found or [])
    combo = set(cross_doc_combo or [])
    if supporting and (cited & supporting):
        return True, "single_doc_cited"
    if cross_doc_support and combo and (cited & combo):
        return True, "cross_doc_cited"
    if not supporting and not cross_doc_support:
        return False, "no_supporting_doc_found"
    if supporting and not (cited & supporting):
        return False, "supporting_doc_not_cited"
    if cross_doc_support and combo and not (cited & combo):
        return False, "cross_doc_not_cited"
    return False, "not_supported"

