"""
src/parsers.py
==============
Robust output parsers for each retained annotation stage.

Stage 1  → JSON object per doc
Stage 2  → JSON object {conflict_reason, answerable_under_evidence}
Stage 3  → JSON object {expected_response, think}
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
#  Low-level JSON extraction helpers
# ─────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_ABSTAIN  = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).replace("```", "").strip()


def _fix_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def _fix_json_control_chars(text: str) -> str:
    """Escape literal control characters (newline, tab, etc.) inside JSON string values.

    Models sometimes output JSON with unescaped newlines inside strings,
    e.g. {"think": "<think>\n...\n</think>"} with actual newline bytes.
    This is invalid JSON. We fix it by scanning for strings and escaping
    control chars only within them, leaving structural whitespace intact.
    """
    result = []
    in_str = False
    esc = False
    for ch in text:
        if esc:
            result.append(ch)
            esc = False
            continue
        if ch == '\\' and in_str:
            result.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            result.append(ch)
            continue
        if in_str:
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract the first complete JSON object from text.
    Handles markdown fences, leading prose, and trailing commas.
    """
    if not text:
        return None
    cleaned = _strip_fences(text)

    # Direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first balanced { … }
    start = cleaned.find("{")
    if start == -1:
        return None

    depth, end = 0, -1
    in_str, esc = False, False
    for i, ch in enumerate(cleaned[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    frag = cleaned[start:end + 1]
    for attempt in (frag, _fix_trailing_commas(frag)):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Last resort: fix literal control characters inside JSON strings
    fixed = _fix_json_control_chars(cleaned)
    try:
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _iter_json_objects(text: str) -> List[Dict[str, Any]]:
    """Return all balanced JSON objects found in text, in order."""
    if not text:
        return []
    cleaned = _strip_fences(text)
    objects: List[Dict[str, Any]] = []
    i = 0
    while i < len(cleaned):
        start = cleaned.find("{", i)
        if start == -1:
            break

        depth, end = 0, -1
        in_str, esc = False, False
        for j, ch in enumerate(cleaned[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break

        if end == -1:
            break

        frag = cleaned[start:end + 1]
        parsed = None
        for attempt in (frag, _fix_trailing_commas(frag), _fix_json_control_chars(frag)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    parsed = obj
                    break
            except Exception:
                pass
        if parsed is not None:
            objects.append(parsed)
        i = max(end + 1, start + 1)
    return objects


def _extract_stage3_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the Stage-3 JSON object, preferring expected_response payloads."""
    first = _extract_json_object(text)
    if first is not None and isinstance(first.get("expected_response"), dict):
        return first
    for obj in _iter_json_objects(text):
        if isinstance(obj.get("expected_response"), dict):
            return obj
    return first


# ─────────────────────────────────────────────
#  Stage 1 parser
# ─────────────────────────────────────────────

STAGE1_REQUIRED_FIELDS = {"doc_id", "verdict", "key_fact", "quote", "verdict_reason", "source_quality"}
STAGE1_VALID_VERDICTS  = {"supports", "partially supports", "irrelevant"}
STAGE1_VALID_QUALITY   = {"high", "low"}


def parse_stage1(raw: str, fallback_doc_id: str = "") -> Tuple[Dict[str, Any], List[str]]:
    """
    Parse Stage-1 LLM output.

    Returns
    -------
    (record, errors)
    record : the parsed (and sanitised) JSON object
    errors : list of validation error strings (empty = OK)
    """
    obj = _extract_json_object(raw)
    if obj is None:
        return _stage1_fallback(fallback_doc_id, f"could not parse JSON from: {raw[:120]}"), \
               [f"JSON parse failure"]

    errors = _validate_stage1(obj)
    # Patch missing doc_id from context
    if not obj.get("doc_id") and fallback_doc_id:
        obj["doc_id"] = fallback_doc_id
    return obj, errors


def _validate_stage1(obj: Dict) -> List[str]:
    errs: List[str] = []
    for field in STAGE1_REQUIRED_FIELDS:
        if field not in obj:
            errs.append(f"missing field: {field}")
    verdict = obj.get("verdict", "")
    if verdict not in STAGE1_VALID_VERDICTS:
        errs.append(f"invalid verdict: {verdict!r}")
    quality = obj.get("source_quality", "")
    if quality not in STAGE1_VALID_QUALITY:
        errs.append(f"invalid source_quality: {quality!r}")
    if verdict == "irrelevant":
        if obj.get("key_fact") or obj.get("quote"):
            errs.append("irrelevant record must have empty key_fact and quote")
    else:
        if not obj.get("key_fact"):
            errs.append("missing key_fact for non-irrelevant verdict")
        if not obj.get("quote"):
            errs.append("missing quote for non-irrelevant verdict")
    return errs


def _stage1_fallback(doc_id: str, reason: str) -> Dict[str, Any]:
    return {
        "doc_id":         doc_id,
        "verdict":        "irrelevant",
        "key_fact":       "",
        "quote":          "",
        "verdict_reason": f"Fallback: {reason}"[:120],
        "source_quality": "low",
        "_parse_error":   True,
    }


# ─────────────────────────────────────────────
#  Stage 2 parser
# ─────────────────────────────────────────────

def parse_stage2(raw: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Parse Stage-2 LLM output.

    Returns
    -------
    (record, errors)
    record : {conflict_reason: str, answerable_under_evidence: bool}
    errors : list of validation error strings
    """
    obj = _extract_json_object(raw)
    if obj is None:
        return {
            "conflict_reason": "JSON parse failure.",
            "answerable_under_evidence": False,
            "_parse_error": True,
        }, ["JSON parse failure"]

    errs: List[str] = []
    if "conflict_reason" not in obj:
        errs.append("missing field: conflict_reason")
    if "answerable_under_evidence" not in obj:
        errs.append("missing field: answerable_under_evidence")
    elif not isinstance(obj["answerable_under_evidence"], bool):
        # Coerce string "true"/"false"
        val = str(obj["answerable_under_evidence"]).lower()
        obj["answerable_under_evidence"] = val == "true"

    return obj, errs


# ─────────────────────────────────────────────
#  Stage 2 refusal parser
# ─────────────────────────────────────────────

# Valid conflict_type labels accepted across the repo.
# Some historical artifacts use "or research outcomes" while others use
# "and research outcomes", so we accept both.
VALID_CONFLICT_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflicting opinions and research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}


def parse_stage2_refusal(raw: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Parse Stage-2 LLM output for the REFUSALS dataset, where the model
    independently determines conflict_type rather than being given it.

    Expected model output schema:
    {
      "conflict_type": "one of the five valid labels",
      "conflict_reason": "≤50 words, references doc IDs",
      "answerable_under_evidence": true | false
    }

    Returns
    -------
    (record, errors)
    record : {conflict_type: str, conflict_reason: str, answerable_under_evidence: bool}
    errors : list of validation error strings (empty = OK)
    """
    obj = _extract_json_object(raw)
    if obj is None:
        return {
            "conflict_type":             "",
            "conflict_reason":           "JSON parse failure.",
            "answerable_under_evidence": False,
            "_parse_error":              True,
        }, ["JSON parse failure"]

    errs: List[str] = []

    # ── conflict_type ──
    ct = obj.get("conflict_type", "")
    if not ct:
        errs.append("missing field: conflict_type")
    elif ct not in VALID_CONFLICT_TYPES:
        errs.append(f"invalid conflict_type: {ct!r}")

    # ── conflict_reason ──
    if "conflict_reason" not in obj:
        errs.append("missing field: conflict_reason")

    # ── answerable_under_evidence ──
    if "answerable_under_evidence" not in obj:
        errs.append("missing field: answerable_under_evidence")
    elif not isinstance(obj["answerable_under_evidence"], bool):
        val = str(obj["answerable_under_evidence"]).lower()
        obj["answerable_under_evidence"] = val == "true"

    return obj, errs


# ─────────────────────────────────────────────
#  Stage 3 parser
# ─────────────────────────────────────────────

_STAGE3_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def parse_stage3(raw: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Parse Stage-3 LLM output.

    Handles two common model output formats:
      A) Single JSON with "think" as an embedded string (e.g. Qwen 72b):
            {"expected_response": {...}, "think": "<think>...</think>"}
      B) Raw <think> block followed by JSON (e.g. Sonnet 4.6, DeepSeek):
            <think>...</think>
            {"expected_response": {...}, "think": "..."}

    Returns
    -------
    (record, errors)
    record : {expected_response: {...}, think: str}
    errors : list of validation error strings
    """
    # ── Attempt 1: direct JSON extraction (works for format A) ──
    obj = _extract_stage3_json_object(raw)
    if obj is not None and isinstance(obj.get("expected_response"), dict):
        # Successfully parsed — ensure think field is populated
        if not obj.get("think"):
            m = _STAGE3_THINK_RE.search(raw)
            if m:
                obj["think"] = f"<think>{m.group(1)}</think>"
        return obj, []

    # ── Attempt 2: strip leading <think>…</think> then re-extract ──
    think_content = ""
    think_match = _STAGE3_THINK_RE.search(raw)
    if think_match:
        think_content = think_match.group(1)
        # Remove the <think>…</think> block so _extract_json_object
        # finds the actual response JSON instead of a doc-reasoning object.
        remainder = raw[think_match.end():]
        obj = _extract_stage3_json_object(remainder)
        if obj is not None and isinstance(obj.get("expected_response"), dict):
            # Re-attach think content
            obj["think"] = f"<think>{think_content}</think>"
            return obj, []

    # ── Attempt 3: last resort — try original extraction but accept it ──
    if obj is None:
        obj = _extract_stage3_json_object(raw)

    if obj is None:
        return {
            "expected_response": {
                "answer":        _ABSTAIN,
                "evidence":      [],
                "abstain":       True,
                "abstain_reason": "JSON parse failure.",
            },
            "think": f"<think>{think_content}</think>" if think_content else "",
            "_parse_error": True,
        }, ["JSON parse failure"]

    errs: List[str] = []
    er = obj.get("expected_response")
    if not isinstance(er, dict):
        errs.append("expected_response is not a dict")
        obj["expected_response"] = {
            "answer": _ABSTAIN, "evidence": [], "abstain": True,
            "abstain_reason": "Malformed response.",
        }
    # Preserve any think content we found
    if think_content and not obj.get("think"):
        obj["think"] = f"<think>{think_content}</think>"

    return obj, errs
