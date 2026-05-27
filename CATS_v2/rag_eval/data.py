# data.py
# -*- coding: utf-8 -*-
"""
Dataset utilities for RAG Mixed Evaluation Toolkit
--------------------------------------------------

Expected record schema (per JSONL line):
{
  "id": "ex_0001",
  "query": "who is commander chief of the military",
  "retrieved_docs": [
    {"doc_id": "d1", "title": "...", "url": "...", "snippet": "...", "date": "..."},
    ...
  ],
  "per_doc_notes": [
    {"doc_id": "d1", "verdict": "supports", "key_fact": "...", "quote": "..."},
    {"doc_id": "d2", "verdict": "irrelevant", "key_fact": "", "quote": ""}
  ],
  "conflict_category_id": 1,
  "conflict_type": "No Conflict",
  "conflict_reason": "All sources agree...",
  "final_grounded_answer": {                  # gold annotation; never used as model_output
    "style_hint": "...",
    "answer": "...",
    "evidence": ["d1","d2"],
    "abstain": false
  },
  "model_output": "...",                      # required for evaluation
  "gold_answer": "President of Nigeria",      # optional, for single-truth recall
  "trace_type": "summarized"
}
"""

import json
from typing import Dict, Any, List, Iterator, Optional


# Verdicts that count as "this doc supports the answer".
# Normalized by lowercasing + replacing underscores; matched as substring on the
# `partial_*` variants so "partially supports", "partial support", "partial_supports"
# all count when accept_partial=True.
_POSITIVE_VERDICTS = {"supports", "support"}
_PARTIAL_TOKENS = ("partial", "weakly support", "weak support")


def _verdict_is_positive(verdict_raw: Optional[str], accept_partial: bool) -> bool:
    v = (verdict_raw or "").strip().lower().replace("_", " ")
    if v in _POSITIVE_VERDICTS:
        return True
    if accept_partial and any(tok in v for tok in _PARTIAL_TOKENS):
        return True
    return False


# -------------------------
# I/O helpers
# -------------------------

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield dataset records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Record-level utilities
# -------------------------

def doc_index_from_record(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for d in record.get("retrieved_docs", []) or []:
        if "doc_id" in d:
            idx[d["doc_id"]] = d
    return idx


def support_doc_ids_from_notes(per_doc_notes: List[Dict[str, Any]], accept_partial: bool = True) -> List[str]:
    """
    Extract doc_ids whose verdict counts as supporting the answer.
    Normalizes verdict text so "Supports", "partially_supports",
    "partial support", "weakly supports" all match when accept_partial=True.
    """
    out: List[str] = []
    seen: set = set()
    for n in per_doc_notes or []:
        if not _verdict_is_positive(n.get("verdict"), accept_partial):
            continue
        did = n.get("doc_id")
        if did and did not in seen:
            out.append(did)
            seen.add(did)
    return out


def gold_answerable_from_notes(per_doc_notes: List[Dict[str, Any]], accept_partial: bool = True) -> bool:
    return len(support_doc_ids_from_notes(per_doc_notes, accept_partial)) > 0


def gold_answerable_from_record(record: Dict[str, Any], accept_partial: bool = True) -> bool:
    """
    Authoritative gold_answerable for a record.

    Prefers the explicit `answerable_under_evidence` field when present — this
    is the annotator's direct verdict and overrides notes-derived inference.
    Falls back to `gold_answerable_from_notes` for records without that field
    (backwards-compatible with the old dataset schema).

    Why this matters: val-split records have `partially supports` verdicts even
    on unanswerable samples (evidence is partial but insufficient). With
    `accept_partial=True`, `gold_answerable_from_notes` returns True for those
    samples, misclassifying correct refusals as wrong-refusals and scoring
    GR=0 instead of GR=1.
    """
    aue = record.get("answerable_under_evidence")
    if aue is not None:
        return bool(aue)
    notes = record.get("per_doc_notes") or []
    return gold_answerable_from_notes(notes, accept_partial=accept_partial)


class MissingModelOutputError(KeyError):
    """Raised when a record has no usable model_output."""
    pass


def get_model_output(record: Dict[str, Any], strict: bool = False) -> str:
    """
    Extract the model's answer text.

    Priority order:
      1. `model_output` — the explicit model output field (standard schema).
      2. `expected_response.answer` — val/gold dataset support.
         For gold evaluation datasets (e.g. stage3_final.jsonl) the pipeline
         stores the expected model answer in `expected_response.answer`.
         The companion `think` field is the stage-1 annotator reasoning, NOT
         the model's thinking trace, so it is intentionally ignored here.
         `expected_response.answer` may contain "CANNOT ANSWER, INSUFFICIENT
         EVIDENCE" for unanswerable samples — this is handled correctly by
         `answered_flags` / `strip_think_trace` downstream.

    `final_grounded_answer.answer` is the GOLD annotation, not the model's
    output — silently falling back to it would score the gold against itself,
    so it is never used here (§5.3 fix, preserved).

    By default, return "" when no usable output field is found (so the sample
    is treated as a refusal). Pass strict=True to raise instead.
    """
    if "model_output" in record:
        val = record["model_output"]
        if val is None:
            val = ""
        return str(val)

    # Val/gold dataset fallback: expected_response is a dict with an 'answer' key.
    er = record.get("expected_response")
    if isinstance(er, dict) and "answer" in er:
        val = er.get("answer") or ""
        return str(val)

    if strict:
        raise MissingModelOutputError(
            f"Record {record.get('id', '<no id>')} has no model_output or "
            "expected_response.answer field."
        )

    # Lenient mode: treat as refusal but DO NOT use any gold annotation field.
    return ""


def get_gold_answer(record: Dict[str, Any]) -> Optional[str]:
    return record.get("gold_answer")


# -------------------------
# Batch utilities
# -------------------------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def dataset_size(path: str) -> int:
    return sum(1 for _ in read_jsonl(path))
