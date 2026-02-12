# data.py
# -*- coding: utf-8 -*-
"""
Dataset utilities for RAG Mixed Evaluation Toolkit
--------------------------------------------------

This module provides helper functions for working with annotated
RAG evaluation datasets.

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
  "final_grounded_answer": {
    "style_hint": "Direct answer grounded in supports/partials with bracketed doc IDs.",
    "answer": "4–6 sentences grounded answer.",
    "evidence": ["d1","d2"],
    "abstain": false
  },
  "gold_answer": "President of Nigeria",   # optional, for single-truth recall
  "trace_type": "summarized"
}
"""

import json
from typing import Dict, Any, List, Iterator, Optional


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
    """Write a list of dataset records to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Record-level utilities
# -------------------------

def doc_index_from_record(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a {doc_id → doc_dict} index for quick lookup."""
    idx = {}
    for d in record.get("retrieved_docs", []) or []:
        if "doc_id" in d:
            idx[d["doc_id"]] = d
    return idx


def support_doc_ids_from_notes(per_doc_notes: List[Dict[str, Any]], accept_partial: bool = True) -> List[str]:
    """
    Extract doc_ids that support (or partially support) the query.
    De-duplicates while preserving order.
    """
    out, seen = [], set()
    for n in per_doc_notes or []:
        verdict = (n.get("verdict") or "").lower()
        if verdict == "supports" or (accept_partial and verdict == "partially supports"):
            did = n["doc_id"]
            if did not in seen:
                out.append(did)
                seen.add(did)
    return out


def gold_answerable_from_notes(per_doc_notes: List[Dict[str, Any]], accept_partial: bool = True) -> bool:
    """
    Return True if at least one support/partial-support doc exists.
    """
    return len(support_doc_ids_from_notes(per_doc_notes, accept_partial)) > 0


def get_model_output(record: Dict[str, Any]) -> str:
    """
    Extract the model's answer text from a record.
    - Prefer explicit `model_output` field if present.
    - Fall back to Stage-3 `final_grounded_answer.answer`.
    """
    if "model_output" in record and record["model_output"]:
        return record["model_output"]
    return record.get("final_grounded_answer", {}).get("answer", "") or ""


def get_gold_answer(record: Dict[str, Any]) -> Optional[str]:
    """
    Extract gold answer string for single-truth recall evaluation.
    """
    return record.get("gold_answer")


# -------------------------
# Batch utilities
# -------------------------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load all records from a JSONL dataset file into a list."""
    return list(read_jsonl(path))


def dataset_size(path: str) -> int:
    """Return number of non-empty records in a JSONL file."""
    return sum(1 for _ in read_jsonl(path))
