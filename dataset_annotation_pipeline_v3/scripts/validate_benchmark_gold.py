#!/usr/bin/env python3
"""
Validate benchmark files at the prepared / stage1 / stage2 stages.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


VALID_CONFLICT_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflicting opinions and research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}
VALID_VERDICTS = {"supports", "partially supports", "irrelevant"}
VALID_SOURCE_QUALITY = {"high", "low"}


def normalize_doc_id(value: Any) -> str:
    return html.unescape(str(value or ""))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_common(row: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for field in [
        "id",
        "query",
        "retrieved_docs",
        "conflict_category_id",
        "conflict_type",
        "conflict_reason",
        "gold_answer",
    ]:
        if field not in row:
            errs.append(f"missing top-level field: {field}")
    docs = row.get("retrieved_docs", [])
    if not isinstance(docs, list) or not docs:
        errs.append("retrieved_docs must be a non-empty list")
    else:
        for i, doc in enumerate(docs):
            for field in ["doc_id", "title", "snippet", "source_url", "timestamp"]:
                if field not in doc:
                    errs.append(f"doc[{i}] missing field: {field}")
    if row.get("conflict_type") not in VALID_CONFLICT_TYPES:
        errs.append(f"invalid conflict_type: {row.get('conflict_type')!r}")
    return errs


def validate_stage1(row: Dict[str, Any]) -> List[str]:
    errs = validate_common(row)
    notes = row.get("per_doc_notes")
    docs = row.get("retrieved_docs", [])
    if not isinstance(notes, list):
        errs.append("per_doc_notes missing or not a list")
        return errs
    if len(notes) != len(docs):
        errs.append(f"per_doc_notes length {len(notes)} != retrieved_docs length {len(docs)}")
    doc_ids = {normalize_doc_id(doc.get("doc_id", "")) for doc in docs}
    for i, note in enumerate(notes):
        for field in ["doc_id", "verdict", "key_fact", "quote", "verdict_reason", "source_quality"]:
            if field not in note:
                errs.append(f"note[{i}] missing field: {field}")
        if note.get("verdict") not in VALID_VERDICTS:
            errs.append(f"note[{i}] invalid verdict: {note.get('verdict')!r}")
        if note.get("source_quality") not in VALID_SOURCE_QUALITY:
            errs.append(f"note[{i}] invalid source_quality: {note.get('source_quality')!r}")
        if normalize_doc_id(note.get("doc_id", "")) not in doc_ids:
            errs.append(f"note[{i}] doc_id not found in retrieved_docs: {note.get('doc_id')!r}")
    return errs


def validate_stage2(row: Dict[str, Any]) -> List[str]:
    errs = validate_stage1(row)
    if "answerable_under_evidence" not in row or not isinstance(row.get("answerable_under_evidence"), bool):
        errs.append("answerable_under_evidence missing or not bool")
    if not isinstance(row.get("conflict_reason"), str) or not row.get("conflict_reason", "").strip():
        errs.append("conflict_reason missing or empty")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate prepared / stage1 / stage2 benchmark files")
    ap.add_argument("--input", required=True, help="Input JSONL to validate")
    ap.add_argument("--stage", choices=["prepared", "stage1", "stage2"], required=True)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    all_errs: List[str] = []
    for row in rows:
        rec_id = row.get("id", "<missing-id>")
        if args.stage == "prepared":
            errs = validate_common(row)
        elif args.stage == "stage1":
            errs = validate_stage1(row)
        else:
            errs = validate_stage2(row)
        all_errs.extend(f"{rec_id}: {err}" for err in errs)

    if all_errs:
        preview = "\n".join(all_errs[:30])
        raise SystemExit(f"Validation failed for {args.input} ({len(all_errs)} issues)\n{preview}")

    ct = Counter(row.get("conflict_type") for row in rows)
    doc_counts = Counter(len(row.get("retrieved_docs", [])) for row in rows)
    print(f"✅ Benchmark validation passed ({args.stage}) → {args.input}")
    print(f"   rows={len(rows)}")
    print(f"   conflict_types={dict(sorted(ct.items()))}")
    print(f"   doc_counts={dict(sorted(doc_counts.items()))}")


if __name__ == "__main__":
    main()
