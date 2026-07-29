#!/usr/bin/env python3
"""
Normalize raw benchmark JSONL into the stagewise pipeline input schema.

This keeps the benchmark-specific gold fields (`gold_answer`,
`conflict_category_id`, original conflict label/reason) while converting
retrieved docs into the field names expected by the v3 stagewise pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_conflict_type(conflict_type: str) -> str:
    ct = (conflict_type or "").strip()
    mapping = {
        "No Conflict": "No conflict",
        "Complementary Information": "Complementary information",
        "Conflicting Opinions or Research Outcomes": "Conflicting opinions or research outcomes",
        "Conflicting Opinions and Research Outcomes": "Conflicting opinions or research outcomes",
        "Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
        "Conflicting opinions or research outcomes": "Conflicting opinions or research outcomes",
        "Outdated Information": "Conflict due to outdated information",
        "Conflict Due to Outdated Information": "Conflict due to outdated information",
        "Conflict due to outdated information": "Conflict due to outdated information",
        "Misinformation": "Conflict due to misinformation",
        "Conflict Due to Misinformation": "Conflict due to misinformation",
        "Conflict due to misinformation": "Conflict due to misinformation",
        "No conflict": "No conflict",
        "Complementary information": "Complementary information",
    }
    return mapping.get(ct, ct)


def validate_raw_record(row: Dict[str, Any]) -> List[str]:
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
        return errs

    for i, doc in enumerate(docs):
        for field in ["doc_id", "title", "snippet", "url", "date"]:
            if field not in doc:
                errs.append(f"doc[{i}] missing field: {field}")
    return errs


def normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "doc_id": doc.get("doc_id", ""),
        "title": doc.get("title", ""),
        "snippet": doc.get("snippet", ""),
        "source_url": doc.get("url", "") or "",
        "timestamp": doc.get("date", "") or "",
        # Preserve original benchmark names too; harmless for stagewise scripts.
        "url": doc.get("url", "") or "",
        "date": doc.get("date", "") or "",
    }
    # Preserve CONFLICTS-style provenance fields when benchmark retrieval provides them.
    # The annotation prompts still consume `snippet`, i.e. the selected short window.
    for extra_field in ["response_str", "short_text"]:
        if extra_field in doc:
            out[extra_field] = doc.get(extra_field, "")
    return out


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    original_ct = row.get("conflict_type", "")
    normalized_ct = normalize_conflict_type(original_ct)

    out: Dict[str, Any] = {
        "id": row.get("id", ""),
        "query": row.get("query", ""),
        "retrieved_docs": [normalize_doc(doc) for doc in row.get("retrieved_docs", [])],
        "conflict_category_id": row.get("conflict_category_id"),
        "benchmark_conflict_category_id": row.get("conflict_category_id"),
        "conflict_type": normalized_ct,
        "conflict_reason": row.get("conflict_reason", ""),
        "gold_answer": row.get("gold_answer", ""),
        "_benchmark_source_conflict_type": original_ct,
        "_benchmark_source_conflict_reason": row.get("conflict_reason", ""),
        "_annotation_target": "benchmark_stagewise",
    }
    if "model_output" in row:
        out["_benchmark_model_output"] = row.get("model_output")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare benchmark JSONL for stagewise annotation")
    ap.add_argument("--input", required=True, help="Raw benchmark JSONL")
    ap.add_argument("--output", required=True, help="Prepared JSONL output")
    ap.add_argument("--limit", type=int, default=None, help="Optional max rows")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    if args.limit:
        rows = rows[:args.limit]

    errors: List[str] = []
    for row in rows:
        for err in validate_raw_record(row):
            errors.append(f"{row.get('id', '<missing-id>')}: {err}")
    if errors:
        preview = "\n".join(errors[:20])
        raise SystemExit(f"Raw benchmark validation failed ({len(errors)} issues)\n{preview}")

    out_rows = [normalize_row(row) for row in rows]
    write_jsonl(Path(args.output), out_rows)

    ct = Counter(r["conflict_type"] for r in out_rows)
    doc_counts = Counter(len(r["retrieved_docs"]) for r in out_rows)
    changed = sum(1 for r in out_rows if r["conflict_type"] != r["_benchmark_source_conflict_type"])
    print(f"✅ Prepared benchmark input → {args.output}")
    print(f"   rows={len(out_rows)} | relabeled_conflict_type={changed}")
    print(f"   conflict_types={dict(sorted(ct.items()))}")
    print(f"   doc_counts={dict(sorted(doc_counts.items()))}")


if __name__ == "__main__":
    main()
