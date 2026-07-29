#!/usr/bin/env python3
"""
Prepare the curated refusal-benchmark slice for local benchmark annotation.

These rows already represent gold refusal cases. We still normalize them into
the benchmark-style stagewise schema so the existing Stage-1 / Stage-2 scripts
and validators can run unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_doc(doc: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "doc_id": doc.get("doc_id", f"d{idx}"),
        "title": doc.get("title", "") or "",
        "snippet": doc.get("snippet", "") or "",
        "source_url": doc.get("source_url", "") or doc.get("url", "") or "",
        "timestamp": doc.get("timestamp", "") or doc.get("date", "") or "",
        "url": doc.get("source_url", "") or doc.get("url", "") or "",
        "date": doc.get("timestamp", "") or doc.get("date", "") or "",
    }


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    docs = [normalize_doc(doc, i + 1) for i, doc in enumerate(row.get("retrieved_docs", []))]
    return {
        "id": row.get("id", ""),
        "query": row.get("query", ""),
        "retrieved_docs": docs,
        "conflict_category_id": -1,
        "benchmark_conflict_category_id": -1,
        "conflict_type": row.get("conflict_type", "No conflict"),
        "conflict_reason": "Gold refusal benchmark item; final refusal-oriented conflict reasoning is generated in Stage 2.",
        "gold_answer": row.get("gold_answer", ""),
        "_benchmark_source_conflict_type": row.get("conflict_type", ""),
        "_benchmark_source_conflict_reason": "Gold refusal benchmark input.",
        "_annotation_target": "benchmark_refusal_stagewise",
        "_gold_refusal_benchmark": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare curated refusal benchmark JSONL for stagewise local annotation")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    out_rows = [normalize_row(row) for row in rows]
    write_jsonl(Path(args.output), out_rows)

    print(f"✅ Prepared refusal benchmark input → {args.output}")
    print(f"   rows={len(out_rows)}")
    print(f"   conflict_types={dict(sorted(Counter(r['conflict_type'] for r in out_rows).items()))}")
    print(f"   doc_counts={dict(sorted(Counter(len(r['retrieved_docs']) for r in out_rows).items()))}")


if __name__ == "__main__":
    main()
