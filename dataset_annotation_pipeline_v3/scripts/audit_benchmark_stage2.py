#!/usr/bin/env python3
"""
Audit benchmark Stage-2 outputs against source benchmark labels.

This is meant to distinguish:
1. likely prompt / annotation misses
2. likely benchmark-retrieval integrity problems
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit benchmark Stage-2 outputs against source labels")
    ap.add_argument("--input", required=True, help="Stage-2 benchmark JSONL")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    mismatches: List[Dict[str, Any]] = []
    for row in rows:
        src = row.get("_gold_conflict_type") or row.get("_benchmark_source_conflict_type")
        voted = row.get("conflict_type")
        if src != voted:
            notes = row.get("per_doc_notes", [])
            non_irrel = [n for n in notes if n.get("verdict") in {"supports", "partially supports"}]
            mismatches.append({
                "id": row.get("id"),
                "query": row.get("query"),
                "source_conflict_type": src,
                "voted_conflict_type": voted,
                "answerable_under_evidence": row.get("answerable_under_evidence"),
                "non_irrelevant_doc_count": len(non_irrel),
                "non_irrelevant_doc_ids": [n.get("doc_id") for n in non_irrel],
            })

    print(f"rows={len(rows)}")
    print(f"mismatches={len(mismatches)}")
    print("voted_distribution=", dict(Counter(r.get('conflict_type') for r in rows)))
    if mismatches:
        print("\nTop mismatches:")
        for row in mismatches[:20]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
