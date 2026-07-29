#!/usr/bin/env python3
"""
Select a balanced, high-confidence benchmark set from stagewise outputs.

Input should be the Stage-2 output produced by scripts/run_benchmark_stagewise.py
in --benchmark-mode. The selector uses committee vote tallies, answerability,
and per-document relevance notes to keep records whose conflict labels are
stable enough to serve as benchmark gold.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFLICT_LABELS = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]

CONFLICT_CATEGORY_IDS = {
    "No conflict": 1,
    "Complementary information": 2,
    "Conflicting opinions or research outcomes": 3,
    "Conflict due to outdated information": 4,
    "Conflict due to misinformation": 5,
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def vote_confidence(row: Dict[str, Any]) -> float:
    tally = row.get("_ct_vote_tally") or {}
    if not tally:
        return 1.0 if row.get("conflict_type") in CONFLICT_LABELS else 0.0
    total = sum(float(v) for v in tally.values())
    if total <= 0:
        return 0.0
    return max(float(v) for v in tally.values()) / total


def non_irrelevant_count(row: Dict[str, Any]) -> int:
    notes = row.get("per_doc_notes") or []
    return sum(1 for note in notes if note.get("verdict") in {"supports", "partially supports"})


def is_eligible(row: Dict[str, Any], min_confidence: float, min_docs: int) -> Tuple[bool, str]:
    label = row.get("conflict_type")
    if label not in CONFLICT_LABELS:
        return False, "invalid_label"
    if vote_confidence(row) < min_confidence:
        return False, "low_confidence"
    if not row.get("answerable_under_evidence", False):
        return False, "not_answerable"
    if len(row.get("retrieved_docs") or []) < min_docs:
        return False, "too_few_docs"
    non_irrelevant = non_irrelevant_count(row)
    if label == "No conflict" and non_irrelevant < 1:
        return False, "no_relevant_docs"
    if label != "No conflict" and non_irrelevant < 2:
        return False, "too_few_relevant_docs_for_conflict"
    if not str(row.get("conflict_reason", "")).strip():
        return False, "missing_reason"
    return True, "eligible"


def select_balanced(rows: List[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["conflict_type"]].append(row)
    for label in CONFLICT_LABELS:
        by_label[label].sort(key=lambda row: (-vote_confidence(row), row.get("id", "")))
        top_band = by_label[label][:]
        rng.shuffle(top_band)
        top_band.sort(key=lambda row: -vote_confidence(row))
        by_label[label] = top_band

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    cursors = {label: 0 for label in CONFLICT_LABELS}

    while len(selected) < target:
        active = [label for label in CONFLICT_LABELS if cursors[label] < len(by_label[label])]
        if not active:
            break
        counts = Counter(row["conflict_type"] for row in selected)
        active.sort(key=lambda label: (counts[label], CONFLICT_LABELS.index(label)))
        made_progress = False
        for label in active:
            bucket = by_label[label]
            while cursors[label] < len(bucket) and bucket[cursors[label]].get("id") in selected_ids:
                cursors[label] += 1
            if cursors[label] >= len(bucket):
                continue
            row = bucket[cursors[label]]
            cursors[label] += 1
            selected.append(row)
            selected_ids.add(row.get("id", ""))
            made_progress = True
            if len(selected) >= target:
                break
        if not made_progress:
            break
    return selected


def to_raw_gold(row: Dict[str, Any]) -> Dict[str, Any]:
    label = row["conflict_type"]
    return {
        "id": row.get("id", ""),
        "query": row.get("query", ""),
        "retrieved_docs": row.get("retrieved_docs", []),
        "conflict_category_id": CONFLICT_CATEGORY_IDS[label],
        "conflict_type": label,
        "conflict_reason": row.get("conflict_reason", ""),
        "gold_answer": row.get("gold_answer", ""),
        "_gold_source": "stagewise_committee_high_confidence",
        "_ct_vote_tally": row.get("_ct_vote_tally", {}),
        "_ans_vote_tally": row.get("_ans_vote_tally", {}),
        "_selection_confidence": vote_confidence(row),
        "_candidate_source": row.get("_candidate_source", {}),
        "_retrieval_metadata": row.get("_retrieval_metadata", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Select high-confidence balanced benchmark records")
    ap.add_argument("--input", required=True, help="Stage-2 benchmark-mode JSONL")
    ap.add_argument("--output", required=True, help="Selected Stage-2 JSONL with full audit fields")
    ap.add_argument("--output-raw-gold", default=None, help="Optional stripped raw-gold JSONL for benchmark reruns")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--min-confidence", type=float, default=0.70)
    ap.add_argument("--min-docs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    reject_reasons: Counter = Counter()
    eligible: List[Dict[str, Any]] = []
    for row in rows:
        ok, reason = is_eligible(row, args.min_confidence, args.min_docs)
        if ok:
            eligible.append(row)
        else:
            reject_reasons[reason] += 1

    selected = select_balanced(eligible, args.target, args.seed)
    write_jsonl(Path(args.output), selected)
    if args.output_raw_gold:
        write_jsonl(Path(args.output_raw_gold), [to_raw_gold(row) for row in selected])

    manifest = {
        "input": args.input,
        "output": args.output,
        "output_raw_gold": args.output_raw_gold,
        "target": args.target,
        "selected": len(selected),
        "min_confidence": args.min_confidence,
        "min_docs": args.min_docs,
        "input_rows": len(rows),
        "eligible_rows": len(eligible),
        "reject_reasons": dict(reject_reasons),
        "eligible_distribution": dict(Counter(row["conflict_type"] for row in eligible)),
        "selected_distribution": dict(Counter(row["conflict_type"] for row in selected)),
        "mean_selected_confidence": (
            sum(vote_confidence(row) for row in selected) / len(selected)
            if selected else 0.0
        ),
        "notes": [
            "If misinformation is scarce, the selector keeps all eligible misinformation rows and balances the remaining labels as evenly as possible.",
            "For publication-grade gold labels, manually review selected rows with low committee margins or high-stakes factual claims.",
        ],
    }
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"selected={len(selected)} eligible={len(eligible)} input_rows={len(rows)}")
    print(f"selected_distribution={manifest['selected_distribution']}")
    print(f"reject_reasons={manifest['reject_reasons']}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
