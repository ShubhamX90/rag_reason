#!/usr/bin/env python3
"""
Compute conflict-type inter-annotator agreement for the training review pass.

Expected inputs are reviewer JSONL files produced by
`scripts/training_conflict_type_review_cli.py`.

Outputs:
- overall raw agreement on doubly reviewed items
- overall Fleiss/Cohen-style kappa for 2 ratings per item
- pairwise agreement and kappa by reviewer pair
- per-label reviewer output distributions
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


CANONICAL_CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]

CONFLICT_TYPE_ALIASES = {
    "Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
}


def canonical(label: Any) -> str:
    raw = str(label or "").strip()
    return CONFLICT_TYPE_ALIASES.get(raw, raw)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Review file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fleiss_kappa_two_raters(label_pairs: Sequence[Tuple[str, str]]) -> float:
    if not label_pairs:
        return float("nan")
    n = 2
    N = len(label_pairs)
    per_item_counts: List[Counter[str]] = []
    total_counts: Counter[str] = Counter()
    for a, b in label_pairs:
        counts = Counter([a, b])
        per_item_counts.append(counts)
        total_counts.update([a, b])

    P_i_sum = 0.0
    for counts in per_item_counts:
        P_i = sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))
        P_i_sum += P_i
    P_bar = P_i_sum / N

    p_j = {label: total_counts[label] / (N * n) for label in set(total_counts)}
    P_e = sum(p * p for p in p_j.values())
    if abs(1.0 - P_e) < 1e-12:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def raw_agreement(label_pairs: Sequence[Tuple[str, str]]) -> float:
    if not label_pairs:
        return float("nan")
    return sum(1 for a, b in label_pairs if a == b) / len(label_pairs)


def load_review_files(paths: Sequence[str]) -> Tuple[Dict[str, List[Dict[str, Any]]], Counter[str], Counter[str]]:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    label_counts: Counter[str] = Counter()
    reviewer_counts: Counter[str] = Counter()
    for path in paths:
        rows = read_jsonl(Path(path))
        for row in rows:
            rid = str(row.get("id") or "").strip()
            if not rid:
                continue
            reviewed_label = canonical(row.get("reviewed_conflict_type"))
            row["_reviewed_conflict_type_canonical"] = reviewed_label
            by_id[rid].append(row)
            label_counts[reviewed_label] += 1
            reviewer_counts[str(row.get("reviewer_id"))] += 1
    return by_id, label_counts, reviewer_counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reviews", nargs="+", required=True, help="Reviewer JSONL files")
    args = ap.parse_args()

    by_id, label_counts, reviewer_counts = load_review_files(args.reviews)

    exact_two: Dict[str, List[Dict[str, Any]]] = {}
    bad_counts: Dict[str, int] = {}
    for rid, rows in by_id.items():
        if len(rows) == 2:
            exact_two[rid] = sorted(rows, key=lambda row: int(row.get("reviewer_id") or 0))
        else:
            bad_counts[rid] = len(rows)

    overall_pairs = [
        (
            rows[0]["_reviewed_conflict_type_canonical"],
            rows[1]["_reviewed_conflict_type_canonical"],
        )
        for rows in exact_two.values()
    ]

    print(f"review_files={len(args.reviews)}")
    print(f"total_unique_ids={len(by_id)}")
    print(f"exactly_two_reviews={len(exact_two)}")
    print(f"non_two_review_ids={len(bad_counts)}")
    if bad_counts:
        preview = list(itertools.islice(sorted(bad_counts.items()), 10))
        print(f"non_two_review_preview={preview}")
    print(f"reviewer_assignment_counts={dict(sorted(reviewer_counts.items()))}")
    print(f"reviewed_label_distribution={dict(label_counts)}")
    print()

    overall_raw = raw_agreement(overall_pairs)
    overall_kappa = fleiss_kappa_two_raters(overall_pairs)
    print(f"overall_raw_agreement={overall_raw:.4f}")
    print(f"overall_kappa={overall_kappa:.4f}")
    print()

    pair_buckets: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
    pair_changes: Dict[Tuple[int, int], Counter[str]] = defaultdict(Counter)
    for rows in exact_two.values():
        r1, r2 = rows
        pair = tuple(sorted((int(r1["reviewer_id"]), int(r2["reviewer_id"]))))
        pair_buckets[pair].append(
            (
                r1["_reviewed_conflict_type_canonical"],
                r2["_reviewed_conflict_type_canonical"],
            )
        )
        for review in rows:
            pair_changes[pair][str(bool(review.get("changed_label")))] += 1

    print("pairwise_metrics:")
    for pair in sorted(pair_buckets):
        pairs = pair_buckets[pair]
        print(
            f"  reviewer_pair={pair[0]}-{pair[1]} "
            f"n={len(pairs)} raw_agreement={raw_agreement(pairs):.4f} "
            f"kappa={fleiss_kappa_two_raters(pairs):.4f} "
            f"changed_label_counts={dict(pair_changes[pair])}"
        )


if __name__ == "__main__":
    main()
