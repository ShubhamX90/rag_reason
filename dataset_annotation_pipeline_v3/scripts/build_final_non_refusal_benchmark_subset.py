#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIRST_REVIEWS_FINAL = (
    PROJECT_ROOT
    / "human_reviews/benchmark/first_pass"
)
OUTPUT_DIR = FIRST_REVIEWS_FINAL / "benchmark_selection_final"
OUTPUT_JSONL = OUTPUT_DIR / "benchmark_non_refusal_selected_800.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "benchmark_non_refusal_selected_800_summary.json"

VALID_NON_NO_CONFLICT = {
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_cleaned_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(FIRST_REVIEWS_FINAL.glob("reviewer_*_reviews.cleaned.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_top_tier_no_conflict(row: Dict[str, Any]) -> bool:
    return (
        row.get("preliminary_conflict_type") == "No conflict"
        and row.get("human_preselect_decision") == "accept"
        and row.get("preselection_confidence") == "high"
        and row.get("retrieval_quality") == "good"
        and row.get("evidence_sufficiency") == "sufficient"
        and row.get("conflict_clarity") == "clear"
        and row.get("query_specificity") == "specific"
        and row.get("source_reliability") == "strong"
        and row.get("relevant_doc_count_bin") == "4-6"
        and row.get("gold_answer_possible") is True
    )


def source_family(row: Dict[str, Any]) -> str:
    return str((row.get("_candidate_source") or {}).get("source_family") or row.get("source_family") or "unknown")


def apportion_counts(total: int, source_counts: Counter[str]) -> Dict[str, int]:
    raw = {src: total * count / sum(source_counts.values()) for src, count in source_counts.items()}
    base = {src: int(raw[src]) for src in source_counts}
    remainder = total - sum(base.values())
    ranked = sorted(source_counts, key=lambda src: (raw[src] - base[src], source_counts[src], src), reverse=True)
    for src in ranked[:remainder]:
        base[src] += 1
    return base


def main() -> None:
    rows = load_cleaned_rows()
    non_no_conflict = [row for row in rows if row.get("preliminary_conflict_type") in VALID_NON_NO_CONFLICT]
    top_tier_no_conflict = [row for row in rows if is_top_tier_no_conflict(row)]

    if len(non_no_conflict) != 555:
        raise SystemExit(f"Expected 555 non-No-conflict rows, found {len(non_no_conflict)}")
    if len(top_tier_no_conflict) < 245:
        raise SystemExit(f"Need at least 245 top-tier No conflict rows, found {len(top_tier_no_conflict)}")

    src_counts = Counter(source_family(row) for row in top_tier_no_conflict)
    src_targets = apportion_counts(245, src_counts)

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in top_tier_no_conflict:
        by_source[source_family(row)].append(row)
    for src in by_source:
        by_source[src].sort(key=lambda row: stable_hash(str(row.get("id"))))

    selected_no_conflict: List[Dict[str, Any]] = []
    for src, target in src_targets.items():
        selected_no_conflict.extend(by_source[src][:target])

    selected_no_conflict.sort(key=lambda row: stable_hash(f"selected:{row.get('id')}"))
    merged = list(non_no_conflict) + list(selected_no_conflict)
    merged.sort(key=lambda row: (row.get("preliminary_conflict_type", ""), stable_hash(f"merged:{row.get('id')}")))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_selected_non_refusal": len(merged),
        "all_non_no_conflict_included": len(non_no_conflict),
        "selected_no_conflict_count": len(selected_no_conflict),
        "selected_label_counts": dict(Counter(row.get("preliminary_conflict_type") for row in merged)),
        "selected_source_counts": dict(Counter(source_family(row) for row in merged)),
        "top_tier_no_conflict_available": len(top_tier_no_conflict),
        "top_tier_no_conflict_source_counts": dict(src_counts),
        "selected_no_conflict_source_targets": dict(src_targets),
        "selected_no_conflict_source_counts": dict(Counter(source_family(row) for row in selected_no_conflict)),
        "top_tier_definition": {
            "human_preselect_decision": "accept",
            "preliminary_conflict_type": "No conflict",
            "preselection_confidence": "high",
            "retrieval_quality": "good",
            "evidence_sufficiency": "sufficient",
            "conflict_clarity": "clear",
            "query_specificity": "specific",
            "source_reliability": "strong",
            "relevant_doc_count_bin": "4-6",
            "gold_answer_possible": True,
        },
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(merged)} rows to {OUTPUT_JSONL}")
    print(f"summary: {OUTPUT_SUMMARY}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
