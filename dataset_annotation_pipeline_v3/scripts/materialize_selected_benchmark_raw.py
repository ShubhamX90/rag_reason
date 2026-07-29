#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_INPUT = (
    PROJECT_ROOT
    / "human_reviews/benchmark/first_pass/benchmark_selection_final/benchmark_non_refusal_selected_800.jsonl"
)
RAW_RETRIEVED_INPUT = PROJECT_ROOT / "data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl"
OUTPUT_RAW = (
    PROJECT_ROOT
    / "human_reviews/benchmark/first_pass/benchmark_selection_final/benchmark_non_refusal_selected_800_raw.jsonl"
)

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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    selected_rows = read_jsonl(SELECTED_INPUT)
    raw_rows = read_jsonl(RAW_RETRIEVED_INPUT)
    raw_by_id = {row["id"]: row for row in raw_rows}

    missing = [row["id"] for row in selected_rows if row["id"] not in raw_by_id]
    if missing:
        raise SystemExit(f"Missing {len(missing)} selected ids in raw retrieved source; first few: {missing[:10]}")

    output_rows: List[Dict[str, Any]] = []
    for review_row in selected_rows:
        base_row = raw_by_id[review_row["id"]]
        conflict_type = review_row["preliminary_conflict_type"]
        conflict_category_id = CONFLICT_CATEGORY_IDS[conflict_type]
        human_gold_answer = (review_row.get("human_gold_answer") or "").strip()
        output_rows.append(
            {
                "id": base_row["id"],
                "query": base_row["query"],
                "retrieved_docs": base_row["retrieved_docs"],
                "conflict_category_id": conflict_category_id,
                "conflict_type": conflict_type,
                "conflict_reason": "Human preselection conflict label retained; annotate conflict_reason in Stage 2 using gold conflict_type.",
                "gold_answer": human_gold_answer,
                "_candidate_source": base_row.get("_candidate_source", {}),
                "_retrieval_metadata": base_row.get("_retrieval_metadata", {}),
                "_human_preselection": {
                    "reviewer_first_name": review_row.get("reviewer_first_name"),
                    "reviewer_id": review_row.get("reviewer_id"),
                    "human_preselect_decision": review_row.get("human_preselect_decision"),
                    "preliminary_conflict_type_id": review_row.get("preliminary_conflict_type_id"),
                    "preliminary_conflict_type": review_row.get("preliminary_conflict_type"),
                    "preselection_confidence": review_row.get("preselection_confidence"),
                    "retrieval_quality": review_row.get("retrieval_quality"),
                    "evidence_sufficiency": review_row.get("evidence_sufficiency"),
                    "conflict_clarity": review_row.get("conflict_clarity"),
                    "query_specificity": review_row.get("query_specificity"),
                    "source_reliability": review_row.get("source_reliability"),
                    "relevant_doc_count_bin": review_row.get("relevant_doc_count_bin"),
                    "gold_answer_possible": review_row.get("gold_answer_possible"),
                    "human_gold_answer": review_row.get("human_gold_answer", ""),
                    "needs_second_reviewer": review_row.get("needs_second_reviewer"),
                    "reviewer_notes": review_row.get("reviewer_notes", ""),
                    "reviewed_at_utc": review_row.get("reviewed_at_utc"),
                },
            }
        )

    write_jsonl(OUTPUT_RAW, output_rows)
    print(f"wrote {len(output_rows)} rows to {OUTPUT_RAW}")


if __name__ == "__main__":
    main()
