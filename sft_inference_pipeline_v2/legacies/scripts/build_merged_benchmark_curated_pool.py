#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CURRENT_BENCH_PATH = ROOT / "data/Benchmark Dataset/benchmark_final_sanitized.jsonl"
OLDER_MERGE_READY_PATH = (
    ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset_merge_ready.jsonl"
)
LIKELY_KEEP_TSV_PATH = (
    ROOT / "outputs/analysis/benchmark_partial_only_likely_keep_non_refusal_2026-06-23.tsv"
)
OUT_PATH = ROOT / "data/Benchmark Dataset/benchmark_curated_merged_current_plus_older.jsonl"
MANIFEST_PATH = (
    ROOT / "data/Benchmark Dataset/benchmark_curated_merged_current_plus_older_manifest.json"
)

CANON_KEYS = [
    "id",
    "query",
    "retrieved_docs",
    "conflict_type",
    "conflict_reason",
    "gold_answer",
    "per_doc_notes",
    "answerable_under_evidence",
    "conflict_category_id",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def load_likely_keep_ids(path: Path) -> set[str]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["id"] for row in reader}


def canonicalize_row(
    row: dict[str, Any],
    *,
    source_pool: str,
    source_file: str,
    source_origin: str,
) -> dict[str, Any]:
    canon = {key: row.get(key) for key in CANON_KEYS}
    canon["_merge_source_pool"] = source_pool
    canon["_merge_source_file"] = source_file
    canon["_merge_source_origin"] = source_origin
    return canon


def main() -> None:
    current_rows = load_jsonl(CURRENT_BENCH_PATH)
    older_rows = load_jsonl(OLDER_MERGE_READY_PATH)
    likely_keep_ids = load_likely_keep_ids(LIKELY_KEEP_TSV_PATH)

    current_by_id = {row["id"]: row for row in current_rows}

    current_support_present_ids = {
        row["id"]
        for row in current_rows
        if row.get("answerable_under_evidence") is True
        and any(note.get("verdict") == "supports" for note in row.get("per_doc_notes", []))
    }
    current_gold_refusal_ids = {
        row["id"] for row in current_rows if row.get("answerable_under_evidence") is False
    }

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add_pool(ids: list[str] | set[str], source_pool: str, source_file: str, source_origin: str) -> None:
        for row_id in ids:
            row = current_by_id[row_id]
            canon = canonicalize_row(
                row,
                source_pool=source_pool,
                source_file=source_file,
                source_origin=source_origin,
            )
            if canon["id"] in seen_ids:
                raise ValueError(f"Duplicate id in merged output: {canon['id']}")
            merged.append(canon)
            seen_ids.add(canon["id"])

    add_pool(
        sorted(current_support_present_ids),
        "current_support_present_nonrefusal",
        str(CURRENT_BENCH_PATH.relative_to(ROOT)),
        "current_benchmark",
    )
    add_pool(
        sorted(likely_keep_ids),
        "current_likely_keep_partial_only_nonrefusal",
        str(CURRENT_BENCH_PATH.relative_to(ROOT)),
        "current_benchmark",
    )
    add_pool(
        sorted(current_gold_refusal_ids),
        "current_gold_refusal",
        str(CURRENT_BENCH_PATH.relative_to(ROOT)),
        "current_benchmark",
    )

    for row in older_rows:
        canon = canonicalize_row(
            row,
            source_pool="older_merge_ready_nonrefusal",
            source_file=str(OLDER_MERGE_READY_PATH.relative_to(ROOT)),
            source_origin="older_benchmark",
        )
        if canon["id"] in seen_ids:
            raise ValueError(f"Duplicate id in merged output: {canon['id']}")
        merged.append(canon)
        seen_ids.add(canon["id"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "inputs": {
            "current_benchmark": str(CURRENT_BENCH_PATH.relative_to(ROOT)),
            "older_merge_ready_subset": str(OLDER_MERGE_READY_PATH.relative_to(ROOT)),
            "current_likely_keep_tsv": str(LIKELY_KEEP_TSV_PATH.relative_to(ROOT)),
        },
        "output": str(OUT_PATH.relative_to(ROOT)),
        "selection_logic": {
            "current_benchmark": {
                "support_present_nonrefusal": "answerable_under_evidence == true and at least one per_doc_notes verdict == supports",
                "likely_keep_partial_only_nonrefusal": "ids from benchmark_partial_only_likely_keep_non_refusal_2026-06-23.tsv",
                "gold_refusal": "answerable_under_evidence == false",
            },
            "older_benchmark": {
                "merge_ready_nonrefusal": "rows from benchmark_older_high_quality_nonrefusal_subset_merge_ready.jsonl",
            },
        },
        "counts": {
            "current_support_present_nonrefusal": len(current_support_present_ids),
            "current_likely_keep_partial_only_nonrefusal": len(likely_keep_ids),
            "current_gold_refusal": len(current_gold_refusal_ids),
            "older_merge_ready_nonrefusal": len(older_rows),
            "total_merged_rows": len(merged),
            "by_source_pool": dict(Counter(row["_merge_source_pool"] for row in merged)),
            "by_conflict_type": dict(Counter(row["conflict_type"] for row in merged)),
            "by_answerable_under_evidence": dict(Counter(row["answerable_under_evidence"] for row in merged)),
        },
        "selected_id_counts_check": {
            "unique_ids": len(seen_ids),
            "rows_written": len(merged),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
