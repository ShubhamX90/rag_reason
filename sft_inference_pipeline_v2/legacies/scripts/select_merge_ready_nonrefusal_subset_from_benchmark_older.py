#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset_strict.jsonl"
INPUT_BROAD_PATH = ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset.jsonl"
OUT_PATH = ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset_merge_ready.jsonl"
MANIFEST_PATH = (
    ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset_merge_ready_manifest.json"
)

# From the manual audit subset:
# justified qacc rows to keep
KEEP_QACC_IDS = {
    "qacc_0025",
    "qacc_0066",
    "qacc_0091",
    "qacc_0156",
    "qacc_0165",
    "qacc_0197",
    "qacc_0030",
    "qacc_0013",
    "qacc_0023",
    "qacc_0037",
    "qacc_0167",
}

KEEP_FULL_SLICES = {
    "wikirevision_outdated_200",
    "hotpotqa_complementary_200",
    "healthcontradict_conflicting_200",
    "misinformation_200",
}

EXTRA_QACC_GOLD_OVERRIDES = {
    "qacc_0025": (
        "Authorship is disputed: Pusha T has publicly claimed to have written the McDonald's "
        "\"I'm Lovin' It\" jingle, while other sources credit or associate Justin Timberlake "
        "and Pharrell Williams with the campaign/song."
    ),
    "qacc_0066": (
        "The song was first recorded by Smokey Robinson & The Miracles in 1966, though that "
        "version was unreleased; Gladys Knight & the Pips were the first to release a major "
        "version, while Marvin Gaye later made the song most famous."
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    rows = load_jsonl(INPUT_PATH)
    broad_rows_by_id = {row["id"]: row for row in load_jsonl(INPUT_BROAD_PATH)}
    selected = []
    reject_counts = Counter()

    for row in rows:
        source_slice = row["_selection_metadata"]["source_slice"]
        if source_slice in KEEP_FULL_SLICES:
            selected.append(row)
            continue
        if source_slice == "qacc_conflicting_200":
            if row["id"] in KEEP_QACC_IDS:
                selected.append(row)
            else:
                reject_counts["qacc_not_merge_ready"] += 1
            continue
        reject_counts["unexpected_slice"] += 1

    # Add a small number of manually approved qacc conflict rows that were
    # intentionally left out of the strict file but are acceptable for the
    # final merge-ready set.
    selected_ids = {row["id"] for row in selected}
    extra_added = []
    for row_id in sorted(KEEP_QACC_IDS):
        if row_id in selected_ids:
            continue
        row = broad_rows_by_id.get(row_id)
        if row is None:
            reject_counts["missing_broad_extra_id"] += 1
            continue
        row = dict(row)
        if row_id in EXTRA_QACC_GOLD_OVERRIDES:
            row["_original_gold_answer"] = row.get("gold_answer", "")
            row["gold_answer"] = EXTRA_QACC_GOLD_OVERRIDES[row_id]
            row["_selection_metadata"] = dict(row["_selection_metadata"])
            row["_selection_metadata"]["merge_ready_gold_answer_source"] = "manual_audit_override"
        selected.append(row)
        selected_ids.add(row_id)
        extra_added.append(row_id)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "input_subset": str(INPUT_PATH.relative_to(ROOT)),
        "output_subset": str(OUT_PATH.relative_to(ROOT)),
        "selection_name": "benchmark_older_high_quality_nonrefusal_merge_ready_v1",
        "selection_logic": {
            "kept_full_slices": sorted(KEEP_FULL_SLICES),
            "qacc_policy": "keep only manually audited-good qacc rows from the strict subset",
            "kept_qacc_ids": sorted(KEEP_QACC_IDS),
            "extra_qacc_rows_added_from_broad_subset": extra_added,
        },
        "counts": {
            "input_rows": len(rows),
            "selected_rows": len(selected),
            "rejected_rows": len(rows) - len(selected),
            "reject_breakdown": dict(reject_counts),
            "by_slice": dict(Counter(row["_selection_metadata"]["source_slice"] for row in selected)),
            "by_conflict_type": dict(Counter(row["conflict_type"] for row in selected)),
        },
        "selected_ids": [row["id"] for row in selected],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
