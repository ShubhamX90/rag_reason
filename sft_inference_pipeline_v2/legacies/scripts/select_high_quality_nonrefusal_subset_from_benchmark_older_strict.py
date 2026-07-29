#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset.jsonl"
OUT_PATH = ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset_strict.jsonl"
MANIFEST_PATH = (
    ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset_strict_manifest.json"
)

QACC_SUSPICIOUS_PATTERNS = [
    r"\bI think\b",
    r"\bbecause\b",
    r"\bcontexts?\b",
    r"\bmost common answer\b",
    r"\bmajority of contexts\b",
    r"\bThat only answer\b",
]
MANUAL_EXCLUDE_IDS = {
    "qacc_0077",  # "13th" for a "how many times" query is almost certainly a bad answer surface.
    "qacc_0083",  # "state" is too underspecified for Zhongguo -> high-quality answer.
    "qacc_0110",  # residual garbage answer text.
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def strip_inline_citations(text: str) -> str:
    text = re.sub(r"\[d\d+\]", "", text)
    text = text.replace("**", "")
    text = " ".join(text.split()).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def sanitize_qacc_strict(answer: str) -> str:
    text = " ".join(answer.split())
    if "." in text:
        text = text.split(".", 1)[0].strip()
    text = text.rstrip(",;: ").strip()
    return text


def sanitize_health_strict(answer: str) -> str:
    return answer.strip().lower().rstrip(".")


def curated_strict_gold_answer(row: dict[str, Any]) -> str:
    source_slice = row["_selection_metadata"]["source_slice"]
    answer = (row.get("gold_answer") or "").strip()
    if source_slice == "wikirevision_outdated_200":
        return strip_inline_citations(answer)
    if source_slice == "qacc_conflicting_200":
        return sanitize_qacc_strict(answer)
    if source_slice == "healthcontradict_conflicting_200":
        return sanitize_health_strict(answer)
    return answer


def keep_qacc_strict(row: dict[str, Any]) -> bool:
    if row["id"] in MANUAL_EXCLUDE_IDS:
        return False
    notes = row.get("per_doc_notes", [])
    supports = sum(note.get("verdict") == "supports" for note in notes)
    irrelevant = sum(note.get("verdict") == "irrelevant" for note in notes)
    if supports < 2 or irrelevant != 0:
        return False
    answer = (row.get("gold_answer") or "").strip()
    return not any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in QACC_SUSPICIOUS_PATTERNS)


def strict_score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    notes = row.get("per_doc_notes", [])
    supports = sum(note.get("verdict") == "supports" for note in notes)
    partial = sum(note.get("verdict") == "partially supports" for note in notes)
    irrelevant = sum(note.get("verdict") == "irrelevant" for note in notes)
    answer_len = len((row.get("gold_answer") or "").split())
    return (supports, -irrelevant, -partial, -answer_len, -int(re.sub(r"\D", "", row["id"]) or "0"))


def main() -> None:
    rows = load_jsonl(INPUT_PATH)
    kept: list[dict[str, Any]] = []
    reject_counts = Counter()

    for row in rows:
        source_slice = row["_selection_metadata"]["source_slice"]
        if source_slice == "qacc_conflicting_200" and not keep_qacc_strict(row):
            reject_counts["qacc_strict_filter"] += 1
            continue

        strict_row = dict(row)
        strict_row["_strict_original_gold_answer"] = row.get("gold_answer", "")
        strict_row["gold_answer"] = curated_strict_gold_answer(row)
        strict_row["_selection_metadata"] = dict(row["_selection_metadata"])
        strict_row["_selection_metadata"]["subset_name"] = "benchmark_older_high_quality_nonrefusal_strict_v1"
        strict_row["_selection_metadata"]["strict_pass_applied"] = True
        strict_row["_selection_metadata"]["strict_gold_answer_source"] = source_slice
        if not strict_row["gold_answer"].strip():
            reject_counts["empty_gold_after_strict_sanitize"] += 1
            continue
        kept.append(strict_row)

    # Deduplicate repeated normalized queries by preferring stronger evidence patterns.
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kept:
        by_query[normalize_query(row["query"])].append(row)

    final_rows: list[dict[str, Any]] = []
    dedup_dropped = []
    for norm_query, candidates in by_query.items():
        if len(candidates) == 1:
            final_rows.append(candidates[0])
            continue
        winner = sorted(candidates, key=strict_score, reverse=True)[0]
        final_rows.append(winner)
        for row in candidates:
            if row["id"] != winner["id"]:
                dedup_dropped.append({"query": norm_query, "dropped_id": row["id"], "kept_id": winner["id"]})

    final_rows.sort(key=lambda row: row["id"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "input_subset": str(INPUT_PATH.relative_to(ROOT)),
        "output_subset": str(OUT_PATH.relative_to(ROOT)),
        "selection_name": "benchmark_older_high_quality_nonrefusal_strict_v1",
        "selection_logic": {
            "base": "start from benchmark_older_high_quality_nonrefusal_subset.jsonl",
            "strict_filters": {
                "qacc_conflicting_200": [
                    "supports >= 2",
                    "irrelevant == 0",
                    "no rationale-like gold answer residue",
                    "manual exclusion of a few still-bad surfaces",
                ],
                "wikirevision_outdated_200": [
                    "keep, but strip inline citations and markdown from curated gold answers",
                    "deduplicate repeated normalized queries",
                ],
                "healthcontradict_conflicting_200": [
                    "keep, normalize yes/no answer text",
                    "deduplicate repeated normalized queries",
                ],
                "other_slices": [
                    "keep if they survived the broad v1 pass",
                ],
            },
        },
        "counts": {
            "input_rows": len(rows),
            "pre_dedup_kept_rows": len(kept),
            "selected_rows": len(final_rows),
            "rejected_rows": len(rows) - len(final_rows),
            "reject_breakdown": dict(reject_counts),
            "dedup_dropped_count": len(dedup_dropped),
            "by_slice": dict(Counter(row["_selection_metadata"]["source_slice"] for row in final_rows)),
            "by_conflict_type": dict(Counter(row["conflict_type"] for row in final_rows)),
        },
        "dedup_dropped": dedup_dropped,
        "selected_ids": [row["id"] for row in final_rows],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
