#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


CONFLICT_LABEL_ALIASES = {
    "no conflict": "No conflict",
    "complementary information": "Complementary information",
    "conflicting opinions or research outcomes": "Conflicting opinions or research outcomes",
    "conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
    "conflict due to outdated information": "Conflict due to outdated information",
    "conflict due to misinformation": "Conflict due to misinformation",
}

SUPPORT_TARGET_QUOTAS = {
    "Complementary information": 10,
    "Conflicting opinions or research outcomes": 10,
    "Conflict due to outdated information": 6,
    "Conflict due to misinformation": 1,
    "No conflict": 18,
}

PARTIAL_ONLY_TARGET_QUOTAS = {
    "No conflict": 3,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_conflict_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value.strip())
    return CONFLICT_LABEL_ALIASES.get(text.lower(), text)


def safe_id_suffix(value: str) -> str:
    value = value.replace("#", "hash_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_")


def classify_doc_ids(row: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    valid_doc_ids = {doc.get("doc_id") for doc in row.get("retrieved_docs") or []}
    supports: list[str] = []
    partial: list[str] = []
    irrelevant: list[str] = []
    for note in row.get("per_doc_notes") or []:
        doc_id = note.get("doc_id")
        if doc_id not in valid_doc_ids:
            continue
        verdict = (note.get("verdict") or "").strip().lower()
        if verdict == "supports":
            supports.append(doc_id)
        elif verdict == "partially supports":
            partial.append(doc_id)
        elif verdict == "irrelevant":
            irrelevant.append(doc_id)
    return supports, partial, irrelevant


def origin_for_variant(conflict_type: str, *, partial_only: bool) -> str:
    if conflict_type == "No conflict":
        return (
            "run_l_short5_no_conflict_partial_only"
            if partial_only
            else "run_l_short5_no_conflict_support"
        )
    return "run_l_short5_partial_only" if partial_only else "run_l_short5_support"


def subset_row(
    row: dict[str, Any],
    selected_doc_ids: list[str],
    *,
    new_id: str,
    origin: str,
) -> dict[str, Any]:
    selected_set = set(selected_doc_ids)
    cloned = deepcopy(row)
    cloned["id"] = new_id
    cloned["_run_l_origin"] = origin
    cloned["_run_l_parent_id"] = row.get("id")
    cloned["_run_l_parent_doc_count"] = len(row.get("retrieved_docs") or [])
    cloned["conflict_type"] = normalize_conflict_type(cloned.get("conflict_type"))
    cloned["retrieved_docs"] = [
        doc for doc in (row.get("retrieved_docs") or []) if doc.get("doc_id") in selected_set
    ]
    cloned["per_doc_notes"] = [
        note for note in (row.get("per_doc_notes") or []) if note.get("doc_id") in selected_set
    ]
    expected = cloned.get("expected_response")
    if isinstance(expected, dict):
        evidence = [doc_id for doc_id in (expected.get("evidence") or []) if doc_id in selected_set]
        if not evidence:
            supports, partial, _ = classify_doc_ids(cloned)
            evidence = supports or partial or selected_doc_ids[:1]
        expected["evidence"] = evidence
    return cloned


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_conflict_type": dict(
            sorted(Counter(normalize_conflict_type(row.get("conflict_type")) for row in rows).items())
        ),
        "by_origin": dict(sorted(Counter(row.get("_run_l_origin", "main") for row in rows).items())),
        "by_parent_doc_count": dict(
            sorted(Counter(int(row.get("_run_l_parent_doc_count", 0)) for row in rows).items())
        ),
    }


def derive_short_context_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    support_candidates: list[dict[str, Any]] = []
    partial_only_candidates: list[dict[str, Any]] = []
    seen_ids = {row.get("id") for row in rows}

    for row in rows:
        if row.get("answerable_under_evidence") is not True:
            continue
        docs = row.get("retrieved_docs") or []
        if len(docs) <= 5:
            continue

        conflict_type = normalize_conflict_type(row.get("conflict_type"))
        supports, partial, irrelevant = classify_doc_ids(row)

        if (
            conflict_type in SUPPORT_TARGET_QUOTAS
            and 1 <= len(supports) <= 3
            and len(supports) + len(partial) + len(irrelevant) >= 5
        ):
            selected_doc_ids = (supports + partial + irrelevant)[:5]
            if len(selected_doc_ids) == 5:
                new_id = f"runl_short5_{safe_id_suffix(str(row.get('id') or 'unknown'))}"
                if new_id not in seen_ids:
                    seen_ids.add(new_id)
                    support_candidates.append(
                        subset_row(
                            row,
                            selected_doc_ids,
                            new_id=new_id,
                            origin=origin_for_variant(conflict_type, partial_only=False),
                        )
                    )

        if (
            conflict_type in PARTIAL_ONLY_TARGET_QUOTAS
            and len(supports) == 0
            and 2 <= len(partial) <= 5
            and len(partial) + len(irrelevant) >= 5
        ):
            selected_doc_ids = (partial + irrelevant)[:5]
            if len(selected_doc_ids) == 5:
                new_id = f"runl_short5_partial_{safe_id_suffix(str(row.get('id') or 'unknown'))}"
                if new_id not in seen_ids:
                    seen_ids.add(new_id)
                    partial_only_candidates.append(
                        subset_row(
                            row,
                            selected_doc_ids,
                            new_id=new_id,
                            origin=origin_for_variant(conflict_type, partial_only=True),
                        )
                    )

    def rank_key(row: dict[str, Any]) -> tuple[int, str]:
        return (
            -int(row.get("_run_l_parent_doc_count", 0)),
            str(row.get("_run_l_parent_id") or ""),
        )

    selected: list[dict[str, Any]] = []
    for conflict_type, limit in SUPPORT_TARGET_QUOTAS.items():
        group = [
            row
            for row in support_candidates
            if normalize_conflict_type(row.get("conflict_type")) == conflict_type
        ]
        group.sort(key=rank_key)
        selected.extend(group[:limit])

    for conflict_type, limit in PARTIAL_ONLY_TARGET_QUOTAS.items():
        group = [
            row
            for row in partial_only_candidates
            if normalize_conflict_type(row.get("conflict_type")) == conflict_type
        ]
        group.sort(key=rank_key)
        selected.extend(group[:limit])

    selected.sort(key=lambda row: row["id"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Run L splits by extending Run J with targeted short-context answerable "
            "variants, including explicit No conflict coverage."
        )
    )
    parser.add_argument(
        "--base_train_jsonl",
        type=Path,
        default=Path("data/splits/run_j/stagewise_train_augmented.jsonl"),
    )
    parser.add_argument(
        "--base_val_jsonl",
        type=Path,
        default=Path("data/splits/run_j/stagewise_val_combined.jsonl"),
    )
    parser.add_argument("--out_dir", type=Path, default=Path("data/splits/run_l"))
    args = parser.parse_args()

    base_train = read_jsonl(args.base_train_jsonl)
    base_val = read_jsonl(args.base_val_jsonl)
    derived = derive_short_context_variants(base_train)
    stagewise_train_augmented = base_train + derived

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "base_train_from_run_j.jsonl", base_train)
    write_jsonl(args.out_dir / "derived_short_context_answerables.jsonl", derived)
    write_jsonl(args.out_dir / "stagewise_train_augmented.jsonl", stagewise_train_augmented)
    write_jsonl(args.out_dir / "stagewise_val_combined.jsonl", base_val)

    summary = {
        "base_train_rows": len(base_train),
        "base_val_rows": len(base_val),
        "derived_summary": summarize(derived),
        "final_train_rows": len(stagewise_train_augmented),
        "final_val_rows": len(base_val),
    }
    with (args.out_dir / "run_l_split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
