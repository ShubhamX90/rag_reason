#!/usr/bin/env python3
"""
Normalize benchmark conflict_category_id values to their canonical taxonomy ids.

This script fixes two classes of issues:
  1. Known row-specific label mistakes.
  2. Invalid numeric sentinels (for example historical refusal rows with -1)
     by mapping them back to the canonical id implied by `conflict_type`.

By default this updates:
  - the canonical benchmark gold file
  - every prepared benchmark evaluator input under
    inputs/prepped_model_eval_inputs/benchmark_set_all_modes/**/input.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


FIXED_IDS: Dict[str, int] = {
    "healthcontradict_0060": 1,
    "qacc_0013": 1,
    "qacc_0023": 1,
}

CONFLICT_TYPE_STR_MAP: Dict[str, int] = {
    "no conflict": 1,
    "complementary information": 2,
    "conflicting opinions and research outcomes": 3,
    "conflicting opinions or research outcomes": 3,
    "conflict due to outdated information": 4,
    "conflict due to outdated information (temporal conflict)": 4,
    "conflict due to misinformation": 5,
}

VALID_CONFLICT_TYPES = {1, 2, 3, 4, 5}


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_conflict_type_id(conflict_type: object) -> Optional[int]:
    if not isinstance(conflict_type, str):
        return None
    return CONFLICT_TYPE_STR_MAP.get(conflict_type.strip().lower())


def apply_fixes(path: Path, *, dry_run: bool) -> int:
    rows = read_jsonl(path)
    changed = 0
    for row in rows:
        rid = row.get("id")
        target = None
        if rid in FIXED_IDS:
            target = FIXED_IDS[rid]
        else:
            raw = row.get("conflict_category_id")
            mapped = canonical_conflict_type_id(row.get("conflict_type"))
            if mapped is not None and raw not in VALID_CONFLICT_TYPES:
                target = mapped

        if target is not None and row.get("conflict_category_id") != target:
            row["conflict_category_id"] = target
            changed += 1
    if changed and not dry_run:
        write_jsonl(path, rows)
    return changed


def iter_target_files(repo_root: Path) -> Iterable[Path]:
    yield repo_root / "data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl"
    input_root = repo_root / "inputs/prepped_model_eval_inputs/benchmark_set_all_modes"
    yield from sorted(input_root.rglob("input.jsonl"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    total_changed = 0
    changed_files = 0
    for path in iter_target_files(repo_root):
        if not path.exists():
            continue
        changed = apply_fixes(path, dry_run=args.dry_run)
        if changed:
            changed_files += 1
            total_changed += changed
            print(f"{path}: fixed {changed} rows")

    print(f"changed_files={changed_files}")
    print(f"changed_rows={total_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
