#!/usr/bin/env python3
"""Split a val49 JSONL file into original-conflicts and original-refusals rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFLICTS = PROJECT_ROOT / "data" / "normalized" / "conflicts_normalized.jsonl"
DEFAULT_REFUSALS = PROJECT_ROOT / "data" / "normalized" / "refusals_normalized.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def id_set(path: Path) -> set[str]:
    return {row["id"] for row in read_jsonl(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--conflicts-source", type=Path, default=DEFAULT_CONFLICTS)
    ap.add_argument("--refusals-source", type=Path, default=DEFAULT_REFUSALS)
    args = ap.parse_args()

    conflicts_ids = id_set(args.conflicts_source)
    refusals_ids = id_set(args.refusals_source)
    rows = read_jsonl(args.input)

    conflicts: List[Dict[str, Any]] = []
    refusals: List[Dict[str, Any]] = []
    unknown: List[str] = []

    for row in rows:
        row_id = row.get("id")
        origin = row.get("_val49_origin")
        if origin == "conflicts" or row_id in conflicts_ids:
            conflicts.append(row)
        elif origin == "refusals" or row_id in refusals_ids:
            refusals.append(row)
        else:
            unknown.append(str(row_id))

    if unknown:
        raise SystemExit(f"Could not infer origin for ids: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conflicts_path = args.output_dir / "val49_conflicts.jsonl"
    refusals_path = args.output_dir / "val49_refusals.jsonl"
    write_jsonl(conflicts_path, conflicts)
    write_jsonl(refusals_path, refusals)

    manifest = {
        "input": str(args.input),
        "outputs": {
            "conflicts": str(conflicts_path),
            "refusals": str(refusals_path),
        },
        "counts": {
            "input": len(rows),
            "conflicts": len(conflicts),
            "refusals": len(refusals),
            "by_conflict_type": dict(Counter(row.get("conflict_type", "") for row in rows)),
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"wrote {conflicts_path}")
    print(f"wrote {refusals_path}")


if __name__ == "__main__":
    main()
