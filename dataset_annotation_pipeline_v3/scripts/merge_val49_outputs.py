#!/usr/bin/env python3
"""Merge conflicts/refusals val49 outputs back into validation-ID order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAL_IDS = PROJECT_ROOT / "data" / "splits" / "92p5_7p5" / "val_ids.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conflicts", required=True, type=Path)
    ap.add_argument("--refusals", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-ids", type=Path, default=DEFAULT_VAL_IDS)
    args = ap.parse_args()

    order = json.loads(args.val_ids.read_text(encoding="utf-8"))
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for path in (args.conflicts, args.refusals):
        for row in read_jsonl(path):
            row_id = row.get("id")
            if row_id in rows_by_id:
                duplicates.append(str(row_id))
            rows_by_id[row_id] = row
    if duplicates:
        raise SystemExit(f"Duplicate IDs while merging: {duplicates}")

    missing = [row_id for row_id in order if row_id not in rows_by_id]
    extra = sorted(row_id for row_id in rows_by_id if row_id not in set(order))
    if missing or extra:
        raise SystemExit(f"Merge mismatch; missing={missing}, extra={extra}")

    merged = [rows_by_id[row_id] for row_id in order]
    write_jsonl(args.output, merged)
    print(f"wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
