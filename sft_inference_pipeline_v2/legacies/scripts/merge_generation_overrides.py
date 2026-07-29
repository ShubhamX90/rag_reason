#!/usr/bin/env python3
"""Merge a base generation file with override rows keyed by id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except Exception as exc:  # pragma: no cover - defensive CLI guard
                raise ValueError(f"{path}:{line_no} bad json: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_jsonl", required=True, type=Path)
    parser.add_argument("--override_jsonl", required=True, type=Path)
    parser.add_argument("--out_jsonl", required=True, type=Path)
    parser.add_argument(
        "--allow_new_ids",
        action="store_true",
        help="Append override ids that do not exist in the base file.",
    )
    args = parser.parse_args()

    base_rows: List[dict] = list(read_jsonl(args.base_jsonl))
    override_by_id: Dict[str, dict] = {}
    for rec in read_jsonl(args.override_jsonl):
        cid = rec.get("id")
        if cid in override_by_id:
            raise ValueError(f"Duplicate override id: {cid}")
        override_by_id[cid] = rec

    merged: List[dict] = []
    replaced = 0
    base_ids = set()
    for rec in base_rows:
        cid = rec.get("id")
        base_ids.add(cid)
        if cid in override_by_id:
            merged.append(override_by_id[cid])
            replaced += 1
        else:
            merged.append(rec)

    extra_ids = [cid for cid in override_by_id if cid not in base_ids]
    if extra_ids and not args.allow_new_ids:
        raise ValueError(
            "Override file contains ids not present in base file: "
            + ", ".join(extra_ids[:10])
            + (" ..." if len(extra_ids) > 10 else "")
        )
    if args.allow_new_ids:
        for cid in extra_ids:
            merged.append(override_by_id[cid])

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for rec in merged:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "base_jsonl": str(args.base_jsonl),
                "override_jsonl": str(args.override_jsonl),
                "out_jsonl": str(args.out_jsonl),
                "base_rows": len(base_rows),
                "override_rows": len(override_by_id),
                "replaced": replaced,
                "appended_new_ids": len(extra_ids) if args.allow_new_ids else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
