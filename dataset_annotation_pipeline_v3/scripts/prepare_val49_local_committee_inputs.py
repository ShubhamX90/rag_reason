#!/usr/bin/env python3
"""Prepare the 49-row OpenRouter validation split for local committee reruns.

The validation split already exists with OpenRouter-produced annotations under
data/splits/92p5_7p5/stagewise_multi/val/.  For a clean head-to-head rerun, this
script uses only the selected IDs and reloads the original normalized input
records, so the local committee never sees the prior OpenRouter per-doc notes,
conflict reasons, or expected responses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAL_FILE = PROJECT_ROOT / "data" / "splits" / "92p5_7p5" / "stagewise_multi" / "val" / "stage3_final.jsonl"
DEFAULT_CONFLICTS = PROJECT_ROOT / "data" / "normalized" / "conflicts_normalized.jsonl"
DEFAULT_REFUSALS = PROJECT_ROOT / "data" / "normalized" / "refusals_normalized.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "local_committee_val49" / "inputs"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not row_id:
            raise ValueError(f"Missing id in {path}")
        by_id[row_id] = row
    return by_id


def clean_input_record(row: Dict[str, Any], origin: str) -> Dict[str, Any]:
    """Keep only fields used by the stagewise prompts plus audit provenance."""
    return {
        "id": row["id"],
        "query": row.get("query", ""),
        "retrieved_docs": row.get("retrieved_docs", []),
        "conflict_type": row.get("conflict_type", ""),
        "gold_answer": row.get("gold_answer", "") or "",
        "_val49_origin": origin,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE)
    ap.add_argument("--conflicts-source", type=Path, default=DEFAULT_CONFLICTS)
    ap.add_argument("--refusals-source", type=Path, default=DEFAULT_REFUSALS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    val_rows = read_jsonl(args.val_file)
    conflicts_by_id = load_by_id(args.conflicts_source)
    refusals_by_id = load_by_id(args.refusals_source)

    all_rows: List[Dict[str, Any]] = []
    conflict_rows: List[Dict[str, Any]] = []
    refusal_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for val_row in val_rows:
        row_id = val_row.get("id")
        if row_id in conflicts_by_id:
            clean = clean_input_record(conflicts_by_id[row_id], "conflicts")
            conflict_rows.append(clean)
        elif row_id in refusals_by_id:
            clean = clean_input_record(refusals_by_id[row_id], "refusals")
            refusal_rows.append(clean)
        else:
            missing.append(str(row_id))
            continue
        all_rows.append(clean)

    if missing:
        raise SystemExit(f"Validation IDs missing from normalized sources: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "val49_all_input.jsonl", all_rows)
    write_jsonl(args.output_dir / "val49_conflicts_input.jsonl", conflict_rows)
    write_jsonl(args.output_dir / "val49_refusals_input.jsonl", refusal_rows)

    manifest = {
        "val_file": str(args.val_file),
        "conflicts_source": str(args.conflicts_source),
        "refusals_source": str(args.refusals_source),
        "outputs": {
            "all": str(args.output_dir / "val49_all_input.jsonl"),
            "conflicts": str(args.output_dir / "val49_conflicts_input.jsonl"),
            "refusals": str(args.output_dir / "val49_refusals_input.jsonl"),
        },
        "counts": {
            "all": len(all_rows),
            "conflicts": len(conflict_rows),
            "refusals": len(refusal_rows),
            "by_origin": dict(Counter(row["_val49_origin"] for row in all_rows)),
            "by_conflict_type": dict(Counter(row.get("conflict_type", "") for row in all_rows)),
        },
        "ordered_ids": [row["id"] for row in all_rows],
    }
    (args.output_dir / "val49_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))
    print(f"wrote inputs under {args.output_dir}")


if __name__ == "__main__":
    main()
