#!/usr/bin/env python3
"""
Merge per-model Stage-2 collect files into one voted final JSONL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.committee_config import configure_committee_for_backend
from src.voting import COMMITTEE_MODELS, merge_stage2_votes


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rec_id = str(row.get("id") or "").strip()
        if rec_id:
            out[rec_id] = row
    return out


def parse_member(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --member {spec!r}; expected model_id=path")
    model_id, path = spec.split("=", 1)
    model_id = model_id.strip()
    path = path.strip()
    if not model_id or not path:
        raise ValueError(f"Invalid --member {spec!r}; expected model_id=path")
    return model_id, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge Stage-2 committee collect files.")
    ap.add_argument("--output", required=True, help="Merged Stage-2 final JSONL")
    ap.add_argument("--committee-config", required=True, help="Committee JSON config")
    ap.add_argument(
        "--member",
        action="append",
        required=True,
        help="Repeated model_id=path pair for each judge collect file",
    )
    ap.add_argument(
        "--mode",
        choices=["benchmark", "refusal", "conflicts"],
        default="benchmark",
        help="Stage-2 merge mode",
    )
    args = ap.parse_args()

    configure_committee_for_backend(
        backend="local_openai",
        config_path=args.committee_config,
    )

    members: Dict[str, Path] = {}
    for spec in args.member:
        model_id, path = parse_member(spec)
        members[model_id] = path

    missing_models = [model for model in COMMITTEE_MODELS if model not in members]
    if missing_models:
        raise SystemExit(f"Missing collect files for models: {missing_models}")

    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    index_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model, path in members.items():
        rows = read_jsonl(path)
        rows_by_model[model] = rows
        index_by_model[model] = build_index(rows)

    base_model = COMMITTEE_MODELS[0]
    base_rows = rows_by_model[base_model]
    vote_conflict_type = args.mode != "conflicts"
    is_refusal = args.mode == "refusal"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for base_row in base_rows:
            rec_id = str(base_row.get("id") or "").strip()
            model_records = {
                model: index_by_model[model].get(rec_id)
                for model in COMMITTEE_MODELS
            }
            merged = merge_stage2_votes(
                model_records,
                is_refusal=is_refusal,
                vote_conflict_type=vote_conflict_type,
            )

            out_row = dict(base_row)
            out_row["answerable_under_evidence"] = merged["answerable_under_evidence"]
            out_row["conflict_reason"] = merged.get("conflict_reason", "")

            if vote_conflict_type:
                if "conflict_type" in out_row:
                    out_row["_gold_conflict_type"] = out_row["conflict_type"]
                out_row["conflict_type"] = merged.get("conflict_type", "")

            for key in ("_ans_vote_tally", "_ans_winner_model", "_ct_vote_tally", "_ct_winner_model"):
                if key in merged:
                    out_row[key] = merged[key]

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} merged stage2 rows to {output_path}")


if __name__ == "__main__":
    main()
