#!/usr/bin/env python3
"""
Merge per-model Stage-1 collect files into one voted final JSONL.

This avoids re-querying cache or model endpoints when we already have one
single-model Stage-1 output per judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.committee_config import configure_committee_for_backend
from src.voting import COMMITTEE_MODELS, merge_stage1_votes


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


def note_by_doc_id(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    notes = record.get("per_doc_notes") or []
    out: Dict[str, Dict[str, Any]] = {}
    for note in notes:
        if isinstance(note, dict):
            doc_id = str(note.get("doc_id") or "").strip()
            if doc_id:
                out[doc_id] = note
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
    ap = argparse.ArgumentParser(description="Merge Stage-1 committee collect files.")
    ap.add_argument("--output", required=True, help="Merged Stage-1 final JSONL")
    ap.add_argument("--committee-config", required=True, help="Committee JSON config")
    ap.add_argument(
        "--member",
        action="append",
        required=True,
        help="Repeated model_id=path pair for each judge collect file",
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for base_row in base_rows:
            rec_id = str(base_row.get("id") or "").strip()
            retrieved_docs = base_row.get("retrieved_docs") or []
            merged_row = dict(base_row)
            merged_notes: List[Dict[str, Any]] = []

            model_note_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for model in COMMITTEE_MODELS:
                row = index_by_model[model].get(rec_id)
                model_note_maps[model] = note_by_doc_id(row or {})

            for doc in retrieved_docs:
                doc_id = str(doc.get("doc_id") or "").strip()
                model_notes: Dict[str, Optional[Dict[str, Any]]] = {
                    model: model_note_maps[model].get(doc_id)
                    for model in COMMITTEE_MODELS
                }
                merged_notes.append(
                    merge_stage1_votes(
                        model_notes,
                        fallback_doc_id=doc_id,
                        fallback_source_url=str(doc.get("source_url", "") or ""),
                    )
                )

            merged_row["per_doc_notes"] = merged_notes
            fout.write(json.dumps(merged_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} merged stage1 rows to {output_path}")


if __name__ == "__main__":
    main()
