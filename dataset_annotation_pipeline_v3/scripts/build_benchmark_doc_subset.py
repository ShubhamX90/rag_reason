#!/usr/bin/env python3
"""Build a deterministic smaller evidence set from retrieved benchmark docs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_exact10.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl"
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("record_id") or "").strip()


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def sample_bucket(record_id: str, docs: Sequence[Dict[str, Any]], take: int, seed: int, bucket_name: str) -> List[Dict[str, Any]]:
    if take > len(docs):
        raise ValueError(f"Cannot sample {take} docs from {bucket_name} bucket with {len(docs)} docs for {record_id}")
    keyed = sorted(
        docs,
        key=lambda doc: stable_hash(seed, record_id, bucket_name, doc.get("doc_id"), doc.get("_rank")),
    )
    return keyed[:take]


def subset_record(
    row: Dict[str, Any],
    input_path: Path,
    seed: int,
    top_window: int,
    bottom_window: int,
    top_take: int,
    bottom_take: int,
) -> Dict[str, Any]:
    rid = row_id(row)
    docs = row.get("retrieved_docs") or []
    expected = top_window + bottom_window
    if len(docs) != expected:
        raise ValueError(f"{rid} has {len(docs)} docs; expected {expected}")

    positioned_docs = []
    for position, doc in enumerate(docs, 1):
        positioned_doc = copy.deepcopy(doc)
        positioned_doc["_original_list_position"] = position
        positioned_docs.append(positioned_doc)

    top_docs = positioned_docs[:top_window]
    bottom_docs = positioned_docs[top_window : top_window + bottom_window]
    selected = sample_bucket(rid, top_docs, top_take, seed, "top") + sample_bucket(
        rid, bottom_docs, bottom_take, seed, "bottom"
    )
    selected = sorted(selected, key=lambda doc: int(doc.get("_original_list_position") or 9999))

    out = copy.deepcopy(row)
    out_docs: List[Dict[str, Any]] = []
    selected_original_positions: List[int] = []
    selected_original_ranks: List[int] = []
    selected_original_doc_ids: List[str] = []
    for position, doc in enumerate(selected, 1):
        new_doc = copy.deepcopy(doc)
        original_doc_id = str(new_doc.get("doc_id") or "")
        original_position = int(new_doc.get("_original_list_position") or position)
        original_rank = int(new_doc.get("_rank") or original_position)
        selected_original_positions.append(original_position)
        selected_original_ranks.append(original_rank)
        selected_original_doc_ids.append(original_doc_id)
        new_doc["_original_doc_id"] = original_doc_id
        new_doc["_original_search_rank"] = original_rank
        new_doc["_original_rank"] = original_position
        new_doc["_rank"] = original_position
        new_doc["_subset_position"] = position
        new_doc["doc_id"] = f"{rid}_doc_{position}"
        out_docs.append(new_doc)

    out["retrieved_docs"] = out_docs
    meta = dict(out.get("_retrieval_metadata") or {})
    meta.update(
        {
            "doc_subset_strategy": f"{top_take}_from_top{top_window}__{bottom_take}_from_bottom{bottom_window}",
            "doc_subset_seed": seed,
            "doc_subset_original_doc_count": len(docs),
            "doc_subset_selected_doc_count": len(out_docs),
            "doc_subset_selected_original_positions": selected_original_positions,
            "doc_subset_selected_original_ranks": selected_original_ranks,
            "doc_subset_selected_original_doc_ids": selected_original_doc_ids,
            "doc_subset_original_input": input_path.as_posix(),
        }
    )
    out["_retrieval_metadata"] = meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic 5-doc benchmark evidence subsets")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=62002)
    parser.add_argument("--top-window", type=int, default=5)
    parser.add_argument("--bottom-window", type=int, default=5)
    parser.add_argument("--top-take", type=int, default=2)
    parser.add_argument("--bottom-take", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = [
        subset_record(
            row=row,
            input_path=input_path,
            seed=args.seed,
            top_window=args.top_window,
            bottom_window=args.bottom_window,
            top_take=args.top_take,
            bottom_take=args.bottom_take,
        )
        for row in read_jsonl(input_path)
    ]
    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} rows to {output_path}")
    print(
        "strategy="
        f"{args.top_take}_from_top{args.top_window} + {args.bottom_take}_from_bottom{args.bottom_window}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
