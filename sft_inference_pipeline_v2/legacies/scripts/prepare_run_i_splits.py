#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_prepare_data_module(project_root: Path):
    module_path = project_root / "code" / "data" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_prefix(row_id: str) -> str:
    if row_id.startswith("trust_align_"):
        return "trust_align"
    if row_id.startswith("#"):
        return "#"
    return row_id.split("_", 1)[0]


def alloc_counts(group_sizes: dict[tuple[str, str], int], target: int) -> dict[tuple[str, str], int]:
    if target <= 0:
        return {key: 0 for key in group_sizes}
    total = sum(group_sizes.values())
    if total < target:
        raise ValueError(f"Cannot allocate {target} rows from only {total} available rows")
    raw = {key: (size * target / total) for key, size in group_sizes.items()}
    base = {key: min(group_sizes[key], int(math.floor(value))) for key, value in raw.items()}
    remaining = target - sum(base.values())
    order = sorted(
        (
            raw[key] - base[key],
            group_sizes[key] - base[key],
            key,
        )
        for key in group_sizes
    )
    order.reverse()
    idx = 0
    while remaining > 0:
        _, capacity_left, key = order[idx % len(order)]
        if capacity_left > 0 and base[key] < group_sizes[key]:
            base[key] += 1
            remaining -= 1
        idx += 1
        if idx > 100000:
            raise RuntimeError("Allocation loop did not converge")
    return base


def split_rows(rows: list[dict[str, Any]], *, val_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row_prefix(row["id"]), row["conflict_type"])].append(row)
    group_sizes = {key: len(group_rows) for key, group_rows in grouped.items()}
    val_alloc = alloc_counts(group_sizes, val_size)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key][:]
        group_rows.sort(key=lambda row: row["id"])
        rng.shuffle(group_rows)
        val_ids = {row["id"] for row in group_rows[: val_alloc[key]]}
        for row in sorted(group_rows, key=lambda row: row["id"]):
            if row["id"] in val_ids:
                val_rows.append(row)
            else:
                train_rows.append(row)
    if len(val_rows) != val_size:
        raise ValueError(f"Expected {val_size} calibration val rows, found {len(val_rows)}")
    return train_rows, val_rows


def evidence_doc_ids(row: dict[str, Any], prep) -> list[str]:
    valid_doc_ids = {doc.get("doc_id") for doc in row.get("retrieved_docs") or []}
    supports = []
    partial = []
    for note in row.get("per_doc_notes") or []:
        doc_id = note.get("doc_id")
        if doc_id not in valid_doc_ids:
            continue
        verdict = prep.normalize_verdict(note.get("verdict"))
        if verdict == "supports":
            supports.append(doc_id)
        elif verdict == "partially supports":
            partial.append(doc_id)
    evidence = supports or partial
    if evidence:
        return evidence
    docs = [doc.get("doc_id") for doc in row.get("retrieved_docs") or [] if doc.get("doc_id")]
    return docs[:1]


def synthesize_answerable_stage3(row: dict[str, Any], prep, *, origin: str) -> tuple[dict[str, Any] | None, str | None]:
    staged = deepcopy(row)
    gold_answer = prep.clean_expected_answer_text(staged.get("gold_answer") or "")
    if not gold_answer:
        return None, "blank_gold_answer"
    staged["expected_response"] = {
        "answer": gold_answer,
        "evidence": evidence_doc_ids(staged, prep),
        "abstain": False,
        "abstain_reason": "",
    }
    staged["think"] = staged.get("think") or ""
    staged["_ans_vote_tally"] = staged.get("_ans_vote_tally") or {"run_i_seed": 1.0}
    staged["_ans_winner_model"] = staged.get("_ans_winner_model") or "run_i_seed"
    staged["_abstain_vote_tally"] = staged.get("_abstain_vote_tally") or {}
    staged["_abstain_winner_model"] = staged.get("_abstain_winner_model") or ""
    staged["_run_i_origin"] = origin
    normalized, _ = prep.normalize_example(staged)
    return normalized, None


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "by_prefix": dict(sorted(Counter(row_prefix(row["id"]) for row in rows).items())),
        "by_conflict_type": dict(sorted(Counter(row["conflict_type"] for row in rows).items())),
        "by_docs": dict(sorted(Counter(len(row.get("retrieved_docs") or []) for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark-like augmentation/calibration splits for Run I.")
    parser.add_argument(
        "--train_jsonl",
        type=Path,
        default=Path("data/splits/stagewise_multi/train/stage3_final.jsonl"),
    )
    parser.add_argument(
        "--val_jsonl",
        type=Path,
        default=Path("data/splits/stagewise_multi/val/stage3_final.jsonl"),
    )
    parser.add_argument(
        "--benchmark_jsonl",
        type=Path,
        default=Path("data/Benchmark Dataset/benchmark_final_v2.jsonl"),
    )
    parser.add_argument(
        "--older_subset_jsonl",
        type=Path,
        default=Path("data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset.jsonl"),
    )
    parser.add_argument("--out_dir", type=Path, default=Path("data/splits/run_i"))
    parser.add_argument("--calibration_val_size", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prep = load_prepare_data_module(project_root)

    current_train = read_jsonl(args.train_jsonl)
    current_val = read_jsonl(args.val_jsonl)
    benchmark_rows = read_jsonl(args.benchmark_jsonl)
    older_rows = read_jsonl(args.older_subset_jsonl)

    benchmark_ids = {row["id"] for row in benchmark_rows}
    blocked_queries = {
        row["query"].strip().lower()
        for row in current_train + current_val + benchmark_rows
        if (row.get("query") or "").strip()
    }

    kept: list[dict[str, Any]] = []
    dropped = Counter()
    for row in older_rows:
        row_id = row.get("id") or ""
        query = (row.get("query") or "").strip().lower()
        if row_id in benchmark_ids:
            dropped["id_overlap_with_benchmark_v2"] += 1
            continue
        if query in blocked_queries:
            dropped["query_overlap_with_train_val_or_benchmark_v2"] += 1
            continue
        if row.get("answerable_under_evidence") is not True:
            dropped["not_answerable_under_evidence"] += 1
            continue
        staged, reason = synthesize_answerable_stage3(row, prep, origin="benchmark_like_pool")
        if staged is None:
            dropped[reason or "synthesis_failed"] += 1
            continue
        kept.append(staged)

    if len(kept) < args.calibration_val_size:
        raise SystemExit(
            f"Not enough disjoint rows after filtering: kept={len(kept)} "
            f"but calibration_val_size={args.calibration_val_size}"
        )

    train_aug, val_calibration = split_rows(
        kept,
        val_size=args.calibration_val_size,
        seed=args.seed,
    )

    for row in train_aug:
        row["_run_i_origin"] = "benchmark_like_train_aug"
    for row in val_calibration:
        row["_run_i_origin"] = "benchmark_like_val_calibration"

    stagewise_train_augmented = current_train + train_aug
    stagewise_val_combined = current_val + val_calibration

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "benchmark_like_pool_disjoint.jsonl", kept)
    write_jsonl(args.out_dir / "benchmark_like_train_aug.jsonl", train_aug)
    write_jsonl(args.out_dir / "benchmark_like_val_calibration.jsonl", val_calibration)
    write_jsonl(args.out_dir / "stagewise_train_augmented.jsonl", stagewise_train_augmented)
    write_jsonl(args.out_dir / "stagewise_val_combined.jsonl", stagewise_val_combined)

    summary = {
        "seed": args.seed,
        "calibration_val_size": args.calibration_val_size,
        "dropped": dict(sorted(dropped.items())),
        "kept_disjoint_total": len(kept),
        "train_aug_rows": len(train_aug),
        "val_calibration_rows": len(val_calibration),
        "train_aug_summary": summarize(train_aug),
        "val_calibration_summary": summarize(val_calibration),
        "augmented_train_rows": len(stagewise_train_augmented),
        "combined_val_rows": len(stagewise_val_combined),
    }
    with (args.out_dir / "run_i_split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
