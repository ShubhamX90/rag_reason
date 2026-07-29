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


def evidence_bucket(row: dict[str, Any], prep) -> str:
    supports = 0
    partial = 0
    valid_doc_ids = {doc.get("doc_id") for doc in row.get("retrieved_docs") or []}
    for note in row.get("per_doc_notes") or []:
        doc_id = note.get("doc_id")
        if doc_id not in valid_doc_ids:
            continue
        verdict = prep.normalize_verdict(note.get("verdict"))
        if verdict == "supports":
            supports += 1
        elif verdict == "partially supports":
            partial += 1
    if supports > 0:
        return "support_present"
    if partial > 0:
        return "partial_only"
    return "no_explicit_support_note"


def normalize_existing_stage3_rows(rows: list[dict[str, Any]], prep) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = deepcopy(row)
        new_row["conflict_type"] = prep.normalize_conflict_type(new_row.get("conflict_type"))
        normalized_rows.append(new_row)
    return normalized_rows


def alloc_counts(group_sizes: dict[Any, int], target: int) -> dict[Any, int]:
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
    staged["_ans_vote_tally"] = staged.get("_ans_vote_tally") or {"run_j_seed": 1.0}
    staged["_ans_winner_model"] = staged.get("_ans_winner_model") or "run_j_seed"
    staged["_abstain_vote_tally"] = staged.get("_abstain_vote_tally") or {}
    staged["_abstain_winner_model"] = staged.get("_abstain_winner_model") or ""
    staged["_run_j_origin"] = origin
    normalized, _ = prep.normalize_example(staged)
    return normalized, None


def summarize(rows: list[dict[str, Any]], prep) -> dict[str, dict[str, int]]:
    return {
        "by_prefix": dict(sorted(Counter(row_prefix(row["id"]) for row in rows).items())),
        "by_conflict_type": dict(sorted(Counter(row["conflict_type"] for row in rows).items())),
        "by_docs": dict(sorted(Counter(len(row.get("retrieved_docs") or []) for row in rows).items())),
        "by_evidence_bucket": dict(sorted(Counter(evidence_bucket(row, prep) for row in rows).items())),
    }


def select_rows(rows: list[dict[str, Any]], target_total: int, seed: int, prep) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["conflict_type"]].append(row)

    type_alloc = alloc_counts({key: len(group_rows) for key, group_rows in by_type.items()}, target_total)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for conflict_type in sorted(by_type):
        group_rows = by_type[conflict_type]
        target = type_alloc[conflict_type]
        subgrouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            subgrouped[(row_prefix(row["id"]), evidence_bucket(row, prep))].append(row)
        subgroup_alloc = alloc_counts({key: len(value) for key, value in subgrouped.items()}, target)
        for key in sorted(subgrouped):
            subgroup_rows = subgrouped[key][:]
            subgroup_rows.sort(key=lambda row: row["id"])
            rng.shuffle(subgroup_rows)
            selected.extend(subgroup_rows[: subgroup_alloc[key]])

    selected.sort(key=lambda row: row["id"])
    if len(selected) != target_total:
        raise ValueError(f"Expected {target_total} selected rows, found {len(selected)}")
    return selected


def split_rows(rows: list[dict[str, Any]], *, val_size: int, seed: int, prep) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row_prefix(row["id"]), row["conflict_type"], evidence_bucket(row, prep))].append(row)
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
        raise ValueError(f"Expected {val_size} val rows, found {len(val_rows)}")
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark-answerable augmentation/calibration splits for Run J.")
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
    parser.add_argument("--out_dir", type=Path, default=Path("data/splits/run_j"))
    parser.add_argument("--selection_fraction", type=float, default=0.26)
    parser.add_argument("--val_fraction_within_selected", type=float, default=0.128)
    parser.add_argument("--minimum_holdout_answerable", type=int, default=540)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0.25 <= args.selection_fraction <= 0.35):
        raise SystemExit("selection_fraction must be between 0.25 and 0.35 inclusive")
    if not (0.05 <= args.val_fraction_within_selected <= 0.3):
        raise SystemExit("val_fraction_within_selected must be between 0.05 and 0.3 inclusive")

    project_root = Path(__file__).resolve().parents[1]
    prep = load_prepare_data_module(project_root)

    current_train = normalize_existing_stage3_rows(read_jsonl(args.train_jsonl), prep)
    current_val = normalize_existing_stage3_rows(read_jsonl(args.val_jsonl), prep)
    benchmark_rows = read_jsonl(args.benchmark_jsonl)

    benchmark_answerable: list[dict[str, Any]] = []
    dropped = Counter()
    seen_ids = set()
    seen_queries = set()
    blocked_ids = {row["id"] for row in current_train + current_val}
    blocked_queries = {
        (row.get("query") or "").strip().lower()
        for row in current_train + current_val
        if (row.get("query") or "").strip()
    }

    for row in benchmark_rows:
        row_id = row.get("id") or ""
        query = (row.get("query") or "").strip().lower()
        if row_id in seen_ids:
            dropped["duplicate_benchmark_id"] += 1
            continue
        seen_ids.add(row_id)
        if query:
            if query in seen_queries:
                dropped["duplicate_benchmark_query"] += 1
                continue
            seen_queries.add(query)
        if row_id in blocked_ids:
            dropped["id_overlap_with_train_or_val"] += 1
            continue
        if query and query in blocked_queries:
            dropped["query_overlap_with_train_or_val"] += 1
            continue
        if not (row.get("gold_answer") or "").strip():
            dropped["blank_gold_answer"] += 1
            continue
        staged, reason = synthesize_answerable_stage3(row, prep, origin="benchmark_v2_pool")
        if staged is None:
            dropped[reason or "synthesis_failed"] += 1
            continue
        benchmark_answerable.append(staged)

    total_answerable = len(benchmark_answerable)
    target_total = int(round(total_answerable * args.selection_fraction))
    holdout_answerable = total_answerable - target_total
    if holdout_answerable < args.minimum_holdout_answerable:
        raise SystemExit(
            f"Selection would leave only {holdout_answerable} answerable holdout rows; "
            f"minimum required is {args.minimum_holdout_answerable}"
        )

    selected_all = select_rows(benchmark_answerable, target_total=target_total, seed=args.seed, prep=prep)
    selected_ids = {row["id"] for row in selected_all}
    holdout_residual = [row for row in benchmark_answerable if row["id"] not in selected_ids]

    val_size = int(round(len(selected_all) * args.val_fraction_within_selected))
    val_size = max(1, val_size)
    train_aug, val_aug = split_rows(selected_all, val_size=val_size, seed=args.seed + 1, prep=prep)

    for row in train_aug:
        row["_run_j_origin"] = "benchmark_v2_train_aug"
    for row in val_aug:
        row["_run_j_origin"] = "benchmark_v2_val_aug"

    stagewise_train_augmented = current_train + train_aug
    stagewise_val_combined = current_val + val_aug

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "benchmark_answerable_pool.jsonl", benchmark_answerable)
    write_jsonl(args.out_dir / "benchmark_selected_all.jsonl", selected_all)
    write_jsonl(args.out_dir / "benchmark_train_aug.jsonl", train_aug)
    write_jsonl(args.out_dir / "benchmark_val_aug.jsonl", val_aug)
    write_jsonl(args.out_dir / "benchmark_holdout_answerable_residual.jsonl", holdout_residual)
    write_jsonl(args.out_dir / "stagewise_train_augmented.jsonl", stagewise_train_augmented)
    write_jsonl(args.out_dir / "stagewise_val_combined.jsonl", stagewise_val_combined)

    summary = {
        "seed": args.seed,
        "selection_fraction": args.selection_fraction,
        "val_fraction_within_selected": args.val_fraction_within_selected,
        "minimum_holdout_answerable": args.minimum_holdout_answerable,
        "dropped": dict(sorted(dropped.items())),
        "benchmark_answerable_total": total_answerable,
        "selected_total": len(selected_all),
        "train_aug_rows": len(train_aug),
        "val_aug_rows": len(val_aug),
        "holdout_answerable_rows": len(holdout_residual),
        "selected_summary": summarize(selected_all, prep),
        "train_aug_summary": summarize(train_aug, prep),
        "val_aug_summary": summarize(val_aug, prep),
        "holdout_answerable_summary": summarize(holdout_residual, prep),
        "augmented_train_rows": len(stagewise_train_augmented),
        "combined_val_rows": len(stagewise_val_combined),
    }
    with (args.out_dir / "run_j_split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
