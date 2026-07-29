#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
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


def backup_file(src: Path, backup_root: Path) -> None:
    rel = src.as_posix().lstrip("/")
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def source_name(row_id: str) -> str:
    if row_id.startswith("trust_align_"):
        return "trust_align"
    return row_id.split("_", 1)[0]


def alloc_counts(group_sizes: dict[tuple[str, str], int], target: int) -> dict[tuple[str, str], int]:
    if target <= 0:
        return {key: 0 for key in group_sizes}
    total = sum(group_sizes.values())
    if total < target:
        raise ValueError(f"Cannot allocate {target} rows from only {total} available rows")
    raw = {key: (size * target / total) for key, size in group_sizes.items()}
    base = {key: min(group_sizes[key], int(math.floor(val))) for key, val in raw.items()}
    remaining = target - sum(base.values())
    remainders = sorted(
        (
            raw[key] - base[key],
            group_sizes[key] - base[key],
            key,
        )
        for key in group_sizes
    )
    remainders.reverse()
    idx = 0
    while remaining > 0:
        _, capacity_left, key = remainders[idx % len(remainders)]
        if capacity_left > 0 and base[key] < group_sizes[key]:
            base[key] += 1
            remaining -= 1
        idx += 1
        if idx > 100000:
            raise RuntimeError("Allocation loop did not converge")
    return base


def select_rows(
    refusal_rows: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in refusal_rows:
        rows_by_group[(source_name(row["id"]), row["conflict_type"])].append(row)

    for rows in rows_by_group.values():
        rows.sort(key=lambda r: r["id"])

    # Keep all scarce non-trust rows; fill remainder from trust-align refusals.
    forced_groups = {
        key for key in rows_by_group if key[0] != "trust_align"
    }
    selected: list[dict[str, Any]] = []
    for key in sorted(forced_groups):
        selected.extend(rows_by_group[key])

    remaining_needed = sample_size - len(selected)
    if remaining_needed < 0:
        raise ValueError("Forced non-trust refusal rows exceed requested sample size")

    trust_groups = {key: len(rows) for key, rows in rows_by_group.items() if key[0] == "trust_align"}
    trust_alloc = alloc_counts(trust_groups, remaining_needed)
    rng = random.Random(seed)
    for key in sorted(trust_alloc):
        rows = rows_by_group[key][:]
        rng.shuffle(rows)
        chosen = sorted(rows[: trust_alloc[key]], key=lambda r: r["id"])
        selected.extend(chosen)

    selected_by_id = {}
    for row in selected:
        selected_by_id[row["id"]] = row
    final = [selected_by_id[row_id] for row_id in sorted(selected_by_id)]
    if len(final) != sample_size:
        raise ValueError(f"Expected {sample_size} selected rows, found {len(final)}")
    return final


def split_selected_rows(
    selected_rows: list[dict[str, Any]],
    *,
    val_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_group[(source_name(row["id"]), row["conflict_type"])].append(row)
    group_sizes = {key: len(rows) for key, rows in by_group.items()}
    val_alloc = alloc_counts(group_sizes, val_size)

    rng = random.Random(seed + 1)
    val_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for key in sorted(by_group):
        rows = by_group[key][:]
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)
        val_ids = {row["id"] for row in rows[: val_alloc[key]]}
        for row in sorted(rows, key=lambda r: r["id"]):
            (val_rows if row["id"] in val_ids else train_rows).append(row)
    if len(val_rows) != val_size:
        raise ValueError(f"Expected {val_size} val rows, found {len(val_rows)}")
    return train_rows, val_rows


def refusal_abstain_reason(row: dict[str, Any], prep) -> str:
    notes = row.get("per_doc_notes") or []
    verdicts = [prep.normalize_verdict(note.get("verdict")) for note in notes]
    support = sum(v == "supports" for v in verdicts)
    partial = sum(v == "partially supports" for v in verdicts)
    irrelevant = sum(v == "irrelevant" for v in verdicts)
    conflict_reason = (row.get("conflict_reason") or "").strip()
    if support == 0 and partial == 0:
        core = "All retrieved documents are irrelevant to the query."
    elif support == 0:
        core = (
            "The retrieved documents are only partially supportive and do not provide enough complete, "
            "query-resolving evidence."
        )
    else:
        core = (
            "Some retrieved documents are relevant, but the overall evidence remains insufficient to answer "
            "the query confidently under the benchmark evidence policy."
        )
    details = f" Verdict pattern: supports={support}, partial={partial}, irrelevant={irrelevant}."
    if conflict_reason:
        details += f" Conflict note: {prep.trim_words(prep.sanitize_doc_ranges(conflict_reason), 60)}"
    return (core + details).strip()


def refusal_explanation(row: dict[str, Any], prep) -> str:
    notes = row.get("per_doc_notes") or []
    verdicts = [prep.normalize_verdict(note.get("verdict")) for note in notes]
    support = sum(v == "supports" for v in verdicts)
    partial = sum(v == "partially supports" for v in verdicts)
    if support == 0 and partial == 0:
        return "Refusal is required because none of the retrieved documents provide usable query-specific evidence."
    if support == 0:
        return (
            "Refusal is required because the retrieved evidence is only partially supportive and does not supply "
            "a complete answer."
        )
    return (
        "Refusal is required because, despite some relevant evidence, the available documents do not support a "
        "confident grounded answer."
    )


def synthesize_stage3_fields(row: dict[str, Any], prep) -> dict[str, Any]:
    row = deepcopy(row)
    doc_array = prep.build_doc_array(row)
    conflict_type = prep.normalize_conflict_type(row.get("conflict_type"))
    conflict_reason = prep.trim_words(
        prep.clean_reasoning_text(prep.sanitize_doc_ranges(row.get("conflict_reason") or "")),
        50,
    )
    if not conflict_reason:
        conflict_reason = prep.default_conflict_reason(conflict_type)
    conflict_reasoning = prep.heuristic_conflict_reasoning(doc_array)
    explanation = refusal_explanation(row, prep)
    row["expected_response"] = {
        "answer": "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
        "evidence": [],
        "abstain": True,
        "abstain_reason": refusal_abstain_reason(row, prep),
    }
    row["think"] = (
        "<think>\n"
        f"{json.dumps(doc_array, ensure_ascii=False, indent=2)}\n\n"
        f"{conflict_type} — {conflict_reason}\n"
        f"{prep.trim_words(prep.clean_reasoning_text(conflict_reasoning), 45)}\n\n"
        f"{prep.trim_words(prep.clean_reasoning_text(explanation), 60)}\n"
        "</think>"
    )
    row["_ans_vote_tally"] = row.get("_ans_vote_tally") or {}
    row["_ans_winner_model"] = row.get("_ans_winner_model") or "benchmark_refusal_import"
    row["_abstain_vote_tally"] = row.get("_abstain_vote_tally") or {"benchmark_refusal_import": 1}
    row["_abstain_winner_model"] = row.get("_abstain_winner_model") or "benchmark_refusal_import"
    return row


def to_stage_rows(row: dict[str, Any], prep) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage3 = synthesize_stage3_fields(row, prep)
    stage3, _ = prep.normalize_example(stage3)
    stage1 = {key: stage3[key] for key in ["id", "query", "retrieved_docs", "per_doc_notes", "conflict_type", "gold_answer"]}
    stage2 = {
        key: stage3[key]
        for key in [
            "id",
            "query",
            "retrieved_docs",
            "per_doc_notes",
            "conflict_type",
            "gold_answer",
            "answerable_under_evidence",
            "conflict_reason",
            "_ans_vote_tally",
            "_ans_winner_model",
        ]
    }
    return stage1, stage2, stage3


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "by_source": dict(sorted(Counter(source_name(row["id"]) for row in rows).items())),
        "by_conflict_type": dict(sorted(Counter(row["conflict_type"] for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=7)
    parser.add_argument("--benchmark-jsonl", type=Path, default=Path("data/Benchmark Dataset/benchmark_final_sanitized.jsonl"))
    parser.add_argument("--benchmark-manifest", type=Path, default=Path("data/Benchmark Dataset/benchmark_final_sanitized_manifest.json"))
    parser.add_argument("--benchmark-split-jsonl", type=Path, default=Path("data/splits/benchmark_final.jsonl"))
    parser.add_argument("--stage1-train", type=Path, default=Path("data/splits/stagewise_multi/train/stage1.jsonl"))
    parser.add_argument("--stage2-train", type=Path, default=Path("data/splits/stagewise_multi/train/stage2.jsonl"))
    parser.add_argument("--stage3-train", type=Path, default=Path("data/splits/stagewise_multi/train/stage3_final.jsonl"))
    parser.add_argument("--stage1-val", type=Path, default=Path("data/splits/stagewise_multi/val/stage1.jsonl"))
    parser.add_argument("--stage2-val", type=Path, default=Path("data/splits/stagewise_multi/val/stage2.jsonl"))
    parser.add_argument("--stage3-val", type=Path, default=Path("data/splits/stagewise_multi/val/stage3_final.jsonl"))
    parser.add_argument("--backup-root", type=Path, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prep = load_prepare_data_module(project_root)

    benchmark_rows = read_jsonl(args.benchmark_jsonl)
    split_rows = read_jsonl(args.benchmark_split_jsonl)
    benchmark_by_id = {row["id"]: row for row in benchmark_rows}
    split_by_id = {row["id"]: row for row in split_rows}
    if set(benchmark_by_id) != set(split_by_id):
        raise ValueError("Benchmark sanitized file and canonical split file do not contain the same IDs")

    refusal_rows = [
        deepcopy(row)
        for row in benchmark_rows
        if not bool(row.get("answerable_under_evidence"))
    ]
    if len(refusal_rows) < args.sample_size:
        raise ValueError(f"Only {len(refusal_rows)} gold refusals available, need {args.sample_size}")

    selected_rows = select_rows(refusal_rows, sample_size=args.sample_size, seed=args.seed)
    train_rows, val_rows = split_selected_rows(selected_rows, val_size=args.val_size, seed=args.seed)
    selected_ids = {row["id"] for row in selected_rows}

    train_stage1 = read_jsonl(args.stage1_train)
    train_stage2 = read_jsonl(args.stage2_train)
    train_stage3 = read_jsonl(args.stage3_train)
    val_stage1 = read_jsonl(args.stage1_val)
    val_stage2 = read_jsonl(args.stage2_val)
    val_stage3 = read_jsonl(args.stage3_val)

    existing_ids = {
        *(row["id"] for row in train_stage3),
        *(row["id"] for row in val_stage3),
    }
    overlap = selected_ids & set(existing_ids)
    if overlap:
        raise ValueError(f"Selected IDs already exist in stagewise splits: {sorted(overlap)[:10]}")

    new_train_stage1, new_train_stage2, new_train_stage3 = [], [], []
    new_val_stage1, new_val_stage2, new_val_stage3 = [], [], []

    for row in train_rows:
        s1, s2, s3 = to_stage_rows(row, prep)
        new_train_stage1.append(s1)
        new_train_stage2.append(s2)
        new_train_stage3.append(s3)
    for row in val_rows:
        s1, s2, s3 = to_stage_rows(row, prep)
        new_val_stage1.append(s1)
        new_val_stage2.append(s2)
        new_val_stage3.append(s3)

    reduced_benchmark_rows = [row for row in benchmark_rows if row["id"] not in selected_ids]
    reduced_split_rows = [row for row in split_rows if row["id"] not in selected_ids]

    if len(reduced_benchmark_rows) != len(benchmark_rows) - args.sample_size:
        raise RuntimeError("Benchmark removal count mismatch")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = args.backup_root or Path("backups") / f"benchmark_refusal_import_{timestamp}"
    for path in [
        args.benchmark_jsonl,
        args.benchmark_manifest,
        args.benchmark_split_jsonl,
        args.stage1_train,
        args.stage2_train,
        args.stage3_train,
        args.stage1_val,
        args.stage2_val,
        args.stage3_val,
    ]:
        backup_file(path, backup_root)

    write_jsonl(args.stage1_train, train_stage1 + new_train_stage1)
    write_jsonl(args.stage2_train, train_stage2 + new_train_stage2)
    write_jsonl(args.stage3_train, train_stage3 + new_train_stage3)
    write_jsonl(args.stage1_val, val_stage1 + new_val_stage1)
    write_jsonl(args.stage2_val, val_stage2 + new_val_stage2)
    write_jsonl(args.stage3_val, val_stage3 + new_val_stage3)
    write_jsonl(args.benchmark_jsonl, reduced_benchmark_rows)
    write_jsonl(args.benchmark_split_jsonl, reduced_split_rows)

    manifest = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    manifest["rows"] = len(reduced_benchmark_rows)
    notes = list(manifest.get("notes") or [])
    notes.append(
        f"On {timestamp}, {args.sample_size} gold-refusal rows were removed for stagewise SFT augmentation "
        f"({len(train_rows)} train, {len(val_rows)} val); see backups and import manifest."
    )
    manifest["notes"] = notes
    args.benchmark_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import_manifest = {
        "timestamp": timestamp,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "val_size": args.val_size,
        "backup_root": str(backup_root),
        "selected_summary": summarize(selected_rows),
        "train_summary": summarize(train_rows),
        "val_summary": summarize(val_rows),
        "selected_ids": [row["id"] for row in selected_rows],
        "train_ids": [row["id"] for row in train_rows],
        "val_ids": [row["id"] for row in val_rows],
        "pre_counts": {
            "benchmark_rows": len(benchmark_rows),
            "stage3_train_rows": len(train_stage3),
            "stage3_val_rows": len(val_stage3),
        },
        "post_counts": {
            "benchmark_rows": len(reduced_benchmark_rows),
            "stage3_train_rows": len(train_stage3) + len(new_train_stage3),
            "stage3_val_rows": len(val_stage3) + len(new_val_stage3),
        },
    }
    manifest_path = backup_root / "import_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(import_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(import_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
