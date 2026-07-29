#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


CONFLICT_LABEL_ALIASES = {
    "no conflict": "No conflict",
    "complementary information": "Complementary information",
    "conflicting opinions or research outcomes": "Conflicting opinions or research outcomes",
    "conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
    "conflict due to outdated information": "Conflict due to outdated information",
    "conflict due to misinformation": "Conflict due to misinformation",
}


def normalize_conflict_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value.strip())
    return CONFLICT_LABEL_ALIASES.get(text.lower(), text)


def slugify_reason(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def parse_named_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=VALUE")
    name, raw_weight = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Name cannot be empty")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float weight in {value!r}") from exc
    if weight < 0:
        raise argparse.ArgumentTypeError("Weight must be >= 0")
    return name, weight


def verdict_counts(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for note in row.get("per_doc_notes") or []:
        counts[(note.get("verdict") or "").strip().lower()] += 1
    return counts


def compute_weight(
    message_row: dict[str, Any],
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, list[str]]:
    docs = len(meta.get("retrieved_docs") or [])
    answerable = bool(meta.get("answerable_under_evidence"))
    origin = (
        meta.get("_run_l_origin")
        or meta.get("_run_k_origin")
        or meta.get("_run_j_origin")
        or meta.get("_run_i_origin")
        or "main_train"
    )
    prefix = row_prefix(meta.get("id") or "")
    task = message_row.get("task") or "unknown"
    decision_task = task in {"e2e_trace", "answer_only"}
    conflict_type = normalize_conflict_type(meta.get("conflict_type"))

    counts = verdict_counts(meta)
    partial_only = counts.get("supports", 0) == 0 and counts.get("partially supports", 0) > 0

    weight = 1.0
    reasons: list[str] = []

    if answerable:
        if docs == args.answerable_exact_docs and args.answerable_exact_weight != 1.0:
            weight *= args.answerable_exact_weight
            reasons.append(f"answerable_docs_{docs}")
        if docs <= args.answerable_short_max_docs:
            weight *= args.answerable_short_weight
            reasons.append("answerable_short")
            if decision_task and args.decision_answerable_short_extra_weight != 1.0:
                weight *= args.decision_answerable_short_extra_weight
                reasons.append("decision_answerable_short")
        elif docs <= args.answerable_mid_max_docs:
            weight *= args.answerable_mid_weight
            reasons.append("answerable_mid")
        if partial_only and args.answerable_partial_only_weight != 1.0:
            weight *= args.answerable_partial_only_weight
            reasons.append("answerable_partial_only")
        if origin in {"benchmark_like_train_aug", "benchmark_v2_train_aug"} and args.benchmark_like_aug_weight != 1.0:
            weight *= args.benchmark_like_aug_weight
            reasons.append("benchmark_like_aug")
        origin_weight = args.answerable_origin_weights.get(origin, 1.0)
        if origin_weight != 1.0:
            weight *= origin_weight
            reasons.append(f"origin_{slugify_reason(origin)}")
        label_weight = args.answerable_conflict_label_weights.get(conflict_type, 1.0)
        if label_weight != 1.0:
            weight *= label_weight
            reasons.append(f"answerable_label_{slugify_reason(conflict_type)}")
        if partial_only:
            partial_label_weight = args.answerable_partial_only_conflict_label_weights.get(
                conflict_type, 1.0
            )
            if partial_label_weight != 1.0:
                weight *= partial_label_weight
                reasons.append(f"answerable_partial_only_label_{slugify_reason(conflict_type)}")
    else:
        if docs <= args.refusal_short_max_docs:
            weight *= args.refusal_short_weight
            reasons.append("refusal_short")
            if decision_task and args.decision_refusal_short_extra_weight != 1.0:
                weight *= args.decision_refusal_short_extra_weight
                reasons.append("decision_refusal_short")
        else:
            weight *= args.refusal_long_weight
            reasons.append("refusal_long")
        if prefix == "trust_align" and args.trust_align_refusal_weight != 1.0:
            weight *= args.trust_align_refusal_weight
            reasons.append("trust_align_refusal")

    return weight, reasons


def summarize(rows: list[dict[str, Any]], metadata_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_answerable = Counter()
    weighted_answerable = defaultdict(float)
    raw_docs = Counter()
    weighted_docs = defaultdict(float)
    raw_task_answerable = Counter()
    weighted_task_answerable = defaultdict(float)

    for row in rows:
        meta = metadata_by_id[row["id"]]
        answerable = bool(meta.get("answerable_under_evidence"))
        docs = len(meta.get("retrieved_docs") or [])
        weight = float(row.get("sample_weight", 1.0))
        raw_answerable[answerable] += 1
        weighted_answerable[answerable] += weight
        raw_docs[(docs, answerable)] += 1
        weighted_docs[(docs, answerable)] += weight
        key = (row.get("task") or "unknown", answerable)
        raw_task_answerable[key] += 1
        weighted_task_answerable[key] += weight

    return {
        "raw_answerable": {str(k): v for k, v in sorted(raw_answerable.items())},
        "weighted_answerable": {str(k): round(v, 4) for k, v in sorted(weighted_answerable.items())},
        "raw_docs": {
            f"docs={docs}|answerable={answerable}": count
            for (docs, answerable), count in sorted(raw_docs.items())
        },
        "weighted_docs": {
            f"docs={docs}|answerable={answerable}": round(total, 4)
            for (docs, answerable), total in sorted(weighted_docs.items())
        },
        "raw_task_answerable": {
            f"task={task}|answerable={answerable}": count
            for (task, answerable), count in sorted(raw_task_answerable.items())
        },
        "weighted_task_answerable": {
            f"task={task}|answerable={answerable}": round(total, 4)
            for (task, answerable), total in sorted(weighted_task_answerable.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate message JSONL rows with Run I sample weights.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--answerable_exact_docs", type=int, default=5)
    parser.add_argument("--answerable_exact_weight", type=float, default=1.8)
    parser.add_argument("--answerable_short_max_docs", type=int, default=7)
    parser.add_argument("--answerable_short_weight", type=float, default=2.0)
    parser.add_argument("--decision_answerable_short_extra_weight", type=float, default=1.5)
    parser.add_argument("--answerable_mid_max_docs", type=int, default=10)
    parser.add_argument("--answerable_mid_weight", type=float, default=1.25)
    parser.add_argument("--answerable_partial_only_weight", type=float, default=1.4)
    parser.add_argument("--benchmark_like_aug_weight", type=float, default=1.5)
    parser.add_argument(
        "--answerable-origin-weight",
        action="append",
        type=parse_named_weight,
        default=[],
        help="Extra multiplier for answerable rows with a specific augmentation origin, using ORIGIN=WEIGHT.",
    )
    parser.add_argument(
        "--answerable-conflict-label-weight",
        action="append",
        type=parse_named_weight,
        default=[],
        help="Extra multiplier for answerable rows with a specific conflict label, using LABEL=WEIGHT.",
    )
    parser.add_argument(
        "--answerable-partial-only-conflict-label-weight",
        action="append",
        type=parse_named_weight,
        default=[],
        help=(
            "Extra multiplier for answerable partial-only rows with a specific conflict label, "
            "using LABEL=WEIGHT."
        ),
    )
    parser.add_argument("--refusal_short_max_docs", type=int, default=5)
    parser.add_argument("--refusal_short_weight", type=float, default=0.5)
    parser.add_argument("--decision_refusal_short_extra_weight", type=float, default=0.7)
    parser.add_argument("--refusal_long_weight", type=float, default=0.65)
    parser.add_argument("--trust_align_refusal_weight", type=float, default=0.8)
    args = parser.parse_args()

    args.answerable_origin_weights = dict(args.answerable_origin_weight)
    args.answerable_conflict_label_weights = {
        normalize_conflict_type(label): weight for label, weight in args.answerable_conflict_label_weight
    }
    args.answerable_partial_only_conflict_label_weights = {
        normalize_conflict_type(label): weight
        for label, weight in args.answerable_partial_only_conflict_label_weight
    }

    messages = read_jsonl(args.input)
    metadata_by_id = {row["id"]: row for row in read_jsonl(args.metadata_jsonl)}

    annotated: list[dict[str, Any]] = []
    group_counts = Counter()
    for row in messages:
        meta = metadata_by_id.get(row["id"])
        if meta is None:
            raise SystemExit(f"Missing metadata for message id={row['id']}")
        weight, reasons = compute_weight(row, meta, args)
        new_row = dict(row)
        new_row["sample_weight"] = round(weight, 6)
        if reasons:
            new_row["sample_weight_group"] = "|".join(reasons)
            group_counts[new_row["sample_weight_group"]] += 1
        else:
            new_row["sample_weight_group"] = "baseline"
            group_counts["baseline"] += 1
        annotated.append(new_row)

    write_jsonl(args.output, annotated)

    summary = summarize(annotated, metadata_by_id)
    summary["sample_weight_groups"] = dict(sorted(group_counts.items()))
    summary["answerable_origin_weights"] = args.answerable_origin_weights
    summary["answerable_conflict_label_weights"] = args.answerable_conflict_label_weights
    summary["answerable_partial_only_conflict_label_weights"] = (
        args.answerable_partial_only_conflict_label_weights
    )
    summary["input"] = str(args.input)
    summary["output"] = str(args.output)
    summary["metadata_jsonl"] = str(args.metadata_jsonl)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
