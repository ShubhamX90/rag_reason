#!/usr/bin/env python3
"""Create a weighted multitask message JSONL by duplicating task rows.

Example:
  python scripts/make_weighted_multitask_messages.py \
    --input data/messages/train_stagewise_multitask_trace_text_messages.jsonl \
    --output data/messages/train_stagewise_multitask_trace_text_conflict3_answer2_messages.jsonl \
    --task-weight e2e_trace=1 \
    --task-weight doc_verdict=1 \
    --task-weight conflict_type=3 \
    --task-weight answer_only=2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_task_weight(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected TASK=N")
    task, raw_weight = value.split("=", 1)
    task = task.strip()
    try:
        weight = int(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer weight in {value!r}") from exc
    if not task:
        raise argparse.ArgumentTypeError("Task name cannot be empty")
    if weight < 0:
        raise argparse.ArgumentTypeError("Task weight must be >= 0")
    return task, weight


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--task-weight",
        action="append",
        type=parse_task_weight,
        default=[],
        help="TASK=N. Unspecified tasks default to weight 1.",
    )
    args = parser.parse_args()

    weights = dict(args.task_weight)
    rows_in = rows_out = 0
    in_counts: Counter[str] = Counter()
    out_counts: Counter[str] = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_f:
        for row in read_jsonl(args.input):
            rows_in += 1
            task = row.get("task") or "unknown"
            in_counts[task] += 1
            weight = weights.get(task, 1)
            for copy_idx in range(weight):
                new_row = dict(row)
                if weight > 1:
                    new_row["weighted_copy_idx"] = copy_idx
                out_f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                rows_out += 1
                out_counts[task] += 1

    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "input_task_counts": dict(sorted(in_counts.items())),
        "output_task_counts": dict(sorted(out_counts.items())),
        "weights": weights,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
