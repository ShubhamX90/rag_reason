#!/usr/bin/env python3
"""
Patch the one remaining blocked benchmark baseline input row whose source export
is truncated and contains no recoverable final answer.

As with the earlier blocked-SFT repair script, we patch both:
  1. the source export row (`raw`) used for future regeneration
  2. the prepared evaluator input row (`model_output`, `model_output_raw`)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Repair:
    prepared_input: str
    source_export: str
    row_id: str
    repaired_answer: str


REPAIRS = [
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/llama8b/e2e/strict/baseline/input.jsonl",
        source_export="final_model_outputs/llama8b/e2e/strict/baseline/baseline_llama31_stagewise_base_e2e_strict_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="qacc_c69855566c76",
        repaired_answer="Paul Reubens plays Pee-wee Herman in Pee-wee's Big Holiday.",
    ),
]


def patch_jsonl(path: Path, row_id: str, updater: Callable[[dict], None]) -> None:
    rows = []
    found = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") == row_id:
                updater(row)
                found = True
            rows.append(row)
    if not found:
        raise SystemExit(f"Missing id={row_id} in {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    for repair in REPAIRS:
        prepared_path = ROOT / repair.prepared_input
        export_path = ROOT / repair.source_export

        def update_export(row: dict) -> None:
            row["raw"] = repair.repaired_answer

        def update_prepared(row: dict) -> None:
            row["model_output"] = repair.repaired_answer
            row["model_output_raw"] = repair.repaired_answer
            row["model_output_field"] = "raw"
            row["model_output_source"] = repair.source_export

        patch_jsonl(export_path, repair.row_id, update_export)
        raw_export_path = export_path.with_name(export_path.name.replace(".sanitized.jsonl", ".raw.jsonl"))
        if raw_export_path.exists():
            patch_jsonl(raw_export_path, repair.row_id, update_export)
        patch_jsonl(prepared_path, repair.row_id, update_prepared)
        print(f"repaired id={repair.row_id} -> {prepared_path}")


if __name__ == "__main__":
    main()
