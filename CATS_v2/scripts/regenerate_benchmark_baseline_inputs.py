#!/usr/bin/env python3
"""
Regenerate the 48 benchmark baseline evaluator inputs from their recorded source
export JSONLs.

Why this script exists:
  - the prepared `inputs/.../baseline/input.jsonl` files are the launch
    artifacts used by CATS
  - a small subset can drift if `model_output` still contains trace scaffolding
  - each prepared file already records its originating export path in
    `model_output_source`, so we can rebuild deterministically from source

The script rewrites every existing baseline `input.jsonl` in place using the
same extraction logic as `prep_model_outputs_for_eval.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prep_model_outputs_for_eval import (
    collect_duplicate_ids,
    prepare_file,
    read_jsonl,
)

INPUT_ROOT = ROOT / "inputs" / "prepped_model_eval_inputs" / "benchmark_set_all_modes"
GOLD_PATH = ROOT / "data" / "benchmark" / "benchmark_final_v2_holdout_clean_736.jsonl"


def discover_baseline_inputs() -> list[Path]:
    return sorted(INPUT_ROOT.glob("**/baseline/input.jsonl"))


def source_export_for(prepared_input: Path) -> Path:
    rows = read_jsonl(prepared_input)
    if not rows:
        raise SystemExit(f"Prepared input is empty: {prepared_input}")
    source = rows[0].get("model_output_source")
    if not isinstance(source, str) or not source.strip():
        raise SystemExit(f"Missing model_output_source in: {prepared_input}")
    export_path = ROOT / source
    if not export_path.exists():
        raise SystemExit(f"Source export missing for {prepared_input}: {export_path}")
    return export_path


def normalize_model_output_source(prepared_input: Path, export_path: Path) -> None:
    rel_source = str(export_path.relative_to(ROOT))
    rows = []
    with prepared_input.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["model_output_source"] = rel_source
            rows.append(row)
    with prepared_input.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    baseline_inputs = discover_baseline_inputs()
    if not baseline_inputs:
        raise SystemExit(f"No baseline input.jsonl files found under {INPUT_ROOT}")

    gold_rows = read_jsonl(GOLD_PATH)
    gold_ids = collect_duplicate_ids(gold_rows, f"Gold file {GOLD_PATH}")
    gold_by_id = {row.get("id"): row for row in gold_rows if row.get("id")}

    print(f"gold_records={len(gold_rows)}")
    print(f"baseline_inputs={len(baseline_inputs)}")

    total_prepared = 0
    total_missing_gold = 0
    total_empty_answers = 0
    total_missing_exports = 0

    for prepared_input in baseline_inputs:
        export_path = source_export_for(prepared_input)
        count, missing_gold, empty_answers, missing_exports = prepare_file(
            gold_by_id=gold_by_id,
            gold_count=len(gold_ids),
            export_path=export_path,
            output_path=prepared_input,
            allow_expected_response_answer=False,
        )
        normalize_model_output_source(prepared_input, export_path)
        total_prepared += count
        total_missing_gold += missing_gold
        total_empty_answers += empty_answers
        total_missing_exports += missing_exports
        print(
            f"rewrote={prepared_input} source={export_path} "
            f"rows={count} missing_gold={missing_gold} "
            f"missing_exports={missing_exports} empty_answers={empty_answers}"
        )

    print("summary")
    print(f"  total_prepared={total_prepared}")
    print(f"  total_missing_gold={total_missing_gold}")
    print(f"  total_missing_exports={total_missing_exports}")
    print(f"  total_empty_answers={total_empty_answers}")


if __name__ == "__main__":
    main()
