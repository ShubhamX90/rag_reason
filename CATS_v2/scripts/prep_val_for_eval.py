#!/usr/bin/env python3
"""
prep_val_for_eval.py
--------------------
Convert CATS v2.0 val-split JSONL records into the format expected by
run_evaluation.py.

What this does:
  1. Maps conflict_type string → conflict_category_id int
  2. Injects model_output from expected_response.answer
     (gold/"perfect" model eval — model_output == expected answer)
     When expected_response.abstain is True, model_output is set to
     the canonical refusal string so the evaluator detects it correctly.
  3. Passes answerable_under_evidence through as-is (evaluator derives
     gold_answerable from per_doc_notes verdicts, which match it exactly).

Usage:
  python scripts/prep_val_for_eval.py \
    --input  data/splits/85_7p5_7p5/monolithic_multi/val/monolithic_final.jsonl \
    --output outputs/val_monolithic_e2e/input_prepped.jsonl

  python scripts/prep_val_for_eval.py \
    --input  data/splits/85_7p5_7p5/stagewise_multi/val/stage3_final.jsonl \
    --output outputs/val_stagewise_e2e/input_prepped.jsonl
"""

import argparse
import json
from pathlib import Path


# Map conflict_type string to conflict_category_id int (CATS taxonomy)
CONFLICT_TYPE_MAP = {
    "no conflict":                                   1,
    "complementary information":                     2,
    "conflicting opinions and research outcomes":    3,
    "conflicting opinions or research outcomes":     3,
    "conflicting opinions":                          3,
    "conflict due to outdated information":          4,
    "outdated information":                          4,
    "misinformation":                                5,
}

REFUSAL_TEXT = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"


def map_conflict_type(raw: str) -> int:
    key = (raw or "").strip().lower()
    if key in CONFLICT_TYPE_MAP:
        return CONFLICT_TYPE_MAP[key]
    # Partial match fallback
    for k, v in CONFLICT_TYPE_MAP.items():
        if k in key or key in k:
            return v
    print(f"  [WARN] unknown conflict_type={raw!r}, defaulting to 1")
    return 1


def prep_record(rec: dict) -> dict:
    out = dict(rec)

    # 1. Numeric conflict_category_id
    if "conflict_category_id" not in out:
        out["conflict_category_id"] = map_conflict_type(out.get("conflict_type", ""))

    # 2. model_output from expected_response
    if "model_output" not in out:
        er = out.get("expected_response") or {}
        if er.get("abstain", False):
            out["model_output"] = REFUSAL_TEXT
        else:
            answer = (er.get("answer") or "").strip()
            out["model_output"] = answer if answer else REFUSAL_TEXT

    return out


def main():
    parser = argparse.ArgumentParser(description="Prep val JSONL for CATS evaluator")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    prepped = [prep_record(r) for r in records]

    # Quick stats
    abstains = sum(1 for r in prepped if r["model_output"] == REFUSAL_TEXT)
    ctypes = {}
    for r in prepped:
        k = r["conflict_category_id"]
        ctypes[k] = ctypes.get(k, 0) + 1

    with open(out_path, "w", encoding="utf-8") as f:
        for r in prepped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(prepped)} records → {out_path}")
    print(f"  conflict_category_id dist: {dict(sorted(ctypes.items()))}")
    print(f"  model_output=REFUSAL (abstain=True): {abstains}/{len(prepped)}")


if __name__ == "__main__":
    main()
