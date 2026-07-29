#!/usr/bin/env python3
"""
Strict pre-launch validator for CATS evaluator input JSONL files.

This is intended to be a hard gate before expensive local-committee runs.
It validates:
  - strict JSONL parsing
  - required top-level fields and types
  - non-empty model_output after sanitization checks
  - retrieved_docs / per_doc_notes structure and doc_id consistency
  - conflict type encoding consistency
  - optional exact alignment against a canonical gold benchmark JSONL

Example:
  python scripts/validate_eval_input_jsonl.py \
    --input inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen7b/e2e/minimal/sft/input.jsonl \
    --mode benchmark_prepped \
    --gold data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl \
    --expected-rows 736
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CONFLICT_TYPE_STR_MAP = {
    "no conflict": 1,
    "complementary information": 2,
    "conflicting opinions and research outcomes": 3,
    "conflicting opinions or research outcomes": 3,
    "conflict due to outdated information": 4,
    "conflict due to outdated information (temporal conflict)": 4,
    "conflict due to misinformation": 5,
}

VALID_CONFLICT_TYPES = {1, 2, 3, 4, 5}
MODEL_OUTPUT_FIELDS = {"model_output", "raw", "expected_response.answer"}
THINK_MARKERS = (
    "<think>",
    "</think>",
    "stage 3 - answer generation",
    "stage 3: answer generation",
    "stage 3 answer generation:",
    "[[end-of-answer]]",
    "[[end of answer]]",
)


def is_refusal_sentinel_row(row: Dict[str, Any]) -> bool:
    gold_answer = row.get("gold_answer")
    return (
        row.get("conflict_category_id") == -1
        and row.get("answerable_under_evidence") is False
        and isinstance(gold_answer, str)
        and not gold_answer.strip()
    )


def load_jsonl_strict(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                raise ValueError(f"{path}: blank line at {lineno}")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON at line {lineno}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {lineno} is {type(obj).__name__}, expected object")
            rows.append(obj)
    return rows


def require(condition: bool, issues: List[str], message: str) -> None:
    if not condition:
        issues.append(message)


def require_non_empty_str(
    row: Dict[str, Any],
    key: str,
    issues: List[str],
    sample_prefix: str,
) -> None:
    val = row.get(key)
    require(isinstance(val, str), issues, f"{sample_prefix}: `{key}` must be a string")
    if isinstance(val, str):
        require(bool(val.strip()), issues, f"{sample_prefix}: `{key}` must be non-empty")


def canonical_conflict_type_id(conflict_type: Any) -> int | None:
    if not isinstance(conflict_type, str):
        return None
    return CONFLICT_TYPE_STR_MAP.get(conflict_type.strip().lower())


def compare_jsonish(lhs: Any, rhs: Any) -> bool:
    return json.dumps(lhs, sort_keys=True, ensure_ascii=False) == json.dumps(
        rhs, sort_keys=True, ensure_ascii=False
    )


def validate_row(
    row: Dict[str, Any],
    idx: int,
    *,
    mode: str,
    gold_row: Dict[str, Any] | None,
    repo_root: Path,
) -> List[str]:
    issues: List[str] = []
    sample_id = row.get("id", f"<missing-id-row-{idx}>")
    sample_prefix = f"row {idx} id={sample_id}"

    required_common = [
        "id",
        "query",
        "retrieved_docs",
        "conflict_type",
        "conflict_reason",
        "gold_answer",
        "per_doc_notes",
        "answerable_under_evidence",
        "conflict_category_id",
        "model_output",
    ]
    for key in required_common:
        require(key in row, issues, f"{sample_prefix}: missing required field `{key}`")

    require_non_empty_str(row, "id", issues, sample_prefix)
    require_non_empty_str(row, "query", issues, sample_prefix)
    require_non_empty_str(row, "conflict_type", issues, sample_prefix)
    require_non_empty_str(row, "conflict_reason", issues, sample_prefix)
    require(isinstance(row.get("gold_answer"), str), issues, f"{sample_prefix}: `gold_answer` must be a string")
    require_non_empty_str(row, "model_output", issues, sample_prefix)

    aue = row.get("answerable_under_evidence")
    require(isinstance(aue, bool), issues, f"{sample_prefix}: `answerable_under_evidence` must be bool")

    ctype_id = row.get("conflict_category_id")
    require(isinstance(ctype_id, int), issues, f"{sample_prefix}: `conflict_category_id` must be int")
    mapped_id = canonical_conflict_type_id(row.get("conflict_type"))
    require(
        mapped_id is not None,
        issues,
        f"{sample_prefix}: unknown `conflict_type` string {row.get('conflict_type')!r}",
    )
    if isinstance(ctype_id, int):
        if ctype_id not in VALID_CONFLICT_TYPES:
            if is_refusal_sentinel_row(row):
                require(
                    mapped_id in VALID_CONFLICT_TYPES,
                    issues,
                    f"{sample_prefix}: refusal sentinel row uses `conflict_category_id=-1` but has unknown `conflict_type` mapping",
                )
            else:
                issues.append(
                    f"{sample_prefix}: `conflict_category_id`={ctype_id} is outside 1..5 and is not an allowed refusal sentinel"
                )
        if mapped_id is not None and ctype_id in VALID_CONFLICT_TYPES:
            require(
                ctype_id == mapped_id,
                issues,
                f"{sample_prefix}: `conflict_category_id`={ctype_id} disagrees with `conflict_type`={row.get('conflict_type')!r} (canonical id {mapped_id})",
            )

    model_output = row.get("model_output")
    if isinstance(model_output, str):
        stripped = model_output.strip()
        require(bool(stripped), issues, f"{sample_prefix}: `model_output` is empty/whitespace")
        lowered = stripped.lower()
        bad_markers = [marker for marker in THINK_MARKERS if marker in lowered]
        require(
            not bad_markers,
            issues,
            f"{sample_prefix}: `model_output` still contains scaffolding markers {bad_markers}",
        )

    docs = row.get("retrieved_docs")
    require(isinstance(docs, list), issues, f"{sample_prefix}: `retrieved_docs` must be list")
    doc_ids: List[str] = []
    if isinstance(docs, list):
        require(bool(docs), issues, f"{sample_prefix}: `retrieved_docs` must be non-empty")
        for doc_idx, doc in enumerate(docs, start=1):
            prefix = f"{sample_prefix}: retrieved_docs[{doc_idx}]"
            require(isinstance(doc, dict), issues, f"{prefix} must be an object")
            if not isinstance(doc, dict):
                continue
            for key in ("doc_id", "snippet"):
                require(
                    isinstance(doc.get(key), str) and bool(doc.get(key, "").strip()),
                    issues,
                    f"{prefix}: `{key}` must be a non-empty string",
                )
            if mode == "benchmark_prepped":
                for key in ("source_url", "timestamp"):
                    require(
                        isinstance(doc.get(key), str),
                        issues,
                        f"{prefix}: `{key}` must be a string in benchmark mode",
                    )
            did = doc.get("doc_id")
            if isinstance(did, str) and did.strip():
                doc_ids.append(did.strip())
        if doc_ids:
            dupes = [did for did, count in Counter(doc_ids).items() if count > 1]
            require(not dupes, issues, f"{sample_prefix}: duplicate retrieved_docs doc_id values {sorted(dupes)}")

    notes = row.get("per_doc_notes")
    require(isinstance(notes, list), issues, f"{sample_prefix}: `per_doc_notes` must be list")
    note_ids: List[str] = []
    if isinstance(notes, list):
        require(bool(notes), issues, f"{sample_prefix}: `per_doc_notes` must be non-empty")
        for note_idx, note in enumerate(notes, start=1):
            prefix = f"{sample_prefix}: per_doc_notes[{note_idx}]"
            require(isinstance(note, dict), issues, f"{prefix} must be an object")
            if not isinstance(note, dict):
                continue
            require(
                isinstance(note.get("doc_id"), str) and bool(note.get("doc_id", "").strip()),
                issues,
                f"{prefix}: `doc_id` must be a non-empty string",
            )
            require(
                isinstance(note.get("verdict"), str) and bool(note.get("verdict", "").strip()),
                issues,
                f"{prefix}: `verdict` must be a non-empty string",
            )
            did = note.get("doc_id")
            if isinstance(did, str) and did.strip():
                note_ids.append(did.strip())
        if note_ids:
            dupes = [did for did, count in Counter(note_ids).items() if count > 1]
            require(not dupes, issues, f"{sample_prefix}: duplicate per_doc_notes doc_id values {sorted(dupes)}")

    if doc_ids and note_ids:
        require(
            set(doc_ids) == set(note_ids),
            issues,
            f"{sample_prefix}: retrieved_docs doc_id set does not exactly match per_doc_notes doc_id set",
        )

    if mode == "benchmark_prepped":
        for key in ("model_output_raw", "model_output_field", "model_output_source"):
            require(key in row, issues, f"{sample_prefix}: missing benchmark-prepped metadata field `{key}`")
        field = row.get("model_output_field")
        require(
            isinstance(field, str) and field in MODEL_OUTPUT_FIELDS,
            issues,
            f"{sample_prefix}: `model_output_field` must be one of {sorted(MODEL_OUTPUT_FIELDS)}",
        )
        src = row.get("model_output_source")
        require(
            isinstance(src, str) and bool(src.strip()),
            issues,
            f"{sample_prefix}: `model_output_source` must be a non-empty string",
        )
        if isinstance(src, str) and src.strip():
            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = repo_root / src_path
            require(src_path.exists(), issues, f"{sample_prefix}: `model_output_source` path does not exist: {src_path}")

    if gold_row is not None:
        gold_prefix = f"{sample_prefix}: gold-alignment"
        for key in (
            "id",
            "query",
            "retrieved_docs",
            "conflict_type",
            "conflict_reason",
            "gold_answer",
            "per_doc_notes",
            "answerable_under_evidence",
            "conflict_category_id",
        ):
            require(
                compare_jsonish(row.get(key), gold_row.get(key)),
                issues,
                f"{gold_prefix}: field `{key}` differs from canonical gold benchmark row",
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Evaluator input JSONL to validate.")
    parser.add_argument(
        "--mode",
        choices=("generic", "benchmark_prepped"),
        default="generic",
        help="Validation profile. benchmark_prepped adds benchmark-specific metadata and gold checks.",
    )
    parser.add_argument("--gold", help="Optional canonical gold JSONL to compare against by id and row order.")
    parser.add_argument("--expected-rows", type=int, help="Fail if row count differs.")
    parser.add_argument(
        "--max-issues",
        type=int,
        default=50,
        help="Maximum number of issues to print before truncating output.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        rows = load_jsonl_strict(input_path)
    except ValueError as exc:
        print(f"VALIDATION FAILED\n- {exc}")
        return 1

    if args.expected_rows is not None and len(rows) != args.expected_rows:
        print("VALIDATION FAILED")
        print(f"- expected {args.expected_rows} rows, found {len(rows)}")
        return 1

    ids = [row.get("id") for row in rows]
    dup_ids = sorted(str(k) for k, v in Counter(ids).items() if k is not None and v > 1)
    if dup_ids:
        print("VALIDATION FAILED")
        print(f"- duplicate ids detected: {dup_ids[:10]}")
        return 1

    gold_rows: List[Dict[str, Any]] | None = None
    gold_by_id: Dict[str, Dict[str, Any]] = {}
    if args.gold:
        gold_path = Path(args.gold).resolve()
        if not gold_path.exists():
            print(f"VALIDATION FAILED\n- gold file not found: {gold_path}")
            return 1
        try:
            gold_rows = load_jsonl_strict(gold_path)
        except ValueError as exc:
            print(f"VALIDATION FAILED\n- invalid gold file: {exc}")
            return 1
        gold_by_id = {row["id"]: row for row in gold_rows}
        gold_ids = [row.get("id") for row in gold_rows]
        if ids != gold_ids:
            print("VALIDATION FAILED")
            print("- input row id order does not exactly match gold benchmark order")
            input_set = set(ids)
            gold_set = set(gold_ids)
            missing = sorted(gold_set - input_set)[:10]
            extra = sorted(input_set - gold_set)[:10]
            if missing:
                print(f"- ids missing from input (first 10): {missing}")
            if extra:
                print(f"- ids not present in gold (first 10): {extra}")
            return 1

    all_issues: List[str] = []
    for idx, row in enumerate(rows, start=1):
        gold_row = gold_by_id.get(row.get("id")) if gold_by_id else None
        all_issues.extend(
            validate_row(
                row,
                idx,
                mode=args.mode,
                gold_row=gold_row,
                repo_root=repo_root,
            )
        )

    if all_issues:
        print("VALIDATION FAILED")
        for issue in all_issues[: args.max_issues]:
            print(f"- {issue}")
        if len(all_issues) > args.max_issues:
            print(f"- ... truncated {len(all_issues) - args.max_issues} additional issues")
        return 1

    print("VALIDATION PASSED")
    print(f"- input: {input_path}")
    print(f"- rows: {len(rows)}")
    print(f"- mode: {args.mode}")
    if args.gold:
        print(f"- gold alignment: exact match against {Path(args.gold).resolve()}")
    print("- duplicate ids: none")
    print("- row-level schema checks: passed")
    print("- model_output checks: passed")
    print("- doc/note consistency checks: passed")
    print("- conflict-type encoding checks: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
