#!/usr/bin/env python3
"""Reconstruct lost benchmark second-review files from documented acceptances.

This utility is intentionally narrow: it recreates a reviewer\'s assigned
second-review records as ``accept_first_review`` records after the reviewer has
confirmed that every assigned first review was accepted.  It refuses to replace
an existing non-empty output unless ``--overwrite`` is explicitly supplied.

The accompanying reconstruction manifest records this exceptional provenance;
the generated JSONL schema is otherwise identical to normal second-review CLI
output.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from benchmark_human_preselection_cli import REVIEWER_NAMES_BY_ID, load_reviews, read_jsonl
from benchmark_human_second_review_cli import build_second_review_record, read_second_assignment_payload, resolve_portable_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIRST_REVIEW_DIR = PROJECT_ROOT / "human_reviews/benchmark/first_pass/reviews"
ASSIGNMENT_DIR = PROJECT_ROOT / "human_reviews/benchmark/second_pass/assignments"
SECOND_REVIEW_DIR = PROJECT_ROOT / "human_reviews/benchmark/second_pass/second_reviews"
MANIFEST_PATH = PROJECT_ROOT / "human_reviews/benchmark/second_pass/reconstruction_manifest.json"


def load_canonical_first_reviews() -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    files = sorted(FIRST_REVIEW_DIR.glob("reviewer_*_reviews.cleaned.jsonl"))
    if not files:
        raise SystemExit(f"No canonical cleaned first-pass review files found in {FIRST_REVIEW_DIR}")
    for path in files:
        for record_id, review in load_reviews(path).items():
            if record_id in merged:
                raise SystemExit(f"Duplicate first-pass review ID across cleaned files: {record_id}")
            merged[record_id] = review
    return merged


def reconstruct_reviewer(reviewer_id: int, first_reviews: Dict[str, Dict[str, Any]], overwrite: bool) -> Dict[str, Any]:
    assignment_path = ASSIGNMENT_DIR / f"reviewer_{reviewer_id}_second_review_ids.json"
    if not assignment_path.exists():
        raise SystemExit(f"Second-review assignment not found: {assignment_path}")
    assignment = read_second_assignment_payload(assignment_path)
    if int(assignment.get("reviewer_id", 0)) != reviewer_id:
        raise SystemExit(f"Assignment reviewer ID mismatch in {assignment_path}")

    input_path = resolve_portable_path(str(assignment.get("input", "")))
    if not input_path.exists():
        raise SystemExit(f"Assigned source input not found: {input_path}")
    candidates = {str(row.get("id") or "").strip(): row for row in read_jsonl(input_path)}

    output_path = SECOND_REVIEW_DIR / f"reviewer_{reviewer_id}_second_reviews.jsonl"
    if output_path.exists() and output_path.stat().st_size and not overwrite:
        raise SystemExit(f"Refusing to replace non-empty output without --overwrite: {output_path}")

    reviewer_name = REVIEWER_NAMES_BY_ID.get(reviewer_id)
    if not reviewer_name:
        raise SystemExit(f"No reviewer name registered for reviewer ID {reviewer_id}")

    output_rows: List[Dict[str, Any]] = []
    seen = set()
    for assignment_row in assignment["assignment_rows"]:
        record_id = str(assignment_row.get("record_id") or "").strip()
        if not record_id or record_id in seen:
            raise SystemExit(f"Invalid or duplicate assignment ID: {record_id!r}")
        seen.add(record_id)
        candidate = candidates.get(record_id)
        first_review = first_reviews.get(record_id)
        if candidate is None or first_review is None:
            raise SystemExit(f"Missing candidate or canonical first review for {record_id}")
        if int(first_review.get("reviewer_id") or 0) != int(assignment_row.get("first_reviewer_id") or 0):
            raise SystemExit(f"First-reviewer mismatch for {record_id}")
        if int(first_review.get("reviewer_id") or 0) == reviewer_id:
            raise SystemExit(f"Self-review assignment detected for {record_id}")

        output_rows.append(
            build_second_review_record(
                row=candidate,
                first_review=first_review,
                reviewer_first_name=reviewer_name,
                reviewer_id=reviewer_id,
                action="accept_first_review",
                resolved_review=first_review,
            )
        )

    SECOND_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "reviewer_id": reviewer_id,
        "reviewer_first_name": reviewer_name,
        "assignment": assignment_path.relative_to(PROJECT_ROOT).as_posix(),
        "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "record_count": len(output_rows),
        "action": "accept_first_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-ids", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if len(set(args.reviewer_ids)) != len(args.reviewer_ids):
        raise SystemExit("Reviewer IDs must be unique")
    first_reviews = load_canonical_first_reviews()
    entries = [reconstruct_reviewer(reviewer_id, first_reviews, args.overwrite) for reviewer_id in args.reviewer_ids]
    prior_entries: List[Dict[str, Any]] = []
    if MANIFEST_PATH.exists():
        try:
            prior_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            prior_entries = list(prior_manifest.get("entries") or [])
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Existing reconstruction manifest is invalid JSON: {MANIFEST_PATH}") from exc
    replaced_ids = {entry["reviewer_id"] for entry in entries}
    merged_entries = [entry for entry in prior_entries if entry.get("reviewer_id") not in replaced_ids] + entries
    merged_entries.sort(key=lambda entry: int(entry["reviewer_id"]))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reconstruction of lost second-review JSONL files after reviewers confirmed every assigned first review was accepted as correct.",
        "canonical_first_pass_source": "human_reviews/benchmark/first_pass/reviews/reviewer_*_reviews.cleaned.jsonl",
        "entries": merged_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for entry in entries:
        print(f"wrote {entry['record_count']} accept-as-is records to {entry['output']}")
    print(f"wrote provenance manifest to {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
