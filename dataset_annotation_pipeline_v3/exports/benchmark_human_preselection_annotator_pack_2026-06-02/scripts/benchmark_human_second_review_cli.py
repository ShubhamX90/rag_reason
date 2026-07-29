#!/usr/bin/env python3
"""
Interactive second review for benchmark human preselection.

This pass is intentionally lightweight:
- show the original query and retrieved snippets
- show the complete first-review annotation
- let the second reviewer accept it as-is, edit selected fields, or reject it

Two common modes:

1. Build second-review assignments from completed first-pass review files:

   python3 scripts/benchmark_human_second_review_cli.py \
     --make-assignments \
     --input data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl \
     --first-pass-reviews data/benchmark_build/human_preselection/reviews/reviewer_*_reviews.jsonl \
     --output-dir data/benchmark_build/human_preselection/second_review_assignments

2. Run second review for the current reviewer:

   python3 scripts/benchmark_human_second_review_cli.py \
     --input data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl \
     --assignment data/benchmark_build/human_preselection/second_review_assignments/reviewer_1_second_review_ids.json \
     --output data/benchmark_build/human_preselection/second_reviews/reviewer_1_second_reviews.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import benchmark_human_preselection_cli as base


DEFAULT_INPUT = None
DEFAULT_FIRST_REVIEW_DIR = PROJECT_ROOT / "data/benchmark_build/human_preselection/reviews"
DEFAULT_SECOND_ASSIGNMENT_DIR = PROJECT_ROOT / "data/benchmark_build/human_preselection/second_review_assignments"
DEFAULT_SECOND_REVIEW_DIR = PROJECT_ROOT / "data/benchmark_build/human_preselection/second_reviews"


def load_first_review_files(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        file_reviews = base.load_reviews(Path(path))
        for rid, review in file_reviews.items():
            merged[rid] = review
    return merged


def iter_review_files(review_dir: Path) -> Iterable[Path]:
    patterns = [
        "reviewer_*_reviews.jsonl",
        "reviewer_*_reviews.cleaned.jsonl",
    ]
    seen = set()
    for pattern in patterns:
        for path in sorted(review_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            yield path


def resolve_portable_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def candidate_first_review_dirs(input_path: Path, explicit_dir: Optional[str]) -> List[Path]:
    dirs: List[Path] = []
    if explicit_dir:
        dirs.append(Path(explicit_dir))

    dirs.append(DEFAULT_FIRST_REVIEW_DIR)
    dirs.append(input_path.parent.parent)

    # If the input comes from an export pack, prefer that pack's sibling review dir too.
    export_review_dir = input_path.parent.parent / "human_preselection" / "reviews"
    dirs.append(export_review_dir)

    seen = set()
    unique: List[Path] = []
    for path in dirs:
        norm = str(path.resolve()) if path.exists() else str(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def second_review_target(first_review: Dict[str, Any]) -> bool:
    decision = str(first_review.get("human_preselect_decision") or "")
    needs_second = bool(first_review.get("needs_second_reviewer"))
    if decision in {"reject", "borderline_reject"}:
        return False
    if decision == "accept" and not needs_second:
        return False
    return True


def choose_second_reviewer(
    row: Dict[str, Any],
    first_reviewer_id: int,
    queues: Dict[int, List[Dict[str, Any]]],
    source_counts: Dict[int, Counter[str]],
    seed: int,
) -> int:
    src = base.source_family(row)
    candidates = [rid for rid in queues if rid != first_reviewer_id]
    return min(
        candidates,
        key=lambda reviewer_id: (
            len(queues[reviewer_id]),
            source_counts[reviewer_id][src],
            base.stable_hash(f"second:{base.row_id(row)}:{first_reviewer_id}:{reviewer_id}", seed),
        ),
    )


def make_second_review_assignments(args: argparse.Namespace) -> None:
    if not args.first_pass_reviews:
        raise SystemExit("--make-assignments requires at least one --first-pass-reviews path")
    if not args.input:
        raise SystemExit("--make-assignments requires --input")

    rows = base.read_jsonl(Path(args.input))
    by_id = {base.row_id(row): row for row in rows if base.row_id(row)}
    first_reviews = load_first_review_files(args.first_pass_reviews)

    targets: List[Dict[str, Any]] = []
    if args.target_all_input_rows:
        for row in rows:
            rid = base.row_id(row)
            if not rid or rid not in first_reviews:
                continue
            targets.append(row)
        target_rule = "all input rows with an available first-pass review"
    else:
        for rid, first_review in first_reviews.items():
            if not second_review_target(first_review):
                continue
            row = by_id.get(rid)
            if row is None:
                continue
            targets.append(row)
        target_rule = "decision not in {reject, borderline_reject} and not (accept with needs_second_reviewer=false)"

    targets.sort(key=base.stable_sort_key)

    queues: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(1, args.reviewers + 1)}
    reviewer_source_counts: Dict[int, Counter[str]] = {idx: Counter() for idx in range(1, args.reviewers + 1)}
    second_assignments: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in targets:
        rid = base.row_id(row)
        first_review = first_reviews[rid]
        first_reviewer_id = int(first_review["reviewer_id"])
        second_reviewer_id = choose_second_reviewer(row, first_reviewer_id, queues, reviewer_source_counts, args.seed)
        payload = {
            "record_id": rid,
            "source_family": base.source_family(row),
            "first_reviewer_id": first_reviewer_id,
            "first_reviewer_first_name": first_review.get("reviewer_first_name", ""),
            "first_review_decision": first_review.get("human_preselect_decision", ""),
            "first_review_conflict_type": first_review.get("preliminary_conflict_type", ""),
        }
        second_assignments[second_reviewer_id].append(payload)
        queues[second_reviewer_id].append(row)
        reviewer_source_counts[second_reviewer_id][base.source_family(row)] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input": base.portable_path(args.input),
        "first_pass_reviews": [base.portable_path(path) for path in args.first_pass_reviews],
        "reviewers": args.reviewers,
        "seed": args.seed,
        "target_rule": target_rule,
        "selected_for_second_review": len(targets),
        "reviewer_files": {},
    }

    for reviewer_id in range(1, args.reviewers + 1):
        assignment_rows = second_assignments.get(reviewer_id, [])
        path = output_dir / f"reviewer_{reviewer_id}_second_review_ids.json"
        payload = {
            "reviewer_id": reviewer_id,
            "input": base.portable_path(args.input),
            "record_count": len(assignment_rows),
            "assignment_rows": assignment_rows,
            "source_counts": dict(Counter(row["source_family"] for row in assignment_rows)),
            "first_reviewer_counts": dict(Counter(str(row["first_reviewer_id"]) for row in assignment_rows)),
            "order_strategy": "source_balanced_excluding_first_reviewer",
        }
        base.write_json(path, payload)
        manifest["reviewer_files"][str(reviewer_id)] = str(path)

    manifest_path = output_dir / "second_review_assignment_manifest.json"
    base.write_json(manifest_path, manifest)
    print(f"wrote second-review assignments to {output_dir}")
    print(f"manifest: {manifest_path}")
    for reviewer_id in range(1, args.reviewers + 1):
        rows_for_reviewer = second_assignments.get(reviewer_id, [])
        counts = Counter(row["source_family"] for row in rows_for_reviewer)
        print(f"reviewer {reviewer_id}: {len(rows_for_reviewer)} records {dict(counts)}")


def read_second_assignment_payload(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("assignment_rows"), list):
        return raw
    raise ValueError(f"Second-review assignment must contain assignment_rows: {path}")


def load_second_assignment(path: Path) -> List[Dict[str, Any]]:
    return list(read_second_assignment_payload(path)["assignment_rows"])


def build_second_review_record(
    row: Dict[str, Any],
    first_review: Dict[str, Any],
    reviewer_first_name: str,
    reviewer_id: int,
    action: str,
    resolved_review: Dict[str, Any],
) -> Dict[str, Any]:
    out = copy.deepcopy(resolved_review)
    out["reviewer_first_name"] = reviewer_first_name
    out["reviewer_id"] = reviewer_id
    out["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["needs_second_reviewer"] = False
    out["second_review_action"] = action
    out["second_review_of_reviewer_id"] = first_review.get("reviewer_id")
    out["second_review_of_reviewer_first_name"] = first_review.get("reviewer_first_name", "")
    out["_first_review"] = copy.deepcopy(first_review)
    out["_candidate_source"] = row.get("_candidate_source", {})
    out["_retrieval_metadata"] = row.get("_retrieval_metadata", {})
    return out


def print_first_review_summary(first_review: Dict[str, Any]) -> None:
    width = base.terminal_width()
    print()
    base.print_section_header("First Review Annotation", width, base.Ansi.GREEN)
    base.print_kv("First reviewer", f"{first_review.get('reviewer_first_name', '-')} (ID {first_review.get('reviewer_id', '-')})", width)
    base.print_kv("Decision", base.badge(first_review.get("human_preselect_decision", "-")), width)
    base.print_kv("Conflict type", base.badge(first_review.get("preliminary_conflict_type", "-")), width)
    base.print_kv("Confidence", base.badge(first_review.get("preselection_confidence", "-")), width)
    base.print_kv("Retrieval quality", base.badge(first_review.get("retrieval_quality", "-")), width)
    base.print_kv("Evidence sufficiency", base.badge(first_review.get("evidence_sufficiency", "-")), width)
    base.print_kv("Conflict clarity", base.badge(first_review.get("conflict_clarity", "-")), width)
    base.print_kv("Query specificity", base.badge(first_review.get("query_specificity", "-")), width)
    base.print_kv("Source reliability", base.badge(first_review.get("source_reliability", "-")), width)
    base.print_kv("Relevant docs", base.badge(first_review.get("relevant_doc_count_bin", "-")), width)
    base.print_kv("Gold answer possible", first_review.get("gold_answer_possible", "-"), width)
    base.print_kv("Gold answer", first_review.get("human_gold_answer", "-"), width)
    base.print_kv("Needs second reviewer", first_review.get("needs_second_reviewer", "-"), width)
    if str(first_review.get("reject_reason") or "").strip():
        base.print_kv("Reject reason", first_review.get("reject_reason", ""), width)
    if str(first_review.get("reviewer_notes") or "").strip():
        base.print_kv("Reviewer notes", first_review.get("reviewer_notes", ""), width)


def prompt_second_review_action() -> str:
    print()
    print(base.section_title("Second review action"))
    print(base.option_style("1", "accept first review as-is"))
    print(base.option_style("2", "edit fields and save second review"))
    print(base.option_style("3", "reject this query now"))
    print()
    print(base.color("Navigation / display commands", base.Ansi.DIM))
    print(base.option_style("s", "skip this record"))
    print(base.option_style("p", "go to previous record"))
    print(base.option_style("q", "save and quit"))
    print(base.option_style("r", "redisplay record"))
    print(base.option_style("f", "toggle full/compact snippets for this record"))
    while True:
        raw = input(base.color("> ", base.Ansi.BOLD, base.Ansi.GREEN)).strip().lower()
        if raw in {"1", "2", "3", "s", "p", "q", "r", "f"}:
            return raw
        print(base.color("Please choose one of the listed options.", base.Ansi.RED))


def confirm_second_review(review: Dict[str, Any], action_label: str) -> Optional[Dict[str, Any]]:
    while True:
        base.print_review_summary(review)
        print()
        print(base.section_title(f"Second review action: {action_label}"))
        print(base.option_style("y", "save and move to next record"))
        print(base.option_style("e", "edit a field"))
        print(base.option_style("a", "restart this record"))
        print(base.option_style("r", "redisplay record"))
        print(base.option_style("s", "skip without saving"))
        print(base.option_style("q", "save existing reviews and quit"))
        raw = input(base.color("> ", base.Ansi.BOLD, base.Ansi.GREEN)).strip().lower()
        if raw in {"y", "yes", ""}:
            return review
        if raw == "e":
            base.edit_review_field(review)
            continue
        if raw == "a":
            return None
        if raw in {"r", "s", "q"}:
            return {"_command": raw}
        print(base.color("Please choose y, e, a, r, s, or q.", base.Ansi.RED))


def review_one_record(
    row: Dict[str, Any],
    first_review: Dict[str, Any],
    args: argparse.Namespace,
    reviewer_first_name: str,
    reviewer_id: int,
    index: int,
    total: int,
    reviewed_count: int,
) -> Optional[Dict[str, Any]]:
    show_taxonomy = not args.hide_taxonomy
    snippet_chars = args.snippet_chars
    compact_snippet_chars = args.compact_snippet_chars
    while True:
        base.clear_screen(not args.no_clear)
        base.render_record(row, index, total, reviewed_count, snippet_chars, show_taxonomy, args.show_search_extract)
        print_first_review_summary(first_review)

        action = prompt_second_review_action()
        if action in {"q", "s", "p"}:
            return {"_command": action}
        if action == "r":
            continue
        if action == "f":
            snippet_chars = base.toggled_snippet_chars(snippet_chars, compact_snippet_chars)
            continue

        if action == "1":
            draft = build_second_review_record(
                row=row,
                first_review=first_review,
                reviewer_first_name=reviewer_first_name,
                reviewer_id=reviewer_id,
                action="accept_first_review",
                resolved_review=copy.deepcopy(first_review),
            )
            confirmed = confirm_second_review(draft, "accept first review as-is")
        elif action == "2":
            draft = build_second_review_record(
                row=row,
                first_review=first_review,
                reviewer_first_name=reviewer_first_name,
                reviewer_id=reviewer_id,
                action="edited_fields",
                resolved_review=copy.deepcopy(first_review),
            )
            confirmed = confirm_second_review(draft, "edit fields")
        else:
            draft_review = copy.deepcopy(first_review)
            draft_review["human_preselect_decision"] = "reject"
            draft_review["reject_reason"] = base.prompt_text(
                "Reject reason",
                str(first_review.get("reject_reason") or ""),
            )
            draft_review["needs_second_reviewer"] = False
            draft = build_second_review_record(
                row=row,
                first_review=first_review,
                reviewer_first_name=reviewer_first_name,
                reviewer_id=reviewer_id,
                action="reject_query",
                resolved_review=draft_review,
            )
            confirmed = confirm_second_review(draft, "reject query")

        if confirmed is None:
            continue
        if confirmed.get("_command") == "r":
            continue
        return confirmed


def default_second_assignment_path(reviewer_id: int) -> Path:
    return DEFAULT_SECOND_ASSIGNMENT_DIR / f"reviewer_{reviewer_id}_second_review_ids.json"


def default_second_review_path(reviewer_id: int) -> Path:
    return DEFAULT_SECOND_REVIEW_DIR / f"reviewer_{reviewer_id}_second_reviews.jsonl"


def print_assignment_overview(
    reviewer_first_name: str,
    reviewer_id: int,
    assignment_rows: Sequence[Dict[str, Any]],
    reviews: Dict[str, Dict[str, Any]],
    assignment_path: Optional[Path],
    output: Path,
) -> None:
    width = base.terminal_width()
    print()
    base.print_rule(width)
    print(base.section_title("Second Review Assignment"))
    base.print_rule(width)
    base.print_kv("Reviewer", f"{reviewer_first_name} (ID {reviewer_id})", width)
    base.print_kv("Total assigned queries", len(assignment_rows), width)
    base.print_kv("Already reviewed", len(reviews), width)
    if assignment_path:
        base.print_kv("Assignment file", assignment_path, width)
    base.print_kv("Second-review output", output, width)
    print()
    print(base.section_title("Assigned queries per dataset"))
    counts = Counter(row["source_family"] for row in assignment_rows)
    for src in base.SOURCE_PRIORITY + sorted(set(counts) - set(base.SOURCE_PRIORITY)):
        print(f"  {base.color(src + ':', base.Ansi.BOLD, base.Ansi.CYAN):<32} {counts[src]}")
    print()
    print(base.section_title("Assigned queries by first reviewer"))
    first_counts = Counter(f"{row['first_reviewer_first_name']} ({row['first_reviewer_id']})" for row in assignment_rows)
    for label, count in sorted(first_counts.items()):
        print(f"  {base.color(label + ':', base.Ansi.BOLD, base.Ansi.MAGENTA):<32} {count}")
    base.print_rule(width, "-")


def save_second_reviews(path: Path, reviews: Dict[str, Dict[str, Any]], record_order: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rid in record_order:
            review = reviews.get(rid)
            if review is not None:
                f.write(json.dumps(review, ensure_ascii=False) + "\n")


def run_second_review(args: argparse.Namespace) -> None:
    reviewer_first_name, reviewer_id = base.prompt_reviewer_name(args)
    assignment_path = Path(args.assignment) if args.assignment else default_second_assignment_path(reviewer_id)
    if not assignment_path.exists():
        raise SystemExit(f"Second-review assignment file not found: {assignment_path}")

    output = Path(args.output) if args.output else default_second_review_path(reviewer_id)
    assignment_payload = read_second_assignment_payload(assignment_path)
    input_ref = args.input or assignment_payload.get("input")
    if not input_ref:
        raise SystemExit("No input provided and assignment file does not define an input path.")
    input_path = resolve_portable_path(str(input_ref))
    input_rows = {base.row_id(row): row for row in base.read_jsonl(input_path)}
    assignment_rows = list(assignment_payload["assignment_rows"])
    first_reviews: Dict[str, Dict[str, Any]] = {}
    used_review_dirs: List[Path] = []
    for review_dir in candidate_first_review_dirs(input_path, args.first_review_source_dir):
        if not review_dir.exists():
            continue
        used_review_dirs.append(review_dir)
        for maybe_path in iter_review_files(review_dir):
            first_reviews.update(base.load_reviews(maybe_path))

    reviews = base.load_reviews(output)
    record_order = [str(row["record_id"]) for row in assignment_rows]

    print_assignment_overview(reviewer_first_name, reviewer_id, assignment_rows, reviews, assignment_path, output)
    input("Press Enter to start second review...")

    index = 0
    while index < len(assignment_rows):
        assignment = assignment_rows[index]
        rid = str(assignment["record_id"])
        if rid in reviews and not args.review_all:
            index += 1
            continue
        row = input_rows.get(rid)
        first_review = first_reviews.get(rid)
        if row is None or first_review is None:
            if row is None:
                raise SystemExit(f"Missing source row for {rid} in input {input_path}")
            raise SystemExit(
                f"Missing first review for {rid}. Checked review dirs: "
                + ", ".join(str(path) for path in used_review_dirs)
            )

        result = review_one_record(row, first_review, args, reviewer_first_name, reviewer_id, index, len(assignment_rows), len(reviews))
        if result is None:
            index += 1
            continue
        command = result.get("_command")
        if command == "q":
            save_second_reviews(output, reviews, record_order)
            print(f"Saved {len(reviews)} second reviews to {output}")
            return
        if command == "s":
            index += 1
            continue
        if command == "p":
            index = max(0, index - 1)
            continue

        reviews[rid] = result
        save_second_reviews(output, reviews, record_order)
        index += 1

    save_second_reviews(output, reviews, record_order)
    print(f"Completed second-review queue. Saved {len(reviews)} rows to {output}")


def main() -> None:
    base.COLOR_ENABLED = True
    ap = argparse.ArgumentParser(description="Second-review CLI for benchmark human preselection")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--assignment", default=None, help="Second-review assignment JSON containing assignment_rows")
    ap.add_argument("--output", default=None, help="Second-review JSONL output path")
    ap.add_argument("--reviewer-first-name", default=None)
    ap.add_argument("--reviewer-id", type=int, default=None)
    ap.add_argument("--reviewers", type=int, default=7)
    ap.add_argument("--snippet-chars", type=int, default=0)
    ap.add_argument("--compact-snippet-chars", type=int, default=base.DEFAULT_COMPACT_SNIPPET_CHARS)
    ap.add_argument("--no-clear", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--hide-taxonomy", action="store_true")
    ap.add_argument("--show-search-extract", action="store_true")
    ap.add_argument("--review-all", action="store_true")

    ap.add_argument("--make-assignments", action="store_true")
    ap.add_argument("--first-pass-reviews", nargs="*", default=[])
    ap.add_argument("--output-dir", default=str(DEFAULT_SECOND_ASSIGNMENT_DIR))
    ap.add_argument("--seed", type=int, default=62002)
    ap.add_argument("--target-all-input-rows", action="store_true")

    ap.add_argument("--first-review-source-dir", default=str(DEFAULT_FIRST_REVIEW_DIR))
    args = ap.parse_args()

    base.COLOR_ENABLED = not args.no_color and "NO_COLOR" not in os.environ

    if args.make_assignments:
        make_second_review_assignments(args)
        return
    run_second_review(args)


if __name__ == "__main__":
    main()
