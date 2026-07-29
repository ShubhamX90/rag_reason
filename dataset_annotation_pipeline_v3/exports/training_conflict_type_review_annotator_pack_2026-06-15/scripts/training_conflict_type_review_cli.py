#!/usr/bin/env python3
"""
Interactive human review of conflict-type labels for the 658-example training set.

This pass is intentionally narrower than benchmark preselection:
- show the query and retrieved evidence
- show the current conflict-type label already assigned by the committee
- let the reviewer either accept that label or change it
- save one lightweight review record per reviewer

It also supports assignment generation where each query is assigned to exactly
two reviewers.

Common usage:

1. Create balanced 2-review assignments for 3 reviewers:

   python3 scripts/training_conflict_type_review_cli.py \
     --make-assignments

2. Run the review UI for one reviewer:

   python3 scripts/training_conflict_type_review_cli.py \
     --assignment data/final_annotations/stagewise_multi/conflict_type_review/assignments/reviewer_1_ids.json \
     --output data/final_annotations/stagewise_multi/conflict_type_review/reviews/reviewer_1_reviews.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import benchmark_human_preselection_cli as base


DEFAULT_INPUT = PROJECT_ROOT / "data/final_annotations/stagewise_multi/stage3_final.jsonl"
DEFAULT_ASSIGNMENT_DIR = PROJECT_ROOT / "data/final_annotations/stagewise_multi/conflict_type_review/assignments"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data/final_annotations/stagewise_multi/conflict_type_review/reviews"
DEFAULT_COMPACT_SNIPPET_CHARS = 1800

REVIEWER_ROSTER = {
    "manan": 1,
    "atharv": 2,
    "parth": 3,
}

REVIEWER_ID_TO_NAME = {reviewer_id: name for name, reviewer_id in REVIEWER_ROSTER.items()}

CANONICAL_CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]

CONFLICT_TYPE_ALIASES = {
    "Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
}

CONFLICT_OPTIONS = [
    ("1", "No conflict"),
    ("2", "Complementary information"),
    ("3", "Conflicting opinions or research outcomes"),
    ("4", "Conflict due to outdated information"),
    ("5", "Conflict due to misinformation"),
]

CONFIDENCE_OPTIONS = [
    ("1", "high"),
    ("2", "medium"),
    ("3", "low"),
]

LABEL_REVIEW_DEFINITIONS = [
    (
        "1",
        "No conflict",
        "Retrieved documents provide equivalent or nearly equivalent answers to the same query. Minor wording or granularity differences do not count as conflict.",
    ),
    (
        "2",
        "Complementary information",
        "Retrieved documents provide different but mutually compatible answers or perspectives, often because the query is underspecified or has multiple valid answers.",
    ),
    (
        "3",
        "Conflicting opinions or research outcomes",
        "Retrieved documents provide non-compatible answers within the same scope, such as contradictory research findings, subjective disagreement, or competing claims.",
    ),
    (
        "4",
        "Conflict due to outdated information",
        "Retrieved documents disagree because some reflect an older state of affairs and others reflect newer updates.",
    ),
    (
        "5",
        "Conflict due to misinformation",
        "Retrieved documents disagree because at least one source appears false, misleading, or inaccurate relative to stronger evidence in the retrieved set.",
    ),
]

REVIEW_GUIDELINES = [
    "Read the query first, then inspect the retrieved snippets as a set before deciding.",
    "Judge from the retrieved evidence shown here; use outside knowledge only as a light sanity check.",
    "Be careful with underspecified queries: different entities, places, or timeframes may explain the mismatch.",
    "Do not get anchored by the current label. Keep it only if the evidence clearly supports it.",
    "Change the label only when the current one is materially wrong under the taxonomy above.",
    "If the case is mixed or unclear, choose the best label, lower your confidence, and leave a short note.",
]


def canonical_conflict_type(label: Any) -> str:
    raw = str(label or "").strip()
    if not raw:
        return ""
    return CONFLICT_TYPE_ALIASES.get(raw, raw)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return base.read_jsonl(path)


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    base.write_jsonl(path, rows)


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or "").strip()


def load_reviews(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    reviews: Dict[str, Dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            review = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid review JSON on line {line_no} of {path}: {exc}") from exc
        rid = str(review.get("id") or "").strip()
        if rid:
            reviews[rid] = review
    return reviews


def save_reviews(path: Path, reviews: Dict[str, Dict[str, Any]], record_order: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rid in record_order:
            review = reviews.get(rid)
            if review is not None:
                f.write(json.dumps(review, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def stable_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    label = canonical_conflict_type(row.get("conflict_type"))
    try:
        idx = CANONICAL_CONFLICT_TYPES.index(label)
    except ValueError:
        idx = len(CANONICAL_CONFLICT_TYPES)
    return idx, row_id(row)


def interleave_by_label(rows: Sequence[Dict[str, Any]], reviewer_id: int, seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, deque[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(canonical_conflict_type(row.get("conflict_type")), deque()).append(row)
    for label, bucket in list(grouped.items()):
        ordered = sorted(bucket, key=lambda row: base.stable_hash(f"{row_id(row)}:{reviewer_id}", seed))
        grouped[label] = deque(ordered)

    order = [label for label in CANONICAL_CONFLICT_TYPES if label in grouped] + sorted(set(grouped) - set(CANONICAL_CONFLICT_TYPES))
    if order:
        shift = (reviewer_id - 1) % len(order)
        order = order[shift:] + order[:shift]

    interleaved: List[Dict[str, Any]] = []
    while any(grouped[label] for label in order):
        for label in order:
            if grouped[label]:
                interleaved.append(grouped[label].popleft())
    return interleaved


def reviewer_target_loads(total_rows: int, reviewers: int) -> List[int]:
    total_reviews = total_rows * 2
    return base.apportion_counts(total_reviews, [1] * reviewers)


def choose_pair_for_row(
    *,
    row: Dict[str, Any],
    reviewer_ids: Sequence[int],
    reviewer_loads: Dict[int, int],
    reviewer_targets: Dict[int, int],
    pair_counts: Counter[Tuple[int, int]],
    pair_label_counts: Dict[Tuple[int, int], Counter[str]],
    seed: int,
) -> Tuple[int, int]:
    label = canonical_conflict_type(row.get("conflict_type"))
    pairs = list(itertools.combinations(reviewer_ids, 2))

    viable = [
        pair
        for pair in pairs
        if reviewer_loads[pair[0]] < reviewer_targets[pair[0]]
        and reviewer_loads[pair[1]] < reviewer_targets[pair[1]]
    ]
    if not viable:
        viable = pairs

    def key(pair: Tuple[int, int]) -> Tuple[float, float, int, str]:
        a, b = pair
        max_load_ratio = max(
            reviewer_loads[a] / max(1, reviewer_targets[a]),
            reviewer_loads[b] / max(1, reviewer_targets[b]),
        )
        return (
            pair_label_counts[pair][label],
            max_load_ratio,
            pair_counts[pair],
            base.stable_hash(f"pair:{row_id(row)}:{pair[0]}:{pair[1]}", seed),
        )

    return min(viable, key=key)


def load_assignment_payload(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("assignment_rows"), list):
        return raw
    raise ValueError(f"Assignment file must contain assignment_rows: {path}")


def load_assignment(path: Path) -> List[Dict[str, Any]]:
    return list(load_assignment_payload(path)["assignment_rows"])


def apply_assignment(rows: List[Dict[str, Any]], assignment_path: Optional[Path]) -> List[Dict[str, Any]]:
    if assignment_path is None:
        return sorted(rows, key=stable_sort_key)
    assignment_rows = load_assignment(assignment_path)
    by_id = {row_id(row): row for row in rows}
    missing = [item["record_id"] for item in assignment_rows if item["record_id"] not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} assignment IDs are missing from input; first missing: {preview}")

    ordered: List[Dict[str, Any]] = []
    for item in assignment_rows:
        row = dict(by_id[item["record_id"]])
        row["_review_assignment"] = item
        ordered.append(row)
    return ordered


def default_assignment_path(reviewer_id: int) -> Path:
    return DEFAULT_ASSIGNMENT_DIR / f"reviewer_{reviewer_id}_ids.json"


def default_review_path_for_reviewer(reviewer_id: int) -> Path:
    return DEFAULT_REVIEW_DIR / f"reviewer_{reviewer_id}_reviews.jsonl"


def validate_existing_reviews(
    reviews: Dict[str, Dict[str, Any]],
    reviewer_first_name: str,
    reviewer_id: int,
    output: Path,
) -> None:
    bad_rows: List[str] = []
    for rid, review in reviews.items():
        if review.get("reviewer_id") != reviewer_id or str(review.get("reviewer_first_name") or "").strip().lower() != reviewer_first_name:
            bad_rows.append(rid)
            if len(bad_rows) >= 5:
                break
    if bad_rows:
        preview = ", ".join(bad_rows)
        raise ValueError(
            f"Existing review file {output} contains reviews from a different reviewer. "
            f"First mismatched IDs: {preview}"
        )


def print_taxonomy(width: int) -> None:
    base.print_section_header("Conflict Taxonomy - Detailed Definitions", width, base.Ansi.CYAN)
    for option, label, definition in LABEL_REVIEW_DEFINITIONS:
        print()
        print(f"  {base.color(option + '.', base.Ansi.BOLD, base.Ansi.YELLOW)} {base.color(label, base.Ansi.BOLD, base.Ansi.CYAN)}")
        print(base.wrap(definition, width - 6, indent="      "))


def print_review_guidelines(width: int) -> None:
    base.print_section_header("Reviewer Guidelines", width, base.Ansi.GREEN)
    for bullet in REVIEW_GUIDELINES:
        print(base.wrap(f"- {bullet}", width - 6, indent="  "))
        print()


def print_training_screen_banner(width: int, index: int, total: int, reviewed_count: int) -> None:
    base.print_rule(width)
    title = "TRAINING CONFLICT-TYPE REVIEW"
    meta = f"record {index + 1}/{total} | reviewed {reviewed_count}"
    print(base.color(title.center(width), base.Ansi.BOLD, base.Ansi.WHITE))
    print(base.color(meta.center(width), base.Ansi.BOLD, base.Ansi.CYAN))
    base.print_rule(width)


def render_record(
    row: Dict[str, Any],
    index: int,
    total: int,
    reviewed_count: int,
    snippet_chars: int,
    show_taxonomy: bool,
) -> None:
    width = base.terminal_width()
    docs = row.get("retrieved_docs") or []
    rid = row_id(row)
    raw_label = str(row.get("conflict_type") or "")
    canon_label = canonical_conflict_type(raw_label)
    assignment = row.get("_review_assignment") or {}

    print_training_screen_banner(width, index, total, reviewed_count)

    base.print_section_header("Training Conflict-Type Review", width, base.Ansi.WHITE)
    print(base.color("Question", base.Ansi.BOLD, base.Ansi.YELLOW))
    print(base.color(base.wrap(row.get("query", ""), width - 4, indent="  "), base.Ansi.BOLD))
    print()
    base.print_kv("ID", rid, width)
    base.print_kv("Current conflict label", base.badge(canon_label or "-"), width)
    if raw_label and raw_label != canon_label:
        base.print_kv("Stored label variant", raw_label, width)
    base.print_kv("Retrieved docs", len(docs), width)
    if assignment:
        base.print_kv("Paired reviewer ID", assignment.get("paired_reviewer_id", "-"), width)

    if show_taxonomy:
        print_taxonomy(width)
        print_review_guidelines(width)

    snippet_mode = "full snippets" if snippet_chars <= 0 else f"compact snippets, {snippet_chars} chars max"
    base.print_section_header(f"Retrieved Documents ({snippet_mode})", width, base.Ansi.CYAN)
    for idx, doc in enumerate(docs, 1):
        url = doc.get("source_url") or doc.get("url") or ""
        title = f"d{idx} | {base.host(url)}"
        timestamp = doc.get("timestamp") or doc.get("date") or "-"
        snippet = str(doc.get("snippet") or "").strip()
        snippet = base.ellipsize(snippet, snippet_chars)

        print()
        print(base.color(f" d{idx} ".ljust(width, "-"), base.Ansi.BOLD, base.Ansi.CYAN))
        print(base.color(base.wrap(title, width - 4, indent="  "), base.Ansi.BOLD))
        print(base.color(base.wrap(f"URL: {url}", width - 4, indent="  "), base.Ansi.DIM))
        print(base.color(base.wrap(f"Timestamp: {timestamp}", width - 4, indent="  "), base.Ansi.DIM))
        base.print_minor_header("Snippet", base.Ansi.CYAN)
        print(base.wrap(snippet, width - 4, indent="    "))
        print(base.color("-" * width, base.Ansi.DIM, base.Ansi.CYAN))


def prompt_choice(label: str, options: Sequence[Tuple[str, str]], allow_commands: bool = True) -> str:
    print()
    print(base.section_title(label))
    for key, value in options:
        print(base.option_style(key, value))
    if allow_commands:
        print()
        print(base.color("Navigation / display commands", base.Ansi.DIM))
        print(base.option_style("s", "skip this record"))
        print(base.option_style("p", "go to previous record"))
        print(base.option_style("q", "save and quit"))
        print(base.option_style("r", "redisplay record"))
        print(base.option_style("f", "toggle full/compact snippets for this record"))
    valid = {key: value for key, value in options}
    while True:
        raw = input(base.color("> ", base.Ansi.BOLD, base.Ansi.GREEN)).strip().lower()
        if allow_commands and raw in {"s", "p", "q", "r", "f"}:
            return raw
        if raw in valid:
            return valid[raw]
        print(base.color("Please enter one of the listed options.", base.Ansi.RED))


def prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{base.section_title(label)}{suffix} {base.color('> ', base.Ansi.BOLD, base.Ansi.GREEN)}").strip()
    return raw if raw else default


def prompt_reviewer_identity(args: argparse.Namespace) -> Tuple[str, int]:
    reviewer_name = (args.reviewer_first_name or "").strip().lower()
    reviewer_id = args.reviewer_id

    if reviewer_name and reviewer_name in REVIEWER_ROSTER:
        mapped_id = REVIEWER_ROSTER[reviewer_name]
        if reviewer_id is not None and reviewer_id != mapped_id:
            print(base.color(
                f"Provided reviewer ID {reviewer_id} does not match reviewer name {reviewer_name}; using mapped ID {mapped_id}.",
                base.Ansi.YELLOW,
            ))
        reviewer_id = mapped_id

    while not reviewer_name:
        raw_name = prompt_text("Reviewer first name").strip().lower()
        if not raw_name:
            print(base.color("Please enter your first name.", base.Ansi.RED))
            continue
        if raw_name not in REVIEWER_ROSTER:
            allowed = ", ".join(sorted(REVIEWER_ROSTER))
            print(base.color(f"Unknown reviewer name. Allowed names: {allowed}", base.Ansi.RED))
            continue
        reviewer_name = raw_name
        reviewer_id = REVIEWER_ROSTER[reviewer_name]
        print()
        print(base.color(f"Reviewer matched {reviewer_name} -> ID {reviewer_id}", base.Ansi.BOLD, base.Ansi.GREEN))

    if reviewer_id is None:
        reviewer_id = REVIEWER_ROSTER.get(reviewer_name)
    if reviewer_id is None or not 1 <= reviewer_id <= args.reviewers:
        raise ValueError(f"Reviewer mapping is invalid for {reviewer_name!r}: {reviewer_id!r}")

    return reviewer_name, reviewer_id


def build_review_record(
    row: Dict[str, Any],
    reviewer_first_name: str,
    reviewer_id: int,
    label_action: str,
    reviewed_conflict_type: str,
    confidence: str,
    change_reason: str,
    notes: str,
) -> Dict[str, Any]:
    assignment = row.get("_review_assignment") or {}
    raw_label = str(row.get("conflict_type") or "")
    canon_label = canonical_conflict_type(raw_label)
    return {
        "id": row_id(row),
        "query": row.get("query", ""),
        "reviewer_first_name": reviewer_first_name,
        "reviewer_id": reviewer_id,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_action": label_action,
        "original_conflict_type_raw": raw_label,
        "original_conflict_type_canonical": canon_label,
        "reviewed_conflict_type": reviewed_conflict_type,
        "changed_label": reviewed_conflict_type != canon_label,
        "review_confidence": confidence,
        "change_reason": change_reason,
        "reviewer_notes": notes,
        "paired_reviewer_id": assignment.get("paired_reviewer_id"),
    }


REVIEW_EDIT_FIELDS = [
    ("1", "label_action", "Label action"),
    ("2", "reviewed_conflict_type", "Reviewed conflict type"),
    ("3", "review_confidence", "Reviewer confidence"),
    ("4", "change_reason", "Change reason"),
    ("5", "reviewer_notes", "Reviewer notes"),
]


def print_review_summary(review: Dict[str, Any]) -> None:
    width = base.terminal_width()
    print()
    base.print_section_header("Review Summary Before Save", width, base.Ansi.YELLOW)
    base.print_kv("Record ID", review.get("id", "-"), width)
    base.print_kv("Query", review.get("query", "-"), width)
    base.print_kv("Original canonical label", base.badge(review.get("original_conflict_type_canonical", "-")), width)
    print()
    for key, field, label in REVIEW_EDIT_FIELDS:
        value = review.get(field, "")
        label_text = base.color(f"{key}. {label + ':':<24}", base.Ansi.BOLD, base.Ansi.CYAN)
        print(f"  {label_text} {base.badge(value)}")
    print()


def edit_review_field(review: Dict[str, Any]) -> None:
    print_review_summary(review)
    print(base.section_title("Choose field to edit"))
    for key, _field, label in REVIEW_EDIT_FIELDS:
        print(base.option_style(key, label))
    print(base.option_style("c", "cancel edit"))
    raw = input(base.color("> ", base.Ansi.BOLD, base.Ansi.GREEN)).strip().lower()
    field_map = {key: field for key, field, _label in REVIEW_EDIT_FIELDS}
    field = field_map.get(raw)
    if raw == "c" or not field:
        return

    if field == "label_action":
        action = prompt_choice(
            "Label action",
            [("1", "accept_as_is"), ("2", "change_label")],
            allow_commands=False,
        )
        review["label_action"] = action
        if action == "accept_as_is":
            review["reviewed_conflict_type"] = review.get("original_conflict_type_canonical", "")
            review["change_reason"] = ""
    elif field == "reviewed_conflict_type":
        review[field] = prompt_choice("Reviewed conflict type", CONFLICT_OPTIONS, allow_commands=False)
        review["label_action"] = (
            "accept_as_is"
            if review[field] == review.get("original_conflict_type_canonical", "")
            else "change_label"
        )
        if review["label_action"] == "accept_as_is":
            review["change_reason"] = ""
    elif field == "review_confidence":
        review[field] = prompt_choice("Reviewer confidence", CONFIDENCE_OPTIONS, allow_commands=False)
    elif field == "change_reason":
        if review.get("label_action") != "change_label":
            print(base.color("Change reason is only used when the reviewer changes the label.", base.Ansi.YELLOW))
        else:
            review[field] = prompt_text("Short reason for label change", review.get(field, ""))
    elif field == "reviewer_notes":
        review[field] = prompt_text("Optional reviewer note", review.get(field, ""))

    review["changed_label"] = review.get("reviewed_conflict_type") != review.get("original_conflict_type_canonical")
    review["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()


def confirm_review(review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    while True:
        print_review_summary(review)
        print(base.section_title("Save this review?"))
        print(base.option_style("y", "save and move to next record"))
        print(base.option_style("e", "edit a field"))
        print(base.option_style("a", "redo annotation for this record"))
        print(base.option_style("r", "redisplay record"))
        print(base.option_style("s", "skip without saving"))
        print(base.option_style("q", "save existing reviews and quit"))
        raw = input(base.color("> ", base.Ansi.BOLD, base.Ansi.GREEN)).strip().lower()
        if raw in {"y", "yes", ""}:
            return review
        if raw == "e":
            edit_review_field(review)
            continue
        if raw == "a":
            return None
        if raw in {"r", "s", "q"}:
            return {"_command": raw}
        print(base.color("Please choose y, e, a, r, s, or q.", base.Ansi.RED))


def review_one_record(
    row: Dict[str, Any],
    args: argparse.Namespace,
    reviewer_first_name: str,
    reviewer_id: int,
    index: int,
    total: int,
    reviewed_count: int,
    existing_review: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    show_taxonomy = not args.hide_taxonomy
    snippet_chars = args.snippet_chars
    compact_snippet_chars = args.compact_snippet_chars
    while True:
        base.clear_screen(not args.no_clear)
        render_record(row, index, total, reviewed_count, snippet_chars, show_taxonomy)

        if existing_review:
            print()
            print(base.section_title("Existing saved review found for this record"))
            print(base.option_style("1", "edit the saved review"))
            print(base.option_style("2", "redo this review from scratch"))
            print(base.option_style("s", "skip and keep saved review as is"))
            print(base.option_style("p", "go to previous record"))
            print(base.option_style("q", "save existing reviews and quit"))
            print(base.option_style("r", "redisplay record"))
            print(base.option_style("f", "toggle full/compact snippets for this record"))
            while True:
                raw = input(base.color("> ", base.Ansi.BOLD, base.Ansi.GREEN)).strip().lower()
                if raw == "1":
                    confirmed = confirm_review(dict(existing_review))
                    if confirmed is None:
                        existing_review = None
                        break
                    if confirmed.get("_command") == "r":
                        break
                    return confirmed
                if raw == "2":
                    existing_review = None
                    break
                if raw in {"s", "p", "q"}:
                    return {"_command": raw}
                if raw == "r":
                    break
                if raw == "f":
                    snippet_chars = base.toggled_snippet_chars(snippet_chars, compact_snippet_chars)
                    break
                print(base.color("Please enter one of the listed options.", base.Ansi.RED))
            if existing_review:
                continue

        print()
        print(base.section_title("Enter Conflict-Type Review"))
        action = prompt_choice(
            "Label action",
            [("1", "accept_as_is"), ("2", "change_label")],
        )
        if action in {"q", "s", "p"}:
            return {"_command": action}
        if action == "r":
            continue
        if action == "f":
            snippet_chars = base.toggled_snippet_chars(snippet_chars, compact_snippet_chars)
            continue

        if action == "accept_as_is":
            reviewed_conflict_type = canonical_conflict_type(row.get("conflict_type"))
            change_reason = ""
        else:
            reviewed_conflict_type = prompt_choice("Reviewed conflict type", CONFLICT_OPTIONS, allow_commands=False)
            change_reason = prompt_text("Short reason for label change")

        confidence = prompt_choice("Reviewer confidence", CONFIDENCE_OPTIONS, allow_commands=False)
        notes = prompt_text("Optional reviewer note")

        review = build_review_record(
            row=row,
            reviewer_first_name=reviewer_first_name,
            reviewer_id=reviewer_id,
            label_action=action,
            reviewed_conflict_type=reviewed_conflict_type,
            confidence=confidence,
            change_reason=change_reason,
            notes=notes,
        )
        confirmed = confirm_review(review)
        if confirmed is None:
            continue
        if confirmed.get("_command") == "r":
            continue
        return confirmed


def print_assignment_overview(
    reviewer_first_name: str,
    reviewer_id: int,
    rows: Sequence[Dict[str, Any]],
    reviews: Dict[str, Dict[str, Any]],
    assignment_path: Optional[Path],
    output: Path,
) -> None:
    width = base.terminal_width()
    print()
    base.print_rule(width)
    print(base.section_title("Training Conflict-Type Reviewer Assignment"))
    base.print_rule(width)
    base.print_kv("Reviewer", f"{reviewer_first_name} (ID {reviewer_id})", width)
    base.print_kv("Total assigned queries", len(rows), width)
    base.print_kv("Already reviewed", len(reviews), width)
    base.print_kv("Remaining", max(0, len(rows) - len(reviews)), width)
    if assignment_path:
        base.print_kv("Assignment file", assignment_path, width)
    base.print_kv("Review output", output, width)
    print()
    print(base.section_title("Assigned queries per conflict type"))
    counts = Counter(canonical_conflict_type(row.get("conflict_type")) for row in rows)
    for label in CANONICAL_CONFLICT_TYPES:
        print(f"  {base.color(label + ':', base.Ansi.BOLD, base.Ansi.CYAN):<52} {counts[label]}")
    print()
    print(base.section_title("Paired reviewer counts"))
    pair_counts = Counter(str((row.get("_review_assignment") or {}).get("paired_reviewer_id", "-")) for row in rows)
    for paired_id in sorted(pair_counts, key=lambda x: (x == "-", x)):
        if paired_id == "-":
            label = "unknown:"
        else:
            paired_name = REVIEWER_ID_TO_NAME.get(int(paired_id), f"reviewer_{paired_id}")
            label = f"{paired_name} ({paired_id}):"
        print(f"  {base.color(label, base.Ansi.BOLD, base.Ansi.CYAN):<24} {pair_counts[paired_id]}")
    base.print_rule(width, "-")


def make_assignments(args: argparse.Namespace) -> None:
    if args.reviewers < 2:
        raise SystemExit("--reviewers must be >= 2 for two-review assignments")

    rows = read_jsonl(Path(args.input))
    rows = [row for row in rows if row_id(row)]
    rows.sort(key=stable_sort_key)

    reviewer_ids = list(range(1, args.reviewers + 1))
    target_loads_list = reviewer_target_loads(len(rows), args.reviewers)
    reviewer_targets = {reviewer_id: target_loads_list[reviewer_id - 1] for reviewer_id in reviewer_ids}
    reviewer_loads = {reviewer_id: 0 for reviewer_id in reviewer_ids}
    reviewer_rows: Dict[int, List[Dict[str, Any]]] = {reviewer_id: [] for reviewer_id in reviewer_ids}
    reviewer_label_counts: Dict[int, Counter[str]] = {reviewer_id: Counter() for reviewer_id in reviewer_ids}
    reviewer_pair_counts: Dict[int, Counter[str]] = {reviewer_id: Counter() for reviewer_id in reviewer_ids}
    pair_counts: Counter[Tuple[int, int]] = Counter()
    pair_label_counts: Dict[Tuple[int, int], Counter[str]] = defaultdict(Counter)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[canonical_conflict_type(row.get("conflict_type"))].append(row)

    ordered_rows: List[Dict[str, Any]] = []
    for label in CANONICAL_CONFLICT_TYPES + sorted(set(grouped) - set(CANONICAL_CONFLICT_TYPES)):
        bucket = grouped.get(label, [])
        bucket.sort(key=lambda row: base.stable_hash(row_id(row), args.seed))
        ordered_rows.extend(bucket)

    for row in ordered_rows:
        pair = choose_pair_for_row(
            row=row,
            reviewer_ids=reviewer_ids,
            reviewer_loads=reviewer_loads,
            reviewer_targets=reviewer_targets,
            pair_counts=pair_counts,
            pair_label_counts=pair_label_counts,
            seed=args.seed,
        )
        pair = tuple(sorted(pair))
        label = canonical_conflict_type(row.get("conflict_type"))
        pair_counts[pair] += 1
        pair_label_counts[pair][label] += 1

        a, b = pair
        for reviewer_id, paired_id in ((a, b), (b, a)):
            reviewer_loads[reviewer_id] += 1
            reviewer_label_counts[reviewer_id][label] += 1
            reviewer_pair_counts[reviewer_id][str(paired_id)] += 1
            row_with_assignment = dict(row)
            row_with_assignment["_review_assignment"] = {
                "record_id": row_id(row),
                "paired_reviewer_id": paired_id,
                "original_conflict_type_raw": row.get("conflict_type", ""),
                "original_conflict_type_canonical": label,
            }
            reviewer_rows[reviewer_id].append(row_with_assignment)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input": base.portable_path(args.input),
        "reviewers": args.reviewers,
        "reviewer_roster": {str(reviewer_id): REVIEWER_ID_TO_NAME.get(reviewer_id, f'reviewer_{reviewer_id}') for reviewer_id in reviewer_ids},
        "seed": args.seed,
        "rows": len(rows),
        "total_review_assignments": len(rows) * 2,
        "reviewer_target_loads": reviewer_targets,
        "reviewer_final_loads": reviewer_loads,
        "pair_counts": {f"{a}-{b}": count for (a, b), count in sorted(pair_counts.items())},
        "canonical_label_distribution": dict(Counter(canonical_conflict_type(row.get("conflict_type")) for row in rows)),
        "reviewer_files": {},
    }

    for reviewer_id in reviewer_ids:
        queue_rows = interleave_by_label(reviewer_rows[reviewer_id], reviewer_id, args.seed + 7000)
        payload_rows = [row["_review_assignment"] for row in queue_rows]
        payload = {
            "reviewer_id": reviewer_id,
            "reviewer_first_name": REVIEWER_ID_TO_NAME.get(reviewer_id, f"reviewer_{reviewer_id}"),
            "input": base.portable_path(args.input),
            "record_count": len(payload_rows),
            "assignment_rows": payload_rows,
            "target_load": reviewer_targets[reviewer_id],
            "label_counts": dict(reviewer_label_counts[reviewer_id]),
            "paired_reviewer_counts": dict(reviewer_pair_counts[reviewer_id]),
            "order_strategy": "two_review_pair_balanced_label_interleaved",
        }
        path = output_dir / f"reviewer_{reviewer_id}_ids.json"
        write_json(path, payload)
        manifest["reviewer_files"][str(reviewer_id)] = base.portable_path(path)

    manifest_path = output_dir / "assignment_manifest.json"
    write_json(manifest_path, manifest)
    print(f"wrote assignments to {output_dir}")
    print(f"manifest: {manifest_path}")
    for reviewer_id in reviewer_ids:
        print(
            f"reviewer {reviewer_id}: {reviewer_loads[reviewer_id]} records "
            f"{dict(reviewer_label_counts[reviewer_id])} paired_with={dict(reviewer_pair_counts[reviewer_id])}"
        )


def summarize_reviews(path: Path) -> None:
    reviews = list(load_reviews(path).values())
    print(f"reviews={len(reviews)} path={path}")
    for field in ["label_action", "reviewed_conflict_type", "review_confidence", "changed_label"]:
        print(f"{field}: {dict(Counter(str(row.get(field, '')) for row in reviews))}")


def run_review(args: argparse.Namespace) -> None:
    reviewer_first_name, reviewer_id = prompt_reviewer_identity(args)
    assignment_path = Path(args.assignment) if args.assignment else default_assignment_path(reviewer_id)
    if not assignment_path.exists():
        print(f"Warning: assignment file not found: {assignment_path}")
        print("Falling back to all input rows in canonical-label order.")
        assignment_path = None
        assignment_payload = None
    else:
        assignment_payload = load_assignment_payload(assignment_path)
        assigned_reviewer_id = assignment_payload.get("reviewer_id")
        if assigned_reviewer_id is not None and int(assigned_reviewer_id) != reviewer_id:
            raise ValueError(
                f"Assignment file {assignment_path} belongs to reviewer {assigned_reviewer_id}, "
                f"but current reviewer is {reviewer_first_name} (ID {reviewer_id})."
            )

    input_path = Path(args.input)
    if assignment_payload and str(args.input) == str(DEFAULT_INPUT):
        assignment_input = assignment_payload.get("input")
        if assignment_input:
            input_path = PROJECT_ROOT / str(assignment_input)

    output = Path(args.output) if args.output else default_review_path_for_reviewer(reviewer_id)
    rows = read_jsonl(input_path)
    rows = apply_assignment(rows, assignment_path)
    if args.limit:
        rows = rows[: args.limit]
    order = [row_id(row) for row in rows]
    if any(not rid for rid in order):
        raise ValueError("Every input row must have a non-empty id.")

    reviews = load_reviews(output)
    validate_existing_reviews(reviews, reviewer_first_name, reviewer_id, output)
    print_assignment_overview(reviewer_first_name, reviewer_id, rows, reviews, assignment_path, output)
    input("Press Enter to start reviewing...")

    index = 0
    revisit_index: Optional[int] = None
    while index < len(rows):
        row = rows[index]
        rid = row_id(row)
        existing_review = reviews.get(rid)
        revisit_this_row = revisit_index == index
        if existing_review and not args.review_all and not revisit_this_row:
            index += 1
            continue

        result = review_one_record(
            row,
            args,
            reviewer_first_name,
            reviewer_id,
            index,
            len(rows),
            len(reviews),
            existing_review=existing_review if (args.review_all or revisit_this_row) else None,
        )
        if result is None:
            revisit_index = None
            index += 1
            continue
        command = result.get("_command")
        if command == "q":
            save_reviews(output, reviews, order)
            print(f"Saved {len(reviews)} reviews to {output}")
            return
        if command == "s":
            revisit_index = None
            index += 1
            continue
        if command == "p":
            index = max(0, index - 1)
            revisit_index = index
            continue

        reviews[rid] = result
        save_reviews(output, reviews, order)
        revisit_index = None
        index += 1

    save_reviews(output, reviews, order)
    print(f"Completed assigned queue. Saved {len(reviews)} reviews to {output}")
    summarize_reviews(output)


def main() -> None:
    base.COLOR_ENABLED = True

    ap = argparse.ArgumentParser(description="Human conflict-type review CLI for the 658-example training set")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--assignment", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--reviewer-first-name", default=None)
    ap.add_argument("--reviewer-id", type=int, default=None)
    ap.add_argument("--reviewers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--snippet-chars", type=int, default=0, help="Max chars per snippet; <=0 shows full snippets")
    ap.add_argument("--compact-snippet-chars", type=int, default=DEFAULT_COMPACT_SNIPPET_CHARS)
    ap.add_argument("--no-clear", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--hide-taxonomy", action="store_true")
    ap.add_argument("--review-all", action="store_true")

    ap.add_argument("--make-assignments", action="store_true")
    ap.add_argument("--output-dir", default=str(DEFAULT_ASSIGNMENT_DIR))
    ap.add_argument("--seed", type=int, default=62058)

    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    base.COLOR_ENABLED = not args.no_color and "NO_COLOR" not in os.environ

    if args.make_assignments:
        make_assignments(args)
        return
    if args.summarize:
        if not args.output:
            raise SystemExit("--summarize requires --output")
        summarize_reviews(Path(args.output))
        return
    run_review(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
