"""
Interactive CLI for human review of Stage-3 JSONL annotations.

This script is designed for human verification of stagewise multi-LLM outputs
such as:

    data/experiments/stagewise_multi_20260330_042140/stage3_final.jsonl

It makes each record substantially easier to inspect than raw JSONL by:
1. formatting the key annotation fields clearly,
2. summarizing per-document Stage-1 notes,
3. allowing on-demand viewing of full retrieved-doc snippets,
4. recording human review decisions in a separate JSONL sidecar.

The input path is always supplied at the command line so the tool remains
generalizable across experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ReviewMap = Dict[str, Dict[str, Any]]


def default_review_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem}_human_review.jsonl"))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


def load_reviews(path: str) -> ReviewMap:
    review_path = Path(path)
    if not review_path.exists():
        return {}

    reviews: ReviewMap = {}
    with review_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid review JSON on line {line_no} of {path}: {exc}") from exc

            record_id = review.get("id")
            if record_id:
                reviews[record_id] = review
    return reviews


def save_reviews(path: str, reviews: ReviewMap, record_order: List[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record_id in record_order:
            review = reviews.get(record_id)
            if review is not None:
                f.write(json.dumps(review, ensure_ascii=False) + "\n")


def terminal_width(default: int = 100) -> int:
    try:
        return max(80, shutil.get_terminal_size((default, 20)).columns)
    except OSError:
        return default


def wrap(text: str, width: int, indent: str = "") -> str:
    value = (text or "").strip()
    if not value:
        return f"{indent}-"
    return textwrap.fill(
        value,
        width=max(40, width),
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def print_rows(rows: List[tuple[str, str]], width: int, label_width: int = 26) -> None:
    for label, value in rows:
        text = (value or "-").strip() or "-"
        wrapped = textwrap.wrap(
            text,
            width=max(30, width - label_width - 2),
            break_long_words=False,
            break_on_hyphens=False,
        ) or ["-"]
        print(f"{label + ':':<{label_width}} {wrapped[0]}")
        for line in wrapped[1:]:
            print(f"{'':<{label_width}} {line}")


def print_block(title: str, text: str, width: int, indent: int = 4) -> None:
    print(title)
    print(wrap(text, width - indent, indent=" " * indent))


def short_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_tally(tally: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in (tally or {}).items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def consensus_metrics(tally: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_tally(tally)
    if not normalized:
        return {
            "badge": "NO_VOTE",
            "winner": None,
            "winner_weight": 0.0,
            "runner_up": None,
            "runner_up_weight": 0.0,
            "margin": 0.0,
        }

    ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    winner, winner_weight = ranked[0]
    runner_up, runner_up_weight = (ranked[1] if len(ranked) > 1 else (None, 0.0))
    margin = round(winner_weight - runner_up_weight, 4)

    if len(ranked) == 1 and abs(winner_weight - 1.0) < 1e-9:
        badge = "UNANIMOUS"
    elif margin < 0.2:
        badge = "CONTESTED"
    else:
        badge = "SPLIT"

    return {
        "badge": badge,
        "winner": winner,
        "winner_weight": round(winner_weight, 4),
        "runner_up": runner_up,
        "runner_up_weight": round(runner_up_weight, 4),
        "margin": margin,
    }


def disagreement_docs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    disagreements: List[Dict[str, Any]] = []
    for note in record.get("per_doc_notes", []):
        metrics = consensus_metrics(note.get("_vote_tally", {}))
        if metrics["badge"] != "UNANIMOUS":
            disagreements.append({
                "doc_id": note.get("doc_id", "?"),
                "badge": metrics["badge"],
                "margin": metrics["margin"],
                "winner": metrics["winner"],
            })
    return disagreements


def record_consensus_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    ans = consensus_metrics(record.get("_ans_vote_tally", {}))
    abstain = consensus_metrics(record.get("_abstain_vote_tally", {}))

    total_docs = len(record.get("per_doc_notes", []))
    disagreements = disagreement_docs(record)
    disagreement_count = len(disagreements)
    unanimous_count = total_docs - disagreement_count
    unanimity_rate = round((unanimous_count / total_docs), 4) if total_docs else 0.0

    if ans["badge"] == "CONTESTED" or abstain["badge"] == "CONTESTED":
        strength = "LOW"
    elif ans["badge"] == "UNANIMOUS" and abstain["badge"] == "UNANIMOUS" and unanimity_rate >= 0.8:
        strength = "STRONG"
    elif unanimity_rate >= 0.6:
        strength = "MODERATE"
    else:
        strength = "LOW"

    return {
        "strength": strength,
        "answerability": ans,
        "abstain": abstain,
        "doc_unanimity_rate": unanimity_rate,
        "unanimous_docs": unanimous_count,
        "total_docs": total_docs,
        "disagreement_docs": disagreements,
    }


def print_rule(width: int, ch: str = "=") -> None:
    print(ch * width)


def print_section(title: str, width: int) -> None:
    print_rule(width, "-")
    print(title)
    print_rule(width, "-")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def explain_consensus_metrics(width: int) -> None:
    print_section("Metric Guide", width)
    print(wrap(
        "UNANIMOUS means every saved vote went to one label. SPLIT means there was disagreement, "
        "but the leading label still had a clear edge. CONTESTED means the leading label only beat "
        "the runner-up by a small amount.",
        width - 2,
    ))
    print()
    print(wrap(
        "vote_margin = winner_weight - runner_up_weight. Larger margin means stronger agreement. "
        "Example: 1.00 means unanimous, 0.60 means a strong split, 0.10 means the vote was close.",
        width - 2,
    ))
    print()
    print(wrap(
        "record_consensus_strength is a quick overall summary built from the answerability vote, "
        "the abstain vote, and how many Stage-1 docs were unanimous.",
        width - 2,
    ))


def render_doc_summary(
    note: Dict[str, Any],
    doc: Optional[Dict[str, Any]],
    final_evidence: List[str],
    width: int,
) -> None:
    doc_id = note.get("doc_id", "?")
    verdict = note.get("verdict", "-")
    quality = note.get("source_quality", "-")
    vote_tally = note.get("_vote_tally", {})
    metrics = consensus_metrics(vote_tally)
    used_in_final = "yes" if doc_id in final_evidence else "no"
    print_rule(width, ".")
    print(f"DOC {doc_id}")
    print_rule(width, ".")
    print_rows(
        [
            ("Verdict", verdict),
            ("Source Quality", quality),
            ("Used In Final Answer", used_in_final),
            ("Consensus Badge", metrics["badge"]),
            ("Vote Margin", str(metrics["margin"])),
            ("Vote Tally", short_json(vote_tally)),
        ],
        width,
    )
    print()
    print_block("Key Fact", note.get("key_fact", ""), width)
    if doc:
        print()
        print_block("Snippet", doc.get("snippet", ""), width)
    quote = note.get("quote", "")
    if quote:
        print()
        print_block("Quote", quote, width)


def build_doc_index(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    docs = {doc.get("doc_id", ""): doc for doc in record.get("retrieved_docs", [])}
    for note in record.get("per_doc_notes", []):
        doc_id = note.get("doc_id", "")
        if doc_id and doc_id not in docs:
            docs[doc_id] = {"doc_id": doc_id}
    return docs


def show_doc_detail(record: Dict[str, Any], doc_id: str, width: int) -> None:
    notes_by_id = {note.get("doc_id", ""): note for note in record.get("per_doc_notes", [])}
    docs_by_id = build_doc_index(record)

    note = notes_by_id.get(doc_id)
    doc = docs_by_id.get(doc_id)
    if doc is None and note is None:
        print(f"No doc found for id '{doc_id}'.")
        return

    print_rule(width)
    print(f"DOC DETAIL: {doc_id}")
    print_rule(width)

    if note:
        metrics = consensus_metrics(note.get("_vote_tally", {}))
        print_rows(
            [
                ("Verdict", note.get("verdict", "-")),
                ("Source Quality", note.get("source_quality", "-")),
                ("Consensus Badge", metrics["badge"]),
                (
                    "Vote Margin",
                    f"{metrics['margin']} (winner_weight={metrics['winner_weight']}, runner_up_weight={metrics['runner_up_weight']})",
                ),
                ("Vote Tally", short_json(note.get("_vote_tally", {}))),
            ],
            width,
        )
        all_verdicts = note.get("_all_verdicts")
        if all_verdicts:
            print_rows([("All Model Verdicts", short_json(all_verdicts))], width)
        print()
        print_block("Verdict Reason", note.get("verdict_reason", ""), width)
        print()
        print_block("Key Fact", note.get("key_fact", ""), width)
        print()
        print_block("Quote", note.get("quote", ""), width)
        print()

    if doc:
        print_rows(
            [
                ("Source URL", doc.get("source_url", "-")),
                ("Timestamp", doc.get("timestamp", "-") or "-"),
            ],
            width,
        )
        print()
        print_block("Full Snippet", doc.get("snippet", ""), width)

    print_rule(width)


def show_think(record: Dict[str, Any], width: int) -> None:
    think = (record.get("think") or "").strip()
    print_rule(width)
    print("THINK TRACE")
    print_rule(width)
    print(think or "-")
    print_rule(width)


def render_record(
    record: Dict[str, Any],
    index: int,
    total: int,
    review: Optional[Dict[str, Any]],
    width: int,
) -> None:
    summary = record_consensus_summary(record)
    ans = summary["answerability"]
    abstain = summary["abstain"]
    disagreement_list = summary["disagreement_docs"]
    expected = record.get("expected_response", {})
    final_evidence = expected.get("evidence", [])
    docs_by_id = build_doc_index(record)

    clear_screen()
    print_rule(width)
    print(f"STAGE-3 HUMAN REVIEW  |  record {index + 1}/{total}  |  id={record.get('id', '-')}")
    print_rule(width)
    print_rows(
        [
            ("Query", record.get("query", "")),
            ("Conflict Type", record.get("conflict_type", "-")),
            ("Answerable Under Evidence", str(record.get("answerable_under_evidence", "-"))),
            ("Gold Answer", record.get("gold_answer", "") or "-"),
            ("Answerability Vote Tally", short_json(record.get("_ans_vote_tally", {}))),
            ("Answerability Winner Model", record.get("_ans_winner_model", "-")),
            ("Abstain Vote Tally", short_json(record.get("_abstain_vote_tally", {}))),
            ("Abstain Winner Model", record.get("_abstain_winner_model", "-")),
        ],
        width,
    )

    if review:
        print()
        print_section("Existing Saved Review", width)
        notes_value = review.get("notes", review.get("review_notes", ""))
        print_rows(
            [
                ("Review Status", review.get("review_status", "-")),
                ("Final Answer OK", str(review.get("final_answer_ok", "-"))),
                ("Evidence OK", str(review.get("evidence_ok", "-"))),
                ("Abstain OK", str(review.get("abstain_ok", "-"))),
                ("Existing Notes", notes_value),
            ],
            width,
        )

    print_section("Consensus Summary", width)
    print_rows(
        [
            ("Record Consensus Strength", summary["strength"]),
            (
                "Answerability Consensus",
                f"{ans['badge']} | winner={ans['winner']} | vote_margin={ans['margin']}",
            ),
            (
                "Abstain Consensus",
                f"{abstain['badge']} | winner={abstain['winner']} | vote_margin={abstain['margin']}",
            ),
            (
                "Doc Unanimity",
                f"{summary['unanimous_docs']}/{summary['total_docs']} ({summary['doc_unanimity_rate'] * 100:.1f}%)",
            ),
        ],
        width,
    )
    if disagreement_list:
        quick = ", ".join(
            f"{item['doc_id']}[{item['badge']}|margin={item['margin']}]"
            for item in disagreement_list
        )
        print_rows([("Docs With Disagreement", quick)], width)
    else:
        print_rows([("Docs With Disagreement", "none")], width)

    explain_consensus_metrics(width)

    print_section("Conflict Reason", width)
    print(wrap(record.get("conflict_reason", ""), width - 4, indent="    "))

    print_section("Final Annotation", width)
    print_rows(
        [
            ("Abstain", str(expected.get("abstain", False))),
            ("Evidence", ", ".join(expected.get("evidence", [])) or "-"),
            ("Abstain Reason", expected.get("abstain_reason", "") or "-"),
        ],
        width,
    )
    print()
    print_block("Answer", expected.get("answer", ""), width)

    print_section("Per-Doc Notes Summary", width)
    for note in record.get("per_doc_notes", []):
        doc = docs_by_id.get(note.get("doc_id", ""))
        render_doc_summary(note, doc, final_evidence, width)
        print()

    print_section("Commands", width)
    print("a = approve/save review")
    print("f = flag/save review")
    print("r = reject/save review")
    print("k = skip for now")
    print("b = previous record")
    print("d <doc_id> = inspect one retrieved document in detail")
    print("t = show think trace")
    print("j <index|id> = jump to record")
    print("q = quit")
    print_rule(width)


def prompt(label: str) -> str:
    try:
        return input(label).strip()
    except EOFError:
        return "q"


def prompt_tri_state(label: str, default: Optional[bool]) -> Optional[bool]:
    if default is True:
        suffix = "[Y/n/u]"
    elif default is False:
        suffix = "[y/N/u]"
    else:
        suffix = "[y/n/U]"

    while True:
        raw = prompt(f"{label} {suffix}: ").lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        if raw in {"u", "unsure", "unknown"}:
            return None
        print("Please answer y, n, or u.")


def rubric_defaults(status: str) -> Dict[str, Optional[bool]]:
    if status == "approved":
        default = True
    elif status == "rejected":
        default = False
    else:
        default = None
    return {
        "final_answer_ok": default,
        "evidence_ok": default,
        "abstain_ok": default,
    }


def build_review_entry(
    record: Dict[str, Any],
    status: str,
    reviewer: str,
    final_answer_ok: Optional[bool],
    evidence_ok: Optional[bool],
    abstain_ok: Optional[bool],
    notes: str,
    suggested_answer: str,
) -> Dict[str, Any]:
    expected = record.get("expected_response", {})
    return {
        "id": record.get("id"),
        "query": record.get("query", ""),
        "review_status": status,
        "final_answer_ok": final_answer_ok,
        "evidence_ok": evidence_ok,
        "abstain_ok": abstain_ok,
        "notes": notes,
        "suggested_answer": suggested_answer or None,
        "final_answer_snapshot": expected.get("answer", ""),
        "final_evidence_snapshot": expected.get("evidence", []),
        "abstain_snapshot": expected.get("abstain", False),
        "reviewer": reviewer or None,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def first_unreviewed_index(records: List[Dict[str, Any]], reviews: ReviewMap) -> int:
    for idx, record in enumerate(records):
        if record.get("id") not in reviews:
            return idx
    return 0


def next_unreviewed_index(
    records: List[Dict[str, Any]],
    reviews: ReviewMap,
    start_idx: int,
) -> int:
    for idx in range(max(0, start_idx), len(records)):
        if records[idx].get("id") not in reviews:
            return idx
    return len(records)


def jump_target(command: str, records: List[Dict[str, Any]]) -> Optional[int]:
    _, _, raw_target = command.partition(" ")
    raw_target = raw_target.strip()
    if not raw_target:
        return None

    if raw_target.startswith("#"):
        for idx, record in enumerate(records):
            if record.get("id") == raw_target:
                return idx
        return None

    if raw_target.isdigit():
        target = int(raw_target)
        if 1 <= target <= len(records):
            return target - 1
    return None


def review_loop(
    records: List[Dict[str, Any]],
    output_path: str,
    reviewer: str,
) -> None:
    reviews = load_reviews(output_path)
    record_order = [record.get("id", "") for record in records]
    current = first_unreviewed_index(records, reviews)
    width = terminal_width()

    while 0 <= current < len(records):
        record = records[current]
        record_id = record.get("id", f"record_{current + 1}")
        review = reviews.get(record_id)
        render_record(record, current, len(records), review, width)

        command = prompt("review> ").strip()
        if not command:
            continue

        if command == "q":
            print(f"Saved reviews to {output_path}")
            return

        if command == "k":
            current = next_unreviewed_index(records, reviews, current + 1)
            continue

        if command == "b":
            current = max(current - 1, 0)
            continue

        if command == "t":
            show_think(record, width)
            prompt("Press Enter to return to the record...")
            continue

        if command.startswith("d "):
            _, _, doc_id = command.partition(" ")
            show_doc_detail(record, doc_id.strip(), width)
            prompt("Press Enter to return to the record...")
            continue

        if command.startswith("j "):
            target = jump_target(command, records)
            if target is None:
                print("Could not resolve jump target.")
                prompt("Press Enter to continue...")
            else:
                current = target
            continue

        if command in {"a", "f", "r"}:
            status_map = {"a": "approved", "f": "flagged", "r": "rejected"}
            status = status_map[command]
            defaults = rubric_defaults(status)
            final_answer_ok = prompt_tri_state("final_answer_ok", defaults["final_answer_ok"])
            evidence_ok = prompt_tri_state("evidence_ok", defaults["evidence_ok"])
            abstain_ok = prompt_tri_state("abstain_ok", defaults["abstain_ok"])
            notes = prompt("notes (optional): ")
            suggested_answer = ""
            if status in {"flagged", "rejected"}:
                suggested_answer = prompt("suggested_answer (optional): ")

            reviews[record_id] = build_review_entry(
                record=record,
                status=status,
                reviewer=reviewer,
                final_answer_ok=final_answer_ok,
                evidence_ok=evidence_ok,
                abstain_ok=abstain_ok,
                notes=notes,
                suggested_answer=suggested_answer,
            )
            save_reviews(output_path, reviews, record_order)
            current = next_unreviewed_index(records, reviews, current + 1)
            continue

        print("Unknown command.")
        prompt("Press Enter to continue...")

    print(f"Saved reviews to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive CLI for human review of Stage-3 JSONL outputs. "
            "Pass any stage3_final.jsonl-style path with --input."
        )
    )
    parser.add_argument("--input", required=True, help="Path to Stage-3 JSONL file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save human review JSONL (default: <input>_human_review.jsonl)",
    )
    parser.add_argument(
        "--reviewer",
        default="",
        help="Optional reviewer name to store with each saved review decision",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if not records:
        raise SystemExit(f"No records found in {args.input}")

    output_path = args.output or default_review_path(args.input)
    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Review output will be saved to {output_path}")
    prompt("Press Enter to start reviewing...")
    review_loop(records, output_path, args.reviewer)


if __name__ == "__main__":
    main()
