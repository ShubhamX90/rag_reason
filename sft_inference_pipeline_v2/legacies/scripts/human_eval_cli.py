#!/usr/bin/env python3
"""
Interactive CLI for human evaluation of model outputs.

The script discovers available generation files from ./outputs, lets an
evaluator choose dataset/model/variant/profile/file, and then presents each
example with:

- query
- retrieved docs
- gold per-doc notes
- gold conflict type
- gold answerable_under_evidence
- model think trace (if present)
- model final answer

Judgments are appended to outputs/human_eval/judgments.jsonl.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
JUDGMENTS_DIR = OUTPUTS_DIR / "human_eval"
JUDGMENTS_PATH = JUDGMENTS_DIR / "judgments.jsonl"

PROMPT_MODE_PARTS = [
    ("oracle_conflict", ["oracle", "conflict"]),
    ("oracle_notes", ["oracle", "notes"]),
    ("oracle_both", ["oracle", "both"]),
    ("e2e", ["e2e"]),
]
KNOWN_PROFILES = {
    "strict": "strict",
    "trace_text": "runtime",
    "minimal": "minimal",
}
SENTINEL = "[[END-OF-ANSWER]]"
THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


@dataclass
class OutputRecord:
    path: Path
    variant: str
    model_name: str
    train_strategy: str
    run_name: str
    prompt_mode: str
    message_tag: str
    prompt_profile: str
    dataset_label: str
    file_kind: str

    @property
    def display_name(self) -> str:
        return (
            f"{self.variant} | {self.model_name} | {self.prompt_profile} | "
            f"{self.dataset_label} | {self.path.name}"
        )


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def load_canon_rows(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        row_id = row.get("id")
        if row_id is not None:
            rows[row_id] = row
    return rows


def terminal_width(default: int = 120) -> int:
    try:
        return max(80, shutil.get_terminal_size((default, 40)).columns)
    except OSError:
        return default


def hr(char: str = "=") -> str:
    return char * terminal_width()


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def wrap(text: str, *, indent: int = 0) -> str:
    width = terminal_width()
    prefix = " " * indent
    return textwrap.fill(
        text or "",
        width=width,
        initial_indent=prefix,
        subsequent_indent=prefix,
        replace_whitespace=False,
        drop_whitespace=False,
    )


def wrap_preserve_lines(text: str, *, indent: int = 0) -> str:
    lines = (text or "").splitlines() or [""]
    wrapped = []
    for line in lines:
        if line.strip():
            wrapped.append(wrap(line, indent=indent))
        else:
            wrapped.append(" " * indent)
    return "\n".join(wrapped)


def prompt_input(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def prompt_choice(title: str, options: List[Tuple[str, Any]]) -> Any:
    while True:
        clear_screen()
        print(title)
        print(hr("-"))
        for idx, (label, _) in enumerate(options, 1):
            print(f"{idx}. {label}")
        print("q. Quit")
        raw = input("\nChoose an option: ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][1]
        input("Invalid choice. Press Enter to continue...")


def prompt_yes_no(label: str, default: Optional[bool] = None) -> Optional[bool]:
    hint = {True: "Y/n", False: "y/N", None: "y/n"}[default]
    while True:
        raw = input(f"{label} [{hint}]: ").strip().lower()
        if not raw and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        if raw in {"na", "n/a"}:
            return None
        print("Please enter y, n, or na.")


def prompt_score(label: str, default: Optional[int] = None, allow_na: bool = True) -> Optional[int]:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{label} (1-5{' or na' if allow_na else ''}){suffix}: ").strip().lower()
        if not raw and default is not None:
            return default
        if allow_na and raw in {"na", "n/a"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("Please enter 1-5 or na.")


def parse_output_record(path: Path, dataset_labels: List[str]) -> Optional[OutputRecord]:
    name = path.name
    if not (name.endswith(".sanitized.jsonl") or name.endswith(".raw.jsonl")):
        return None
    file_kind = "sanitized" if name.endswith(".sanitized.jsonl") else "raw"
    stem = name[: -len(".sanitized.jsonl")] if file_kind == "sanitized" else name[: -len(".raw.jsonl")]
    parts = stem.split("_")
    if not parts or parts[0] not in {"sft", "baseline"}:
        return None

    strategy_idx = None
    for i, token in enumerate(parts):
        if token in {"stagewise", "monolithic"}:
            strategy_idx = i
            break
    if strategy_idx is None or strategy_idx < 2:
        return None

    variant = parts[0]
    model_name = "_".join(parts[1:strategy_idx])
    train_strategy = parts[strategy_idx]

    dataset_parts = None
    dataset_label = None
    for candidate in sorted(dataset_labels, key=len, reverse=True):
        cand_parts = candidate.split("_")
        if parts[-len(cand_parts) :] == cand_parts:
            dataset_parts = cand_parts
            dataset_label = candidate
            break
    if dataset_parts is None or dataset_label is None:
        return None

    middle = parts[strategy_idx + 1 : -len(dataset_parts)]
    mode_start = None
    mode_name = None
    mode_parts = None
    for idx in range(len(middle)):
        for candidate_name, candidate_parts in PROMPT_MODE_PARTS:
            if middle[idx : idx + len(candidate_parts)] == candidate_parts:
                mode_start = idx
                mode_name = candidate_name
                mode_parts = candidate_parts
    if mode_start is None or mode_name is None or mode_parts is None:
        return None

    run_name = "_".join(middle[:mode_start])
    tag_parts = middle[mode_start + len(mode_parts) :]
    message_tag = "_".join(tag_parts)
    prompt_profile = KNOWN_PROFILES.get(message_tag, "default" if not message_tag else message_tag)

    return OutputRecord(
        path=path,
        variant=variant,
        model_name=model_name,
        train_strategy=train_strategy,
        run_name=run_name,
        prompt_mode=mode_name,
        message_tag=message_tag,
        prompt_profile=prompt_profile,
        dataset_label=dataset_label,
        file_kind=file_kind,
    )


def discover_output_records() -> List[OutputRecord]:
    dataset_labels = sorted(p.stem for p in SPLITS_DIR.glob("*.jsonl"))
    records = []
    for path in sorted(OUTPUTS_DIR.glob("*.sanitized.jsonl")):
        record = parse_output_record(path, dataset_labels)
        if record:
            records.append(record)
    return records


def load_judgments() -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not JUDGMENTS_PATH.exists():
        return latest
    for row in read_jsonl(JUDGMENTS_PATH):
        key = (str(row.get("evaluator_id")), str(row.get("output_file")), str(row.get("example_id")))
        latest[key] = row
    return latest


def save_judgment(row: Dict[str, Any]) -> None:
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with JUDGMENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_model_sections(text: str) -> Tuple[str, str]:
    text = (text or "").replace(SENTINEL, "").strip()
    think_match = THINK_BLOCK.search(text)
    if think_match:
        think = think_match.group(1).strip()
        final = (text[: think_match.start()] + text[think_match.end() :]).strip()
    else:
        think = ""
        final = text
    final = re.sub(r"\s+", " ", final).strip()
    return think, final


def format_doc(doc: Dict[str, Any], note_by_id: Dict[str, Dict[str, Any]], idx: int) -> str:
    doc_id = doc.get("doc_id", f"d{idx}")
    note = note_by_id.get(doc_id, {})
    pieces = [
        f"[{idx}] {doc_id}",
    ]
    title = doc.get("title") or ""
    if title:
        pieces.append(f"Title: {title}")
    snippet = doc.get("snippet") or ""
    if snippet:
        pieces.append(f"Snippet: {snippet}")
    source_url = doc.get("source_url") or doc.get("url") or ""
    if source_url:
        pieces.append(f"Source: {source_url}")
    timestamp = doc.get("timestamp") or doc.get("date") or ""
    if timestamp:
        pieces.append(f"Time: {timestamp}")

    if note:
        pieces.append(
            "Gold Note: "
            f"verdict={note.get('verdict', '')}; "
            f"source_quality={note.get('source_quality', '')}; "
            f"key_fact={note.get('key_fact', '')}"
        )
        if note.get("quote"):
            pieces.append(f"Quote: {note.get('quote')}")
        if note.get("verdict_reason"):
            pieces.append(f"Reason: {note.get('verdict_reason')}")

    return "\n".join(wrap(piece, indent=2) for piece in pieces)


def render_example(
    evaluator_name: str,
    evaluator_id: str,
    record: OutputRecord,
    example: Dict[str, Any],
    generation: Dict[str, Any],
    example_index: int,
    total_examples: int,
    existing_judgment: Optional[Dict[str, Any]],
) -> None:
    clear_screen()
    width = terminal_width()
    print(hr("="))
    print(
        wrap(
            f"Evaluator: {evaluator_name} ({evaluator_id}) | "
            f"Example {example_index + 1}/{total_examples} | "
            f"Dataset: {record.dataset_label} | Model: {record.model_name} | "
            f"Variant: {record.variant} | Profile: {record.prompt_profile}",
        )
    )
    print(hr("="))
    print(wrap(f"ID: {example.get('id')}"))
    print()

    print("Query")
    print(hr("-"))
    print(wrap(example.get("query", "")))
    print()

    print("Gold Summary")
    print(hr("-"))
    print(wrap(f"Conflict Type: {example.get('conflict_type', '')}"))
    print(wrap(f"Conflict Reason: {example.get('conflict_reason', '')}"))
    print(wrap(f"Answerable Under Evidence: {example.get('answerable_under_evidence', '')}"))
    if "gold_answer" in example:
        print(wrap(f"Gold Answer: {example.get('gold_answer', '')}"))
    print()

    print("Retrieved Docs + Gold Per-Doc Notes")
    print(hr("-"))
    notes = {note.get("doc_id"): note for note in (example.get("per_doc_notes") or [])}
    for idx, doc in enumerate(example.get("retrieved_docs") or [], 1):
        print(format_doc(doc, notes, idx))
        print()

    raw_text = generation.get("raw", "")

    print("Model Output")
    print(hr("-"))
    print("Exact Raw Output")
    print(wrap_preserve_lines(raw_text or "<empty>", indent=2))
    print()

    if existing_judgment:
        print("Existing Judgment")
        print(hr("-"))
        summary = (
            f"overall={existing_judgment.get('overall_score')} | "
            f"answer={existing_judgment.get('final_answer_quality')} | "
            f"citations={existing_judgment.get('citation_grounding_quality')} | "
            f"abstain={existing_judgment.get('abstention_behavior')}"
        )
        print(wrap(summary))
        if existing_judgment.get("notes"):
            print(wrap(f"Notes: {existing_judgment.get('notes')}"))
        print()

    print(hr("="))
    print("Actions: [j] judge/save  [n] next  [p] previous  [g] goto index/id  [u] unevaluated next  [q] quit")
    print(hr("="))


def build_judgment(
    evaluator_name: str,
    evaluator_id: str,
    record: OutputRecord,
    example: Dict[str, Any],
    generation: Dict[str, Any],
    existing: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    defaults = existing or {}
    think, final = extract_model_sections(generation.get("raw", ""))
    print("\nEnter judgment values. Use 'na' where appropriate.\n")
    trace_default = defaults.get("trace_quality")
    if not think:
        trace_default = None
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_name": evaluator_name,
        "evaluator_id": evaluator_id,
        "dataset_label": record.dataset_label,
        "output_file": record.path.name,
        "output_path": str(record.path.relative_to(PROJECT_ROOT)),
        "model_name": record.model_name,
        "model_variant": record.variant,
        "train_strategy": record.train_strategy,
        "run_name": record.run_name,
        "prompt_mode": record.prompt_mode,
        "prompt_profile": record.prompt_profile,
        "message_tag": record.message_tag,
        "example_id": example.get("id"),
        "trace_quality": prompt_score("Trace quality", trace_default, allow_na=True),
        "final_answer_quality": prompt_score("Final answer quality", defaults.get("final_answer_quality"), allow_na=False),
        "citation_grounding_quality": prompt_score("Citation / grounding quality", defaults.get("citation_grounding_quality"), allow_na=False),
        "conflict_handling_quality": prompt_score("Conflict handling quality", defaults.get("conflict_handling_quality"), allow_na=False),
        "doc_note_alignment_quality": prompt_score("Alignment with gold per-doc notes", defaults.get("doc_note_alignment_quality"), allow_na=False),
        "abstention_behavior": prompt_input(
            "Abstention behavior (correct / incorrect / not_applicable)",
            str(defaults.get("abstention_behavior", "not_applicable")),
        ),
        "hallucination_present": prompt_yes_no("Hallucination present?", defaults.get("hallucination_present")),
        "citation_issue_present": prompt_yes_no("Citation issue present?", defaults.get("citation_issue_present")),
        "format_issue_present": prompt_yes_no("Format issue present?", defaults.get("format_issue_present")),
        "overall_score": prompt_score("Overall score", defaults.get("overall_score"), allow_na=False),
        "notes": prompt_input("Freeform notes", str(defaults.get("notes", ""))),
    }
    return row


def choose_output_record(records: List[OutputRecord]) -> OutputRecord:
    dataset = prompt_choice(
        "Choose dataset",
        [(label, label) for label in sorted({r.dataset_label for r in records})],
    )
    filtered = [r for r in records if r.dataset_label == dataset]

    model_name = prompt_choice(
        "Choose model",
        [(label, label) for label in sorted({r.model_name for r in filtered})],
    )
    filtered = [r for r in filtered if r.model_name == model_name]

    variant = prompt_choice(
        "Choose variant",
        [(label, label) for label in sorted({r.variant for r in filtered})],
    )
    filtered = [r for r in filtered if r.variant == variant]

    profile = prompt_choice(
        "Choose prompt profile",
        [(label, label) for label in sorted({r.prompt_profile for r in filtered})],
    )
    filtered = [r for r in filtered if r.prompt_profile == profile]

    file_options = []
    for r in sorted(filtered, key=lambda x: x.path.name):
        label = f"{r.path.name} | run={r.run_name}"
        file_options.append((label, r))
    return prompt_choice("Choose output file", file_options)


def goto_example(prompt: str, rows: List[Dict[str, Any]]) -> Optional[int]:
    raw = input(prompt).strip()
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(rows):
            return idx
    for idx, row in enumerate(rows):
        if str(row.get("id")) == raw:
            return idx
    return None


def next_unevaluated_index(
    start_idx: int,
    rows: List[Dict[str, Any]],
    record: OutputRecord,
    evaluator_id: str,
    judgments: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Optional[int]:
    total = len(rows)
    for offset in range(1, total + 1):
        idx = (start_idx + offset) % total
        key = (evaluator_id, record.path.name, str(rows[idx].get("id")))
        if key not in judgments:
            return idx
    return None


def run_session(record: OutputRecord, evaluator_name: str, evaluator_id: str) -> None:
    canon_path = SPLITS_DIR / f"{record.dataset_label}.jsonl"
    if not canon_path.exists():
        raise FileNotFoundError(f"Canon file not found: {canon_path}")

    canon_rows = load_canon_rows(canon_path)
    gens_by_id = load_jsonl_by_id(record.path)
    rows = [row for row in canon_rows if row.get("id") in gens_by_id]
    judgments = load_judgments()
    idx = 0

    while True:
        example = rows[idx]
        gen = gens_by_id[example.get("id")]
        key = (evaluator_id, record.path.name, str(example.get("id")))
        existing = judgments.get(key)
        render_example(evaluator_name, evaluator_id, record, example, gen, idx, len(rows), existing)
        action = input("\nAction: ").strip().lower()
        if action in {"q", "quit"}:
            return
        if action in {"n", ""}:
            idx = min(len(rows) - 1, idx + 1)
            continue
        if action == "p":
            idx = max(0, idx - 1)
            continue
        if action == "g":
            target = goto_example("Enter 1-based index or example id: ", rows)
            if target is not None:
                idx = target
            continue
        if action == "u":
            target = next_unevaluated_index(idx, rows, record, evaluator_id, judgments)
            if target is not None:
                idx = target
            else:
                input("No unevaluated examples remain. Press Enter to continue...")
            continue
        if action == "j":
            judgment = build_judgment(evaluator_name, evaluator_id, record, example, gen, existing)
            save_judgment(judgment)
            judgments[key] = judgment
            input("\nSaved. Press Enter to continue...")
            continue


def main() -> int:
    records = discover_output_records()
    if not records:
        print("No sanitized output files discovered under outputs/.")
        return 1

    try:
        clear_screen()
        print("Human Evaluation CLI")
        print(hr("="))
        evaluator_name = prompt_input("Evaluator name")
        evaluator_id = prompt_input("Evaluator id")
        while True:
            record = choose_output_record(records)
            run_session(record, evaluator_name, evaluator_id)
    except KeyboardInterrupt:
        print("\nExiting.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
