#!/usr/bin/env python3
"""
Interactive human preselection for CONFLICTS-style benchmark candidates.

This tool is intentionally positioned before the multi-LLM annotation pipeline:
humans inspect retrieved documents, decide whether a candidate is worth sending
to the expensive stagewise annotation pipeline, and assign a preliminary
conflict type.

Two common modes:

1. Create small reviewer assignment files:

   python3 scripts/benchmark_human_preselection_cli.py \
     --make-assignments \
     --input data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl \
     --output-dir data/benchmark_build/human_preselection/assignments \
     --reviewers 7

2. Review a specific assignment:

   python3 scripts/benchmark_human_preselection_cli.py \
     --input data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl \
     --assignment data/benchmark_build/human_preselection/assignments/reviewer_1_ids.json \
     --output data/benchmark_build/human_preselection/reviews/reviewer_1_reviews.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import textwrap
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl"
)
DEFAULT_ASSIGNMENT_DIR = PROJECT_ROOT / "data/benchmark_build/human_preselection/assignments"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data/benchmark_build/human_preselection/reviews"
DEFAULT_COMPACT_SNIPPET_CHARS = 1800
DEFAULT_REQUIRED_DOC_COUNT = 5
COLOR_ENABLED = True

REVIEWER_ROSTER = {
    "shubham": 1,
    "harsh": 2,
    "gorang": 3,
    "shiv": 4,
    "atharv": 5,
    "manan": 6,
    "parth": 7,
}
REVIEWER_NAMES_BY_ID = {reviewer_id: name for name, reviewer_id in REVIEWER_ROSTER.items()}

CONFLICT_OPTIONS = [
    ("1", "No conflict"),
    ("2", "Complementary information"),
    ("3", "Conflicting opinions or research outcomes"),
    ("4", "Conflict due to outdated information"),
    ("5", "Conflict due to misinformation"),
    ("6", "Other conflict type"),
    ("7", "Unsure"),
]

VALID_PIPELINE_CONFLICT_TYPES = [label for _, label in CONFLICT_OPTIONS[:5]]

CONFLICT_CATEGORY_IDS = {
    "No conflict": 1,
    "Complementary information": 2,
    "Conflicting opinions or research outcomes": 3,
    "Conflict due to outdated information": 4,
    "Conflict due to misinformation": 5,
}

DECISION_OPTIONS = [
    ("1", "accept"),
    ("2", "borderline_accept"),
    ("3", "borderline_reject"),
    ("4", "reject"),
]

CONFIDENCE_OPTIONS = [
    ("1", "high"),
    ("2", "medium"),
    ("3", "low"),
]

RETRIEVAL_QUALITY_OPTIONS = [
    ("1", "good"),
    ("2", "partial"),
    ("3", "bad"),
]

EVIDENCE_SUFFICIENCY_OPTIONS = [
    ("1", "sufficient"),
    ("2", "borderline"),
    ("3", "insufficient"),
]

CONFLICT_CLARITY_OPTIONS = [
    ("1", "clear"),
    ("2", "somewhat_clear"),
    ("3", "unclear"),
]

QUERY_SPECIFICITY_OPTIONS = [
    ("1", "specific"),
    ("2", "somewhat_underspecified"),
    ("3", "too_underspecified"),
]

SOURCE_RELIABILITY_OPTIONS = [
    ("1", "strong"),
    ("2", "mixed"),
    ("3", "weak"),
]

RELEVANT_DOC_COUNT_OPTIONS = [
    ("1", "0-1"),
    ("2", "2-3"),
    ("3", "4-6"),
    ("4", "7-10"),
]

CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
QUALITY_RANK = {"bad": 1, "partial": 2, "good": 3}
SUFFICIENCY_RANK = {"insufficient": 1, "borderline": 2, "sufficient": 3}
DECISION_RANK = {"reject": 0, "borderline_reject": 1, "borderline_accept": 2, "accept": 3}
CLARITY_RANK = {"unclear": 1, "somewhat_clear": 2, "clear": 3}
SPECIFICITY_RANK = {"too_underspecified": 1, "somewhat_underspecified": 2, "specific": 3}
RELIABILITY_RANK = {"weak": 1, "mixed": 2, "strong": 3}

SOURCE_PRIORITY = [
    "conflictingqa",
    "freshqa",
    "situatedqa_temp",
    "qacc",
    "situatedqa_geo",
]

SOURCE_DESCRIPTIONS: Dict[str, List[str]] = {
    "conflictingqa": [
        "Conflict-oriented QA source from the RAG convincingness / ConflictingQA material.",
        "Best for opinion, research-outcome, policy, legal, and conditional conflicts.",
        "Usually worth reviewing first because many questions are conflict-bearing by design.",
        "Still verify that the retrieved top-10 actually preserves the conflict.",
    ],
    "freshqa": [
        "FreshQA contains recent/current factual questions where answers may change over time.",
        "Good for clean current facts, temporal drift, and false-premise or misinformation-like cases.",
        "Often has a useful seed answer, but retrieval may fail on very recent or obscure questions.",
        "Check publication dates and whether snippets are current enough for the query.",
    ],
    "situatedqa_temp": [
        "SituatedQA temporal split contains questions whose answer depends on time or date context.",
        "Best for outdated-information and currentness conflicts.",
        "Relative-time wording can be useful, but only when retrieved docs preserve the target event.",
        "Reject if the snippets answer different timeframes without a coherent temporal story.",
    ],
    "qacc": [
        "QACC-style QA records are mostly compact factual questions.",
        "Good for clean agreement, entity disambiguation, and occasional scope conflicts.",
        "Often less naturally conflict-bearing than ConflictingQA or SituatedQA temporal records.",
        "Keep if retrieved docs are clean and the candidate adds useful benchmark diversity.",
    ],
    "situatedqa_geo": [
        "SituatedQA geographic split contains location-sensitive questions.",
        "Potentially useful for jurisdiction/geographic scope conflicts.",
        "High-risk source: many questions omit country/state context in the standalone query.",
        "Reject aggressively if retrieval guesses the wrong location or mixes unrelated geographies.",
    ],
}

CONFLICT_TYPE_DEFINITIONS = [
    (
        "1",
        "No conflict",
        "Documents answer the same query with equivalent or nearly equivalent facts/entities/concepts. Minor wording or granularity differences do not count as conflict; related docs that do not answer the query should not create conflict.",
    ),
    (
        "2",
        "Complementary information",
        "Documents give different but mutually compatible answers or facets. A reasonable answer can reconcile them, often because multiple answers are valid or the query is underspecified by scope, entity, location, or perspective.",
    ),
    (
        "3",
        "Conflicting opinions or research outcomes",
        "Documents give non-compatible answers within the same scope, such as subjective disagreement, contradictory research findings, historical disagreement, or sources arguing for different sides of the same claim.",
    ),
    (
        "4",
        "Conflict due to outdated information",
        "Documents disagree because some reflect an older state of affairs and others contain newer updates. Prefer the up-to-date evidence while optionally noting that older sources are outdated.",
    ),
    (
        "5",
        "Conflict due to misinformation",
        "Documents disagree because at least one retrieved source appears false, misleading, or inaccurate relative to stronger/reliable evidence in the retrieved set. The right behavior is to disregard the inaccurate source.",
    ),
]

REVIEW_CHECKLIST = [
    "Are enough retrieved snippets directly answering this query?",
    "Do snippets agree, complement each other, or conflict under one of the taxonomy labels?",
    "Is the query too underspecified for a reliable benchmark item?",
    "Would this be worth sending to the expensive multi-LLM annotation pipeline?",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def portable_path(value: Any) -> str:
    path = Path(str(value))
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return str(value)


def row_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("record_id") or "").strip()


def source_family(row: Dict[str, Any]) -> str:
    return str((row.get("_candidate_source") or {}).get("source_family") or "unknown")


def stable_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    src = source_family(row)
    priority = SOURCE_PRIORITY.index(src) if src in SOURCE_PRIORITY else len(SOURCE_PRIORITY)
    return priority, row_id(row)


def stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def parse_reviewer_loads(raw: Optional[str], reviewers: int, total: int) -> List[int]:
    if raw:
        loads = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if len(loads) != reviewers:
            raise ValueError(f"--reviewer-loads must contain exactly {reviewers} comma-separated integers")
        if any(load < 0 for load in loads):
            raise ValueError("--reviewer-loads cannot contain negative values")
        if sum(loads) != total:
            raise ValueError(f"--reviewer-loads must sum to selected total {total}; got {sum(loads)}")
        return loads

    if reviewers == 7 and total == 1878:
        return [150, 150, 300, 300, 326, 326, 326]
    if reviewers == 6 and total == 1878:
        return [150, 150, 315, 315, 474, 474]

    base = total // reviewers
    loads = [base for _ in range(reviewers)]
    for idx in range(total % reviewers):
        loads[idx] += 1
    return loads


def apportion_counts(total: int, weights: Sequence[int], offset: int = 0) -> List[int]:
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [0 for _ in weights]
    raw = [(total * weight / weight_sum) for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    ranked = sorted(
        range(len(weights)),
        key=lambda idx: (raw[idx] - counts[idx], -((idx - offset) % len(weights))),
        reverse=True,
    )
    for idx in ranked[:remainder]:
        counts[idx] += 1
    return counts


def interleave_by_source(rows: Sequence[Dict[str, Any]], reviewer_id: int, seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, deque[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(source_family(row), deque()).append(row)
    for src, bucket in list(grouped.items()):
        ordered = sorted(bucket, key=lambda row: stable_hash(row_id(row), seed + reviewer_id * 97))
        grouped[src] = deque(ordered)

    order = [src for src in SOURCE_PRIORITY if src in grouped] + sorted(set(grouped) - set(SOURCE_PRIORITY))
    if order:
        shift = (reviewer_id - 1) % len(order)
        order = order[shift:] + order[:shift]

    interleaved: List[Dict[str, Any]] = []
    while any(grouped[src] for src in order):
        for src in order:
            if grouped[src]:
                interleaved.append(grouped[src].popleft())
    return interleaved


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
    with path.open("w", encoding="utf-8") as f:
        for rid in record_order:
            review = reviews.get(rid)
            if review is not None:
                f.write(json.dumps(review, ensure_ascii=False) + "\n")


def terminal_width(default: int = 120) -> int:
    try:
        return max(90, shutil.get_terminal_size((default, 30)).columns)
    except OSError:
        return default


def clear_screen(enabled: bool) -> None:
    if not enabled:
        return
    if os.name == "nt":
        os.system("cls")
        return
    # Clear visible screen, scrollback, and return cursor home. This is more
    # reliable than shelling out to `clear` in several terminals.
    print("\033[2J\033[3J\033[H", end="", flush=True)


class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    REVERSE = "\033[7m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def color(text: Any, *codes: str) -> str:
    value = str(text)
    if not COLOR_ENABLED or not codes:
        return value
    return "".join(codes) + value + Ansi.RESET


def section_title(text: str) -> str:
    return color(text, Ansi.BOLD, Ansi.CYAN)


def muted(text: Any) -> str:
    return color(text, Ansi.DIM, Ansi.GRAY)


def option_style(key: str, value: str) -> str:
    return f"  {color(key + '.', Ansi.BOLD, Ansi.YELLOW)} {badge(value)}"


def pill(label: str, value: Any, accent: str = Ansi.CYAN) -> str:
    return color(f" {label}: ", Ansi.BOLD, accent) + color(f"{value} ", Ansi.BOLD)


def badge(value: Any) -> str:
    text = str(value)
    lowered = text.lower()
    if lowered in {"accept", "good", "sufficient", "high", "clear", "specific", "strong", "7-10"}:
        return color(text, Ansi.BOLD, Ansi.GREEN)
    if "borderline" in lowered or lowered in {"medium", "partial", "mixed", "somewhat_clear", "somewhat_underspecified", "4-6", "2-3"}:
        return color(text, Ansi.BOLD, Ansi.YELLOW)
    if lowered in {"reject", "bad", "insufficient", "low", "unclear", "too_underspecified", "weak", "0-1"}:
        return color(text, Ansi.BOLD, Ansi.RED)
    if lowered in {"false"}:
        return color(text, Ansi.GREEN)
    if lowered in {"true"}:
        return color(text, Ansi.YELLOW)
    return text


def wrap(text: Any, width: int, indent: str = "") -> str:
    value = str(text or "").strip()
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


def ellipsize(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 15].rstrip() + " ... [truncated]"


def toggled_snippet_chars(current: int, compact_chars: int) -> int:
    if current <= 0:
        return compact_chars
    return 0


def print_rule(width: int, char: str = "=") -> None:
    line = char * width
    if char == "=":
        print(color(line, Ansi.BLUE))
    elif char == "-":
        print(color(line, Ansi.GRAY))
    else:
        print(line)


def print_screen_banner(width: int, index: int, total: int, reviewed_count: int) -> None:
    print()
    print_rule(width)
    title = " BENCHMARK HUMAN PRESELECTION "
    progress = f" record {index + 1}/{total} | reviewed {reviewed_count} "
    print(color(title.center(width), Ansi.BOLD, Ansi.REVERSE, Ansi.WHITE))
    print(color(progress.center(width), Ansi.BOLD, Ansi.BLUE))
    print_rule(width)


def print_section_header(text: str, width: int, accent: str = Ansi.CYAN) -> None:
    print()
    print(color(text, Ansi.BOLD, accent))
    print(color("=" * width, Ansi.BOLD, accent))
    print()


def print_minor_header(text: str, accent: str = Ansi.CYAN) -> None:
    print()
    print(color(text, Ansi.BOLD, accent))
    print(color("-" * len(text), accent))


def print_pill_row(items: Sequence[str], width: int, indent: str = "  ") -> None:
    line = indent
    for item in items:
        visible_padding = 4
        if len(line) + len(item) + visible_padding > width:
            print(line.rstrip())
            line = indent
        line += item + "  "
    if line.strip():
        print(line.rstrip())


def print_kv(label: str, value: Any, width: int, label_width: int = 24) -> None:
    if value is None:
        text = "-"
    else:
        text = str(value).strip()
        if text == "":
            text = "-"
    lines = textwrap.wrap(
        text,
        width=max(30, width - label_width - 2),
        break_long_words=False,
        break_on_hyphens=False,
    ) or ["-"]
    label_text = f"{label + ':':<{label_width}}"
    print(f"{color(label_text, Ansi.BOLD, Ansi.CYAN)} {lines[0]}")
    for line in lines[1:]:
        print(f"{'':<{label_width}} {line}")


def fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def host(url: str) -> str:
    parsed = urlparse(url or "")
    return parsed.netloc.lower().removeprefix("www.") or "-"


def score_summary(docs: Sequence[Dict[str, Any]]) -> str:
    scores = []
    window_scores = []
    for doc in docs:
        try:
            scores.append(float(doc.get("_search_score")))
        except (TypeError, ValueError):
            pass
        try:
            window_scores.append(float(doc.get("_window_score")))
        except (TypeError, ValueError):
            pass
    parts = []
    if scores:
        parts.append(f"search_score min/avg/max={min(scores):.3f}/{sum(scores)/len(scores):.3f}/{max(scores):.3f}")
    if window_scores:
        parts.append(
            f"window_score min/avg/max={min(window_scores):.3f}/{sum(window_scores)/len(window_scores):.3f}/{max(window_scores):.3f}"
        )
    return "; ".join(parts) if parts else "-"


def retrieval_pills(docs: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> List[str]:
    hosts = {host(str(doc.get("url") or doc.get("source_url") or "")) for doc in docs}
    hosts.discard("-")
    exact_docs = len(docs) == 10
    cache_value = meta.get("search_cache_hit", "-")
    credits = (meta.get("tavily_usage") or {}).get("credits", "-")
    return [
        pill("docs", len(docs), Ansi.GREEN if exact_docs else Ansi.YELLOW),
        pill("unique hosts", len(hosts), Ansi.CYAN),
        pill("requested/raw", f"{meta.get('requested_results', '-')}/{meta.get('raw_docs_before_filter', '-')}", Ansi.MAGENTA),
        pill("cache", badge(cache_value), Ansi.GREEN if cache_value is False else Ansi.YELLOW),
        pill("credits", credits, Ansi.BLUE),
        pill("attempts", meta.get("tavily_attempt_count", "-"), Ansi.BLUE),
    ]


def explain_retrieval_method(method: str, meta: Dict[str, Any]) -> List[str]:
    value = str(method or "").strip()
    if value == "tavily_top10_from20_fullhtml_cloudscraper_justext_tasb512_stride256":
        steps = [
            "Search Tavily for 20 web results for the query.",
            "Fetch readable page/full HTML content when available.",
            "Clean webpages with cloudscraper + jusText to remove boilerplate.",
            "Split cleaned text into overlapping 512-token windows with stride 256.",
            "Use TAS-B semantic scoring to choose the best evidence window per result.",
            "Keep a ranked top-10 candidate evidence pool.",
        ]
        if meta.get("doc_subset_strategy"):
            steps.append(
                "For human preselection, show 5 docs: 2 sampled from current positions 1-5 and 3 sampled from current positions 6-10."
            )
        else:
            steps.append("Keep the top 10 retrieved documents for human review.")
        return steps
    if value:
        return [value.replace("_", " ")]
    requested = meta.get("requested_results", "-")
    return [f"Search provider retrieval; requested {requested} results and kept the ranked evidence windows."]


def doc_accent(_index: int) -> str:
    return Ansi.CYAN


def print_taxonomy(width: int) -> None:
    print_section_header("Conflict Taxonomy - Detailed Definitions", width, Ansi.CYAN)
    print(muted("Use these labels for preliminary selection. Full annotation still happens later in the multi-LLM pipeline."))
    for option, label, definition in CONFLICT_TYPE_DEFINITIONS:
        print()
        print(f"  {color(option + '.', Ansi.BOLD, Ansi.YELLOW)} {color(label, Ansi.BOLD, Ansi.CYAN)}")
        print(wrap(definition, width - 6, indent="      "))


def print_review_checklist(width: int, src: str) -> None:
    print_section_header("Reviewer Focus", width, Ansi.GREEN)
    for bullet in REVIEW_CHECKLIST:
        print(wrap(f"- {bullet}", width - 6, indent="  "))
    if src == "situatedqa_geo":
        print()
        print(color(wrap("Watch out: this source often has underspecified geography. Reject if retrieved docs guess the wrong place or mix unrelated locations.", width - 6, indent="  "), Ansi.BOLD, Ansi.YELLOW))


def render_record(
    row: Dict[str, Any],
    index: int,
    total: int,
    reviewed_count: int,
    snippet_chars: int,
    show_taxonomy: bool,
    show_search_extract: bool,
) -> None:
    width = terminal_width()
    docs = row.get("retrieved_docs") or []
    src = source_family(row)
    source = row.get("_candidate_source") or {}
    meta = row.get("_retrieval_metadata") or {}
    rid = row_id(row)

    print_screen_banner(width, index, total, reviewed_count)

    print_section_header("Candidate", width, Ansi.YELLOW)
    print(color("Question", Ansi.BOLD, Ansi.YELLOW))
    print(color(wrap(row.get("query", ""), width - 4, indent="  "), Ansi.BOLD))
    print()
    print_kv("ID", rid, width)
    print_kv("Source family", color(src, Ansi.BOLD, Ansi.MAGENTA), width)
    print_kv("Source dataset", source.get("source_dataset", ""), width)
    print_kv("Source record ID", source.get("source_record_id", ""), width)
    print_kv("Seed/source answer", color(row.get("gold_answer", "") or source.get("source_answer", "") or "-", Ansi.GREEN), width)
    if source.get("source_metadata"):
        print_kv("Source metadata", json.dumps(source.get("source_metadata"), ensure_ascii=False), width)

    print_review_checklist(width, src)

    if show_taxonomy:
        print_taxonomy(width)

    print_section_header("QA Source Notes", width, Ansi.GREEN)
    for bullet in SOURCE_DESCRIPTIONS.get(src, ["No source description available."]):
        print(wrap(f"- {bullet}", width - 6, indent="  "))
        print()

    print_section_header("Retrieval Summary", width, Ansi.MAGENTA)
    print_pill_row(retrieval_pills(docs, meta), width)
    print()
    print(color("Retrieval method", Ansi.BOLD, Ansi.MAGENTA))
    for step in explain_retrieval_method(str(meta.get("method", "")), meta):
        print(wrap(f"- {step}", width - 6, indent="  "))
    print()
    print_kv("Raw method ID", meta.get("method", ""), width)
    if meta.get("doc_subset_strategy"):
        print_kv("Doc subset", meta.get("doc_subset_strategy", ""), width)
        print_kv("Selected top-10 positions", meta.get("doc_subset_selected_original_positions", ""), width)
    print_kv("Tavily statuses", meta.get("tavily_statuses", "-"), width)
    print_kv("Scores", score_summary(docs), width)

    snippet_mode = "full snippets" if snippet_chars <= 0 else f"compact snippets, {snippet_chars} chars max"
    print_section_header(f"Retrieved Documents ({snippet_mode})", width, Ansi.CYAN)
    for idx, doc in enumerate(docs, 1):
        url = doc.get("url") or doc.get("source_url") or ""
        title = doc.get("title") or "-"
        date = doc.get("date") or doc.get("timestamp") or "-"
        text = doc.get("short_text") or doc.get("snippet") or doc.get("_search_snippet") or ""
        text = ellipsize(text, snippet_chars)
        accent = doc_accent(idx)
        top10_position = doc.get("_original_rank") or doc.get("_rank", idx)
        search_rank = doc.get("_original_search_rank")
        rank_part = f"top10_pos={top10_position}"
        if search_rank and search_rank != top10_position:
            rank_part += f" | search_rank={search_rank}"
        metadata_line = (
            f"{rank_part} | host={host(url)} | date={date} | "
            f"search={fmt_float(doc.get('_search_score'))} | window={fmt_float(doc.get('_window_score'))} | "
            f"fetch={doc.get('_fetch_status', '-')} | full_chars={doc.get('_full_text_chars', '-')}"
        )

        print()
        print(color(f" d{idx} ".ljust(width, "-"), Ansi.BOLD, accent))
        print(color(wrap(str(title), width - 4, indent="  "), Ansi.BOLD))
        print(color(wrap(metadata_line, width - 4, indent="  "), Ansi.DIM))
        print(wrap(str(url), width - 4, indent="  "))
        print_minor_header("Snippet", accent)
        print(wrap(text, width - 4, indent="    "))
        search_snippet = (doc.get("_search_snippet") or "").strip()
        if show_search_extract and search_snippet and search_snippet not in text:
            print_minor_header("Search Extract", Ansi.MAGENTA)
            search_extract_chars = 800 if snippet_chars <= 0 else min(snippet_chars, 800)
            print(wrap(ellipsize(search_snippet, search_extract_chars), width - 4, indent="    "))
        print(color("-" * width, Ansi.DIM, accent))


def prompt_choice(label: str, options: Sequence[Tuple[str, str]], allow_commands: bool = True) -> str:
    print()
    print(section_title(label))
    for key, value in options:
        print(option_style(key, value))
    if allow_commands:
        print()
        print(color("Navigation / display commands", Ansi.DIM))
        print(option_style("s", "skip this record"))
        print(option_style("p", "go to previous record"))
        print(option_style("q", "save and quit"))
        print(option_style("r", "redisplay record"))
        print(option_style("f", "toggle full/compact snippets for this record"))
    valid = {key: value for key, value in options}
    while True:
        raw = input(color("> ", Ansi.BOLD, Ansi.GREEN)).strip().lower()
        if allow_commands and raw in {"s", "p", "q", "r", "f"}:
            return raw
        if raw in valid:
            return valid[raw]
        print(color("Please enter one of the listed options.", Ansi.RED))


def prompt_bool(label: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{section_title(label)} [{suffix}] {color('> ', Ansi.BOLD, Ansi.GREEN)}").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "true", "1"}:
            return True
        if raw in {"n", "no", "false", "0"}:
            return False
        print(color("Please enter y or n.", Ansi.RED))


def prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{section_title(label)}{suffix} {color('> ', Ansi.BOLD, Ansi.GREEN)}").strip()
    return raw if raw else default


def normalize_reviewer_name(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def prompt_reviewer_name(args: argparse.Namespace) -> Tuple[str, int]:
    while True:
        first = args.reviewer_first_name or prompt_text("Reviewer first name")
        normalized = normalize_reviewer_name(first)
        reviewer_id = REVIEWER_ROSTER.get(normalized)

        if args.reviewer_id is not None:
            if not 1 <= args.reviewer_id <= args.reviewers:
                raise SystemExit(f"--reviewer-id must be between 1 and {args.reviewers}")
            expected_name = REVIEWER_NAMES_BY_ID.get(args.reviewer_id)
            if reviewer_id is not None and reviewer_id != args.reviewer_id:
                raise SystemExit(f"Reviewer name '{first}' maps to ID {reviewer_id}, not --reviewer-id {args.reviewer_id}")
            return expected_name or normalized or first, args.reviewer_id

        if reviewer_id is not None:
            if reviewer_id > args.reviewers:
                raise SystemExit(f"Reviewer '{first}' maps to ID {reviewer_id}, but --reviewers is {args.reviewers}")
            print()
            print(f"{section_title('Reviewer matched')} {color(first.strip(), Ansi.BOLD)} -> ID {color(reviewer_id, Ansi.BOLD, Ansi.GREEN)}")
            return normalized, reviewer_id

        print(color("Name not found in reviewer roster. Please enter one of:", Ansi.RED))
        print("  " + ", ".join(f"{name} ({reviewer_id})" for name, reviewer_id in REVIEWER_ROSTER.items()))
        if args.reviewer_first_name:
            raise SystemExit("Unknown --reviewer-first-name")


def build_review_record(
    row: Dict[str, Any],
    reviewer_first_name: str,
    reviewer_id: int,
    decision: str,
    conflict_type: str,
    conflict_type_other: str,
    confidence: str,
    retrieval_quality: str,
    evidence_sufficiency: str,
    conflict_clarity: str,
    query_specificity: str,
    source_reliability: str,
    relevant_doc_count: str,
    gold_answer_possible: bool,
    gold_answer: str,
    reject_reason: str,
    notes: str,
    needs_second: bool,
) -> Dict[str, Any]:
    conflict_type_id = next(key for key, value in CONFLICT_OPTIONS if value == conflict_type)
    return {
        "id": row_id(row),
        "query": row.get("query", ""),
        "source_family": source_family(row),
        "reviewer_first_name": reviewer_first_name,
        "reviewer_id": reviewer_id,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "human_preselect_decision": decision,
        "preliminary_conflict_type_id": int(conflict_type_id),
        "preliminary_conflict_type": conflict_type,
        "preliminary_conflict_type_other": conflict_type_other,
        "preselection_confidence": confidence,
        "retrieval_quality": retrieval_quality,
        "evidence_sufficiency": evidence_sufficiency,
        "conflict_clarity": conflict_clarity,
        "query_specificity": query_specificity,
        "source_reliability": source_reliability,
        "relevant_doc_count_bin": relevant_doc_count,
        "gold_answer_possible": gold_answer_possible,
        "human_gold_answer": gold_answer,
        "reject_reason": reject_reason,
        "needs_second_reviewer": needs_second,
        "reviewer_notes": notes,
        "_candidate_source": row.get("_candidate_source", {}),
        "_retrieval_metadata": row.get("_retrieval_metadata", {}),
    }


REVIEW_EDIT_FIELDS = [
    ("1", "human_preselect_decision", "Human preselect decision"),
    ("2", "preliminary_conflict_type", "Preliminary conflict type"),
    ("3", "preselection_confidence", "Preselection confidence"),
    ("4", "retrieval_quality", "Retrieval quality"),
    ("5", "evidence_sufficiency", "Evidence sufficiency"),
    ("6", "conflict_clarity", "Conflict-type clarity"),
    ("7", "query_specificity", "Query specificity"),
    ("8", "source_reliability", "Source reliability"),
    ("9", "relevant_doc_count_bin", "Approx. relevant docs"),
    ("10", "gold_answer_possible", "Gold answer possible"),
    ("11", "human_gold_answer", "Gold answer text"),
    ("12", "reject_reason", "Reject/borderline reason"),
    ("13", "needs_second_reviewer", "Needs second reviewer"),
    ("14", "reviewer_notes", "Reviewer notes"),
]


def print_review_summary(review: Dict[str, Any]) -> None:
    width = terminal_width()
    print()
    print_section_header("Review Summary Before Save", width, Ansi.YELLOW)
    print_kv("Record ID", review.get("id", "-"), width)
    print_kv("Query", review.get("query", "-"), width)
    print()
    for key, field, label in REVIEW_EDIT_FIELDS:
        value = review.get(field, "")
        if field == "human_gold_answer" and not review.get("gold_answer_possible"):
            value = "-"
        label_text = color(f"{key}. {label + ':':<28}", Ansi.BOLD, Ansi.CYAN)
        print(f"  {label_text} {badge(value)}")
    print()


def edit_review_field(review: Dict[str, Any]) -> None:
    print_review_summary(review)
    print(section_title("Choose field to edit"))
    for key, _field, label in REVIEW_EDIT_FIELDS:
        print(option_style(key, label))
    print(option_style("c", "cancel edit"))
    raw = input(color("> ", Ansi.BOLD, Ansi.GREEN)).strip().lower()
    field_map = {key: field for key, field, _label in REVIEW_EDIT_FIELDS}
    field = field_map.get(raw)
    if raw == "c" or not field:
        return

    if field == "human_preselect_decision":
        review[field] = prompt_choice("Human preselect decision", DECISION_OPTIONS, allow_commands=False)
    elif field == "preliminary_conflict_type":
        conflict_type = prompt_choice("Preliminary conflict type", CONFLICT_OPTIONS, allow_commands=False)
        review[field] = conflict_type
        review["preliminary_conflict_type_id"] = int(next(key for key, value in CONFLICT_OPTIONS if value == conflict_type))
        review["preliminary_conflict_type_other"] = (
            prompt_text("Briefly name other conflict type")
            if conflict_type == "Other conflict type"
            else ""
        )
    elif field == "preselection_confidence":
        review[field] = prompt_choice("Preselection confidence", CONFIDENCE_OPTIONS, allow_commands=False)
    elif field == "retrieval_quality":
        review[field] = prompt_choice("Retrieval quality", RETRIEVAL_QUALITY_OPTIONS, allow_commands=False)
    elif field == "evidence_sufficiency":
        review[field] = prompt_choice("Evidence sufficiency", EVIDENCE_SUFFICIENCY_OPTIONS, allow_commands=False)
    elif field == "conflict_clarity":
        review[field] = prompt_choice("Conflict-type clarity", CONFLICT_CLARITY_OPTIONS, allow_commands=False)
    elif field == "query_specificity":
        review[field] = prompt_choice("Query specificity", QUERY_SPECIFICITY_OPTIONS, allow_commands=False)
    elif field == "source_reliability":
        review[field] = prompt_choice("Source reliability", SOURCE_RELIABILITY_OPTIONS, allow_commands=False)
    elif field == "relevant_doc_count_bin":
        review[field] = prompt_choice("Approx. relevant docs", RELEVANT_DOC_COUNT_OPTIONS, allow_commands=False)
    elif field == "gold_answer_possible":
        possible = prompt_bool("Is a gold answer possible for this query?", default=bool(review.get(field)))
        review[field] = possible
        review["human_gold_answer"] = prompt_text("One-line gold answer", review.get("human_gold_answer", "")) if possible else ""
    elif field == "human_gold_answer":
        if not review.get("gold_answer_possible"):
            print(color("Gold answer is currently marked impossible. Turn field 10 to yes first.", Ansi.YELLOW))
        else:
            review[field] = prompt_text("One-line gold answer", review.get(field, ""))
    elif field == "reject_reason":
        review[field] = prompt_text("Reject/borderline reason", review.get(field, ""))
    elif field == "needs_second_reviewer":
        review[field] = prompt_bool("Needs second reviewer", default=bool(review.get(field)))
    elif field == "reviewer_notes":
        review[field] = prompt_text("Reviewer notes", review.get(field, ""))

    review["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()


def confirm_review(review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    while True:
        print_review_summary(review)
        print(section_title("Save this review?"))
        print(option_style("y", "save and move to next record"))
        print(option_style("e", "edit a field"))
        print(option_style("a", "redo annotation for this record"))
        print(option_style("r", "redisplay record"))
        print(option_style("s", "skip without saving"))
        print(option_style("q", "save existing reviews and quit"))
        raw = input(color("> ", Ansi.BOLD, Ansi.GREEN)).strip().lower()
        if raw in {"y", "yes", ""}:
            return review
        if raw == "e":
            edit_review_field(review)
            continue
        if raw == "a":
            return None
        if raw in {"r", "s", "q"}:
            return {"_command": raw}
        print(color("Please choose y, e, a, r, s, or q.", Ansi.RED))


def review_one_record(
    row: Dict[str, Any],
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
        clear_screen(not args.no_clear)
        render_record(row, index, total, reviewed_count, snippet_chars, show_taxonomy, args.show_search_extract)

        print()
        print(section_title("Enter Preselection Annotation"))
        print(color("Tip: use f/r/p/s/q from the first menu to change view or navigate.", Ansi.DIM))
        decision = prompt_choice("Human preselect decision", DECISION_OPTIONS)
        if decision in {"q", "s", "p"}:
            return {"_command": decision}
        if decision == "r":
            continue
        if decision == "f":
            snippet_chars = toggled_snippet_chars(snippet_chars, compact_snippet_chars)
            continue

        conflict_type = prompt_choice("Preliminary conflict type", CONFLICT_OPTIONS, allow_commands=False)
        conflict_type_other = ""
        if conflict_type == "Other conflict type":
            conflict_type_other = prompt_text("Briefly name other conflict type")

        confidence = prompt_choice("Preselection confidence", CONFIDENCE_OPTIONS, allow_commands=False)
        retrieval_quality = prompt_choice("Retrieval quality", RETRIEVAL_QUALITY_OPTIONS, allow_commands=False)
        evidence_sufficiency = prompt_choice("Evidence sufficiency", EVIDENCE_SUFFICIENCY_OPTIONS, allow_commands=False)
        conflict_clarity = prompt_choice("Conflict-type clarity", CONFLICT_CLARITY_OPTIONS, allow_commands=False)
        query_specificity = prompt_choice("Query specificity", QUERY_SPECIFICITY_OPTIONS, allow_commands=False)
        source_reliability = prompt_choice("Source reliability", SOURCE_RELIABILITY_OPTIONS, allow_commands=False)
        relevant_doc_count = prompt_choice("Approx. relevant docs", RELEVANT_DOC_COUNT_OPTIONS, allow_commands=False)
        gold_answer_possible = prompt_bool("Is a gold answer possible for this query?", default=False)
        gold_answer = prompt_text("One-line gold answer") if gold_answer_possible else ""
        reject_reason = ""
        if (
            decision in {"borderline_reject", "reject"}
            or retrieval_quality == "bad"
            or evidence_sufficiency == "insufficient"
            or conflict_clarity == "unclear"
            or query_specificity == "too_underspecified"
        ):
            reject_reason = prompt_text("Reject/borderline reason")
        notes = prompt_text("Optional short reviewer note")
        needs_second = prompt_bool(
            "Needs second reviewer",
            default=(
                decision.startswith("borderline")
                or conflict_type in {"Other conflict type", "Unsure"}
                or confidence == "low"
                or conflict_clarity == "unclear"
                or query_specificity != "specific"
            ),
        )

        review = build_review_record(
            row=row,
            reviewer_first_name=reviewer_first_name,
            reviewer_id=reviewer_id,
            decision=decision,
            conflict_type=conflict_type,
            conflict_type_other=conflict_type_other,
            confidence=confidence,
            retrieval_quality=retrieval_quality,
            evidence_sufficiency=evidence_sufficiency,
            conflict_clarity=conflict_clarity,
            query_specificity=query_specificity,
            source_reliability=source_reliability,
            relevant_doc_count=relevant_doc_count,
            gold_answer_possible=gold_answer_possible,
            gold_answer=gold_answer,
            reject_reason=reject_reason,
            notes=notes,
            needs_second=needs_second,
        )
        confirmed = confirm_review(review)
        if confirmed is None:
            continue
        if confirmed.get("_command") == "r":
            continue
        return confirmed


def load_assignment(path: Path) -> List[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict) and isinstance(raw.get("record_ids"), list):
        return [str(x) for x in raw["record_ids"]]
    raise ValueError(f"Assignment file must be a list or contain record_ids: {path}")


def apply_assignment(rows: List[Dict[str, Any]], assignment_path: Optional[Path]) -> List[Dict[str, Any]]:
    if assignment_path is None:
        return sorted(rows, key=stable_sort_key)
    ids = load_assignment(assignment_path)
    by_id = {row_id(row): row for row in rows}
    missing = [rid for rid in ids if rid not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} assignment IDs are not in input; first missing: {preview}")
    return [by_id[rid] for rid in ids]


def default_assignment_path(reviewer_id: int) -> Path:
    return DEFAULT_ASSIGNMENT_DIR / f"reviewer_{reviewer_id}_ids.json"


def default_review_path_for_reviewer(reviewer_id: int) -> Path:
    return DEFAULT_REVIEW_DIR / f"reviewer_{reviewer_id}_reviews.jsonl"


def print_assignment_overview(
    reviewer_first_name: str,
    reviewer_id: int,
    rows: Sequence[Dict[str, Any]],
    reviews: Dict[str, Dict[str, Any]],
    assignment_path: Optional[Path],
    output: Path,
) -> None:
    width = terminal_width()
    print()
    print_rule(width)
    print(section_title("Reviewer Assignment"))
    print_rule(width)
    print_kv("Reviewer", f"{reviewer_first_name} (ID {reviewer_id})", width)
    print_kv("Total assigned queries", len(rows), width)
    print_kv("Already reviewed", len(reviews), width)
    if assignment_path:
        print_kv("Assignment file", assignment_path, width)
    print_kv("Review output", output, width)
    print()
    print(section_title("Assigned queries per dataset"))
    counts = Counter(source_family(row) for row in rows)
    for src in SOURCE_PRIORITY + sorted(set(counts) - set(SOURCE_PRIORITY)):
        print(f"  {color(src + ':', Ansi.BOLD, Ansi.CYAN):<32} {counts[src]}")
    print_rule(width, "-")


def make_assignments(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.input))
    rows = [row for row in rows if row_id(row)]
    if args.required_doc_count is not None:
        rows = [row for row in rows if len(row.get("retrieved_docs") or []) == args.required_doc_count]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[source_family(row)].append(row)

    selected: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    target_total = args.target_total or len(rows)

    # Fill in priority order, preserving all records when target_total >= len(rows).
    for src in SOURCE_PRIORITY + sorted(set(grouped) - set(SOURCE_PRIORITY)):
        bucket = grouped.get(src, [])
        bucket.sort(key=lambda row: stable_hash(row_id(row), args.seed))
        if len(selected) >= target_total:
            break
        remaining = target_total - len(selected)
        take = bucket if target_total >= len(rows) else bucket[:remaining]
        selected.extend(take)
        source_counts[src] += len(take)

    reviewer_loads = parse_reviewer_loads(args.reviewer_loads, args.reviewers, len(selected))

    # Source-balanced assignment across reviewers with explicit reviewer capacity.
    queues: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(1, args.reviewers + 1)}
    reviewer_source_counts: Dict[int, Counter[str]] = {idx: Counter() for idx in range(1, args.reviewers + 1)}
    selected_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_source[source_family(row)].append(row)

    for src in SOURCE_PRIORITY + sorted(set(selected_by_source) - set(SOURCE_PRIORITY)):
        bucket = selected_by_source.get(src, [])
        bucket.sort(key=lambda row: stable_hash(row_id(row), args.seed + 1000))
        for row in bucket:
            candidates = [
                reviewer_id
                for reviewer_id in range(1, args.reviewers + 1)
                if len(queues[reviewer_id]) < reviewer_loads[reviewer_id - 1]
            ]
            if not candidates:
                raise RuntimeError("No reviewer capacity left while assigning records")
            reviewer = min(
                candidates,
                key=lambda reviewer_id: (
                    reviewer_source_counts[reviewer_id][src] / max(1, reviewer_loads[reviewer_id - 1]),
                    len(queues[reviewer_id]) / max(1, reviewer_loads[reviewer_id - 1]),
                    stable_hash(f"{src}:{row_id(row)}:{reviewer_id}", args.seed + 2000),
                ),
            )
            queues[reviewer].append(row)
            reviewer_source_counts[reviewer][src] += 1

    queues = {
        reviewer_id: interleave_by_source(queue_rows, reviewer_id, args.seed + 3000)
        for reviewer_id, queue_rows in queues.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input": portable_path(args.input),
        "reviewers": args.reviewers,
        "target_total": target_total,
        "selected": len(selected),
        "required_doc_count": args.required_doc_count,
        "seed": args.seed,
        "reviewer_loads": reviewer_loads,
        "source_priority": SOURCE_PRIORITY,
        "source_counts": dict(Counter(source_family(row) for row in selected)),
        "reviewer_files": {},
    }

    for reviewer_id, queue_rows in queues.items():
        record_ids = [row_id(row) for row in queue_rows]
        payload = {
            "reviewer_id": reviewer_id,
            "input": portable_path(args.input),
            "record_ids": record_ids,
            "record_count": len(record_ids),
            "target_load": reviewer_loads[reviewer_id - 1],
            "source_counts": dict(Counter(source_family(row) for row in queue_rows)),
            "order_strategy": "source_interleaved_capacity_balanced",
        }
        path = output_dir / f"reviewer_{reviewer_id}_ids.json"
        write_json(path, payload)
        manifest["reviewer_files"][str(reviewer_id)] = str(path)

    manifest_path = output_dir / "assignment_manifest.json"
    write_json(manifest_path, manifest)
    print(f"wrote assignments to {output_dir}")
    print(f"manifest: {manifest_path}")
    for reviewer_id in range(1, args.reviewers + 1):
        counts = Counter(source_family(row) for row in queues[reviewer_id])
        print(f"reviewer {reviewer_id}: {len(queues[reviewer_id])} records {dict(counts)}")


def summarize_reviews(path: Path) -> None:
    reviews = list(load_reviews(path).values())
    print(f"reviews={len(reviews)} path={path}")
    for field in [
        "human_preselect_decision",
        "preliminary_conflict_type",
        "preselection_confidence",
        "retrieval_quality",
        "evidence_sufficiency",
        "conflict_clarity",
        "query_specificity",
        "source_reliability",
        "relevant_doc_count_bin",
        "source_family",
    ]:
        print(f"{field}: {dict(Counter(str(row.get(field, '')) for row in reviews))}")
    print(f"needs_second_reviewer: {dict(Counter(bool(row.get('needs_second_reviewer')) for row in reviews))}")


def load_review_files(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for rid, review in load_reviews(Path(path)).items():
            merged[rid] = review
    return merged


def review_is_export_eligible(review: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    decision = review.get("human_preselect_decision")
    allowed_decisions = {"accept"}
    if args.include_borderline_accept:
        allowed_decisions.add("borderline_accept")
    if decision not in allowed_decisions:
        return False, "decision"

    label = review.get("preliminary_conflict_type")
    if label not in VALID_PIPELINE_CONFLICT_TYPES:
        return False, "invalid_or_unsure_label"

    if CONFIDENCE_RANK.get(review.get("preselection_confidence"), 0) < CONFIDENCE_RANK[args.min_preselection_confidence]:
        return False, "low_preselection_confidence"
    if QUALITY_RANK.get(review.get("retrieval_quality"), 0) < QUALITY_RANK[args.min_retrieval_quality]:
        return False, "low_retrieval_quality"
    if SUFFICIENCY_RANK.get(review.get("evidence_sufficiency"), 0) < SUFFICIENCY_RANK[args.min_evidence_sufficiency]:
        return False, "low_evidence_sufficiency"
    if CLARITY_RANK.get(review.get("conflict_clarity", "clear"), 0) < CLARITY_RANK[args.min_conflict_clarity]:
        return False, "low_conflict_clarity"
    if SPECIFICITY_RANK.get(review.get("query_specificity", "specific"), 0) < SPECIFICITY_RANK[args.min_query_specificity]:
        return False, "low_query_specificity"
    if RELIABILITY_RANK.get(review.get("source_reliability", "strong"), 0) < RELIABILITY_RANK[args.min_source_reliability]:
        return False, "low_source_reliability"
    return True, "eligible"


def export_score(review: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int, str]:
    return (
        DECISION_RANK.get(review.get("human_preselect_decision"), 0),
        CONFIDENCE_RANK.get(review.get("preselection_confidence"), 0),
        QUALITY_RANK.get(review.get("retrieval_quality"), 0),
        SUFFICIENCY_RANK.get(review.get("evidence_sufficiency"), 0),
        CLARITY_RANK.get(review.get("conflict_clarity", "clear"), 0),
        SPECIFICITY_RANK.get(review.get("query_specificity", "specific"), 0),
        RELIABILITY_RANK.get(review.get("source_reliability", "strong"), 0),
        str(review.get("id", "")),
    )


def select_balanced_by_human_label(rows: List[Dict[str, Any]], target: Optional[int]) -> List[Dict[str, Any]]:
    if not target or len(rows) <= target:
        return rows

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        review = row.get("_human_preselection") or {}
        by_label[row["conflict_type"]].append(row)
    for label in VALID_PIPELINE_CONFLICT_TYPES:
        by_label[label].sort(key=lambda row: export_score(row.get("_human_preselection") or {}), reverse=True)

    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    cursors = {label: 0 for label in VALID_PIPELINE_CONFLICT_TYPES}
    while len(selected) < target:
        active = [label for label in VALID_PIPELINE_CONFLICT_TYPES if cursors[label] < len(by_label[label])]
        if not active:
            break
        counts = Counter(row["conflict_type"] for row in selected)
        active.sort(key=lambda label: (counts[label], VALID_PIPELINE_CONFLICT_TYPES.index(label)))
        made_progress = False
        for label in active:
            bucket = by_label[label]
            while cursors[label] < len(bucket) and row_id(bucket[cursors[label]]) in selected_ids:
                cursors[label] += 1
            if cursors[label] >= len(bucket):
                continue
            row = bucket[cursors[label]]
            cursors[label] += 1
            selected.append(row)
            selected_ids.add(row_id(row))
            made_progress = True
            if len(selected) >= target:
                break
        if not made_progress:
            break
    return selected


def export_selected(args: argparse.Namespace) -> None:
    if not args.reviews:
        raise SystemExit("--export-selected requires at least one --reviews path")
    if not args.output:
        raise SystemExit("--export-selected requires --output")

    rows = read_jsonl(Path(args.input))
    reviews = load_review_files(args.reviews)
    by_id = {row_id(row): row for row in rows}
    eligible: List[Dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()

    for rid, review in reviews.items():
        row = by_id.get(rid)
        if row is None:
            reject_reasons["missing_input_row"] += 1
            continue
        ok, reason = review_is_export_eligible(review, args)
        if not ok:
            reject_reasons[reason] += 1
            continue

        label = review["preliminary_conflict_type"]
        out = dict(row)
        out["conflict_type"] = label
        out["conflict_category_id"] = CONFLICT_CATEGORY_IDS[label]
        out["conflict_reason"] = "Human preselection placeholder; run benchmark-mode Stage-2 to generate final conflict reason."
        if str(review.get("human_gold_answer", "")).strip():
            out["gold_answer"] = str(review["human_gold_answer"]).strip()
        out["_human_preselection"] = review
        eligible.append(out)

    eligible.sort(key=lambda row: export_score(row.get("_human_preselection") or {}), reverse=True)
    selected = select_balanced_by_human_label(eligible, args.target_total)
    write_jsonl(Path(args.output), selected)

    manifest = {
        "input": args.input,
        "reviews": args.reviews,
        "output": args.output,
        "review_rows": len(reviews),
        "eligible_rows": len(eligible),
        "selected_rows": len(selected),
        "target_total": args.target_total,
        "include_borderline_accept": args.include_borderline_accept,
        "min_preselection_confidence": args.min_preselection_confidence,
        "min_retrieval_quality": args.min_retrieval_quality,
        "min_evidence_sufficiency": args.min_evidence_sufficiency,
        "min_conflict_clarity": args.min_conflict_clarity,
        "min_query_specificity": args.min_query_specificity,
        "min_source_reliability": args.min_source_reliability,
        "reject_reasons": dict(reject_reasons),
        "eligible_distribution": dict(Counter(row["conflict_type"] for row in eligible)),
        "selected_distribution": dict(Counter(row["conflict_type"] for row in selected)),
    }
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(f"selected={len(selected)} eligible={len(eligible)} reviews={len(reviews)}")
    print(f"selected_distribution={manifest['selected_distribution']}")
    print(f"reject_reasons={manifest['reject_reasons']}")
    print(f"manifest={manifest_path}")


def run_review(args: argparse.Namespace) -> None:
    reviewer_first_name, reviewer_id = prompt_reviewer_name(args)
    assignment_path = Path(args.assignment) if args.assignment else default_assignment_path(reviewer_id)
    if not assignment_path.exists():
        print(f"Warning: assignment file not found: {assignment_path}")
        print("Falling back to all input rows sorted by source priority.")
        assignment_path = None

    output = Path(args.output) if args.output else default_review_path_for_reviewer(reviewer_id)

    rows = read_jsonl(Path(args.input))
    rows = apply_assignment(rows, assignment_path)
    if args.limit:
        rows = rows[: args.limit]
    order = [row_id(row) for row in rows]
    if any(not rid for rid in order):
        raise ValueError("Every input row must have a non-empty id.")

    reviews = load_reviews(output)

    print_assignment_overview(reviewer_first_name, reviewer_id, rows, reviews, assignment_path, output)
    input("Press Enter to start reviewing...")

    index = 0
    while index < len(rows):
        row = rows[index]
        rid = row_id(row)
        if rid in reviews and not args.review_all:
            index += 1
            continue

        result = review_one_record(row, args, reviewer_first_name, reviewer_id, index, len(rows), len(reviews))
        if result is None:
            index += 1
            continue
        command = result.get("_command")
        if command == "q":
            save_reviews(output, reviews, order)
            print(f"Saved {len(reviews)} reviews to {output}")
            return
        if command == "s":
            index += 1
            continue
        if command == "p":
            index = max(0, index - 1)
            continue

        reviews[rid] = result
        save_reviews(output, reviews, order)
        index += 1

    save_reviews(output, reviews, order)
    print(f"Completed assigned queue. Saved {len(reviews)} reviews to {output}")
    summarize_reviews(output)


def default_review_output(args: argparse.Namespace) -> Path:
    input_path = Path(args.input)
    if args.assignment:
        assignment = Path(args.assignment)
        return Path("data/benchmark_build/human_preselection/reviews") / f"{assignment.stem}_reviews.jsonl"
    return input_path.with_name(f"{input_path.stem}_human_preselection_reviews.jsonl")


def main() -> None:
    global COLOR_ENABLED
    ap = argparse.ArgumentParser(description="Human preselection CLI for benchmark candidates")
    ap.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Retrieved benchmark JSONL with candidate records (default: {DEFAULT_INPUT})",
    )
    ap.add_argument("--assignment", default=None, help="Optional reviewer assignment JSON containing record_ids")
    ap.add_argument("--output", default=None, help="Review JSONL output path")
    ap.add_argument("--reviewer-first-name", default=None)
    ap.add_argument("--reviewer-id", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--snippet-chars", type=int, default=0, help="Max chars displayed per doc snippet; <=0 shows full snippets. Default shows full snippets.")
    ap.add_argument("--compact-snippet-chars", type=int, default=DEFAULT_COMPACT_SNIPPET_CHARS, help="Max chars used when pressing f to switch into compact snippet view")
    ap.add_argument("--no-clear", action="store_true", help="Do not clear terminal between records")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    ap.add_argument("--hide-taxonomy", action="store_true", help="Hide taxonomy guidance in the per-record view")
    ap.add_argument("--show-search-extract", action="store_true", help="Also display raw search-provider extract text under each selected snippet")
    ap.add_argument("--review-all", action="store_true", help="Review records even if they already have saved reviews")

    ap.add_argument("--make-assignments", action="store_true", help="Create reviewer assignment ID files and exit")
    ap.add_argument("--output-dir", default=str(DEFAULT_ASSIGNMENT_DIR))
    ap.add_argument("--reviewers", type=int, default=7)
    ap.add_argument("--reviewer-loads", default=None, help="Optional comma-separated assignment counts, e.g. 150,150,300,300,326,326,326")
    ap.add_argument("--target-total", type=int, default=None, help="Optional max number of records to assign")
    ap.add_argument("--seed", type=int, default=62002)
    ap.add_argument("--required-doc-count", type=int, default=DEFAULT_REQUIRED_DOC_COUNT, help="Only assign rows with this many retrieved docs; use -1 to disable")
    ap.add_argument("--exact10-only", action="store_true", help=argparse.SUPPRESS)

    ap.add_argument("--summarize", action="store_true", help="Summarize an existing review output and exit")
    ap.add_argument("--export-selected", action="store_true", help="Merge review files and export selected rows for the LLM pipeline")
    ap.add_argument("--reviews", nargs="*", default=[], help="Review JSONL files for --export-selected")
    ap.add_argument("--include-borderline-accept", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-preselection-confidence", choices=["high", "medium", "low"], default="medium")
    ap.add_argument("--min-retrieval-quality", choices=["good", "partial", "bad"], default="partial")
    ap.add_argument("--min-evidence-sufficiency", choices=["sufficient", "borderline", "insufficient"], default="borderline")
    ap.add_argument("--min-conflict-clarity", choices=["clear", "somewhat_clear", "unclear"], default="somewhat_clear")
    ap.add_argument("--min-query-specificity", choices=["specific", "somewhat_underspecified", "too_underspecified"], default="somewhat_underspecified")
    ap.add_argument("--min-source-reliability", choices=["strong", "mixed", "weak"], default="mixed")
    args = ap.parse_args()

    COLOR_ENABLED = not args.no_color and "NO_COLOR" not in os.environ

    if args.reviewers < 1:
        raise SystemExit("--reviewers must be >= 1")
    if args.exact10_only:
        args.required_doc_count = 10
    if args.required_doc_count is not None and args.required_doc_count < 0:
        args.required_doc_count = None
    if args.make_assignments:
        make_assignments(args)
        return
    if args.summarize:
        output = Path(args.output) if args.output else default_review_output(args)
        summarize_reviews(output)
        return
    if args.export_selected:
        export_selected(args)
        return
    run_review(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
