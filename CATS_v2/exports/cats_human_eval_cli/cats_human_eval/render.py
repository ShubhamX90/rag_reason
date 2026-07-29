from __future__ import annotations

import shutil
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .prompts import BEHAVIOR_GUIDE, BEHAVIOR_RUBRIC, FG_GUIDE, STR_GUIDE, behavior_provenance_lines


def app_console() -> Console:
    width = min(shutil.get_terminal_size((140, 40)).columns, 140)
    return Console(highlight=False, width=width)


def _kv_table(title: str, rows: List[tuple[str, str]]) -> Table:
    table = Table(title=title, show_header=False, box=box.SIMPLE, pad_edge=False, expand=False)
    table.add_column("field", style="bold cyan", width=28)
    table.add_column("value", style="white")
    for key, value in rows:
        table.add_row(key, value)
    return table


def _confidence_text(payload: Dict[str, Any]) -> str:
    label = str(payload.get("confidence_label") or "").strip()
    numeric = payload.get("confidence")
    if label and numeric is not None:
        return f"{label} ({numeric})"
    if label:
        return label
    return str(numeric)


def render_dashboard(
    console: Console,
    study_name: str,
    reviewer: str,
    progress: Dict[str, int],
    sample_count: int,
) -> None:
    title = Text(f"{study_name}", style="bold bright_magenta")
    subtitle = Text(
        f"reviewer={reviewer}   submitted={progress['submitted']}/{sample_count}   drafts={progress['drafts']}",
        style="bold white",
    )
    console.print(Rule(style="bright_magenta"))
    console.print(Panel.fit(subtitle, title=title, border_style="bright_magenta", padding=(0, 2)))
    console.print(Rule(style="bright_magenta"))


def render_sample_overview(
    console: Console,
    sample: Dict[str, Any],
    position: int,
    total: int,
    autosave_label: str,
) -> None:
    console.print(Rule(f"[bold bright_magenta]CATS HUMAN EVAL[/bold bright_magenta]   sample {position}/{total}", style="bright_magenta"))
    console.print(
        _kv_table(
            "Query Header",
            [
                ("Sample ID", sample["sample_id"]),
                ("Gold Conflict Label", f"{sample.get('conflict_category_id')} | {sample.get('conflict_type') or 'Unknown'}"),
                ("Gold answerable", str(sample.get("gold_answerable"))),
                ("Correct refusal", str(sample.get("correct_refusal"))),
                ("Pred answered", str(sample.get("pred_answered"))),
                ("Autosave status", autosave_label),
            ],
        )
    )
    if sample.get("conflict_reason"):
        console.print(Panel(sample["conflict_reason"], title="Gold Conflict Reason", border_style="green", padding=(0, 1)))
    if sample.get("gold_answer"):
        gold_answer_text = Text(str(sample["gold_answer"]), no_wrap=False, overflow="fold")
        console.print(Panel(gold_answer_text, title="Gold Answer", border_style="green", padding=(0, 1)))
    console.print(Panel(sample.get("query", ""), title="Query", border_style="cyan", padding=(0, 1)))
    think = sample.get("think_trace") or ""
    if think:
        think_block = f"<think>\n{think}\n</think>"
        think_text = Text(think_block, no_wrap=False, overflow="fold")
        console.print(Panel(think_text, title="Model Think Trace", border_style="magenta", padding=(0, 1)))
    response = sample.get("stripped_answer") or "(empty answer)"
    response_text = Text(response, no_wrap=False, overflow="fold")
    console.print(Panel(response_text, title="Model Final Answer", border_style="yellow", padding=(0, 1)))
    console.print(
        _kv_table(
            "Claims / Evidence Summary",
            [
                ("Extracted claims", str(len(sample.get("claims_with_citations") or []))),
                ("Retrieved docs", str(len(sample.get("retrieved_docs") or []))),
                ("FG-eligible docs", str(len(sample.get("fg_eligible_docs") or []))),
                ("STR applicable", str(sample.get("single_truth_applicable"))),
            ],
        )
    )


def render_help(console: Console) -> None:
    table = Table(title="Session Commands", box=box.SIMPLE, pad_edge=False, expand=False)
    table.add_column("key", style="bold green", width=6)
    table.add_column("action", style="white")
    rows = [
        ("o", "redisplay sample overview"),
        ("d", "browse retrieved docs"),
        ("b", "edit behavior judgment"),
        ("f", "edit factual grounding judgments"),
        ("t", "edit single-truth recall judgment"),
        ("m", "edit reviewer notes"),
        ("r", "show current review summary"),
        ("s", "save current state"),
        ("x", "submit sample"),
        ("n", "next sample"),
        ("p", "previous sample"),
        ("j", "jump to assigned sample number"),
        ("q", "save current state and quit"),
        ("h", "show command help"),
    ]
    for key, action in rows:
        table.add_row(key, action)
    console.print(table)
    console.print("[dim]Tip: use overview as the home screen, then open docs or metric-specific pages as needed.[/dim]")


def render_docs(console: Console, sample: Dict[str, Any]) -> None:
    console.print(Rule("[bold cyan]Retrieved Documents[/bold cyan]", style="cyan"))
    docs = sample.get("docs_with_notes") or []
    for doc in docs:
        header = f"{doc.get('doc_id')} | verdict={doc.get('verdict') or 'n/a'} | date={doc.get('timestamp') or 'unknown'}"
        meta = doc.get("source_url") or ""
        body_parts = []
        if doc.get("title"):
            body_parts.append(f"Title: {doc['title']}")
        if meta:
            body_parts.append(meta)
        if doc.get("key_fact"):
            body_parts.append(f"Gold key fact: {doc['key_fact']}")
        if doc.get("quote"):
            body_parts.append(f"Gold quote: {doc['quote']}")
        if doc.get("snippet"):
            body_parts.append(f"Snippet:\n{doc['snippet']}")
        body_text = Text("\n\n".join(body_parts) if body_parts else "(no document content)", no_wrap=False, overflow="fold")
        console.print(Panel(body_text, title=header, border_style="cyan", padding=(0, 1)))


def render_behavior_guide(console: Console, sample: Dict[str, Any]) -> None:
    conflict_id = int(sample.get("conflict_category_id") or 1)
    rubric = BEHAVIOR_RUBRIC.get(conflict_id, BEHAVIOR_RUBRIC[1])
    console.print(
        _kv_table(
            "Behavior Review Context",
            [
                ("Sample ID", sample["sample_id"]),
                ("Gold Conflict Label", f"{sample.get('conflict_category_id')} | {sample.get('conflict_type') or 'Unknown'}"),
                ("Gold answerable", str(sample.get("gold_answerable"))),
                ("Correct refusal", str(sample.get("correct_refusal"))),
            ],
        )
    )
    if sample.get("conflict_reason"):
        reason_text = Text(str(sample["conflict_reason"]), no_wrap=False, overflow="fold")
        console.print(Panel(reason_text, title="Gold Conflict Reason", border_style="green", padding=(0, 1)))
    if sample.get("gold_answer"):
        gold_answer_text = Text(str(sample["gold_answer"]), no_wrap=False, overflow="fold")
        console.print(Panel(gold_answer_text, title="Gold Answer", border_style="green", padding=(0, 1)))
    query_text = Text(sample.get("query", ""), no_wrap=False, overflow="fold")
    console.print(Panel(query_text, title="Query", border_style="cyan", padding=(0, 1)))
    response = sample.get("stripped_answer") or "(empty answer)"
    response_text = Text(response, no_wrap=False, overflow="fold")
    console.print(Panel(response_text, title="Model Final Answer", border_style="yellow", padding=(0, 1)))

    guide_lines = [BEHAVIOR_GUIDE, "", f"Current conflict-type rubric:\n- {rubric}"]
    provenance_lines = behavior_provenance_lines(conflict_id, sample.get("docs_with_notes") or [])
    if provenance_lines:
        guide_lines.extend(["", "Document provenance for this behavior judgment:", *provenance_lines])
    guide_text = Text("\n".join(guide_lines), no_wrap=False, overflow="fold")
    console.print(Panel(guide_text, title="Behavior Guide", border_style="green", padding=(0, 1)))

    docs = sample.get("docs_with_notes") or []
    if docs:
        doc_table = Table(title="Gold Per-Doc Notes", box=box.SIMPLE, pad_edge=False, expand=True)
        doc_table.add_column("doc_id", style="bold cyan", width=8)
        doc_table.add_column("gold verdict", style="green", width=20)
        doc_table.add_column("gold key fact", style="white", overflow="fold")
        for doc in docs:
            doc_table.add_row(
                doc.get("doc_id") or "",
                doc.get("verdict") or "n/a",
                doc.get("key_fact") or "",
            )
        console.print(doc_table)


def render_fg_guide(console: Console, sample: Dict[str, Any]) -> None:
    console.print(
        _kv_table(
            "FG Review Context",
            [
                ("Sample ID", sample["sample_id"]),
                ("Gold Conflict Label", f"{sample.get('conflict_category_id')} | {sample.get('conflict_type') or 'Unknown'}"),
                ("Gold answerable", str(sample.get("gold_answerable"))),
                ("Correct refusal", str(sample.get("correct_refusal"))),
                ("Extracted claims", str(len(sample.get("claims_with_citations") or []))),
                ("FG-eligible docs", str(len(sample.get("fg_eligible_docs") or []))),
            ],
        )
    )
    if sample.get("conflict_reason"):
        reason_text = Text(str(sample["conflict_reason"]), no_wrap=False, overflow="fold")
        console.print(Panel(reason_text, title="Gold Conflict Reason", border_style="green", padding=(0, 1)))
    if sample.get("gold_answer"):
        gold_answer_text = Text(str(sample["gold_answer"]), no_wrap=False, overflow="fold")
        console.print(Panel(gold_answer_text, title="Gold Answer", border_style="green", padding=(0, 1)))
    query_text = Text(sample.get("query", ""), no_wrap=False, overflow="fold")
    console.print(Panel(query_text, title="Query", border_style="cyan", padding=(0, 1)))
    response = sample.get("stripped_answer") or "(empty answer)"
    response_text = Text(response, no_wrap=False, overflow="fold")
    console.print(Panel(response_text, title="Model Final Answer", border_style="yellow", padding=(0, 1)))
    console.print(Panel(FG_GUIDE, title="FG Guide", border_style="blue", padding=(0, 1)))

    docs = sample.get("fg_eligible_docs") or []
    if docs:
        doc_table = Table(title="FG-Eligible Gold Per-Doc Notes", box=box.SIMPLE, pad_edge=False, expand=True)
        doc_table.add_column("doc_id", style="bold cyan", width=8)
        doc_table.add_column("gold verdict", style="green", width=20)
        doc_table.add_column("gold key fact", style="white", overflow="fold")
        for doc in docs:
            doc_table.add_row(
                doc.get("doc_id") or "",
                doc.get("verdict") or "n/a",
                doc.get("key_fact") or "",
            )
        console.print(doc_table)


def render_str_guide(console: Console, sample: Dict[str, Any]) -> None:
    console.print(
        _kv_table(
            "STR Review Context",
            [
                ("Sample ID", sample["sample_id"]),
                ("Gold Conflict Label", f"{sample.get('conflict_category_id')} | {sample.get('conflict_type') or 'Unknown'}"),
                ("Gold answerable", str(sample.get("gold_answerable"))),
                ("STR applicable", str(sample.get("single_truth_applicable"))),
                ("Correct refusal", str(sample.get("correct_refusal"))),
            ],
        )
    )
    if sample.get("conflict_reason"):
        reason_text = Text(str(sample["conflict_reason"]), no_wrap=False, overflow="fold")
        console.print(Panel(reason_text, title="Gold Conflict Reason", border_style="green", padding=(0, 1)))
    if sample.get("gold_answer"):
        gold_answer_text = Text(str(sample["gold_answer"]), no_wrap=False, overflow="fold")
        console.print(Panel(gold_answer_text, title="Gold Answer", border_style="green", padding=(0, 1)))
    query_text = Text(sample.get("query", ""), no_wrap=False, overflow="fold")
    console.print(Panel(query_text, title="Query", border_style="cyan", padding=(0, 1)))
    response = sample.get("stripped_answer") or "(empty answer)"
    response_text = Text(response, no_wrap=False, overflow="fold")
    console.print(Panel(response_text, title="Model Final Answer", border_style="yellow", padding=(0, 1)))
    console.print(Panel(STR_GUIDE, title="STR Guide", border_style="magenta", padding=(0, 1)))


def render_claim(console: Console, claim: Dict[str, Any], idx: int, total: int, eligible_docs: List[Dict[str, Any]]) -> None:
    cited = ", ".join(claim.get("cited_docs") or []) or "none"
    console.print(Rule(f"[bold blue]Claim {idx}/{total}[/bold blue]", style="blue"))
    console.print(Panel(claim.get("text") or "", title=f"Cited docs: {cited}", border_style="blue", padding=(0, 1)))
    eligible = Table(title="FG-eligible doc pool for this claim review", box=box.SIMPLE, pad_edge=False, expand=False)
    eligible.add_column("doc_id", style="bold cyan", width=8)
    eligible.add_column("verdict", style="green", width=18)
    eligible.add_column("key fact", style="white")
    for doc in eligible_docs:
        eligible.add_row(doc.get("doc_id") or "", doc.get("verdict") or "", (doc.get("key_fact") or doc.get("snippet") or "")[:95])
    console.print(eligible)


def render_review_summary(console: Console, sample: Dict[str, Any], draft: Dict[str, Any]) -> None:
    console.print(Rule("[bold yellow]Review Summary[/bold yellow]", style="yellow"))
    rows = [
        ("Sample ID", sample["sample_id"]),
        ("Behavior complete", str(bool(draft.get("behavior")) or sample.get("correct_refusal"))),
        ("FG complete", str(bool(draft.get("fg")) or sample.get("correct_refusal"))),
        ("STR complete", str(bool(draft.get("str")) or not sample.get("single_truth_applicable"))),
        ("Notes", draft.get("meta", {}).get("notes", "")),
    ]
    console.print(_kv_table("Summary", rows))
    if draft.get("behavior"):
        behavior = draft["behavior"]
        console.print(
            _kv_table(
                "Behavior",
                [
                    ("Adherent", str(behavior.get("adherent"))),
                    ("Confidence", _confidence_text(behavior)),
                    ("Rationale", behavior.get("rationale") or ""),
                ],
            )
        )
    if draft.get("fg"):
        console.print(_kv_table("FG", [("Grounding ratio", str(draft["fg"].get("grounding_ratio"))), ("Claims reviewed", str(len(draft["fg"].get("claim_details") or [])))]))
    if draft.get("str"):
        str_judgment = draft["str"]
        console.print(
            _kv_table(
                "STR",
                [
                    ("Adherent", str(str_judgment.get("adherent"))),
                    ("Confidence", _confidence_text(str_judgment)),
                    ("Rationale", str_judgment.get("rationale") or ""),
                ],
            )
        )
