#!/usr/bin/env python3
"""Build targeted retry messages for abstaining generations.

This script is intentionally inference-only: it uses the model's previous draft
to decide which rows deserve a correction pass, without consulting gold labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ABSTAIN_ANSWER = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
DOC_VERDICT_RE = re.compile(
    r"^\s*-\s*(d\d+)\s*:\s*(supports|partially supports|irrelevant)\b",
    re.IGNORECASE | re.MULTILINE,
)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except Exception as exc:  # pragma: no cover - defensive CLI guard
                raise ValueError(f"{path}:{line_no} bad json: {exc}") from exc


def append_retry_note(
    original_user: str,
    support_docs: List[str],
    partial_docs: List[str],
    mode: str,
) -> str:
    lines = [original_user.rstrip(), "", "Retry note:"]
    if support_docs:
        lines.append(
            "- In the previous draft, you identified direct support in docs: "
            + ", ".join(support_docs)
            + "."
        )
        lines.append("- Because direct support exists, the query is answerable under the retrieved evidence.")
    if partial_docs:
        lines.append(
            "- You also identified partial support in docs: "
            + ", ".join(partial_docs)
            + "."
        )
    if mode == "support":
        lines.append("- Regenerate the full response and do not output the abstention phrase.")
    else:
        lines.append(
            "- Regenerate the full response by synthesizing the supporting and partial evidence. "
            "Do not abstain unless a necessary information gap still remains after integrating them."
        )
    lines.append(
        "- Keep the same output structure: one <think>...</think> block, then the final answer with citations, then [[END-OF-ANSWER]]."
    )
    return "\n".join(lines)


def parse_trace_verdicts(raw: str) -> Tuple[List[str], List[str], List[str]]:
    match = THINK_BLOCK_RE.search(raw or "")
    if not match:
        return [], [], []
    support_docs: List[str] = []
    partial_docs: List[str] = []
    irrelevant_docs: List[str] = []
    for doc_id, verdict in DOC_VERDICT_RE.findall(match.group(1)):
        verdict = verdict.lower()
        if verdict == "supports":
            support_docs.append(doc_id)
        elif verdict == "partially supports":
            partial_docs.append(doc_id)
        elif verdict == "irrelevant":
            irrelevant_docs.append(doc_id)
    return support_docs, partial_docs, irrelevant_docs


def select_retry_rows(
    source_messages: Dict[str, dict],
    prior_outputs: Iterable[dict],
    mode: str,
    partial_min_count: int,
    partial_max_irrelevant: int,
) -> Tuple[List[dict], dict]:
    selected: List[dict] = []
    stats = {
        "seen": 0,
        "abstaining": 0,
        "selected": 0,
        "selected_support": 0,
        "selected_partial_only": 0,
        "missing_source_message": 0,
    }

    for rec in prior_outputs:
        stats["seen"] += 1
        raw = rec.get("raw", "")
        if ABSTAIN_ANSWER not in raw:
            continue
        stats["abstaining"] += 1

        support_docs, partial_docs, irrelevant_docs = parse_trace_verdicts(raw)
        take = False
        if mode == "support":
            take = bool(support_docs)
        elif mode == "support_or_partial":
            take = bool(support_docs) or (
                len(partial_docs) >= partial_min_count
                and len(irrelevant_docs) <= partial_max_irrelevant
            )
        else:  # pragma: no cover - argparse constrains this
            raise ValueError(f"Unsupported mode: {mode}")

        if not take:
            continue

        cid = rec.get("id")
        base = source_messages.get(cid)
        if base is None:
            stats["missing_source_message"] += 1
            continue

        new_rec = copy.deepcopy(base)
        messages = new_rec.get("messages") or []
        user_found = False
        for msg in messages:
            if msg.get("role") == "user":
                msg["content"] = append_retry_note(
                    str(msg.get("content", "")),
                    support_docs=support_docs,
                    partial_docs=partial_docs,
                    mode=mode,
                )
                user_found = True
                break
        if not user_found:
            raise ValueError(f"{cid}: retry source message is missing a user turn")

        new_rec["retry_metadata"] = {
            "retry_mode": mode,
            "source_id": cid,
            "support_doc_ids": support_docs,
            "partial_doc_ids": partial_docs,
            "irrelevant_doc_ids": irrelevant_docs,
            "source_abstained": True,
        }
        selected.append(new_rec)
        stats["selected"] += 1
        if support_docs:
            stats["selected_support"] += 1
        elif partial_docs:
            stats["selected_partial_only"] += 1

    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_messages", required=True, type=Path)
    parser.add_argument("--prior_outputs", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=["support", "support_or_partial"],
        default="support",
        help="Which abstaining rows should receive a correction pass.",
    )
    parser.add_argument(
        "--partial_min_count",
        type=int,
        default=5,
        help="When mode=support_or_partial, retry partial-only rows with at least this many partial docs.",
    )
    parser.add_argument(
        "--partial_max_irrelevant",
        type=int,
        default=99,
        help="When mode=support_or_partial, retry partial-only rows with at most this many irrelevant docs.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    source_messages = {rec["id"]: rec for rec in read_jsonl(args.source_messages)}
    selected, stats = select_retry_rows(
        source_messages=source_messages,
        prior_outputs=read_jsonl(args.prior_outputs),
        mode=args.mode,
        partial_min_count=args.partial_min_count,
        partial_max_irrelevant=args.partial_max_irrelevant,
    )

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(selected)
    if args.limit > 0:
        selected = selected[: args.limit]
        stats["selected_after_limit"] = len(selected)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for rec in selected:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source_messages": str(args.source_messages),
                "prior_outputs": str(args.prior_outputs),
                "output_jsonl": str(args.output_jsonl),
                "mode": args.mode,
                "partial_min_count": args.partial_min_count,
                "partial_max_irrelevant": args.partial_max_irrelevant,
                "written": len(selected),
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
