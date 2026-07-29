#!/usr/bin/env python3
"""Sanity-check trace-text SFT message files before launching a long job."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FORBIDDEN = re.compile(
    r"\b(?:gold_answer|gold answer|provided[_ ]gold_answer|refusal-required sample|"
    r"refusal required sample|sample because)\b|could not parse json|fallback:",
    flags=re.IGNORECASE,
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def assistant_text(row: dict) -> str:
    messages = row.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return row.get("assistant") or row.get("raw") or ""


def check_file(path: Path, *, require_think: bool) -> dict:
    rows = 0
    missing_assistant = []
    missing_think = []
    forbidden_think = []
    missing_sentinel = []
    forbidden_hits = []
    training_task_prefix_hits = []

    for line_no, row in iter_jsonl(path):
        rows += 1
        text = assistant_text(row)
        if not text:
            missing_assistant.append(line_no)
            continue
        low = text.lower()
        if require_think and ("<think>" not in low or "</think>" not in low):
            missing_think.append(line_no)
        if getattr(check_file, "forbid_think", False) and ("<think>" in low or "</think>" in low):
            forbidden_think.append(line_no)
        if "[[END-OF-ANSWER]]" not in text:
            missing_sentinel.append(line_no)
        if FORBIDDEN.search(text):
            forbidden_hits.append(line_no)
        messages = row.get("messages") or []
        for msg in messages:
            if msg.get("role") == "user" and str(msg.get("content") or "").startswith("Training subtask:"):
                training_task_prefix_hits.append(line_no)
                break

    return {
        "path": str(path),
        "rows": rows,
        "missing_assistant": missing_assistant[:10],
        "missing_think": missing_think[:10],
        "forbidden_think": forbidden_think[:10],
        "missing_sentinel": missing_sentinel[:10],
        "forbidden_hits": forbidden_hits[:10],
        "training_task_prefix_hits": training_task_prefix_hits[:10],
        "ok": bool(rows)
        and not missing_assistant
        and not missing_think
        and not forbidden_think
        and not missing_sentinel
        and not forbidden_hits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--require_think", "--require-think", action="store_true")
    ap.add_argument("--forbid_think", "--forbid-think", action="store_true")
    ap.add_argument("--forbid_task_prefix", "--forbid-task-prefix", action="store_true")
    args = ap.parse_args()

    ok = True
    check_file.forbid_think = args.forbid_think
    for path in args.jsonl:
        if not path.is_file():
            print(json.dumps({"path": str(path), "ok": False, "error": "missing_file"}))
            ok = False
            continue
        result = check_file(path, require_think=args.require_think)
        if args.forbid_task_prefix and result["training_task_prefix_hits"]:
            result["ok"] = False
        print(json.dumps(result, sort_keys=True))
        ok = ok and result["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
