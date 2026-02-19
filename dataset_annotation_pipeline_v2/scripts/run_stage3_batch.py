#!/usr/bin/env python3
"""
scripts/run_stage3_batch.py
============================
Stage-3 batch runner using Anthropic's Message Batches API.

One request per query (reads everything from Stage-2 output).

Usage:
    python scripts/run_stage3_batch.py \\
        --input  data/stage2_outputs/stage2_out.jsonl \\
        --output data/stage3_outputs/stage3_batch.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient, Provider
from src.parsers    import parse_stage3

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage3.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage3.txt"
STAGE3_MAX_TOKENS  = 2500


import re as _re

def sanitize_custom_id(s: str) -> str:
    """
    Ensure the string satisfies Anthropic's custom_id constraint:
      ^[a-zA-Z0-9_-]{1,64}$
    Any disallowed character is replaced with '_', then truncated to 64 chars.
    An empty result falls back to 'id'.
    """
    sanitized = _re.sub(r"[^a-zA-Z0-9_\-]", "_", s)[:64]
    return sanitized or "id"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def brace_safe_fill(template: str, mapping: Dict[str, str]) -> str:
    temp = template
    for k in mapping:
        temp = temp.replace("{" + k + "}", f"@@{k}@@")
    temp = temp.replace("{", "{{").replace("}", "}}")
    for k, v in mapping.items():
        temp = temp.replace(f"@@{k}@@", v or "")
    return temp


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_user_prompt(template: str, record: Dict[str, Any]) -> str:
    return brace_safe_fill(template, {
        "query":                    record.get("query", ""),
        "retrieved_docs":           json.dumps(record.get("retrieved_docs", []),
                                               ensure_ascii=False, indent=2),
        "per_doc_notes":            json.dumps(record.get("per_doc_notes", []),
                                               ensure_ascii=False, indent=2),
        "conflict_type":            record.get("conflict_type", ""),
        "conflict_reason":          record.get("conflict_reason", ""),
        "answerable_under_evidence": str(record.get("answerable_under_evidence", True)).lower(),
        "gold_answer":              record.get("gold_answer", "") or "",
        "ranked_doc_ids":           ", ".join(
            n.get("doc_id", "") for n in record.get("per_doc_notes", [])
            if n.get("verdict") != "irrelevant"
        ),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-3 batch annotation (Anthropic Batch API).")
    ap.add_argument("--input",         required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--model",         default=None)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--limit",         type=int, default=None)
    ap.add_argument("--batch-id",      dest="batch_id",      default=None)
    ap.add_argument("--batch-id-file", dest="batch_id_file", default=None)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--system-prompt", dest="system_prompt", default=None)
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None)
    args = ap.parse_args()

    system_prompt = load_text(Path(args.system_prompt) if args.system_prompt else SYSTEM_PROMPT_PATH)
    user_template = load_text(Path(args.user_prompt)   if args.user_prompt   else USER_PROMPT_PATH)

    client = LLMClient(provider=Provider(args.provider), model=args.model or None)
    records = load_records(args.input)
    if args.limit:
        records = records[:args.limit]

    batch_id = args.batch_id
    if batch_id is None and args.batch_id_file and Path(args.batch_id_file).exists():
        batch_id = Path(args.batch_id_file).read_text().strip() or None

    if batch_id is None:
        reqs = [
            {
                "custom_id": sanitize_custom_id(record.get("id", f"rec_{i}")),
                "system":    system_prompt,
                "user":      build_user_prompt(user_template, record),
            }
            for i, record in enumerate(records)
        ]
        print(f"📤 Submitting Stage-3 batch | model={client.model} | records={len(reqs)}")
        batch_id = client.create_batch(reqs, max_tokens=STAGE3_MAX_TOKENS)
        print(f"   Batch ID: {batch_id}")
        if args.batch_id_file:
            Path(args.batch_id_file).write_text(batch_id)
    else:
        print(f"⏩ Resuming batch: {batch_id}")

    print("⏳ Waiting for batch…")
    batch_results = client.wait_batch(batch_id, poll_interval=args.poll_interval, verbose=True)

    result_map = {r["custom_id"]: r for r in batch_results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats = {"answered": 0, "abstained": 0, "failed": 0}

    with open(args.output, "w", encoding="utf-8") as fout:
        for record in records:
            rec_id = record.get("id", "")
            res    = result_map.get(sanitize_custom_id(rec_id))
            if res and not res.get("error"):
                parsed, errors = parse_stage3(res["content"])
                record["expected_response"] = parsed.get("expected_response", {})
                record["think"]             = parsed.get("think", "")
                if errors:
                    record["_stage3_errors"] = errors
                if record["expected_response"].get("abstain"):
                    stats["abstained"] += 1
                else:
                    stats["answered"] += 1
            else:
                record["expected_response"] = {
                    "answer":        "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                    "evidence":      [],
                    "abstain":       True,
                    "abstain_reason": f"Batch error: {res.get('error') if res else 'missing'}",
                }
                record["think"]       = ""
                record["_batch_error"] = True
                stats["failed"] += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Stage-3 batch complete → {args.output}")
    print(f"   {stats}")


if __name__ == "__main__":
    main()