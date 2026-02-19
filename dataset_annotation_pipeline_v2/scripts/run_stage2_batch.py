#!/usr/bin/env python3
"""
scripts/run_stage2_batch.py
============================
Stage-2 batch runner using Anthropic's Message Batches API.

One request per query (reads per_doc_notes from Stage-1 output).

Usage:
    python scripts/run_stage2_batch.py \\
        --input  data/stage1_outputs/stage1_out.jsonl \\
        --output data/stage2_outputs/stage2_batch.jsonl
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
from src.parsers    import parse_stage2

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage2.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage2.txt"
STAGE2_MAX_TOKENS  = 350


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 batch annotation (Anthropic Batch API).")
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

    # ── Resolve batch ID ──
    batch_id = args.batch_id
    if batch_id is None and args.batch_id_file and Path(args.batch_id_file).exists():
        batch_id = Path(args.batch_id_file).read_text().strip() or None

    if batch_id is None:
        reqs = []
        for record in records:
            rec_id        = record.get("id", "")
            conflict_type = record.get("conflict_type", "UNKNOWN")
            per_doc_notes = record.get("per_doc_notes", [])
            user_prompt   = brace_safe_fill(user_template, {
                "QUERY":              record.get("query", ""),
                "CONFLICT_TYPE":      conflict_type,
                "PER_DOC_NOTES_JSON": json.dumps(per_doc_notes, ensure_ascii=False, indent=2),
            })
            reqs.append({"custom_id": sanitize_custom_id(rec_id), "system": system_prompt, "user": user_prompt})

        print(f"📤 Submitting Stage-2 batch | model={client.model} | records={len(reqs)}")
        batch_id = client.create_batch(reqs, max_tokens=STAGE2_MAX_TOKENS)
        print(f"   Batch ID: {batch_id}")

        if args.batch_id_file:
            Path(args.batch_id_file).write_text(batch_id)

    else:
        print(f"⏩ Resuming batch: {batch_id}")

    print("⏳ Waiting for batch…")
    batch_results = client.wait_batch(batch_id, poll_interval=args.poll_interval, verbose=True)

    result_map = {r["custom_id"]: r for r in batch_results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fout:
        for record in records:
            rec_id = record.get("id", "")
            res    = result_map.get(sanitize_custom_id(rec_id))
            if res and not res.get("error"):
                parsed, errors = parse_stage2(res["content"])
                record.update(parsed)
                if errors:
                    record["_stage2_errors"] = errors
            else:
                record.update({
                    "conflict_reason":           f"Batch error: {res.get('error') if res else 'missing'}",
                    "answerable_under_evidence":  False,
                    "_batch_error": True,
                })
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Stage-2 batch complete → {args.output}")


if __name__ == "__main__":
    main()