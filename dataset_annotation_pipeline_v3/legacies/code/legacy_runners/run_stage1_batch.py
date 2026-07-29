#!/usr/bin/env python3
"""
scripts/run_stage1_batch.py
============================
Stage-1 batch runner using Anthropic's Message Batches API.

Submits ALL (query, doc) pairs as a single batch, polls until completion,
then writes the annotated JSONL output.  Much cheaper than async for
large corpora because batch requests are billed at 50% of normal cost.

Batch mode is Anthropic-only. For OpenRouter/Qwen use the async script.

Usage:
    python scripts/run_stage1_batch.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/stage1_outputs/stage1_batch.jsonl

    # Resume an existing batch (if the script was interrupted):
    python scripts/run_stage1_batch.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/stage1_outputs/stage1_batch.jsonl \\
        --batch-id msgbatch_XXXXXXXXXXXXX

    # Custom model / tokens:
    python scripts/run_stage1_batch.py \\
        --input data/normalized/conflicts_normalized.jsonl \\
        --output data/stage1_outputs/stage1_batch.jsonl \\
        --model claude-haiku-4-5-20251001
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
from src.parsers    import parse_stage1

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage1.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage1.txt"
STAGE1_MAX_TOKENS  = 512

# Separator in custom_id: "{record_id}__{doc_id}"
# NOTE: Must only contain [a-zA-Z0-9_-] to satisfy Anthropic Batch API constraints.
_SEP = "__"


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


# ─────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_user_prompt(template: str, query: str, doc: Dict[str, Any]) -> str:
    return (
        template
        .replace("{QUERY}",     query)
        .replace("{DOC_ID}",    doc.get("doc_id", ""))
        .replace("{URL}",       doc.get("source_url", "") or "")
        .replace("{TEXT}",      doc.get("snippet",    "") or "")
        .replace("{TIMESTAMP}", doc.get("timestamp",  "") or "")
    )


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_batch_requests(
    records: List[Dict[str, Any]],
    system_prompt: str,
    user_template: str,
) -> List[Dict[str, Any]]:
    """Build the list of request dicts for LLMClient.create_batch()."""
    reqs = []
    for record in records:
        rec_id = record.get("id", "")
        query  = record.get("query", "")
        for doc in record.get("retrieved_docs", []):
            doc_id    = doc.get("doc_id", "")
            custom_id = sanitize_custom_id(f"{rec_id}{_SEP}{doc_id}")
            reqs.append({
                "custom_id": custom_id,
                "system":    system_prompt,
                "user":      fill_user_prompt(user_template, query, doc),
            })
    return reqs


def group_results(
    batch_results: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    system_prompt: str,
    user_template: str,
) -> List[Dict[str, Any]]:
    """
    Group batch results back by record ID, assigning per_doc_notes to each record.
    """
    # Map custom_id → content / error
    result_map: Dict[str, Dict] = {}
    for r in batch_results:
        result_map[r["custom_id"]] = r

    annotated = []
    for record in records:
        rec_id = record.get("id", "")
        per_doc_notes = []
        for doc in record.get("retrieved_docs", []):
            doc_id    = doc.get("doc_id", "")
            custom_id = sanitize_custom_id(f"{rec_id}{_SEP}{doc_id}")
            res       = result_map.get(custom_id)
            if res is None or res.get("error"):
                note = {
                    "doc_id":         doc_id,
                    "verdict":        "irrelevant",
                    "key_fact":       "",
                    "quote":          "",
                    "verdict_reason": f"Batch error: {res.get('error') if res else 'missing result'}",
                    "source_quality": "low",
                    "_batch_error":   True,
                }
            else:
                note, errors = parse_stage1(res["content"], fallback_doc_id=doc_id)
                if errors:
                    note["_validation_errors"] = errors
            per_doc_notes.append(note)

        record["per_doc_notes"] = per_doc_notes
        annotated.append(record)
    return annotated


# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1 batch annotation (Anthropic Batch API).")
    ap.add_argument("--input",      required=True,  help="Input JSONL (normalized dataset)")
    ap.add_argument("--output",     required=True,  help="Output JSONL path")
    ap.add_argument("--model",      default=None,   help="Anthropic model (default: claude-sonnet-4-6)")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--batch-id",   dest="batch_id", default=None,
                    help="Existing batch ID to resume polling (skip submission)")
    ap.add_argument("--batch-id-file", dest="batch_id_file", default=None,
                    help="File to save/load batch ID for resumption")
    ap.add_argument("--poll-interval", type=int, default=30,
                    help="Seconds between batch status checks (default: 30)")
    ap.add_argument("--system-prompt", dest="system_prompt", default=None)
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None)
    args = ap.parse_args()

    system_prompt = load_text(Path(args.system_prompt) if args.system_prompt else SYSTEM_PROMPT_PATH)
    user_template = load_text(Path(args.user_prompt)   if args.user_prompt   else USER_PROMPT_PATH)

    client = LLMClient(provider=Provider(args.provider), model=args.model or None)

    records = load_records(args.input)
    if args.limit:
        records = records[:args.limit]

    # ── Determine batch ID ──
    batch_id = args.batch_id
    if batch_id is None and args.batch_id_file and Path(args.batch_id_file).exists():
        batch_id = Path(args.batch_id_file).read_text().strip()
        if batch_id:
            print(f"⏩ Resuming batch from file: {batch_id}")

    if batch_id is None:
        print(f"📤 Submitting Stage-1 batch | model={client.model} | records={len(records)}")
        reqs = build_batch_requests(records, system_prompt, user_template)
        print(f"   Total (query, doc) pairs: {len(reqs)}")
        batch_id = client.create_batch(reqs, max_tokens=STAGE1_MAX_TOKENS)
        print(f"   Batch submitted: {batch_id}")

        if args.batch_id_file:
            Path(args.batch_id_file).write_text(batch_id)
            print(f"   Batch ID saved to: {args.batch_id_file}")
    else:
        print(f"⏩ Resuming batch: {batch_id}")

    # ── Poll & fetch results ──
    print("⏳ Waiting for batch to complete…")
    batch_results = client.wait_batch(batch_id, poll_interval=args.poll_interval, verbose=True)
    print(f"   Received {len(batch_results)} result(s)")

    # ── Group back into records ──
    annotated = group_results(batch_results, records, system_prompt, user_template)

    # ── Write output ──
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in annotated:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Stage-1 batch complete → {args.output}")
    n_errors = sum(
        1 for rec in annotated
        for note in rec.get("per_doc_notes", [])
        if note.get("_batch_error") or note.get("_parse_error")
    )
    if n_errors:
        print(f"   ⚠️  {n_errors} doc-level errors (see _batch_error / _parse_error fields)")


if __name__ == "__main__":
    main()