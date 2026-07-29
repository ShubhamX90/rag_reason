#!/usr/bin/env python3
"""
scripts/run_stage1_async.py
============================
Async Stage-1 runner: per-document evidence adjudication.

For each query in the input JSONL, each retrieved document is adjudicated
concurrently, producing a per-doc note (verdict, key_fact, quote, etc.).

Supports:
  - Provider: anthropic (claude-sonnet-4-6) or openrouter (qwen 2.5)
  - Resume: skips already-processed query IDs
  - Rate-limiting: configurable concurrency semaphore

Usage:
    # Anthropic (default)
    python scripts/run_stage1_async.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/stage1_outputs/stage1_out.jsonl \\
        --concurrency 10

    # OpenRouter / Qwen
    python scripts/run_stage1_async.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/stage1_outputs/stage1_qwen.jsonl \\
        --provider openrouter \\
        --model  qwen/qwen2.5-72b-instruct \\
        --concurrency 8
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm_asyncio

# ── project imports ──
THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient, Provider
from src.parsers    import parse_stage1

# ── defaults ──
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage1.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage1.txt"
STAGE1_MAX_TOKENS  = 512


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_user_prompt(template: str, query: str, doc: Dict[str, Any]) -> str:
    """Safe string substitution for stage-1 user template."""
    return (
        template
        .replace("{QUERY}",     query)
        .replace("{DOC_ID}",    doc.get("doc_id", ""))
        .replace("{URL}",       doc.get("source_url", "") or "")
        .replace("{TEXT}",      doc.get("snippet", "") or "")
        .replace("{TIMESTAMP}", doc.get("timestamp", "") or "")
    )


def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping invalid JSONL line: {e}")
    return records


def load_processed_ids(output_path: str) -> set:
    done = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec_id = json.loads(line).get("id")
                if rec_id:
                    done.add(rec_id)
            except Exception:
                pass
    return done


# ─────────────────────────────────────────────
#  Async adjudication
# ─────────────────────────────────────────────

async def adjudicate_doc(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_template: str,
    query: str,
    doc: Dict[str, Any],
    record_id: str,
) -> Dict[str, Any]:
    """Adjudicate one document against the query."""
    async with semaphore:
        doc_id      = doc.get("doc_id", "")
        user_prompt = fill_user_prompt(user_template, query, doc)
        idempotency = f"{record_id}:{doc_id}"

        extra_headers = None
        if client.provider == Provider.ANTHROPIC:
            extra_headers = {"X-Idempotency-Key": idempotency}

        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=STAGE1_MAX_TOKENS,
                extra_headers=extra_headers,
            )
            note, errors = parse_stage1(raw, fallback_doc_id=doc_id)
            if errors:
                note["_validation_errors"] = errors
            return note

        except Exception as exc:
            return {
                "doc_id":         doc_id,
                "verdict":        "irrelevant",
                "key_fact":       "",
                "quote":          "",
                "verdict_reason": f"Exception during adjudication: {exc}"[:120],
                "source_quality": "low",
                "_error":         str(exc),
            }


async def process_record(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_template: str,
    record: Dict[str, Any],
    out_lock: asyncio.Lock,
    output_path: str,
) -> None:
    """Adjudicate all docs for one query record and write output."""
    record_id = record.get("id", "")
    query     = record.get("query", "")
    docs      = record.get("retrieved_docs", [])

    per_doc_notes = await asyncio.gather(
        *[
            adjudicate_doc(
                client, semaphore, system_prompt, user_template, query, doc, record_id
            )
            for doc in docs
        ]
    )
    record["per_doc_notes"] = list(per_doc_notes)

    async with out_lock:
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    system_path = Path(args.system_prompt) if args.system_prompt else SYSTEM_PROMPT_PATH
    user_path   = Path(args.user_prompt)   if args.user_prompt   else USER_PROMPT_PATH

    system_prompt = load_text(system_path)
    user_template = load_text(user_path)

    client = LLMClient(
        provider    = Provider(args.provider),
        model       = args.model or None,
        temperature = args.temperature,
        max_retries = args.max_retries,
    )

    records     = load_records(args.input)
    done_ids    = load_processed_ids(args.output)
    records     = [r for r in records if r.get("id") not in done_ids]

    if args.limit:
        records = records[:args.limit]

    if not records:
        print("✅ Nothing to process (all records already in output).")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"⚙️  Stage-1 async | provider={args.provider} | model={client.model} | "
        f"records={len(records)} | concurrency={args.concurrency}"
    )
    if done_ids:
        print(f"⏩ Resuming: {len(done_ids)} already processed")

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock  = asyncio.Lock()

    tasks = [
        process_record(client, semaphore, system_prompt, user_template, rec, out_lock, args.output)
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Stage-1 async")

    print(f"\n✅ Stage-1 complete → {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Async Stage-1 evidence adjudication.")
    ap.add_argument("--input",       required=True,  help="Input JSONL (normalized dataset)")
    ap.add_argument("--output",      required=True,  help="Output JSONL path")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "openrouter"],
                    help="LLM provider (default: anthropic)")
    ap.add_argument("--model",       default=None,   help="Model name (provider-specific default if omitted)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int,  default=10, help="Max concurrent API calls")
    ap.add_argument("--limit",       type=int,  default=None, help="Max records to process")
    ap.add_argument("--max-retries", type=int,  default=3,  help="Retries per failed call")
    ap.add_argument("--system-prompt", dest="system_prompt", default=None,
                    help="Override system prompt path")
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None,
                    help="Override user prompt path")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
