#!/usr/bin/env python3
"""
scripts/run_stage2_async.py
============================
Async Stage-2 runner: conflict-level macro reasoning.

Takes Stage-1 output JSONL (with per_doc_notes) and produces a conflict
reason + answerability judgment for each query.

Usage:
    python scripts/run_stage2_async.py \\
        --input  data/stage1_outputs/stage1_out.jsonl \\
        --output data/stage2_outputs/stage2_out.jsonl \\
        --concurrency 12

    # OpenRouter / Qwen
    python scripts/run_stage2_async.py \\
        --input  data/stage1_outputs/stage1_out.jsonl \\
        --output data/stage2_outputs/stage2_qwen.jsonl \\
        --provider openrouter --model qwen/qwen2.5-72b-instruct
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm.asyncio import tqdm_asyncio

THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient, Provider
from src.parsers    import parse_stage2

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage2.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage2.txt"
STAGE2_MAX_TOKENS  = 350


# ─────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def brace_safe_fill(template: str, mapping: Dict[str, str]) -> str:
    """Fill {KEY} placeholders without breaking other braces."""
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


def load_processed_ids(output_path: str) -> set:
    done = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line).get("id"))
            except Exception:
                pass
    return done


# ─────────────────────────────────────────────

async def process_record(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_template: str,
    record: Dict[str, Any],
    out_lock: asyncio.Lock,
    output_path: str,
) -> None:
    async with semaphore:
        rec_id       = record.get("id", "")
        query        = record.get("query", "")
        conflict_type = record.get("conflict_type", "UNKNOWN")
        per_doc_notes = record.get("per_doc_notes", [])

        user_prompt = brace_safe_fill(user_template, {
            "QUERY":              query,
            "CONFLICT_TYPE":      conflict_type,
            "PER_DOC_NOTES_JSON": json.dumps(per_doc_notes, ensure_ascii=False, indent=2),
        })

        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=STAGE2_MAX_TOKENS,
                extra_headers={"X-Idempotency-Key": rec_id} if client.provider == Provider.ANTHROPIC else None,
            )
            parsed, errors = parse_stage2(raw)
            record.update(parsed)
            if errors:
                record["_stage2_errors"] = errors
        except Exception as exc:
            record.update({
                "conflict_reason":          f"Exception: {exc}",
                "answerable_under_evidence": False,
                "_error": str(exc),
            })

        async with out_lock:
            with open(output_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    system_prompt = load_text(Path(args.system_prompt) if args.system_prompt else SYSTEM_PROMPT_PATH)
    user_template = load_text(Path(args.user_prompt)   if args.user_prompt   else USER_PROMPT_PATH)

    client = LLMClient(
        provider    = Provider(args.provider),
        model       = args.model or None,
        temperature = args.temperature,
        max_retries = args.max_retries,
    )

    records  = load_records(args.input)
    done_ids = load_processed_ids(args.output)
    records  = [r for r in records if r.get("id") not in done_ids]

    if args.limit:
        records = records[:args.limit]

    if not records:
        print("✅ Nothing to process.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"⚙️  Stage-2 async | provider={args.provider} | model={client.model} | "
        f"records={len(records)} | concurrency={args.concurrency}"
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock  = asyncio.Lock()

    tasks = [
        process_record(client, semaphore, system_prompt, user_template, rec, out_lock, args.output)
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Stage-2 async")
    print(f"\n✅ Stage-2 complete → {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Async Stage-2 conflict macro reasoning.")
    ap.add_argument("--input",       required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "openrouter"])
    ap.add_argument("--model",       default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int,   default=12)
    ap.add_argument("--limit",       type=int,   default=None)
    ap.add_argument("--max-retries", type=int,   default=3)
    ap.add_argument("--system-prompt", dest="system_prompt", default=None)
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
