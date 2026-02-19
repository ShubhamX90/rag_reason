#!/usr/bin/env python3
"""
scripts/run_stage3_async.py
============================
Async Stage-3 runner: grounded expected-response generation.

Takes Stage-2 output JSONL and generates the final expected response
(with <think> trace, citations, abstain judgement) for each query.

Usage:
    python scripts/run_stage3_async.py \\
        --input  data/stage2_outputs/stage2_out.jsonl \\
        --output data/stage3_outputs/stage3_out.jsonl \\
        --concurrency 8

    # OpenRouter / Qwen
    python scripts/run_stage3_async.py \\
        --input  data/stage2_outputs/stage2_out.jsonl \\
        --output data/stage3_outputs/stage3_qwen.jsonl \\
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
from src.parsers    import parse_stage3

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage3.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage3.txt"
STAGE3_MAX_TOKENS  = 2500


# ─────────────────────────────────────────────

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


def load_processed_ids(output_path: str) -> set:
    done = set()
    if not Path(output_path).exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
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
        rec_id = record.get("id", "")

        user_prompt = brace_safe_fill(user_template, {
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

        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=STAGE3_MAX_TOKENS,
                extra_headers={"X-Idempotency-Key": rec_id} if client.provider == Provider.ANTHROPIC else None,
            )
            parsed, errors = parse_stage3(raw)
            record["expected_response"] = parsed.get("expected_response", {})
            record["think"]             = parsed.get("think", "")
            if errors:
                record["_stage3_errors"] = errors
        except Exception as exc:
            record["expected_response"] = {
                "answer":        "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                "evidence":      [],
                "abstain":       True,
                "abstain_reason": str(exc),
            }
            record["think"]  = ""
            record["_error"] = str(exc)

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
    stats = {"answered": 0, "abstained": 0, "failed": 0}

    print(
        f"⚙️  Stage-3 async | provider={args.provider} | model={client.model} | "
        f"records={len(records)} | concurrency={args.concurrency}"
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock  = asyncio.Lock()

    tasks = [
        process_record(client, semaphore, system_prompt, user_template, rec, out_lock, args.output)
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Stage-3 async")

    # Quick stats scan
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                er = json.loads(line).get("expected_response", {})
                if er.get("abstain"):
                    stats["abstained"] += 1
                elif er.get("_error"):
                    stats["failed"] += 1
                else:
                    stats["answered"] += 1
            except Exception:
                pass

    print(f"\n✅ Stage-3 complete → {args.output}")
    print(f"   {stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Async Stage-3 expected-response synthesis.")
    ap.add_argument("--input",       required=True)
    ap.add_argument("--output",      required=True)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "openrouter"])
    ap.add_argument("--model",       default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int,   default=8)
    ap.add_argument("--limit",       type=int,   default=None)
    ap.add_argument("--max-retries", type=int,   default=3)
    ap.add_argument("--system-prompt", dest="system_prompt", default=None)
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
