#!/usr/bin/env python3
"""
scripts/run_monolithic_async.py
================================
Monolithic async annotation: ONE LLM call per query produces ALL annotation
(per-doc verdicts + conflict reasoning + expected response).

This is faster than 3-stage for large corpora (fewer round trips) and
useful for comparing annotation quality between strategies.

The monolithic prompt takes:
  • query
  • conflict_type (ground-truth label from the dataset)
  • retrieved_docs (all docs for the query)

And produces (text-mode output):
  • per_doc_notes  → Stage-1 equivalent
  • conflict_reason → Stage-2 equivalent
  • expected_response → Stage-3 equivalent

Output schema is identical to the 3-stage pipeline for full interoperability.

Usage:
    # Anthropic (default)
    python scripts/run_monolithic_async.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/monolithic_outputs/monolithic_out.jsonl \\
        --concurrency 8

    # OpenRouter / Qwen
    python scripts/run_monolithic_async.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/monolithic_outputs/monolithic_qwen.jsonl \\
        --provider openrouter --model qwen/qwen2.5-72b-instruct \\
        --concurrency 6
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
from src.parsers    import parse_monolithic

SYSTEM_PROMPT_PATH  = PROJECT_ROOT / "prompts" / "system_monolithic.txt"
USER_PROMPT_PATH    = PROJECT_ROOT / "prompts" / "user_monolithic.txt"
MONOLITHIC_MAX_TOKENS = 3000


# ─────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_user_prompt(template: str, record: Dict[str, Any]) -> str:
    """Substitute placeholders in the monolithic user template."""
    query         = record.get("query", "")
    conflict_type = record.get("conflict_type", "")
    # Strip source_url/timestamp for brevity; keep doc_id and snippet
    docs_slim = [
        {
            "doc_id":     d.get("doc_id", ""),
            "source_url": d.get("source_url", ""),
            "snippet":    d.get("snippet", ""),
            "timestamp":  d.get("timestamp", ""),
        }
        for d in record.get("retrieved_docs", [])
    ]
    return (
        template
        .replace("{query}",          query)
        .replace("{conflict_type}",  conflict_type)
        .replace("{retrieved_docs_json}", json.dumps(docs_slim, ensure_ascii=False, indent=2))
    )


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
        doc_ids      = [d.get("doc_id", "") for d in record.get("retrieved_docs", [])]
        user_prompt  = fill_user_prompt(user_template, record)

        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=MONOLITHIC_MAX_TOKENS,
                extra_headers={"X-Idempotency-Key": rec_id} if client.provider == Provider.ANTHROPIC else None,
            )
            parsed, errors = parse_monolithic(raw, expected_doc_ids=doc_ids)

            # Merge parsed annotations into record
            record["per_doc_notes"]             = parsed.get("per_doc_notes", [])
            record["conflict_reason"]           = parsed.get("conflict_reason", "")
            record["answerable_under_evidence"] = parsed.get("answerable_under_evidence", False)
            record["expected_response"]         = parsed.get("expected_response", {})
            record["think"]                     = parsed.get("think", "")
            record["_annotation_strategy"]      = "monolithic"
            if errors:
                record["_monolithic_errors"] = errors

        except Exception as exc:
            record["per_doc_notes"]             = []
            record["conflict_reason"]           = f"Exception: {exc}"
            record["answerable_under_evidence"] = False
            record["expected_response"]         = {
                "answer":        "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                "evidence":      [],
                "abstain":       True,
                "abstain_reason": str(exc),
            }
            record["think"]                = ""
            record["_annotation_strategy"] = "monolithic"
            record["_error"]               = str(exc)

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
        f"⚙️  Monolithic async | provider={args.provider} | model={client.model} | "
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
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Monolithic async")

    # Stats
    abstained, answered, errors_count = 0, 0, 0
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("_error"):
                    errors_count += 1
                elif rec.get("expected_response", {}).get("abstain"):
                    abstained += 1
                else:
                    answered += 1
            except Exception:
                pass

    print(f"\n✅ Monolithic async complete → {args.output}")
    print(f"   answered={answered}  abstained={abstained}  errors={errors_count}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Monolithic async annotation (all stages in 1 call).")
    ap.add_argument("--input",       required=True,  help="Input JSONL (normalized dataset)")
    ap.add_argument("--output",      required=True,  help="Output JSONL path")
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
