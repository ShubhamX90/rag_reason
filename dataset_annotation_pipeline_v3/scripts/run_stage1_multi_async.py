#!/usr/bin/env python3
"""
scripts/run_stage1_multi_async.py
==================================
Multi-LLM Stage-1: per-document evidence adjudication with weighted
majority voting across the annotation committee.

Every (query, doc) pair is sent to ALL committee models concurrently.
`verdict` is decided by weighted majority vote; the complete text fields
(key_fact, quote, verdict_reason, source_quality) are adopted from the
highest-weight model that voted for the winning verdict.

All models are accessed via OpenRouter (OPENROUTER_API_KEY required).

Concurrency
-----------
Each (model × doc) pair is one API call.  With 5 committee models and
N docs per query, one query spawns 5×N calls.  The --concurrency flag
caps *total* simultaneous in-flight calls across all models and docs.
Default 25 ≈ one record (5 docs × 5 models) processed in parallel.

Usage:
    # Conflicts dataset
    python scripts/run_stage1_multi_async.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/stage1_outputs/stage1_multi.jsonl \\
        --concurrency 25

    # Refusals dataset
    python scripts/run_stage1_multi_async.py \\
        --input  data/normalized/refusals_normalized.jsonl \\
        --output data/stage1_outputs/refusals_stage1_multi.jsonl \\
        --concurrency 25
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
from src.parsers    import parse_stage1
from src.voting     import COMMITTEE_MODELS, MODEL_WEIGHTS, merge_stage1_votes
from src.cost_tracker import CostTracker, default_cost_report_path

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage1.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage1.txt"
STAGE1_MAX_TOKENS  = 512


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_user_prompt(template: str, query: str, doc: Dict[str, Any]) -> str:
    """Substitute stage-1 user template placeholders."""
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
    done: set = set()
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
#  Per-model single call
# ─────────────────────────────────────────────

async def call_one_model(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    doc_id: str,
    tracker: CostTracker,
) -> Dict[str, Any]:
    """One API call: one committee model × one (query, doc) pair."""
    async with semaphore:
        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=STAGE1_MAX_TOKENS,
                cost_tracker=tracker,
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
                "verdict_reason": f"API error: {str(exc)[:120]}",
                "source_quality": "low",
                "_error":         str(exc),
            }


# ─────────────────────────────────────────────
#  Committee adjudication for one doc
# ─────────────────────────────────────────────

async def adjudicate_doc_committee(
    clients: Dict[str, LLMClient],
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_template: str,
    query: str,
    doc: Dict[str, Any],
    tracker: CostTracker,
) -> Dict[str, Any]:
    """
    Run ALL committee models concurrently on one (query, doc) pair.
    Each model call independently acquires a semaphore slot.
    Returns the consensus note after weighted majority vote on verdict.
    """
    doc_id      = doc.get("doc_id", "")
    user_prompt = fill_user_prompt(user_template, query, doc)

    coros = [
        call_one_model(clients[model], semaphore, system_prompt, user_prompt, doc_id, tracker)
        for model in COMMITTEE_MODELS
    ]
    raw_results = await asyncio.gather(*coros)
    model_notes = {model: raw_results[i] for i, model in enumerate(COMMITTEE_MODELS)}

    return merge_stage1_votes(model_notes, fallback_doc_id=doc_id)


# ─────────────────────────────────────────────
#  Per-record processing
# ─────────────────────────────────────────────

async def process_record(
    clients: Dict[str, LLMClient],
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_template: str,
    record: Dict[str, Any],
    out_lock: asyncio.Lock,
    output_path: str,
    tracker: CostTracker,
) -> None:
    """Adjudicate all docs for one query record via the committee."""
    query = record.get("query", "")
    docs  = record.get("retrieved_docs", [])

    per_doc_notes = await asyncio.gather(*[
        adjudicate_doc_committee(
            clients, semaphore, system_prompt, user_template, query, doc, tracker
        )
        for doc in docs
    ])
    record["per_doc_notes"] = list(per_doc_notes)

    async with out_lock:
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    system_path = Path(args.system_prompt) if args.system_prompt else SYSTEM_PROMPT_PATH
    user_path   = Path(args.user_prompt)   if args.user_prompt   else USER_PROMPT_PATH

    system_prompt = load_text(system_path)
    user_template = load_text(user_path)

    # One LLMClient per committee model — all share the same OpenRouter key
    clients = {
        model: LLMClient(
            provider    = Provider.OPENROUTER,
            model       = model,
            temperature = args.temperature,
            max_retries = args.max_retries,
        )
        for model in COMMITTEE_MODELS
    }

    records  = load_records(args.input)
    done_ids = load_processed_ids(args.output)
    records  = [r for r in records if r.get("id") not in done_ids]

    if args.limit:
        records = records[:args.limit]

    if not records:
        print("✅ Nothing to process (all records already in output).")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"⚙️  Stage-1 multi-LLM | committee={len(COMMITTEE_MODELS)} models | "
        f"records={len(records)} | concurrency={args.concurrency}"
    )
    for model, weight in MODEL_WEIGHTS.items():
        print(f"   {weight:.0%}  {model}")
    if done_ids:
        print(f"⏩ Resuming: {len(done_ids)} already processed")

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock  = asyncio.Lock()
    tracker   = CostTracker(stage="stage1")

    tasks = [
        process_record(
            clients, semaphore, system_prompt, user_template, rec, out_lock, args.output, tracker
        )
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Stage-1 multi-LLM")
    print(f"\n✅ Stage-1 multi-LLM complete → {args.output}")

    # ── Fetch exact costs from OpenRouter and print breakdown ─────────────
    report_path = args.cost_report or default_cost_report_path(args.output)
    await tracker.fetch_and_report(save_json_path=report_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Multi-LLM Stage-1 evidence adjudication with weighted majority voting.\n"
            "All committee models are accessed via OpenRouter."
        )
    )
    ap.add_argument("--input",       required=True,  help="Input JSONL (normalized conflicts or refusals dataset)")
    ap.add_argument("--output",      required=True,  help="Output JSONL path")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int,   default=25,
                    help="Total concurrent API calls across ALL committee models (default: 25)")
    ap.add_argument("--limit",       type=int,   default=None, help="Max records to process")
    ap.add_argument("--max-retries", type=int,   default=3,    help="Retries per failed call")
    ap.add_argument("--system-prompt", dest="system_prompt", default=None,
                    help="Override system prompt path (default: prompts/system_stage1.txt)")
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None,
                    help="Override user prompt path (default: prompts/user_stage1.txt)")
    ap.add_argument("--cost-report",   dest="cost_report",   default=None,
                    help="Path to save cost report JSON (default: <output>_cost_report.json)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
