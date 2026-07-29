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

By default, models are accessed via OpenRouter (OPENROUTER_API_KEY required).
Use --committee-backend local_openai with a JSON committee config for local
OpenAI-compatible endpoints.

Concurrency
-----------
Each (model × doc) pair is one API call. With the current 4-model committee and
N docs per query, one query spawns 4×N calls. The --concurrency flag
caps *total* simultaneous in-flight calls across all models and docs.
Default 25 is enough to keep multiple docs in flight for one record.

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

from src.llm_client import LLMClient
from src.llm_cache import CacheMissError
from src.committee_config import (
    build_clients_for_committee,
    cache_mode_for_backend,
    client_max_tokens,
    configure_committee_for_backend,
    describe_committee,
    max_concurrency_for_backend,
)
from src.parsers    import parse_stage1
from src.voting     import COMMITTEE_MODELS, MODEL_WEIGHTS, merge_stage1_votes
from src.cost_tracker import (
    CostTracker,
    default_cost_ledger_path,
    default_cost_report_path,
    default_cumulative_cost_report_path,
)

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
    cache_enabled: bool,
    cache_mode: str,
) -> Dict[str, Any] | None:
    """One API call: one committee model × one (query, doc) pair."""
    async with semaphore:
        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=client_max_tokens(client, STAGE1_MAX_TOKENS),
                cost_tracker=tracker,
                cache_enabled=cache_enabled,
                cache_mode=cache_mode,
                cache_namespace="stage1_multi",
            )
            note, errors = parse_stage1(raw, fallback_doc_id=doc_id)
            if errors:
                note["_validation_errors"] = errors
            return note
        except CacheMissError:
            raise
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
    cache_enabled: bool,
    cache_mode: str,
) -> Dict[str, Any]:
    """
    Run ALL committee models concurrently on one (query, doc) pair.
    Each model call independently acquires a semaphore slot.
    Returns the consensus note after weighted majority vote on verdict.
    """
    doc_id      = doc.get("doc_id", "")
    user_prompt = fill_user_prompt(user_template, query, doc)

    coros = [
        call_one_model(
            clients[model], semaphore, system_prompt, user_prompt, doc_id,
            tracker, cache_enabled, cache_mode
        )
        for model in COMMITTEE_MODELS
    ]
    raw_results = await asyncio.gather(*coros)
    model_notes = {model: raw_results[i] for i, model in enumerate(COMMITTEE_MODELS)}

    return merge_stage1_votes(
        model_notes,
        fallback_doc_id=doc_id,
        fallback_source_url=str(doc.get("source_url", "")),
    )


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
    cache_enabled: bool,
    cache_mode: str,
) -> None:
    """Adjudicate all docs for one query record via the committee."""
    query = record.get("query", "")
    docs  = record.get("retrieved_docs", [])

    per_doc_notes = await asyncio.gather(*[
        adjudicate_doc_committee(
            clients, semaphore, system_prompt, user_template, query, doc,
            tracker, cache_enabled, cache_mode
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

    committee_cfg = configure_committee_for_backend(
        backend=args.committee_backend,
        config_path=args.committee_config,
        cache_mode_override=args.cache_mode,
        cache_dir_override=args.cache_dir,
    )
    args.concurrency = max_concurrency_for_backend(args.concurrency, committee_cfg)
    resolved_cache_mode = cache_mode_for_backend(
        backend=args.committee_backend,
        committee_config=committee_cfg,
        use_cache=args.use_cache,
        cache_mode_override=args.cache_mode,
    )
    clients = build_clients_for_committee(
        backend=args.committee_backend,
        committee_config=committee_cfg,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )

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
        f"⚙️  Stage-1 multi-LLM | backend={args.committee_backend} | "
        f"committee={len(COMMITTEE_MODELS)} models | "
        f"records={len(records)} | concurrency={args.concurrency}"
    )
    for line in describe_committee(backend=args.committee_backend, committee_config=committee_cfg):
        print(line)
    if done_ids:
        print(f"⏩ Resuming: {len(done_ids)} already processed")
    print(f"   cache_mode={resolved_cache_mode}")

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock  = asyncio.Lock()
    tracker   = CostTracker(stage="stage1")

    tasks = [
        process_record(
            clients, semaphore, system_prompt, user_template,
            rec, out_lock, args.output, tracker,
            resolved_cache_mode != "off",
            resolved_cache_mode,
        )
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Stage-1 multi-LLM")
    print(f"\n✅ Stage-1 multi-LLM complete → {args.output}")

    # ── Fetch exact costs from OpenRouter and print breakdown ─────────────
    report_path = args.cost_report or default_cost_report_path(args.output)
    ledger_path = args.cost_ledger or default_cost_ledger_path(args.output)
    cumulative_path = (
        args.cumulative_cost_report
        or default_cumulative_cost_report_path(args.output)
    )
    await tracker.fetch_and_report(
        save_json_path=report_path,
        ledger_jsonl_path=ledger_path,
        cumulative_summary_path=cumulative_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Multi-LLM Stage-1 evidence adjudication with weighted majority voting.\n"
            "Default backend is OpenRouter; local_openai is available via --committee-config."
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
    ap.add_argument("--cost-ledger",   dest="cost_ledger",   default=None,
                    help="Path to append cumulative cost ledger JSONL (default: <output>_cost_ledger.jsonl)")
    ap.add_argument("--cumulative-cost-report", dest="cumulative_cost_report", default=None,
                    help="Path to save cumulative cost summary JSON (default: <output>_cost_cumulative.json)")
    ap.add_argument("--use-cache",     dest="use_cache", action="store_true", default=False,
                    help="Reuse/write local raw-response cache for exact matching calls")
    ap.add_argument("--committee-backend", choices=["openrouter", "local_openai"],
                    default="openrouter",
                    help="Committee backend (default: openrouter)")
    ap.add_argument("--committee-config", default=None,
                    help="JSON local_openai committee config")
    ap.add_argument("--cache-mode", choices=["off", "read_write", "read_only", "write_only"],
                    default=None,
                    help="Explicit raw-response cache mode")
    ap.add_argument("--cache-dir", default=None,
                    help="Override response cache root directory")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except CacheMissError as exc:
        raise SystemExit(f"Read-only cache miss: {exc}") from None


if __name__ == "__main__":
    main()
