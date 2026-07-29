#!/usr/bin/env python3
"""
scripts/run_stage3_multi_async.py
==================================
Multi-LLM Stage-3: grounded expected-response generation with weighted
majority voting on the abstain decision across the annotation committee.

`expected_response.abstain` is decided by weighted majority vote.
The complete expected_response block (answer, evidence, abstain_reason)
and the think trace are adopted wholesale from the highest-weight model
that voted for the winning abstain value.

By default, models are accessed via OpenRouter (OPENROUTER_API_KEY required).
Use --committee-backend local_openai with a JSON committee config for local
OpenAI-compatible endpoints.

Default mode uses the standard conflicts prompts.
With --refusal-mode, refusal-specific prompts are used so the final
expected_response is forced to refuse.

Usage:
    # Conflicts
    python scripts/run_stage3_multi_async.py \\
        --input  data/stage2_outputs/stage2_multi.jsonl \\
        --output data/stage3_outputs/stage3_multi.jsonl \\
        --concurrency 15

    # Refusals
    python scripts/run_stage3_multi_async.py \\
        --input  data/stage2_outputs/refusals_stage2_multi.jsonl \\
        --output data/stage3_outputs/refusals_stage3_multi.jsonl \\
        --concurrency 15
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
from src.parsers    import parse_stage3
from src.voting     import COMMITTEE_MODELS, MODEL_WEIGHTS, merge_stage3_votes
from src.cost_tracker import (
    CostTracker,
    default_cost_ledger_path,
    default_cost_report_path,
    default_cumulative_cost_report_path,
)

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage3.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage3.txt"
SYSTEM_REFUSAL_PATH = PROJECT_ROOT / "prompts" / "system_stage3_refusal.txt"
USER_REFUSAL_PATH   = PROJECT_ROOT / "prompts" / "user_stage3_refusal.txt"
STAGE3_MAX_TOKENS  = 6000


# ─────────────────────────────────────────────
#  Helpers
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


def build_user_prompt(template: str, record: Dict[str, Any]) -> str:
    return brace_safe_fill(template, {
        "query":                     record.get("query", ""),
        "retrieved_docs":            json.dumps(
            record.get("retrieved_docs", []), ensure_ascii=False, indent=2
        ),
        "per_doc_notes":             json.dumps(
            record.get("per_doc_notes", []), ensure_ascii=False, indent=2
        ),
        "conflict_type":             record.get("conflict_type", ""),
        "conflict_reason":           record.get("conflict_reason", ""),
        "answerable_under_evidence": str(
            record.get("answerable_under_evidence", True)
        ).lower(),
        "gold_answer":               record.get("gold_answer", "") or "",
        "ranked_doc_ids":            ", ".join(
            n.get("doc_id", "")
            for n in record.get("per_doc_notes", [])
            if n.get("verdict") != "irrelevant"
        ),
    })


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_processed_ids(output_path: str) -> set:
    done: set = set()
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
#  Per-model single call
# ─────────────────────────────────────────────

async def call_one_model(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    tracker: CostTracker,
    cache_enabled: bool,
    cache_mode: str,
    is_refusal: bool,
) -> Dict[str, Any] | None:
    """One API call: one committee model for one record."""
    async with semaphore:
        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=client_max_tokens(client, STAGE3_MAX_TOKENS),
                cost_tracker=tracker,
                cache_enabled=cache_enabled,
                cache_mode=cache_mode,
                cache_namespace="stage3_multi_refusal" if is_refusal else "stage3_multi",
            )
            parsed, errors = parse_stage3(raw)
            if errors:
                parsed["_stage3_errors"] = errors
                parsed["_raw_output"]    = raw[:500]
            return parsed
        except CacheMissError:
            raise
        except Exception as exc:
            return {
                "expected_response": {
                    "answer":        "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                    "evidence":      [],
                    "abstain":       True,
                    "abstain_reason": str(exc),
                },
                "think":  "",
                "_error": str(exc),
            }


# ─────────────────────────────────────────────
#  Per-record processing
# ─────────────────────────────────────────────

async def process_record(
    clients: Dict[str, LLMClient],
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    record: Dict[str, Any],
    out_lock: asyncio.Lock,
    output_path: str,
    tracker: CostTracker,
    cache_enabled: bool,
    cache_mode: str,
    is_refusal: bool,
) -> None:
    coros = [
        call_one_model(
            clients[model], semaphore, system_prompt, user_prompt, tracker,
            cache_enabled, cache_mode, is_refusal
        )
        for model in COMMITTEE_MODELS
    ]
    raw_results   = await asyncio.gather(*coros)
    model_records = {model: raw_results[i] for i, model in enumerate(COMMITTEE_MODELS)}

    merged = merge_stage3_votes(model_records)

    record["expected_response"] = merged.get("expected_response", {})
    record["think"]             = merged.get("think", "")

    for k in ("_abstain_vote_tally", "_abstain_winner_model"):
        if k in merged:
            record[k] = merged[k]

    async with out_lock:
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    default_system = SYSTEM_REFUSAL_PATH if args.refusal_mode else SYSTEM_PROMPT_PATH
    default_user = USER_REFUSAL_PATH if args.refusal_mode else USER_PROMPT_PATH
    system_prompt = load_text(
        Path(args.system_prompt) if args.system_prompt else default_system
    )
    user_template = load_text(
        Path(args.user_prompt) if args.user_prompt else default_user
    )

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
        print("✅ Nothing to process.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"⚙️  Stage-3 multi-LLM | backend={args.committee_backend} | "
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
    tracker   = CostTracker(stage="stage3")

    tasks = [
        process_record(
            clients, semaphore,
            system_prompt, build_user_prompt(user_template, rec),
            rec, out_lock, args.output, tracker,
            resolved_cache_mode != "off",
            resolved_cache_mode,
            args.refusal_mode,
        )
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Stage-3 multi-LLM")

    # Quick stats scan
    stats = {"answered": 0, "abstained": 0}
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                er = json.loads(line).get("expected_response", {})
                if er.get("abstain"):
                    stats["abstained"] += 1
                else:
                    stats["answered"] += 1
            except Exception:
                pass

    print(f"\n✅ Stage-3 multi-LLM complete → {args.output}")
    print(f"   {stats}")

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
            "Multi-LLM Stage-3 response synthesis with weighted majority vote on abstain.\n"
            "Default backend is OpenRouter; local_openai is available via --committee-config.\n"
            "Default (no --refusal-mode): conflicts prompts.\n"
            "With --refusal-mode: refusal-specific prompts force abstaining outputs."
        )
    )
    ap.add_argument("--input",         required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--refusal-mode",  dest="refusal_mode", action="store_true", default=False,
                    help="Use refusal-specific Stage-3 prompts for refusal-required samples")
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--concurrency",   type=int,   default=15,
                    help="Total concurrent API calls across ALL committee models (default: 15)")
    ap.add_argument("--limit",         type=int,   default=None)
    ap.add_argument("--max-retries",   type=int,   default=3)
    ap.add_argument("--system-prompt", dest="system_prompt", default=None)
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None)
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
