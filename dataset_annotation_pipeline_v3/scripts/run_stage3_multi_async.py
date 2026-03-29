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

All models are accessed via OpenRouter (OPENROUTER_API_KEY required).

This script works identically for both the conflicts and refusals pipelines
since Stage 3 only uses per_doc_notes, conflict_type, conflict_reason,
and answerable_under_evidence — all of which are already resolved by Stage 2.

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

from src.llm_client import LLMClient, Provider
from src.parsers    import parse_stage3
from src.voting     import COMMITTEE_MODELS, MODEL_WEIGHTS, merge_stage3_votes
from src.cost_tracker import CostTracker

SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_stage3.txt"
USER_PROMPT_PATH   = PROJECT_ROOT / "prompts" / "user_stage3.txt"
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
) -> Dict[str, Any]:
    """One API call: one committee model for one record."""
    async with semaphore:
        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=STAGE3_MAX_TOKENS,
                cost_tracker=tracker,
            )
            parsed, errors = parse_stage3(raw)
            if errors:
                parsed["_stage3_errors"] = errors
                parsed["_raw_output"]    = raw[:500]
            return parsed
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
) -> None:
    coros = [
        call_one_model(clients[model], semaphore, system_prompt, user_prompt, tracker)
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
    system_prompt = load_text(
        Path(args.system_prompt) if args.system_prompt else SYSTEM_PROMPT_PATH
    )
    user_template = load_text(
        Path(args.user_prompt) if args.user_prompt else USER_PROMPT_PATH
    )

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
        print("✅ Nothing to process.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(
        f"⚙️  Stage-3 multi-LLM | committee={len(COMMITTEE_MODELS)} models | "
        f"records={len(records)} | concurrency={args.concurrency}"
    )
    for model, weight in MODEL_WEIGHTS.items():
        print(f"   {weight:.0%}  {model}")
    if done_ids:
        print(f"⏩ Resuming: {len(done_ids)} already processed")

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock  = asyncio.Lock()
    tracker   = CostTracker(stage="stage3")

    tasks = [
        process_record(
            clients, semaphore,
            system_prompt, build_user_prompt(user_template, rec),
            rec, out_lock, args.output, tracker,
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
    await tracker.fetch_and_report()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Multi-LLM Stage-3 response synthesis with weighted majority vote on abstain.\n"
            "All committee models are accessed via OpenRouter.\n"
            "Works for both conflicts and refusals datasets."
        )
    )
    ap.add_argument("--input",         required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--concurrency",   type=int,   default=15,
                    help="Total concurrent API calls across ALL committee models (default: 15)")
    ap.add_argument("--limit",         type=int,   default=None)
    ap.add_argument("--max-retries",   type=int,   default=3)
    ap.add_argument("--system-prompt", dest="system_prompt", default=None)
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
