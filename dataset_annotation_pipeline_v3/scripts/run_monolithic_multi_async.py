#!/usr/bin/env python3
"""
scripts/run_monolithic_multi_async.py
======================================
Multi-LLM Committee version of the MONOLITHIC annotation strategy.

Each query record is sent to ALL 5 committee models (via OpenRouter) using
the same monolithic prompt. Results are merged via weighted majority voting:

  • Per-doc verdict          : voted across models (like stage1 merge)
  • abstain                  : voted across models (like stage3 merge)
  • All text fields          : adopted wholesale from the highest-weight model
                               that voted for the winning abstain value

All models are accessed via OpenRouter (OPENROUTER_API_KEY required).
No direct Anthropic or OpenAI keys are used.

Committee (from src/voting.py):
  anthropic/claude-sonnet-4.6  weight 0.30
  openai/gpt-5.4               weight 0.25
  qwen/qwen3.5-27b             weight 0.20
  deepseek/deepseek-v3.2       weight 0.15
  x-ai/grok-4.1-fast           weight 0.10

Usage:
    python scripts/run_monolithic_multi_async.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/monolithic_outputs/monolithic_multi.jsonl \\
        --concurrency 10 \\
        --limit 5
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm_asyncio

THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client   import LLMClient, Provider
from src.parsers      import parse_monolithic
from src.voting       import (
    COMMITTEE_MODELS, MODEL_WEIGHTS,
    weighted_majority_vote, select_winner_model, _build_votes,
)
from src.cost_tracker import CostTracker, default_cost_report_path

SYSTEM_PROMPT_PATH    = PROJECT_ROOT / "prompts" / "system_monolithic.txt"
USER_PROMPT_PATH      = PROJECT_ROOT / "prompts" / "user_monolithic.txt"
MONOLITHIC_MAX_TOKENS = 3000


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_user_prompt(template: str, record: Dict[str, Any]) -> str:
    """Substitute placeholders in the monolithic user template."""
    query         = record.get("query", "")
    conflict_type = record.get("conflict_type", "")
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
        .replace("{query}",               query)
        .replace("{conflict_type}",       conflict_type)
        .replace("{retrieved_docs_json}", json.dumps(docs_slim, ensure_ascii=False, indent=2))
    )


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_processed_ids(output_path: str) -> set:
    done: set = set()
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
#  Per-model monolithic call
# ─────────────────────────────────────────────

async def call_one_model(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    doc_ids: List[str],
    tracker: CostTracker,
) -> Optional[Dict[str, Any]]:
    """
    One API call: one committee model for one query (monolithic).
    Returns the parsed monolithic dict, or None on irrecoverable error.
    """
    async with semaphore:
        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=MONOLITHIC_MAX_TOKENS,
                cost_tracker=tracker,
            )
            parsed, errors = parse_monolithic(raw, expected_doc_ids=doc_ids)
            if errors:
                parsed["_monolithic_errors"] = errors
            return parsed
        except Exception as exc:
            return {
                "per_doc_notes": [],
                "conflict_reason": f"API error: {str(exc)[:120]}",
                "answerable_under_evidence": False,
                "expected_response": {
                    "answer":        "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                    "evidence":      [],
                    "abstain":       True,
                    "abstain_reason": str(exc),
                },
                "think":   "",
                "_error":  str(exc),
            }


# ─────────────────────────────────────────────
#  Committee merge for monolithic outputs
# ─────────────────────────────────────────────

def _merge_per_doc_notes(
    model_results: Dict[str, Optional[Dict[str, Any]]],
    doc_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    For each doc_id, collect each model's verdict for that doc and run
    weighted majority vote.  Text fields (key_fact, quote, verdict_reason,
    source_quality) are adopted from the highest-weight model that voted
    for the winning verdict — identical logic to stage-1 merge.
    """
    merged_notes = []
    for doc_id in doc_ids:
        # Collect per-model note for this doc_id
        model_notes: Dict[str, Optional[Dict[str, Any]]] = {}
        for model, result in model_results.items():
            if result is None:
                model_notes[model] = None
                continue
            pdn = result.get("per_doc_notes", [])
            note = next((n for n in pdn if n.get("doc_id") == doc_id), None)
            model_notes[model] = note

        # Build verdict votes
        votes = [
            (model, (note or {}).get("verdict", "irrelevant"), MODEL_WEIGHTS.get(model, 0.0))
            for model, note in model_notes.items()
            if note is not None
        ]

        if not votes:
            merged_notes.append({
                "doc_id": doc_id, "verdict": "irrelevant",
                "key_fact": "", "quote": "", "verdict_reason": "",
                "source_quality": "low",
            })
            continue

        winning_verdict, tally = weighted_majority_vote(votes)
        winning_model   = select_winner_model(votes, winning_verdict)

        base: Dict[str, Any] = (model_notes.get(winning_model) or {}).copy()
        base["verdict"]       = winning_verdict
        base["doc_id"]        = doc_id
        base["_vote_tally"]   = {str(k): round(v, 4) for k, v in tally.items()}
        base["_winner_model"] = winning_model
        base["_all_verdicts"] = {
            m: (model_notes.get(m) or {}).get("verdict")
            for m in COMMITTEE_MODELS
        }
        merged_notes.append(base)

    return merged_notes


def merge_monolithic_votes(
    model_results: Dict[str, Optional[Dict[str, Any]]],
    doc_ids: List[str],
) -> Dict[str, Any]:
    """
    Merge per-model monolithic outputs.

    Votes on:
      • per-doc verdict (per doc_id, like stage-1 merge)
      • abstain flag in expected_response (like stage-3 merge)

    Adopts text fields from the highest-weight model that voted for the
    winning abstain value (conflict_reason, expected_response, think).
    """
    # ── 1. Per-doc verdict voting ────────────────────────────────────────────
    merged_per_doc = _merge_per_doc_notes(model_results, doc_ids)

    # ── 2. Abstain voting (mirrors merge_stage3_votes) ───────────────────────
    flat_abstain: Dict[str, Optional[Dict[str, Any]]] = {
        model: {
            "abstain": ((res or {}).get("expected_response") or {}).get("abstain", False)
        }
        for model, res in model_results.items()
        if res is not None
    }

    abstain_votes = [
        (model, (rec or {}).get("abstain", False), MODEL_WEIGHTS.get(model, 0.0))
        for model, rec in flat_abstain.items()
        if rec is not None
    ]

    if not abstain_votes:
        # All models failed — full fallback
        return {
            "per_doc_notes":             merged_per_doc,
            "conflict_reason":           "",
            "answerable_under_evidence": False,
            "expected_response": {
                "answer":        "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                "evidence":      [],
                "abstain":       True,
                "abstain_reason": "All committee models failed.",
            },
            "think": "",
            "_all_models_failed": True,
        }

    winning_abstain, abstain_tally = weighted_majority_vote(abstain_votes)
    abstain_winner = select_winner_model(abstain_votes, winning_abstain)

    # Adopt full output from the abstain-winning model
    base: Dict[str, Any] = (model_results.get(abstain_winner) or {}).copy()

    # Overwrite per_doc_notes with the voted consensus notes
    base["per_doc_notes"] = merged_per_doc

    # Enforce voted abstain value
    er = base.setdefault("expected_response", {})
    er["abstain"] = winning_abstain

    # answerable_under_evidence: derived from voted per_doc_notes
    non_irr = [
        n for n in merged_per_doc
        if n.get("verdict") in ("supports", "partially supports")
    ]
    base["answerable_under_evidence"] = len(non_irr) > 0

    # Audit fields
    base["_abstain_vote_tally"]   = {str(k): round(v, 4) for k, v in abstain_tally.items()}
    base["_abstain_winner_model"] = abstain_winner
    base["_annotation_strategy"]  = "monolithic_multi"

    return base


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
    doc_ids    = [d.get("doc_id", "") for d in record.get("retrieved_docs", [])]
    user_prompt = fill_user_prompt(user_template, record)

    # Fire ALL 5 committee models concurrently — each acquires semaphore independently
    coros = [
        call_one_model(clients[model], semaphore, system_prompt, user_prompt, doc_ids, tracker)
        for model in COMMITTEE_MODELS
    ]
    raw_results  = await asyncio.gather(*coros)
    model_results = {model: raw_results[i] for i, model in enumerate(COMMITTEE_MODELS)}

    merged = merge_monolithic_votes(model_results, doc_ids)

    # Write merged fields back into the record
    for key in (
        "per_doc_notes", "conflict_reason", "answerable_under_evidence",
        "expected_response", "think", "_annotation_strategy",
        "_abstain_vote_tally", "_abstain_winner_model",
        "_monolithic_errors", "_all_models_failed",
    ):
        if key in merged:
            record[key] = merged[key]

    # Preserve any per-record parse errors from winning model
    if "_monolithic_errors" in merged:
        record["_monolithic_errors"] = merged["_monolithic_errors"]

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

    # One LLMClient per committee model — all via OpenRouter
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
        f"⚙️  Monolithic multi-LLM | committee={len(COMMITTEE_MODELS)} models | "
        f"records={len(records)} | concurrency={args.concurrency}"
    )
    for model, weight in MODEL_WEIGHTS.items():
        print(f"   {weight:.0%}  {model}")
    if done_ids:
        print(f"⏩ Resuming: {len(done_ids)} already processed")

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock  = asyncio.Lock()
    tracker   = CostTracker(stage="monolithic_multi")

    tasks = [
        process_record(
            clients, semaphore,
            system_prompt, user_template,
            rec, out_lock, args.output, tracker,
        )
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Monolithic multi-LLM")

    # Stats
    abstained, answered, errors_count = 0, 0, 0
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("_error") or rec.get("_all_models_failed"):
                    errors_count += 1
                elif rec.get("expected_response", {}).get("abstain"):
                    abstained += 1
                else:
                    answered += 1
            except Exception:
                pass

    print(f"\n✅ Monolithic multi-LLM complete → {args.output}")
    print(f"   answered={answered}  abstained={abstained}  errors={errors_count}")

    # ── Fetch exact costs from OpenRouter and print breakdown ─────────────────
    report_path = args.cost_report or default_cost_report_path(args.output)
    await tracker.fetch_and_report(save_json_path=report_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Multi-LLM Committee Monolithic annotation.\n"
            "Sends the monolithic prompt to ALL 5 committee models via OpenRouter,\n"
            "then merges via weighted majority voting (per-doc verdict + abstain).\n"
            "Only OPENROUTER_API_KEY is needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",         required=True,
                    help="Input JSONL (normalized conflicts or refusals dataset)")
    ap.add_argument("--output",        required=True,
                    help="Output JSONL path")
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--concurrency",   type=int,   default=10,
                    help=(
                        "Total concurrent API calls across ALL committee models "
                        "(default: 10 ≈ 2 records × 5 models in parallel)"
                    ))
    ap.add_argument("--limit",         type=int,   default=None,
                    help="Max records to process (default: all)")
    ap.add_argument("--max-retries",   type=int,   default=3,
                    help="Retries per failed API call (default: 3)")
    ap.add_argument("--system-prompt", dest="system_prompt", default=None,
                    help="Override system prompt path (default: prompts/system_monolithic.txt)")
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None,
                    help="Override user prompt path (default: prompts/user_monolithic.txt)")
    ap.add_argument("--cost-report",   dest="cost_report",   default=None,
                    help="Path to save cost report JSON (default: <output>_cost_report.json)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
