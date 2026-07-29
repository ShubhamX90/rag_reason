#!/usr/bin/env python3
"""
scripts/run_stage2_multi_async.py
==================================
Multi-LLM Stage-2: conflict reasoning + answerability with weighted
majority voting across the annotation committee.

CONFLICTS dataset  (run WITHOUT --refusal-mode):
    Every record has a gold human-annotated conflict_type.
    Votes on  : answerable_under_evidence
    Adopts    : conflict_reason from the answerable-vote winner
    Prompts   : system_stage2.txt / user_stage2.txt  (conflict_type given verbatim)

REFUSALS dataset  (run WITH --refusal-mode):
    Use this when the refusal requirement is ground truth for the run.
    Votes on   : conflict_type  AND  answerable_under_evidence  (independently)
    Adopts     : conflict_reason from the conflict_type-vote winner
    Prompts    : system_stage2_refusal.txt / user_stage2_refusal.txt
                 (model knows the sample is refusal-required and still annotates conflict_type)

Usage:
    # Conflicts (human-annotated conflict_type — no re-annotation)
    python scripts/run_stage2_multi_async.py \\
        --input  data/stage1_outputs/stage1_multi.jsonl \\
        --output data/stage2_outputs/stage2_multi.jsonl \\
        --concurrency 20

    # Refusals (committee re-annotation of conflict_type)
    python scripts/run_stage2_multi_async.py \\
        --input  data/stage1_outputs/refusals_stage1_multi.jsonl \\
        --output data/stage2_outputs/refusals_stage2_multi.jsonl \\
        --refusal-mode \\
        --concurrency 20
BENCHMARK dataset  (run WITH --benchmark-mode):
    Use this when benchmark labels should be re-annotated by the committee
    from per-document notes, without forcing a refusal outcome.
    Votes on   : conflict_type  AND  answerable_under_evidence independently
    Adopts     : conflict_reason from the conflict_type-vote winner
    Prompts    : system_stage2_benchmark.txt / user_stage2_benchmark.txt
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
from src.parsers    import parse_stage2, parse_stage2_refusal
from src.voting     import COMMITTEE_MODELS, MODEL_WEIGHTS, merge_stage2_votes
from src.cost_tracker import (
    CostTracker,
    default_cost_ledger_path,
    default_cost_report_path,
    default_cumulative_cost_report_path,
)

# ── Prompt paths ──────────────────────────────────────────────────────────────
SYSTEM_CONFLICTS = PROJECT_ROOT / "prompts" / "system_stage2.txt"
USER_CONFLICTS   = PROJECT_ROOT / "prompts" / "user_stage2.txt"
SYSTEM_REFUSALS  = PROJECT_ROOT / "prompts" / "system_stage2_refusal.txt"
USER_REFUSALS    = PROJECT_ROOT / "prompts" / "user_stage2_refusal.txt"
SYSTEM_BENCHMARK = PROJECT_ROOT / "prompts" / "system_stage2_benchmark.txt"
USER_BENCHMARK   = PROJECT_ROOT / "prompts" / "user_stage2_benchmark.txt"

STAGE2_MAX_TOKENS = 400   # slightly more than v2 to accommodate conflict_type field in refusal output


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def brace_safe_fill(template: str, mapping: Dict[str, str]) -> str:
    """Fill {KEY} placeholders without corrupting JSON braces in values."""
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


def build_user_prompt_conflicts(template: str, record: Dict[str, Any]) -> str:
    """Build Stage-2 user prompt for the conflicts dataset."""
    return brace_safe_fill(template, {
        "QUERY":              record.get("query", ""),
        "CONFLICT_TYPE":      record.get("conflict_type", ""),
        "PER_DOC_NOTES_JSON": json.dumps(record.get("per_doc_notes", []), ensure_ascii=False, indent=2),
    })


def build_user_prompt_refusals(template: str, record: Dict[str, Any]) -> str:
    """Build Stage-2 user prompt for refusal-required samples."""
    return brace_safe_fill(template, {
        "QUERY":              record.get("query", ""),
        "PER_DOC_NOTES_JSON": json.dumps(record.get("per_doc_notes", []), ensure_ascii=False, indent=2),
    })


# ─────────────────────────────────────────────
#  Per-model single call
# ─────────────────────────────────────────────

async def call_one_model(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    mode: str,
    tracker: CostTracker,
    cache_enabled: bool,
    cache_mode: str,
) -> Dict[str, Any] | None:
    """One API call: one committee model for one record."""
    async with semaphore:
        try:
            raw = await client.acomplete(
                system=system_prompt,
                user=user_prompt,
                max_tokens=client_max_tokens(client, STAGE2_MAX_TOKENS),
                cost_tracker=tracker,
                cache_enabled=cache_enabled,
                cache_mode=cache_mode,
                cache_namespace=(
                    "stage2_multi_refusal" if mode == "refusal"
                    else "stage2_multi_benchmark" if mode == "benchmark"
                    else "stage2_multi"
                ),
            )
            if mode in {"refusal", "benchmark"}:
                parsed, errors = parse_stage2_refusal(raw)
            else:
                parsed, errors = parse_stage2(raw)
            if errors:
                parsed["_validation_errors"] = errors
            return parsed
        except CacheMissError:
            raise
        except Exception as exc:
            base: Dict[str, Any] = {
                "conflict_reason":           f"API error: {str(exc)[:120]}",
                "answerable_under_evidence": False,
                "_error":                    str(exc),
            }
            if mode in {"refusal", "benchmark"}:
                base["conflict_type"] = ""
            return base


# ─────────────────────────────────────────────
#  Per-record processing
# ─────────────────────────────────────────────

async def process_record(
    clients: Dict[str, LLMClient],
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_prompt: str,
    record: Dict[str, Any],
    mode: str,
    out_lock: asyncio.Lock,
    output_path: str,
    tracker: CostTracker,
    cache_enabled: bool,
    cache_mode: str,
) -> None:
    # Run all committee models concurrently for this single record
    coros = [
        call_one_model(
            clients[model], semaphore, system_prompt, user_prompt,
            mode, tracker, cache_enabled, cache_mode
        )
        for model in COMMITTEE_MODELS
    ]
    raw_results   = await asyncio.gather(*coros)
    model_records = {model: raw_results[i] for i, model in enumerate(COMMITTEE_MODELS)}

    merged = merge_stage2_votes(model_records, is_refusal=(mode == "refusal"), vote_conflict_type=(mode != "conflicts"))

    # Write consensus back into the record
    record["answerable_under_evidence"] = merged["answerable_under_evidence"]
    record["conflict_reason"]           = merged.get("conflict_reason", "")

    if mode != "conflicts":
        # Preserve original input label for analysis; overwrite with committee vote
        if "conflict_type" in record:
            record["_gold_conflict_type"] = record["conflict_type"]
        record["conflict_type"] = merged.get("conflict_type", "")

    # Copy vote-audit metadata
    for k in ("_ans_vote_tally", "_ans_winner_model", "_ct_vote_tally", "_ct_winner_model"):
        if k in merged:
            record[k] = merged[k]

    async with out_lock:
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    if args.refusal_mode and args.benchmark_mode:
        raise ValueError("Choose only one of --refusal-mode or --benchmark-mode")

    if args.refusal_mode:
        mode = "refusal"
        if not SYSTEM_REFUSALS.exists():
            raise FileNotFoundError(
                f"Refusal system prompt not found: {SYSTEM_REFUSALS}\n"
                "Ensure prompts/system_stage2_refusal.txt exists."
            )
        if not USER_REFUSALS.exists():
            raise FileNotFoundError(
                f"Refusal user prompt not found: {USER_REFUSALS}\n"
                "Ensure prompts/user_stage2_refusal.txt exists."
            )
        system_prompt = load_text(SYSTEM_REFUSALS)
        user_template = load_text(USER_REFUSALS)
        mode_label    = "REFUSAL (ground-truth refusal; committee votes conflict_type)"
    elif args.benchmark_mode:
        mode = "benchmark"
        system_path = Path(args.system_prompt) if args.system_prompt else SYSTEM_BENCHMARK
        user_path = Path(args.user_prompt) if args.user_prompt else USER_BENCHMARK
        if not system_path.exists():
            raise FileNotFoundError(
                f"Benchmark system prompt not found: {system_path}\n"
                "Ensure prompts/system_stage2_benchmark.txt exists."
            )
        if not user_path.exists():
            raise FileNotFoundError(
                f"Benchmark user prompt not found: {user_path}\n"
                "Ensure prompts/user_stage2_benchmark.txt exists."
            )
        system_prompt = load_text(system_path)
        user_template = load_text(user_path)
        mode_label = "BENCHMARK (committee votes conflict_type and answerability)"
    else:
        mode = "conflicts"
        system_prompt = load_text(
            Path(args.system_prompt) if args.system_prompt else SYSTEM_CONFLICTS
        )
        user_template = load_text(
            Path(args.user_prompt) if args.user_prompt else USER_CONFLICTS
        )
        mode_label = "CONFLICTS (uses gold conflict_type)"

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
        f"⚙️  Stage-2 multi-LLM | backend={args.committee_backend} | mode={mode_label} | "
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
    tracker   = CostTracker(stage="stage2")

    # Pre-build user prompts per record (done outside async for clarity)
    def make_user_prompt(record: Dict[str, Any]) -> str:
        if mode in {"refusal", "benchmark"}:
            return build_user_prompt_refusals(user_template, record)
        return build_user_prompt_conflicts(user_template, record)

    tasks = [
        process_record(
            clients, semaphore,
            system_prompt, make_user_prompt(rec),
            rec, mode,
            out_lock, args.output,
            tracker,
            resolved_cache_mode != "off",
            resolved_cache_mode,
        )
        for rec in records
    ]
    await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Stage-2 multi-LLM")
    print(f"\n✅ Stage-2 multi-LLM complete → {args.output}")

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
            "Multi-LLM Stage-2 conflict reasoning + answerability with weighted majority voting.\n"
            "Default backend is OpenRouter; local_openai is available via --committee-config.\n\n"
            "  Default (no --refusal-mode): CONFLICTS dataset — gold conflict_type used as-is.\n"
            "  With --refusal-mode: REFUSALS dataset — refusal is ground truth; committee still votes conflict_type.\n"
            "  With --benchmark-mode: BENCHMARK dataset — committee votes conflict_type and answerability."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",         required=True,
                    help="Input JSONL (stage1 multi output)")
    ap.add_argument("--output",        required=True,
                    help="Output JSONL path")
    ap.add_argument("--refusal-mode",  dest="refusal_mode", action="store_true", default=False,
                    help=(
                        "Activate refusal ground-truth mode: committee still votes conflict_type "
                        "while producing refusal-oriented Stage-2 annotations."
                    ))
    ap.add_argument("--benchmark-mode", dest="benchmark_mode", action="store_true", default=False,
                    help=(
                        "Activate benchmark mode: committee votes conflict_type and "
                        "answerable_under_evidence without forcing refusal behavior."
                    ))
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--concurrency",   type=int,   default=20,
                    help="Total concurrent API calls across ALL committee models (default: 20)")
    ap.add_argument("--limit",         type=int,   default=None)
    ap.add_argument("--max-retries",   type=int,   default=3)
    ap.add_argument("--system-prompt", dest="system_prompt", default=None,
                    help="Override conflicts system prompt path (ignored in refusal-mode)")
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None,
                    help="Override conflicts user prompt path (ignored in refusal-mode)")
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
