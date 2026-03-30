"""
src/cost_tracker.py
===================
Per-run cost tracking for the multi-LLM committee pipeline, using
OpenRouter's official generation endpoint to fetch exact USD costs.

How it works
------------
1. Every call to LLMClient.acomplete() records (generation_id, model, stage)
   in a shared CostTracker instance.
2. After the pipeline stage completes, call tracker.fetch_and_report() which:
     a. Hits GET https://openrouter.ai/api/v1/generation?id={gen_id}
        for each recorded call (concurrently, in batches).
     b. Sums total_cost (USD) per model and overall.
     c. Prints a formatted cost breakdown table.

This uses OpenRouter's own billing data — no pricing table, no estimation.
The same numbers appear in your OpenRouter dashboard.

Thread / async safety
---------------------
record() uses a threading.Lock so it is safe to call from concurrent asyncio
tasks (the event loop runs in one thread, but Lock is still correct here).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Max concurrent requests to the generation endpoint during the cost fetch.
_COST_FETCH_CONCURRENCY = 30
# Seconds to wait before fetching costs — OpenRouter may need a moment to
# finalize billing for very recent generations.
_FETCH_DELAY_SECONDS = 2.0


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CallRecord:
    generation_id: str
    model:         str
    stage:         str          # "stage1" | "stage2" | "stage3"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # filled after cost fetch:
    cost_usd:      float = 0.0
    fetch_error:   str  = ""


class CostTracker:
    """
    Thread-safe accumulator for per-call cost records.

    Usage
    -----
    tracker = CostTracker(stage="stage1")
    # pass tracker=tracker to LLMClient.acomplete()
    # after pipeline stage:
    await tracker.fetch_and_report()
    """

    def __init__(self, stage: str = "unknown") -> None:
        self.stage     = stage
        self._records: List[_CallRecord] = []
        self._lock     = threading.Lock()

    # ── Recording ────────────────────────────────────────────────────────────

    def record(
        self,
        generation_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Called by LLMClient.acomplete() after every successful OpenRouter call."""
        rec = _CallRecord(
            generation_id     = generation_id,
            model             = model,
            stage             = self.stage,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
        )
        with self._lock:
            self._records.append(rec)

    # ── Cost fetching ────────────────────────────────────────────────────────

    async def _fetch_one(
        self,
        session,          # httpx.AsyncClient
        record: _CallRecord,
        sem: asyncio.Semaphore,
    ) -> None:
        """Fetch exact USD cost for one generation from OpenRouter."""
        api_key = _get_openrouter_key()
        url     = f"https://openrouter.ai/api/v1/generation?id={record.generation_id}"
        async with sem:
            for attempt in range(3):
                try:
                    resp = await session.get(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=15.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        record.cost_usd = float(data.get("total_cost", 0.0))
                        return
                    elif resp.status_code == 404:
                        # Generation not yet indexed — wait and retry
                        await asyncio.sleep(2.0 * (attempt + 1))
                    else:
                        record.fetch_error = f"HTTP {resp.status_code}"
                        return
                except Exception as exc:
                    record.fetch_error = str(exc)[:80]
                    if attempt < 2:
                        await asyncio.sleep(2.0)

    async def fetch_costs(self) -> None:
        """
        Fetch exact costs from OpenRouter for all recorded calls.
        Runs concurrently (up to _COST_FETCH_CONCURRENCY parallel requests).
        Call this after the pipeline stage finishes.
        """
        if not self._records:
            return

        # Brief delay to let OpenRouter finalize billing for recent generations
        await asyncio.sleep(_FETCH_DELAY_SECONDS)

        try:
            import httpx
        except ImportError:
            print(
                "⚠  httpx not installed — cannot fetch exact costs from OpenRouter.\n"
                "   Run: pip install httpx\n"
                "   Falling back to token-count-only summary."
            )
            self._print_tokens_only()
            return

        sem = asyncio.Semaphore(_COST_FETCH_CONCURRENCY)
        async with httpx.AsyncClient() as session:
            await asyncio.gather(*[
                self._fetch_one(session, rec, sem)
                for rec in self._records
            ])

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> None:
        """Print a formatted cost breakdown. Call after fetch_costs()."""
        if not self._records:
            print(f"\n💰 Cost report ({self.stage}): no calls recorded.")
            return

        total_cost   = sum(r.cost_usd for r in self._records)
        total_in     = sum(r.prompt_tokens for r in self._records)
        total_out    = sum(r.completion_tokens for r in self._records)
        fetch_errors = [r for r in self._records if r.fetch_error]
        n_calls      = len(self._records)

        # Per-model breakdown
        model_stats: Dict[str, Dict] = {}
        for r in self._records:
            s = model_stats.setdefault(r.model, {
                "calls": 0, "cost": 0.0, "in_tok": 0, "out_tok": 0
            })
            s["calls"]   += 1
            s["cost"]    += r.cost_usd
            s["in_tok"]  += r.prompt_tokens
            s["out_tok"] += r.completion_tokens

        # Format
        W = 54   # model column width
        print(f"\n{'━'*72}")
        print(f"  💰  Cost Report — {self.stage.upper()}")
        print(f"{'━'*72}")
        print(f"  {'Model':<{W}} {'Calls':>6}  {'In tok':>8}  {'Out tok':>8}  {'Cost':>10}")
        print(f"  {'-'*W}  {'------':>6}  {'--------':>8}  {'--------':>8}  {'----------':>10}")
        for model, s in sorted(model_stats.items(), key=lambda x: -x[1]["cost"]):
            cost_str = f"${s['cost']:.5f}" if s["cost"] < 1 else f"${s['cost']:.4f}"
            print(
                f"  {model:<{W}} {s['calls']:>6}  {s['in_tok']:>8,}  "
                f"{s['out_tok']:>8,}  {cost_str:>10}"
            )
        print(f"  {'-'*W}  {'------':>6}  {'--------':>8}  {'--------':>8}  {'----------':>10}")
        total_cost_str = f"${total_cost:.5f}" if total_cost < 1 else f"${total_cost:.4f}"
        print(
            f"  {'TOTAL':<{W}} {n_calls:>6}  {total_in:>8,}  "
            f"{total_out:>8,}  {total_cost_str:>10}"
        )
        print(f"{'━'*72}")

        if fetch_errors:
            print(
                f"\n  ⚠  {len(fetch_errors)} call(s) had cost-fetch errors "
                f"(shown as $0.00 above). Generation IDs saved to tracker.records."
            )

    def summary_dict(self) -> Dict[str, Any]:
        total_cost = sum(r.cost_usd for r in self._records)
        total_in = sum(r.prompt_tokens for r in self._records)
        total_out = sum(r.completion_tokens for r in self._records)

        model_stats: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            s = model_stats.setdefault(r.model, {
                "model": r.model,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            })
            s["calls"] += 1
            s["prompt_tokens"] += r.prompt_tokens
            s["completion_tokens"] += r.completion_tokens
            s["cost_usd"] += r.cost_usd

        models = sorted(
            model_stats.values(),
            key=lambda item: (-item["cost_usd"], item["model"]),
        )
        records = [asdict(r) for r in self._records]

        return {
            "stage": self.stage,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "calls": len(self._records),
            "total_prompt_tokens": total_in,
            "total_completion_tokens": total_out,
            "total_cost_usd": total_cost,
            "fetch_error_count": sum(1 for r in self._records if r.fetch_error),
            "models": models,
            "records": records,
        }

    def save_json(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary_dict(), f, ensure_ascii=False, indent=2)
        print(f"  💾  Cost report saved → {out_path}")

    async def fetch_and_report(self, save_json_path: Optional[str] = None) -> float:
        """
        Convenience: fetch costs then print report.
        Returns total cost in USD.
        """
        await self.fetch_costs()
        self.report()
        if save_json_path:
            self.save_json(save_json_path)
        return sum(r.cost_usd for r in self._records)

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def records(self) -> List[_CallRecord]:
        """All recorded call records (after fetch_costs, .cost_usd is populated)."""
        return list(self._records)

    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def _print_tokens_only(self) -> None:
        total_in  = sum(r.prompt_tokens  for r in self._records)
        total_out = sum(r.completion_tokens for r in self._records)
        print(f"\n💰 Token usage ({self.stage}): {total_in:,} in / {total_out:,} out")


# ─── Key helper ──────────────────────────────────────────────────────────────

def _get_openrouter_key() -> str:
    """Resolve OPENROUTER_API_KEY (same logic as llm_client.py)."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        p = os.path.expanduser("~/.openrouter_key")
        if os.path.exists(p):
            key = open(p).read().strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Export it or write it to ~/.openrouter_key"
        )
    return key


def default_cost_report_path(output_path: str) -> str:
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_cost_report.json"))
