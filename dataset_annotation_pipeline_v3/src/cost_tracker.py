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
import hashlib
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

    def _summary_dict_for_records(
        self,
        records_source: List[Dict[str, Any]],
        *,
        generated_at_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        total_cost = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in records_source)
        total_in = sum(int(r.get("prompt_tokens", 0) or 0) for r in records_source)
        total_out = sum(int(r.get("completion_tokens", 0) or 0) for r in records_source)

        model_stats: Dict[str, Dict[str, Any]] = {}
        for r in records_source:
            model = str(r.get("model", ""))
            s = model_stats.setdefault(model, {
                "model": model,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            })
            s["calls"] += 1
            s["prompt_tokens"] += int(r.get("prompt_tokens", 0) or 0)
            s["completion_tokens"] += int(r.get("completion_tokens", 0) or 0)
            s["cost_usd"] += float(r.get("cost_usd", 0.0) or 0.0)

        models = sorted(
            model_stats.values(),
            key=lambda item: (-item["cost_usd"], item["model"]),
        )

        return {
            "stage": self.stage,
            "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
            "calls": len(records_source),
            "total_prompt_tokens": total_in,
            "total_completion_tokens": total_out,
            "total_cost_usd": total_cost,
            "fetch_error_count": sum(1 for r in records_source if r.get("fetch_error")),
            "models": models,
            "records": records_source,
        }

    def summary_dict(self) -> Dict[str, Any]:
        records = [asdict(r) for r in self._records]
        return self._summary_dict_for_records(records)

    def cumulative_summary_from_ledger(self, ledger_path: str) -> Dict[str, Any]:
        records = _dedupe_ledger_records(_read_ledger_records(Path(ledger_path)))
        return self._summary_dict_for_records(records)

    def append_ledger(self, ledger_path: str) -> int:
        """
        Append this run's fetched call records to an append-only JSONL ledger.

        Existing generation IDs are not appended again, so re-running report
        generation for the same calls remains idempotent. Returns the number of
        newly appended ledger rows.
        """
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = _ledger_records_by_generation_id(path)
        appended = 0
        run_id = _run_id_for_records(self._records)

        with path.open("a", encoding="utf-8") as f:
            for rec in self._records:
                old = existing.get(rec.generation_id)
                is_correction = bool(
                    old
                    and old.get("fetch_error")
                    and not rec.fetch_error
                )
                if old and not is_correction:
                    continue
                row = asdict(rec)
                row["ledger_appended_at_utc"] = datetime.now(timezone.utc).isoformat()
                row["run_id"] = run_id
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[rec.generation_id] = row
                appended += 1

        print(f"  🧾  Cost ledger appended {appended} row(s) → {path}")
        return appended

    def save_cumulative_summary_json(self, ledger_path: str, summary_path: str) -> None:
        out_path = Path(summary_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.cumulative_summary_from_ledger(ledger_path)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  🧾  Cumulative cost summary saved → {out_path}")

    def print_cumulative_summary(self, ledger_path: str) -> None:
        summary = self.cumulative_summary_from_ledger(ledger_path)
        if not summary["calls"]:
            return

        total_cost = summary["total_cost_usd"]
        total_cost_str = f"${total_cost:.5f}" if total_cost < 1 else f"${total_cost:.4f}"
        print(
            f"  🧾  Cumulative ledger total ({self.stage}): "
            f"{summary['calls']} calls, {summary['total_prompt_tokens']:,} in / "
            f"{summary['total_completion_tokens']:,} out, {total_cost_str}"
        )

    def save_cumulative_ledger(
        self,
        ledger_path: str,
        summary_path: Optional[str] = None,
    ) -> int:
        appended = self.append_ledger(ledger_path)
        self.print_cumulative_summary(ledger_path)
        if summary_path:
            self.save_cumulative_summary_json(ledger_path, summary_path)
        return appended

    def save_json(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary_dict(), f, ensure_ascii=False, indent=2)
        print(f"  💾  Cost report saved → {out_path}")

    async def fetch_and_report(
        self,
        save_json_path: Optional[str] = None,
        ledger_jsonl_path: Optional[str] = None,
        cumulative_summary_path: Optional[str] = None,
    ) -> float:
        """
        Convenience: fetch costs then print report.
        Returns total cost in USD.
        """
        await self.fetch_costs()
        self.report()
        if save_json_path:
            self.save_json(save_json_path)
        if ledger_jsonl_path:
            self.save_cumulative_ledger(
                ledger_jsonl_path,
                summary_path=cumulative_summary_path,
            )
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


def default_cost_ledger_path(output_path: str) -> str:
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_cost_ledger.jsonl"))


def default_cumulative_cost_report_path(output_path: str) -> str:
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_cost_cumulative.json"))


def _read_ledger_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    return rows


def _ledger_records_by_generation_id(path: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["generation_id"]): row
        for row in _dedupe_ledger_records(_read_ledger_records(path))
        if row.get("generation_id")
    }


def _dedupe_ledger_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse append-only ledger rows to one effective row per generation.

    Normal resume/report calls append each generation once. If a previous cost
    fetch had an error and a later run appends a successful correction for the
    same generation_id, prefer the successful row so cumulative totals repair
    themselves without rewriting ledger history.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    no_id: List[Dict[str, Any]] = []
    for row in rows:
        generation_id = row.get("generation_id")
        if not generation_id:
            no_id.append(row)
            continue
        key = str(generation_id)
        current = by_id.get(key)
        if current is None:
            by_id[key] = row
            continue
        current_failed = bool(current.get("fetch_error"))
        row_succeeded = not bool(row.get("fetch_error"))
        if current_failed and row_succeeded:
            by_id[key] = row
            continue
        current_time = str(current.get("ledger_appended_at_utc", ""))
        row_time = str(row.get("ledger_appended_at_utc", ""))
        if current_failed == bool(row.get("fetch_error")) and row_time > current_time:
            by_id[key] = row
    return no_id + list(by_id.values())


def _run_id_for_records(records: List[_CallRecord]) -> str:
    material = "|".join(sorted(r.generation_id for r in records if r.generation_id))
    if not material:
        material = datetime.now(timezone.utc).isoformat()
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
