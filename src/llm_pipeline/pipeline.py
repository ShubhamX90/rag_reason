"""Async pipeline orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Set

import jinja2
from tqdm.asyncio import tqdm_asyncio

from .client import LLMClient
from .config import PipelineConfig
from .retry import PermanentAPIError, with_retry

logger = logging.getLogger(__name__)

# Jinja2 environment: missing vars → empty string (no exception)
_jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.Undefined,
    keep_trailing_newline=True,
)


def _render(template_str: str, context: dict) -> str:
    return _jinja_env.from_string(template_str).render(**context)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d in %s: %s", lineno, path, e)
    return records


def _load_processed_ids(output_path: Path, id_field: Optional[str]) -> Set:
    """Scan existing output file for already-processed record IDs."""
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get(id_field) if id_field else None
                if key is not None:
                    processed.add(key)
            except json.JSONDecodeError:
                pass
    return processed


def _record_id(record: dict, index: int, id_field: Optional[str]) -> Any:
    if id_field and id_field in record:
        return record[id_field]
    return index


def _build_context(record: dict) -> dict:
    ctx = dict(record)          # flat fields: {{ query }}, {{ id }}, …
    ctx["record"] = record      # nested access: {{ record.field }}
    ctx["run_date"] = datetime.now(timezone.utc).date().isoformat()
    return ctx


async def _process_record(
    record: dict,
    index: int,
    system_tpl: str,
    user_tpl: str,
    client: LLMClient,
    cfg: PipelineConfig,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    out_file,
    counters: dict,
) -> None:
    record_id = _record_id(record, index, cfg.io.id_field)

    async with semaphore:
        ctx = _build_context(record)
        system_prompt = _render(system_tpl, ctx)
        user_prompt = _render(user_tpl, ctx)

        timestamp = datetime.now(timezone.utc).isoformat()
        raw_output = None
        parsed_output = None
        attempt_count = 0
        error_msg = None

        try:
            raw_output, parsed_output, attempt_count = await with_retry(
                coro_fn=lambda: client.complete(system_prompt, user_prompt),
                validation_cfg=cfg.validation,
                retry_cfg=cfg.retry,
                record_id=record_id,
            )
            counters["success"] += 1
            logger.debug("Record %s completed in %d attempt(s).", record_id, attempt_count)

        except PermanentAPIError as exc:
            error_msg = f"PermanentAPIError({exc.status_code}): {exc}"
            counters["failure"] += 1
            logger.error("Record %s — permanent error: %s", record_id, error_msg)

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            counters["failure"] += 1
            logger.error("Record %s — failed after all retries: %s", record_id, error_msg)

        # Assemble output record
        out_record = dict(record)

        if cfg.output.include_raw_output:
            out_record["llm_output"] = raw_output

        if cfg.output.include_parsed:
            out_record["llm_parsed"] = parsed_output

        if cfg.output.include_meta:
            out_record["_pipeline_meta"] = {
                "timestamp_utc": timestamp,
                "provider": cfg.provider,
                "model": cfg.model,
                "attempt_count": attempt_count,
                "error": error_msg,
            }

        async with lock:
            out_file.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_file.flush()


async def run_pipeline(cfg: PipelineConfig) -> None:
    records = _read_jsonl(cfg.io.input)
    total = len(records)
    logger.info("Loaded %d records from %s", total, cfg.io.input)

    # Resume: skip already-processed records
    processed_ids = _load_processed_ids(cfg.io.output, cfg.io.id_field)
    if processed_ids:
        logger.info("Resuming: %d records already processed.", len(processed_ids))

    pending = []
    for i, rec in enumerate(records):
        rid = _record_id(rec, i, cfg.io.id_field)
        if rid in processed_ids:
            continue
        pending.append((i, rec))

    # Apply --limit
    if cfg.limit is not None:
        pending = pending[: cfg.limit]

    if not pending:
        logger.info("Nothing to process.")
        return

    logger.info("Processing %d records (concurrency=%d).", len(pending), cfg.concurrency)

    # Read templates
    system_tpl = cfg.prompts.system.read_text(encoding="utf-8")
    user_tpl = cfg.prompts.user.read_text(encoding="utf-8")

    counters = {"success": 0, "failure": 0}
    semaphore = asyncio.Semaphore(cfg.concurrency)
    lock = asyncio.Lock()

    # Ensure output dir exists
    cfg.io.output.parent.mkdir(parents=True, exist_ok=True)

    async with LLMClient(
        provider=cfg.provider,
        model=cfg.model,
        temperature=cfg.model_params.temperature,
        max_tokens=cfg.model_params.max_tokens,
        **cfg.model_params.extra,
    ) as client:
        with open(cfg.io.output, "a", encoding="utf-8") as out_file:
            tasks = [
                _process_record(
                    record=rec,
                    index=idx,
                    system_tpl=system_tpl,
                    user_tpl=user_tpl,
                    client=client,
                    cfg=cfg,
                    semaphore=semaphore,
                    lock=lock,
                    out_file=out_file,
                    counters=counters,
                )
                for idx, rec in pending
            ]
            await tqdm_asyncio.gather(*tasks, desc="Processing")

    print(
        f"\nDone. Success: {counters['success']}, "
        f"Failure: {counters['failure']}, "
        f"Total: {len(pending)}"
    )
