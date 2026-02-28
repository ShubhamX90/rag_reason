"""Retry loop, validation, and exception types."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Coroutine, Optional, Tuple

from .config import RetryConfig, ValidationConfig

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Output did not match expected format."""

    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


class PermanentAPIError(Exception):
    """Non-retryable HTTP error (e.g., 400, 401, 403, 404)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _extract_json(text: str) -> Optional[dict]:
    """Robustly extract the first JSON object from text."""
    # Strip markdown code fences
    fenced = re.sub(r"```(?:json)?\s*", "", text)
    fenced = fenced.replace("```", "").strip()

    # Try direct parse first
    for candidate in (fenced, text.strip()):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        blob = match.group(0)
        # Fix trailing commas
        blob = re.sub(r",\s*([}\]])", r"\1", blob)
        try:
            result = json.loads(blob)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def validate_output(raw: str, cfg: ValidationConfig) -> Optional[dict]:
    """
    Validate raw LLM output against cfg.
    Returns parsed dict (or None for mode=none).
    Raises ValidationError on failure.
    """
    if cfg.mode == "none":
        return None

    if cfg.mode == "regex":
        if not cfg.pattern:
            raise ValueError("validation.pattern must be set when mode=regex")
        if not re.search(cfg.pattern, raw):
            raise ValidationError(
                f"Output did not match pattern {cfg.pattern!r}", raw
            )
        return None

    if cfg.mode == "json_schema":
        import jsonschema

        parsed = _extract_json(raw)
        if parsed is None:
            raise ValidationError("Could not extract JSON from output", raw)
        if cfg.schema:
            try:
                jsonschema.validate(instance=parsed, schema=cfg.schema)
            except jsonschema.ValidationError as e:
                raise ValidationError(f"JSON schema validation failed: {e.message}", raw)
        return parsed

    raise ValueError(f"Unknown validation mode: {cfg.mode!r}")


async def with_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, str]],
    validation_cfg: ValidationConfig,
    retry_cfg: RetryConfig,
    record_id: Any = None,
) -> Tuple[str, Optional[dict], int]:
    """
    Retry wrapper.
    Returns (raw_output, parsed_output, attempt_count).
    """
    delay = retry_cfg.base_delay
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, retry_cfg.max_attempts + 1):
        try:
            raw = await coro_fn()
            parsed = validate_output(raw, validation_cfg)
            return raw, parsed, attempt

        except ValidationError as exc:
            last_exc = exc
            logger.warning(
                "Record %s attempt %d/%d — validation failed: %s",
                record_id, attempt, retry_cfg.max_attempts, exc,
            )

        except Exception as exc:
            last_exc = exc
            status_code = _get_status_code(exc)

            if status_code in retry_cfg.non_retryable_status:
                raise PermanentAPIError(
                    f"Non-retryable HTTP {status_code}: {exc}", status_code
                )

            effective_delay = max(delay, 15.0) if status_code == 429 else delay
            logger.warning(
                "Record %s attempt %d/%d — %s (status=%s). Retrying in %.1fs.",
                record_id, attempt, retry_cfg.max_attempts, exc, status_code, effective_delay,
            )

        if attempt < retry_cfg.max_attempts:
            await asyncio.sleep(delay)
            delay = min(delay * 2, retry_cfg.max_delay)

    raise last_exc


def _get_status_code(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from SDK exceptions."""
    for attr in ("status_code", "http_status", "status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    # Anthropic wraps in APIStatusError; OpenAI in APIError
    return None
