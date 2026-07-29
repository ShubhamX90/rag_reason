"""
src/llm_cache.py
================
Small, disk-backed cache for raw LLM responses.

The cache key includes the exact prompt text, model, provider, temperature,
max_tokens, and request options that can affect the response. A cache hit
returns the same text that the live LLM call returned earlier, then the normal
parser/voting code runs unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / ".llm_cache"
CACHE_VERSION = "llm-response-cache-v1"


class CacheMissError(RuntimeError):
    """Raised when a read-only cache lookup cannot satisfy an LLM request."""


VALID_CACHE_MODES = {"off", "read_write", "read_only", "write_only"}


def normalize_cache_mode(cache_mode: str | None, cache_enabled: bool = False) -> str:
    """Normalize legacy boolean cache flags and explicit cache modes."""
    if cache_mode is None:
        return "read_write" if cache_enabled else "off"
    mode = str(cache_mode).strip().lower()
    if mode == "on":
        mode = "read_write"
    if mode not in VALID_CACHE_MODES:
        raise ValueError(
            f"Invalid cache mode {cache_mode!r}; expected one of "
            f"{', '.join(sorted(VALID_CACHE_MODES))}"
        )
    return mode


def cache_root() -> Path:
    override = os.getenv("RAG_REASON_LLM_CACHE_DIR", "").strip()
    return Path(override).expanduser() if override else DEFAULT_CACHE_DIR


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_cache_key(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    request_options: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "provider": provider,
        "model": model,
        "system": system,
        "user": user,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "request_options": request_options or {},
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def cache_path(cache_key: str, namespace: str = "default") -> Path:
    safe_namespace = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in (namespace or "default")
    )
    return cache_root() / safe_namespace / cache_key[:2] / f"{cache_key}.json"


def read_cached_response(cache_key: str, namespace: str = "default") -> Optional[str]:
    path = cache_path(cache_key, namespace)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        response = obj.get("response_text")
        return response if isinstance(response, str) else None
    except Exception:
        return None


def write_cached_response(
    *,
    cache_key: str,
    response_text: str,
    namespace: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = cache_path(cache_key, namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "response_text": response_text,
    }

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
