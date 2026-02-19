"""
src/llm_client.py
=================
Unified LLM client supporting:
  - Anthropic API  : claude-sonnet-4-6 (default), claude-haiku-4-5, etc.
                     Supports sync, async, and Anthropic Batch API.
  - OpenAI API     : gpt-4o (default), gpt-4o-mini, gpt-5, etc.
                     Supports sync, async, and OpenAI Batch API.
  - OpenRouter API : qwen/qwen-2.5-72b-instruct (default), any provider/model slug.
                     Supports sync and async (no batch).

Usage:
    from src.llm_client import LLMClient, Provider

    client = LLMClient(provider=Provider.ANTHROPIC)
    client = LLMClient(provider=Provider.OPENAI, model="gpt-4o")
    client = LLMClient(provider=Provider.OPENROUTER, model="qwen/qwen-2.5-72b-instruct")
    # Any OpenRouter model slug works:
    client = LLMClient(provider=Provider.OPENROUTER, model="mistralai/mistral-7b-instruct")
    client = LLMClient(provider=Provider.OPENROUTER, model="meta-llama/llama-3.1-70b-instruct")

    # Sync
    text = client.complete(system="...", user="...", max_tokens=512)

    # Async
    text = await client.acomplete(system="...", user="...", max_tokens=512)

    # Batch (Anthropic or OpenAI only)
    batch_id = client.create_batch(requests=[{"custom_id": "q1", "system": ..., "user": ...}])
    results  = client.wait_batch(batch_id)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
#  Lazy imports
# ─────────────────────────────────────────────

def _anthropic():
    try:
        import anthropic as _a
        return _a
    except ImportError:
        raise ImportError("Run:  pip install anthropic")

def _openai():
    try:
        import openai as _o
        return _o
    except ImportError:
        raise ImportError("Run:  pip install openai")


# ─────────────────────────────────────────────
#  Enums & Defaults
# ─────────────────────────────────────────────

class Provider(str, Enum):
    ANTHROPIC  = "anthropic"
    OPENAI     = "openai"
    OPENROUTER = "openrouter"


ANTHROPIC_DEFAULT_MODEL  = "claude-sonnet-4-6"
OPENAI_DEFAULT_MODEL     = "gpt-4o"
OPENROUTER_DEFAULT_MODEL = "qwen/qwen-2.5-72b-instruct"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_BATCH_POLL_INTERVAL = 30
_BATCH_MAX_WAIT_H    = 24   # hours before TimeoutError


# ─────────────────────────────────────────────
#  OpenRouter model alias table
#
#  Allows shorthands like "qwen", "qwen-72b", etc.
#  Any string containing '/' that is NOT in this table is passed
#  through unchanged — so any valid OpenRouter slug works directly
#  without code changes (e.g. "mistralai/mistral-7b-instruct").
# ─────────────────────────────────────────────

_OPENROUTER_ALIASES: Dict[str, str] = {
    # ── Qwen 2.5 72B ──────────────────────────────────────────────
    "qwen":                          "qwen/qwen-2.5-72b-instruct",
    "qwen2.5":                       "qwen/qwen-2.5-72b-instruct",
    "qwen-72b":                      "qwen/qwen-2.5-72b-instruct",
    "qwen72b":                       "qwen/qwen-2.5-72b-instruct",
    "qwen2.5-72b":                   "qwen/qwen-2.5-72b-instruct",
    "qwen2.5-72b-instruct":          "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen2.5-72b-instruct":     "qwen/qwen-2.5-72b-instruct",  # old slug
    "qwen/qwen-2.5-72b-instruct":    "qwen/qwen-2.5-72b-instruct",  # canonical
    # ── Qwen 2.5 7B ───────────────────────────────────────────────
    "qwen-7b":                       "qwen/qwen-2.5-7b-instruct",
    "qwen7b":                        "qwen/qwen-2.5-7b-instruct",
    "qwen2.5-7b":                    "qwen/qwen-2.5-7b-instruct",
    "qwen2.5-7b-instruct":           "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen2.5-7b-instruct":      "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-7b-instruct":     "qwen/qwen-2.5-7b-instruct",
}


def resolve_openrouter_model(model: str) -> str:
    """
    Expand shorthand / legacy OpenRouter model names to their canonical slug.

    Rules
    -----
    1. Lowercased name found in alias table  →  use canonical slug.
    2. Name already contains '/'            →  pass through unchanged.
       This means ANY valid OpenRouter slug (e.g. "mistralai/mistral-7b-instruct",
       "meta-llama/llama-3.1-70b-instruct") works without touching this file.
    3. Bare name with no '/' and no alias   →  raise ValueError with guidance.
    """
    key = model.strip().lower()
    if key in _OPENROUTER_ALIASES:
        resolved = _OPENROUTER_ALIASES[key]
        if resolved != model:
            print(f"  [llm_client] OpenRouter alias: {model!r} → {resolved!r}")
        return resolved
    if "/" in model:
        return model   # already a valid slug
    raise ValueError(
        f"\n[llm_client] Unrecognised OpenRouter model shorthand: {model!r}\n"
        f"Use the full 'provider/model-name' slug from openrouter.ai/models, e.g.:\n"
        f"  qwen/qwen-2.5-72b-instruct\n"
        f"  mistralai/mistral-7b-instruct\n"
        f"  meta-llama/llama-3.1-70b-instruct\n"
        f"Built-in shorthands: {sorted(_OPENROUTER_ALIASES.keys())}"
    )


# ─────────────────────────────────────────────
#  API Key helpers
# ─────────────────────────────────────────────

def _load_key(env_var: str, file_name: str) -> str:
    val = os.getenv(env_var, "").strip()
    if val:
        return val
    path = os.path.expanduser(f"~/.{file_name}")
    if os.path.exists(path):
        return open(path).read().strip()
    raise RuntimeError(
        f"No API key found.\n"
        f"  Set the {env_var} environment variable, or\n"
        f"  create ~/.{file_name} with the key."
    )

def anthropic_key()  -> str: return _load_key("ANTHROPIC_API_KEY",  "anthropic_key")
def openai_key()     -> str: return _load_key("OPENAI_API_KEY",      "openai_key")
def openrouter_key() -> str: return _load_key("OPENROUTER_API_KEY",  "openrouter_key")


# ─────────────────────────────────────────────
#  LLMClient
# ─────────────────────────────────────────────

class LLMClient:
    """
    Provider-agnostic LLM client.

    Parameters
    ----------
    provider    : Provider.ANTHROPIC | Provider.OPENAI | Provider.OPENROUTER
    model       : Model name; omit to use the provider default.
    temperature : Generation temperature (default 0.0).
    max_retries : Retries per call on transient errors (default 3).
    """

    def __init__(
        self,
        provider: Provider = Provider.ANTHROPIC,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self.provider = Provider(provider)

        _defaults = {
            Provider.ANTHROPIC:  ANTHROPIC_DEFAULT_MODEL,
            Provider.OPENAI:     OPENAI_DEFAULT_MODEL,
            Provider.OPENROUTER: OPENROUTER_DEFAULT_MODEL,
        }
        _raw = model or _defaults[self.provider]

        # Normalise OpenRouter model names (alias expansion + slug validation)
        if self.provider == Provider.OPENROUTER:
            _raw = resolve_openrouter_model(_raw)

        self.model       = _raw
        self.temperature = temperature
        self.max_retries = max_retries
        self._sync_client:  Any = None
        self._async_client: Any = None

    async def __aenter__(self): return self
    async def __aexit__(self, *_):
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception:
                pass

    # ── Client factories ──────────────────────────────────────────

    def _get_sync(self) -> Any:
        if self._sync_client is None:
            if self.provider == Provider.ANTHROPIC:
                self._sync_client = _anthropic().Anthropic(api_key=anthropic_key())
            elif self.provider == Provider.OPENAI:
                self._sync_client = _openai().OpenAI(api_key=openai_key())
            else:  # OPENROUTER
                self._sync_client = _openai().OpenAI(
                    api_key=openrouter_key(), base_url=OPENROUTER_BASE_URL
                )
        return self._sync_client

    def _get_async(self) -> Any:
        if self._async_client is None:
            if self.provider == Provider.ANTHROPIC:
                self._async_client = _anthropic().AsyncAnthropic(api_key=anthropic_key())
            elif self.provider == Provider.OPENAI:
                self._async_client = _openai().AsyncOpenAI(api_key=openai_key())
            else:  # OPENROUTER
                self._async_client = _openai().AsyncOpenAI(
                    api_key=openrouter_key(), base_url=OPENROUTER_BASE_URL
                )
        return self._async_client

    # ── Sync completion ───────────────────────────────────────────

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        extra_headers: Optional[Dict] = None,
    ) -> str:
        """Blocking completion with exponential back-off retries."""
        backoff, last_err = 1.0, RuntimeError("Unknown error")
        for attempt in range(self.max_retries):
            try:
                return self._call_sync(system, user, max_tokens, extra_headers)
            except Exception as exc:
                last_err = exc
                status = getattr(exc, "status_code", None)
                # Non-retryable: bad request, auth, not found — fail immediately
                if status in (400, 401, 403, 404):
                    raise
                if attempt < self.max_retries - 1:
                    # 429 rate-limit: back off much longer
                    if status == 429:
                        backoff = max(backoff, 15.0)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
        raise last_err

    def _call_sync(
        self, system: str, user: str, max_tokens: int, extra_headers: Optional[Dict]
    ) -> str:
        client = self._get_sync()
        if self.provider == Provider.ANTHROPIC:
            kw: Dict[str, Any] = {}
            if extra_headers:
                kw["extra_headers"] = extra_headers
            resp = client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=self.temperature,
                system=system, messages=[{"role": "user", "content": user}], **kw,
            )
            return resp.content[0].text
        else:   # OPENAI or OPENROUTER (both use openai SDK)
            resp = client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

    # ── Async completion ──────────────────────────────────────────

    async def acomplete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        extra_headers: Optional[Dict] = None,
    ) -> str:
        """Non-blocking completion with exponential back-off retries."""
        backoff, last_err = 1.0, RuntimeError("Unknown error")
        for attempt in range(self.max_retries):
            try:
                return await self._call_async(system, user, max_tokens, extra_headers)
            except Exception as exc:
                last_err = exc
                status = getattr(exc, "status_code", None)
                # Non-retryable: bad request, auth, not found — fail immediately
                if status in (400, 401, 403, 404):
                    raise
                if attempt < self.max_retries - 1:
                    # 429 rate-limit: back off much longer
                    if status == 429:
                        backoff = max(backoff, 15.0)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
        raise last_err

    async def _call_async(
        self, system: str, user: str, max_tokens: int, extra_headers: Optional[Dict]
    ) -> str:
        client = self._get_async()
        if self.provider == Provider.ANTHROPIC:
            kw: Dict[str, Any] = {}
            if extra_headers:
                kw["extra_headers"] = extra_headers
            resp = await client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=self.temperature,
                system=system, messages=[{"role": "user", "content": user}], **kw,
            )
            return resp.content[0].text
        else:   # OPENAI or OPENROUTER
            resp = await client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

    # ── Batch API ─────────────────────────────────────────────────

    def create_batch(
        self,
        requests: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> str:
        """
        Submit a batch of annotation requests.

        Each dict in `requests` must contain:
            custom_id : str   unique identifier per request
            system    : str   system prompt text
            user      : str   user prompt text

        Returns the batch ID string (use with wait_batch / poll_batch).
        Supported providers: ANTHROPIC, OPENAI.
        OpenRouter does not have a native batch API — use async mode instead.
        """
        if self.provider == Provider.ANTHROPIC:
            return self._create_batch_anthropic(requests, max_tokens)
        elif self.provider == Provider.OPENAI:
            return self._create_batch_openai(requests, max_tokens)
        else:
            raise ValueError(
                "Batch mode is supported for Anthropic and OpenAI only.\n"
                "For OpenRouter use the async scripts instead."
            )

    def _create_batch_anthropic(
        self, requests: List[Dict[str, Any]], max_tokens: int
    ) -> str:
        client = self._get_sync()
        api_requests = [
            {
                "custom_id": req["custom_id"],
                "params": {
                    "model":       self.model,
                    "max_tokens":  max_tokens,
                    "temperature": self.temperature,
                    "system":      req["system"],
                    "messages":    [{"role": "user", "content": req["user"]}],
                },
            }
            for req in requests
        ]
        batch = client.beta.messages.batches.create(requests=api_requests)
        return batch.id

    def _create_batch_openai(
        self, requests: List[Dict[str, Any]], max_tokens: int
    ) -> str:
        """
        Upload a JSONL batch file then create an OpenAI Batch job.
        Uses the /v1/chat/completions endpoint with a 24h completion window.
        """
        client = self._get_sync()

        lines = [
            json.dumps({
                "custom_id": req["custom_id"],
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":       self.model,
                    "max_tokens":  max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": req["system"]},
                        {"role": "user",   "content": req["user"]},
                    ],
                },
            })
            for req in requests
        ]
        jsonl_bytes = "\n".join(lines).encode("utf-8")

        file_obj = client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    # ── Batch polling ─────────────────────────────────────────────

    def poll_batch(self, batch_id: str) -> Dict[str, Any]:
        """Non-blocking status check. Returns a normalised status dict."""
        if self.provider == Provider.ANTHROPIC:
            return self._poll_anthropic(batch_id)
        elif self.provider == Provider.OPENAI:
            return self._poll_openai(batch_id)
        else:
            raise ValueError("Batch polling not supported for OpenRouter.")

    def _poll_anthropic(self, batch_id: str) -> Dict[str, Any]:
        b = self._get_sync().beta.messages.batches.retrieve(batch_id)
        return {
            "id":                b.id,
            "processing_status": b.processing_status,
            "request_counts": {
                "processing": b.request_counts.processing,
                "succeeded":  b.request_counts.succeeded,
                "errored":    b.request_counts.errored,
                "canceled":   b.request_counts.canceled,
                "expired":    b.request_counts.expired,
            },
        }

    def _poll_openai(self, batch_id: str) -> Dict[str, Any]:
        b = self._get_sync().batches.retrieve(batch_id)
        rc = b.request_counts or {}
        return {
            "id":                b.id,
            "processing_status": b.status,
            "request_counts": {
                "total":     getattr(rc, "total",     0),
                "completed": getattr(rc, "completed", 0),
                "failed":    getattr(rc, "failed",    0),
            },
            "output_file_id": b.output_file_id,
            "error_file_id":  b.error_file_id,
        }

    # ── Batch waiting & result collection ────────────────────────

    def wait_batch(
        self,
        batch_id: str,
        poll_interval: int = _BATCH_POLL_INTERVAL,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Block until the batch finishes, then return results as:
            [{"custom_id": ..., "content": <str>, "error": <str|None>}, ...]
        """
        if self.provider == Provider.ANTHROPIC:
            return self._wait_anthropic(batch_id, poll_interval, verbose)
        elif self.provider == Provider.OPENAI:
            return self._wait_openai(batch_id, poll_interval, verbose)
        else:
            raise ValueError("Batch not supported for OpenRouter.")

    def _wait_anthropic(
        self, batch_id: str, poll_interval: int, verbose: bool
    ) -> List[Dict[str, Any]]:
        deadline = time.time() + _BATCH_MAX_WAIT_H * 3600
        while time.time() < deadline:
            status = self._poll_anthropic(batch_id)
            if verbose:
                c = status["request_counts"]
                print(
                    f"  [batch {batch_id[:14]}…] "
                    f"status={status['processing_status']}  "
                    f"done={c['succeeded'] + c['errored']}  "
                    f"processing={c['processing']}"
                )
            if status["processing_status"] == "ended":
                break
            if poll_interval > 0:
                time.sleep(poll_interval)
        else:
            raise TimeoutError(f"Batch {batch_id} did not complete within {_BATCH_MAX_WAIT_H}h")

        client  = self._get_sync()
        results = []
        for item in client.beta.messages.batches.results(batch_id):
            cid = item.custom_id
            if item.result.type == "succeeded":
                results.append({
                    "custom_id": cid,
                    "content":   item.result.message.content[0].text,
                    "error":     None,
                })
            else:
                results.append({
                    "custom_id": cid,
                    "content":   "",
                    "error":     str(getattr(item.result, "error", "unknown")),
                })
        return results

    def _wait_openai(
        self, batch_id: str, poll_interval: int, verbose: bool
    ) -> List[Dict[str, Any]]:
        client   = self._get_sync()
        deadline = time.time() + _BATCH_MAX_WAIT_H * 3600
        terminal = {"completed", "failed", "expired", "cancelled"}

        while time.time() < deadline:
            status = self._poll_openai(batch_id)
            if verbose:
                c = status["request_counts"]
                print(
                    f"  [batch {batch_id[:14]}…] "
                    f"status={status['processing_status']}  "
                    f"completed={c.get('completed', 0)}  "
                    f"failed={c.get('failed', 0)}"
                )
            if status["processing_status"] in terminal:
                break
            if poll_interval > 0:
                time.sleep(poll_interval)
        else:
            raise TimeoutError(f"Batch {batch_id} did not complete within {_BATCH_MAX_WAIT_H}h")

        output_file_id = status.get("output_file_id")
        results: List[Dict[str, Any]] = []
        if output_file_id:
            raw = client.files.content(output_file_id).text
            for line in raw.splitlines():
                if not line.strip():
                    continue
                obj  = json.loads(line)
                cid  = obj.get("custom_id", "")
                resp = obj.get("response", {})
                if resp.get("status_code") == 200:
                    content = resp["body"]["choices"][0]["message"]["content"] or ""
                    results.append({"custom_id": cid, "content": content, "error": None})
                else:
                    err = str(resp.get("error") or resp.get("body", {}).get("error", "unknown"))
                    results.append({"custom_id": cid, "content": "", "error": err})

        # Supplement with any error-file entries
        error_file_id = status.get("error_file_id")
        if error_file_id:
            existing_ids = {r["custom_id"] for r in results}
            err_raw = client.files.content(error_file_id).text
            for line in err_raw.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id", "")
                if cid not in existing_ids:
                    results.append({
                        "custom_id": cid,
                        "content":   "",
                        "error":     str(obj.get("error", "unknown")),
                    })
        return results

    # ── Async batch waiting (non-blocking) ───────────────────────

    async def await_batch(
        self,
        batch_id: str,
        poll_interval: int = _BATCH_POLL_INTERVAL,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Async version of wait_batch — yields control between polls."""
        if self.provider not in (Provider.ANTHROPIC, Provider.OPENAI):
            raise ValueError("Batch not supported for OpenRouter.")

        deadline = time.time() + _BATCH_MAX_WAIT_H * 3600
        terminal = {"ended", "completed", "failed", "expired", "cancelled"}

        while time.time() < deadline:
            status = self.poll_batch(batch_id)
            if verbose:
                c = status.get("request_counts", {})
                if self.provider == Provider.ANTHROPIC:
                    print(
                        f"  [batch {batch_id[:14]}…] "
                        f"status={status['processing_status']}  "
                        f"done={c.get('succeeded', 0) + c.get('errored', 0)}  "
                        f"processing={c.get('processing', 0)}"
                    )
                else:
                    print(
                        f"  [batch {batch_id[:14]}…] "
                        f"status={status['processing_status']}  "
                        f"completed={c.get('completed', 0)}  "
                        f"failed={c.get('failed', 0)}"
                    )
            if status["processing_status"] in terminal:
                break
            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(f"Batch {batch_id} did not complete within {_BATCH_MAX_WAIT_H}h")

        # Result collection is I/O but not async-native — run in executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: (
                self._wait_anthropic(batch_id, 0, False)
                if self.provider == Provider.ANTHROPIC
                else self._wait_openai(batch_id, 0, False)
            ),
        )