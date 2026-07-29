#!/usr/bin/env python3
"""Probe a local OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

import httpx


def parse_extra_body(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit("--extra-body-json must decode to a JSON object")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe a local OpenAI-compatible chat endpoint")
    ap.add_argument("--base-url", required=True, help="Base URL including /v1, e.g. http://node:8001/v1")
    ap.add_argument("--model", required=True, help="Served model name")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--extra-body-json", default=None, help="Optional extra request body JSON object")
    ap.add_argument("--api-key", default="local-openai", help="Bearer token value if the server requires one")
    args = ap.parse_args()

    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Respond only with JSON."},
            {
                "role": "user",
                "content": (
                    'Return exactly {"adherent": true, "confidence": 0.9, '
                    '"rationale": "local endpoint ok"}.'
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
    }
    payload.update(parse_extra_body(args.extra_body_json))

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            with httpx.Client(base_url=args.base_url, headers=headers, timeout=args.timeout) as client:
                models = client.get("/models")
                print(f"GET /models status={models.status_code}")
                print(models.text[:1000])

                start = time.time()
                response = client.post("/chat/completions", json=payload)
                elapsed = time.time() - start
                print(f"POST /chat/completions status={response.status_code} elapsed_s={elapsed:.2f}")
                print(response.text[:4000])
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"].get("content") or ""
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                print("content:", content)
                parsed = json.loads(content)
                if parsed.get("adherent") is not True:
                    raise RuntimeError("Probe JSON did not contain adherent=true")
                return 0
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt}/{args.retries} failed: {exc}", file=sys.stderr)
            if attempt < args.retries:
                time.sleep(5)

    print(f"probe failed: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
