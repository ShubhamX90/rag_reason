#!/usr/bin/env python3
"""Probe a local OpenAI-compatible judge server with a tiny JSON task."""

import argparse
import json
import sys
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Endpoint base URL, e.g. http://node:8001/v1")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Add Qwen-style chat_template_kwargs.enable_thinking=false.",
    )
    args = parser.parse_args()

    payload = {
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
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            with httpx.Client(timeout=args.timeout) as client:
                models = client.get(f"{args.base_url}/models")
                print(f"GET /models status={models.status_code}")
                print(models.text[:1000])

                start = time.time()
                response = client.post(f"{args.base_url}/chat/completions", json=payload)
                elapsed = time.time() - start
                print(f"POST /chat/completions status={response.status_code} elapsed_s={elapsed:.2f}")
                print(response.text[:4000])
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print("content:", content)
                if "adherent" not in content:
                    raise RuntimeError("Response did not contain expected JSON-ish judge field")
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
