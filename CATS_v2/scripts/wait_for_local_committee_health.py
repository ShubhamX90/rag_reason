#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional
from urllib import request


TERMINAL_FAILURE = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "STOPPED",
    "SUSPENDED",
    "TIMEOUT",
}


@dataclass
class Endpoint:
    name: str
    base_url: str
    model: str
    job_id: Optional[str]
    disable_thinking: bool = False


def run_cmd(cmd: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def query_job_state(job_id: str) -> str:
    sq = run_cmd(["squeue", "-h", "-j", job_id, "-o", "%T"], check=False)
    if sq:
        return sq.strip().upper()

    sacct = run_cmd(
        ["sacct", "-X", "-j", job_id, "--parsable2", "--format=JobIDRaw,State", "-n"],
        check=False,
    )
    for line in sacct.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id_raw, state = parts[:2]
        if job_id_raw == job_id:
            return state.strip().upper().rstrip("+")
    return "UNKNOWN"


def _read_json_response(req: request.Request, timeout: float) -> dict:
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        print(body[:4000], flush=True)
        return json.loads(body)


def probe_endpoint(endpoint: Endpoint, timeout: float, retries: int) -> bool:
    payload = {
        "model": endpoint.model,
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
    if endpoint.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            models_req = request.Request(f"{endpoint.base_url}/models", method="GET")
            with request.urlopen(models_req, timeout=timeout) as resp:
                models_body = resp.read().decode("utf-8", errors="replace")
                print(f"GET /models status={resp.status}", flush=True)
                print(models_body[:1000], flush=True)

            body = json.dumps(payload).encode("utf-8")
            completion_req = request.Request(
                f"{endpoint.base_url}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            start = time.time()
            data = _read_json_response(completion_req, timeout=timeout)
            elapsed = time.time() - start
            print(f"POST /chat/completions status=200 elapsed_s={elapsed:.2f}", flush=True)

            content = data["choices"][0]["message"]["content"]
            print(f"content: {content}", flush=True)
            if "adherent" not in content:
                raise RuntimeError("Response did not contain expected JSON-ish judge field")
            return True
        except Exception as exc:
            last_error = exc
            print(f"attempt {attempt}/{retries} failed: {exc}", file=sys.stderr, flush=True)
            if attempt < retries:
                time.sleep(5)

    if last_error is not None:
        print(f"probe failed: {last_error}", file=sys.stderr, flush=True)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait until local committee judge endpoints are healthy.")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--probe-timeout", type=float, default=60.0)
    parser.add_argument("--probe-retries", type=int, default=1)
    parser.add_argument("--max-wait-seconds", type=int, default=86400)

    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-model", default="local/qwen3.5-397b-a17b")
    parser.add_argument("--qwen-job-id")

    parser.add_argument("--mistral-base-url")
    parser.add_argument("--mistral-model", default="local/mistral-small-4")
    parser.add_argument("--mistral-job-id")

    parser.add_argument("--deepseek-base-url")
    parser.add_argument("--deepseek-model", default="local/deepseek-r1-distill-32b")
    parser.add_argument("--deepseek-job-id")
    return parser.parse_args()


def build_endpoints(args: argparse.Namespace) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    if args.qwen_base_url:
        endpoints.append(
            Endpoint(
                name="qwen397",
                base_url=args.qwen_base_url,
                model=args.qwen_model,
                job_id=args.qwen_job_id,
                disable_thinking=True,
            )
        )
    if args.mistral_base_url:
        endpoints.append(
            Endpoint(
                name="mistral4",
                base_url=args.mistral_base_url,
                model=args.mistral_model,
                job_id=args.mistral_job_id,
            )
        )
    if args.deepseek_base_url:
        endpoints.append(
            Endpoint(
                name="deepseek32",
                base_url=args.deepseek_base_url,
                model=args.deepseek_model,
                job_id=args.deepseek_job_id,
            )
        )
    return endpoints


def main() -> int:
    args = parse_args()
    endpoints = build_endpoints(args)
    if not endpoints:
        raise SystemExit("At least one endpoint must be provided.")

    deadline = time.time() + args.max_wait_seconds
    pending = {endpoint.name: endpoint for endpoint in endpoints}

    while time.time() < deadline:
        progressed = False

        for name in list(pending):
            endpoint = pending[name]
            if endpoint.job_id:
                state = query_job_state(endpoint.job_id)
                print(f"[{endpoint.name}] job_id={endpoint.job_id} state={state}", flush=True)
                if state in TERMINAL_FAILURE:
                    print(f"[{endpoint.name}] server job entered terminal failure state={state}", file=sys.stderr, flush=True)
                    return 1
                if state not in {"RUNNING", "COMPLETING"}:
                    continue

            print(f"[{endpoint.name}] probing {endpoint.base_url} model={endpoint.model}", flush=True)
            if probe_endpoint(endpoint, timeout=args.probe_timeout, retries=args.probe_retries):
                print(f"[{endpoint.name}] healthy", flush=True)
                pending.pop(name, None)
                progressed = True
            else:
                print(f"[{endpoint.name}] probe failed; will retry", flush=True)

        if not pending:
            print("all_endpoints_healthy=1", flush=True)
            return 0

        if not progressed:
            print(f"waiting_for={','.join(sorted(pending))}", flush=True)
        time.sleep(args.poll_interval)

    print(f"health gate timed out after {args.max_wait_seconds}s", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
