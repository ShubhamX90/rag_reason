#!/usr/bin/env python3
"""
Render a per-file benchmark local-committee config for Sharanga runs.

This keeps cache/output paths unique per prepared benchmark input file while
reusing the fixed 3-judge committee shape:

  - local/qwen3.5-397b-a17b        priority 6
  - local/mistral-small-4          priority 3
  - local/deepseek-r1-distill-32b  priority 2

Usage examples:

  python scripts/render_benchmark_local_committee_run_config.py \
    --mode collect \
    --judge qwen397 \
    --run-dir outputs/.../staged/qwen397_collect \
    --cache-dir outputs/.../response_cache \
    --base-url http://gpunodeX:8001/v1 \
    --output-config /tmp/run_config.yaml

  python scripts/render_benchmark_local_committee_run_config.py \
    --mode final \
    --run-dir outputs/.../final \
    --cache-dir outputs/.../response_cache \
    --output-config /tmp/run_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


FINAL_JUDGE_META = {
    "qwen397": dict(
        model_id="local/qwen3.5-397b-a17b",
        priority=6,
        max_tokens=400,
        disable_thinking=True,
    ),
    "mistral4": dict(
        model_id="local/mistral-small-4",
        priority=3,
        max_tokens=400,
        disable_thinking=False,
    ),
    "deepseek32": dict(
        model_id="local/deepseek-r1-distill-32b",
        priority=2,
        max_tokens=500,
        disable_thinking=False,
    ),
}


def judge_block(indent: str, *, model_id: str, base_url: str, priority: int, max_tokens: int, request_timeout: int, disable_thinking: bool = False) -> str:
    lines = [
        f'{indent}- model_id: "{model_id}"',
        f'{indent}  base_url: "{base_url}"',
        f"{indent}  priority: {priority}",
        f"{indent}  max_tokens: {max_tokens}",
        f"{indent}  request_timeout: {request_timeout}",
    ]
    if disable_thinking:
        lines.extend(
            [
                f"{indent}  extra_body:",
                f"{indent}    chat_template_kwargs:",
                f"{indent}      enable_thinking: false",
            ]
        )
    return "\n".join(lines)


def render_collect(args: argparse.Namespace) -> str:
    judge_map = {
        "qwen397": dict(
            model_id="local/qwen3.5-397b-a17b",
            priority=6,
            max_tokens=400,
            disable_thinking=True,
        ),
        "mistral4": dict(
            model_id="local/mistral-small-4",
            priority=3,
            max_tokens=400,
            disable_thinking=False,
        ),
        "deepseek32": dict(
            model_id="local/deepseek-r1-distill-32b",
            priority=2,
            max_tokens=500,
            disable_thinking=False,
        ),
    }
    meta = judge_map[args.judge]
    judges = judge_block(
        "      ",
        model_id=meta["model_id"],
        base_url=args.base_url,
        priority=meta["priority"],
        max_tokens=meta["max_tokens"],
        request_timeout=args.request_timeout,
        disable_thinking=meta["disable_thinking"],
    )
    return f"""outputs_dir: "{args.run_dir}"
report_md: "{args.run_dir}/eval_report.md"
detailed_results_json: "{args.run_dir}/detailed_results.json"

pipeline:
  batch_size: {args.batch_size}
  verbose: true

conflict_eval:
  enable: true
  use_judge_committee: true
  correct_refusal_full_credit: true
  require_cross_doc_verification: false
  max_claims_per_answer: 8
  allow_paraphrases: true

  committee:
    type: "local_openai"
    voting_strategy: "weighted_majority"
    max_concurrent_requests: {args.max_concurrent_requests}
    timeout_seconds: {args.request_timeout}
    response_cache_dir: "{args.cache_dir}"
    cache_mode: "read_write"
    judges:
{judges}
"""


def render_final(args: argparse.Namespace) -> str:
    base_url_map = {
        "qwen397": args.qwen_base_url,
        "mistral4": args.mistral_base_url,
        "deepseek32": args.deepseek_base_url,
    }
    judges_to_render = [name.strip() for name in args.final_judges.split(",") if name.strip()]
    invalid = [name for name in judges_to_render if name not in FINAL_JUDGE_META]
    if invalid:
        raise SystemExit(f"Invalid --final-judges entries: {', '.join(invalid)}")
    if not judges_to_render:
        raise SystemExit("--final-judges must include at least one judge")

    rendered_judges = []
    for name in judges_to_render:
        meta = FINAL_JUDGE_META[name]
        rendered_judges.append(
            judge_block(
                "      ",
                model_id=meta["model_id"],
                base_url=base_url_map[name],
                priority=meta["priority"],
                max_tokens=meta["max_tokens"],
                request_timeout=args.request_timeout,
                disable_thinking=meta["disable_thinking"],
            )
        )
    judges_block = "\n".join(rendered_judges)
    return f"""outputs_dir: "{args.run_dir}"
report_md: "{args.run_dir}/eval_report.md"
detailed_results_json: "{args.run_dir}/detailed_results.json"

pipeline:
  batch_size: {args.batch_size}
  verbose: true

conflict_eval:
  enable: true
  use_judge_committee: true
  correct_refusal_full_credit: true
  require_cross_doc_verification: false
  max_claims_per_answer: 8
  allow_paraphrases: true

  committee:
    type: "local_openai"
    voting_strategy: "weighted_majority"
    max_concurrent_requests: {args.max_concurrent_requests}
    timeout_seconds: {args.request_timeout}
    response_cache_dir: "{args.cache_dir}"
    cache_mode: "read_only"
    judges:
{judges_block}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "final"], required=True)
    parser.add_argument("--judge", choices=["qwen397", "mistral4", "deepseek32"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--mistral-base-url", default="http://127.0.0.1:8004/v1")
    parser.add_argument("--deepseek-base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--final-judges", default="qwen397,mistral4,deepseek32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-concurrent-requests", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()

    if args.mode == "collect":
        if not args.judge or not args.base_url:
            raise SystemExit("--mode collect requires --judge and --base-url")
        rendered = render_collect(args)
    else:
        rendered = render_final(args)

    out = Path(args.output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
