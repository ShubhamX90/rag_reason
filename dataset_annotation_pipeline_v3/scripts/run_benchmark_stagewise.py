#!/usr/bin/env python3
"""
Orchestrate stagewise benchmark preparation + Stage-1 + Stage-2.

This intentionally reuses the main stagewise annotation scripts so benchmark
runs inherit the same facilities as training-data runs:
  - resume behavior
  - cache support
  - per-stage cost report / ledger / cumulative report
  - concurrency controls

For local_openai staged cache collection, use the individual Stage-1/Stage-2
scripts as documented in configs/local_committee/README.md. Stage-2 collection
must run against the final read-only aggregated Stage-1 file, not against a
single-judge Stage-1 output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def default_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_path.stem
    return (
        output_dir / f"{stem}_prepared.jsonl",
        output_dir / f"{stem}_stage1.jsonl",
        output_dir / f"{stem}_stage2.jsonl",
    )


def run_cmd(cmd: list[str]) -> None:
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run benchmark prep with the stagewise committee pipeline. "
            "For staged local_openai cache collection, use the explicit "
            "workflow in configs/local_committee/README.md."
        )
    )
    ap.add_argument("--input", required=True, help="Raw benchmark JSONL")
    ap.add_argument("--output-dir", required=True, help="Directory for prepared/stage outputs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency-stage1", type=int, default=25)
    ap.add_argument("--concurrency-stage2", type=int, default=20)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--use-cache", action="store_true", default=False)
    ap.add_argument("--committee-backend", choices=["openrouter", "local_openai"],
                    default="openrouter")
    ap.add_argument("--committee-config", default=None,
                    help="JSON local_openai committee config passed to Stage-1 and Stage-2")
    ap.add_argument("--cache-mode", choices=["off", "read_write", "read_only", "write_only"],
                    default=None)
    ap.add_argument("--cache-dir", default=None,
                    help="Override response cache root directory")
    ap.add_argument("--stage2-system-prompt", default=None,
                    help="Optional Stage-2 system prompt override")
    ap.add_argument("--stage2-user-prompt", default=None,
                    help="Optional Stage-2 user prompt override")
    ap.add_argument("--prepare-only", action="store_true", default=False)
    ap.add_argument("--skip-prepare", action="store_true", default=False)
    args = ap.parse_args()

    if (
        args.committee_backend == "local_openai"
        and args.committee_config
        and "collect" in Path(args.committee_config).stem
        and not args.prepare_only
    ):
        raise SystemExit(
            "Do not run the full benchmark orchestrator with a one-model "
            "*_collect.json config. Collect Stage-1 cache per model, aggregate "
            "Stage-1, then collect Stage-2 cache per model from the final "
            "Stage-1 file. See configs/local_committee/README.md."
        )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prepared_path, stage1_path, stage2_path = default_paths(input_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        cmd = [
            PYTHON, "scripts/prepare_benchmark_stagewise_input.py",
            "--input", str(input_path),
            "--output", str(prepared_path),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run_cmd(cmd)
    else:
        print(f"↷ Skipping prepare step; using existing {prepared_path}")

    run_cmd([
        PYTHON, "scripts/validate_benchmark_gold.py",
        "--input", str(prepared_path),
        "--stage", "prepared",
    ])

    if args.prepare_only:
        print("✅ Prepare-only run complete.")
        return

    cmd_stage1 = [
        PYTHON, "scripts/run_stage1_multi_async.py",
        "--input", str(prepared_path),
        "--output", str(stage1_path),
        "--concurrency", str(args.concurrency_stage1),
        "--temperature", str(args.temperature),
        "--max-retries", str(args.max_retries),
        "--system-prompt", "prompts/system_stage1_benchmark.txt",
        "--user-prompt", "prompts/user_stage1_benchmark.txt",
    ]
    if args.limit:
        cmd_stage1 += ["--limit", str(args.limit)]
    if args.use_cache:
        cmd_stage1.append("--use-cache")
    cmd_stage1 += ["--committee-backend", args.committee_backend]
    if args.committee_config:
        cmd_stage1 += ["--committee-config", args.committee_config]
    if args.cache_mode:
        cmd_stage1 += ["--cache-mode", args.cache_mode]
    if args.cache_dir:
        cmd_stage1 += ["--cache-dir", args.cache_dir]
    run_cmd(cmd_stage1)

    run_cmd([
        PYTHON, "scripts/validate_benchmark_gold.py",
        "--input", str(stage1_path),
        "--stage", "stage1",
    ])

    cmd_stage2 = [
        PYTHON, "scripts/run_stage2_multi_async.py",
        "--input", str(stage1_path),
        "--output", str(stage2_path),
        "--benchmark-mode",
        "--concurrency", str(args.concurrency_stage2),
        "--temperature", str(args.temperature),
        "--max-retries", str(args.max_retries),
    ]
    if args.stage2_system_prompt:
        cmd_stage2 += ["--system-prompt", str(args.stage2_system_prompt)]
    if args.stage2_user_prompt:
        cmd_stage2 += ["--user-prompt", str(args.stage2_user_prompt)]
    if args.limit:
        cmd_stage2 += ["--limit", str(args.limit)]
    if args.use_cache:
        cmd_stage2.append("--use-cache")
    cmd_stage2 += ["--committee-backend", args.committee_backend]
    if args.committee_config:
        cmd_stage2 += ["--committee-config", args.committee_config]
    if args.cache_mode:
        cmd_stage2 += ["--cache-mode", args.cache_mode]
    if args.cache_dir:
        cmd_stage2 += ["--cache-dir", args.cache_dir]
    run_cmd(cmd_stage2)

    run_cmd([
        PYTHON, "scripts/validate_benchmark_gold.py",
        "--input", str(stage2_path),
        "--stage", "stage2",
    ])

    print("✅ Benchmark stagewise pipeline complete.")
    print(f"   prepared: {prepared_path}")
    print(f"   stage1:   {stage1_path}")
    print(f"   stage2:   {stage2_path}")


if __name__ == "__main__":
    main()
