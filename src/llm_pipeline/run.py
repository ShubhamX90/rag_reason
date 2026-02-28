"""CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-pipeline",
        description="Process a JSONL file through an LLM with retry and validation.",
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--input", help="Override io.input path")
    parser.add_argument("--output", help="Override io.output path")
    parser.add_argument("--concurrency", type=int, help="Override concurrency")
    parser.add_argument("--limit", type=int, help="Process at most N records (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    # CLI overrides
    if args.input:
        from pathlib import Path
        cfg.io.input = Path(args.input).resolve()
    if args.output:
        from pathlib import Path
        cfg.io.output = Path(args.output).resolve()
    if args.concurrency is not None:
        cfg.concurrency = args.concurrency
    if args.limit is not None:
        cfg.limit = args.limit

    try:
        asyncio.run(run_pipeline(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved — re-run to resume.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        logging.exception("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
