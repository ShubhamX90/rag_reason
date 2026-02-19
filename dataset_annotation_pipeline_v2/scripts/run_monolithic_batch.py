#!/usr/bin/env python3
"""
scripts/run_monolithic_batch.py
================================
Monolithic batch annotation using Anthropic's Message Batches API.

One batch request per query — cheapest approach for large corpora.
Combines all annotation stages into a single call per query.

Usage:
    python scripts/run_monolithic_batch.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/monolithic_outputs/monolithic_batch.jsonl

    # Resume existing batch:
    python scripts/run_monolithic_batch.py \\
        --input  data/normalized/conflicts_normalized.jsonl \\
        --output data/monolithic_outputs/monolithic_batch.jsonl \\
        --batch-id msgbatch_XXXXXXXXXXXXX
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_FILE    = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_client import LLMClient, Provider
from src.parsers    import parse_monolithic

SYSTEM_PROMPT_PATH    = PROJECT_ROOT / "prompts" / "system_monolithic.txt"
USER_PROMPT_PATH      = PROJECT_ROOT / "prompts" / "user_monolithic.txt"
MONOLITHIC_MAX_TOKENS = 3000


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_user_prompt(template: str, record: Dict[str, Any]) -> str:
    docs_slim = [
        {
            "doc_id":     d.get("doc_id", ""),
            "source_url": d.get("source_url", ""),
            "snippet":    d.get("snippet", ""),
            "timestamp":  d.get("timestamp", ""),
        }
        for d in record.get("retrieved_docs", [])
    ]
    return (
        template
        .replace("{query}",               record.get("query", ""))
        .replace("{conflict_type}",       record.get("conflict_type", ""))
        .replace("{retrieved_docs_json}", json.dumps(docs_slim, ensure_ascii=False, indent=2))
    )


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Monolithic batch annotation (Anthropic Batch API).")
    ap.add_argument("--input",         required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--model",         default=None)
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--limit",         type=int, default=None)
    ap.add_argument("--batch-id",      dest="batch_id",      default=None)
    ap.add_argument("--batch-id-file", dest="batch_id_file", default=None)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--system-prompt", dest="system_prompt", default=None)
    ap.add_argument("--user-prompt",   dest="user_prompt",   default=None)
    args = ap.parse_args()

    system_prompt = load_text(Path(args.system_prompt) if args.system_prompt else SYSTEM_PROMPT_PATH)
    user_template = load_text(Path(args.user_prompt)   if args.user_prompt   else USER_PROMPT_PATH)

    client = LLMClient(provider=Provider(args.provider), model=args.model or None)
    records = load_records(args.input)
    if args.limit:
        records = records[:args.limit]

    batch_id = args.batch_id
    if batch_id is None and args.batch_id_file and Path(args.batch_id_file).exists():
        batch_id = Path(args.batch_id_file).read_text().strip() or None

    if batch_id is None:
        reqs = [
            {
                "custom_id": record.get("id", f"rec_{i}"),
                "system":    system_prompt,
                "user":      fill_user_prompt(user_template, record),
            }
            for i, record in enumerate(records)
        ]
        print(f"📤 Submitting Monolithic batch | model={client.model} | records={len(reqs)}")
        batch_id = client.create_batch(reqs, max_tokens=MONOLITHIC_MAX_TOKENS)
        print(f"   Batch ID: {batch_id}")
        if args.batch_id_file:
            Path(args.batch_id_file).write_text(batch_id)
    else:
        print(f"⏩ Resuming batch: {batch_id}")

    print("⏳ Waiting for batch…")
    batch_results = client.wait_batch(batch_id, poll_interval=args.poll_interval, verbose=True)

    result_map = {r["custom_id"]: r for r in batch_results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    stats = {"answered": 0, "abstained": 0, "failed": 0}

    with open(args.output, "w", encoding="utf-8") as fout:
        for record in records:
            rec_id  = record.get("id", "")
            doc_ids = [d.get("doc_id", "") for d in record.get("retrieved_docs", [])]
            res     = result_map.get(rec_id)

            if res and not res.get("error"):
                parsed, errors = parse_monolithic(res["content"], expected_doc_ids=doc_ids)
                record["per_doc_notes"]             = parsed.get("per_doc_notes", [])
                record["conflict_reason"]           = parsed.get("conflict_reason", "")
                record["answerable_under_evidence"] = parsed.get("answerable_under_evidence", False)
                record["expected_response"]         = parsed.get("expected_response", {})
                record["think"]                     = parsed.get("think", "")
                record["_annotation_strategy"]      = "monolithic"
                if errors:
                    record["_monolithic_errors"] = errors
                if record["expected_response"].get("abstain"):
                    stats["abstained"] += 1
                else:
                    stats["answered"] += 1
            else:
                record["per_doc_notes"]             = []
                record["conflict_reason"]           = f"Batch error: {res.get('error') if res else 'missing'}"
                record["answerable_under_evidence"] = False
                record["expected_response"]         = {
                    "answer":        "CANNOT ANSWER, INSUFFICIENT EVIDENCE",
                    "evidence":      [],
                    "abstain":       True,
                    "abstain_reason": "Batch error or missing result.",
                }
                record["think"]                = ""
                record["_annotation_strategy"] = "monolithic"
                record["_batch_error"]         = True
                stats["failed"] += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Monolithic batch complete → {args.output}")
    print(f"   {stats}")


if __name__ == "__main__":
    main()
