#!/usr/bin/env python3
"""
Build a deduplicated candidate-query pool for a CONFLICTS-style benchmark.

This script only chooses seed queries. It deliberately does not assign gold
conflict labels, because the CONFLICTS paper annotates conflict type after
retrieval rather than inheriting labels from source datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external_sources"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "benchmark_build"

SOURCE_ORDER = [
    "conflictingqa",
    "situatedqa_geo",
    "situatedqa_temp",
    "freshqa",
    "qacc",
]


def normalize_query(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def clean_json_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, dict):
        return {k: clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json_value(v) for v in value]
    return value


def clean_text_value(value: Any) -> str:
    value = clean_json_value(value)
    return "" if value is None else str(value)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_conflicts_exclusion_set() -> set[str]:
    exclusions: set[str] = set()
    for path in [
        EXTERNAL_ROOT / "rag_conflicts" / "conflicts.jsonl",
        PROJECT_ROOT / "data" / "normalized" / "conflicts_normalized.jsonl",
        PROJECT_ROOT / "data" / "raw" / "conflicts.jsonl",
    ]:
        for row in read_jsonl(path) or []:
            query = row.get("question") or row.get("query") or ""
            norm = normalize_query(query)
            if norm:
                exclusions.add(norm)
    return exclusions


def candidate(
    source_family: str,
    source_dataset: str,
    query: str,
    source_record_id: str,
    source_split: str = "",
    source_answer: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "candidate_id": stable_id(source_family, f"{source_dataset}|{source_record_id}|{query}"),
        "query": query.strip(),
        "query_norm": normalize_query(query),
        "source_family": source_family,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "source_split": source_split,
        "source_answer": clean_text_value(source_answer),
        "source_metadata": clean_json_value(metadata or {}),
    }


def load_freshqa() -> List[Dict[str, Any]]:
    path = EXTERNAL_ROOT / "freshqa" / "downloads" / "freshqa_2026-04-21.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header_idx = next(i for i, row in enumerate(rows) if row and row[0] == "id")
    header = rows[header_idx]
    out: List[Dict[str, Any]] = []
    for raw in rows[header_idx + 1:]:
        if not raw or not any(cell.strip() for cell in raw):
            continue
        rec = dict(zip(header, raw))
        answers = [
            rec.get(f"answer_{i}", "").strip()
            for i in range(10)
            if rec.get(f"answer_{i}", "").strip()
        ]
        out.append(candidate(
            source_family="freshqa",
            source_dataset="freshqa_2026-04-21",
            query=rec.get("question", ""),
            source_record_id=rec.get("id", ""),
            source_split=rec.get("split", ""),
            source_answer="; ".join(answers[:3]),
            metadata={
                "effective_year": rec.get("effective_year", ""),
                "next_review": rec.get("next_review", ""),
                "false_premise": rec.get("false_premise", ""),
                "fact_type": rec.get("fact_type", ""),
                "source": rec.get("source", ""),
            },
        ))
    return out


def load_situatedqa() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    qa_root = EXTERNAL_ROOT / "situatedqa" / "data" / "qa_data"
    for source_family in ["situatedqa_geo", "situatedqa_temp"]:
        short = "geo" if source_family.endswith("geo") else "temp"
        for split in ["train", "dev", "test"]:
            path = qa_root / f"{short}.{split}.jsonl"
            for idx, rec in enumerate(read_jsonl(path) or []):
                answer = rec.get("answer", "")
                if isinstance(answer, list):
                    answer = "; ".join(clean_text_value(x) for x in answer[:3])
                out.append(candidate(
                    source_family=source_family,
                    source_dataset=f"situatedqa_{short}",
                    query=rec.get("question", ""),
                    source_record_id=str(rec.get("id", idx)),
                    source_split=split,
                    source_answer=answer,
                    metadata={
                        "location": rec.get("location", ""),
                        "edited_question": rec.get("edited_question", ""),
                        "any_answer": rec.get("any_answer", ""),
                    },
                ))
    return out


def load_qacc() -> List[Dict[str, Any]]:
    path = EXTERNAL_ROOT / "qacc" / "data" / "ConflictQA_Dataset.json"
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        out.append(candidate(
            source_family="qacc",
            source_dataset="qacc_conflictqa",
            query=rec.get("question", ""),
            source_record_id=str(rec.get("annotation_task_id", idx)),
            source_split=rec.get("split", ""),
            source_answer=clean_text_value(rec.get("correctAnswer", "")),
            metadata={
                "first_answer": rec.get("firstAnswer", ""),
                "second_answer": rec.get("secondAnswer", ""),
                "third_answer": rec.get("thirdAnswer", ""),
                "fourth_answer_exists": rec.get("fourthAnswerExist", ""),
            },
        ))
    return out


def load_conflictingqa() -> List[Dict[str, Any]]:
    path = EXTERNAL_ROOT / "conflictingqa_rag_convincingness" / "downloads" / "data.pkl"
    if not path.exists():
        return []
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required to read the ConflictingQA pickle") from exc

    df = pd.read_pickle(path)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, rec in df.iterrows():
        query = str(rec.get("search_query", "")).strip()
        norm = normalize_query(query)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(candidate(
            source_family="conflictingqa",
            source_dataset="rag_convincingness_conflictingqa",
            query=query,
            source_record_id=norm,
            source_split="",
            source_answer="",
            metadata={"category": rec.get("category", "")},
        ))
    return out


def load_all_candidates() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(load_conflictingqa())
    rows.extend(load_situatedqa())
    rows.extend(load_freshqa())
    rows.extend(load_qacc())
    return [row for row in rows if row["query_norm"]]


def dedupe_candidates(rows: List[Dict[str, Any]], exclusions: set[str]) -> tuple[List[Dict[str, Any]], Counter]:
    stats: Counter = Counter()
    seen: set[str] = set()
    kept: List[Dict[str, Any]] = []
    for row in rows:
        norm = row["query_norm"]
        if norm in exclusions:
            stats["excluded_conflicts_overlap"] += 1
            continue
        if norm in seen:
            stats["excluded_duplicate_query"] += 1
            continue
        seen.add(norm)
        kept.append(row)
    stats["kept"] = len(kept)
    return kept, stats


def select_balanced(rows: List[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_family"]].append(row)
    for bucket in by_source.values():
        rng.shuffle(bucket)

    selected: List[Dict[str, Any]] = []
    used: set[str] = set()
    cursors = {source: 0 for source in SOURCE_ORDER}

    while len(selected) < target:
        made_progress = False
        source_counts = Counter(row["source_family"] for row in selected)
        active_sources = [
            source for source in SOURCE_ORDER
            if cursors.get(source, 0) < len(by_source.get(source, []))
        ]
        if not active_sources:
            break
        active_sources.sort(key=lambda source: (source_counts[source], SOURCE_ORDER.index(source)))
        for source in active_sources:
            bucket = by_source[source]
            while cursors[source] < len(bucket) and bucket[cursors[source]]["query_norm"] in used:
                cursors[source] += 1
            if cursors[source] >= len(bucket):
                continue
            row = bucket[cursors[source]]
            cursors[source] += 1
            used.add(row["query_norm"])
            selected.append(row)
            made_progress = True
            if len(selected) >= target:
                break
        if not made_progress:
            break

    for i, row in enumerate(selected, start=1):
        row["pool_rank"] = i
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deduped candidate queries for a CONFLICTS-style benchmark")
    ap.add_argument("--target", type=int, default=2000, help="Number of candidate queries to select")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default=str(OUTPUT_ROOT / "candidates" / "query_pool_2000.jsonl"))
    ap.add_argument("--all-output", default=str(OUTPUT_ROOT / "candidates" / "all_deduped_candidates.jsonl"))
    ap.add_argument("--manifest", default=str(OUTPUT_ROOT / "candidates" / "manifest_2000.json"))
    args = ap.parse_args()

    exclusions = load_conflicts_exclusion_set()
    all_rows = load_all_candidates()
    deduped, dedupe_stats = dedupe_candidates(all_rows, exclusions)
    selected = select_balanced(deduped, args.target, args.seed)

    write_jsonl(Path(args.all_output), deduped)
    write_jsonl(Path(args.output), selected)

    manifest = {
        "target": args.target,
        "seed": args.seed,
        "conflicts_exclusion_queries": len(exclusions),
        "raw_candidates": len(all_rows),
        "dedupe_stats": dict(dedupe_stats),
        "deduped_source_distribution": dict(Counter(row["source_family"] for row in deduped)),
        "selected_rows": len(selected),
        "selected_source_distribution": dict(Counter(row["source_family"] for row in selected)),
        "output": args.output,
        "all_output": args.all_output,
        "notes": [
            "These are seed queries only, not gold-labeled benchmark examples.",
            "Gold conflict labels must be assigned after CONFLICTS-style retrieval and high-confidence committee/human annotation.",
        ],
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote selected query pool: {args.output}")
    print(f"selected_rows={len(selected)}")
    print(f"selected_source_distribution={manifest['selected_source_distribution']}")
    print(f"dedupe_stats={manifest['dedupe_stats']}")


if __name__ == "__main__":
    main()
