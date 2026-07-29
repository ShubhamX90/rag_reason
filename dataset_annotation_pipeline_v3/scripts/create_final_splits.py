#!/usr/bin/env python3
"""
Create shared train/val/test splits for the retained stagewise annotation outputs.

The same ID split is applied to:
  - stagewise_multi/{stage1,stage2,stage3_final}.jsonl

Stratification uses a combined label:
  (refusal | nonrefusal) + normalized conflict_type

This keeps both refusal proportion and conflict-type distribution aligned
across splits as closely as possible while honoring exact target sizes.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGEWISE_STAGE1 = PROJECT_ROOT / "data" / "final_annotations" / "stagewise_multi" / "stage1.jsonl"
STAGEWISE_STAGE2 = PROJECT_ROOT / "data" / "final_annotations" / "stagewise_multi" / "stage2.jsonl"
STAGEWISE_STAGE3 = PROJECT_ROOT / "data" / "final_annotations" / "stagewise_multi" / "stage3_final.jsonl"


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_conflict_type(conflict_type: str) -> str:
    ct = (conflict_type or "").strip()
    mapping = {
        "No Conflict": "No conflict",
        "Complementary Information": "Complementary information",
        "Conflicting Opinions or Research Outcomes": "Conflicting opinions or research outcomes",
        "Conflicting Opinions and Research Outcomes": "Conflicting opinions or research outcomes",
        "Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
        "Conflicting opinions or research outcomes": "Conflicting opinions or research outcomes",
        "Outdated Information": "Conflict due to outdated information",
        "Conflict Due to Outdated Information": "Conflict due to outdated information",
        "Conflict due to outdated information": "Conflict due to outdated information",
        "Misinformation": "Conflict due to misinformation",
        "Conflict Due to Misinformation": "Conflict due to misinformation",
        "Conflict due to misinformation": "Conflict due to misinformation",
    }
    return mapping.get(ct, ct)


def is_refusal_record(row: Dict) -> bool:
    er = row.get("expected_response", {}) or {}
    return (
        er.get("abstain") is True
        and er.get("answer") == "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
    )


def strat_key(row: Dict) -> str:
    prefix = "refusal" if is_refusal_record(row) else "nonrefusal"
    return f"{prefix}__{normalize_conflict_type(row.get('conflict_type', ''))}"


def allocate_counts(
    strata_sizes: Dict[str, int],
    target_total: int,
    total_size: int,
    preserve_train_example: bool = True,
) -> Dict[str, int]:
    """
    Largest-remainder apportionment for one split.

    We allocate split counts per stratum to sum exactly to target_total.
    When preserve_train_example=True, we avoid allocating all examples from
    a stratum away from train when stratum size > 0.
    """
    ideals = {k: (v * target_total / total_size) for k, v in strata_sizes.items()}
    alloc = {k: int(ideals[k]) for k in strata_sizes}

    used = sum(alloc.values())
    remaining = target_total - used

    def room_left(k: str) -> int:
        limit = strata_sizes[k]
        if preserve_train_example and strata_sizes[k] > 0:
            limit = max(0, limit - 1)
        return max(0, limit - alloc[k])

    order = sorted(
        strata_sizes,
        key=lambda k: (ideals[k] - alloc[k], strata_sizes[k], k),
        reverse=True,
    )

    while remaining > 0:
        placed = False
        for k in order:
            if room_left(k) > 0:
                alloc[k] += 1
                remaining -= 1
                placed = True
                if remaining == 0:
                    break
        if not placed:
            raise RuntimeError(f"Could not allocate remaining {remaining} items")

    return alloc


def build_split_ids(
    rows: List[Dict],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> Dict[str, List[str]]:
    if train_n + val_n + test_n != len(rows):
        raise ValueError("Split counts must sum to dataset size")

    groups: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        groups[strat_key(row)].append(row["id"])

    rng = random.Random(seed)
    for ids in groups.values():
        rng.shuffle(ids)

    strata_sizes = {k: len(v) for k, v in groups.items()}

    val_alloc = allocate_counts(strata_sizes, val_n, len(rows), preserve_train_example=True)
    test_alloc = allocate_counts(strata_sizes, test_n, len(rows), preserve_train_example=True)

    # Ensure val + test never exhausts a stratum when possible.
    for k, size in strata_sizes.items():
        max_holdout = max(0, size - 1)
        holdout = val_alloc[k] + test_alloc[k]
        if holdout > max_holdout:
            overflow = holdout - max_holdout
            # Remove overflow from the split with the smaller remainder pressure: test first.
            dec_test = min(overflow, test_alloc[k])
            test_alloc[k] -= dec_test
            overflow -= dec_test
            if overflow:
                val_alloc[k] -= overflow

    # Re-balance exact totals after safety correction.
    def rebalance(alloc: Dict[str, int], target: int, other: Dict[str, int]) -> None:
        current = sum(alloc.values())
        if current == target:
            return
        ideals = {k: strata_sizes[k] * target / len(rows) for k in strata_sizes}
        if current < target:
            need = target - current
            order = sorted(
                strata_sizes,
                key=lambda k: (ideals[k] - alloc[k], strata_sizes[k], k),
                reverse=True,
            )
            while need > 0:
                placed = False
                for k in order:
                    if alloc[k] + other[k] < max(0, strata_sizes[k] - 1):
                        alloc[k] += 1
                        need -= 1
                        placed = True
                        if need == 0:
                            break
                if not placed:
                    raise RuntimeError("Could not rebalance split upward")
        else:
            extra = current - target
            order = sorted(
                strata_sizes,
                key=lambda k: (alloc[k] - ideals[k], alloc[k], k),
                reverse=True,
            )
            while extra > 0:
                removed = False
                for k in order:
                    if alloc[k] > 0:
                        alloc[k] -= 1
                        extra -= 1
                        removed = True
                        if extra == 0:
                            break
                if not removed:
                    raise RuntimeError("Could not rebalance split downward")

    rebalance(val_alloc, val_n, test_alloc)
    rebalance(test_alloc, test_n, val_alloc)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for key, ids in sorted(groups.items()):
        v = val_alloc[key]
        t = test_alloc[key]
        val_ids.extend(ids[:v])
        test_ids.extend(ids[v:v + t])
        train_ids.extend(ids[v + t:])

    if not (len(train_ids) == train_n and len(val_ids) == val_n and len(test_ids) == test_n):
        raise RuntimeError(
            f"Bad split sizes: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
        )

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def summarize(rows: List[Dict], ids: List[str]) -> Dict[str, Dict[str, int]]:
    keep = {row["id"] for row in rows if row["id"] in set(ids)}
    selected = [row for row in rows if row["id"] in keep]
    out = {
        "size": len(selected),
        "refusal_vs_nonrefusal": Counter(),
        "combined_strata": Counter(),
        "normalized_conflict_type": Counter(),
    }
    for row in selected:
        ref_key = "refusal" if is_refusal_record(row) else "nonrefusal"
        out["refusal_vs_nonrefusal"][ref_key] += 1
        out["combined_strata"][strat_key(row)] += 1
        out["normalized_conflict_type"][normalize_conflict_type(row.get("conflict_type", ""))] += 1
    return {
        "size": out["size"],
        "refusal_vs_nonrefusal": dict(sorted(out["refusal_vs_nonrefusal"].items())),
        "combined_strata": dict(sorted(out["combined_strata"].items())),
        "normalized_conflict_type": dict(sorted(out["normalized_conflict_type"].items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create shared stratified train/val/test splits for stagewise annotations"
    )
    ap.add_argument("--train", type=int, default=609)
    ap.add_argument("--val", type=int, default=49)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument(
        "--outdir",
        default="data/splits/92p5_7p5",
        help="Output directory for split manifests and JSONLs",
    )
    args = ap.parse_args()

    stage1_rows = read_jsonl(STAGEWISE_STAGE1)
    stage2_rows = read_jsonl(STAGEWISE_STAGE2)
    stage3_rows = read_jsonl(STAGEWISE_STAGE3)

    stage3_ids = {r["id"] for r in stage3_rows}
    if not (
        stage3_ids == {r["id"] for r in stage1_rows}
        == {r["id"] for r in stage2_rows}
    ):
        raise RuntimeError("Input files do not share the same ID set")

    split_ids = build_split_ids(stage3_rows, args.train, args.val, args.test, args.seed)

    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in split_ids.items():
        ids_path = outdir / f"{split_name}_ids.json"
        ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")

    split_sets = {k: set(v) for k, v in split_ids.items()}
    stagewise_dir = outdir / "stagewise_multi"

    def write_split(rows: List[Dict], split_name: str, path: Path) -> None:
        subset = [r for r in rows if r["id"] in split_sets[split_name]]
        write_jsonl(path, subset)

    for split_name in ("train", "val", "test"):
        write_split(stage1_rows, split_name, stagewise_dir / split_name / "stage1.jsonl")
        write_split(stage2_rows, split_name, stagewise_dir / split_name / "stage2.jsonl")
        write_split(stage3_rows, split_name, stagewise_dir / split_name / "stage3_final.jsonl")

    manifest = {
        "seed": args.seed,
        "counts": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "total": len(stage3_rows),
        },
        "notes": {
            "shared_ids_across_stagewise_files": True,
            "stratification_key": "is_refusal + normalized_conflict_type",
            "normalization_note": (
                "'Conflicting opinions and research outcomes' and "
                "'Conflicting opinions or research outcomes' are treated as one stratum"
            ),
        },
        "summaries": {
            split_name: summarize(stage3_rows, ids)
            for split_name, ids in split_ids.items()
        },
    }
    (outdir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
