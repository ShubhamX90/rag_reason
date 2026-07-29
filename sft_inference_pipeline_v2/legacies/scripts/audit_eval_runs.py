#!/usr/bin/env python3
"""Create a per-id audit table for local eval runs.

The compact summary table is useful for comparing runs, but it hides whether
failures come from the same examples, from trace emergence, or from semantic
conflict mistakes. This script joins canonical labels, generations, and report
diagnostics into one CSV/JSONL table that is easier to sort and inspect.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


CONFLICT_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}

ABSTAIN_CANON = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
ABSTAIN_PAT = re.compile(
    r"^\s*cannot\s+answer\s*[,:\-]?\s*insufficient\s+evidence\.?\s*$",
    re.IGNORECASE,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
VERDICT_RE = re.compile(
    r"^\s*-\s*(d\d+)\s*:\s*(supports|partially supports|irrelevant)\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
CITE_RE = re.compile(r"\[d(\d+)\]")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_block(raw: str) -> str:
    m = THINK_RE.search(raw or "")
    return m.group(1) if m else ""


def extract_tail(raw: str) -> str:
    m = THINK_RE.search(raw or "")
    tail = raw[m.end() :] if m else raw or ""
    return tail.replace("[[END-OF-ANSWER]]", "").strip()


def extract_conflict(raw: str) -> str:
    block = extract_block(raw)
    for line in block.splitlines():
        line = line.strip()
        if line.lower().startswith("conflict type:"):
            label = line.split(":", 1)[1].strip()
            return label if label in CONFLICT_TYPES else "PRED_INVALID"
        for sep in (" - ", " — ", " – ", ":"):
            if sep in line:
                left = line.split(sep, 1)[0].strip()
                if left in CONFLICT_TYPES:
                    return left
    return "PRED_MISSING"


def extract_doc_verdicts(raw: str) -> dict[str, str]:
    block = extract_block(raw)
    verdicts: dict[str, str] = {}
    for match in VERDICT_RE.finditer(block):
        verdicts[match.group(1)] = match.group(2).lower()
    return verdicts


def is_abstain(raw: str) -> bool:
    tail = extract_tail(raw)
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return False
    return lines[0] == ABSTAIN_CANON or bool(ABSTAIN_PAT.match(lines[0]))


def sentence_split(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def citation_sentence_coverage(raw: str) -> float | str:
    if is_abstain(raw):
        return "NA"
    final = " ".join(line.strip() for line in extract_tail(raw).splitlines() if line.strip())
    sentences = sentence_split(final)
    if not sentences:
        return 0.0
    cited = sum(1 for sentence in sentences if CITE_RE.search(sentence))
    return round(cited / len(sentences), 4)


def final_preview(raw: str, limit: int = 180) -> str:
    tail = " ".join(line.strip() for line in extract_tail(raw).splitlines() if line.strip())
    return tail[:limit]


def report_problems(report_dir: Path) -> dict[str, list[str]]:
    contract = load_json(report_dir / "contract.json")
    problems: dict[str, list[str]] = {}
    for item in contract.get("problems", []):
        problems[item.get("id", "")] = item.get("problems", [])
    return problems


def final_answer_flags(report_dir: Path) -> dict[str, str]:
    final = load_json(report_dir / "final_answer.json")
    flags: dict[str, str] = {}
    abstain = final.get("abstain") or {}
    for cid in abstain.get("false_positive_ids", []):
        flags[cid] = "false_positive"
    for cid in abstain.get("false_negative_ids", []):
        flags[cid] = "false_negative"
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--canon-jsonl", default="data/splits/val_stagewise.jsonl")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-jsonl")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    canon_by_id = {row["id"]: row for row in read_jsonl(Path(args.canon_jsonl))}
    rows: list[dict[str, Any]] = []

    for run in args.runs:
        gen_path = outputs_root / f"{run}.sanitized.jsonl"
        report_dir = outputs_root / "reports" / run
        problems_by_id = report_problems(report_dir)
        final_flags = final_answer_flags(report_dir)

        for gen in read_jsonl(gen_path):
            cid = gen.get("id", "")
            raw = gen.get("raw", "")
            canon = canon_by_id.get(cid, {})
            gold_doc = {
                note.get("doc_id"): (note.get("verdict") or "").strip().lower()
                for note in canon.get("per_doc_notes") or []
            }
            pred_doc = extract_doc_verdicts(raw)
            doc_total = len(gold_doc)
            doc_correct = sum(1 for did, verdict in gold_doc.items() if pred_doc.get(did) == verdict)
            gold_conflict = canon.get("conflict_type", "")
            pred_conflict = extract_conflict(raw)
            gold_abstain = (canon.get("expected_response") or {}).get("abstain")
            pred_abstain = is_abstain(raw)
            rows.append(
                {
                    "run": run,
                    "id": cid,
                    "gold_conflict": gold_conflict,
                    "pred_conflict": pred_conflict,
                    "conflict_match": int(gold_conflict == pred_conflict),
                    "gold_abstain": gold_abstain,
                    "pred_abstain": pred_abstain,
                    "abstain_match": int(gold_abstain == pred_abstain) if isinstance(gold_abstain, bool) else "NA",
                    "final_abstain_error": final_flags.get(cid, ""),
                    "has_think": int("<think>" in raw and "</think>" in raw),
                    "has_sentinel": int("[[END-OF-ANSWER]]" in raw),
                    "doc_correct": doc_correct,
                    "doc_total": doc_total,
                    "doc_acc": round(doc_correct / doc_total, 4) if doc_total else "NA",
                    "pred_support_count": sum(1 for v in pred_doc.values() if v == "supports"),
                    "pred_partial_count": sum(1 for v in pred_doc.values() if v == "partially supports"),
                    "citation_sentence_coverage": citation_sentence_coverage(raw),
                    "problems": ";".join(problems_by_id.get(cid, [])),
                    "final_preview": final_preview(raw),
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.out_jsonl:
        out_jsonl = Path(args.out_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"rows": len(rows), "out_csv": str(out_csv), "out_jsonl": args.out_jsonl}, indent=2))


if __name__ == "__main__":
    main()
