#!/usr/bin/env python3
"""Summarize local eval reports for one or more experiment run names.

The script accepts run directories either directly under outputs/ or under
outputs/reports/. It prints a compact TSV so runs can be compared without
copy-pasting several jq commands.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ABSTAIN_PAT = re.compile(
    r"^\s*cannot\s+answer\s*[,:\-]?\s*insufficient\s+evidence\.?\s*$",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_report_dir(outputs_root: Path, run_name: str) -> Path:
    candidates = [
        outputs_root / run_name,
        outputs_root / "reports" / run_name,
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in ("contract.json", "doc_verdicts.json", "conflict_type.json")):
            return candidate
    raise FileNotFoundError(f"Could not find reports for {run_name} under {outputs_root} or {outputs_root / 'reports'}")


def count_jsonl(path: Path) -> tuple[int | str, int | str, int | str]:
    if not path.exists():
        return "NA", "NA", "NA"
    rows = sentinel = think = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            raw = json.loads(line).get("raw", "")
            sentinel += int("[[END-OF-ANSWER]]" in raw)
            think += int("<think>" in raw and "</think>" in raw)
    return rows, sentinel, think


def is_abstain(raw: str) -> bool:
    if "</think>" in raw:
        tail = raw.split("</think>", 1)[1]
    else:
        tail = raw
    tail = tail.replace("[[END-OF-ANSWER]]", "")
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return False
    return lines[0] == "CANNOT ANSWER, INSUFFICIENT EVIDENCE" or bool(ABSTAIN_PAT.match(lines[0]))


def abstain_gold_accuracy(sanitized_path: Path, canon_path: Path) -> str:
    if not sanitized_path.exists() or not canon_path.exists():
        return "NA"
    canon: dict[str, bool] = {}
    with canon_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            expected = obj.get("expected_response") or {}
            if isinstance(expected.get("abstain"), bool):
                canon[obj["id"]] = expected["abstain"]
    total = correct = 0
    with sanitized_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id") not in canon:
                continue
            total += 1
            correct += int(is_abstain(obj.get("raw", "")) == canon[obj["id"]])
    return "NA" if total == 0 else round(100 * correct / total, 1)


def adjusted_contract(contract: dict[str, Any]) -> tuple[int | str, float | str]:
    ignored = {
        "abstain_violation_support_present",
        "abstain_with_partial_support_present",
    }
    if "ok_ignoring_abstain_evidence_violation" in contract:
        return (
            contract.get("ok_ignoring_abstain_evidence_violation", "NA"),
            contract.get("ok_ignoring_abstain_evidence_violation_rate_pct", "NA"),
        )
    if "ok_ignoring_abstain_support_violation" in contract:
        return (
            contract.get("ok_ignoring_abstain_support_violation", "NA"),
            contract.get("ok_ignoring_abstain_support_violation_rate_pct", "NA"),
        )
    ok = int(contract.get("ok_all_checks", 0))
    recovered = 0
    for item in contract.get("problems", []):
        remaining = [p for p in item.get("problems", []) if p not in ignored]
        if not remaining:
            recovered += 1
    total = int(contract.get("total", 0))
    adjusted = ok + recovered
    return adjusted, ("NA" if total == 0 else round(100 * adjusted / total, 1))


def top_problems(contract: dict[str, Any], n: int = 3) -> str:
    counts: dict[str, int] = {}
    for item in contract.get("problems", []):
        for problem in item.get("problems", []):
            counts[problem] = counts.get(problem, 0) + 1
    if not counts:
        return ""
    return ";".join(f"{problem}:{count}" for problem, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Experiment run names without .jsonl suffix")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--format", choices=["tsv", "markdown"], default="tsv")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    header = [
        "run",
        "rows",
        "sentinel",
        "think",
        "contract_ok",
        "contract_pct",
        "contract_adj_ok",
        "contract_adj_pct",
        "abstain",
        "abstain_gold_acc",
        "doc_micro",
        "doc_macro",
        "conf_acc",
        "conf_support",
        "final_abs_acc",
        "final_pred_abstain",
        "final_token_f1",
        "final_rouge_l",
        "final_cite_cov",
        "top_problems",
    ]
    table_rows = []

    for run in args.runs:
        report_dir = find_report_dir(outputs_root, run)
        contract = load_json(report_dir / "contract.json")
        doc = load_json(report_dir / "doc_verdicts.json")
        conflict = load_json(report_dir / "conflict_type.json")
        final_answer = optional_json(report_dir / "final_answer.json")
        row_count, sentinel, think = count_jsonl(outputs_root / f"{run}.sanitized.jsonl")
        abstain_gold = contract.get("abstain_gold") or {}
        adj_ok, adj_pct = adjusted_contract(contract)
        abstain_gold_acc = abstain_gold.get("accuracy_pct", "NA")
        if abstain_gold_acc == "NA":
            abstain_gold_acc = abstain_gold_accuracy(
                outputs_root / f"{run}.sanitized.jsonl",
                Path("data/splits/val_stagewise.jsonl"),
            )

        values = [
            run,
            row_count,
            sentinel,
            think,
            contract.get("ok_all_checks", "NA"),
            contract.get("ok_rate_pct", "NA"),
            adj_ok,
            adj_pct,
            contract.get("abstain_count", "NA"),
            abstain_gold_acc,
            doc.get("totals", {}).get("micro_accuracy_doc_level", "NA"),
            doc.get("overall", {}).get("macro_f1", "NA"),
            conflict.get("overall", {}).get("accuracy", "NA"),
            conflict.get("overall", {}).get("support", "NA"),
            final_answer.get("abstain", {}).get("accuracy_pct", "NA"),
            final_answer.get("abstain", {}).get("pred_abstain_count", "NA"),
            final_answer.get("lexical_overlap_non_abstain", {}).get("avg_token_f1", "NA"),
            final_answer.get("lexical_overlap_non_abstain", {}).get("avg_rouge_l_f1", "NA"),
            final_answer.get("citations", {}).get("avg_sentence_coverage", "NA"),
            top_problems(contract),
        ]
        table_rows.append([str(value) for value in values])

    if args.format == "markdown":
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join(["---"] * len(header)) + " |")
        for row in table_rows:
            print("| " + " | ".join(row) + " |")
    else:
        print("\t".join(header))
        for row in table_rows:
            print("\t".join(row))


if __name__ == "__main__":
    main()
