#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/benchmark_local_committee_3judge")
MASTER_DIR = ROOT / "master_results"
CSV_PATH = MASTER_DIR / "cats_master_results_20260708.csv"
JSON_PATH = MASTER_DIR / "cats_master_results_20260708.json"
REPORT_JSON_PATH = MASTER_DIR / "gr_metric_verification_20260709.json"
REPORT_MD_PATH = MASTER_DIR / "gr_metric_verification_20260709.md"

TOL = 1e-12


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def compute_from_per_sample(samples: list[dict[str, Any]]) -> dict[str, float]:
    tp = sum(1 for s in samples if bool(s["pred_answered"]) and bool(s["gold_answerable"]))
    fp = sum(1 for s in samples if bool(s["pred_answered"]) and not bool(s["gold_answerable"]))
    fn = sum(1 for s in samples if not bool(s["pred_answered"]) and bool(s["gold_answerable"]))
    tn = sum(1 for s in samples if not bool(s["pred_answered"]) and not bool(s["gold_answerable"]))

    answer_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    answer_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    answer_f1 = (
        2 * answer_precision * answer_recall / (answer_precision + answer_recall)
        if (answer_precision + answer_recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / max(1, len(samples))

    refusal_tp = tn
    refusal_fp = fn
    refusal_fn = fp
    refusal_tn = tp
    refusal_precision = refusal_tp / (refusal_tp + refusal_fp) if (refusal_tp + refusal_fp) > 0 else 0.0
    refusal_recall = refusal_tp / (refusal_tp + refusal_fn) if (refusal_tp + refusal_fn) > 0 else 0.0
    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if (refusal_precision + refusal_recall) > 0
        else 0.0
    )

    return {
        "n": len(samples),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "answer_precision": answer_precision,
        "answer_recall": answer_recall,
        "answer_f1": answer_f1,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "accuracy": accuracy,
        "refusal_tn": refusal_tn,
    }


def load_master_csv() -> dict[str, dict[str, str]]:
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        return {row["source_relpath"]: row for row in reader}


def load_master_json() -> dict[str, dict[str, Any]]:
    rows = json.loads(JSON_PATH.read_text())
    return {row["source_relpath"]: row for row in rows}


def verify() -> dict[str, Any]:
    master_csv = load_master_csv()
    master_json = load_master_json()

    detail_paths = sorted(
        p for p in ROOT.rglob("detailed_results.json") if p.parent != MASTER_DIR
    )

    summary_mismatches: list[dict[str, Any]] = []
    csv_mismatches: list[dict[str, Any]] = []
    json_mismatches: list[dict[str, Any]] = []
    structural_issues: list[dict[str, Any]] = []

    for path in detail_paths:
        rel = str(path.relative_to(ROOT))
        data = json.loads(path.read_text())
        per_sample = data.get("per_sample")
        if not isinstance(per_sample, list):
            structural_issues.append({"source_relpath": rel, "issue": "missing_per_sample_list"})
            continue

        calc = compute_from_per_sample(per_sample)
        summary = data["summary"]
        gr = summary["gr_dataset_metrics"]
        overall = summary["conflict_overall"]

        expected_summary = {
            "precision": calc["answer_precision"],
            "recall": calc["answer_recall"],
            "f1": calc["answer_f1"],
            "accuracy": calc["accuracy"],
            "abstain_precision": calc["refusal_precision"],
            "abstain_recall": calc["refusal_recall"],
            "abstain_f1": calc["refusal_f1"],
            "tp": calc["tp"],
            "fp": calc["fp"],
            "fn": calc["fn"],
            "tn": calc["tn"],
        }
        for field, expected in expected_summary.items():
            actual = gr[field]
            ok = close(float(actual), float(expected)) if isinstance(expected, float) else actual == expected
            if not ok:
                summary_mismatches.append(
                    {
                        "source_relpath": rel,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )

        if overall.get("n") != calc["n"]:
            structural_issues.append(
                {
                    "source_relpath": rel,
                    "issue": "overall_n_mismatch",
                    "expected": calc["n"],
                    "actual": overall.get("n"),
                }
            )

        csv_row = master_csv.get(rel)
        if csv_row is None:
            csv_mismatches.append({"source_relpath": rel, "field": "row_presence", "expected": "present", "actual": "missing"})
        else:
            csv_expected = {
                "gr_answer_precision": calc["answer_precision"],
                "gr_answer_recall": calc["answer_recall"],
                "gr_answer_f1": calc["answer_f1"],
                "gr_refusal_precision": calc["refusal_precision"],
                "gr_refusal_recall": calc["refusal_recall"],
                "gr_refusal_f1": calc["refusal_f1"],
            }
            for field, expected in csv_expected.items():
                actual = float(csv_row[field])
                if not close(actual, expected):
                    csv_mismatches.append(
                        {
                            "source_relpath": rel,
                            "field": field,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

        json_row = master_json.get(rel)
        if json_row is None:
            json_mismatches.append({"source_relpath": rel, "field": "row_presence", "expected": "present", "actual": "missing"})
        else:
            json_expected = {
                "gr_answer_precision": calc["answer_precision"],
                "gr_answer_recall": calc["answer_recall"],
                "gr_answer_f1": calc["answer_f1"],
                "gr_refusal_precision": calc["refusal_precision"],
                "gr_refusal_recall": calc["refusal_recall"],
                "gr_refusal_f1": calc["refusal_f1"],
            }
            for field, expected in json_expected.items():
                actual = float(json_row[field])
                if not close(actual, expected):
                    json_mismatches.append(
                        {
                            "source_relpath": rel,
                            "field": field,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

    report = {
        "verified_source_file_count": len(detail_paths),
        "summary_mismatch_count": len(summary_mismatches),
        "csv_mismatch_count": len(csv_mismatches),
        "json_mismatch_count": len(json_mismatches),
        "structural_issue_count": len(structural_issues),
        "summary_mismatches_preview": summary_mismatches[:50],
        "csv_mismatches_preview": csv_mismatches[:50],
        "json_mismatches_preview": json_mismatches[:50],
        "structural_issues_preview": structural_issues[:50],
        "overall_ok": not summary_mismatches and not csv_mismatches and not json_mismatches and not structural_issues,
    }
    return report


def write_report(report: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# GR Metric Verification",
        "",
        f"- Verified source result files: `{report['verified_source_file_count']}`",
        f"- Summary mismatches: `{report['summary_mismatch_count']}`",
        f"- Master CSV mismatches: `{report['csv_mismatch_count']}`",
        f"- Master JSON mismatches: `{report['json_mismatch_count']}`",
        f"- Structural issues: `{report['structural_issue_count']}`",
        f"- Overall OK: `{report['overall_ok']}`",
        "",
        "This report independently recomputes all six GR-answer / GR-refusal metrics",
        "from each run's `per_sample` records and checks them against both the",
        "run-local `summary.gr_dataset_metrics` values and the master CSV/JSON files.",
        "",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    report = verify()
    write_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
