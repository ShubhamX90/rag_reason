#!/usr/bin/env python3
"""Compile refreshed val-set export metrics into CSV and JSON artifacts.

This script maps every file under model_output_exports/val set to its matching
output JSONL via exact file hash, then reads the corresponding report bundle
under outputs/reports/<run>/.

Outputs:
- outputs/analysis/val_export_metrics_summary.csv
- outputs/analysis/val_export_metrics_overall_only.csv
- outputs/analysis/val_export_metrics_summary.json

By default this compiler excludes qwen3_32b rows because those exports are not
wanted in the downstream comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten(value, next_prefix))
        return flat
    if isinstance(obj, list):
        flat[f"{prefix}.count" if prefix else "count"] = len(obj)
        return flat
    flat[prefix] = obj
    return flat


def infer_run_stem(output_path: Path) -> str:
    name = output_path.name
    for suffix in (".sanitized.jsonl", ".raw.jsonl", ".resanitized.jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return output_path.stem


def prompt_label_from_export(export_path: Path) -> str:
    name = export_path.stem
    if name == "runtime_helper_prompt_outputs":
        return "runtime"
    if name == "strict_prompt_outputs":
        return "strict"
    if name == "minimal_prompt_outputs":
        return "minimal"
    return name


def build_row(
    root: Path,
    export_path: Path,
    matched_output: Path,
    identical_outputs: list[Path],
    report_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(report_dir / "contract.json")
    doc = load_json(report_dir / "doc_verdicts.json")
    conflict = load_json(report_dir / "conflict_type.json")
    final_answer = load_json(report_dir / "final_answer.json")

    export_rel = export_path.relative_to(root)
    parts = export_rel.parts
    task_family = parts[2]
    variant_family = parts[3]
    model = parts[4]
    prompt = prompt_label_from_export(export_path)

    summary_row = {
        "export_path": str(export_rel),
        "task_family": task_family,
        "variant_family": variant_family,
        "model": model,
        "prompt": prompt,
        "run_name": report_dir.name,
        "matched_output": str(matched_output.relative_to(root)),
        "identical_output_count": len(identical_outputs),
        "report_dir": str(report_dir.relative_to(root)),
        "total_examples": final_answer.get("total", contract.get("total")),
        "contract_ok_pct": contract.get("ok_rate_pct"),
        "citation_pass_pct": contract.get("citation_coverage", {}).get("pass_rate_pct"),
        "sentence_coverage": contract.get("citation_coverage", {}).get("avg_sentence_coverage"),
        "doc_micro_accuracy": doc.get("totals", {}).get("micro_accuracy_doc_level"),
        "doc_macro_f1": doc.get("overall", {}).get("macro_f1"),
        "conflict_accuracy": conflict.get("overall", {}).get("accuracy"),
        "conflict_support": conflict.get("overall", {}).get("support"),
        "abstain_accuracy": final_answer.get("abstain", {}).get("accuracy_pct"),
        "pred_abstain_count": final_answer.get("abstain", {}).get("pred_abstain_count"),
        "avg_token_f1": final_answer.get("lexical_overlap_non_abstain", {}).get("avg_token_f1"),
        "avg_rouge_l_f1": final_answer.get("lexical_overlap_non_abstain", {}).get("avg_rouge_l_f1"),
        "avg_citation_count": final_answer.get("citations", {}).get("avg_citation_count"),
        "avg_unique_citations": final_answer.get("citations", {}).get("avg_unique_citations"),
        "invalid_citation_rows": final_answer.get("citations", {}).get("rows_with_invalid_citations"),
        "think_count": final_answer.get("trace_presence", {}).get("think_count"),
        "sentinel_count": final_answer.get("trace_presence", {}).get("sentinel_count"),
    }

    detailed_entry = {
        **summary_row,
        "identical_outputs": [str(p.relative_to(root)) for p in identical_outputs],
        "metrics": {
            "contract": contract,
            "doc_verdicts": doc,
            "conflict_type": conflict,
            "final_answer": final_answer,
        },
        "flat_metrics": {
            **{f"contract.{k}": v for k, v in flatten(contract).items()},
            **{f"doc_verdicts.{k}": v for k, v in flatten(doc).items()},
            **{f"conflict_type.{k}": v for k, v in flatten(conflict).items()},
            **{f"final_answer.{k}": v for k, v in flatten(final_answer).items()},
        },
    }
    return summary_row, detailed_entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--csv-out",
        default="outputs/analysis/val_export_metrics_summary.csv",
    )
    parser.add_argument(
        "--overall-csv-out",
        default="outputs/analysis/val_export_metrics_overall_only.csv",
    )
    parser.add_argument(
        "--json-out",
        default="outputs/analysis/val_export_metrics_summary.json",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=["qwen3_32b"],
        help="Model directory names to exclude from the compiled outputs.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    exports = sorted((root / "model_output_exports" / "val set").glob("**/*.jsonl"))
    outputs = sorted((root / "outputs").glob("*.jsonl"))

    output_by_hash: dict[str, list[Path]] = {}
    for output_path in outputs:
        output_by_hash.setdefault(md5(output_path), []).append(output_path)

    rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []

    for export_path in exports:
        identical_outputs = sorted(output_by_hash[md5(export_path)])
        matched_output = next(
            (path for path in identical_outputs if path.name.endswith(".sanitized.jsonl")),
            identical_outputs[0],
        )
        report_dir = root / "outputs" / "reports" / infer_run_stem(matched_output)
        summary_row, detailed_entry = build_row(
            root=root,
            export_path=export_path,
            matched_output=matched_output,
            identical_outputs=identical_outputs,
            report_dir=report_dir,
        )
        if summary_row["model"] in set(args.exclude_models):
            continue
        rows.append(summary_row)
        detailed.append(detailed_entry)

    csv_out = (root / args.csv_out).resolve()
    overall_csv_out = (root / args.overall_csv_out).resolve()
    json_out = (root / args.json_out).resolve()
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    overall_csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall_fieldnames = [
        "export_path",
        "task_family",
        "variant_family",
        "model",
        "prompt",
        "run_name",
        "matched_output",
        "report_dir",
        "total_examples",
        "contract_ok_pct",
        "citation_pass_pct",
        "sentence_coverage",
        "doc_micro_accuracy",
        "doc_macro_f1",
        "conflict_accuracy",
        "abstain_accuracy",
        "avg_token_f1",
        "avg_rouge_l_f1",
        "avg_citation_count",
        "avg_unique_citations",
        "invalid_citation_rows",
        "pred_abstain_count",
        "think_count",
        "sentinel_count",
    ]
    with overall_csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in overall_fieldnames})

    payload = {
        "summary": {
            "total_exports": len(rows),
            "csv_path": str(csv_out.relative_to(root)),
            "overall_csv_path": str(overall_csv_out.relative_to(root)),
            "source_root": "model_output_exports/val set",
            "report_root": "outputs/reports",
        },
        "rows": detailed,
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(csv_out.relative_to(root))
    print(overall_csv_out.relative_to(root))
    print(json_out.relative_to(root))


if __name__ == "__main__":
    main()
