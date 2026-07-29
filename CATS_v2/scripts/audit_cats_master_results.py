#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("outputs/benchmark_local_committee_3judge")
MASTER_DIR = ROOT / "master_results"
CSV_PATH = MASTER_DIR / "cats_master_results_20260708.csv"
JSON_PATH = MASTER_DIR / "cats_master_results_20260708.json"
MD_PATH = MASTER_DIR / "cats_master_results_20260708.md"
AUDIT_JSON_PATH = MASTER_DIR / "cats_master_results_20260709_audit.json"
AUDIT_MD_PATH = MASTER_DIR / "cats_master_results_20260709_audit.md"

CSV_FIELDS = [
    "collection",
    "category",
    "variant",
    "model_family",
    "eval_family",
    "prompt_mode",
    "run_type",
    "gr_answer_precision",
    "gr_answer_recall",
    "gr_answer_f1",
    "gr_refusal_precision",
    "gr_refusal_recall",
    "gr_refusal_f1",
    "gr_accuracy",
    "str",
    "fg",
    "behavior",
    "final_cats",
    "n",
    "behavior_n",
    "fg_n",
    "str_n",
    "correct_refusals",
    "source_relpath",
]

STANDARD_MODEL_ORDER = ["llama8b", "mistral7b", "qwen7b", "qwen32b"]
OTHER_MODEL_ORDER = ["llama", "mistral", "qwen"]


@dataclass(frozen=True)
class SourceRow:
    source_relpath: str
    collection: str
    category: str
    variant: str
    model_family: str
    eval_family: str
    prompt_mode: str
    run_type: str
    gr_answer_precision: float
    gr_answer_recall: float
    gr_answer_f1: float
    gr_refusal_precision: float
    gr_refusal_recall: float
    gr_refusal_f1: float
    gr_accuracy: float
    str_score: float
    fg: float
    behavior: float
    final_cats: float
    n: int
    behavior_n: int
    fg_n: int
    str_n: int
    correct_refusals: int

    def to_master_dict(self) -> dict[str, Any]:
        return {
            "source_relpath": self.source_relpath,
            "gr_answer_precision": self.gr_answer_precision,
            "gr_answer_recall": self.gr_answer_recall,
            "gr_answer_f1": self.gr_answer_f1,
            "gr_refusal_precision": self.gr_refusal_precision,
            "gr_refusal_recall": self.gr_refusal_recall,
            "gr_refusal_f1": self.gr_refusal_f1,
            "gr_accuracy": self.gr_accuracy,
            "str": self.str_score,
            "fg": self.fg,
            "behavior": self.behavior,
            "final_cats": self.final_cats,
            "n": self.n,
            "behavior_n": self.behavior_n,
            "fg_n": self.fg_n,
            "str_n": self.str_n,
            "correct_refusals": self.correct_refusals,
            "collection": self.collection,
            "category": self.category,
            "variant": self.variant,
            "model_family": self.model_family,
            "eval_family": self.eval_family,
            "prompt_mode": self.prompt_mode,
            "run_type": self.run_type,
        }

    def to_csv_dict(self) -> dict[str, str]:
        raw = self.to_master_dict()
        return {field: str(raw[field]) for field in CSV_FIELDS}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def infer_metadata(rel: Path) -> dict[str, str]:
    parts = rel.parts
    if parts[0] == "benchmark_set_all_modes":
        if parts[1] == "answer_only_sft":
            if len(parts) != 8 or parts[-2:] != ("final", "detailed_results.json"):
                raise ValueError(f"Unexpected answer_only path shape: {rel}")
            return {
                "collection": "benchmark_set_all_modes",
                "category": "answer_only_sft",
                "variant": "answer_only_sft",
                "model_family": parts[2],
                "eval_family": parts[3],
                "prompt_mode": parts[4],
                "run_type": parts[5],
            }
        if len(parts) != 7 or parts[-2:] != ("final", "detailed_results.json"):
            raise ValueError(f"Unexpected standard benchmark path shape: {rel}")
        return {
            "collection": "benchmark_set_all_modes",
            "category": "standard_benchmark",
            "variant": "standard",
            "model_family": parts[1],
            "eval_family": parts[2],
            "prompt_mode": parts[3],
            "run_type": parts[4],
        }
    if parts[0] == "other_techniques":
        if len(parts) != 5 or parts[-2:] != ("final", "detailed_results.json"):
            raise ValueError(f"Unexpected other-techniques path shape: {rel}")
        return {
            "collection": "other_techniques",
            "category": "other_techniques",
            "variant": parts[1],
            "model_family": parts[2],
            "eval_family": parts[1],
            "prompt_mode": "comparison",
            "run_type": "committee_eval",
        }
    raise ValueError(f"Unexpected result root: {rel}")


def load_source_rows() -> list[SourceRow]:
    rows: list[SourceRow] = []
    for path in sorted(ROOT.rglob("detailed_results.json")):
        if path.parent == MASTER_DIR:
            continue
        rel = path.relative_to(ROOT)
        data = load_json(path)
        summary = data["summary"]
        overall = summary["conflict_overall"]
        gr = summary["gr_dataset_metrics"]
        meta = infer_metadata(rel)
        rows.append(
            SourceRow(
                source_relpath=str(rel),
                collection=meta["collection"],
                category=meta["category"],
                variant=meta["variant"],
                model_family=meta["model_family"],
                eval_family=meta["eval_family"],
                prompt_mode=meta["prompt_mode"],
                run_type=meta["run_type"],
                gr_answer_precision=gr["precision"],
                gr_answer_recall=gr["recall"],
                gr_answer_f1=gr["f1"],
                gr_refusal_precision=gr["abstain_precision"],
                gr_refusal_recall=gr["abstain_recall"],
                gr_refusal_f1=gr["abstain_f1"],
                gr_accuracy=overall["gr_accuracy"],
                str_score=overall["single_truth_recall"],
                fg=overall["factual_grounding"],
                behavior=overall["behavior"],
                final_cats=overall["cats_score"],
                n=overall["n"],
                behavior_n=overall["behavior_n"],
                fg_n=overall["factual_grounding_n"],
                str_n=overall["single_truth_recall_n"],
                correct_refusals=overall["correct_refusals"],
            )
        )
    return rows


def sort_key(row: SourceRow) -> tuple[Any, ...]:
    return (
        row.collection,
        row.category,
        row.model_family,
        row.eval_family,
        row.prompt_mode,
        row.run_type,
        row.source_relpath,
    )


def compare_csv(expected_rows: list[SourceRow]) -> dict[str, Any]:
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
        csv_fields = reader.fieldnames

    expected_by_path = {r.source_relpath: r.to_csv_dict() for r in expected_rows}
    actual_by_path = {r["source_relpath"]: r for r in csv_rows}

    missing = sorted(set(expected_by_path) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - set(expected_by_path))

    duplicates: list[str] = []
    seen: set[str] = set()
    for row in csv_rows:
        rel = row["source_relpath"]
        if rel in seen:
            duplicates.append(rel)
        seen.add(rel)

    mismatches: list[dict[str, Any]] = []
    for rel in sorted(set(expected_by_path) & set(actual_by_path)):
        expected = expected_by_path[rel]
        actual = actual_by_path[rel]
        for field in CSV_FIELDS:
            if actual[field] != expected[field]:
                mismatches.append(
                    {
                        "source_relpath": rel,
                        "field": field,
                        "expected": expected[field],
                        "actual": actual[field],
                    }
                )

    return {
        "path": str(CSV_PATH),
        "row_count": len(csv_rows),
        "fieldnames_match_expected": csv_fields == CSV_FIELDS,
        "actual_fieldnames": csv_fields,
        "expected_fieldnames": CSV_FIELDS,
        "duplicate_source_relpaths": duplicates,
        "missing_in_csv": missing,
        "extra_in_csv": extra,
        "mismatch_count": len(mismatches),
        "mismatches_preview": mismatches[:50],
        "ok": not duplicates and not missing and not extra and not mismatches and csv_fields == CSV_FIELDS,
    }


def compare_master_json(expected_rows: list[SourceRow]) -> dict[str, Any]:
    actual_rows = load_json(JSON_PATH)
    expected_by_path = {r.source_relpath: r.to_master_dict() for r in expected_rows}
    actual_by_path = {r["source_relpath"]: r for r in actual_rows}

    missing = sorted(set(expected_by_path) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - set(expected_by_path))

    mismatches: list[dict[str, Any]] = []
    for rel in sorted(set(expected_by_path) & set(actual_by_path)):
        expected = expected_by_path[rel]
        actual = actual_by_path[rel]
        for field in expected:
            if actual[field] != expected[field]:
                mismatches.append(
                    {
                        "source_relpath": rel,
                        "field": field,
                        "expected": expected[field],
                        "actual": actual[field],
                    }
                )

    return {
        "path": str(JSON_PATH),
        "row_count": len(actual_rows),
        "missing_in_master_json": missing,
        "extra_in_master_json": extra,
        "mismatch_count": len(mismatches),
        "mismatches_preview": mismatches[:50],
        "ok": not missing and not extra and not mismatches,
    }


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_master_outputs(expected_rows: list[SourceRow]) -> None:
    csv_rows = [row.to_csv_dict() for row in expected_rows]
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    json_rows = [row.to_master_dict() for row in expected_rows]
    JSON_PATH.write_text(json.dumps(json_rows, indent=2) + "\n")
    MD_PATH.write_text(generate_markdown(expected_rows))


def generate_markdown(expected_rows: list[SourceRow]) -> str:
    standard = [r for r in expected_rows if r.category == "standard_benchmark"]
    answer_only = [r for r in expected_rows if r.category == "answer_only_sft"]
    other = [r for r in expected_rows if r.category == "other_techniques"]

    lines: list[str] = []
    lines.append("# CATS Master Results Matrix")
    lines.append("")
    lines.append(
        "This report is rebuilt directly from the synced `detailed_results.json` files under "
        "`outputs/benchmark_local_committee_3judge`, not from any derived CSV. In the standard "
        "benchmark section, each exact `model + eval family + prompt` configuration is shown with "
        "the `baseline` row immediately followed by the matching `sft` row whenever both are available locally."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Total synced result files included: `{len(expected_rows)}`")
    lines.append(f"- Standard benchmark result files: `{len(standard)}`")
    lines.append(f"- Answer-only SFT result files: `{len(answer_only)}`")
    lines.append(f"- Other-techniques result files: `{len(other)}`")

    standard_groups: dict[tuple[str, str, str], list[SourceRow]] = {}
    for row in standard:
        standard_groups.setdefault((row.model_family, row.eval_family, row.prompt_mode), []).append(row)
    paired = sum(
        1
        for rows in standard_groups.values()
        if {r.run_type for r in rows} == {"baseline", "sft"}
    )
    baseline_only = sum(1 for rows in standard_groups.values() if {r.run_type for r in rows} == {"baseline"})
    sft_only = sum(1 for rows in standard_groups.values() if {r.run_type for r in rows} == {"sft"})
    lines.append(f"- Complete standard benchmark baseline+SFT pairs: `{paired}`")
    lines.append(f"- Standard benchmark baseline-only configurations still present locally: `{baseline_only}`")
    lines.append(f"- Standard benchmark SFT-only configurations still present locally: `{sft_only}`")
    lines.append("")
    lines.append("## Model Distribution")
    lines.append("")
    lines.append("| Model | Synced result files |")
    lines.append("| --- | ---: |")
    model_counts: dict[str, int] = {}
    for row in expected_rows:
        model_counts[row.model_family] = model_counts.get(row.model_family, 0) + 1
    for model in STANDARD_MODEL_ORDER + OTHER_MODEL_ORDER:
        lines.append(f"| {model} | {model_counts.get(model, 0)} |")
    lines.append("")
    lines.append("## Standard Benchmark Matrix")
    lines.append("")
    lines.append(
        "Within each model table below, rows are ordered by `eval family`, then `prompt`, and then "
        "by `run`, with `baseline` always shown before `sft`. The `Delta vs baseline` column is "
        "populated only on the `sft` row, so it is visually obvious which row is the baseline and "
        "which row is the SFT counterpart."
    )
    lines.append("")

    for model in STANDARD_MODEL_ORDER:
        model_rows = [r for r in standard if r.model_family == model]
        if not model_rows:
            continue
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS | Delta vs baseline |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        grouped: dict[tuple[str, str], dict[str, SourceRow]] = {}
        for row in model_rows:
            grouped.setdefault((row.eval_family, row.prompt_mode), {})[row.run_type] = row

        for eval_family, prompt_mode in sorted(grouped):
            runs = grouped[(eval_family, prompt_mode)]
            baseline = runs.get("baseline")
            sft = runs.get("sft")
            if baseline:
                lines.append(
                    f"| {eval_family} | {prompt_mode} | baseline | {format_float(baseline.gr_answer_precision)} | "
                    f"{format_float(baseline.gr_answer_recall)} | {format_float(baseline.gr_answer_f1)} | "
                    f"{format_float(baseline.gr_refusal_precision)} | {format_float(baseline.gr_refusal_recall)} | "
                    f"{format_float(baseline.gr_refusal_f1)} | "
                    f"{format_float(baseline.str_score)} | {format_float(baseline.fg)} | "
                    f"{format_float(baseline.behavior)} | {format_float(baseline.final_cats)} | — |"
                )
            if sft:
                delta = sft.final_cats - (baseline.final_cats if baseline else 0.0)
                lines.append(
                    f"| {eval_family} | {prompt_mode} | sft | {format_float(sft.gr_answer_precision)} | "
                    f"{format_float(sft.gr_answer_recall)} | {format_float(sft.gr_answer_f1)} | "
                    f"{format_float(sft.gr_refusal_precision)} | {format_float(sft.gr_refusal_recall)} | "
                    f"{format_float(sft.gr_refusal_f1)} | "
                    f"{format_float(sft.str_score)} | {format_float(sft.fg)} | "
                    f"{format_float(sft.behavior)} | {format_float(sft.final_cats)} | {format_float(delta)} |"
                )
        lines.append("")

    lines.append("## Answer-only SFT")
    lines.append("")
    lines.append(
        "These six runs are methodologically distinct from the standard benchmark family, so they "
        "are kept in their own section. All of them are `sft` runs."
    )
    lines.append("")
    lines.append("| Model | Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(answer_only, key=lambda r: (r.model_family, r.eval_family, r.prompt_mode, r.run_type)):
        lines.append(
            f"| {row.model_family} | {row.eval_family} | {row.prompt_mode} | {row.run_type} | "
            f"{format_float(row.gr_answer_precision)} | {format_float(row.gr_answer_recall)} | {format_float(row.gr_answer_f1)} | "
            f"{format_float(row.gr_refusal_precision)} | {format_float(row.gr_refusal_recall)} | {format_float(row.gr_refusal_f1)} | "
            f"{format_float(row.str_score)} | {format_float(row.fg)} | {format_float(row.behavior)} | "
            f"{format_float(row.final_cats)} |"
        )
    lines.append("")
    lines.append("## Other Techniques")
    lines.append("")
    lines.append(
        "These rows summarize the currently synced `CoN` and `CoT fewshot` comparisons. These are "
        "committee-evaluated comparison runs, not baseline/SFT prompt-family pairs."
    )
    lines.append("")
    lines.append("| Model | Technique | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Final CATS |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(
        other,
        key=lambda r: (OTHER_MODEL_ORDER.index(r.model_family), r.variant, r.run_type),
    ):
        lines.append(
            f"| {row.model_family} | {row.variant} | {row.run_type} | {format_float(row.gr_answer_precision)} | "
            f"{format_float(row.gr_answer_recall)} | {format_float(row.gr_answer_f1)} | "
            f"{format_float(row.gr_refusal_precision)} | {format_float(row.gr_refusal_recall)} | {format_float(row.gr_refusal_f1)} | "
            f"{format_float(row.str_score)} | "
            f"{format_float(row.fg)} | {format_float(row.behavior)} | {format_float(row.final_cats)} |"
        )
    lines.append("")
    lines.append("## Metric Notes")
    lines.append("")
    lines.append("- `GR-answer Precision`, `GR-answer Recall`, and `GR-answer F1` are read directly from `summary.gr_dataset_metrics.{precision, recall, f1}` in each synced `detailed_results.json`.")
    lines.append("- `GR-refusal Precision`, `GR-refusal Recall`, and `GR-refusal F1` are read directly from `summary.gr_dataset_metrics.{abstain_precision, abstain_recall, abstain_f1}` in each synced `detailed_results.json`.")
    lines.append("- The two GR families come from the same dataset-level answer/refuse confusion table; the answer family treats `answered` as positive, while the refusal family treats `refused` as positive.")
    lines.append("- `STR`, `FG`, `Behavior`, and `Final CATS` are read from `summary.conflict_overall`.")
    lines.append("- This Markdown was regenerated directly from the synced JSON result files, not from the derived `cats_master_results_20260708.csv` file.")
    lines.append("- `Delta vs baseline` is shown only on the `sft` row and is computed as `sft Final CATS - baseline Final CATS`.")
    lines.append("")
    return "\n".join(lines)


def compare_markdown(expected_rows: list[SourceRow]) -> dict[str, Any]:
    actual = MD_PATH.read_text()
    expected = generate_markdown(expected_rows)
    if actual == expected:
        return {
            "path": str(MD_PATH),
            "exact_match": True,
            "first_difference_line": None,
            "actual_line": None,
            "expected_line": None,
            "ok": True,
        }

    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    if actual_lines == expected_lines:
        return {
            "path": str(MD_PATH),
            "exact_match": False,
            "first_difference_line": None,
            "actual_line": None,
            "expected_line": None,
            "note": "Line content matches exactly; only EOF newline/byte-level formatting differs.",
            "ok": True,
        }

    max_len = max(len(actual_lines), len(expected_lines))
    diff_line = None
    for idx in range(max_len):
        a = actual_lines[idx] if idx < len(actual_lines) else None
        e = expected_lines[idx] if idx < len(expected_lines) else None
        if a != e:
            diff_line = idx + 1
            return {
                "path": str(MD_PATH),
                "exact_match": False,
                "first_difference_line": diff_line,
                "actual_line": a,
                "expected_line": e,
                "ok": False,
            }
    return {
        "path": str(MD_PATH),
        "exact_match": False,
        "first_difference_line": None,
        "actual_line": None,
        "expected_line": None,
        "ok": False,
    }


def audit_source_inventory(expected_rows: list[SourceRow]) -> dict[str, Any]:
    detailed = sorted(ROOT.rglob("detailed_results.json"))
    detailed = [p for p in detailed if p.parent != MASTER_DIR]
    missing_siblings: list[dict[str, str]] = []
    for path in detailed:
        for sibling in ("eval_report.md", "run_config.yaml"):
            if not (path.parent / sibling).exists():
                missing_siblings.append(
                    {
                        "source_relpath": str(path.relative_to(ROOT)),
                        "missing_sibling": sibling,
                    }
                )

    combo_seen: dict[tuple[str, ...], int] = {}
    for row in expected_rows:
        if row.category == "standard_benchmark":
            key = (
                row.category,
                row.model_family,
                row.eval_family,
                row.prompt_mode,
                row.run_type,
            )
        elif row.category == "answer_only_sft":
            key = (
                row.category,
                row.model_family,
                row.eval_family,
                row.prompt_mode,
                row.run_type,
            )
        else:
            key = (
                row.category,
                row.variant,
                row.model_family,
                row.run_type,
            )
        combo_seen[key] = combo_seen.get(key, 0) + 1

    non_unique_keys = [
        {"key": key, "count": count}
        for key, count in sorted(combo_seen.items(), key=lambda item: item[0])
        if count != 1
    ]

    parse_failures: list[str] = []
    for path in detailed:
        try:
            load_json(path)
        except Exception as exc:  # pragma: no cover - audit only
            parse_failures.append(f"{path.relative_to(ROOT)} :: {exc}")

    return {
        "source_detailed_results_count": len(detailed),
        "source_relpaths_unique": len({r.source_relpath for r in expected_rows}) == len(expected_rows),
        "json_parse_failures": parse_failures,
        "missing_siblings": missing_siblings,
        "non_unique_configuration_keys": non_unique_keys,
        "ok": not parse_failures and not missing_siblings and not non_unique_keys,
    }


def summarize_counts(expected_rows: list[SourceRow]) -> dict[str, Any]:
    counts_by_category: dict[str, int] = {}
    counts_by_collection: dict[str, int] = {}
    counts_by_run_type: dict[str, int] = {}
    counts_by_model: dict[str, int] = {}
    for row in expected_rows:
        counts_by_category[row.category] = counts_by_category.get(row.category, 0) + 1
        counts_by_collection[row.collection] = counts_by_collection.get(row.collection, 0) + 1
        counts_by_run_type[row.run_type] = counts_by_run_type.get(row.run_type, 0) + 1
        counts_by_model[row.model_family] = counts_by_model.get(row.model_family, 0) + 1
    return {
        "total_rows": len(expected_rows),
        "counts_by_category": counts_by_category,
        "counts_by_collection": counts_by_collection,
        "counts_by_run_type": counts_by_run_type,
        "counts_by_model": counts_by_model,
    }


def write_audit_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# CATS Master Results Audit",
        "",
        f"- Source result files audited: `{report['inventory']['source_detailed_results_count']}`",
        f"- CSV rows audited: `{report['csv']['row_count']}`",
        f"- Master JSON rows audited: `{report['master_json']['row_count']}`",
        f"- CSV OK: `{report['csv']['ok']}`",
        f"- Master JSON OK: `{report['master_json']['ok']}`",
        f"- Markdown OK: `{report['markdown']['ok']}`",
        f"- Overall OK: `{report['overall_ok']}`",
        "",
        "## Coverage Counts",
        "",
        f"- Total rows: `{report['counts']['total_rows']}`",
        f"- By category: `{json.dumps(report['counts']['counts_by_category'], sort_keys=True)}`",
        f"- By collection: `{json.dumps(report['counts']['counts_by_collection'], sort_keys=True)}`",
        f"- By run type: `{json.dumps(report['counts']['counts_by_run_type'], sort_keys=True)}`",
        f"- By model: `{json.dumps(report['counts']['counts_by_model'], sort_keys=True)}`",
        "",
        "## Notes",
        "",
        f"- CSV duplicate relpaths: `{len(report['csv']['duplicate_source_relpaths'])}`",
        f"- CSV missing rows: `{len(report['csv']['missing_in_csv'])}`",
        f"- CSV extra rows: `{len(report['csv']['extra_in_csv'])}`",
        f"- CSV mismatches: `{report['csv']['mismatch_count']}`",
        f"- Master JSON missing rows: `{len(report['master_json']['missing_in_master_json'])}`",
        f"- Master JSON extra rows: `{len(report['master_json']['extra_in_master_json'])}`",
        f"- Master JSON mismatches: `{report['master_json']['mismatch_count']}`",
        f"- Markdown exact match: `{report['markdown']['exact_match']}`",
        "",
    ]
    AUDIT_MD_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rewrite-master",
        action="store_true",
        help="Rewrite the master CSV/JSON/Markdown from synced detailed_results.json files before auditing.",
    )
    args = parser.parse_args()

    expected_rows = sorted(load_source_rows(), key=sort_key)
    if args.rewrite_master:
        write_master_outputs(expected_rows)
    inventory = audit_source_inventory(expected_rows)
    csv_report = compare_csv(expected_rows)
    json_report = compare_master_json(expected_rows)
    markdown_report = compare_markdown(expected_rows)
    counts = summarize_counts(expected_rows)

    report = {
        "inventory": inventory,
        "counts": counts,
        "csv": csv_report,
        "master_json": json_report,
        "markdown": markdown_report,
        "overall_ok": inventory["ok"] and csv_report["ok"] and json_report["ok"] and markdown_report["ok"],
    }

    AUDIT_JSON_PATH.write_text(json.dumps(report, indent=2))
    write_audit_markdown(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
