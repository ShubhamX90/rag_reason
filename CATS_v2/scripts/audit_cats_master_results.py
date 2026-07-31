#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_eval.evaluator import aggregate_sample_results

ROOT = Path("outputs/benchmark_local_committee_3judge")
MASTER_DIR = ROOT / "master_results"
LEGACY_SCOPE_CSV = MASTER_DIR / "cats_master_results_20260708.csv"
CSV_PATH = MASTER_DIR / "cats_master_results_20260731_hierarchical.csv"
JSON_PATH = MASTER_DIR / "cats_master_results_20260731_hierarchical.json"
MD_PATH = MASTER_DIR / "cats_master_results_20260731_hierarchical.md"
AUDIT_JSON_PATH = MASTER_DIR / "cats_master_results_20260731_hierarchical_audit.json"
AUDIT_MD_PATH = MASTER_DIR / "cats_master_results_20260731_hierarchical_audit.md"

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
    "answer_quality",
    "final_cats_prevalence",
    "final_cats_balanced",
    "legacy_flat_cats",
    "n",
    "behavior_n",
    "fg_n",
    "str_n",
    "answer_quality_n",
    "correct_refusals",
    "cats_complete",
    "cats_unscorable_n",
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
    answer_quality: float
    final_cats_prevalence: float | None
    final_cats_balanced: float | None
    legacy_flat_cats: float
    n: int
    behavior_n: int
    fg_n: int
    str_n: int
    answer_quality_n: int
    correct_refusals: int
    cats_complete: bool
    cats_unscorable_n: int

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
            "answer_quality": self.answer_quality,
            "final_cats_prevalence": self.final_cats_prevalence,
            "final_cats_balanced": self.final_cats_balanced,
            "legacy_flat_cats": self.legacy_flat_cats,
            "n": self.n,
            "behavior_n": self.behavior_n,
            "fg_n": self.fg_n,
            "str_n": self.str_n,
            "answer_quality_n": self.answer_quality_n,
            "correct_refusals": self.correct_refusals,
            "cats_complete": self.cats_complete,
            "cats_unscorable_n": self.cats_unscorable_n,
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
        return {field: "" if raw[field] is None else str(raw[field]) for field in CSV_FIELDS}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_scope_paths() -> list[str]:
    """Return the authoritative experiment scope used by the master matrix.

    The results tree also contains staged collection artifacts and comparison
    runs that are intentionally not part of the 108-row master matrix. The
    historical master CSV is the stable row-level scope for this workbook.
    """
    with LEGACY_SCOPE_CSV.open() as f:
        rows = list(csv.DictReader(f))
    paths = [row["source_relpath"] for row in rows]
    if len(paths) != 108 or len(set(paths)) != 108:
        raise ValueError(
            f"Expected 108 unique master-scope result paths in {LEGACY_SCOPE_CSV}, "
            f"found {len(paths)} rows and {len(set(paths))} unique paths"
        )
    return paths


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
    if parts[0] in {"other_techniques", "other_techniques_fixed"}:
        if len(parts) == 5 and parts[-2:] == ("final", "detailed_results.json"):
            return {
                "collection": parts[0],
                "category": parts[0],
                "variant": parts[1],
                "model_family": parts[2],
                "eval_family": parts[1],
                "prompt_mode": "comparison",
                "run_type": "committee_eval",
            }
        if len(parts) == 6 and parts[-1] == "detailed_results.json":
            return {
                "collection": parts[0],
                "category": parts[0],
                "variant": parts[1],
                "model_family": parts[2],
                "eval_family": parts[1],
                "prompt_mode": parts[3],
                "run_type": parts[4],
            }
        return {
            "collection": parts[0],
            "category": parts[0],
            "variant": parts[1] if len(parts) > 1 else "unknown",
            "model_family": parts[2] if len(parts) > 2 else "unknown",
            "eval_family": parts[1] if len(parts) > 1 else "unknown",
            "prompt_mode": "unexpected_shape",
            "run_type": str(rel),
        }
    raise ValueError(f"Unexpected result root: {rel}")


def load_source_rows() -> list[SourceRow]:
    rows: list[SourceRow] = []
    for rel_text in load_scope_paths():
        rel = Path(rel_text)
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Master-scope result file is missing: {rel}")
        data = load_json(path)
        if "per_sample" not in data:
            raise ValueError(f"Missing per_sample payload required for hierarchical recompute: {rel}")
        overall, _per_type, gr = aggregate_sample_results(data["per_sample"])
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
                answer_quality=overall.get("answer_quality", 0.0),
                final_cats_prevalence=overall.get("cats_prevalence_score", overall["cats_score"]),
                final_cats_balanced=overall.get("cats_balanced_score"),
                legacy_flat_cats=overall.get("cats_flat_legacy_score", overall["cats_score"]),
                n=overall["n"],
                behavior_n=overall["behavior_n"],
                fg_n=overall["factual_grounding_n"],
                str_n=overall["single_truth_recall_n"],
                answer_quality_n=overall.get("answer_quality_n", 0),
                correct_refusals=overall["correct_refusals"],
                cats_complete=bool(overall.get("cats_complete", False)),
                cats_unscorable_n=int(overall.get("cats_unscorable_n", 0)),
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


def format_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


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
    other = [r for r in expected_rows if r.category in {"other_techniques", "other_techniques_fixed"}]

    lines: list[str] = []
    lines.append("# CATS Master Results Matrix")
    lines.append("")
    lines.append(
        "This report is rebuilt directly from the synced `detailed_results.json` files under "
        "`outputs/benchmark_local_committee_3judge`. The secondary CATS summaries shown here are "
        "recomputed from each file's stored `per_sample` payload using the latest example-level "
        "hierarchical aggregation, rather than trusting the historical `summary.conflict_overall.cats_score`."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Total synced result files included: `{len(expected_rows)}`")
    lines.append(f"- Standard benchmark result files: `{len(standard)}`")
    lines.append(f"- Answer-only SFT result files: `{len(answer_only)}`")
    lines.append(f"- Other-techniques result files: `{len(other)}`")
    lines.append(
        "- The four redone Mistral/Qwen comparison runs are included from "
        "`other_techniques_fixed/{con,cot_fewshot}`; older unfixed Mistral/Qwen "
        "comparison JSONs are excluded from the 108-row master scope."
    )
    complete_rows = sum(row.cats_complete for row in expected_rows)
    unscorable_rows = sum(row.cats_unscorable_n for row in expected_rows)
    lines.append(f"- Complete CATS-H result files: `{complete_rows}`")
    lines.append(f"- Incomplete CATS-H result files: `{len(expected_rows) - complete_rows}`")
    lines.append(f"- Example rows without a computable CATS score: `{unscorable_rows}`")
    lines.append(
        "- Correct refusals contribute their grounded-refusal decision-correctness score; "
        "behavior, grounding, and recall remain non-applicable for those examples."
    )

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
        lines.append("| Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal | Delta Prev vs baseline | Delta Bal vs baseline |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

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
                    f"{format_float(baseline.behavior)} | {format_float(baseline.answer_quality)} | "
                    f"{format_float(baseline.final_cats_prevalence)} | {format_float(baseline.final_cats_balanced)} | — | — |"
                )
            if sft:
                delta_prev = (
                    sft.final_cats_prevalence - baseline.final_cats_prevalence
                    if baseline and sft.final_cats_prevalence is not None and baseline.final_cats_prevalence is not None
                    else None
                )
                delta_bal = (
                    sft.final_cats_balanced - baseline.final_cats_balanced
                    if baseline and sft.final_cats_balanced is not None and baseline.final_cats_balanced is not None
                    else None
                )
                lines.append(
                    f"| {eval_family} | {prompt_mode} | sft | {format_float(sft.gr_answer_precision)} | "
                    f"{format_float(sft.gr_answer_recall)} | {format_float(sft.gr_answer_f1)} | "
                    f"{format_float(sft.gr_refusal_precision)} | {format_float(sft.gr_refusal_recall)} | "
                    f"{format_float(sft.gr_refusal_f1)} | "
                    f"{format_float(sft.str_score)} | {format_float(sft.fg)} | "
                    f"{format_float(sft.behavior)} | {format_float(sft.answer_quality)} | "
                    f"{format_float(sft.final_cats_prevalence)} | {format_float(sft.final_cats_balanced)} | "
                    f"{format_float(delta_prev)} | {format_float(delta_bal)} |"
                )
        lines.append("")

    lines.append("## Answer-only SFT")
    lines.append("")
    lines.append(
        "These six runs are methodologically distinct from the standard benchmark family, so they "
        "are kept in their own section. All of them are `sft` runs."
    )
    lines.append("")
    lines.append("| Model | Eval family | Prompt | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(answer_only, key=lambda r: (r.model_family, r.eval_family, r.prompt_mode, r.run_type)):
        lines.append(
            f"| {row.model_family} | {row.eval_family} | {row.prompt_mode} | {row.run_type} | "
            f"{format_float(row.gr_answer_precision)} | {format_float(row.gr_answer_recall)} | {format_float(row.gr_answer_f1)} | "
            f"{format_float(row.gr_refusal_precision)} | {format_float(row.gr_refusal_recall)} | {format_float(row.gr_refusal_f1)} | "
            f"{format_float(row.str_score)} | {format_float(row.fg)} | {format_float(row.behavior)} | "
            f"{format_float(row.answer_quality)} | {format_float(row.final_cats_prevalence)} | "
            f"{format_float(row.final_cats_balanced)} |"
        )
    lines.append("")
    lines.append("## Other Techniques")
    lines.append("")
    lines.append(
        "These rows summarize the currently synced `CoN` and `CoT fewshot` comparisons. These are "
        "committee-evaluated comparison runs, not baseline/SFT prompt-family pairs."
    )
    lines.append("")
    lines.append("| Model | Technique | Run | GR-answer Precision | GR-answer Recall | GR-answer F1 | GR-refusal Precision | GR-refusal Recall | GR-refusal F1 | STR | FG | Behavior | Answer Quality | CATS-Prev | CATS-Bal |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(
        other,
        key=lambda r: (OTHER_MODEL_ORDER.index(r.model_family), r.variant, r.run_type),
    ):
        lines.append(
            f"| {row.model_family} | {row.variant} | {row.run_type} | {format_float(row.gr_answer_precision)} | "
            f"{format_float(row.gr_answer_recall)} | {format_float(row.gr_answer_f1)} | "
            f"{format_float(row.gr_refusal_precision)} | {format_float(row.gr_refusal_recall)} | {format_float(row.gr_refusal_f1)} | "
            f"{format_float(row.str_score)} | "
            f"{format_float(row.fg)} | {format_float(row.behavior)} | {format_float(row.answer_quality)} | "
            f"{format_float(row.final_cats_prevalence)} | {format_float(row.final_cats_balanced)} |"
        )
    lines.append("")
    lines.append("## Metric Notes")
    lines.append("")
    lines.append("- `GR-answer Precision`, `GR-answer Recall`, and `GR-answer F1` are read directly from `summary.gr_dataset_metrics.{precision, recall, f1}` in each synced `detailed_results.json`.")
    lines.append("- `GR-refusal Precision`, `GR-refusal Recall`, and `GR-refusal F1` are read directly from `summary.gr_dataset_metrics.{abstain_precision, abstain_recall, abstain_f1}` in each synced `detailed_results.json`.")
    lines.append("- The two GR families come from the same dataset-level answer/refuse confusion table; the answer family treats `answered` as positive, while the refusal family treats `refused` as positive.")
    lines.append("- `STR`, `FG`, and `Behavior` are reported from the stored per-sample judgments after recomputing aggregate means.")
    lines.append("- `Answer Quality` is the example-level fusion of FG and STR: `sqrt(FG * STR)` when STR applies, else `FG`.")
    lines.append("- `CATS-Prev` is the benchmark-prevalence-weighted CATS-Harmonized summary. `CATS-Bal` gives equal weight to conflict types and balances answerable/refusal-required subgroups within each type when both are present.")
    lines.append("- Final CATS values are recomputed from per-example hierarchical scores; correct refusals use decision correctness only.")
    lines.append(f"- This Markdown was regenerated directly from the synced JSON result files, not from the derived `{CSV_PATH.name}` file.")
    lines.append("- `Delta Prev vs baseline` and `Delta Bal vs baseline` are shown only on the `sft` row.")
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
    expected_paths = {row.source_relpath for row in expected_rows}
    all_detailed = sorted(
        path for path in ROOT.rglob("detailed_results.json")
        if path.parent != MASTER_DIR
    )
    detailed = [
        path for path in all_detailed
        if str(path.relative_to(ROOT)) in expected_paths
    ]
    ignored = [
        str(path.relative_to(ROOT)) for path in all_detailed
        if str(path.relative_to(ROOT)) not in expected_paths
    ]
    discovered_paths = {str(path.relative_to(ROOT)) for path in detailed}
    missing_scoped = sorted(expected_paths - discovered_paths)
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
        "ignored_out_of_scope_detailed_results_count": len(ignored),
        "ignored_out_of_scope_detailed_results": ignored,
        "missing_scoped_results": missing_scoped,
        "source_relpaths_unique": len({r.source_relpath for r in expected_rows}) == len(expected_rows),
        "json_parse_failures": parse_failures,
        "missing_siblings": missing_siblings,
        "non_unique_configuration_keys": non_unique_keys,
        "ok": not parse_failures and not missing_siblings and not non_unique_keys and not missing_scoped,
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
        "cats_complete_rows": sum(row.cats_complete for row in expected_rows),
        "cats_incomplete_rows": sum(not row.cats_complete for row in expected_rows),
        "cats_unscorable_total": sum(row.cats_unscorable_n for row in expected_rows),
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
        f"- Out-of-scope detailed result files ignored: `{report['inventory']['ignored_out_of_scope_detailed_results_count']}`",
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
        f"- CATS-complete rows: `{report['counts']['cats_complete_rows']}`",
        f"- CATS-incomplete rows: `{report['counts']['cats_incomplete_rows']}`",
        f"- Examples without a computable CATS score: `{report['counts']['cats_unscorable_total']}`",
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
