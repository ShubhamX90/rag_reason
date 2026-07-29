#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("outputs/model_output_eval_runs/val_set_all_modes")
OUT_DIR = Path("outputs/compiled_results")
EXCLUDED_MODELS = {"qwen3_32b"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize_strategy(name: str) -> str:
    return name.removesuffix("_prompt_outputs")


def conflict_type_columns(conflict_types: set[str]) -> list[str]:
    cols: list[str] = []
    for ct in sorted(conflict_types, key=lambda x: int(x) if str(x).isdigit() else x):
        prefix = f"ct{ct}_"
        cols.extend(
            [
                f"{prefix}n",
                f"{prefix}correct_refusals",
                f"{prefix}gr_accuracy",
                f"{prefix}gr_f1",
                f"{prefix}behavior",
                f"{prefix}behavior_n",
                f"{prefix}factual_grounding",
                f"{prefix}factual_grounding_n",
                f"{prefix}single_truth_recall",
                f"{prefix}single_truth_recall_n",
                f"{prefix}cats_score",
            ]
        )
    return cols


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    json_runs: list[dict] = []
    conflict_types_seen: set[str] = set()
    excluded_runs: list[str] = []

    for result_path in sorted(ROOT.rglob("detailed_results.json")):
        rel_parts = result_path.relative_to(ROOT).parts
        if len(rel_parts) != 5:
            continue

        mode, family, model, strategy_dir, _ = rel_parts
        if model in EXCLUDED_MODELS:
            excluded_runs.append(str(result_path.relative_to(ROOT).parent))
            continue

        result = load_json(result_path)
        summary = result["summary"]
        overall = summary["conflict_overall"]
        per_type = summary.get("conflict_per_type", {})
        gr_dataset = summary.get("gr_dataset_metrics", {})
        cost_summary = summary.get("cost_summary", {})

        for ct in per_type:
            conflict_types_seen.add(str(ct))

        row = {
            "mode": mode,
            "family": family,
            "model": model,
            "strategy": normalize_strategy(strategy_dir),
            "strategy_dir": strategy_dir,
            "relative_run_dir": str(result_path.parent.relative_to(ROOT)),
            "report_path": str((result_path.parent / "eval_report.md").relative_to(".")),
            "details_path": str(result_path.relative_to(".")),
            "samples_n": overall.get("n"),
            "correct_refusals": overall.get("correct_refusals"),
            "gr_accuracy": overall.get("gr_accuracy"),
            "gr_f1": overall.get("gr_f1"),
            "behavior": overall.get("behavior"),
            "behavior_n": overall.get("behavior_n"),
            "factual_grounding": overall.get("factual_grounding"),
            "factual_grounding_n": overall.get("factual_grounding_n"),
            "single_truth_recall": overall.get("single_truth_recall"),
            "single_truth_recall_n": overall.get("single_truth_recall_n"),
            "cats_score": overall.get("cats_score"),
            "gr_precision": gr_dataset.get("precision"),
            "gr_recall": gr_dataset.get("recall"),
            "gr_tp": gr_dataset.get("tp"),
            "gr_fp": gr_dataset.get("fp"),
            "gr_fn": gr_dataset.get("fn"),
            "gr_tn": gr_dataset.get("tn"),
            "total_cost_usd": cost_summary.get("total_cost_usd"),
            "decisions_made": cost_summary.get("decisions_made"),
            "avg_cost_per_decision": cost_summary.get("avg_cost_per_decision"),
        }

        for ct, values in per_type.items():
            prefix = f"ct{ct}_"
            row[f"{prefix}n"] = values.get("n")
            row[f"{prefix}correct_refusals"] = values.get("correct_refusals")
            row[f"{prefix}gr_accuracy"] = values.get("gr_accuracy")
            row[f"{prefix}gr_f1"] = values.get("gr_f1")
            row[f"{prefix}behavior"] = values.get("behavior")
            row[f"{prefix}behavior_n"] = values.get("behavior_n")
            row[f"{prefix}factual_grounding"] = values.get("factual_grounding")
            row[f"{prefix}factual_grounding_n"] = values.get("factual_grounding_n")
            row[f"{prefix}single_truth_recall"] = values.get("single_truth_recall")
            row[f"{prefix}single_truth_recall_n"] = values.get("single_truth_recall_n")
            row[f"{prefix}cats_score"] = values.get("cats_score")

        run_rows.append(row)
        json_runs.append(
            {
                "mode": mode,
                "family": family,
                "model": model,
                "strategy": normalize_strategy(strategy_dir),
                "strategy_dir": strategy_dir,
                "relative_run_dir": str(result_path.parent.relative_to(ROOT)),
                "report_path": str((result_path.parent / "eval_report.md").relative_to(".")),
                "details_path": str(result_path.relative_to(".")),
                "summary": summary,
                "per_sample": result.get("per_sample", []),
            }
        )

    base_columns = [
        "mode",
        "family",
        "model",
        "strategy",
        "strategy_dir",
        "relative_run_dir",
        "report_path",
        "details_path",
        "samples_n",
        "correct_refusals",
        "gr_accuracy",
        "gr_f1",
        "behavior",
        "behavior_n",
        "factual_grounding",
        "factual_grounding_n",
        "single_truth_recall",
        "single_truth_recall_n",
        "cats_score",
        "gr_precision",
        "gr_recall",
        "gr_tp",
        "gr_fp",
        "gr_fn",
        "gr_tn",
        "total_cost_usd",
        "decisions_made",
        "avg_cost_per_decision",
    ]
    fieldnames = base_columns + conflict_type_columns(conflict_types_seen)

    csv_path = OUT_DIR / "val_set_all_modes_compiled_results_excluding_qwen3_32b.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    json_path = OUT_DIR / "val_set_all_modes_compiled_results_excluding_qwen3_32b.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT),
        "excluded_models": sorted(EXCLUDED_MODELS),
        "included_run_count": len(json_runs),
        "excluded_run_count": len(excluded_runs),
        "excluded_runs": excluded_runs,
        "conflict_types_seen": sorted(conflict_types_seen, key=lambda x: int(x) if x.isdigit() else x),
        "runs": json_runs,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(
        {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "included_run_count": len(json_runs),
            "excluded_run_count": len(excluded_runs),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
