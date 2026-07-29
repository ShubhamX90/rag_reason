#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run deterministic post-hoc citation-quality diagnostics for all synced CATS runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from compute_posthoc_citation_quality import compute_posthoc_citation_quality


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "outputs" / "benchmark_local_committee_3judge"
INPUTS_ROOT = REPO_ROOT / "inputs" / "prepped_model_eval_inputs"
SYNC_AUDIT_ROOT = RESULTS_ROOT / "sync_audits" / "20260709_response_cache_sync"
OUTPUT_ROOT = RESULTS_ROOT / "citation_quality_posthoc"
MASTER_DIR = OUTPUT_ROOT / "master"


def _relative_variant_from_result(path: Path) -> str:
    return str(path.relative_to(RESULTS_ROOT)).removesuffix("/final/detailed_results.json")


def _input_path_for_variant(variant: str) -> Path:
    return INPUTS_ROOT / variant / "input.jsonl"


def _load_provenance() -> Dict[str, Dict[str, Any]]:
    provenance: Dict[str, Dict[str, Any]] = {}
    for name, provenance_type in [
        ("canonical_response_cache_exact_manifest.json", "exact_cache_match"),
        ("canonical_response_cache_missing_exact.json", "final_result_only_missing_exact_cache"),
    ]:
        path = SYNC_AUDIT_ROOT / name
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            provenance[row["variant"]] = {
                "account": row.get("account"),
                "launch": row.get("launch"),
                "remote_cache": row.get("remote_cache"),
                "source_final": row.get("source_final"),
                "source_final_sha256": row.get("source_final_sha256"),
                "provenance_type": provenance_type,
            }
    return provenance


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    provenance = _load_provenance()
    result_files = sorted(RESULTS_ROOT.rglob("detailed_results.json"))

    summary_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    missing_inputs: List[Dict[str, str]] = []

    for detailed_results_path in result_files:
        variant = _relative_variant_from_result(detailed_results_path)
        input_jsonl = _input_path_for_variant(variant)
        if not input_jsonl.exists():
            missing_inputs.append({
                "variant": variant,
                "expected_input_jsonl": str(input_jsonl),
            })
            continue

        result = compute_posthoc_citation_quality(detailed_results_path, input_jsonl)
        summary = result["summary"]
        prov = provenance.get(variant, {})

        out_dir = OUTPUT_ROOT / variant
        _write_json(
            out_dir / "citation_quality_summary.json",
            {
                "variant": variant,
                "provenance": prov,
                "summary": summary,
                "worst_soft_extra_examples": result["worst_soft_extra_examples"],
                "worst_hard_negative_examples": result["worst_hard_negative_examples"],
            },
        )
        _write_jsonl(out_dir / "citation_quality_claim_rows.jsonl", result["claim_rows"])

        row = {
            "variant": variant,
            "input_jsonl": str(input_jsonl),
            "detailed_results_path": str(detailed_results_path),
            "source_account": prov.get("account"),
            "source_launch": prov.get("launch"),
            "source_remote_cache": prov.get("remote_cache"),
            "source_final": prov.get("source_final"),
            "source_final_sha256": prov.get("source_final_sha256"),
            "source_provenance_type": prov.get("provenance_type"),
        }
        row.update(summary)
        summary_rows.append(row)

        audit_rows.append({
            "variant": variant,
            "claim_rows_count": len(result["claim_rows"]),
            "claims_with_citations": summary.get("claims_with_citations"),
            "claims_with_committee_support_set": summary.get("claims_with_committee_support_set"),
            "claims_with_any_approved_citation": summary.get("claims_with_any_approved_citation"),
            "claims_with_any_gold_positive_citation": summary.get("claims_with_any_gold_positive_citation"),
        })

    if missing_inputs:
        _write_json(MASTER_DIR / "missing_inputs.json", {"missing_inputs": missing_inputs})
        raise SystemExit(f"Missing input JSONL files for {len(missing_inputs)} variants")

    summary_rows.sort(key=lambda row: row["variant"])
    audit_rows.sort(key=lambda row: row["variant"])

    master_json = {
        "total_variants": len(summary_rows),
        "summary_rows": summary_rows,
        "audit_rows": audit_rows,
    }
    _write_json(MASTER_DIR / "citation_quality_master.json", master_json)

    fieldnames = [
        "variant",
        "source_account",
        "source_launch",
        "source_provenance_type",
        "total_samples",
        "total_claims",
        "claims_with_citations",
        "claims_with_committee_support_set",
        "claims_with_any_approved_citation",
        "claims_with_any_gold_positive_citation",
        "committee_alignment_precision_macro",
        "committee_alignment_precision_micro",
        "gold_positive_citation_precision_macro",
        "gold_positive_citation_precision_micro",
        "committee_support_citation_recall_macro",
        "committee_support_citation_recall_micro",
        "soft_extra_citation_rate_micro",
        "hard_negative_citation_rate_micro",
        "claims_with_any_soft_extra_rate",
        "claims_with_any_hard_negative_rate",
        "strict_committee_alignment_claim_rate",
        "strict_gold_clean_claim_rate",
        "strict_grounded_committee_clean_rate_all_claims",
        "strict_grounded_gold_clean_rate_all_claims",
        "total_cited_docs",
        "total_approved_cited_docs",
        "total_gold_positive_cited_docs",
        "total_soft_extra_cited_docs",
        "total_hard_negative_cited_docs",
        "detailed_results_path",
        "input_jsonl",
        "source_final",
        "source_final_sha256",
        "source_remote_cache",
    ]
    csv_path = MASTER_DIR / "citation_quality_master.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(
        json.dumps(
            {
                "total_variants": len(summary_rows),
                "master_json": str(MASTER_DIR / "citation_quality_master.json"),
                "master_csv": str(csv_path),
                "per_run_output_root": str(OUTPUT_ROOT),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
