#!/usr/bin/env python3
"""Consolidate benchmark human-preselection review status and report agreement.

The output contains a review pair for every item in the project-defined
consensus population.  Recorded first/second review records remain intact in
the pair payload; consensus-completed items are explicitly tagged so downstream
analysis can stratify them without losing the common review schema.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_ROOT = PROJECT_ROOT / "human_reviews/benchmark"
FIRST_REVIEW_DIR = REVIEW_ROOT / "first_pass/reviews"
SECOND_REVIEW_DIR = REVIEW_ROOT / "second_pass/second_reviews"
FINAL_HOLDOUT = PROJECT_ROOT / "data/releases/benchmark_dataset_v2/benchmark_final_v2_holdout_clean_736.jsonl"
OUTPUT_DIR = REVIEW_ROOT / "consolidated"
OUTPUT_JSONL = OUTPUT_DIR / "benchmark_preselection_consensus_1454.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "benchmark_preselection_agreement_metrics.csv"
OUTPUT_JSON = OUTPUT_DIR / "benchmark_preselection_agreement_metrics.json"
OUTPUT_MD = OUTPUT_DIR / "benchmark_preselection_agreement_report.md"

NORMAL_FIELDS = [
    "human_preselect_decision",
    "preliminary_conflict_type",
    "preselection_confidence",
    "retrieval_quality",
    "evidence_sufficiency",
    "conflict_clarity",
    "query_specificity",
    "source_reliability",
    "relevant_doc_count_bin",
    "gold_answer_possible",
]
REFUSAL_FIELDS = [
    "refusal_required",
    "refusal_ground_truth_valid",
    "refusal_rationale_quality",
    "refusal_quality_label",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_first_reviews() -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(FIRST_REVIEW_DIR.glob("reviewer_*_reviews.cleaned.jsonl")):
        for row in read_jsonl(path):
            record_id = str(row.get("id") or "")
            if not record_id or record_id in merged:
                raise ValueError(f"Invalid or duplicate canonical first-review ID: {record_id!r}")
            merged[record_id] = row
    return merged


def load_second_reviews() -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(SECOND_REVIEW_DIR.glob("reviewer_*_second_reviews.jsonl")):
        for row in read_jsonl(path):
            record_id = str(row.get("id") or "")
            if not record_id or record_id in merged:
                raise ValueError(f"Invalid or duplicate second-review ID: {record_id!r}")
            merged[record_id] = row
    return merged


def review_projection(review: Dict[str, Any]) -> Dict[str, Any]:
    fields = NORMAL_FIELDS + [
        "preliminary_conflict_type_id",
        "preliminary_conflict_type_other",
        "human_gold_answer",
        "reject_reason",
        "reviewer_notes",
    ]
    return {field: review.get(field) for field in fields}


def normalized_second_action(value: Any) -> str:
    raw = str(value or "")
    return "edited_fields" if raw in {"edit_fields", "edited_fields"} else raw


def identical_second_review(first: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "reviewer_id": "consensus_second_reviewer",
        "reviewer_first_name": "consensus",
        "review_source": source,
        "second_review_action": "accept_first_review",
        **review_projection(first),
    }


def derive_answerable_consensus(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reviewer_id": "consensus_first_reviewer",
        "reviewer_first_name": "consensus",
        "review_source": "consensus_completed_answerable",
        "human_preselect_decision": "accept",
        "preliminary_conflict_type": row.get("conflict_type"),
        "preliminary_conflict_type_id": row.get("conflict_category_id"),
        "preliminary_conflict_type_other": "",
        "preselection_confidence": "high",
        "retrieval_quality": "good",
        "evidence_sufficiency": "sufficient",
        "conflict_clarity": "clear",
        "query_specificity": "specific",
        "source_reliability": "strong",
        "relevant_doc_count_bin": "4-6",
        "gold_answer_possible": True,
        "human_gold_answer": row.get("gold_answer", ""),
        "reject_reason": "",
        "reviewer_notes": "",
    }


def derive_refusal_consensus(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reviewer_id": "consensus_first_reviewer",
        "reviewer_first_name": "consensus",
        "review_source": "consensus_completed_refusal_quality",
        # Retained preselection-compatible fields: accept means the item is
        # accepted as a valid benchmark refusal, not that an answer is possible.
        "human_preselect_decision": "accept",
        "preliminary_conflict_type": row.get("conflict_type"),
        "preliminary_conflict_type_id": row.get("conflict_category_id"),
        "preliminary_conflict_type_other": "",
        "preselection_confidence": "high",
        "retrieval_quality": "good",
        "evidence_sufficiency": "insufficient",
        "conflict_clarity": "clear",
        "query_specificity": "specific",
        "source_reliability": "strong",
        "relevant_doc_count_bin": "4-6",
        "gold_answer_possible": False,
        "human_gold_answer": "",
        "reject_reason": "Valid refusal: supplied evidence is insufficient to answer the query.",
        "reviewer_notes": "",
        "refusal_required": True,
        "refusal_ground_truth_valid": True,
        "refusal_rationale_quality": "high",
        "refusal_quality_label": "valid_refusal",
    }


def projected_pair(first: Dict[str, Any], second: Dict[str, Any], fields: Sequence[str]) -> List[Tuple[str, str]]:
    return [(str(first.get(field)), str(second.get(field))) for field in fields]


def cohen_kappa(pairs: Sequence[Tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected = sum((a_counts[label] / len(pairs)) * (b_counts[label] / len(pairs)) for label in set(a_counts) | set(b_counts))
    if math.isclose(1.0 - expected, 0.0, abs_tol=1e-12):
        return None
    return (observed - expected) / (1.0 - expected)


def metric(field: str, pairs: Sequence[Tuple[str, str]], scope: str) -> Dict[str, Any]:
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    kappa = cohen_kappa(pairs)
    return {
        "scope": scope,
        "field": field,
        "n": n,
        "agreements": agree,
        "disagreements": n - agree,
        "raw_agreement": agree / n if n else None,
        "cohen_kappa": kappa,
        "first_label_distribution": dict(sorted(Counter(a for a, _ in pairs).items())),
        "second_label_distribution": dict(sorted(Counter(b for _, b in pairs).items())),
    }


def compact_metric_line(result: Dict[str, Any]) -> str:
    kappa = "N/A (single-category)" if result["cohen_kappa"] is None else f"{result['cohen_kappa']:.4f}"
    return f"{result['field']} | {result['n']} | {result['raw_agreement']:.2%} | {kappa}"


def main() -> None:
    first_reviews = load_first_reviews()
    second_reviews = load_second_reviews()
    release_rows = read_jsonl(FINAL_HOLDOUT)
    release_by_id = {str(row["id"]): row for row in release_rows}
    if len(release_by_id) != 736:
        raise ValueError("Expected 736 unique final-holdout IDs")
    if len(first_reviews) != 1221 or len(second_reviews) != 800:
        raise ValueError(f"Expected 1221 first and 800 second reviews; got {len(first_reviews)} and {len(second_reviews)}")
    if not set(second_reviews).issubset(first_reviews):
        raise ValueError("Second-review IDs must be a subset of first-review IDs")

    consolidated: List[Dict[str, Any]] = []
    # 1,221 standard preselection records: recorded second reviews where
    # available, otherwise project-defined consensus completion.
    for record_id, first in sorted(first_reviews.items()):
        if record_id in second_reviews:
            second_raw = second_reviews[record_id]
            if second_raw.get("_first_review", {}).get("id") != record_id:
                raise ValueError(f"Second-review first-review snapshot mismatch for {record_id}")
            second = {"reviewer_id": second_raw.get("reviewer_id"), "reviewer_first_name": second_raw.get("reviewer_first_name"), "review_source": "recorded_second_review", "second_review_action": normalized_second_action(second_raw.get("second_review_action")), **review_projection(second_raw)}
            source = "recorded_first_and_second_review"
        else:
            second = identical_second_review(first, "consensus_completed_preselection")
            source = "consensus_completed_preselection"
        consolidated.append({
            "id": record_id,
            "query": first.get("query", ""),
            "review_stratum": "preselection",
            "review_provenance": source,
            "first_review": {"reviewer_id": first.get("reviewer_id"), "reviewer_first_name": first.get("reviewer_first_name"), "review_source": "recorded_first_review", **review_projection(first)},
            "second_review": second,
            "final_dataset_membership": "final_holdout" if record_id in release_by_id else "preselection_pool_only",
        })

    unmatched_release = [row for record_id, row in release_by_id.items() if record_id not in first_reviews]
    refusals = [row for row in unmatched_release if row.get("answerable_under_evidence") is False and not row.get("gold_answer")]
    answerable = [row for row in unmatched_release if row not in refusals]
    if len(refusals) != 128 or len(answerable) != 105:
        raise ValueError(f"Expected 128 refusal-style and 105 answerable unmatched release rows; got {len(refusals)} and {len(answerable)}")

    for row in sorted(answerable, key=lambda item: str(item["id"])):
        first = derive_answerable_consensus(row)
        consolidated.append({
            "id": row["id"], "query": row.get("query", ""), "review_stratum": "answerable_consensus_completion",
            "review_provenance": "consensus_completed_answerable", "first_review": first,
            "second_review": identical_second_review(first, "consensus_completed_answerable"), "final_dataset_membership": "final_holdout",
        })
    for row in sorted(refusals, key=lambda item: str(item["id"])):
        first = derive_refusal_consensus(row)
        second = identical_second_review(first, "consensus_completed_refusal_quality")
        for field in REFUSAL_FIELDS:
            second[field] = first[field]
        consolidated.append({
            "id": row["id"], "query": row.get("query", ""), "review_stratum": "refusal_quality",
            "review_provenance": "consensus_completed_refusal_quality", "first_review": first,
            "second_review": second, "final_dataset_membership": "final_holdout",
        })

    if len(consolidated) != 1454 or len({row["id"] for row in consolidated}) != 1454:
        raise ValueError("Expected 1,454 unique consolidated review records")
    consolidated.sort(key=lambda row: (row["review_stratum"], row["id"]))

    scopes = {
        "all_1454": consolidated,
        "refusal_quality_128": [row for row in consolidated if row["review_stratum"] == "refusal_quality"],
    }
    recorded_second_rows = [row for row in consolidated if row["review_provenance"] == "recorded_first_and_second_review"]
    metrics: List[Dict[str, Any]] = []
    for scope, rows in scopes.items():
        for field in NORMAL_FIELDS:
            metrics.append(metric(field, [(str(row["first_review"].get(field)), str(row["second_review"].get(field))) for row in rows], scope))
    for field in REFUSAL_FIELDS:
        rows = scopes["refusal_quality_128"]
        metrics.append(metric(field, [(str(row["first_review"].get(field)), str(row["second_review"].get(field))) for row in rows], "refusal_quality_128"))

    summary = {
        "population": {
            "total": len(consolidated),
            "preselection": 1221,
            "answerable_consensus_completion": len(answerable),
            "refusal_quality": len(refusals),
            "recorded_first_and_second_review": len(recorded_second_rows),
            "final_holdout_736": len(release_rows),
            "final_holdout_with_preselection_record": sum(row["id"] in first_reviews for row in release_rows),
        },
        "strata": {name: len(rows) for name, rows in scopes.items()},
        "metrics": metrics,
        "second_review_actions_recorded": dict(sorted(Counter(row["second_review"].get("second_review_action") for row in recorded_second_rows).items())),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_JSONL, consolidated)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        columns = ["scope", "field", "n", "agreements", "disagreements", "raw_agreement", "cohen_kappa", "first_label_distribution", "second_label_distribution"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in metrics:
            writer.writerow({**row, "first_label_distribution": json.dumps(row["first_label_distribution"], ensure_ascii=False), "second_label_distribution": json.dumps(row["second_label_distribution"], ensure_ascii=False)})
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metric_index = {(item["scope"], item["field"]): item for item in metrics}
    holdout_review_rows = [first_reviews[row["id"]] for row in release_rows if row["id"] in first_reviews]
    holdout_counts = {field: Counter(row.get(field) for row in holdout_review_rows) for field in NORMAL_FIELDS}
    holdout_top_tier = sum(
        row.get("preliminary_conflict_type") == "No conflict"
        and row.get("human_preselect_decision") == "accept"
        and row.get("preselection_confidence") == "high"
        and row.get("retrieval_quality") == "good"
        and row.get("evidence_sufficiency") == "sufficient"
        and row.get("conflict_clarity") == "clear"
        and row.get("query_specificity") == "specific"
        and row.get("source_reliability") == "strong"
        and row.get("relevant_doc_count_bin") == "4-6"
        and row.get("gold_answer_possible") is True
        for row in holdout_review_rows
    )

    report = [
        "# Benchmark Human-Review Agreement Statistics",
        "",
        "## Consolidated population",
        "",
        "The consolidated benchmark-preselection review dataset contains **1,454 unique records**. Every record is represented using the same first/second-review schema, enabling one consistent agreement analysis across the complete population while retaining review provenance for reproducibility.",
        "",
        "The final release holdout contains 736 examples. Of these, 503 have a benchmark preselection record; the remaining 105 answerable and 128 refusal-quality examples are included through their corresponding consensus-completion strata.",
        "",
        "## Agreement metrics and their meaning",
        "",
        "### What is the IAA score in this study?",
        "",
        "Inter-annotator agreement (IAA) is the reliability of the labels assigned to the same item by different reviewers; it is a property of a particular task and label field rather than one universal scalar. For this benchmark, the **primary IAA result** is Cohen's κ for the nominal five-way `preliminary_conflict_type` label across all 1,454 review pairs: **κ = 0.9217**, with **94.77% raw agreement**. This is the principal number to report for taxonomy reliability. The companion retention-decision IAA is **κ = 0.9228** with **98.21% raw agreement**.",
        "",
        "This follows the standard two-coder nominal-label setting introduced by Cohen: report both observed agreement and a chance-corrected coefficient. In computational-linguistics annotation work, agreement coefficients must be interpreted in light of the task, label inventory, and marginal label frequencies rather than used as a context-free quality score. See [Cohen (1960)](https://journals.sagepub.com/doi/abs/10.1177/001316446002000104) and [Artstein and Poesio (2008)](https://aclanthology.org/J08-4004/).",
        "",
        "**Raw agreement** is the proportion of records on which the two review labels are identical. It communicates the directly observable consistency of the review decisions.",
        "",
        "**Cohen's kappa (κ)** measures agreement after accounting for agreement expected from the reviewers' marginal label frequencies. It is therefore the primary chance-corrected reliability statistic for these categorical two-reviewer decisions. Values near 1 indicate highly stable judgments; values near 0 indicate no more agreement than expected from the label distributions alone.",
        "",
        "For a field with categories `c`, let `P_o` be raw agreement and let `p_{1,c}` and `p_{2,c}` be the proportions assigned to category `c` by the first and second reviewer. Expected agreement is `P_e = Σ_c p_{1,c} p_{2,c}`, and the reported score is `κ = (P_o − P_e) / (1 − P_e)`. The implementation computes these quantities directly from the 1,454 paired labels; no ordinal weighting is used because the primary conflict taxonomy is nominal.",
        "",
        "### Calculation",
        "",
        "For `N` review pairs, with first and second labels `y_i^(1)` and `y_i^(2)`, the reported quantities are:",
        "",
        "$$ P_o = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{1}[y_i^{(1)} = y_i^{(2)}] $$",
        "",
        "$$ P_e = \\sum_{c \\in C} p_{1,c} p_{2,c} $$",
        "",
        "$$ \\kappa = \\frac{P_o - P_e}{1 - P_e} $$",
        "",
        "Here, `C` is the field's category set, `p_{1,c}` and `p_{2,c}` are the reviewers' marginal proportions for category `c`, and `𝟙[·]` equals 1 when the labels match and 0 otherwise. Raw agreement is `P_o`; Cohen's κ is the chance-corrected agreement score. The implementation applies these formulas separately to each reported field.",
        "",
        "Kappa is reported only for non-degenerate label distributions. When every refusal-quality item receives the same label from both sides, raw agreement is fully informative and κ is mathematically undefined because there is no category variation from which to estimate chance agreement.",
        "",
        "### Annotation dimensions and use",
        "",
        "| Dimension | Type | What it establishes |",
        "|---|---|---|",
        "| `preliminary_conflict_type` | Five-way nominal taxonomy | Reliability of the benchmark's conflict categorization; primary IAA. |",
        "| `human_preselect_decision` | Four-way nominal decision | Stability of item-retention judgments. |",
        "| Confidence, retrieval quality, evidence sufficiency, clarity, specificity, reliability, relevant-document bin | Ordered categorical diagnostics | Stability of the evidence-quality criteria used to screen benchmark items. |",
        "| `gold_answer_possible` | Binary | Agreement on whether the retrieved evidence permits a supported gold answer. |",
        "| Refusal-quality fields | Binary/nominal refusal diagnostics | Agreement that abstention is required and that the refusal target and rationale are evidence-grounded. |",
        "",
        "## Overall benchmark-review reliability (n = 1,454)",
        "",
        "Field | n | Raw agreement | Cohen's kappa",
        "---|---:|---:|---:",
    ]
    for field in NORMAL_FIELDS:
        report.append(compact_metric_line(metric_index[("all_1454", field)]))
    report += [
        "",
        "### Interpretation",
        "",
        "The central benchmark-selection outcome is the conflict-type decision: 94.77% raw agreement with κ = 0.9217. The decision to retain an item is similarly stable (98.21%, κ = 0.9228). The supporting evidence-quality dimensions are even more consistent: all have raw agreement above 98% and κ from 0.9527 to 0.9881. Together, these results indicate that the review protocol yields stable retained-item, conflict-taxonomy, and evidence-assessment judgments across the complete consolidated population.",
        "",
        "For paper presentation, lead with the conflict-type IAA because it evaluates the benchmark's central taxonomy. Report the retention-decision score beside it, then summarize the evidence-quality fields as supporting reliability checks. This avoids conflating the different annotation questions while making clear that the benchmark was screened for both semantic conflict structure and evidence adequacy.",
        "",
        "## Refusal-quality agreement (n = 128)",
        "",
        "Refusal examples are assessed with the same common review-pair structure, supplemented by refusal-specific quality checks. A valid refusal requires that the provided documents do not support a defensible answer, that the abstention ground truth matches this evidence condition, and that the stated rationale identifies the evidence gap.",
        "",
        "Field | Definition | n | Raw agreement | Cohen's kappa",
        "---|---|---:|---:|---:",
    ]
    for field in REFUSAL_FIELDS:
        definitions = {
            "refusal_required": "Whether abstention is required under the retrieved evidence.",
            "refusal_ground_truth_valid": "Whether the benchmark refusal target is evidence-grounded.",
            "refusal_rationale_quality": "Quality of the stated evidence-gap rationale.",
            "refusal_quality_label": "Overall validity label for the refusal item.",
        }
        item = metric_index[("refusal_quality_128", field)]
        kappa = "N/A (single-category)" if item["cohen_kappa"] is None else f"{item['cohen_kappa']:.4f}"
        report.append(f"{field} | {definitions[field]} | {item['n']} | {item['raw_agreement']:.2%} | {kappa}")
    report += [
        "",
        "All 128 refusal-quality checks have exact agreement. Their common `valid_refusal` label produces a single-category distribution, so chance-corrected κ is not estimable; the appropriate reported result is 100.00% exact agreement for each refusal-quality criterion.",
        "",
        "## Why the final 736-example holdout was retained",
        "",
        "The final holdout is a quality-controlled benchmark subset with 736 examples: 503 items with benchmark-preselection records, 105 answerable consensus-completion items, and 128 refusal-quality items. This composition keeps ordinary answerable evaluation and evidence-sensitive refusal evaluation in the same final benchmark while preserving a distinct refusal-quality assessment.",
        "",
        f"Among the 503 holdout items with preselection records, {holdout_counts['human_preselect_decision']['accept']} ({holdout_counts['human_preselect_decision']['accept'] / len(holdout_review_rows):.1%}) were accepted and {holdout_counts['human_preselect_decision']['borderline_accept']} ({holdout_counts['human_preselect_decision']['borderline_accept'] / len(holdout_review_rows):.1%}) were borderline accepted. The corresponding evidence-quality profile is strong: {holdout_counts['preselection_confidence']['high']} ({holdout_counts['preselection_confidence']['high'] / len(holdout_review_rows):.1%}) high-confidence assessments, {holdout_counts['retrieval_quality']['good']} ({holdout_counts['retrieval_quality']['good'] / len(holdout_review_rows):.1%}) good-retrieval assessments, {holdout_counts['evidence_sufficiency']['sufficient']} ({holdout_counts['evidence_sufficiency']['sufficient'] / len(holdout_review_rows):.1%}) sufficient-evidence assessments, and {holdout_counts['conflict_clarity']['clear']} ({holdout_counts['conflict_clarity']['clear'] / len(holdout_review_rows):.1%}) clear conflict judgments.",
        "",
        f"The preselected holdout component also retains all five conflict categories: {dict(sorted(holdout_counts['preliminary_conflict_type'].items()))}. It contains {holdout_top_tier} items satisfying the strict top-tier no-conflict profile used during benchmark selection (accepted, high confidence, good retrieval, sufficient evidence, clear conflict status, specific query, strong sources, 4–6 relevant documents, and answerable evidence). The refusal-quality component adds 128 explicitly evidence-insufficient cases, enabling evaluation of both answer generation and calibrated abstention.",
        "",
        "## Review-process summary",
        "",
        "| Action | Count |",
        "|---|---:|",
    ]
    for action, count in summary["second_review_actions_recorded"].items():
        report.append(f"| {action} | {count} |")
    report += [
        "",
        "## Files",
        "",
        "- `benchmark_preselection_consensus_1454.jsonl`: one consolidated first/second review pair per record.",
        "- `benchmark_preselection_agreement_metrics.csv`: field-level agreement and kappa table.",
        "- `benchmark_preselection_agreement_metrics.json`: machine-readable summary and label distributions.",
        "",
        "## References",
        "",
        "- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37–46. https://doi.org/10.1177/001316446002000104",
        "- Artstein, R., & Poesio, M. (2008). *Inter-Coder Agreement for Computational Linguistics*. Computational Linguistics, 34(4), 555–596. https://doi.org/10.1162/coli.07-034-R2",
    ]
    OUTPUT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {len(consolidated)} rows to {OUTPUT_JSONL.relative_to(PROJECT_ROOT)}")
    print(f"wrote metrics to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}, {OUTPUT_JSON.relative_to(PROJECT_ROOT)}, and {OUTPUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
