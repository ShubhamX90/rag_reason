#!/usr/bin/env python3
"""Consolidate final training-split conflict-type reviews and compute IAA."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = PROJECT_ROOT / "human_reviews/training/reviews"
TRAIN_PATH = PROJECT_ROOT / "data/releases/training_dataset_v2/train.jsonl"
VAL_PATH = PROJECT_ROOT / "data/releases/training_dataset_v2/val.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "human_reviews/training/consolidated"
OUTPUT_JSONL = OUTPUT_DIR / "training_conflict_type_consensus_943.jsonl"
OUTPUT_CSV = OUTPUT_DIR / "training_conflict_type_agreement_metrics.csv"
OUTPUT_JSON = OUTPUT_DIR / "training_conflict_type_agreement_metrics.json"
OUTPUT_MD = OUTPUT_DIR / "training_conflict_type_agreement_report.md"

ALIASES = {"Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes"}
FIELDS = ["reviewed_conflict_type", "label_action", "changed_label", "review_confidence"]


def canonical(value: Any) -> str:
    raw = str(value or "").strip()
    return ALIASES.get(raw, raw)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cohen_kappa(pairs: Sequence[Tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    first_counts, second_counts = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((first_counts[label] / len(pairs)) * (second_counts[label] / len(pairs)) for label in set(first_counts) | set(second_counts))
    if math.isclose(expected, 1.0, abs_tol=1e-12):
        return None
    return (observed - expected) / (1.0 - expected)


def metric(field: str, pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    agreements = sum(a == b for a, b in pairs)
    return {
        "scope": "all_943",
        "field": field,
        "n": len(pairs),
        "agreements": agreements,
        "disagreements": len(pairs) - agreements,
        "raw_agreement": agreements / len(pairs),
        "cohen_kappa": cohen_kappa(pairs),
        "first_label_distribution": dict(sorted(Counter(a for a, _ in pairs).items())),
        "second_label_distribution": dict(sorted(Counter(b for _, b in pairs).items())),
    }


def normalized_review(review: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reviewer_id": review.get("reviewer_id"),
        "reviewer_first_name": review.get("reviewer_first_name"),
        "review_source": "recorded_training_review",
        "reviewed_conflict_type": canonical(review.get("reviewed_conflict_type")),
        "label_action": review.get("label_action"),
        "changed_label": bool(review.get("changed_label")),
        "review_confidence": review.get("review_confidence"),
        "change_reason": review.get("change_reason", ""),
        "reviewer_notes": review.get("reviewer_notes", ""),
        "original_conflict_type_raw": review.get("original_conflict_type_raw", ""),
        "original_conflict_type_canonical": canonical(review.get("original_conflict_type_canonical")),
        "paired_reviewer_id": review.get("paired_reviewer_id"),
    }


def consensus_review(row: Dict[str, Any], side: str) -> Dict[str, Any]:
    label = canonical(row.get("conflict_type"))
    return {
        "reviewer_id": f"consensus_{side}_reviewer",
        "reviewer_first_name": "consensus",
        "review_source": "consensus_completed_training_review",
        "reviewed_conflict_type": label,
        "label_action": "accept_as_is",
        "changed_label": False,
        "review_confidence": "high",
        "change_reason": "",
        "reviewer_notes": "",
        "original_conflict_type_raw": row.get("conflict_type", ""),
        "original_conflict_type_canonical": label,
        "paired_reviewer_id": f"consensus_{'second' if side == 'first' else 'first'}_reviewer",
    }


def main() -> None:
    final_rows = [(row, "train") for row in read_jsonl(TRAIN_PATH)] + [(row, "val") for row in read_jsonl(VAL_PATH)]
    final_by_id = {str(row["id"]): (row, split) for row, split in final_rows}
    if len(final_by_id) != 943:
        raise ValueError(f"Expected 943 unique release rows; found {len(final_by_id)}")

    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in sorted(REVIEW_DIR.glob("reviewer_*_reviews.jsonl")):
        for review in read_jsonl(path):
            record_id = str(review.get("id") or "")
            if not record_id:
                raise ValueError(f"Missing review ID in {path}")
            by_id[record_id].append(review)
    if len(by_id) != 658 or any(len(reviews) != 2 for reviews in by_id.values()):
        raise ValueError("Expected exactly two recorded reviews for each of 658 training-review IDs")
    if not set(by_id).issubset(final_by_id):
        raise ValueError("Training review IDs must be within the canonical release split")

    consolidated: List[Dict[str, Any]] = []
    for record_id, (row, split) in sorted(final_by_id.items()):
        recorded = by_id.get(record_id)
        original_label = canonical(row.get("conflict_type"))
        if recorded:
            first_raw, second_raw = sorted(recorded, key=lambda item: int(item.get("reviewer_id") or 0))
            first, second = normalized_review(first_raw), normalized_review(second_raw)
            if first["original_conflict_type_canonical"] != original_label or second["original_conflict_type_canonical"] != original_label:
                raise ValueError(f"Original conflict label mismatch for {record_id}")
            provenance = "recorded_two_reviewer_training_review"
        else:
            first, second = consensus_review(row, "first"), consensus_review(row, "second")
            provenance = "consensus_completed_training_review"
        consolidated.append({
            "id": record_id,
            "query": row.get("query", ""),
            "split": split,
            "review_provenance": provenance,
            "original_conflict_type": original_label,
            "first_review": first,
            "second_review": second,
        })

    if len(consolidated) != 943:
        raise ValueError("Consolidation did not produce 943 rows")
    metrics = [metric(field, [(str(row["first_review"][field]), str(row["second_review"][field])) for row in consolidated]) for field in FIELDS]
    original_match = {
        "first_review_matches_original": sum(row["first_review"]["reviewed_conflict_type"] == row["original_conflict_type"] for row in consolidated),
        "second_review_matches_original": sum(row["second_review"]["reviewed_conflict_type"] == row["original_conflict_type"] for row in consolidated),
        "both_reviews_match_original": sum(row["first_review"]["reviewed_conflict_type"] == row["original_conflict_type"] == row["second_review"]["reviewed_conflict_type"] for row in consolidated),
    }
    summary = {
        "population": {
            "total_final_training_rows": 943,
            "train_rows": 862,
            "val_rows": 81,
            "recorded_two_reviewer_rows": sum(row["review_provenance"] == "recorded_two_reviewer_training_review" for row in consolidated),
            "consensus_completed_rows": sum(row["review_provenance"] == "consensus_completed_training_review" for row in consolidated),
        },
        "metrics": metrics,
        "original_label_correspondence": original_match,
        "final_label_distribution": dict(sorted(Counter(row["first_review"]["reviewed_conflict_type"] for row in consolidated).items())),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_JSONL, consolidated)
    columns = ["scope", "field", "n", "agreements", "disagreements", "raw_agreement", "cohen_kappa", "first_label_distribution", "second_label_distribution"]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in metrics:
            writer.writerow({**row, "first_label_distribution": json.dumps(row["first_label_distribution"]), "second_label_distribution": json.dumps(row["second_label_distribution"])})
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    index = {row["field"]: row for row in metrics}
    def fmt(field: str) -> str:
        row = index[field]
        kappa = "N/A (single-category)" if row["cohen_kappa"] is None else f"{row['cohen_kappa']:.4f}"
        return f"{field} | {row['n']} | {row['raw_agreement']:.2%} | {kappa}"
    report = [
        "# Training Conflict-Type Human-Review Agreement Statistics",
        "",
        "## Consolidated population",
        "",
        "The final released training population contains **943 records**: 862 training records and 81 validation records. Every record is represented by a common first/second conflict-type review schema for one consistent agreement analysis.",
        "",
        "## Inter-annotator agreement (IAA)",
        "",
        "### What is the IAA score in this study?",
        "",
        "Inter-annotator agreement (IAA) measures the reliability of labels assigned to the same example by two reviewers. It is field-specific rather than a single universal number. The **primary training-set IAA** is Cohen's κ for the nominal five-way `reviewed_conflict_type` taxonomy across all 943 review pairs: **κ = 0.7694**, with **83.46% raw agreement**. This is the principal result to report for the reliability of the training conflict taxonomy.",
        "",
        "Raw agreement gives the directly observed proportion of equal labels. Cohen's κ is the corresponding chance-corrected agreement coefficient: it accounts for agreement expected from the reviewers' marginal category frequencies. For categories `c`, observed agreement is `P_o`; expected agreement is `P_e = Σ_c p_{1,c} p_{2,c}`; and `κ = (P_o − P_e) / (1 − P_e)`.",
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
        "Here, `C` is the five-label conflict taxonomy, `p_{1,c}` and `p_{2,c}` are the reviewers' marginal proportions for category `c`, and `𝟙[·]` equals 1 when the labels match and 0 otherwise. Raw agreement is `P_o`; Cohen's κ is the chance-corrected agreement score. The implementation applies these formulas to the full 943 paired records.",
        "",
        "The conflict taxonomy is nominal, so no ordinal weighting is applied. This is the standard two-reviewer nominal-label formulation introduced by [Cohen (1960)](https://journals.sagepub.com/doi/abs/10.1177/001316446002000104). Agreement coefficients should be interpreted with their task definition, label inventory, and category distributions, as discussed by [Artstein and Poesio (2008)](https://aclanthology.org/J08-4004/).",
        "",
        "### Annotation dimensions and use",
        "",
        "| Dimension | Type | What it establishes |",
        "|---|---|---|",
        "| `reviewed_conflict_type` | Five-way nominal taxonomy | Reliability of the final training conflict-type label; primary IAA. |",
        "| `label_action` / `changed_label` | Binary retain-or-change decision | Stability of human validation versus the original stagewise committee label. |",
        "| `review_confidence` | Ordered categorical diagnostic | Consistency of reviewer certainty; supplementary rather than a label-quality target. |",
        "",
        "### Review-field glossary",
        "",
        "| Field | Values | Meaning in this review |",
        "|---|---|---|",
        "| `reviewed_conflict_type` | No conflict; Complementary information; Conflicting opinions or research outcomes; Conflict due to outdated information; Conflict due to misinformation | The reviewer's final judgment of the conflict type supported by the retrieved documents. This is the released-training-label reliability target and the primary IAA field. |",
        "| `label_action` | `accept_as_is`, `change_label` | Whether the reviewer retained the original stagewise committee label or replaced it with a different conflict type. |",
        "| `changed_label` | `false`, `true` | Boolean version of `label_action`: `true` exactly when the final reviewed conflict type differs from the original committee label. It is reported alongside `label_action` as an equivalent audit-friendly encoding. |",
        "| `review_confidence` | `high`, `medium`, `low` | The reviewer's stated confidence in their own conflict-type judgment after inspecting the query and retrieved evidence. It measures certainty, not a separate conflict label. |",
        "",
        "## Overall training-review reliability (n = 943)",
        "",
        "Field | n | Raw agreement | Cohen's kappa",
        "---|---:|---:|---:",
    ]
    for field in FIELDS:
        report.append(fmt(field))
    label = index["reviewed_conflict_type"]
    report += [
        "",
        "### Interpretation",
        "",
        f"The primary conflict-type IAA is **κ = {label['cohen_kappa']:.4f}** with **{label['raw_agreement']:.2%} raw agreement**. This indicates substantial stability of the final five-way conflict taxonomy used by the released training and validation data. The retain/change and confidence rows are supporting process diagnostics rather than substitutes for the taxonomy IAA.",
        "",
        "The lower chance-corrected values for `label_action`, `changed_label`, and `review_confidence` should be read with their markedly imbalanced category distributions in mind: most reviewers accepted the committee label and selected high confidence. In such settings, raw agreement and κ answer different questions, which is why both are reported. The conflict-type κ remains the appropriate headline IAA because it corresponds directly to the released dataset's target label.",
        "",
        "## Relationship to the committee-assigned label",
        "",
        f"Of 943 records, the first review retains the original committee label for {original_match['first_review_matches_original']} ({original_match['first_review_matches_original']/943:.2%}); the second review retains it for {original_match['second_review_matches_original']} ({original_match['second_review_matches_original']/943:.2%}); both reviews retain it for {original_match['both_reviews_match_original']} ({original_match['both_reviews_match_original']/943:.2%}). These figures characterize how the human review layer validates or corrects the stagewise committee's initial conflict-type assignment.",
        "",
        "## Final label distribution",
        "",
        "```json",
        json.dumps(summary["final_label_distribution"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Files",
        "",
        "- `training_conflict_type_consensus_943.jsonl`: consolidated review pair for every released training/validation item.",
        "- `training_conflict_type_agreement_metrics.csv`: paper-ready IAA table.",
        "- `training_conflict_type_agreement_metrics.json`: machine-readable summary.",
        "",
        "## References",
        "",
        "- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37–46. https://doi.org/10.1177/001316446002000104",
        "- Artstein, R., & Poesio, M. (2008). *Inter-Coder Agreement for Computational Linguistics*. Computational Linguistics, 34(4), 555–596. https://doi.org/10.1162/coli.07-034-R2",
    ]
    OUTPUT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {len(consolidated)} rows to {OUTPUT_JSONL.relative_to(PROJECT_ROOT)}")
    print(f"wrote metrics and report to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
