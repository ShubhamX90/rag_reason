from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_STUDY_DIR = (
    Path(__file__).resolve().parent.parent
    / "studies"
    / "qwen_llama_e2e_sft_baseline_balanced_4reviewers"
)
DEFAULT_OUTPUT_LABEL = "2026-07-30_full_receipts"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_assignments(study_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    assignments_path = study_dir / "assignments" / "assignments.json"
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    sample_to_reviewers: Dict[str, List[str]] = defaultdict(list)
    for reviewer, sample_ids in assignments.items():
        for sample_id in sample_ids:
            sample_to_reviewers[sample_id].append(reviewer)
    for sample_id in sample_to_reviewers:
        sample_to_reviewers[sample_id].sort()
    return assignments, dict(sample_to_reviewers)


def load_sample_index(study_dir: Path) -> Dict[str, Dict[str, Any]]:
    samples_path = study_dir / "data" / "samples.jsonl"
    return {row["sample_id"]: row for row in _iter_jsonl(samples_path)}


def return_directories(returns_root: Path) -> List[Tuple[str, Path]]:
    dirs: List[Tuple[str, Path]] = []
    if not returns_root.exists():
        return dirs
    for reviewer_dir in sorted(path for path in returns_root.iterdir() if path.is_dir()):
        reviewer = reviewer_dir.name
        for return_dir in sorted(path for path in reviewer_dir.iterdir() if path.is_dir()):
            dirs.append((reviewer, return_dir))
    return dirs


def sanitize_row(
    row: Dict[str, Any],
    reviewer: str,
    return_dir: Path,
    sample_to_reviewers: Dict[str, List[str]],
) -> Dict[str, Any]:
    sample_id = row["sample_id"]
    assigned_reviewers = sample_to_reviewers.get(sample_id, [])
    judgment = row.get("judgment") or {}
    sanitized = {
        "study_name": row.get("study_name"),
        "reviewer_id": row.get("reviewer_id"),
        "sample_id": sample_id,
        "status": row.get("status"),
        "revision": row.get("revision"),
        "query": row.get("query"),
        "conflict_category_id": row.get("conflict_category_id"),
        "conflict_type": row.get("conflict_type"),
        "gold_answerable": row.get("gold_answerable"),
        "correct_refusal": row.get("correct_refusal"),
        "claims_with_citations": row.get("claims_with_citations"),
        "judgment": judgment,
        "organizer_meta": {
            "return_reviewer_dir": reviewer,
            "return_id": f"{reviewer}/{return_dir.name}",
            "return_label": return_dir.name,
            "reviewer_folder_match": row.get("reviewer_id") == reviewer,
            "assignment_verified": reviewer in assigned_reviewers,
            "assigned_reviewers_for_sample": assigned_reviewers,
            "accepted_for_final_metrics": bool(
                row.get("status") == "submitted"
                and row.get("reviewer_id") == reviewer
                and reviewer in assigned_reviewers
            ),
        },
    }
    return sanitized


def parse_model_axes(sample_id: str) -> Dict[str, str]:
    parts = sample_id.split("__", 3)
    if len(parts) < 4:
        return {"model": "", "prompt": "", "train_type": "", "base_id": sample_id}
    model, prompt, train_type, base_id = parts
    return {
        "model": model,
        "prompt": prompt,
        "train_type": train_type,
        "base_id": base_id,
    }


def consolidate(study_dir: Path, output_dir: Path) -> Dict[str, Any]:
    assignments, sample_to_reviewers = load_assignments(study_dir)
    samples = load_sample_index(study_dir)
    returns_root = study_dir / "reviewer_returns"

    submitted_rows: List[Dict[str, Any]] = []
    draft_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    duplicate_issues: List[Dict[str, Any]] = []
    return_inventory: List[Dict[str, Any]] = []

    best_submitted_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    best_draft_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for reviewer, return_dir in return_directories(returns_root):
        enriched_path = return_dir / "active_judgments_enriched.jsonl"
        if not enriched_path.exists():
            return_inventory.append(
                {
                    "reviewer_dir": reviewer,
                    "return_id": f"{reviewer}/{return_dir.name}",
                    "path": str(return_dir),
                    "has_enriched": False,
                    "rows": 0,
                }
            )
            continue

        rows = 0
        statuses = Counter()
        file_reviewers = Counter()
        accepted_submitted = 0
        accepted_drafts = 0
        invalid_count = 0
        seen_in_file: Dict[Tuple[str, str], int] = {}

        for raw_row in _iter_jsonl(enriched_path):
            rows += 1
            statuses[str(raw_row.get("status"))] += 1
            file_reviewers[str(raw_row.get("reviewer_id"))] += 1

            sanitized = sanitize_row(raw_row, reviewer, return_dir, sample_to_reviewers)
            key = (sanitized["reviewer_id"], sanitized["sample_id"])
            revision = int(sanitized.get("revision") or 0)
            prior_revision = seen_in_file.get(key)
            if prior_revision is not None and revision <= prior_revision:
                duplicate_issues.append(
                    {
                        "kind": "within_file_non_increasing_revision",
                        "return_id": sanitized["organizer_meta"]["return_id"],
                        "reviewer_id": sanitized["reviewer_id"],
                        "sample_id": sanitized["sample_id"],
                        "revision": revision,
                        "prior_revision": prior_revision,
                    }
                )
            seen_in_file[key] = max(prior_revision or 0, revision)

            accepted = sanitized["organizer_meta"]["accepted_for_final_metrics"]
            status = sanitized["status"]
            if accepted and status == "submitted":
                prior = best_submitted_by_key.get(key)
                if prior is not None:
                    duplicate_issues.append(
                        {
                            "kind": "cross_return_duplicate_submitted",
                            "reviewer_id": sanitized["reviewer_id"],
                            "sample_id": sanitized["sample_id"],
                            "existing_return_id": prior["organizer_meta"]["return_id"],
                            "existing_revision": prior["revision"],
                            "new_return_id": sanitized["organizer_meta"]["return_id"],
                            "new_revision": sanitized["revision"],
                        }
                    )
                    if int(sanitized["revision"] or 0) <= int(prior["revision"] or 0):
                        continue
                best_submitted_by_key[key] = sanitized
                accepted_submitted += 1
                continue

            if status == "draft" and sanitized["organizer_meta"]["reviewer_folder_match"]:
                prior = best_draft_by_key.get(key)
                if prior is None or int(sanitized["revision"] or 0) > int(prior["revision"] or 0):
                    best_draft_by_key[key] = sanitized
                accepted_drafts += 1
                continue

            invalid_count += 1
            invalid_rows.append(sanitized)

        return_inventory.append(
            {
                "reviewer_dir": reviewer,
                "return_id": f"{reviewer}/{return_dir.name}",
                "path": str(return_dir),
                "has_enriched": True,
                "rows": rows,
                "statuses": dict(statuses),
                "embedded_reviewers": dict(file_reviewers),
                "accepted_submitted_rows_seen": accepted_submitted,
                "draft_rows_seen": accepted_drafts,
                "invalid_rows_seen": invalid_count,
            }
        )

    submitted_rows = sorted(
        best_submitted_by_key.values(),
        key=lambda row: (row["sample_id"], row["reviewer_id"]),
    )
    draft_rows = sorted(
        best_draft_by_key.values(),
        key=lambda row: (row["sample_id"], row["reviewer_id"]),
    )

    submitted_by_sample: Dict[str, List[str]] = defaultdict(list)
    draft_by_sample: Dict[str, List[str]] = defaultdict(list)
    submitted_by_reviewer = Counter()
    draft_by_reviewer = Counter()
    for row in submitted_rows:
        submitted_by_sample[row["sample_id"]].append(row["reviewer_id"])
        submitted_by_reviewer[row["reviewer_id"]] += 1
    for row in draft_rows:
        draft_by_sample[row["sample_id"]].append(row["reviewer_id"])
        draft_by_reviewer[row["reviewer_id"]] += 1

    coverage_rows: List[Dict[str, Any]] = []
    coverage_counter = Counter()
    for sample_id, assigned_reviewers in sorted(sample_to_reviewers.items()):
        sample = samples.get(sample_id, {})
        submitted_reviewers = sorted(submitted_by_sample.get(sample_id, []))
        draft_reviewers = sorted(draft_by_sample.get(sample_id, []))
        missing_reviewers = [reviewer for reviewer in assigned_reviewers if reviewer not in submitted_reviewers]
        axes = parse_model_axes(sample_id)
        coverage = {
            "sample_id": sample_id,
            "base_id": axes["base_id"],
            "model": axes["model"],
            "prompt": axes["prompt"],
            "train_type": axes["train_type"],
            "query": sample.get("query"),
            "conflict_category_id": sample.get("conflict_category_id"),
            "conflict_type": sample.get("conflict_type"),
            "assigned_reviewers": assigned_reviewers,
            "submitted_reviewers": submitted_reviewers,
            "draft_reviewers": draft_reviewers,
            "missing_reviewers": missing_reviewers,
            "submitted_count": len(submitted_reviewers),
            "draft_count": len(draft_reviewers),
            "coverage_status": (
                "complete" if len(submitted_reviewers) == len(assigned_reviewers)
                else "partial" if submitted_reviewers
                else "none"
            ),
        }
        coverage_rows.append(coverage)
        coverage_counter[(coverage["submitted_count"], coverage["coverage_status"])] += 1

    reviewers_expected = {reviewer: len(sample_ids) for reviewer, sample_ids in assignments.items()}
    reviewer_progress = {}
    for reviewer in sorted(assignments):
        reviewer_progress[reviewer] = {
            "assigned": reviewers_expected[reviewer],
            "submitted_count": submitted_by_reviewer.get(reviewer, 0),
            "draft_count": draft_by_reviewer.get(reviewer, 0),
            "missing_count": reviewers_expected[reviewer] - submitted_by_reviewer.get(reviewer, 0),
            "has_any_return_folder": any(item["reviewer_dir"] == reviewer for item in return_inventory),
        }

    summary = {
        "study_dir": str(study_dir),
        "output_dir": str(output_dir),
        "accepted_submitted_rows": len(submitted_rows),
        "accepted_draft_rows": len(draft_rows),
        "invalid_rows": len(invalid_rows),
        "duplicate_issue_count": len(duplicate_issues),
        "reviewer_progress": reviewer_progress,
        "return_inventory": return_inventory,
        "coverage_distribution": {
            f"{submitted_count}_{status}": count
            for (submitted_count, status), count in sorted(coverage_counter.items())
        },
        "fully_complete_samples": sum(1 for row in coverage_rows if row["coverage_status"] == "complete"),
        "partially_complete_samples": sum(1 for row in coverage_rows if row["coverage_status"] == "partial"),
        "unstarted_samples": sum(1 for row in coverage_rows if row["coverage_status"] == "none"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "submitted_judgments_enriched.jsonl", submitted_rows)
    _write_jsonl(output_dir / "draft_judgments_enriched.jsonl", draft_rows)
    _write_jsonl(output_dir / "invalid_judgments_enriched.jsonl", invalid_rows)
    _write_jsonl(output_dir / "sample_coverage.jsonl", coverage_rows)
    (output_dir / "duplicate_issues.json").write_text(
        json.dumps(duplicate_issues, indent=2),
        encoding="utf-8",
    )
    (output_dir / "consolidation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "consolidation_summary.md").write_text(
        build_markdown_summary(summary),
        encoding="utf-8",
    )
    return summary


def build_markdown_summary(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Human Eval Consolidation Summary")
    lines.append("")
    lines.append(f"- Accepted submitted judgments: `{summary['accepted_submitted_rows']}`")
    lines.append(f"- Accepted draft judgments kept for audit only: `{summary['accepted_draft_rows']}`")
    lines.append(f"- Invalid rows excluded from final-countable merge: `{summary['invalid_rows']}`")
    lines.append(f"- Duplicate issues detected: `{summary['duplicate_issue_count']}`")
    lines.append("")
    lines.append("## Reviewer Progress")
    lines.append("")
    for reviewer, data in summary["reviewer_progress"].items():
        lines.append(f"### {reviewer}")
        lines.append("")
        lines.append(f"- Assigned: `{data['assigned']}`")
        lines.append(f"- Submitted currently countable: `{data['submitted_count']}`")
        lines.append(f"- Drafts preserved for audit: `{data['draft_count']}`")
        lines.append(f"- Remaining missing submissions: `{data['missing_count']}`")
        lines.append(f"- Return folder present: `{data['has_any_return_folder']}`")
        lines.append("")
    lines.append("## Sample Coverage")
    lines.append("")
    lines.append(f"- Fully complete samples: `{summary['fully_complete_samples']}`")
    lines.append(f"- Partially complete samples: `{summary['partially_complete_samples']}`")
    lines.append(f"- Unstarted samples: `{summary['unstarted_samples']}`")
    lines.append(f"- Coverage distribution: `{summary['coverage_distribution']}`")
    lines.append("")
    lines.append("## Return Inventory")
    lines.append("")
    for item in summary["return_inventory"]:
        lines.append(f"### {item['return_id']}")
        lines.append("")
        lines.append(f"- Path: `{item['path']}`")
        lines.append(f"- Enriched file present: `{item['has_enriched']}`")
        lines.append(f"- Rows seen: `{item.get('rows', 0)}`")
        if "statuses" in item:
            lines.append(f"- Status mix: `{item['statuses']}`")
        if "embedded_reviewers" in item:
            lines.append(f"- Embedded reviewer ids: `{item['embedded_reviewers']}`")
        if "accepted_submitted_rows_seen" in item:
            lines.append(f"- Accepted submitted rows seen: `{item['accepted_submitted_rows_seen']}`")
        if "draft_rows_seen" in item:
            lines.append(f"- Draft rows seen: `{item['draft_rows_seen']}`")
        if "invalid_rows_seen" in item:
            lines.append(f"- Invalid rows seen: `{item['invalid_rows_seen']}`")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanitize and consolidate received human-eval reviewer returns.")
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=DEFAULT_STUDY_DIR,
        help="Study directory containing assignments, data, and reviewer_returns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sanitized consolidated artifacts. Defaults under study_dir/consolidated/.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_OUTPUT_LABEL,
        help="Output label used when --output-dir is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (study_dir / "consolidated" / args.label)
    summary = consolidate(study_dir, output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
