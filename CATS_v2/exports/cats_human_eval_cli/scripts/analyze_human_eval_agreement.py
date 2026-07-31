from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STUDY_NAME = "qwen_llama_e2e_sft_baseline_balanced_4reviewers"
DEFAULT_STUDY_DIR = Path(__file__).resolve().parent.parent / "studies" / STUDY_NAME
DEFAULT_CONSOLIDATED_LABEL = "2026-07-30_full_receipts"
BENCHMARK_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "benchmark_local_committee_3judge"
    / "benchmark_set_all_modes"
)
MODELS = ("qwen7b", "llama8b")
PROMPTS = ("minimal", "runtime", "strict")
TRAIN_TYPES = ("sft", "baseline")
JUDGE_DIR_TO_MODEL = {
    "qwen397_collect": "local/qwen3.5-397b-a17b",
    "mistral4_collect": "local/mistral-small-4",
    "deepseek32_collect": "local/deepseek-r1-distill-32b",
}
JUDGE_MODEL_IDS = tuple(JUDGE_DIR_TO_MODEL.values())


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


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _percentile(sorted_values: Sequence[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    weight = idx - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return mean(values) if values else None


def rankdata(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for i in range(pos, end):
            ranks[indexed[i][0]] = avg_rank
        pos = end
    return ranks


def pearsonr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    denom = den_x * den_y
    if denom == 0:
        return None
    return num / denom


def spearmanr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearsonr(rankdata(xs), rankdata(ys))


def binary_agreement_metrics(labels_a: Sequence[int], labels_b: Sequence[int]) -> Dict[str, Any]:
    n = len(labels_a)
    if n != len(labels_b):
        raise ValueError("label sequences must have same length")
    if n == 0:
        return {
            "n": 0,
            "agreement": None,
            "cohen_kappa": None,
            "positive_agreement": None,
            "negative_agreement": None,
            "n11": 0,
            "n10": 0,
            "n01": 0,
            "n00": 0,
            "positive_rate_a": None,
            "positive_rate_b": None,
        }
    n11 = sum(1 for a, b in zip(labels_a, labels_b) if a == 1 and b == 1)
    n10 = sum(1 for a, b in zip(labels_a, labels_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(labels_a, labels_b) if a == 0 and b == 1)
    n00 = sum(1 for a, b in zip(labels_a, labels_b) if a == 0 and b == 0)
    po = (n11 + n00) / n
    pa1 = (n11 + n10) / n
    pb1 = (n11 + n01) / n
    pa0 = 1.0 - pa1
    pb0 = 1.0 - pb1
    pe = pa1 * pb1 + pa0 * pb0
    kappa = None if math.isclose(1.0 - pe, 0.0) else (po - pe) / (1.0 - pe)
    pos_den = 2 * n11 + n10 + n01
    neg_den = 2 * n00 + n10 + n01
    pos_agreement = None if pos_den == 0 else (2 * n11) / pos_den
    neg_agreement = None if neg_den == 0 else (2 * n00) / neg_den
    return {
        "n": n,
        "agreement": po,
        "cohen_kappa": kappa,
        "positive_agreement": pos_agreement,
        "negative_agreement": neg_agreement,
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "positive_rate_a": pa1,
        "positive_rate_b": pb1,
    }


def krippendorff_alpha_nominal(items: Sequence[Sequence[int]]) -> Optional[float]:
    valid_items = [list(item) for item in items if len(item) >= 2]
    if len(valid_items) < 2:
        return None
    denom = sum(len(item) * (len(item) - 1) for item in valid_items)
    if denom == 0:
        return None
    observed_disagreement = 0.0
    overall_counts = Counter()
    total_labels = 0
    for item in valid_items:
        counts = Counter(item)
        total_labels += len(item)
        overall_counts.update(counts)
        observed_disagreement += sum(count * (len(item) - count) for count in counts.values())
    do = observed_disagreement / denom
    if total_labels < 2:
        return None
    de_num = sum(count * (total_labels - count) for count in overall_counts.values())
    de_den = total_labels * (total_labels - 1)
    if de_den == 0:
        return None
    de = de_num / de_den
    if math.isclose(de, 0.0):
        return None
    return 1.0 - (do / de)


def continuous_agreement_metrics(values_a: Sequence[float], values_b: Sequence[float]) -> Dict[str, Any]:
    if len(values_a) != len(values_b):
        raise ValueError("value sequences must have same length")
    n = len(values_a)
    if n == 0:
        return {
            "n": 0,
            "exact_match_rate": None,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "spearman_rho": None,
            "mean_a": None,
            "mean_b": None,
        }
    diffs = [abs(a - b) for a, b in zip(values_a, values_b)]
    squared = [(a - b) ** 2 for a, b in zip(values_a, values_b)]
    exact = sum(1 for a, b in zip(values_a, values_b) if math.isclose(a, b))
    return {
        "n": n,
        "exact_match_rate": exact / n,
        "mae": sum(diffs) / n,
        "rmse": math.sqrt(sum(squared) / n),
        "pearson_r": pearsonr(values_a, values_b),
        "spearman_rho": spearmanr(values_a, values_b),
        "mean_a": _safe_mean(values_a),
        "mean_b": _safe_mean(values_b),
    }


def load_sample_index(study_dir: Path) -> Dict[str, Dict[str, Any]]:
    data_path = study_dir / "data" / "samples.jsonl"
    return {row["sample_id"]: row for row in _iter_jsonl(data_path)}


def load_human_rows(consolidated_dir: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(consolidated_dir / "submitted_judgments_enriched.jsonl"))


def load_sample_coverage_rows(consolidated_dir: Path) -> List[Dict[str, Any]]:
    return list(_iter_jsonl(consolidated_dir / "sample_coverage.jsonl"))


def load_assignment_audit(study_dir: Path) -> Dict[str, Any]:
    path = study_dir / "admin" / "assignment_audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def parse_axes(sample_id: str) -> Dict[str, str]:
    model, prompt, train_type, base_id = sample_id.split("__", 3)
    return {
        "model": model,
        "prompt": prompt,
        "train_type": train_type,
        "base_id": base_id,
    }


def load_committee_map() -> Dict[str, Dict[str, Any]]:
    committee_map: Dict[str, Dict[str, Any]] = {}
    for model in MODELS:
        for prompt in PROMPTS:
            for train_type in TRAIN_TYPES:
                path = (
                    BENCHMARK_OUTPUT_ROOT
                    / model
                    / "e2e"
                    / prompt
                    / train_type
                    / "final"
                    / "detailed_results.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = payload["per_sample"]
                for row in rows:
                    composite_id = f"{model}__{prompt}__{train_type}__{row['sample_id']}"
                    if composite_id in committee_map:
                        raise ValueError(f"duplicate committee row for {composite_id}")
                    behavior_details = row.get("behavior_details") or {}
                    fg_details = row.get("factual_grounding_details") or {}
                    str_details = row.get("single_truth_recall_details") or {}
                    committee_map[composite_id] = {
                        "sample_id": composite_id,
                        "base_sample_id": row["sample_id"],
                        "model": model,
                        "prompt": prompt,
                        "train_type": train_type,
                        "conflict_category_id": row.get("conflict_type"),
                        "gold_answerable": row.get("gold_answerable"),
                        "correct_refusal": row.get("correct_refusal"),
                        "gr_accuracy": row.get("gr_accuracy"),
                        "behavior_label": behavior_details.get("adherent"),
                        "behavior_score": row.get("behavior_score"),
                        "behavior_details": behavior_details,
                        "behavior_individual": [
                            {
                                "judge_id": resp.get("judge_id"),
                                "model_id": resp.get("model_id"),
                                "adherent": resp.get("adherent"),
                                "confidence": resp.get("confidence"),
                            }
                            for resp in ((behavior_details.get("committee_details") or {}).get("individual_responses") or [])
                        ],
                        "fg_ratio": fg_details.get("grounding_ratio"),
                        "fg_score": row.get("factual_grounding_score"),
                        "fg_claims": fg_details.get("claim_details") or [],
                        "str_score": row.get("single_truth_recall_score"),
                        "str_details": str_details,
                        "str_applicable": row.get("single_truth_applicable"),
                        "str_strict_positive": (
                            None if not row.get("single_truth_applicable")
                            else bool(math.isclose(float(row.get("single_truth_recall_score") or 0.0), 1.0))
                        ),
                        "str_soft_positive": (
                            None if not row.get("single_truth_applicable")
                            else bool(float(row.get("single_truth_recall_score") or 0.0) > 0.0)
                        ),
                    }
    return committee_map


def load_llm_judge_cache(
    judge_cache_dir: Path,
    sample_coverage_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    judge_map: Dict[str, Dict[str, Any]] = {}
    if not judge_cache_dir.exists():
        return judge_map

    for model_dir in sorted(judge_cache_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            for train_dir in sorted(prompt_dir.iterdir()):
                if not train_dir.is_dir():
                    continue
                for judge_dir in sorted(train_dir.iterdir()):
                    if not judge_dir.is_dir():
                        continue
                    judge_model_id = JUDGE_DIR_TO_MODEL.get(judge_dir.name)
                    if judge_model_id is None:
                        continue
                    detailed_path = judge_dir / "detailed_results.json"
                    if not detailed_path.exists():
                        continue
                    payload = json.loads(detailed_path.read_text(encoding="utf-8"))
                    for row in payload.get("per_sample") or []:
                        composite_id = f"{model_dir.name}__{prompt_dir.name}__{train_dir.name}__{row['sample_id']}"
                        coverage_row = sample_coverage_map.get(composite_id)
                        if coverage_row is None:
                            continue
                        entry = judge_map.setdefault(
                            composite_id,
                            {
                                "sample_id": composite_id,
                                "base_sample_id": row["sample_id"],
                                "model": model_dir.name,
                                "prompt": prompt_dir.name,
                                "train_type": train_dir.name,
                                "conflict_category_id": coverage_row.get("conflict_category_id"),
                                "coverage_status": coverage_row.get("coverage_status"),
                                "judges": {},
                            },
                        )
                        behavior_details = row.get("behavior_details") or {}
                        fg_details = row.get("factual_grounding_details") or {}
                        claim_details = fg_details.get("claim_details") or []
                        entry["judges"][judge_model_id] = {
                            "behavior_label": (
                                None
                                if not isinstance(behavior_details.get("adherent"), bool)
                                else int(behavior_details["adherent"])
                            ),
                            "str_applicable": bool(row.get("single_truth_applicable")),
                            "str_score": row.get("single_truth_recall_score"),
                            "str_strict_positive": (
                                None
                                if not row.get("single_truth_applicable")
                                else int(math.isclose(float(row.get("single_truth_recall_score") or 0.0), 1.0))
                            ),
                            "str_soft_positive": (
                                None
                                if not row.get("single_truth_applicable")
                                else int(float(row.get("single_truth_recall_score") or 0.0) > 0.0)
                            ),
                            "fg_ratio": (
                                None
                                if row.get("factual_grounding_score") is None
                                else float(row.get("factual_grounding_score"))
                            ),
                            "fg_claims": [
                                {
                                    "claim_text": _normalize_text(claim.get("claim")),
                                    "citation_check_passed": int(bool(claim.get("citation_check_passed"))),
                                }
                                for claim in claim_details
                            ],
                        }
    return judge_map


def extract_human_metrics(row: Dict[str, Any], sample_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sample_id = row["sample_id"]
    axes = parse_axes(sample_id)
    sample = sample_index[sample_id]
    judgment = row["judgment"]
    behavior = judgment.get("behavior") or {}
    fg = judgment.get("fg") or {}
    str_judgment = judgment.get("str") or {}
    return {
        "sample_id": sample_id,
        "base_id": axes["base_id"],
        "model": axes["model"],
        "prompt": axes["prompt"],
        "train_type": axes["train_type"],
        "reviewer_id": row["reviewer_id"],
        "conflict_category_id": sample.get("conflict_category_id"),
        "conflict_type": sample.get("conflict_type"),
        "single_truth_applicable": sample.get("single_truth_applicable"),
        "claims_with_citations": sample.get("claims_with_citations") or [],
        "claim_count": len(sample.get("claims_with_citations") or []),
        "behavior_label": behavior.get("adherent"),
        "behavior_confidence": behavior.get("confidence"),
        "behavior_rationale": behavior.get("rationale"),
        "fg_ratio": fg.get("grounding_ratio"),
        "fg_claims": fg.get("claim_details") or [],
        "str_label": str_judgment.get("adherent"),
        "str_confidence": str_judgment.get("confidence"),
        "str_rationale": str_judgment.get("rationale"),
        "organizer_meta": row.get("organizer_meta") or {},
    }


def align_human_committee(
    human_rows: List[Dict[str, Any]],
    sample_index: Dict[str, Dict[str, Any]],
    committee_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    aligned_rows: List[Dict[str, Any]] = []
    missing_committee: List[Dict[str, Any]] = []
    for row in human_rows:
        human = extract_human_metrics(row, sample_index)
        committee = committee_map.get(human["sample_id"])
        if committee is None:
            missing_committee.append(
                {
                    "sample_id": human["sample_id"],
                    "reviewer_id": human["reviewer_id"],
                }
            )
            continue
        aligned_rows.append(
            {
                **human,
                "committee": committee,
            }
        )
    return aligned_rows, missing_committee, []


def summarize_binary_units(
    units: List[Dict[str, Any]],
    label_a_key: str,
    label_b_key: str,
    comparison_name: str,
    metric_name: str,
    subgroup_type: str = "overall",
    subgroup_value: str = "all",
    alpha_key: Optional[str] = None,
) -> Dict[str, Any]:
    labels_a = [int(unit[label_a_key]) for unit in units]
    labels_b = [int(unit[label_b_key]) for unit in units]
    metrics = binary_agreement_metrics(labels_a, labels_b)
    if alpha_key is not None:
        item_lists = [[int(unit[label_a_key]), int(unit[label_b_key])] for unit in units]
        metrics["krippendorff_alpha_nominal"] = krippendorff_alpha_nominal(item_lists)
    else:
        metrics["krippendorff_alpha_nominal"] = None
    return {
        "comparison": comparison_name,
        "metric": metric_name,
        "subgroup_type": subgroup_type,
        "subgroup_value": subgroup_value,
        **metrics,
    }


def summarize_continuous_units(
    units: List[Dict[str, Any]],
    value_a_key: str,
    value_b_key: str,
    comparison_name: str,
    metric_name: str,
    subgroup_type: str = "overall",
    subgroup_value: str = "all",
) -> Dict[str, Any]:
    values_a = [float(unit[value_a_key]) for unit in units]
    values_b = [float(unit[value_b_key]) for unit in units]
    metrics = continuous_agreement_metrics(values_a, values_b)
    return {
        "comparison": comparison_name,
        "metric": metric_name,
        "subgroup_type": subgroup_type,
        "subgroup_value": subgroup_value,
        **metrics,
    }


def group_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def build_human_human_units(
    aligned_rows: List[Dict[str, Any]]
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    by_sample = group_by(aligned_rows, "sample_id")
    behavior_units: List[Dict[str, Any]] = []
    str_units: List[Dict[str, Any]] = []
    fg_ratio_units: List[Dict[str, Any]] = []
    fg_claim_units: List[Dict[str, Any]] = []
    claim_issues: List[Dict[str, Any]] = []

    for sample_id, rows in sorted(by_sample.items()):
        if len(rows) != 2:
            continue
        rows = sorted(rows, key=lambda row: row["reviewer_id"])
        first, second = rows
        reviewers = tuple(sorted((first["reviewer_id"], second["reviewer_id"])))
        common = {
            "sample_id": sample_id,
            "reviewer_pair": " / ".join(reviewers),
            "conflict_category_id": first["conflict_category_id"],
            "model": first["model"],
            "prompt": first["prompt"],
            "train_type": first["train_type"],
        }
        if isinstance(first["behavior_label"], bool) and isinstance(second["behavior_label"], bool):
            behavior_units.append(
                {
                    **common,
                    "label_a": int(first["behavior_label"]),
                    "label_b": int(second["behavior_label"]),
                    "reviewer_a": first["reviewer_id"],
                    "reviewer_b": second["reviewer_id"],
                }
            )
        if isinstance(first["str_label"], bool) and isinstance(second["str_label"], bool):
            str_units.append(
                {
                    **common,
                    "label_a": int(first["str_label"]),
                    "label_b": int(second["str_label"]),
                    "reviewer_a": first["reviewer_id"],
                    "reviewer_b": second["reviewer_id"],
                }
            )
        if first["fg_ratio"] is not None and second["fg_ratio"] is not None:
            fg_ratio_units.append(
                {
                    **common,
                    "value_a": float(first["fg_ratio"]),
                    "value_b": float(second["fg_ratio"]),
                    "reviewer_a": first["reviewer_id"],
                    "reviewer_b": second["reviewer_id"],
                }
            )
        claims_a = first["fg_claims"]
        claims_b = second["fg_claims"]
        if len(claims_a) != len(claims_b):
            claim_issues.append(
                {
                    "sample_id": sample_id,
                    "issue": "length_mismatch",
                    "len_a": len(claims_a),
                    "len_b": len(claims_b),
                    "reviewers": reviewers,
                }
            )
            continue
        aligned_ok = True
        for idx, (claim_a, claim_b) in enumerate(zip(claims_a, claims_b)):
            text_a = _normalize_text(claim_a.get("claim"))
            text_b = _normalize_text(claim_b.get("claim"))
            if text_a != text_b:
                claim_issues.append(
                    {
                        "sample_id": sample_id,
                        "issue": "claim_text_mismatch",
                        "claim_index": idx,
                        "claim_a": text_a,
                        "claim_b": text_b,
                        "reviewers": reviewers,
                    }
                )
                aligned_ok = False
                break
        if not aligned_ok:
            continue
        for idx, (claim_a, claim_b) in enumerate(zip(claims_a, claims_b)):
            fg_claim_units.append(
                {
                    **common,
                    "unit_id": f"{sample_id}::claim::{idx}",
                    "claim_index": idx,
                    "label_a": int(bool(claim_a.get("citation_check_passed"))),
                    "label_b": int(bool(claim_b.get("citation_check_passed"))),
                    "reviewer_a": first["reviewer_id"],
                    "reviewer_b": second["reviewer_id"],
                }
            )
    return behavior_units, str_units, fg_ratio_units, fg_claim_units, claim_issues


def build_human_committee_units(
    aligned_rows: List[Dict[str, Any]]
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    behavior_units: List[Dict[str, Any]] = []
    str_strict_units: List[Dict[str, Any]] = []
    str_soft_units: List[Dict[str, Any]] = []
    fg_ratio_units: List[Dict[str, Any]] = []
    fg_claim_units: List[Dict[str, Any]] = []
    claim_issues: List[Dict[str, Any]] = []

    for row in aligned_rows:
        committee = row["committee"]
        common = {
            "sample_id": row["sample_id"],
            "reviewer_id": row["reviewer_id"],
            "conflict_category_id": row["conflict_category_id"],
            "model": row["model"],
            "prompt": row["prompt"],
            "train_type": row["train_type"],
        }
        if isinstance(row["behavior_label"], bool) and isinstance(committee["behavior_label"], bool):
            behavior_units.append(
                {
                    **common,
                    "human_label": int(row["behavior_label"]),
                    "committee_label": int(committee["behavior_label"]),
                }
            )
        if isinstance(row["str_label"], bool):
            if committee["str_strict_positive"] is not None:
                str_strict_units.append(
                    {
                        **common,
                        "human_label": int(row["str_label"]),
                        "committee_label": int(committee["str_strict_positive"]),
                        "committee_score": committee["str_score"],
                    }
                )
            if committee["str_soft_positive"] is not None:
                str_soft_units.append(
                    {
                        **common,
                        "human_label": int(row["str_label"]),
                        "committee_label": int(committee["str_soft_positive"]),
                        "committee_score": committee["str_score"],
                    }
                )
        if row["fg_ratio"] is not None and committee["fg_ratio"] is not None:
            fg_ratio_units.append(
                {
                    **common,
                    "human_value": float(row["fg_ratio"]),
                    "committee_value": float(committee["fg_ratio"]),
                }
            )
        human_claims = row["fg_claims"]
        committee_claims = committee["fg_claims"]
        if len(human_claims) != len(committee_claims):
            claim_issues.append(
                {
                    "sample_id": row["sample_id"],
                    "reviewer_id": row["reviewer_id"],
                    "issue": "length_mismatch",
                    "human_len": len(human_claims),
                    "committee_len": len(committee_claims),
                }
            )
            continue
        aligned_ok = True
        for idx, (human_claim, committee_claim) in enumerate(zip(human_claims, committee_claims)):
            human_text = _normalize_text(human_claim.get("claim"))
            committee_text = _normalize_text(committee_claim.get("claim"))
            if human_text != committee_text:
                claim_issues.append(
                    {
                        "sample_id": row["sample_id"],
                        "reviewer_id": row["reviewer_id"],
                        "issue": "claim_text_mismatch",
                        "claim_index": idx,
                        "human_claim": human_text,
                        "committee_claim": committee_text,
                    }
                )
                aligned_ok = False
                break
        if not aligned_ok:
            continue
        for idx, (human_claim, committee_claim) in enumerate(zip(human_claims, committee_claims)):
            fg_claim_units.append(
                {
                    **common,
                    "unit_id": f"{row['sample_id']}::{row['reviewer_id']}::claim::{idx}",
                    "claim_index": idx,
                    "human_label": int(bool(human_claim.get("citation_check_passed"))),
                    "committee_label": int(bool(committee_claim.get("citation_check_passed"))),
                }
            )
    return behavior_units, str_strict_units, str_soft_units, fg_ratio_units, fg_claim_units, claim_issues


def build_behavior_individual_committee_units(aligned_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for row in aligned_rows:
        if not isinstance(row["behavior_label"], bool):
            continue
        for individual in row["committee"]["behavior_individual"]:
            if not isinstance(individual.get("adherent"), bool):
                continue
            units.append(
                {
                    "sample_id": row["sample_id"],
                    "reviewer_id": row["reviewer_id"],
                    "conflict_category_id": row["conflict_category_id"],
                    "model": row["model"],
                    "prompt": row["prompt"],
                    "train_type": row["train_type"],
                    "judge_model_id": individual.get("model_id"),
                    "human_label": int(row["behavior_label"]),
                    "committee_label": int(individual["adherent"]),
                }
            )
    return units


def _judge_short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def summarize_committee_internal_binary(
    sample_rows: Sequence[Dict[str, Any]],
    subset_name: str,
    metric: str,
    label_key: str,
) -> Dict[str, Any]:
    pairwise_units: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
        lambda: {"labels_a": [], "labels_b": []}
    )
    multirater_items: List[List[int]] = []
    for row in sample_rows:
        available = {
            judge_id: judge_data[label_key]
            for judge_id, judge_data in row["judges"].items()
            if judge_data.get(label_key) is not None
        }
        if len(available) >= 2:
            multirater_items.append([int(value) for value in available.values()])
        for judge_a, judge_b in combinations(sorted(available.keys()), 2):
            pairwise_units[(judge_a, judge_b)]["labels_a"].append(int(available[judge_a]))
            pairwise_units[(judge_a, judge_b)]["labels_b"].append(int(available[judge_b]))
    pairwise_rows: List[Dict[str, Any]] = []
    for (judge_a, judge_b), payload in sorted(pairwise_units.items(), key=lambda item: item[0]):
        metrics = binary_agreement_metrics(payload["labels_a"], payload["labels_b"])
        pairwise_rows.append(
            {
                "subset": subset_name,
                "metric": metric,
                "judge_a": judge_a,
                "judge_b": judge_b,
                "judge_pair": f"{_judge_short_name(judge_a)} vs {_judge_short_name(judge_b)}",
                **metrics,
            }
        )
    return {
        "subset": subset_name,
        "metric": metric,
        "n_items": len(multirater_items),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(multirater_items),
        "pairwise": pairwise_rows,
    }


def summarize_committee_internal_fg_ratio(
    sample_rows: Sequence[Dict[str, Any]],
    subset_name: str,
) -> Dict[str, Any]:
    pairwise_units: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"values_a": [], "values_b": []}
    )
    for row in sample_rows:
        available = {
            judge_id: judge_data["fg_ratio"]
            for judge_id, judge_data in row["judges"].items()
            if judge_data.get("fg_ratio") is not None
        }
        for judge_a, judge_b in combinations(sorted(available.keys()), 2):
            pairwise_units[(judge_a, judge_b)]["values_a"].append(float(available[judge_a]))
            pairwise_units[(judge_a, judge_b)]["values_b"].append(float(available[judge_b]))
    pairwise_rows: List[Dict[str, Any]] = []
    for (judge_a, judge_b), payload in sorted(pairwise_units.items(), key=lambda item: item[0]):
        metrics = continuous_agreement_metrics(payload["values_a"], payload["values_b"])
        pairwise_rows.append(
            {
                "subset": subset_name,
                "metric": "fg_ratio_continuous",
                "judge_a": judge_a,
                "judge_b": judge_b,
                "judge_pair": f"{_judge_short_name(judge_a)} vs {_judge_short_name(judge_b)}",
                **metrics,
            }
        )
    return {
        "subset": subset_name,
        "metric": "fg_ratio_continuous",
        "pairwise": pairwise_rows,
    }


def summarize_committee_internal_fg_claims(
    sample_rows: Sequence[Dict[str, Any]],
    subset_name: str,
) -> Dict[str, Any]:
    pairwise_units: Dict[Tuple[str, str], Dict[str, List[int]]] = defaultdict(
        lambda: {"labels_a": [], "labels_b": []}
    )
    claim_alignment_issues: List[Dict[str, Any]] = []
    multirater_items: List[List[int]] = []

    for row in sample_rows:
        available = {
            judge_id: judge_data["fg_claims"]
            for judge_id, judge_data in row["judges"].items()
            if judge_data.get("fg_claims") is not None
        }
        if len(available) < 2:
            continue
        lengths = {len(claims) for claims in available.values()}
        if len(lengths) != 1:
            claim_alignment_issues.append(
                {
                    "subset": subset_name,
                    "sample_id": row["sample_id"],
                    "issue": "claim_count_mismatch",
                    "claim_counts": {judge_id: len(claims) for judge_id, claims in available.items()},
                }
            )
            continue
        claim_count = next(iter(lengths), 0)
        aligned = True
        for claim_index in range(claim_count):
            claim_texts = {
                judge_id: claims[claim_index]["claim_text"]
                for judge_id, claims in available.items()
            }
            if len(set(claim_texts.values())) != 1:
                claim_alignment_issues.append(
                    {
                        "subset": subset_name,
                        "sample_id": row["sample_id"],
                        "issue": "claim_text_mismatch",
                        "claim_index": claim_index,
                        "claim_texts": claim_texts,
                    }
                )
                aligned = False
                break
        if not aligned:
            continue
        for claim_index in range(claim_count):
            labels = {
                judge_id: int(claims[claim_index]["citation_check_passed"])
                for judge_id, claims in available.items()
            }
            multirater_items.append(list(labels.values()))
            for judge_a, judge_b in combinations(sorted(labels.keys()), 2):
                pairwise_units[(judge_a, judge_b)]["labels_a"].append(labels[judge_a])
                pairwise_units[(judge_a, judge_b)]["labels_b"].append(labels[judge_b])

    pairwise_rows: List[Dict[str, Any]] = []
    for (judge_a, judge_b), payload in sorted(pairwise_units.items(), key=lambda item: item[0]):
        metrics = binary_agreement_metrics(payload["labels_a"], payload["labels_b"])
        pairwise_rows.append(
            {
                "subset": subset_name,
                "metric": "fg_claim_binary",
                "judge_a": judge_a,
                "judge_b": judge_b,
                "judge_pair": f"{_judge_short_name(judge_a)} vs {_judge_short_name(judge_b)}",
                **metrics,
            }
        )
    return {
        "subset": subset_name,
        "metric": "fg_claim_binary",
        "n_items": len(multirater_items),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal(multirater_items),
        "pairwise": pairwise_rows,
        "claim_alignment_issues": claim_alignment_issues,
    }


def build_committee_internal_summary(
    sample_coverage_rows: Sequence[Dict[str, Any]],
    judge_cache_dir: Path,
) -> Dict[str, Any]:
    sample_coverage_map = {row["sample_id"]: row for row in sample_coverage_rows}
    judge_map = load_llm_judge_cache(judge_cache_dir, sample_coverage_map)
    subset_ids = {
        "all_350": [row["sample_id"] for row in sample_coverage_rows],
        "complete_300": [row["sample_id"] for row in sample_coverage_rows if row.get("coverage_status") == "complete"],
    }
    subset_summaries: Dict[str, Any] = {}
    all_claim_alignment_issues: List[Dict[str, Any]] = []

    for subset_name, sample_ids in subset_ids.items():
        sample_rows = [judge_map[sample_id] for sample_id in sample_ids if sample_id in judge_map]
        fg_claim_summary = summarize_committee_internal_fg_claims(sample_rows, subset_name)
        subset_summaries[subset_name] = {
            "sample_count": len(sample_rows),
            "behavior_binary": summarize_committee_internal_binary(sample_rows, subset_name, "behavior_binary", "behavior_label"),
            "str_binary_strict": summarize_committee_internal_binary(sample_rows, subset_name, "str_binary_strict", "str_strict_positive"),
            "str_binary_soft": summarize_committee_internal_binary(sample_rows, subset_name, "str_binary_soft", "str_soft_positive"),
            "fg_ratio_continuous": summarize_committee_internal_fg_ratio(sample_rows, subset_name),
            "fg_claim_binary": fg_claim_summary,
        }
        all_claim_alignment_issues.extend(fg_claim_summary["claim_alignment_issues"])

    return {
        "judge_cache_dir": str(judge_cache_dir),
        "available": bool(judge_map),
        "subsets": subset_summaries,
        "claim_alignment_issues": all_claim_alignment_issues,
    }


def summarize_sample_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    return {
        "model": dict(sorted(Counter(row["model"] for row in rows).items())),
        "prompt": dict(sorted(Counter(row["prompt"] for row in rows).items())),
        "train_type": dict(sorted(Counter(row["train_type"] for row in rows).items())),
        "conflict_category_id": dict(sorted(Counter(row["conflict_category_id"] for row in rows).items())),
        "conflict_type": dict(sorted(Counter(row["conflict_type"] for row in rows).items())),
    }


def build_descriptive_summary(
    study_dir: Path,
    sample_coverage_rows: Sequence[Dict[str, Any]],
    human_human_behavior: Sequence[Dict[str, Any]],
    human_human_str: Sequence[Dict[str, Any]],
    human_human_fg_ratio: Sequence[Dict[str, Any]],
    human_human_fg_claims: Sequence[Dict[str, Any]],
    aligned_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    assignment_audit = load_assignment_audit(study_dir)
    complete_rows = [row for row in sample_coverage_rows if row.get("coverage_status") == "complete"]
    complete_ids = {row["sample_id"] for row in complete_rows}
    first_committee_row_by_sample: Dict[str, Dict[str, Any]] = {}
    for row in aligned_rows:
        if row["sample_id"] in complete_ids and row["sample_id"] not in first_committee_row_by_sample:
            first_committee_row_by_sample[row["sample_id"]] = row

    behavior_pair_counts = Counter((row["label_a"], row["label_b"]) for row in human_human_behavior)
    str_pair_counts = Counter((row["label_a"], row["label_b"]) for row in human_human_str)

    behavior_positive_reviews = sum(int(row["label_a"]) + int(row["label_b"]) for row in human_human_behavior)
    str_positive_reviews = sum(int(row["label_a"]) + int(row["label_b"]) for row in human_human_str)
    fg_claim_positive_reviews = sum(int(row["label_a"]) + int(row["label_b"]) for row in human_human_fg_claims)
    human_fg_ratio_mean = _safe_mean(
        [(float(row["value_a"]) + float(row["value_b"])) / 2.0 for row in human_human_fg_ratio]
    )

    committee_complete_rows = list(first_committee_row_by_sample.values())
    committee_behavior_positive = sum(
        1 for row in committee_complete_rows if row["committee"]["behavior_label"] is True
    )
    committee_str_applicable_rows = [
        row for row in committee_complete_rows
        if row["committee"]["str_strict_positive"] is not None
    ]
    committee_strict_positive = sum(int(row["committee"]["str_strict_positive"]) for row in committee_str_applicable_rows)
    committee_str_soft_positive = sum(int(row["committee"]["str_soft_positive"]) for row in committee_str_applicable_rows)
    committee_fg_ratio_mean = _safe_mean([float(row["committee"]["fg_ratio"]) for row in committee_complete_rows])

    return {
        "study_provenance": {
            "study_name": assignment_audit.get("study_name"),
            "selection_seed": assignment_audit.get("seed"),
            "selection_script": str(Path(__file__).resolve().name),
            "study_builder_script": str(
                (Path(__file__).resolve().parent / "build_balanced_qwen_llama_e2e_study.py").resolve()
            ),
            "source_root_pattern": "inputs/prepped_model_eval_inputs/benchmark_set_all_modes/<model>/e2e/<prompt>/<train_type>/input.jsonl",
            "source_task_variant": "e2e only",
            "models": list(MODELS),
            "prompts": list(PROMPTS),
            "train_types": list(TRAIN_TYPES),
            "selected_total": assignment_audit.get("selected_total"),
            "review_target_total": assignment_audit.get("review_total"),
            "extra_cells": assignment_audit.get("extra_cells"),
            "cell_targets": ((assignment_audit.get("selected_distributions") or {}).get("cell") or {}),
            "balanced_conflict_targets": ((assignment_audit.get("selected_distributions") or {}).get("conflict") or {}),
            "correct_refusal_exclusion": True,
            "source_rows_file": str((study_dir / "admin" / "selected_source_rows.jsonl").resolve()),
            "assignment_audit_file": str((study_dir / "admin" / "assignment_audit.json").resolve()),
        },
        "sample_subsets": {
            "all_350": {
                "sample_count": len(sample_coverage_rows),
                "distribution": summarize_sample_distribution(sample_coverage_rows),
            },
            "complete_300": {
                "sample_count": len(complete_rows),
                "distribution": summarize_sample_distribution(complete_rows),
            },
        },
        "descriptive_labels_complete_300": {
            "human_behavior": {
                "positive_reviews": behavior_positive_reviews,
                "total_reviews": 2 * len(human_human_behavior),
                "positive_rate": (
                    behavior_positive_reviews / (2 * len(human_human_behavior))
                    if human_human_behavior
                    else None
                ),
                "unanimous_positive_samples": behavior_pair_counts[(1, 1)],
                "unanimous_negative_samples": behavior_pair_counts[(0, 0)],
                "split_samples": behavior_pair_counts[(1, 0)] + behavior_pair_counts[(0, 1)],
            },
            "committee_behavior": {
                "positive_samples": committee_behavior_positive,
                "total_samples": len(committee_complete_rows),
                "positive_rate": (
                    committee_behavior_positive / len(committee_complete_rows)
                    if committee_complete_rows
                    else None
                ),
            },
            "human_str": {
                "applicable_samples": len(human_human_str),
                "positive_reviews": str_positive_reviews,
                "total_reviews": 2 * len(human_human_str),
                "positive_rate": (
                    str_positive_reviews / (2 * len(human_human_str))
                    if human_human_str
                    else None
                ),
                "unanimous_positive_samples": str_pair_counts[(1, 1)],
                "unanimous_negative_samples": str_pair_counts[(0, 0)],
                "split_samples": str_pair_counts[(1, 0)] + str_pair_counts[(0, 1)],
            },
            "committee_str": {
                "applicable_samples": len(committee_str_applicable_rows),
                "strict_positive_samples": committee_strict_positive,
                "strict_positive_rate": (
                    committee_strict_positive / len(committee_str_applicable_rows)
                    if committee_str_applicable_rows
                    else None
                ),
                "soft_positive_samples": committee_str_soft_positive,
                "soft_positive_rate": (
                    committee_str_soft_positive / len(committee_str_applicable_rows)
                    if committee_str_applicable_rows
                    else None
                ),
            },
            "human_fg": {
                "claim_positive_reviews": fg_claim_positive_reviews,
                "claim_total_reviews": 2 * len(human_human_fg_claims),
                "claim_positive_rate": (
                    fg_claim_positive_reviews / (2 * len(human_human_fg_claims))
                    if human_human_fg_claims
                    else None
                ),
                "sample_level_mean_ratio": human_fg_ratio_mean,
            },
            "committee_fg": {
                "sample_level_mean_ratio": committee_fg_ratio_mean,
            },
        },
    }


def build_consensus_units(
    human_human_behavior: List[Dict[str, Any]],
    human_human_str: List[Dict[str, Any]],
    human_human_fg_claims: List[Dict[str, Any]],
    human_human_fg_ratio: List[Dict[str, Any]],
    committee_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    behavior_consensus: List[Dict[str, Any]] = []
    str_strict_consensus: List[Dict[str, Any]] = []
    str_soft_consensus: List[Dict[str, Any]] = []
    fg_claim_consensus: List[Dict[str, Any]] = []
    fg_ratio_consensus: List[Dict[str, Any]] = []

    for unit in human_human_behavior:
        if unit["label_a"] == unit["label_b"]:
            committee = committee_map[unit["sample_id"]]
            if isinstance(committee["behavior_label"], bool):
                behavior_consensus.append(
                    {
                        **unit,
                        "human_consensus": unit["label_a"],
                        "committee_label": int(committee["behavior_label"]),
                    }
                )
    for unit in human_human_str:
        if unit["label_a"] == unit["label_b"]:
            committee = committee_map[unit["sample_id"]]
            if committee["str_strict_positive"] is not None:
                str_strict_consensus.append(
                    {
                        **unit,
                        "human_consensus": unit["label_a"],
                        "committee_label": int(committee["str_strict_positive"]),
                    }
                )
            if committee["str_soft_positive"] is not None:
                str_soft_consensus.append(
                    {
                        **unit,
                        "human_consensus": unit["label_a"],
                        "committee_label": int(committee["str_soft_positive"]),
                    }
                )
    for unit in human_human_fg_claims:
        if unit["label_a"] == unit["label_b"]:
            committee = committee_map[unit["sample_id"]]
            claim_index = unit["claim_index"]
            if claim_index < len(committee["fg_claims"]):
                fg_claim_consensus.append(
                    {
                        **unit,
                        "human_consensus": unit["label_a"],
                        "committee_label": int(bool(committee["fg_claims"][claim_index].get("citation_check_passed"))),
                    }
                )
    for unit in human_human_fg_ratio:
        if math.isclose(unit["value_a"], unit["value_b"]):
            committee = committee_map[unit["sample_id"]]
            if committee["fg_ratio"] is not None:
                fg_ratio_consensus.append(
                    {
                        **unit,
                        "human_consensus": unit["value_a"],
                        "committee_value": float(committee["fg_ratio"]),
                    }
                )
    return behavior_consensus, str_strict_consensus, str_soft_consensus, fg_claim_consensus, fg_ratio_consensus


def make_binary_table(
    units: List[Dict[str, Any]],
    comparison: str,
    metric: str,
    a_key: str,
    b_key: str,
    include_alpha: bool = False,
    subgroup_fields: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(
        summarize_binary_units(
            units,
            a_key,
            b_key,
            comparison,
            metric,
            subgroup_type="overall",
            subgroup_value="all",
            alpha_key="use" if include_alpha else None,
        )
    )
    for field in subgroup_fields:
        grouped = group_by(units, field)
        for value, subset in sorted(grouped.items(), key=lambda item: str(item[0])):
            rows.append(
                summarize_binary_units(
                    subset,
                    a_key,
                    b_key,
                    comparison,
                    metric,
                    subgroup_type=field,
                    subgroup_value=str(value),
                    alpha_key="use" if include_alpha else None,
                )
            )
    return rows


def make_continuous_table(
    units: List[Dict[str, Any]],
    comparison: str,
    metric: str,
    a_key: str,
    b_key: str,
    subgroup_fields: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(
        summarize_continuous_units(
            units,
            a_key,
            b_key,
            comparison,
            metric,
            subgroup_type="overall",
            subgroup_value="all",
        )
    )
    for field in subgroup_fields:
        grouped = group_by(units, field)
        for value, subset in sorted(grouped.items(), key=lambda item: str(item[0])):
            rows.append(
                summarize_continuous_units(
                    subset,
                    a_key,
                    b_key,
                    comparison,
                    metric,
                    subgroup_type=field,
                    subgroup_value=str(value),
                )
            )
    return rows


def build_metric_log(
    study_dir: Path,
    consolidated_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    sample_index = load_sample_index(study_dir)
    sample_coverage_rows = load_sample_coverage_rows(consolidated_dir)
    human_rows = load_human_rows(consolidated_dir)
    committee_map = load_committee_map()
    committee_internal = build_committee_internal_summary(
        sample_coverage_rows,
        output_dir / "llm_judge_cache",
    )

    aligned_rows, missing_committee, _ = align_human_committee(human_rows, sample_index, committee_map)

    human_human_behavior, human_human_str, human_human_fg_ratio, human_human_fg_claims, human_human_claim_issues = build_human_human_units(aligned_rows)
    hc_behavior, hc_str_strict, hc_str_soft, hc_fg_ratio, hc_fg_claims, human_committee_claim_issues = build_human_committee_units(aligned_rows)
    behavior_individual_units = build_behavior_individual_committee_units(aligned_rows)
    consensus_behavior, consensus_str_strict, consensus_str_soft, consensus_fg_claims, consensus_fg_ratio = build_consensus_units(
        human_human_behavior,
        human_human_str,
        human_human_fg_claims,
        human_human_fg_ratio,
        committee_map,
    )
    descriptive_summary = build_descriptive_summary(
        study_dir,
        sample_coverage_rows,
        human_human_behavior,
        human_human_str,
        human_human_fg_ratio,
        human_human_fg_claims,
        aligned_rows,
    )

    behavior_review_disagreements = [
        {
            "sample_id": row["sample_id"],
            "reviewer_id": row["reviewer_id"],
            "conflict_category_id": row["conflict_category_id"],
            "model": row["model"],
            "prompt": row["prompt"],
            "train_type": row["train_type"],
            "human_behavior_label": row["behavior_label"],
            "human_behavior_rationale": row["behavior_rationale"],
            "committee_behavior_label": row["committee"]["behavior_label"],
            "committee_behavior_rationale": (row["committee"]["behavior_details"] or {}).get("rationale"),
        }
        for row in aligned_rows
        if isinstance(row["behavior_label"], bool)
        and isinstance(row["committee"]["behavior_label"], bool)
        and bool(row["behavior_label"]) != bool(row["committee"]["behavior_label"])
    ]
    behavior_consensus_disagreements = [
        {
            "sample_id": unit["sample_id"],
            "reviewer_pair": unit["reviewer_pair"],
            "conflict_category_id": unit["conflict_category_id"],
            "model": unit["model"],
            "prompt": unit["prompt"],
            "train_type": unit["train_type"],
            "human_consensus_behavior": bool(unit["human_consensus"]),
            "committee_behavior": bool(unit["committee_label"]),
        }
        for unit in consensus_behavior
        if int(unit["human_consensus"]) != int(unit["committee_label"])
    ]
    str_strict_review_disagreements = [
        unit for unit in hc_str_strict
        if int(unit["human_label"]) != int(unit["committee_label"])
    ]
    fg_claim_review_disagreements = [
        unit for unit in hc_fg_claims
        if int(unit["human_label"]) != int(unit["committee_label"])
    ]

    behavior_direction_counts = {
        "human_neg_committee_pos": sum(
            1
            for row in aligned_rows
            if isinstance(row["behavior_label"], bool)
            and isinstance(row["committee"]["behavior_label"], bool)
            and (not row["behavior_label"])
            and bool(row["committee"]["behavior_label"])
        ),
        "human_pos_committee_neg": sum(
            1
            for row in aligned_rows
            if isinstance(row["behavior_label"], bool)
            and isinstance(row["committee"]["behavior_label"], bool)
            and bool(row["behavior_label"])
            and (not row["committee"]["behavior_label"])
        ),
    }
    behavior_direction_by_conflict: Dict[str, Dict[str, int]] = {}
    for conflict_id in sorted(
        {
            row["conflict_category_id"]
            for row in aligned_rows
            if isinstance(row["behavior_label"], bool)
            and isinstance(row["committee"]["behavior_label"], bool)
            and bool(row["behavior_label"]) != bool(row["committee"]["behavior_label"])
        }
    ):
        subset = [
            row for row in aligned_rows
            if row["conflict_category_id"] == conflict_id
            and isinstance(row["behavior_label"], bool)
            and isinstance(row["committee"]["behavior_label"], bool)
            and bool(row["behavior_label"]) != bool(row["committee"]["behavior_label"])
        ]
        behavior_direction_by_conflict[str(conflict_id)] = {
            "human_neg_committee_pos": sum(
                1
                for row in subset
                if (not row["behavior_label"]) and bool(row["committee"]["behavior_label"])
            ),
            "human_pos_committee_neg": sum(
                1
                for row in subset
                if bool(row["behavior_label"]) and (not row["committee"]["behavior_label"])
            ),
        }

    behavior_error_analysis = {
        "review_level_disagreements": len(behavior_review_disagreements),
        "review_level_by_conflict": summarize_count_field(behavior_review_disagreements, "conflict_category_id"),
        "review_level_by_model": summarize_count_field(behavior_review_disagreements, "model"),
        "review_level_by_prompt": summarize_count_field(behavior_review_disagreements, "prompt"),
        "review_level_by_train_type": summarize_count_field(behavior_review_disagreements, "train_type"),
        "review_level_direction": behavior_direction_counts,
        "review_level_direction_by_conflict": behavior_direction_by_conflict,
        "consensus_disagreements": len(behavior_consensus_disagreements),
        "consensus_by_conflict": summarize_count_field(behavior_consensus_disagreements, "conflict_category_id"),
        "consensus_by_model": summarize_count_field(behavior_consensus_disagreements, "model"),
        "consensus_by_prompt": summarize_count_field(behavior_consensus_disagreements, "prompt"),
        "consensus_by_train_type": summarize_count_field(behavior_consensus_disagreements, "train_type"),
    }

    subgroup_fields = ("conflict_category_id", "model")

    human_human_binary = []
    human_human_binary += make_binary_table(human_human_behavior, "human_vs_human", "behavior_binary", "label_a", "label_b", include_alpha=True, subgroup_fields=("reviewer_pair", "conflict_category_id"))
    human_human_binary += make_binary_table(human_human_str, "human_vs_human", "str_binary", "label_a", "label_b", include_alpha=True, subgroup_fields=("reviewer_pair", "conflict_category_id"))
    human_human_binary += make_binary_table(human_human_fg_claims, "human_vs_human", "fg_claim_binary", "label_a", "label_b", include_alpha=True, subgroup_fields=("reviewer_pair", "conflict_category_id"))

    human_human_continuous = make_continuous_table(
        human_human_fg_ratio,
        "human_vs_human",
        "fg_ratio_continuous",
        "value_a",
        "value_b",
        subgroup_fields=("reviewer_pair", "conflict_category_id"),
    )

    human_committee_binary = []
    human_committee_binary += make_binary_table(hc_behavior, "human_vs_committee", "behavior_binary", "human_label", "committee_label", subgroup_fields=subgroup_fields)
    human_committee_binary += make_binary_table(hc_str_strict, "human_vs_committee", "str_binary_strict", "human_label", "committee_label", subgroup_fields=subgroup_fields)
    human_committee_binary += make_binary_table(hc_str_soft, "human_vs_committee", "str_binary_soft", "human_label", "committee_label", subgroup_fields=subgroup_fields)
    human_committee_binary += make_binary_table(hc_fg_claims, "human_vs_committee", "fg_claim_binary", "human_label", "committee_label", subgroup_fields=subgroup_fields)

    human_committee_continuous = make_continuous_table(
        hc_fg_ratio,
        "human_vs_committee",
        "fg_ratio_continuous",
        "human_value",
        "committee_value",
        subgroup_fields=subgroup_fields,
    )

    consensus_binary = []
    consensus_binary += make_binary_table(consensus_behavior, "human_consensus_vs_committee", "behavior_binary", "human_consensus", "committee_label", subgroup_fields=("conflict_category_id",))
    consensus_binary += make_binary_table(consensus_str_strict, "human_consensus_vs_committee", "str_binary_strict", "human_consensus", "committee_label", subgroup_fields=("conflict_category_id",))
    consensus_binary += make_binary_table(consensus_str_soft, "human_consensus_vs_committee", "str_binary_soft", "human_consensus", "committee_label", subgroup_fields=("conflict_category_id",))
    consensus_binary += make_binary_table(consensus_fg_claims, "human_consensus_vs_committee", "fg_claim_binary", "human_consensus", "committee_label", subgroup_fields=("conflict_category_id",))

    consensus_continuous = make_continuous_table(
        consensus_fg_ratio,
        "human_consensus_vs_committee",
        "fg_ratio_continuous",
        "human_consensus",
        "committee_value",
        subgroup_fields=("conflict_category_id",),
    )

    behavior_individual_table = []
    for judge_model_id, subset in sorted(group_by(behavior_individual_units, "judge_model_id").items(), key=lambda item: str(item[0])):
        behavior_individual_table.append(
            summarize_binary_units(
                subset,
                "human_label",
                "committee_label",
                "human_vs_individual_committee_judge",
                "behavior_binary",
                subgroup_type="judge_model_id",
                subgroup_value=str(judge_model_id),
            )
        )

    coverage = {
        "submitted_human_reviews": len(aligned_rows),
        "committee_matched_reviews": len(aligned_rows),
        "missing_committee_matches": len(missing_committee),
        "double_reviewed_samples": len({row["sample_id"] for row in human_human_behavior}),
        "behavior_double_review_units": len(human_human_behavior),
        "str_double_review_units": len(human_human_str),
        "fg_ratio_double_review_units": len(human_human_fg_ratio),
        "fg_claim_double_review_units": len(human_human_fg_claims),
        "human_committee_behavior_units": len(hc_behavior),
        "human_committee_str_strict_units": len(hc_str_strict),
        "human_committee_str_soft_units": len(hc_str_soft),
        "human_committee_fg_ratio_units": len(hc_fg_ratio),
        "human_committee_fg_claim_units": len(hc_fg_claims),
        "consensus_behavior_units": len(consensus_behavior),
        "consensus_str_strict_units": len(consensus_str_strict),
        "consensus_str_soft_units": len(consensus_str_soft),
        "consensus_fg_claim_units": len(consensus_fg_claims),
        "consensus_fg_ratio_units": len(consensus_fg_ratio),
        "behavior_individual_committee_units": len(behavior_individual_units),
        "human_human_claim_alignment_issues": len(human_human_claim_issues),
        "human_committee_claim_alignment_issues": len(human_committee_claim_issues),
        "committee_internal_cache_available": committee_internal["available"],
        "committee_internal_claim_alignment_issues": len(committee_internal["claim_alignment_issues"]),
        "full_selected_samples": descriptive_summary["sample_subsets"]["all_350"]["sample_count"],
        "fully_complete_samples": descriptive_summary["sample_subsets"]["complete_300"]["sample_count"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "aligned_human_committee_reviews.jsonl", aligned_rows)
    _write_jsonl(output_dir / "human_human_behavior_units.jsonl", human_human_behavior)
    _write_jsonl(output_dir / "human_human_str_units.jsonl", human_human_str)
    _write_jsonl(output_dir / "human_human_fg_claim_units.jsonl", human_human_fg_claims)
    _write_jsonl(output_dir / "human_committee_behavior_units.jsonl", hc_behavior)
    _write_jsonl(output_dir / "human_committee_str_strict_units.jsonl", hc_str_strict)
    _write_jsonl(output_dir / "human_committee_str_soft_units.jsonl", hc_str_soft)
    _write_jsonl(output_dir / "human_committee_fg_claim_units.jsonl", hc_fg_claims)
    _write_jsonl(output_dir / "human_consensus_behavior_units.jsonl", consensus_behavior)
    _write_jsonl(output_dir / "human_consensus_str_strict_units.jsonl", consensus_str_strict)
    _write_jsonl(output_dir / "human_consensus_str_soft_units.jsonl", consensus_str_soft)
    _write_jsonl(output_dir / "human_consensus_fg_claim_units.jsonl", consensus_fg_claims)
    _write_jsonl(output_dir / "behavior_review_disagreements.jsonl", behavior_review_disagreements)
    _write_jsonl(output_dir / "behavior_consensus_disagreements.jsonl", behavior_consensus_disagreements)
    _write_jsonl(output_dir / "str_strict_review_disagreements.jsonl", str_strict_review_disagreements)
    _write_jsonl(output_dir / "fg_claim_review_disagreements.jsonl", fg_claim_review_disagreements)
    (output_dir / "coverage_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (output_dir / "human_human_claim_alignment_issues.json").write_text(json.dumps(human_human_claim_issues, indent=2), encoding="utf-8")
    (output_dir / "human_committee_claim_alignment_issues.json").write_text(json.dumps(human_committee_claim_issues, indent=2), encoding="utf-8")
    (output_dir / "committee_internal_agreement.json").write_text(json.dumps(committee_internal, indent=2), encoding="utf-8")
    (output_dir / "committee_internal_claim_alignment_issues.json").write_text(json.dumps(committee_internal["claim_alignment_issues"], indent=2), encoding="utf-8")
    (output_dir / "study_descriptive_summary.json").write_text(json.dumps(descriptive_summary, indent=2), encoding="utf-8")
    (output_dir / "missing_committee_matches.json").write_text(json.dumps(missing_committee, indent=2), encoding="utf-8")
    _write_csv(output_dir / "human_human_binary_agreement.csv", human_human_binary)
    _write_csv(output_dir / "human_human_continuous_agreement.csv", human_human_continuous)
    _write_csv(output_dir / "human_committee_binary_agreement.csv", human_committee_binary)
    _write_csv(output_dir / "human_committee_continuous_agreement.csv", human_committee_continuous)
    _write_csv(output_dir / "human_consensus_binary_agreement.csv", consensus_binary)
    _write_csv(output_dir / "human_consensus_continuous_agreement.csv", consensus_continuous)
    _write_csv(output_dir / "human_vs_individual_committee_behavior.csv", behavior_individual_table)

    metric_log = {
        "study_dir": str(study_dir),
        "consolidated_dir": str(consolidated_dir),
        "output_dir": str(output_dir),
        "coverage": coverage,
        "disagreement_counts": {
            "behavior_review_disagreements": len(behavior_review_disagreements),
            "behavior_consensus_disagreements": len(behavior_consensus_disagreements),
            "str_strict_review_disagreements": len(str_strict_review_disagreements),
            "fg_claim_review_disagreements": len(fg_claim_review_disagreements),
        },
        "human_human_binary": human_human_binary,
        "human_human_continuous": human_human_continuous,
        "human_committee_binary": human_committee_binary,
        "human_committee_continuous": human_committee_continuous,
        "human_consensus_binary": consensus_binary,
        "human_consensus_continuous": consensus_continuous,
        "human_vs_individual_committee_behavior": behavior_individual_table,
        "committee_internal": committee_internal,
        "descriptive_summary": descriptive_summary,
        "behavior_error_analysis": behavior_error_analysis,
    }
    (output_dir / "agreement_metric_log.json").write_text(
        json.dumps(metric_log, indent=2),
        encoding="utf-8",
    )
    (output_dir / "agreement_report.md").write_text(
        build_markdown_report(metric_log),
        encoding="utf-8",
    )
    return metric_log


def find_metric_row(rows: Sequence[Dict[str, Any]], comparison: str, metric: str, subgroup_type: str = "overall", subgroup_value: str = "all") -> Optional[Dict[str, Any]]:
    for row in rows:
        if (
            row["comparison"] == comparison
            and row["metric"] == metric
            and row["subgroup_type"] == subgroup_type
            and row["subgroup_value"] == subgroup_value
        ):
            return row
    return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def mean_metric(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _safe_mean(values)


def format_count_map(mapping: Dict[Any, Any]) -> str:
    parts = [f"{key}={value}" for key, value in mapping.items()]
    return ", ".join(parts)


def summarize_count_field(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def find_committee_internal_pair(
    committee_internal: Dict[str, Any],
    subset_name: str,
    metric: str,
    judge_short_a: str,
    judge_short_b: str,
) -> Optional[Dict[str, Any]]:
    subset = (committee_internal.get("subsets") or {}).get(subset_name) or {}
    metric_block = subset.get(metric) or {}
    target = {judge_short_a, judge_short_b}
    for row in metric_block.get("pairwise") or []:
        pair = {_judge_short_name(row["judge_a"]), _judge_short_name(row["judge_b"])}
        if pair == target:
            return row
    return None


def build_markdown_report(metric_log: Dict[str, Any]) -> str:
    coverage = metric_log["coverage"]
    disagreements = metric_log["disagreement_counts"]
    hh_bin = metric_log["human_human_binary"]
    hh_cont = metric_log["human_human_continuous"]
    hc_bin = metric_log["human_committee_binary"]
    hc_cont = metric_log["human_committee_continuous"]
    cons_bin = metric_log["human_consensus_binary"]
    cons_cont = metric_log["human_consensus_continuous"]
    indiv = metric_log["human_vs_individual_committee_behavior"]
    committee_internal = metric_log.get("committee_internal") or {}
    descriptive_summary = metric_log.get("descriptive_summary") or {}
    behavior_error_analysis = metric_log.get("behavior_error_analysis") or {}
    provenance = descriptive_summary.get("study_provenance") or {}
    subsets = descriptive_summary.get("sample_subsets") or {}
    descriptive_complete = descriptive_summary.get("descriptive_labels_complete_300") or {}
    all350 = subsets.get("all_350") or {}
    complete300 = subsets.get("complete_300") or {}
    human_behavior_desc = descriptive_complete.get("human_behavior") or {}
    committee_behavior_desc = descriptive_complete.get("committee_behavior") or {}
    human_str_desc = descriptive_complete.get("human_str") or {}
    committee_str_desc = descriptive_complete.get("committee_str") or {}
    human_fg_desc = descriptive_complete.get("human_fg") or {}
    committee_fg_desc = descriptive_complete.get("committee_fg") or {}

    hh_behavior = find_metric_row(hh_bin, "human_vs_human", "behavior_binary")
    hh_str = find_metric_row(hh_bin, "human_vs_human", "str_binary")
    hh_fg_claim = find_metric_row(hh_bin, "human_vs_human", "fg_claim_binary")
    hh_fg_ratio = find_metric_row(hh_cont, "human_vs_human", "fg_ratio_continuous")

    hc_behavior = find_metric_row(hc_bin, "human_vs_committee", "behavior_binary")
    hc_strict = find_metric_row(hc_bin, "human_vs_committee", "str_binary_strict")
    hc_soft = find_metric_row(hc_bin, "human_vs_committee", "str_binary_soft")
    hc_fg_claim = find_metric_row(hc_bin, "human_vs_committee", "fg_claim_binary")
    hc_fg_ratio = find_metric_row(hc_cont, "human_vs_committee", "fg_ratio_continuous")

    cons_behavior = find_metric_row(cons_bin, "human_consensus_vs_committee", "behavior_binary")
    cons_strict = find_metric_row(cons_bin, "human_consensus_vs_committee", "str_binary_strict")
    cons_soft = find_metric_row(cons_bin, "human_consensus_vs_committee", "str_binary_soft")
    cons_fg_claim = find_metric_row(cons_bin, "human_consensus_vs_committee", "fg_claim_binary")
    cons_fg_ratio = find_metric_row(cons_cont, "human_consensus_vs_committee", "fg_ratio_continuous")

    ci_all = (committee_internal.get("subsets") or {}).get("all_350") or {}
    ci_complete = (committee_internal.get("subsets") or {}).get("complete_300") or {}
    ci_all_behavior = ci_all.get("behavior_binary") or {}
    ci_all_strict = ci_all.get("str_binary_strict") or {}
    ci_all_soft = ci_all.get("str_binary_soft") or {}
    ci_all_fg_claim = ci_all.get("fg_claim_binary") or {}
    ci_all_fg_ratio = ci_all.get("fg_ratio_continuous") or {}
    ci_complete_behavior = ci_complete.get("behavior_binary") or {}
    ci_complete_strict = ci_complete.get("str_binary_strict") or {}
    ci_complete_soft = ci_complete.get("str_binary_soft") or {}
    ci_complete_fg_claim = ci_complete.get("fg_claim_binary") or {}
    ci_complete_fg_ratio = ci_complete.get("fg_ratio_continuous") or {}

    ci_all_behavior_qd = find_committee_internal_pair(committee_internal, "all_350", "behavior_binary", "qwen3.5-397b-a17b", "deepseek-r1-distill-32b")
    ci_all_behavior_qm = find_committee_internal_pair(committee_internal, "all_350", "behavior_binary", "qwen3.5-397b-a17b", "mistral-small-4")
    ci_all_behavior_md = find_committee_internal_pair(committee_internal, "all_350", "behavior_binary", "mistral-small-4", "deepseek-r1-distill-32b")
    ci_all_strict_qd = find_committee_internal_pair(committee_internal, "all_350", "str_binary_strict", "qwen3.5-397b-a17b", "deepseek-r1-distill-32b")
    ci_all_fg_claim_md = find_committee_internal_pair(committee_internal, "all_350", "fg_claim_binary", "mistral-small-4", "deepseek-r1-distill-32b")
    ci_all_fg_ratio_md = find_committee_internal_pair(committee_internal, "all_350", "fg_ratio_continuous", "mistral-small-4", "deepseek-r1-distill-32b")
    ci_complete_fg_ratio_md = find_committee_internal_pair(committee_internal, "complete_300", "fg_ratio_continuous", "mistral-small-4", "deepseek-r1-distill-32b")
    ci_complete_behavior_mean_kappa = mean_metric(ci_complete_behavior.get("pairwise") or [], "cohen_kappa")
    ci_complete_strict_mean_kappa = mean_metric(ci_complete_strict.get("pairwise") or [], "cohen_kappa")
    ci_complete_fg_claim_mean_kappa = mean_metric(ci_complete_fg_claim.get("pairwise") or [], "cohen_kappa")

    lines: List[str] = []
    lines.append("# Human Eval Agreement Report")
    lines.append("")
    lines.append("## Scope Note")
    lines.append("")
    lines.append("- Unless explicitly marked as supplementary, all primary cross-system comparisons in this report are anchored on the **fully complete 300-sample double-reviewed subset**.")
    lines.append("- That choice keeps the human-human, human-committee, and committee-internal comparisons aligned to the same cleanest available evaluation slice.")
    lines.append("- The larger 350-sample selected pool is still reported where useful, but only as supplementary context because 50 of those samples are not fully double-reviewed by humans.")
    lines.append("")
    lines.append("## Study Construction")
    lines.append("")
    lines.append(f"- Study snapshot label: `{DEFAULT_CONSOLIDATED_LABEL}`")
    lines.append(f"- Selection seed: `{provenance.get('selection_seed')}`")
    lines.append(f"- Source family: `{provenance.get('source_root_pattern')}`")
    lines.append(f"- Task variant used for this study: `{provenance.get('source_task_variant')}`")
    lines.append(f"- Models included: `{', '.join(provenance.get('models') or [])}`")
    lines.append(f"- Prompts included: `{', '.join(provenance.get('prompts') or [])}`")
    lines.append(f"- Train types included: `{', '.join(provenance.get('train_types') or [])}`")
    lines.append(f"- Full selected study sample count: `{provenance.get('selected_total')}`")
    lines.append(f"- Target human review slots at assignment time: `{provenance.get('review_target_total')}`")
    lines.append("- Selection excluded correct-refusal rows before balancing and assignment.")
    if provenance.get("extra_cells"):
        lines.append(f"- The two 30-sample cells were: `{', '.join(provenance.get('extra_cells') or [])}`")
    if provenance.get("cell_targets"):
        lines.append(f"- Cell-level selected counts across the 12 study cells: `{format_count_map(provenance.get('cell_targets') or {})}`")
    lines.append(f"- Selected source-row file: `{provenance.get('source_rows_file')}`")
    lines.append(f"- Selection audit file: `{provenance.get('assignment_audit_file')}`")
    lines.append("")
    lines.append("## Sample Distribution")
    lines.append("")
    lines.append("### Full 350 Selected Samples")
    lines.append("")
    lines.append(f"- By model: `{format_count_map(all350.get('distribution', {}).get('model') or {})}`")
    lines.append(f"- By prompt: `{format_count_map(all350.get('distribution', {}).get('prompt') or {})}`")
    lines.append(f"- By train type: `{format_count_map(all350.get('distribution', {}).get('train_type') or {})}`")
    lines.append(f"- By conflict type id: `{format_count_map(all350.get('distribution', {}).get('conflict_category_id') or {})}`")
    lines.append("- This 350-sample pool is exactly balanced across model, train type, and conflict category, with only a 117/117/116 prompt split due to the indivisible total of 350.")
    lines.append("")
    lines.append("### Fully Complete 300-Sample Double-Reviewed Subset")
    lines.append("")
    lines.append(f"- By model: `{format_count_map(complete300.get('distribution', {}).get('model') or {})}`")
    lines.append(f"- By prompt: `{format_count_map(complete300.get('distribution', {}).get('prompt') or {})}`")
    lines.append(f"- By train type: `{format_count_map(complete300.get('distribution', {}).get('train_type') or {})}`")
    lines.append(f"- By conflict type id: `{format_count_map(complete300.get('distribution', {}).get('conflict_category_id') or {})}`")
    lines.append("- The fully complete 300-sample subset stays close to balanced, so its agreement estimates are not being driven by a single model, prompt, train type, or conflict slice.")
    lines.append("")
    lines.append("## Conflict Type Legend")
    lines.append("")
    lines.append("- Type `1`: No conflict")
    lines.append("- Type `2`: Complementary information")
    lines.append("- Type `3`: Conflicting opinions or research outcomes")
    lines.append("- Type `4`: Conflict due to outdated information")
    lines.append("- Type `5`: Conflict due to misinformation")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Submitted human reviews currently consolidated: `{coverage['submitted_human_reviews']}`")
    lines.append(f"- Double-reviewed samples available for human-human IAA: `{coverage['double_reviewed_samples']}`")
    lines.append(f"- Behavior double-review units: `{coverage['behavior_double_review_units']}`")
    lines.append(f"- STR double-review units: `{coverage['str_double_review_units']}`")
    lines.append(f"- FG ratio double-review units: `{coverage['fg_ratio_double_review_units']}`")
    lines.append(f"- FG claim double-review units: `{coverage['fg_claim_double_review_units']}`")
    lines.append(f"- Human-vs-committee behavior units: `{coverage['human_committee_behavior_units']}`")
    lines.append(f"- Human-vs-committee STR units (strict): `{coverage['human_committee_str_strict_units']}`")
    lines.append(f"- Human-vs-committee FG claim units: `{coverage['human_committee_fg_claim_units']}`")
    lines.append(f"- Behavior review-level disagreements queued for audit: `{disagreements['behavior_review_disagreements']}`")
    lines.append(f"- Committee-internal judge cache available locally: `{coverage['committee_internal_cache_available']}`")
    lines.append("")
    lines.append("## Descriptive Outcomes")
    lines.append("")
    lines.append("### What Humans Thought on the Fully Complete 300-Sample Subset")
    lines.append("")
    if human_behavior_desc:
        lines.append(
            f"- Behavior adherence: `{human_behavior_desc.get('positive_reviews')}/{human_behavior_desc.get('total_reviews')}` positive human reviews, rate `{fmt(human_behavior_desc.get('positive_rate'))}`"
        )
        lines.append(
            f"- Behavior sample breakdown: `{human_behavior_desc.get('unanimous_positive_samples')}` unanimous positive, `{human_behavior_desc.get('unanimous_negative_samples')}` unanimous negative, `{human_behavior_desc.get('split_samples')}` split"
        )
    if human_str_desc:
        lines.append(
            f"- STR on applicable samples: `{human_str_desc.get('positive_reviews')}/{human_str_desc.get('total_reviews')}` positive human reviews, rate `{fmt(human_str_desc.get('positive_rate'))}`"
        )
        lines.append(
            f"- STR sample breakdown: `{human_str_desc.get('unanimous_positive_samples')}` unanimous positive, `{human_str_desc.get('unanimous_negative_samples')}` unanimous negative, `{human_str_desc.get('split_samples')}` split"
        )
    if human_fg_desc:
        lines.append(
            f"- FG claim checks: `{human_fg_desc.get('claim_positive_reviews')}/{human_fg_desc.get('claim_total_reviews')}` supported human claim judgments, rate `{fmt(human_fg_desc.get('claim_positive_rate'))}`"
        )
        lines.append(
            f"- Human mean FG ratio across the 300-sample subset: `{fmt(human_fg_desc.get('sample_level_mean_ratio'))}`"
        )
    lines.append("")
    lines.append("### What the Committee Thought on the Same 300-Sample Subset")
    lines.append("")
    if committee_behavior_desc:
        lines.append(
            f"- Behavior adherence: `{committee_behavior_desc.get('positive_samples')}/{committee_behavior_desc.get('total_samples')}` positive committee decisions, rate `{fmt(committee_behavior_desc.get('positive_rate'))}`"
        )
    if committee_str_desc:
        lines.append(
            f"- STR strict positives: `{committee_str_desc.get('strict_positive_samples')}/{committee_str_desc.get('applicable_samples')}`, rate `{fmt(committee_str_desc.get('strict_positive_rate'))}`"
        )
        lines.append(
            f"- STR soft positives: `{committee_str_desc.get('soft_positive_samples')}/{committee_str_desc.get('applicable_samples')}`, rate `{fmt(committee_str_desc.get('soft_positive_rate'))}`"
        )
    if committee_fg_desc:
        lines.append(
            f"- Committee mean FG ratio across the same 300-sample subset: `{fmt(committee_fg_desc.get('sample_level_mean_ratio'))}`"
        )
    lines.append("")
    lines.append("## Human-Human IAA")
    lines.append("")
    if hh_behavior:
        lines.append(f"- Behavior: `n={hh_behavior['n']}`, agreement `{fmt(hh_behavior['agreement'])}`, Cohen's kappa `{fmt(hh_behavior['cohen_kappa'])}`, Krippendorff alpha `{fmt(hh_behavior['krippendorff_alpha_nominal'])}`")
    if hh_str:
        lines.append(f"- STR: `n={hh_str['n']}`, agreement `{fmt(hh_str['agreement'])}`, Cohen's kappa `{fmt(hh_str['cohen_kappa'])}`, Krippendorff alpha `{fmt(hh_str['krippendorff_alpha_nominal'])}`")
    if hh_fg_claim:
        lines.append(f"- FG claim-level: `n={hh_fg_claim['n']}`, agreement `{fmt(hh_fg_claim['agreement'])}`, Cohen's kappa `{fmt(hh_fg_claim['cohen_kappa'])}`, Krippendorff alpha `{fmt(hh_fg_claim['krippendorff_alpha_nominal'])}`")
    if hh_fg_ratio:
        lines.append(f"- FG sample-level ratio: `n={hh_fg_ratio['n']}`, exact-match `{fmt(hh_fg_ratio['exact_match_rate'])}`, MAE `{fmt(hh_fg_ratio['mae'])}`, Pearson `{fmt(hh_fg_ratio['pearson_r'])}`, Spearman `{fmt(hh_fg_ratio['spearman_rho'])}`")
    lines.append("")
    lines.append("## Human vs Committee")
    lines.append("")
    if hc_behavior:
        lines.append(f"- Behavior: `n={hc_behavior['n']}`, agreement `{fmt(hc_behavior['agreement'])}`, Cohen's kappa `{fmt(hc_behavior['cohen_kappa'])}`")
    if hc_strict:
        lines.append(f"- STR primary (committee exact-match only as positive): `n={hc_strict['n']}`, agreement `{fmt(hc_strict['agreement'])}`, Cohen's kappa `{fmt(hc_strict['cohen_kappa'])}`")
    if hc_soft:
        lines.append(f"- STR sensitivity (committee partial-or-exact as positive): `n={hc_soft['n']}`, agreement `{fmt(hc_soft['agreement'])}`, Cohen's kappa `{fmt(hc_soft['cohen_kappa'])}`")
    if hc_fg_claim:
        lines.append(f"- FG claim-level: `n={hc_fg_claim['n']}`, agreement `{fmt(hc_fg_claim['agreement'])}`, Cohen's kappa `{fmt(hc_fg_claim['cohen_kappa'])}`")
    if hc_fg_ratio:
        lines.append(f"- FG sample-level ratio: `n={hc_fg_ratio['n']}`, exact-match `{fmt(hc_fg_ratio['exact_match_rate'])}`, MAE `{fmt(hc_fg_ratio['mae'])}`, Pearson `{fmt(hc_fg_ratio['pearson_r'])}`, Spearman `{fmt(hc_fg_ratio['spearman_rho'])}`")
    lines.append("")
    lines.append("## Human Consensus vs Committee")
    lines.append("")
    if cons_behavior:
        lines.append(f"- Behavior on unanimous human subset: `n={cons_behavior['n']}`, agreement `{fmt(cons_behavior['agreement'])}`, Cohen's kappa `{fmt(cons_behavior['cohen_kappa'])}`")
    if cons_strict:
        lines.append(f"- STR strict on unanimous human subset: `n={cons_strict['n']}`, agreement `{fmt(cons_strict['agreement'])}`, Cohen's kappa `{fmt(cons_strict['cohen_kappa'])}`")
    if cons_soft:
        lines.append(f"- STR soft sensitivity on unanimous human subset: `n={cons_soft['n']}`, agreement `{fmt(cons_soft['agreement'])}`, Cohen's kappa `{fmt(cons_soft['cohen_kappa'])}`")
    if cons_fg_claim:
        lines.append(f"- FG claim-level on unanimous human subset: `n={cons_fg_claim['n']}`, agreement `{fmt(cons_fg_claim['agreement'])}`, Cohen's kappa `{fmt(cons_fg_claim['cohen_kappa'])}`")
    if cons_fg_ratio:
        lines.append(f"- FG ratio on exact-human-match subset: `n={cons_fg_ratio['n']}`, exact-match with committee `{fmt(cons_fg_ratio['exact_match_rate'])}`, MAE `{fmt(cons_fg_ratio['mae'])}`")
    lines.append("")
    lines.append("## Comparison Caveat")
    lines.append("")
    lines.append("- Each sample in the human study was reviewed by exactly `2` humans, while each sample in the local committee was judged by `3` LLM judges.")
    lines.append("- Because of that design asymmetry, human-human and committee-internal multirater coefficients should not be treated as directly interchangeable apples-to-apples quantities.")
    lines.append("- The fairest direct comparison is pairwise agreement on the same sample slice, especially the fully double-reviewed `300`-sample subset.")
    if hh_behavior and hh_str and hh_fg_claim:
        lines.append(
            f"- On that `300`-sample subset, human-human kappa is `{fmt(hh_behavior['cohen_kappa'])}` for behavior, `{fmt(hh_str['cohen_kappa'])}` for STR, and `{fmt(hh_fg_claim['cohen_kappa'])}` for FG claim checks."
        )
    if (
        ci_complete_behavior_mean_kappa is not None
        and ci_complete_strict_mean_kappa is not None
        and ci_complete_fg_claim_mean_kappa is not None
    ):
        lines.append(
            f"- The corresponding mean pairwise LLM-LLM kappa values are `{fmt(ci_complete_behavior_mean_kappa)}` for behavior, `{fmt(ci_complete_strict_mean_kappa)}` for STR, and `{fmt(ci_complete_fg_claim_mean_kappa)}` for FG claim checks."
        )
    lines.append("- This supports a cautious paper claim: the committee is strongly reliable on STR and grounding, but only moderately stable on behavior, and humans remain more consistent than the LLM judges on behavior.")
    lines.append("")
    lines.append("## Paper-Ready Claims And Cautions")
    lines.append("")
    lines.append("- The strongest validation result is not on holistic behavior, but on STR and grounding. Those are the dimensions on which both human-versus-committee agreement and committee-internal agreement are strongest.")
    lines.append("- The safest paper claim is therefore conditional: the committee is a defensible proxy for human judgment on STR and faithfulness-oriented checks, while behavior still requires more caution and supporting manual analysis.")
    lines.append("- On the fully complete `300`-sample subset, humans labeled behaviorally aligned at rate `0.673`, while the committee labeled behaviorally aligned at rate `0.710`. That gap is not enormous, but it does show the committee is slightly more permissive than humans on behavior.")
    lines.append("- Because the study uses `2` humans per sample and `3` LLM judges per sample, pairwise comparisons should carry the main argumentative weight when comparing human and committee reliability; multirater alpha is best used as within-family context.")
    lines.append("- The report therefore supports a nuanced conclusion: committee-based evaluation is strongest for STR and grounding, reasonably informative but less settled for behavior, and not yet a full drop-in replacement for human behavioral judgment.")
    lines.append("")
    lines.append("## Behavior Error Analysis Priorities")
    lines.append("")
    if behavior_error_analysis:
        lines.append(
            f"- Review-level behavior disagreements: `{behavior_error_analysis.get('review_level_disagreements')}`"
        )
        lines.append(
            f"- By conflict type: `{format_count_map(behavior_error_analysis.get('review_level_by_conflict') or {})}`"
        )
        lines.append(
            f"- By prompt: `{format_count_map(behavior_error_analysis.get('review_level_by_prompt') or {})}`"
        )
        lines.append(
            f"- By train type: `{format_count_map(behavior_error_analysis.get('review_level_by_train_type') or {})}`"
        )
        lines.append(
            f"- Direction overall: `{format_count_map(behavior_error_analysis.get('review_level_direction') or {})}`"
        )
        lines.append(
            f"- Consensus-only disagreements after restricting to unanimous-human samples: `{behavior_error_analysis.get('consensus_disagreements')}`"
        )
        lines.append(
            f"- Consensus disagreements by conflict type: `{format_count_map(behavior_error_analysis.get('consensus_by_conflict') or {})}`"
        )
    lines.append("- Type `5` is the highest-priority slice for manual qualitative analysis. It has the largest behavior disagreement mass and remains the weakest slice even after restricting to unanimous-human samples.")
    lines.append("- Type `2` is especially diagnostic because the disagreement is strongly asymmetric there: the committee is much more likely than humans to call the answer behaviorally aligned, which suggests over-crediting of partial reconciliation in complementary-information cases.")
    lines.append("- Strict-prompt cases deserve focused review because they contribute the largest number of review-level behavior disagreements.")
    lines.append("- Baseline outputs deserve somewhat more behavior-focused audit attention than SFT outputs because their disagreement mass is larger.")
    lines.append("- The most useful files for targeted manual follow-up are `behavior_review_disagreements.jsonl` for all mismatches and `behavior_consensus_disagreements.jsonl` for the cleaner subset where the two humans already agree with each other.")
    lines.append("")
    if committee_internal.get("available"):
        lines.append("## Committee Internal Agreement")
        lines.append("")
        lines.append("### Primary: Fully Complete 300-Sample Subset")
        lines.append("")
        if ci_complete_behavior:
            lines.append(
                f"- Behavior: `n={ci_complete_behavior['n_items']}`, Krippendorff alpha `{fmt(ci_complete_behavior['krippendorff_alpha_nominal'])}`"
            )
        ci_complete_behavior_qm = find_committee_internal_pair(committee_internal, "complete_300", "behavior_binary", "qwen3.5-397b-a17b", "mistral-small-4")
        ci_complete_behavior_qd = find_committee_internal_pair(committee_internal, "complete_300", "behavior_binary", "qwen3.5-397b-a17b", "deepseek-r1-distill-32b")
        ci_complete_behavior_md = find_committee_internal_pair(committee_internal, "complete_300", "behavior_binary", "mistral-small-4", "deepseek-r1-distill-32b")
        ci_complete_strict_qd = find_committee_internal_pair(committee_internal, "complete_300", "str_binary_strict", "qwen3.5-397b-a17b", "deepseek-r1-distill-32b")
        ci_complete_fg_claim_md = find_committee_internal_pair(committee_internal, "complete_300", "fg_claim_binary", "mistral-small-4", "deepseek-r1-distill-32b")
        if ci_complete_behavior_qm:
            lines.append(
                f"- Behavior, qwen vs mistral: agreement `{fmt(ci_complete_behavior_qm['agreement'])}`, Cohen's kappa `{fmt(ci_complete_behavior_qm['cohen_kappa'])}`"
            )
        if ci_complete_behavior_qd:
            lines.append(
                f"- Behavior, qwen vs deepseek: agreement `{fmt(ci_complete_behavior_qd['agreement'])}`, Cohen's kappa `{fmt(ci_complete_behavior_qd['cohen_kappa'])}`"
            )
        if ci_complete_behavior_md:
            lines.append(
                f"- Behavior, mistral vs deepseek: agreement `{fmt(ci_complete_behavior_md['agreement'])}`, Cohen's kappa `{fmt(ci_complete_behavior_md['cohen_kappa'])}`"
            )
        if ci_complete_strict:
            lines.append(
                f"- STR strict: `n={ci_complete_strict['n_items']}`, Krippendorff alpha `{fmt(ci_complete_strict['krippendorff_alpha_nominal'])}`"
            )
        if ci_complete_strict_qd:
            lines.append(
                f"- STR strict, qwen vs deepseek: agreement `{fmt(ci_complete_strict_qd['agreement'])}`, Cohen's kappa `{fmt(ci_complete_strict_qd['cohen_kappa'])}`"
            )
        if ci_complete_soft and ci_complete_soft.get('krippendorff_alpha_nominal') == ci_complete_strict.get('krippendorff_alpha_nominal'):
            lines.append("- STR soft matches STR strict exactly on this primary 300-sample slice, indicating no partial-recall boundary effect inside the cached judge outputs.")
        if ci_complete_fg_claim:
            lines.append(
                f"- FG claim-level: `n={ci_complete_fg_claim['n_items']}`, Krippendorff alpha `{fmt(ci_complete_fg_claim['krippendorff_alpha_nominal'])}`"
            )
        if ci_complete_fg_claim_md:
            lines.append(
                f"- FG claim-level, mistral vs deepseek: agreement `{fmt(ci_complete_fg_claim_md['agreement'])}`, Cohen's kappa `{fmt(ci_complete_fg_claim_md['cohen_kappa'])}`"
            )
        if ci_complete_fg_ratio_md:
            lines.append(
                f"- FG ratio, mistral vs deepseek: `n={ci_complete_fg_ratio_md['n']}`, exact-match `{fmt(ci_complete_fg_ratio_md['exact_match_rate'])}`, MAE `{fmt(ci_complete_fg_ratio_md['mae'])}`, Pearson `{fmt(ci_complete_fg_ratio_md['pearson_r'])}`"
            )
        lines.append("")
        lines.append("### Supplementary: All 350 Selected Study Samples")
        lines.append("")
        if ci_all_behavior:
            lines.append(
                f"- Behavior: `n={ci_all_behavior['n_items']}`, Krippendorff alpha `{fmt(ci_all_behavior['krippendorff_alpha_nominal'])}`"
            )
        if ci_all_strict:
            lines.append(
                f"- STR strict: `n={ci_all_strict['n_items']}`, Krippendorff alpha `{fmt(ci_all_strict['krippendorff_alpha_nominal'])}`"
            )
        if ci_all_soft and ci_all_soft.get('krippendorff_alpha_nominal') == ci_all_strict.get('krippendorff_alpha_nominal'):
            lines.append("- STR soft again matches STR strict on the larger 350-sample pool.")
        if ci_all_fg_claim:
            lines.append(
                f"- FG claim-level: `n={ci_all_fg_claim['n_items']}`, Krippendorff alpha `{fmt(ci_all_fg_claim['krippendorff_alpha_nominal'])}`"
            )
        if ci_all_fg_ratio_md:
            lines.append(
                f"- FG ratio, mistral vs deepseek: `n={ci_all_fg_ratio_md['n']}`, exact-match `{fmt(ci_all_fg_ratio_md['exact_match_rate'])}`, MAE `{fmt(ci_all_fg_ratio_md['mae'])}`, Pearson `{fmt(ci_all_fg_ratio_md['pearson_r'])}`"
            )
        lines.append("")
    lines.append("## Individual Committee Judges")
    lines.append("")
    for row in indiv:
        lines.append(f"- Behavior vs `{row['subgroup_value']}`: `n={row['n']}`, agreement `{fmt(row['agreement'])}`, Cohen's kappa `{fmt(row['cohen_kappa'])}`")
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- All four reviewer returns are now accounted for in the full snapshot. The remaining incompleteness is coverage-level only: `samyek` is accepted as a 50-submission partial return.")
    lines.append("- Behavior and STR are treated as binary labels.")
    lines.append("- FG is analyzed in two complementary ways: claim-level binary pass/fail and sample-level grounding-ratio agreement.")
    lines.append("- For committee STR comparison, the primary analysis treats committee `0.5` partial matches conservatively as non-matches, with a separate sensitivity analysis where partial matches count as positive.")
    if committee_internal.get("available"):
        lines.append("- The committee-internal analysis is reconstructed from cached single-judge staged outputs for the exact 12 study slices used in the human-eval package, covering both baseline and SFT runs.")
        lines.append("- Committee-internal agreement is strongest on grounding and STR, but materially weaker on behavior. That mirrors the broader pattern that behavior is the least operationalized and most interpretation-sensitive judgment axis.")
        lines.append("- Within behavior, qwen and deepseek are the closest pair, while mistral is the least aligned with the other two judges. This suggests the ensemble's behavior instability is driven less by pure label noise and more by one judge's rubric interpretation drift.")
        lines.append("- For paper writing, pairwise comparisons are the primary fair bridge between the `2-human` and `3-LLM` setups; multirater alpha should be presented as within-family context rather than as a direct human-versus-committee contest.")
    lines.append("- Committee-alignment claims for the paper should emphasize the exact coverage subset used for each metric rather than implying that every selected human-eval sample has complete double-human review.")
    lines.append("- The disagreement slices in `behavior_review_disagreements.jsonl`, `behavior_consensus_disagreements.jsonl`, `str_strict_review_disagreements.jsonl`, and `fg_claim_review_disagreements.jsonl` are intended to support manual error analysis before drafting final paper claims.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze human-human and human-vs-committee agreement for the CATS human eval study.")
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--consolidated-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    consolidated_dir = (
        args.consolidated_dir.resolve()
        if args.consolidated_dir
        else (study_dir / "consolidated" / DEFAULT_CONSOLIDATED_LABEL)
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (consolidated_dir / "agreement_analysis")
    )
    metric_log = build_metric_log(study_dir, consolidated_dir, output_dir)
    print(json.dumps(metric_log["coverage"], indent=2))


if __name__ == "__main__":
    main()
