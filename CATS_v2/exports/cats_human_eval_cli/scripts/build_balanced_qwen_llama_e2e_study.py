from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from cats_human_eval.storage import append_event, init_db, register_reviewers, replace_assignments
from cats_human_eval.study import create_study_bundle
from cats_human_eval.logic import answered_flags, get_model_output, gold_answerable_from_record, strip_think_trace


SEED = 20260715
REVIEWERS = ["manan", "atharv", "parth", "samyek"]
REVIEWER_CAPS = {"manan": 200, "atharv": 200, "parth": 200, "samyek": 100}
MODELS = ("qwen7b", "llama8b")
PROMPTS = ("minimal", "runtime", "strict")
TRAIN_TYPES = ("sft", "baseline")
CONFLICT_TYPES = (1, 2, 3, 4, 5)
BASE_CELL_COUNT = 29
TOTAL_SELECTED = 350
STUDY_NAME = "qwen_llama_e2e_sft_baseline_balanced_4reviewers"
STUDY_DIR = PACKAGE_ROOT / "studies" / STUDY_NAME
SOURCE_INPUT_TEMP_PATH = PACKAGE_ROOT / "studies" / f"{STUDY_NAME}__selected_source_rows.jsonl"
RAW_INPUT_COPY_PATH = STUDY_DIR / "admin" / "selected_source_rows.jsonl"
AUDIT_JSON_PATH = STUDY_DIR / "admin" / "assignment_audit.json"
AUDIT_MD_PATH = STUDY_DIR / "admin" / "assignment_audit.md"
SAMPLE_MATRIX_CSV = STUDY_DIR / "admin" / "selected_samples_with_reviewers.csv"


@dataclass(frozen=True)
class Cell:
    model: str
    prompt: str
    train_type: str
    source_path: Path

    @property
    def key(self) -> str:
        return f"{self.model}|{self.prompt}|{self.train_type}"

    @property
    def short_label(self) -> str:
        return f"{self.model}/{self.prompt}/{self.train_type}"


@dataclass
class SelectedRow:
    sample_id: str
    base_id: str
    query: str
    conflict_category_id: int
    conflict_type: str
    model: str
    prompt: str
    train_type: str
    source_path: str
    row: Dict[str, object]


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _largest_remainder(total: int, weights: Dict[str, int], tie_order: List[str]) -> Dict[str, int]:
    weight_sum = sum(weights.values())
    floors: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    allocated = 0
    for key in tie_order:
        exact = total * weights[key] / weight_sum
        floor = math.floor(exact)
        floors[key] = floor
        allocated += floor
        remainders.append((exact - floor, key))
    remaining = total - allocated
    for _, key in sorted(remainders, key=lambda item: (-item[0], tie_order.index(item[1]))):
        if remaining <= 0:
            break
        floors[key] += 1
        remaining -= 1
    return floors


def discover_cells() -> List[Cell]:
    cells: List[Cell] = []
    base = REPO_ROOT / "inputs" / "prepped_model_eval_inputs" / "benchmark_set_all_modes"
    for model in MODELS:
        for prompt in PROMPTS:
            for train_type in TRAIN_TYPES:
                source_path = base / model / "e2e" / prompt / train_type / "input.jsonl"
                if not source_path.exists():
                    raise FileNotFoundError(f"Missing eligible input file: {source_path}")
                cells.append(Cell(model=model, prompt=prompt, train_type=train_type, source_path=source_path))
    return cells


def choose_extra_cells(cells: List[Cell]) -> List[str]:
    best_pair: Tuple[float, Tuple[str, str]] | None = None
    for i, first in enumerate(cells):
        for second in cells[i + 1 :]:
            if first.model == second.model:
                continue
            if first.train_type == second.train_type:
                continue
            prompt_totals = {prompt: 4 * BASE_CELL_COUNT for prompt in PROMPTS}
            prompt_totals[first.prompt] += 1
            prompt_totals[second.prompt] += 1
            mean = TOTAL_SELECTED / len(PROMPTS)
            variance = sum((prompt_totals[prompt] - mean) ** 2 for prompt in PROMPTS)
            pair = (first.key, second.key)
            candidate = (variance, pair)
            if best_pair is None or candidate < best_pair:
                best_pair = candidate
    if best_pair is None:
        raise RuntimeError("Could not find a balanced extra-cell pair.")
    return list(best_pair[1])


def build_cell_quotas(cells: List[Cell], extra_cells: List[str]) -> Tuple[Dict[str, int], Dict[str, Dict[int, int]]]:
    cell_counts = {cell.key: BASE_CELL_COUNT for cell in cells}
    for key in extra_cells:
        cell_counts[key] += 1

    reduced_conflicts = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    smaller_cells = sorted([cell.key for cell in cells if cell.key not in extra_cells])
    if len(smaller_cells) != len(reduced_conflicts):
        raise RuntimeError("Unexpected smaller-cell count while building quotas.")

    quotas: Dict[str, Dict[int, int]] = {}
    reduced_map = dict(zip(smaller_cells, reduced_conflicts))
    for cell in cells:
        if cell.key in extra_cells:
            quotas[cell.key] = {conflict: 6 for conflict in CONFLICT_TYPES}
            continue
        reduced_conflict = reduced_map[cell.key]
        quotas[cell.key] = {conflict: (5 if conflict == reduced_conflict else 6) for conflict in CONFLICT_TYPES}
    return cell_counts, quotas


def load_pool(cells: List[Cell]) -> Dict[str, Dict[int, List[SelectedRow]]]:
    pool: Dict[str, Dict[int, List[SelectedRow]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        for row in _iter_jsonl(cell.source_path):
            gold_answerable = gold_answerable_from_record(row, accept_partial=True)
            stripped_answer = strip_think_trace(get_model_output(row))
            pred_answered = answered_flags([stripped_answer])[0]
            correct_refusal = (not gold_answerable) and (not pred_answered)
            if correct_refusal:
                continue
            base_id = str(row["id"])
            sample_id = f"{cell.model}__{cell.prompt}__{cell.train_type}__{base_id}"
            combined = dict(row)
            combined["id"] = sample_id
            combined["human_eval_source_model"] = cell.model
            combined["human_eval_source_prompt"] = cell.prompt
            combined["human_eval_source_train_type"] = cell.train_type
            combined["human_eval_source_input_jsonl"] = str(cell.source_path)
            combined["human_eval_base_sample_id"] = base_id
            selected = SelectedRow(
                sample_id=sample_id,
                base_id=base_id,
                query=str(row.get("query", "")),
                conflict_category_id=int(row["conflict_category_id"]),
                conflict_type=str(row.get("conflict_type", "")),
                model=cell.model,
                prompt=cell.prompt,
                train_type=cell.train_type,
                source_path=str(cell.source_path),
                row=combined,
            )
            pool[cell.key][selected.conflict_category_id].append(selected)
    for per_conflict in pool.values():
        for rows in per_conflict.values():
            rows.sort(key=lambda row: row.base_id)
    return pool


def select_rows(
    cells: List[Cell],
    pool: Dict[str, Dict[int, List[SelectedRow]]],
    quotas: Dict[str, Dict[int, int]],
    seed: int,
) -> List[SelectedRow]:
    rng = random.Random(seed)
    selected_rows: List[SelectedRow] = []
    selected_ids: set[str] = set()
    base_use_by_conflict: Dict[int, Counter[str]] = defaultdict(Counter)

    for conflict in CONFLICT_TYPES:
        slots: List[str] = []
        for cell in cells:
            slots.extend([cell.key] * quotas[cell.key][conflict])
        rng.shuffle(slots)
        max_occurrences = 2 if conflict == 5 else 1
        for cell_key in slots:
            candidates = [
                row
                for row in pool[cell_key][conflict]
                if row.sample_id not in selected_ids and base_use_by_conflict[conflict][row.base_id] < max_occurrences
            ]
            if not candidates:
                raise RuntimeError(f"No candidates left for cell={cell_key} conflict={conflict}")
            min_use = min(base_use_by_conflict[conflict][row.base_id] for row in candidates)
            candidates = [row for row in candidates if base_use_by_conflict[conflict][row.base_id] == min_use]
            chosen = rng.choice(candidates)
            selected_rows.append(chosen)
            selected_ids.add(chosen.sample_id)
            base_use_by_conflict[conflict][chosen.base_id] += 1

    if len(selected_rows) != TOTAL_SELECTED:
        raise RuntimeError(f"Expected {TOTAL_SELECTED} selected rows, found {len(selected_rows)}")
    return selected_rows


def build_pair_targets() -> Dict[Tuple[str, str], int]:
    major_reviewers = [reviewer for reviewer in REVIEWERS if reviewer != "samyek"]
    small_pair_counts = _largest_remainder(REVIEWER_CAPS["samyek"], {reviewer: 1 for reviewer in major_reviewers}, major_reviewers)
    remaining = {reviewer: REVIEWER_CAPS[reviewer] - small_pair_counts[reviewer] for reviewer in major_reviewers}

    ab = (remaining[major_reviewers[0]] + remaining[major_reviewers[1]] - remaining[major_reviewers[2]]) // 2
    ac = (remaining[major_reviewers[0]] + remaining[major_reviewers[2]] - remaining[major_reviewers[1]]) // 2
    bc = (remaining[major_reviewers[1]] + remaining[major_reviewers[2]] - remaining[major_reviewers[0]]) // 2

    pair_targets = {
        tuple(sorted((major_reviewers[0], major_reviewers[1]))): ab,
        tuple(sorted((major_reviewers[0], major_reviewers[2]))): ac,
        tuple(sorted((major_reviewers[1], major_reviewers[2]))): bc,
        tuple(sorted((major_reviewers[0], "samyek"))): small_pair_counts[major_reviewers[0]],
        tuple(sorted((major_reviewers[1], "samyek"))): small_pair_counts[major_reviewers[1]],
        tuple(sorted((major_reviewers[2], "samyek"))): small_pair_counts[major_reviewers[2]],
    }
    return pair_targets


def build_reviewer_targets(selected_rows: List[SelectedRow]) -> Dict[str, Dict[str, Dict[str, int]]]:
    capacities = REVIEWER_CAPS
    order = REVIEWERS

    conflict_totals = Counter(str(row.conflict_category_id) for row in selected_rows)
    model_totals = Counter(row.model for row in selected_rows)
    train_totals = Counter(row.train_type for row in selected_rows)
    prompt_totals = Counter(row.prompt for row in selected_rows)

    targets: Dict[str, Dict[str, Dict[str, int]]] = {reviewer: {"conflict": {}, "model": {}, "train": {}, "prompt": {}} for reviewer in REVIEWERS}
    for bucket_name, totals in (
        ("conflict", conflict_totals),
        ("model", model_totals),
        ("train", train_totals),
        ("prompt", prompt_totals),
    ):
        for bucket_value in sorted(totals):
            per_reviewer = _largest_remainder(int(totals[bucket_value]) * 2, capacities, order)
            for reviewer in REVIEWERS:
                targets[reviewer][bucket_name][str(bucket_value)] = per_reviewer[reviewer]
    return targets


def initial_assignment_state(pair_targets: Dict[Tuple[str, str], int]) -> Dict[str, object]:
    return {
        "pair_remaining": dict(pair_targets),
        "pair_assigned": Counter(),
        "reviewer_seen_base_ids": {reviewer: set() for reviewer in REVIEWERS},
        "reviewer_counts": {
            reviewer: {
                "total": 0,
                "conflict": Counter(),
                "model": Counter(),
                "train": Counter(),
                "prompt": Counter(),
            }
            for reviewer in REVIEWERS
        },
        "sample_to_pair": {},
    }


def _pair_score(
    sample: SelectedRow,
    pair: Tuple[str, str],
    state: Dict[str, object],
    reviewer_targets: Dict[str, Dict[str, Dict[str, int]]],
) -> float:
    pair_remaining: Dict[Tuple[str, str], int] = state["pair_remaining"]  # type: ignore[assignment]
    if pair_remaining[pair] <= 0:
        return float("inf")

    reviewer_seen: Dict[str, set[str]] = state["reviewer_seen_base_ids"]  # type: ignore[assignment]
    reviewer_counts: Dict[str, Dict[str, Counter[str] | int]] = state["reviewer_counts"]  # type: ignore[assignment]
    score = 0.0
    pair_fill_ratio = (state["pair_assigned"][pair] + 1) / (state["pair_assigned"][pair] + pair_remaining[pair])  # type: ignore[index]
    score += 0.25 * pair_fill_ratio

    dims = (
        ("conflict", str(sample.conflict_category_id), 5.0),
        ("model", sample.model, 3.0),
        ("train", sample.train_type, 3.0),
        ("prompt", sample.prompt, 1.0),
    )
    for reviewer in pair:
        counts = reviewer_counts[reviewer]
        current_total = int(counts["total"])
        total_target = REVIEWER_CAPS[reviewer]
        score += 0.75 * ((current_total + 1) / total_target)
        if sample.base_id in reviewer_seen[reviewer]:
            score += 9.0
        for dim_name, dim_value, weight in dims:
            current_dim = counts[dim_name][dim_value]  # type: ignore[index]
            target_dim = reviewer_targets[reviewer][dim_name][dim_value]
            if target_dim <= 0:
                score += 1000.0
                continue
            after_ratio = (current_dim + 1) / target_dim
            overflow = max(0.0, current_dim + 1 - target_dim)
            score += weight * after_ratio
            score += weight * 8.0 * overflow
    return score


def _clone_state(state: Dict[str, object]) -> Dict[str, object]:
    return {
        "pair_remaining": dict(state["pair_remaining"]),  # type: ignore[index]
        "pair_assigned": Counter(state["pair_assigned"]),  # type: ignore[arg-type]
        "reviewer_seen_base_ids": {reviewer: set(base_ids) for reviewer, base_ids in state["reviewer_seen_base_ids"].items()},  # type: ignore[index]
        "reviewer_counts": {
            reviewer: {
                "total": int(counts["total"]),
                "conflict": Counter(counts["conflict"]),
                "model": Counter(counts["model"]),
                "train": Counter(counts["train"]),
                "prompt": Counter(counts["prompt"]),
            }
            for reviewer, counts in state["reviewer_counts"].items()  # type: ignore[index]
        },
        "sample_to_pair": dict(state["sample_to_pair"]),  # type: ignore[index]
    }


def _apply_pair(sample: SelectedRow, pair: Tuple[str, str], state: Dict[str, object]) -> None:
    pair_remaining: Dict[Tuple[str, str], int] = state["pair_remaining"]  # type: ignore[assignment]
    pair_assigned: Counter[Tuple[str, str]] = state["pair_assigned"]  # type: ignore[assignment]
    reviewer_seen: Dict[str, set[str]] = state["reviewer_seen_base_ids"]  # type: ignore[assignment]
    reviewer_counts: Dict[str, Dict[str, Counter[str] | int]] = state["reviewer_counts"]  # type: ignore[assignment]
    pair_remaining[pair] -= 1
    pair_assigned[pair] += 1
    state["sample_to_pair"][sample.sample_id] = pair  # type: ignore[index]
    for reviewer in pair:
        reviewer_seen[reviewer].add(sample.base_id)
        counts = reviewer_counts[reviewer]
        counts["total"] = int(counts["total"]) + 1
        counts["conflict"][str(sample.conflict_category_id)] += 1  # type: ignore[index]
        counts["model"][sample.model] += 1  # type: ignore[index]
        counts["train"][sample.train_type] += 1  # type: ignore[index]
        counts["prompt"][sample.prompt] += 1  # type: ignore[index]


def assign_pairs(
    selected_rows: List[SelectedRow],
    pair_targets: Dict[Tuple[str, str], int],
    reviewer_targets: Dict[str, Dict[str, Dict[str, int]]],
    seed: int,
) -> Dict[str, Tuple[str, str]]:
    rng = random.Random(seed)
    by_base_id: Dict[str, List[SelectedRow]] = defaultdict(list)
    for row in selected_rows:
        by_base_id[row.base_id].append(row)

    duplicate_groups = [rows for rows in by_base_id.values() if len(rows) == 2]
    single_rows = [rows[0] for rows in by_base_id.values() if len(rows) == 1]
    duplicate_groups.sort(key=lambda rows: (rows[0].conflict_category_id, rows[0].base_id))

    all_pairs = list(pair_targets.keys())
    pair_combos = [(all_pairs[i], all_pairs[j]) for i in range(len(all_pairs)) for j in range(i + 1, len(all_pairs))]

    for attempt in range(40):
        state = initial_assignment_state(pair_targets)
        failed = False

        singles = list(single_rows)
        rng.shuffle(singles)
        for sample in singles:
            best_pair: Tuple[float, Tuple[str, str]] | None = None
            for pair in all_pairs:
                score = _pair_score(sample, pair, state, reviewer_targets)
                if math.isinf(score):
                    continue
                choice = (score, pair)
                if best_pair is None or choice < best_pair:
                    best_pair = choice
            if best_pair is None:
                failed = True
                break
            _apply_pair(sample, best_pair[1], state)

        if failed:
            continue

        groups = [list(rows) for rows in duplicate_groups]
        rng.shuffle(groups)
        for rows in groups:
            best_choice: Tuple[float, Tuple[Tuple[str, str], Tuple[str, str]], Tuple[SelectedRow, SelectedRow]] | None = None
            sample_orders = [(rows[0], rows[1]), (rows[1], rows[0])]
            for pair_one, pair_two in pair_combos:
                for first_sample, second_sample in sample_orders:
                    temp_state = _clone_state(state)
                    score_one = _pair_score(first_sample, pair_one, temp_state, reviewer_targets)
                    if math.isinf(score_one):
                        continue
                    _apply_pair(first_sample, pair_one, temp_state)
                    score_two = _pair_score(second_sample, pair_two, temp_state, reviewer_targets)
                    total_score = score_one + score_two
                    if math.isinf(total_score):
                        continue
                    overlap = len(set(pair_one) & set(pair_two))
                    total_score += overlap * 4.0
                    choice = (total_score, (pair_one, pair_two), (first_sample, second_sample))
                    if best_choice is None or total_score < best_choice[0]:
                        best_choice = choice
            if best_choice is None:
                failed = True
                break
            _, (pair_one, pair_two), (first_sample, second_sample) = best_choice
            _apply_pair(first_sample, pair_one, state)
            _apply_pair(second_sample, pair_two, state)

        if failed:
            continue

        reviewer_counts: Dict[str, Dict[str, Counter[str] | int]] = state["reviewer_counts"]  # type: ignore[assignment]
        pair_remaining: Dict[Tuple[str, str], int] = state["pair_remaining"]  # type: ignore[assignment]
        if any(remaining != 0 for remaining in pair_remaining.values()):
            continue
        if any(int(reviewer_counts[reviewer]["total"]) != REVIEWER_CAPS[reviewer] for reviewer in REVIEWERS):
            continue
        return state["sample_to_pair"]  # type: ignore[return-value]

    raise RuntimeError("Could not assign reviewer pairs while satisfying quotas and no-repeat reviewer/base-id constraints.")


def build_assignment_map(sample_to_pair: Dict[str, Tuple[str, str]], seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    reviewer_to_samples: Dict[str, List[str]] = {reviewer: [] for reviewer in REVIEWERS}
    ordered_sample_ids = list(sample_to_pair.keys())
    rng.shuffle(ordered_sample_ids)
    for sample_id in ordered_sample_ids:
        for reviewer in sample_to_pair[sample_id]:
            reviewer_to_samples[reviewer].append(sample_id)
    return reviewer_to_samples


def summarize(
    selected_rows: List[SelectedRow],
    sample_to_pair: Dict[str, Tuple[str, str]],
    pair_targets: Dict[Tuple[str, str], int],
    extra_cells: List[str],
    cell_counts: Dict[str, int],
) -> Dict[str, object]:
    by_sample_id = {row.sample_id: row for row in selected_rows}
    reviewer_summary: Dict[str, Dict[str, object]] = {
        reviewer: {
            "total": 0,
            "conflict": Counter(),
            "model": Counter(),
            "train": Counter(),
            "prompt": Counter(),
            "cell": Counter(),
            "base_id": Counter(),
        }
        for reviewer in REVIEWERS
    }
    pair_summary = Counter(tuple(pair) for pair in sample_to_pair.values())
    selected_conflicts = Counter(str(row.conflict_category_id) for row in selected_rows)
    selected_models = Counter(row.model for row in selected_rows)
    selected_trains = Counter(row.train_type for row in selected_rows)
    selected_prompts = Counter(row.prompt for row in selected_rows)
    selected_cells = Counter(f"{row.model}|{row.prompt}|{row.train_type}" for row in selected_rows)
    base_id_frequency = Counter(row.base_id for row in selected_rows)

    for sample_id, pair in sample_to_pair.items():
        row = by_sample_id[sample_id]
        for reviewer in pair:
            summary = reviewer_summary[reviewer]
            summary["total"] = int(summary["total"]) + 1
            summary["conflict"][str(row.conflict_category_id)] += 1  # type: ignore[index]
            summary["model"][row.model] += 1  # type: ignore[index]
            summary["train"][row.train_type] += 1  # type: ignore[index]
            summary["prompt"][row.prompt] += 1  # type: ignore[index]
            summary["cell"][f"{row.model}|{row.prompt}|{row.train_type}"] += 1  # type: ignore[index]
            summary["base_id"][row.base_id] += 1  # type: ignore[index]

    return {
        "seed": SEED,
        "study_name": STUDY_NAME,
        "selected_total": len(selected_rows),
        "review_total": sum(REVIEWER_CAPS.values()),
        "extra_cells": extra_cells,
        "cell_targets": cell_counts,
        "selected_distributions": {
            "conflict": dict(sorted(selected_conflicts.items(), key=lambda item: int(item[0]))),
            "model": dict(sorted(selected_models.items())),
            "train": dict(sorted(selected_trains.items())),
            "prompt": dict(sorted(selected_prompts.items())),
            "cell": dict(sorted(selected_cells.items())),
            "base_id_frequency": {
                "max_occurrence": max(base_id_frequency.values()),
                "duplicated_base_ids": sum(1 for value in base_id_frequency.values() if value > 1),
            },
        },
        "pair_targets": {" / ".join(pair): pair_targets[pair] for pair in sorted(pair_targets)},
        "pair_actual": {" / ".join(pair): pair_summary[pair] for pair in sorted(pair_targets)},
        "reviewer_summary": {
            reviewer: {
                "total": reviewer_summary[reviewer]["total"],
                "conflict": dict(sorted(reviewer_summary[reviewer]["conflict"].items(), key=lambda item: int(item[0]))),  # type: ignore[index]
                "model": dict(sorted(reviewer_summary[reviewer]["model"].items())),  # type: ignore[index]
                "train": dict(sorted(reviewer_summary[reviewer]["train"].items())),  # type: ignore[index]
                "prompt": dict(sorted(reviewer_summary[reviewer]["prompt"].items())),  # type: ignore[index]
                "cell": dict(sorted(reviewer_summary[reviewer]["cell"].items())),  # type: ignore[index]
                "repeated_base_ids": {
                    "count": sum(1 for value in reviewer_summary[reviewer]["base_id"].values() if value > 1),  # type: ignore[index]
                    "max_repeat": max(reviewer_summary[reviewer]["base_id"].values(), default=1),  # type: ignore[index]
                },
            }
            for reviewer in REVIEWERS
        },
    }


def write_audit_markdown(summary: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append(f"# {summary['study_name']}")
    lines.append("")
    lines.append(f"- Seed: `{summary['seed']}`")
    lines.append(f"- Selected unique sample-variants: `{summary['selected_total']}`")
    lines.append(f"- Total review assignments: `{summary['review_total']}`")
    lines.append(f"- Extra 30-sample cells: `{', '.join(summary['extra_cells'])}`")
    lines.append("")
    lines.append("## Selected Pool")
    lines.append("")
    lines.append(f"- Conflict counts: `{summary['selected_distributions']['conflict']}`")
    lines.append(f"- Model counts: `{summary['selected_distributions']['model']}`")
    lines.append(f"- Train counts: `{summary['selected_distributions']['train']}`")
    lines.append(f"- Prompt counts: `{summary['selected_distributions']['prompt']}`")
    lines.append(f"- Cell counts: `{summary['selected_distributions']['cell']}`")
    lines.append(f"- Base-id duplication: `{summary['selected_distributions']['base_id_frequency']}`")
    lines.append("")
    lines.append("## Pair Quotas")
    lines.append("")
    lines.append("| Pair | Target | Actual |")
    lines.append("| --- | ---: | ---: |")
    for pair_label, target in summary["pair_targets"].items():
        actual = summary["pair_actual"][pair_label]
        lines.append(f"| {pair_label} | {target} | {actual} |")
    lines.append("")
    lines.append("## Reviewer Summary")
    lines.append("")
    for reviewer, data in summary["reviewer_summary"].items():
        lines.append(f"### {reviewer}")
        lines.append("")
        lines.append(f"- Total: `{data['total']}`")
        lines.append(f"- Conflict: `{data['conflict']}`")
        lines.append(f"- Model: `{data['model']}`")
        lines.append(f"- Train: `{data['train']}`")
        lines.append(f"- Prompt: `{data['prompt']}`")
        lines.append(f"- Repeated base ids: `{data['repeated_base_ids']}`")
        lines.append(f"- Cell mix: `{data['cell']}`")
        lines.append("")
    return "\n".join(lines)


def write_sample_matrix(selected_rows: List[SelectedRow], sample_to_pair: Dict[str, Tuple[str, str]]) -> None:
    by_sample_id = {row.sample_id: row for row in selected_rows}
    SAMPLE_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_MATRIX_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "base_id",
                "query",
                "conflict_category_id",
                "conflict_type",
                "model",
                "prompt",
                "train_type",
                "reviewer_1",
                "reviewer_2",
                "source_path",
            ],
        )
        writer.writeheader()
        for sample_id in sorted(sample_to_pair):
            row = by_sample_id[sample_id]
            reviewer_1, reviewer_2 = sample_to_pair[sample_id]
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "base_id": row.base_id,
                    "query": row.query,
                    "conflict_category_id": row.conflict_category_id,
                    "conflict_type": row.conflict_type,
                    "model": row.model,
                    "prompt": row.prompt,
                    "train_type": row.train_type,
                    "reviewer_1": reviewer_1,
                    "reviewer_2": reviewer_2,
                    "source_path": row.source_path,
                }
            )


def persist_study(
    selected_rows: List[SelectedRow],
    reviewer_to_samples: Dict[str, List[str]],
    summary: Dict[str, object],
    sample_to_pair: Dict[str, Tuple[str, str]],
) -> None:
    if STUDY_DIR.exists():
        shutil.rmtree(STUDY_DIR)
    SOURCE_INPUT_TEMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(SOURCE_INPUT_TEMP_PATH, [row.row for row in selected_rows])

    create_study_bundle(SOURCE_INPUT_TEMP_PATH, STUDY_DIR, STUDY_NAME, overwrite=True)
    init_db(STUDY_DIR)
    register_reviewers(STUDY_DIR, REVIEWERS)
    replace_assignments(STUDY_DIR, reviewer_to_samples)
    (STUDY_DIR / "admin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_INPUT_TEMP_PATH, RAW_INPUT_COPY_PATH)
    append_event(
        STUDY_DIR,
        "build_balanced_qwen_llama_e2e_study",
        {
            "reviewers": REVIEWERS,
            "reviewer_caps": REVIEWER_CAPS,
            "selected_total": len(selected_rows),
            "seed": SEED,
        },
    )

    assignments_path = STUDY_DIR / "assignments" / "assignments.json"
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path.write_text(json.dumps(reviewer_to_samples, indent=2), encoding="utf-8")
    for reviewer, sample_ids in reviewer_to_samples.items():
        reviewer_path = STUDY_DIR / "assignments" / f"{reviewer}_sample_ids.txt"
        reviewer_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    AUDIT_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    AUDIT_MD_PATH.write_text(write_audit_markdown(summary), encoding="utf-8")
    write_sample_matrix(selected_rows, sample_to_pair)


def main() -> None:
    cells = discover_cells()
    extra_cells = choose_extra_cells(cells)
    cell_counts, cell_conflict_quotas = build_cell_quotas(cells, extra_cells)
    pool = load_pool(cells)
    selected_rows = select_rows(cells, pool, cell_conflict_quotas, SEED)
    pair_targets = build_pair_targets()
    reviewer_targets = build_reviewer_targets(selected_rows)
    sample_to_pair = assign_pairs(selected_rows, pair_targets, reviewer_targets, SEED)
    reviewer_to_samples = build_assignment_map(sample_to_pair, SEED)
    summary = summarize(selected_rows, sample_to_pair, pair_targets, extra_cells, cell_counts)
    persist_study(selected_rows, reviewer_to_samples, summary, sample_to_pair)
    print(json.dumps({"study_dir": str(STUDY_DIR), "audit_json": str(AUDIT_JSON_PATH)}, indent=2))


if __name__ == "__main__":
    main()
