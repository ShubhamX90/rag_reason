from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .logic import (
    answered_flags,
    eligible_fg_docs,
    extract_claims_with_citations,
    get_model_output,
    gold_answerable_from_record,
    merge_docs_with_notes,
    split_think_trace,
    strip_think_trace,
)

_SINGLE_TRUTH_TYPES = {1, 2, 4, 5}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StudyStats:
    total_samples: int
    correct_refusals: int
    behavior_applicable: int
    fg_applicable: int
    str_applicable: int


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def single_truth_applicable(record: Dict[str, Any]) -> bool:
    # Mirror main CATS evaluator semantics without depending on the parent repo.
    return bool(record.get("gold_answer")) and record.get("conflict_category_id") in _SINGLE_TRUTH_TYPES


def normalize_record(record: Dict[str, Any], order_index: int, source_path: Path) -> Dict[str, Any]:
    raw_output = record.get("model_output_raw")
    canonical_output = get_model_output(record)
    if raw_output:
        think_trace, raw_answer = split_think_trace(str(raw_output))
    else:
        think_trace, raw_answer = "", ""
    stripped_answer = strip_think_trace(canonical_output or raw_answer)
    gold_answerable = gold_answerable_from_record(record, accept_partial=True)
    pred_answered = answered_flags([stripped_answer])[0]
    correct_refusal = (not gold_answerable) and (not pred_answered)
    merged_docs = merge_docs_with_notes(record)
    claims_with_citations = extract_claims_with_citations(stripped_answer, max_claims=12)
    normalized = {
        "sample_id": record.get("id", f"sample_{order_index:06d}"),
        "order_index": order_index,
        "query": record.get("query", ""),
        "retrieved_docs": record.get("retrieved_docs") or [],
        "per_doc_notes": record.get("per_doc_notes") or [],
        "docs_with_notes": merged_docs,
        "fg_eligible_docs": eligible_fg_docs(merged_docs),
        "conflict_category_id": record.get("conflict_category_id"),
        "conflict_type": record.get("conflict_type", ""),
        "conflict_reason": record.get("conflict_reason", ""),
        "gold_answer": record.get("gold_answer"),
        "answerable_under_evidence": record.get("answerable_under_evidence"),
        "gold_answerable": gold_answerable,
        "model_output": canonical_output,
        "model_output_raw": raw_output or canonical_output,
        "stripped_answer": stripped_answer,
        "think_trace": think_trace,
        "pred_answered": pred_answered,
        "correct_refusal": correct_refusal,
        "claims_with_citations": claims_with_citations,
        "single_truth_applicable": single_truth_applicable(record),
        "metadata": {
            "source_input_jsonl": str(source_path),
            "model_output_field": record.get("model_output_field"),
            "model_output_source": record.get("model_output_source"),
            "model_output_raw_present": raw_output is not None,
        },
    }
    return normalized


def create_study_bundle(
    input_jsonl: Path,
    study_dir: Path,
    study_name: str,
    overwrite: bool = False,
) -> Dict[str, Any]:
    input_jsonl = input_jsonl.resolve()
    study_dir = study_dir.resolve()
    if study_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Study directory already exists: {study_dir}")
        shutil.rmtree(study_dir)
    (study_dir / "data").mkdir(parents=True, exist_ok=True)
    (study_dir / "state").mkdir(parents=True, exist_ok=True)
    (study_dir / "assignments").mkdir(parents=True, exist_ok=True)
    (study_dir / "exports").mkdir(parents=True, exist_ok=True)

    normalized_rows: List[Dict[str, Any]] = []
    correct_refusals = 0
    str_applicable = 0
    for idx, record in enumerate(_iter_jsonl(input_jsonl)):
        normalized = normalize_record(record, idx, input_jsonl)
        normalized_rows.append(normalized)
        if normalized["correct_refusal"]:
            correct_refusals += 1
        if normalized["single_truth_applicable"]:
            str_applicable += 1

    _write_jsonl(study_dir / "data" / "samples.jsonl", normalized_rows)

    stats = StudyStats(
        total_samples=len(normalized_rows),
        correct_refusals=correct_refusals,
        behavior_applicable=len(normalized_rows) - correct_refusals,
        fg_applicable=len(normalized_rows) - correct_refusals,
        str_applicable=str_applicable,
    )
    manifest = {
        "study_id": study_name.lower().replace(" ", "_"),
        "study_name": study_name,
        "created_at": utc_now_iso(),
        "source_input_jsonl": str(input_jsonl),
        "data_file": "data/samples.jsonl",
        "metrics_version": "cats_human_eval_cli_v0_1",
        "claim_extraction_version": "cats_v2_deterministic_extract_claims_with_citations",
        "stats": stats.__dict__,
    }
    with (study_dir / "study.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)
    return manifest


def load_manifest(study_dir: Path) -> Dict[str, Any]:
    with (study_dir / "study.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_samples(study_dir: Path) -> List[Dict[str, Any]]:
    manifest = load_manifest(study_dir)
    data_path = study_dir / manifest["data_file"]
    return list(_iter_jsonl(data_path))


def sample_index(study_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {sample["sample_id"]: sample for sample in load_samples(study_dir)}
