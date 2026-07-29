#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "data/Benchmark Dataset/benchmark_older.jsonl"
RUNTIME_OUTPUT_PATH = (
    ROOT / "model_output_exports/benchmark set/e2e/sft/qwen25_7b/runtime_helper_prompt_outputs.jsonl"
)
OUT_PATH = ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset.jsonl"
MANIFEST_PATH = (
    ROOT / "data/Benchmark Dataset/benchmark_older_high_quality_nonrefusal_subset_manifest.json"
)

EXCLUDED_SLICES = {"natural_questions_no_conflict_200", "refusals_200"}
EXCLUDED_IDS = {"qacc_0175"}  # "not clear ..." answer style; too borderline for a high-confidence subset.


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def predicted_abstain(raw: str) -> bool:
    return "CANNOT ANSWER, INSUFFICIENT EVIDENCE" in raw


def extract_model_final_answer(raw: str) -> str:
    text = raw.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    text = text.replace("[[END-OF-ANSWER]]", "").strip()
    return text


def sanitize_qacc_gold_answer(answer: str) -> str:
    text = " ".join(answer.split())
    patterns = [
        r"\.\s+I chose it because\b",
        r"\.\s+I think\b",
        r"\.\s+While\b",
        r"\.\s+Context\d+\b",
        r"\.\s+The answer\b",
        r"\.\s+All contexts\b",
        r"\.\s+Based on\b",
        r"\.\s+Given\b",
        r"\.\s+It is\b",
        r"\.\s+This\b",
    ]
    cut = len(text)
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            cut = min(cut, match.start())
    text = text[:cut].strip()
    return text.rstrip(".").strip() if text else answer.strip()


def sanitize_health_answer(answer: str) -> str:
    match = re.search(r":\s*(yes|no)\.?\s*$", answer.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return answer.strip()


def curated_gold_answer(row: dict[str, Any], model_final: str) -> tuple[str, str]:
    source_slice = row.get("_benchmark_slice")
    original = (row.get("gold_answer") or "").strip()
    if source_slice == "wikirevision_outdated_200":
        return model_final, "model_final_answer_runtime_qwen25_7b"
    if source_slice == "healthcontradict_conflicting_200":
        return sanitize_health_answer(original), "sanitized_health_gold_answer"
    if source_slice == "qacc_conflicting_200":
        return sanitize_qacc_gold_answer(original), "sanitized_qacc_gold_answer"
    return original, "original_gold_answer"


def main() -> None:
    bench_rows = load_jsonl(BENCH_PATH)
    runtime_rows = {row["id"]: row for row in load_jsonl(RUNTIME_OUTPUT_PATH)}

    selected: list[dict[str, Any]] = []
    reject_counts = Counter()

    for row in bench_rows:
        row_id = row["id"]
        if row.get("_benchmark_slice") in EXCLUDED_SLICES:
            reject_counts["excluded_slice"] += 1
            continue
        if row_id in EXCLUDED_IDS:
            reject_counts["excluded_id"] += 1
            continue
        if row.get("answerable_under_evidence") is not True:
            reject_counts["not_answerable"] += 1
            continue
        runtime = runtime_rows.get(row_id)
        if runtime is None:
            reject_counts["missing_runtime_output"] += 1
            continue
        if predicted_abstain(runtime["raw"]):
            reject_counts["model_abstained"] += 1
            continue

        verdicts = [note.get("verdict") for note in row.get("per_doc_notes", [])]
        verdict_counter = Counter(verdicts)
        if verdict_counter["supports"] < 1:
            reject_counts["no_support_doc"] += 1
            continue

        model_final = extract_model_final_answer(runtime["raw"])
        gold_answer, gold_source = curated_gold_answer(row, model_final)
        if not gold_answer.strip():
            reject_counts["empty_curated_gold"] += 1
            continue

        curated = dict(row)
        curated["_original_gold_answer"] = row.get("gold_answer", "")
        curated["gold_answer"] = gold_answer
        curated["_selection_metadata"] = {
            "subset_name": "benchmark_older_high_quality_nonrefusal_v1",
            "selected_because": [
                "answerable_under_evidence_true",
                "qwen25_7b_run_f_runtime_answered_not_abstained",
                "at_least_one_supports_doc_present",
                "excluded_noisy_slices_and_borderline_ids",
            ],
            "gold_answer_source": gold_source,
            "source_slice": row.get("_benchmark_slice"),
            "support_doc_count": verdict_counter["supports"],
            "partial_doc_count": verdict_counter["partially supports"],
            "irrelevant_doc_count": verdict_counter["irrelevant"],
        }
        curated["_runtime_qwen25_7b_final_answer"] = model_final
        selected.append(curated)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "input_benchmark": str(BENCH_PATH.relative_to(ROOT)),
        "runtime_output": str(RUNTIME_OUTPUT_PATH.relative_to(ROOT)),
        "output_subset": str(OUT_PATH.relative_to(ROOT)),
        "selection_name": "benchmark_older_high_quality_nonrefusal_v1",
        "selection_logic": {
            "required": [
                "answerable_under_evidence == true",
                "qwen25_7b Run-F runtime output is non-abstaining",
                "at least one per_doc_notes verdict == supports",
            ],
            "excluded_slices": sorted(EXCLUDED_SLICES),
            "excluded_ids": sorted(EXCLUDED_IDS),
            "gold_answer_curation": {
                "wikirevision_outdated_200": "replace templated gold_answer with qwen25_7b runtime final answer",
                "healthcontradict_conflicting_200": "reduce template to yes/no",
                "qacc_conflicting_200": "strip rationale tail from answer text",
                "default": "keep original gold_answer",
            },
        },
        "counts": {
            "input_rows": len(bench_rows),
            "selected_rows": len(selected),
            "rejected_rows": len(bench_rows) - len(selected),
            "reject_breakdown": dict(reject_counts),
            "by_slice": dict(Counter(row.get("_benchmark_slice") for row in selected)),
            "by_conflict_type": dict(Counter(row.get("conflict_type") for row in selected)),
            "gold_answer_source_breakdown": dict(
                Counter(row["_selection_metadata"]["gold_answer_source"] for row in selected)
            ),
        },
        "selected_ids": [row["id"] for row in selected],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")

    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
