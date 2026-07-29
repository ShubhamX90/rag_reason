#!/usr/bin/env python3
"""
Prepare the 6 local other-techniques benchmark files for CATS local committee.

These source files already contain the benchmark row ids/order and model outputs,
but they are not in the strict benchmark-prepped schema used by the current
local-committee pipeline. This script rebuilds evaluator-ready inputs by:

  1. loading the canonical 736-row benchmark gold JSONL,
  2. loading each source file under inputs/other_techniques,
  3. deterministically sanitizing the source model output when needed,
  4. merging the sanitized model output onto the canonical gold row, and
  5. writing benchmark-prepped `input.jsonl` files under
     inputs/prepped_model_eval_inputs/other_techniques/...

It also emits an audit JSON describing any structural or content anomalies.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prep_model_outputs_for_eval import extract_final_answer


EMPTY_PLACEHOLDER = "[EMPTY MODEL OUTPUT]"


@dataclass(frozen=True)
class FileSpec:
    source_rel: str
    output_subdir: str
    label: str


STANDARD_FILE_SPECS: List[FileSpec] = [
    FileSpec(
        source_rel="inputs/other_techniques/CoN/con_llama.cats.jsonl",
        output_subdir="con/llama",
        label="con_llama",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/CoN/con_mistral_holdout.cats.jsonl",
        output_subdir="con/mistral",
        label="con_mistral",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/CoN/con_qwen_holdout.cats.jsonl",
        output_subdir="con/qwen",
        label="con_qwen",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/CoT/cot_fewshot_llama.cats.jsonl",
        output_subdir="cot_fewshot/llama",
        label="cot_fewshot_llama",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/CoT/cot_fewshot_mistral_holdout.cats.jsonl",
        output_subdir="cot_fewshot/mistral",
        label="cot_fewshot_mistral",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/CoT/cot_fewshot_qwen_holdout.cats.jsonl",
        output_subdir="cot_fewshot/qwen",
        label="cot_fewshot_qwen",
    ),
]


FIXED_FILE_SPECS: List[FileSpec] = [
    FileSpec(
        source_rel="inputs/other_techniques/CoN/con_llama.cats.jsonl",
        output_subdir="con/llama",
        label="con_llama",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/fixed files/con_mistral_holdout.cats.jsonl",
        output_subdir="con/mistral",
        label="con_mistral",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/fixed files/con_qwen_holdout.cats.jsonl",
        output_subdir="con/qwen",
        label="con_qwen",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/CoT/cot_fewshot_llama.cats.jsonl",
        output_subdir="cot_fewshot/llama",
        label="cot_fewshot_llama",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/fixed files/cot_fewshot_mistral_holdout.cats.jsonl",
        output_subdir="cot_fewshot/mistral",
        label="cot_fewshot_mistral",
    ),
    FileSpec(
        source_rel="inputs/other_techniques/fixed files/cot_fewshot_qwen_holdout.cats.jsonl",
        output_subdir="cot_fewshot/qwen",
        label="cot_fewshot_qwen",
    ),
]


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                raise SystemExit(f"{path}: blank line at {lineno}")
            rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_conversation_leakage(text: str) -> tuple[str, bool]:
    marker = "\nHuman:"
    if marker not in text:
        return text, False
    return text.split(marker, 1)[0].rstrip(), True


def sanitize_other_technique_output(text: str) -> tuple[str, dict]:
    raw = text if isinstance(text, str) else ""
    raw = raw.replace("\r\n", "\n")
    stripped = raw.strip()

    leak_stripped, had_human_leak = strip_conversation_leakage(stripped)
    cleaned = extract_final_answer(leak_stripped)
    if not cleaned.strip():
        cleaned = leak_stripped.strip()

    was_empty = not cleaned.strip()
    if was_empty:
        cleaned = EMPTY_PLACEHOLDER

    meta = {
        "had_human_leakage": had_human_leak,
        "used_empty_placeholder": was_empty,
        "changed": cleaned != stripped,
    }
    return cleaned, meta


def prepare_rows(
    *,
    spec: FileSpec,
    repo_root: Path,
    gold_ids: List[str],
    gold_by_id: Dict[str, dict],
    output_root_rel: str,
) -> tuple[List[dict], dict]:
    source_path = repo_root / spec.source_rel
    rows = read_jsonl(source_path)
    ids = [row.get("id") for row in rows]

    if ids != gold_ids:
        raise SystemExit(
            f"{source_path}: row id order does not exactly match benchmark gold order"
        )

    prepared: List[dict] = []
    leak_count = 0
    placeholder_ids: List[str] = []
    changed_count = 0
    sanitized_outputs: List[str] = []

    for source_row in rows:
        rid = source_row["id"]
        gold_row = dict(gold_by_id[rid])

        cleaned_output, meta = sanitize_other_technique_output(source_row.get("model_output", ""))
        leak_count += int(meta["had_human_leakage"])
        changed_count += int(meta["changed"])
        if meta["used_empty_placeholder"]:
            placeholder_ids.append(rid)
        sanitized_outputs.append(cleaned_output)

        gold_row["model_output"] = cleaned_output
        gold_row["model_output_raw"] = source_row.get("model_output", "")
        gold_row["model_output_field"] = "model_output"
        gold_row["model_output_source"] = spec.source_rel
        gold_row["other_technique_label"] = spec.label
        gold_row["other_technique_had_human_leakage"] = meta["had_human_leakage"]
        gold_row["other_technique_used_empty_placeholder"] = meta["used_empty_placeholder"]
        prepared.append(gold_row)

    counts = Counter(sanitized_outputs)
    most_common_output, most_common_count = counts.most_common(1)[0]
    audit = {
        "label": spec.label,
        "source_path": spec.source_rel,
        "output_path": f"{output_root_rel}/{spec.output_subdir}/input.jsonl",
        "rows": len(rows),
        "unique_source_outputs": len(Counter((row.get("model_output", "") or "").strip() for row in rows)),
        "unique_sanitized_outputs": len(counts),
        "sanitizer_changed_rows": changed_count,
        "human_leakage_rows": leak_count,
        "empty_placeholder_rows": len(placeholder_ids),
        "empty_placeholder_ids": placeholder_ids,
        "most_common_sanitized_output_count": most_common_count,
        "most_common_sanitized_output_preview": most_common_output[:240],
        "degenerate_single_output_after_sanitization": most_common_count == len(rows),
    }
    return prepared, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-set",
        choices=("standard", "fixed"),
        default="standard",
        help="Which other-techniques source set to prepare.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root relative to repo root. Defaults depend on --source-set.",
    )
    parser.add_argument(
        "--gold",
        default="data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl",
        help="Canonical benchmark gold JSONL.",
    )
    parser.add_argument(
        "--audit-out",
        default=None,
        help="Where to write the prep audit JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    file_specs = STANDARD_FILE_SPECS if args.source_set == "standard" else FIXED_FILE_SPECS
    output_root_rel = args.output_root
    if output_root_rel is None:
        if args.source_set == "standard":
            output_root_rel = "inputs/prepped_model_eval_inputs/other_techniques"
        else:
            output_root_rel = "inputs/prepped_model_eval_inputs/other_techniques_fixed"

    gold_path = (repo_root / args.gold).resolve()
    gold_rows = read_jsonl(gold_path)
    gold_ids = [row["id"] for row in gold_rows]
    gold_by_id = {row["id"]: row for row in gold_rows}

    audits = []
    for spec in file_specs:
        prepared, audit = prepare_rows(
            spec=spec,
            repo_root=repo_root,
            gold_ids=gold_ids,
            gold_by_id=gold_by_id,
            output_root_rel=output_root_rel,
        )
        output_path = repo_root / output_root_rel / spec.output_subdir / "input.jsonl"
        write_jsonl(output_path, prepared)
        audits.append(audit)
        print(f"WROTE {output_path} ({len(prepared)} rows)")

    audit_out = args.audit_out
    if audit_out is None:
        if args.source_set == "standard":
            audit_out = "outputs/benchmark_local_committee_3judge/prep_audits/other_techniques_benchmark_prep_audit.json"
        else:
            audit_out = "outputs/benchmark_local_committee_3judge/prep_audits/other_techniques_fixed_benchmark_prep_audit.json"

    audit_path = repo_root / audit_out
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "source_set": args.source_set,
        "gold_path": str(gold_path.relative_to(repo_root)),
        "output_root": output_root_rel,
        "files": audits,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
