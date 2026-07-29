#!/usr/bin/env python3
"""
Prepare the benchmark dataset for message-based inference.

This keeps the existing generation pipeline unchanged by producing:
  - a canonical benchmark JSONL under data/splits/
  - message-format JSONLs under data/messages/

Unlike prepare_data.py, this script does not assume Stage-3 fields such as
expected_response or think traces. It emits only system+user turns because the
generator ignores assistant targets at inference time.
"""

import argparse
import importlib.util
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path


def load_prepare_data_module(project_root: Path):
    module_path = project_root / "code" / "data" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def normalize_example(ex, prep):
    ex = deepcopy(ex)

    docs = ex.get("retrieved_docs") or []
    notes = ex.get("per_doc_notes") or []

    old_doc_ids = [doc.get("doc_id", "") for doc in docs]
    new_doc_ids = [f"d{i}" for i in range(1, len(old_doc_ids) + 1)]
    mapping = {
        old_doc_id: new_doc_id
        for old_doc_id, new_doc_id in zip(old_doc_ids, new_doc_ids)
        if old_doc_id and old_doc_id != new_doc_id
    }

    for idx, doc in enumerate(docs, 1):
        doc["doc_id"] = f"d{idx}"

    notes_by_old_id = {}
    for note in notes:
        if isinstance(note, dict):
            notes_by_old_id[note.get("doc_id", "")] = deepcopy(note)

    rebuilt_notes = []
    for old_doc_id, new_doc_id, doc in zip(old_doc_ids, new_doc_ids, docs):
        note = notes_by_old_id.get(old_doc_id, {"doc_id": old_doc_id})
        note["doc_id"] = new_doc_id
        note["verdict"] = prep.normalize_verdict(note.get("verdict"))
        note["verdict_reason"] = prep.trim_words(
            prep.sanitize_doc_ranges(note.get("verdict_reason") or ""),
            80,
        )
        note["key_fact"] = prep.trim_words(note.get("key_fact") or "", 80)
        if note["verdict"] == "irrelevant":
            note["key_fact"] = ""
        note["source_quality"] = prep.normalize_source_quality(note.get("source_quality"))
        if not note.get("verdict_reason"):
            note["verdict_reason"] = prep.default_doc_verdict_reason(note["verdict"], doc)
        rebuilt_notes.append(note)
    ex["per_doc_notes"] = rebuilt_notes

    ex["conflict_type"] = prep.normalize_conflict_type(ex.get("conflict_type"))
    if ex["conflict_type"] not in prep.CANON_TYPES_SET:
        ex["conflict_type"] = "No conflict"

    ex["conflict_reason"] = prep.trim_words(
        prep.sanitize_doc_ranges(ex.get("conflict_reason") or ""),
        50,
    )

    answerable = parse_bool(ex.get("answerable_under_evidence"))
    ex["answerable_under_evidence"] = bool(answerable) if answerable is not None else False

    if mapping:
        for key in (
            "conflict_reason",
            "gold_answer",
            "_benchmark_source_conflict_reason",
            "_gold_conflict_type",
        ):
            if isinstance(ex.get(key), str):
                ex[key] = prep.remap_doc_ids_in_text(ex.get(key) or "", mapping)

    return ex


def build_messages_for_mode(examples, prompts, prep, mode):
    system_prompt, user_template = prompts[mode]
    rows = []
    for ex in examples:
        payload = prep.build_user_payload(ex, mode)
        user_msg = user_template.format(**payload)
        rows.append(
            {
                "id": ex.get("id"),
                "task": "e2e_trace",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
        )
    return rows


def audit_examples(examples):
    conflict_types = Counter()
    answerable_counts = Counter()
    slice_counts = Counter()
    docs_per_example = []
    note_mismatches = 0
    for ex in examples:
        conflict_types[ex.get("conflict_type")] += 1
        answerable_counts[ex.get("answerable_under_evidence")] += 1
        slice_counts[ex.get("_benchmark_slice")] += 1
        docs = ex.get("retrieved_docs") or []
        notes = ex.get("per_doc_notes") or []
        docs_per_example.append(len(docs))
        if [d.get("doc_id") for d in docs] != [n.get("doc_id") for n in notes]:
            note_mismatches += 1
    print(f"[Benchmark] rows={len(examples)}")
    if docs_per_example:
        print(
            "[Benchmark] docs/example "
            f"min={min(docs_per_example)} max={max(docs_per_example)} "
            f"avg={sum(docs_per_example)/len(docs_per_example):.2f}"
        )
    print(f"[Benchmark] conflict_types={dict(sorted(conflict_types.items()))}")
    print(f"[Benchmark] answerable_under_evidence={dict(answerable_counts)}")
    print(f"[Benchmark] slices={dict(sorted(slice_counts.items()))}")
    print(f"[Benchmark] doc_note_alignment_issues={note_mismatches}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        default="data/Benchmark Dataset/benchmark_final_sanitized.jsonl",
    )
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--prompts_dir", default="prompts")
    parser.add_argument("--dataset_label", default="benchmark_final")
    parser.add_argument(
        "--prompt_profile",
        default="default",
        choices=["default", "minimal", "legacy_text_contract", "runtime", "final_only"],
    )
    parser.add_argument("--message_tag", default="")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["e2e", "oracle_conflict", "oracle_notes", "oracle_both"],
        choices=["e2e", "oracle_conflict", "oracle_notes", "oracle_both"],
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prep = load_prepare_data_module(project_root)

    message_tag = args.message_tag.strip()
    if not message_tag and args.prompt_profile != "default":
        message_tag = args.prompt_profile

    examples = [normalize_example(ex, prep) for ex in prep.read_jsonl(args.input_jsonl)]
    audit_examples(examples)

    out_dir = Path(args.out_dir)
    splits_dir = out_dir / "splits"
    messages_dir = out_dir / "messages"
    splits_dir.mkdir(parents=True, exist_ok=True)
    messages_dir.mkdir(parents=True, exist_ok=True)

    canon_path = splits_dir / f"{args.dataset_label}.jsonl"
    prep.write_jsonl(canon_path, examples)
    print(f"[Write] canonical benchmark -> {canon_path}")

    prompts = prep.prompt_paths(args.prompts_dir, prompt_profile=args.prompt_profile)
    for mode in args.modes:
        rows = build_messages_for_mode(examples, prompts, prep, mode)
        out_path = messages_dir / prep.message_filename(
            args.dataset_label,
            mode,
            message_tag,
            task="e2e_trace",
        )
        prep.write_jsonl(out_path, rows)
        print(f"[Messages] {args.dataset_label}/{mode}: kept={len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
