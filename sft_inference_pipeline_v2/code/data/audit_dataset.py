#!/usr/bin/env python3
"""
audit_dataset.py - Inspect split JSONL files for schema and label issues
=======================================================================

Useful for train/val splits and future benchmark files.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


CANON_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}
TYPE_ALIASES = {
    "Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
    "Conflict due outdated information": "Conflict due to outdated information",
}
ALLOWED_VERDICTS = {"supports", "partially supports", "irrelevant"}
ALLOWED_SOURCE_QUALITY = {"high", "low"}
THINK_OPEN_RE = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"\s*</think>", re.IGNORECASE)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno} bad json: {exc}")


def normalize_conflict_type(label):
    label = re.sub(r"\s+", " ", (label or "").strip())
    return TYPE_ALIASES.get(label, label)


def extract_think_ok(think):
    if not think:
        return False
    m1 = THINK_OPEN_RE.search(think)
    m2 = THINK_CLOSE_RE.search(think)
    return bool(m1 and m2 and m2.start() > m1.end())


def audit_file(path):
    rows = list(read_jsonl(path))
    findings = Counter()
    conflict_types = Counter()
    doc_counts = []
    note_counts = []
    abstain_counts = Counter()
    answerable_counts = Counter()
    sample_issues = []

    for ex in rows:
        cid = ex.get("id")
        docs = ex.get("retrieved_docs") or []
        notes = ex.get("per_doc_notes") or []
        expected = ex.get("expected_response") or {}

        doc_ids = [doc.get("doc_id") for doc in docs if isinstance(doc, dict)]
        note_ids = [note.get("doc_id") for note in notes if isinstance(note, dict)]
        canon_doc_ids = [f"d{i}" for i in range(1, len(doc_ids) + 1)]

        label = normalize_conflict_type(ex.get("conflict_type"))
        conflict_types[label] += 1
        if label not in CANON_TYPES:
            findings["unknown_conflict_type"] += 1
            sample_issues.append((cid, "unknown_conflict_type", ex.get("conflict_type")))

        doc_counts.append(len(doc_ids))
        note_counts.append(len(note_ids))
        abstain_counts[expected.get("abstain")] += 1
        answerable_counts[ex.get("answerable_under_evidence")] += 1

        if doc_ids != canon_doc_ids:
            findings["noncanonical_doc_ids"] += 1
            sample_issues.append((cid, "noncanonical_doc_ids", doc_ids))
        if note_ids != doc_ids:
            findings["note_doc_alignment_issues"] += 1
            sample_issues.append((cid, "note_doc_alignment_issues", note_ids))

        for note in notes:
            verdict = note.get("verdict")
            if verdict not in ALLOWED_VERDICTS:
                findings["bad_verdict"] += 1
                sample_issues.append((cid, "bad_verdict", verdict))
                break
            sq = str(note.get("source_quality") or "").strip().lower()
            if sq not in ALLOWED_SOURCE_QUALITY:
                findings["bad_source_quality"] += 1
                sample_issues.append((cid, "bad_source_quality", sq))
                break
            if verdict == "irrelevant" and (note.get("key_fact") or "").strip():
                findings["irrelevant_has_key_fact"] += 1
                sample_issues.append((cid, "irrelevant_has_key_fact", note.get("doc_id")))
                break

        evidence = expected.get("evidence") or []
        if any(doc_id not in set(doc_ids) for doc_id in evidence):
            findings["evidence_out_of_bounds"] += 1
            sample_issues.append((cid, "evidence_out_of_bounds", evidence))

        abstain = expected.get("abstain")
        if abstain is True and (expected.get("answer") or "").strip() not in {"", "CANNOT ANSWER, INSUFFICIENT EVIDENCE"}:
            findings["abstain_noncanonical_answer"] += 1
            sample_issues.append((cid, "abstain_noncanonical_answer", expected.get("answer")))
        if abstain is not None and ex.get("answerable_under_evidence") is not None:
            if bool(abstain) == bool(ex.get("answerable_under_evidence")):
                findings["abstain_answerable_same_value"] += 1
                sample_issues.append((cid, "abstain_answerable_same_value", (abstain, ex.get("answerable_under_evidence"))))

        if "think" in ex and not extract_think_ok(ex.get("think")):
            findings["think_malformed"] += 1
            sample_issues.append((cid, "think_malformed", (ex.get("think") or "")[:120]))

    print(f"\n=== {path} ===")
    print(f"rows={len(rows)}")
    if doc_counts:
        print(
            f"docs/example min={min(doc_counts)} max={max(doc_counts)} "
            f"avg={sum(doc_counts)/len(doc_counts):.2f}"
        )
    if note_counts:
        print(
            f"notes/example min={min(note_counts)} max={max(note_counts)} "
            f"avg={sum(note_counts)/len(note_counts):.2f}"
        )
    print(f"conflict_types={dict(sorted(conflict_types.items()))}")
    print(f"abstain={dict(abstain_counts)}")
    print(f"answerable_under_evidence={dict(answerable_counts)}")
    if findings:
        print(f"findings={dict(findings)}")
        print(f"sample_issues={sample_issues[:15]}")
    else:
        print("findings={}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", help="One or more split JSONL files to audit")
    args = ap.parse_args()

    for path in args.jsonl:
        audit_file(path)


if __name__ == "__main__":
    main()
