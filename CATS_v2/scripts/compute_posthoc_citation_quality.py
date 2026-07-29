#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute deterministic post-hoc citation-quality diagnostics from an existing CATS run.

This script does not re-run any LLM judge. Instead, it combines:
  1. existing claim-level FG committee outputs from detailed_results.json, and
  2. gold per-doc notes from the original input JSONL.

We report two different notions of citation quality:

  A. committee alignment
     How many cited docs fall inside the committee-approved claim support set
     for that specific claim?

  B. gold-positive cleanliness
     How many cited docs are at least gold-positive at the sample level
     ("supports" / "partially supports"), even if they were not selected by the
     committee for that exact claim?

This is intentionally more scientifically cautious than treating every
non-committee-approved citation as simply "wrong". A cited doc that is
gold-positive but outside the winning claim-specific support set is tracked as a
soft extra citation, not a hard error.

Paper-facing guidance:
  - Treat gold-positive citation precision / hard-negative citation rate as the
    primary deterministic citation-correctness pair.
  - Treat committee-alignment precision as a stricter secondary diagnostic.
  - Treat committee-support citation recall as the citation-sufficiency metric:
    among claims where the committee found a claim-specific support set, how
    often did the model cite at least one of those support docs?
  - The old strict_* metrics are retained for backward compatibility, but their
    names are now explicitly mirrored by clearer "strict_grounded_*" aliases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


_POSITIVE_VERDICTS = {"supports", "support"}
_PARTIAL_TOKENS = ("partial", "weakly support", "weak support")


def _verdict_is_positive(verdict_raw: Optional[str], accept_partial: bool = True) -> bool:
    verdict = (verdict_raw or "").strip().lower().replace("_", " ")
    if verdict in _POSITIVE_VERDICTS:
        return True
    if accept_partial and any(token in verdict for token in _PARTIAL_TOKENS):
        return True
    return False


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_doc_note_map(input_jsonl: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in _read_jsonl(input_jsonl):
        sample_id = row.get("id")
        if not sample_id:
            continue
        notes_by_doc: Dict[str, Dict[str, Any]] = {}
        for note in row.get("per_doc_notes") or []:
            doc_id = note.get("doc_id")
            if doc_id:
                notes_by_doc[doc_id] = {
                    "verdict": note.get("verdict") or "",
                    "key_fact": note.get("key_fact") or "",
                    "quote": note.get("quote") or "",
                    "verdict_reason": note.get("verdict_reason") or "",
                    "source_quality": note.get("source_quality") or "",
                }
        out[sample_id] = notes_by_doc
    return out


def compute_posthoc_citation_quality(
    detailed_results_path: Path,
    input_jsonl: Path,
) -> Dict[str, Any]:
    detailed = json.loads(detailed_results_path.read_text(encoding="utf-8"))
    per_sample = detailed.get("per_sample") or []
    note_map_by_sample = _load_doc_note_map(input_jsonl)

    claim_rows: List[Dict[str, Any]] = []
    committee_alignment_values: List[float] = []
    gold_positive_precision_values: List[float] = []
    total_cited_docs = 0
    total_approved_cited_docs = 0
    total_gold_positive_cited_docs = 0
    total_soft_extra_cited_docs = 0
    total_hard_negative_cited_docs = 0
    strict_committee_alignment_pass_count = 0
    strict_gold_clean_pass_count = 0
    claims_with_any_soft_extra = 0
    claims_with_any_hard_negative = 0
    claims_with_citations = 0
    claims_with_committee_support_set = 0
    claims_with_any_approved_citation = 0
    claims_with_any_gold_positive_citation = 0
    committee_support_citation_recall_values: List[float] = []

    for sample in per_sample:
        sample_id = sample.get("sample_id")
        fg_details = sample.get("factual_grounding_details") or {}
        claim_details = fg_details.get("claim_details") or []
        sample_notes = note_map_by_sample.get(sample_id, {})
        gold_positive_docs = {
            doc_id
            for doc_id, note in sample_notes.items()
            if _verdict_is_positive(note.get("verdict"))
        }

        for idx, claim in enumerate(claim_details):
            cited_docs = list(dict.fromkeys(claim.get("cited_docs") or []))
            supporting_docs = set(claim.get("supporting_docs_found") or [])
            if claim.get("cross_doc_support"):
                supporting_docs.update(claim.get("cross_doc_combo") or [])
            committee_approved_docs = supporting_docs
            has_committee_support_set = bool(committee_approved_docs)

            approved_cited = sorted(set(cited_docs) & committee_approved_docs)
            gold_positive_cited = sorted(set(cited_docs) & gold_positive_docs)
            soft_extra_cited_docs = sorted(
                doc_id
                for doc_id in cited_docs
                if doc_id in gold_positive_docs and doc_id not in committee_approved_docs
            )
            hard_negative_cited_docs = sorted(
                doc_id
                for doc_id in cited_docs
                if doc_id not in gold_positive_docs
            )

            committee_alignment_precision: Optional[float] = None
            gold_positive_precision: Optional[float] = None
            if cited_docs:
                claims_with_citations += 1
                committee_alignment_precision = len(approved_cited) / len(cited_docs)
                gold_positive_precision = len(gold_positive_cited) / len(cited_docs)
                committee_alignment_values.append(committee_alignment_precision)
                gold_positive_precision_values.append(gold_positive_precision)
                total_cited_docs += len(cited_docs)
                total_approved_cited_docs += len(approved_cited)
                total_gold_positive_cited_docs += len(gold_positive_cited)
                total_soft_extra_cited_docs += len(soft_extra_cited_docs)
                total_hard_negative_cited_docs += len(hard_negative_cited_docs)
                if approved_cited:
                    claims_with_any_approved_citation += 1
                if gold_positive_cited:
                    claims_with_any_gold_positive_citation += 1
                if soft_extra_cited_docs:
                    claims_with_any_soft_extra += 1
                if hard_negative_cited_docs:
                    claims_with_any_hard_negative += 1

            if has_committee_support_set:
                claims_with_committee_support_set += 1
                committee_support_citation_recall_values.append(
                    1.0 if approved_cited else 0.0
                )

            strict_committee_alignment_pass = (
                bool(claim.get("supported")) and bool(cited_docs) and not soft_extra_cited_docs and not hard_negative_cited_docs
            )
            if strict_committee_alignment_pass:
                strict_committee_alignment_pass_count += 1

            strict_gold_clean_pass = (
                bool(claim.get("supported")) and bool(cited_docs) and not hard_negative_cited_docs
            )
            if strict_gold_clean_pass:
                strict_gold_clean_pass_count += 1

            claim_rows.append({
                "sample_id": sample_id,
                "claim_index": idx,
                "claim": claim.get("claim"),
                "cited_docs": cited_docs,
                "claim_has_committee_support_set": has_committee_support_set,
                "claim_has_approved_citation": bool(approved_cited),
                "claim_has_gold_positive_citation": bool(gold_positive_cited),
                "claim_has_soft_extra_citation": bool(soft_extra_cited_docs),
                "claim_has_hard_negative_citation": bool(hard_negative_cited_docs),
                "committee_approved_docs": sorted(committee_approved_docs),
                "approved_cited_docs": approved_cited,
                "gold_positive_docs": sorted(gold_positive_docs),
                "gold_positive_cited_docs": gold_positive_cited,
                "soft_extra_cited_docs": soft_extra_cited_docs,
                "hard_negative_cited_docs": hard_negative_cited_docs,
                "committee_alignment_precision": committee_alignment_precision,
                "gold_positive_precision": gold_positive_precision,
                "fg_supported": bool(claim.get("supported")),
                "strict_committee_alignment_pass": strict_committee_alignment_pass,
                "strict_gold_clean_pass": strict_gold_clean_pass,
                "reason": claim.get("reason"),
                "doc_note_snapshot": {
                    doc_id: sample_notes.get(doc_id, {})
                    for doc_id in cited_docs
                },
            })

    total_claims = len(claim_rows)

    summary = {
        "detailed_results_path": str(detailed_results_path),
        "input_jsonl": str(input_jsonl),
        "total_samples": len(per_sample),
        "total_claims": total_claims,
        "claims_with_citations": claims_with_citations,
        "claims_with_committee_support_set": claims_with_committee_support_set,
        "claims_with_any_approved_citation": claims_with_any_approved_citation,
        "claims_with_any_gold_positive_citation": claims_with_any_gold_positive_citation,
        "committee_alignment_precision_macro": (
            mean(committee_alignment_values) if committee_alignment_values else None
        ),
        "committee_alignment_precision_micro": (
            total_approved_cited_docs / total_cited_docs if total_cited_docs else None
        ),
        "gold_positive_citation_precision_macro": (
            mean(gold_positive_precision_values) if gold_positive_precision_values else None
        ),
        "gold_positive_citation_precision_micro": (
            total_gold_positive_cited_docs / total_cited_docs if total_cited_docs else None
        ),
        "soft_extra_citation_rate_micro": (
            total_soft_extra_cited_docs / total_cited_docs if total_cited_docs else None
        ),
        "hard_negative_citation_rate_micro": (
            total_hard_negative_cited_docs / total_cited_docs if total_cited_docs else None
        ),
        "committee_support_citation_recall_macro": (
            mean(committee_support_citation_recall_values)
            if committee_support_citation_recall_values else None
        ),
        "committee_support_citation_recall_micro": (
            claims_with_any_approved_citation / claims_with_committee_support_set
            if claims_with_committee_support_set else None
        ),
        "claims_with_any_soft_extra_rate": (
            claims_with_any_soft_extra / claims_with_citations if claims_with_citations else None
        ),
        "claims_with_any_hard_negative_rate": (
            claims_with_any_hard_negative / claims_with_citations if claims_with_citations else None
        ),
        "strict_committee_alignment_claim_rate": (
            strict_committee_alignment_pass_count / total_claims if total_claims else None
        ),
        "strict_gold_clean_claim_rate": (
            strict_gold_clean_pass_count / total_claims if total_claims else None
        ),
        # Clearer aliases for paper-facing use. The legacy strict_* names are
        # preserved above for backward compatibility.
        "strict_grounded_committee_clean_rate_all_claims": (
            strict_committee_alignment_pass_count / total_claims if total_claims else None
        ),
        "strict_grounded_gold_clean_rate_all_claims": (
            strict_gold_clean_pass_count / total_claims if total_claims else None
        ),
        "total_cited_docs": total_cited_docs,
        "total_approved_cited_docs": total_approved_cited_docs,
        "total_gold_positive_cited_docs": total_gold_positive_cited_docs,
        "total_soft_extra_cited_docs": total_soft_extra_cited_docs,
        "total_hard_negative_cited_docs": total_hard_negative_cited_docs,
    }

    worst_soft_extra_examples = sorted(
        (
            row for row in claim_rows
            if row["committee_alignment_precision"] is not None and row["soft_extra_cited_docs"]
        ),
        key=lambda row: (row["committee_alignment_precision"], -len(row["soft_extra_cited_docs"])),
    )[:20]

    worst_hard_negative_examples = sorted(
        (
            row for row in claim_rows
            if row["gold_positive_precision"] is not None and row["hard_negative_cited_docs"]
        ),
        key=lambda row: (row["gold_positive_precision"], -len(row["hard_negative_cited_docs"])),
    )[:20]

    return {
        "summary": summary,
        "worst_soft_extra_examples": worst_soft_extra_examples,
        "worst_hard_negative_examples": worst_hard_negative_examples,
        "claim_rows": claim_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed-results", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-claims-jsonl", type=Path, default=None)
    args = parser.parse_args()

    result = compute_posthoc_citation_quality(args.detailed_results, args.input_jsonl)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "summary": result["summary"],
                    "worst_soft_extra_examples": result["worst_soft_extra_examples"],
                    "worst_hard_negative_examples": result["worst_hard_negative_examples"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    if args.output_claims_jsonl:
        args.output_claims_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_claims_jsonl.open("w", encoding="utf-8") as handle:
            for row in result["claim_rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
