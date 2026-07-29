#!/usr/bin/env python3
"""
prep_model_outputs_for_eval.py
------------------------------
Prepare model output export JSONLs for the CATS evaluator.

Workflow:
  1. Load the stagewise val gold records (or another supplied gold JSONL).
  2. Load exported model outputs keyed by `id` from `data/model_output_exports/...`.
  3. Strip reasoning traces / scaffolding from `raw` or sanitize `model_output`.
  4. Write evaluator-ready records with `model_output` populated.

The evaluator will then use `model_output` instead of `expected_response.answer`.

Default usage:
  python scripts/prep_model_outputs_for_eval.py

Custom usage:
  python scripts/prep_model_outputs_for_eval.py \
    --gold data/splits/92p5_7p5/stagewise_multi/val/stage3_final.jsonl \
    --exports-root 'data/model_output_exports/val set' \
    --output-root inputs/prepped_model_eval_inputs/val_set_all_modes

Benchmark usage:
  python scripts/prep_model_outputs_for_eval.py \
    --gold data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl \
    --exports-root final_model_outputs \
    --output-root inputs/prepped_model_eval_inputs/benchmark_set_all_modes \
    --fixed-output-name input.jsonl \
    --export-glob '*.sanitized.jsonl'
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_GOLD = "data/splits/92p5_7p5/stagewise_multi/val/stage3_final.jsonl"
DEFAULT_EXPORTS_ROOT = "data/model_output_exports/val set"
DEFAULT_OUTPUT_ROOT = "inputs/prepped_model_eval_inputs/val_set_all_modes"

END_MARKERS = (
    "[[END-OF-ANSWER]]",
    "[[END OF ANSWER]]",
    "[END-OF-ANSWER]",
    "END-OF-ANSWER",
)

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_FULL_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"\bfinal answer\s*:\s*", re.IGNORECASE)
_GENERIC_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:final answer|answer|final|答案|最终答案)\s*[:：]\s*",
    re.IGNORECASE,
)
_STAGE3_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*stage\s*3\s*(?:[-:]\s*)?answer generation\s*:\s*",
    re.IGNORECASE,
)
_STAGE_HEADER_RE = re.compile(
    r"^(?:stage\s+\d+|conflict type:|reason:|evidence pattern:|answer plan:)\b",
    re.IGNORECASE,
)
_CITATION_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:\[\[?d\d+\]?\]\s*)+$", re.IGNORECASE)
_BRACKET_DOC_RE = re.compile(r"\[\[?(d\d+)\]?\]", re.IGNORECASE)

CONFLICT_TYPE_STR_MAP = {
    "no conflict": 1,
    "complementary information": 2,
    "conflicting opinions and research outcomes": 3,
    "conflicting opinions or research outcomes": 3,
    "conflict due to outdated information": 4,
    "conflict due to outdated information (temporal conflict)": 4,
    "conflict due to misinformation": 5,
}
VALID_CONFLICT_TYPES = {1, 2, 3, 4, 5}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _canonical_conflict_type_id(conflict_type: object) -> int | None:
    if not isinstance(conflict_type, str):
        return None
    return CONFLICT_TYPE_STR_MAP.get(conflict_type.strip().lower())


def _normalize_conflict_category_id(row: dict) -> None:
    raw = row.get("conflict_category_id")
    if raw in VALID_CONFLICT_TYPES:
        return
    mapped = _canonical_conflict_type_id(row.get("conflict_type"))
    if mapped is not None:
        row["conflict_category_id"] = mapped


def collect_duplicate_ids(rows: List[dict], label: str) -> Set[str]:
    seen: Set[str] = set()
    dupes: Set[str] = set()
    for row in rows:
        rid = row.get("id")
        if not rid:
            continue
        if rid in seen:
            dupes.add(str(rid))
        else:
            seen.add(str(rid))
    if dupes:
        joined = ", ".join(sorted(dupes)[:10])
        raise SystemExit(f"{label} contains duplicate ids ({len(dupes)} total). Examples: {joined}")
    return seen


def _strip_end_markers(text: str) -> str:
    out = text
    for marker in END_MARKERS:
        if marker in out:
            out = out.split(marker, 1)[0]
    # Some exports contain malformed partial sentinel lines such as `[[END-`
    # before the real marker. Drop any trailing sentinel-like lines directly.
    lines = out.splitlines()
    while lines:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and re.match(r"^\s*\[\[?END", lines[-1], flags=re.IGNORECASE):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def _strip_dangling_think_tags(text: str) -> str:
    """Drop stray outer think tags that survive malformed exports.

    Some benchmark exports contain an otherwise clean final answer followed by an
    extra trailing `</think>` line, or begin with a naked `<think>` token after
    earlier sanitization steps. Remove only the unmatched outer tags and leave
    interior content untouched.
    """
    out = text.strip()
    out = re.sub(r"^\s*<think>\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*</think>\s*$", "", out, flags=re.IGNORECASE)
    return out.strip()


def _is_citation_only_chunk(chunk: str) -> bool:
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    return bool(lines) and all(_CITATION_LINE_RE.match(ln) for ln in lines)


def _normalize_citation_chunk(chunk: str) -> str:
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    doc_ids: List[str] = []
    seen = set()
    for ln in lines:
        for did in _BRACKET_DOC_RE.findall(ln):
            did = did.lower()
            if did not in seen:
                doc_ids.append(did)
                seen.add(did)
    if not doc_ids:
        return chunk.strip()
    return "\n".join(f"[{did}]" for did in doc_ids)


def _strip_balanced_outer_square_brackets(text: str) -> str:
    out = text.strip()
    while len(out) >= 2 and out.startswith("[") and out.endswith("]"):
        depth = 0
        balanced = True
        for i, ch in enumerate(out):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
                if depth == 0 and i != len(out) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        out = out[1:-1].strip()
    return out


def _extract_answer_from_think_blocks(text: str) -> str:
    for block in reversed(_THINK_BLOCK_RE.findall(text)):
        block = block.strip()
        if not block:
            continue
        matches = list(_STAGE3_ANSWER_RE.finditer(block))
        if not matches:
            continue
        candidate = block[matches[-1].end():].strip()
        candidate = _strip_end_markers(candidate)
        candidate = _strip_balanced_outer_square_brackets(candidate.strip())
        if candidate:
            return candidate
    return ""


def extract_final_answer(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    text = _strip_end_markers(text)
    think_answer = _extract_answer_from_think_blocks(text)
    text = _FULL_THINK_BLOCK_RE.sub("", text).strip()
    if not text and think_answer:
        return _strip_dangling_think_tags(think_answer)

    if _FINAL_ANSWER_RE.search(text):
        text = _FINAL_ANSWER_RE.split(text)[-1].strip()
        text = _strip_end_markers(text)
        text = _strip_balanced_outer_square_brackets(text.strip())
        return _strip_dangling_think_tags(text or think_answer)

    generic_matches = list(_GENERIC_ANSWER_RE.finditer(text))
    if generic_matches:
        text = text[generic_matches[-1].end():].strip()
        text = _strip_end_markers(text)
        text = _strip_balanced_outer_square_brackets(text.strip())
        return _strip_dangling_think_tags(text or think_answer)

    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    if not chunks:
        return _strip_dangling_think_tags(think_answer or text.strip())

    # Minimal-prompt outputs are often already just the final answer.
    if len(chunks) == 1 and not _STAGE_HEADER_RE.match(chunks[0]):
        return _strip_dangling_think_tags(_strip_balanced_outer_square_brackets(chunks[0].strip()))

    # Find the last non-stage chunk.
    last_idx = len(chunks) - 1
    while last_idx >= 0 and _STAGE_HEADER_RE.match(chunks[last_idx]):
        last_idx -= 1
    if last_idx < 0:
        return _strip_dangling_think_tags(think_answer or _strip_balanced_outer_square_brackets(text.strip()))

    # If the tail chunk is citation-only, join it to the previous answer chunk.
    if _is_citation_only_chunk(chunks[last_idx]) and last_idx > 0:
        answer_chunk = chunks[last_idx - 1].strip()
        citations = _normalize_citation_chunk(chunks[last_idx])
        return _strip_dangling_think_tags(
            _strip_balanced_outer_square_brackets(f"{answer_chunk}\n\n{citations}".strip())
        )

    final_chunk = _strip_balanced_outer_square_brackets(chunks[last_idx].strip())
    return _strip_dangling_think_tags(final_chunk or think_answer)


def get_export_answer(row: dict, allow_expected_response_answer: bool = False) -> Tuple[str, str]:
    if row.get("model_output") is not None:
        return extract_final_answer(row.get("model_output", "")), "model_output"
    if row.get("raw") is not None:
        return extract_final_answer(row.get("raw", "")), "raw"
    if allow_expected_response_answer:
        expected = row.get("expected_response")
        if isinstance(expected, dict) and expected.get("answer") is not None:
            return extract_final_answer(expected.get("answer", "")), "expected_response.answer"
    return "", "missing"


def prepare_file(
    gold_by_id: Dict[str, dict],
    gold_count: int,
    export_path: Path,
    output_path: Path,
    allow_expected_response_answer: bool = False,
) -> Tuple[int, int, int, int]:
    exports = read_jsonl(export_path)
    export_ids = collect_duplicate_ids(exports, f"Export file {export_path}")
    prepped: List[dict] = []
    missing_gold = 0
    empty_answers = 0

    for row in exports:
        rid = row.get("id")
        base = gold_by_id.get(rid)
        if base is None:
            missing_gold += 1
            continue

        out = dict(base)
        _normalize_conflict_category_id(out)
        final_answer, answer_source = get_export_answer(
            row, allow_expected_response_answer=allow_expected_response_answer
        )
        if not final_answer.strip():
            empty_answers += 1

        out["model_output"] = final_answer.strip()
        out["model_output_raw"] = row.get("raw", row.get("model_output", ""))
        out["model_output_field"] = answer_source
        out["model_output_source"] = str(export_path)
        prepped.append(out)

    missing_exports = gold_count - len(export_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in prepped:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(prepped), missing_gold, empty_answers, missing_exports


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exported model outputs for CATS evaluation.")
    parser.add_argument("--gold", default=DEFAULT_GOLD, help="Gold/base JSONL to merge onto.")
    parser.add_argument("--exports-root", default=DEFAULT_EXPORTS_ROOT, help="Root directory of exported model JSONLs.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for prepared evaluator inputs.")
    parser.add_argument(
        "--fixed-output-name",
        default=None,
        help="If set, write every prepared file using this basename inside its leaf directory (for example: input.jsonl).",
    )
    parser.add_argument(
        "--export-glob",
        default="*.jsonl",
        help="Glob used under exports-root to select input files (default: *.jsonl).",
    )
    parser.add_argument(
        "--allow-expected-response-answer",
        action="store_true",
        help="If model_output/raw is missing, use expected_response.answer as the exported answer source.",
    )
    parser.add_argument(
        "--fail-on-incomplete-id-coverage",
        action="store_true",
        help="Exit non-zero if any export file is missing gold ids or contains non-gold ids.",
    )
    parser.add_argument(
        "--fail-on-empty-answers",
        action="store_true",
        help="Exit non-zero if any prepared output row has an empty extracted model_output.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    exports_root = Path(args.exports_root)
    output_root = Path(args.output_root)

    gold_rows = read_jsonl(gold_path)
    gold_ids = collect_duplicate_ids(gold_rows, f"Gold file {gold_path}")
    gold_by_id = {row.get("id"): row for row in gold_rows if row.get("id")}

    export_files = sorted(exports_root.rglob(args.export_glob))
    if not export_files:
        raise SystemExit(f"No export files matching {args.export_glob!r} found under {exports_root}")

    print(f"Gold records: {len(gold_rows)} from {gold_path}")
    print(f"Gold unique ids: {len(gold_ids)}")
    print(f"Export files: {len(export_files)} from {exports_root} (glob={args.export_glob!r})")

    total_records = 0
    total_missing_gold = 0
    total_empty_answers = 0
    total_missing_exports = 0
    had_error = False

    for export_path in export_files:
        rel = export_path.relative_to(exports_root)
        if args.fixed_output_name:
            rel = rel.with_name(args.fixed_output_name)
        output_path = output_root / rel
        count, missing_gold, empty_answers, missing_exports = prepare_file(
            gold_by_id,
            len(gold_ids),
            export_path,
            output_path,
            allow_expected_response_answer=args.allow_expected_response_answer,
        )
        total_records += count
        total_missing_gold += missing_gold
        total_empty_answers += empty_answers
        total_missing_exports += missing_exports
        if (
            args.fail_on_incomplete_id_coverage
            and (missing_gold > 0 or missing_exports > 0)
        ):
            had_error = True
        if args.fail_on_empty_answers and empty_answers > 0:
            had_error = True
        print(
            f"Wrote {count:>2} records -> {output_path} "
            f"(missing_gold={missing_gold}, missing_exports={missing_exports}, empty_answers={empty_answers})"
        )

    print("\nSummary")
    print(f"  prepared_records: {total_records}")
    print(f"  missing_gold_ids: {total_missing_gold}")
    print(f"  missing_export_ids: {total_missing_exports}")
    print(f"  empty_model_outputs: {total_empty_answers}")

    if had_error:
        print("\nOne or more strict validation checks failed.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
