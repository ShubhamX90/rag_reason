#!/usr/bin/env python3
"""
eval_contract.py
----------------
Contract-checker for trace-text generations.

- canon_jsonl: Stage-3 dev schema with retrieved_docs and conflict_type
  (e.g., data/splits/dev.jsonl).
- gens_jsonl: model generations with fields {"id": ..., "raw": ...}
  (from generate_textmode_v5.py, optionally sanitized).

Checks:
- Exactly one <think>...</think> block; no nested tags.
- Stage 1 doc verdict lines cover the retrieved doc ids in order. Legacy
  JSON-array traces are still accepted for backward-compatible diagnostics.
- Stage 2 contains a canonical conflict label line.
- FINAL section after </think> either abstains with the canonical phrase or
  includes in-range citations on enough final-answer sentences.
- Contract abstention failures are only triggered when true supporting docs
  are present; partial-only evidence is tracked diagnostically.
- Sentence-level citation coverage must reach 75%.
- Sentinel [[END-OF-ANSWER]] must appear somewhere in the tail.

Usage:
  python code/eval/eval_text_ccontract_v5.py \
    --canon_jsonl data/splits/dev.jsonl \
    --gens_jsonl  outputs/dev_generations/sft_qlora_v5_run2.sanitized.jsonl \
    --report_json outputs/eval_reports/sft_qlora_v5_run2.text_contract.json
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

THINK_OPEN = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE = re.compile(r"\s*</think>", re.IGNORECASE)
DOC_RANGE = re.compile(r"d\d+\s*[–-]\s*d\d+", re.IGNORECASE)
CITE = re.compile(r"\[d(\d+)\]")
ZERO_WIDTH = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")

CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(CONFLICT_TYPES)}
IDX_TO_TYPE = {i: t for t, i in TYPE_TO_IDX.items()}

ABSTAIN_CANON = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
ABSTAIN_PAT = re.compile(
    r"^\s*cannot\s+answer\s*[,:\-]?\s*insufficient\s+evidence\.?\s*$",
    re.IGNORECASE,
)
MIN_CITATION_SENTENCE_COVERAGE = 0.75


def read_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{p}:{ln} bad json: {e}")


def extract_think_block(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = ZERO_WIDTH.sub("", text or "")
    m1 = THINK_OPEN.search(text)
    m2 = THINK_CLOSE.search(text)
    if not m1 or not m2 or m2.start() <= m1.end():
        return None, "think_block_missing_or_misaligned"
    before = text[: m1.start()] + text[m2.end() :]
    if THINK_OPEN.search(before) or THINK_CLOSE.search(before):
        return None, "think_block_not_unique"
    return text[m1.end() : m2.start()], None


def json_array_from_block(block: str) -> Tuple[Optional[List[Any]], Optional[str], Optional[int]]:
    start = block.find("[]")
    start = block.find("[") if start == -1 else start
    if start < 0:
        return None, "no_json_array", None
    depth = 0
    in_str = False
    esc = False
    end_idx = None
    for i in range(start, len(block)):
        ch = block[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
    if end_idx is None:
        return None, "json_array_unbalanced", None
    arr_text = block[start:end_idx]
    try:
        arr = json.loads(arr_text)
    except Exception as e:
        return None, f"json_array_parse_error: {e}", None
    return arr, None, end_idx


def text_doc_verdicts_from_block(block: str) -> Dict[str, str]:
    verdicts: Dict[str, str] = {}
    for line in (block or "").splitlines():
        m = re.match(
            r"^\s*-\s*(d\d+)\s*:\s*(supports|partially supports|irrelevant)\b",
            line,
            flags=re.IGNORECASE,
        )
        if m:
            verdicts[m.group(1)] = m.group(2).lower()
    return verdicts


def conflict_line_from_block(block: str, max_conflict_reason_words: int) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    arr, err, end_idx = json_array_from_block(block)
    tail = block.strip() if err else block[end_idx:].strip()
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return None, "conflict_line_missing"
    line = ""
    for candidate in lines:
        if candidate.lower().startswith("conflict type:"):
            t = candidate.split(":", 1)[1].strip()
            if t not in CONFLICT_TYPES:
                return None, "conflict_type_invalid"
            return (t, ""), None
        for sep in (" - ", " — ", " – ", ":"):
            if sep not in candidate:
                continue
            left = candidate.split(sep, 1)[0].strip()
            if left in CONFLICT_TYPES:
                line = candidate
                break
        if line:
            break
    if not line:
        return None, "conflict_line_missing"
    # Prefer the canonical ASCII separator first so dashes inside the reason
    # do not corrupt label parsing.
    if " - " in line:
        t, r = line.split(" - ", 1)
    elif " — " in line:
        t, r = line.split(" — ", 1)
    elif " – " in line:
        t, r = line.split(" – ", 1)
    elif ":" in line:
        t, r = line.split(":", 1)
    else:
        return None, "conflict_line_bad_dash"
    if t not in CONFLICT_TYPES:
        return None, "conflict_type_invalid"
    if max_conflict_reason_words > 0 and len(r.split()) > max_conflict_reason_words:
        return None, "conflict_reason_too_long"
    if DOC_RANGE.search(line):
        return None, "doc_range_in_conflict_reason"
    return (t, r), None


def sentence_split(s: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return [p for p in parts if p]


def is_abstain_tail(tail: str) -> bool:
    t = ZERO_WIDTH.sub("", tail or "")
    t = t.replace("[[END-OF-ANSWER]]", "")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0]
    if first == ABSTAIN_CANON:
        return True
    if ABSTAIN_PAT.match(first):
        return True
    return any(ln == ABSTAIN_CANON for ln in lines)


def gold_abstain_from_canon(canon: Dict[str, Any]) -> Optional[bool]:
    expected = canon.get("expected_response") or {}
    value = expected.get("abstain")
    if isinstance(value, bool):
        return value
    answerable = canon.get("answerable_under_evidence")
    if isinstance(answerable, bool):
        return not answerable
    return None


def canon_has_supporting_doc(canon: Dict[str, Any]) -> bool:
    for note in canon.get("per_doc_notes") or []:
        verdict = (note.get("verdict") or "").strip().lower()
        if verdict == "supports":
            return True
    return False


def canon_has_partial_doc(canon: Dict[str, Any]) -> bool:
    for note in canon.get("per_doc_notes") or []:
        verdict = (note.get("verdict") or "").strip().lower()
        if verdict == "partially supports":
            return True
    return False


def validate_example(
    gen: Dict[str, Any],
    canon: Dict[str, Any],
    max_verdict_reason_words: int,
    max_conflict_reason_words: int,
) -> List[str]:
    text = gen.get("raw", "")
    problems: List[str] = []

    # sentinel check
    if "[[END-OF-ANSWER]]" not in text:
        problems.append("sentinel_missing")

    block, err = extract_think_block(text)
    if err:
        problems.append(err)
        return problems

    exp_docs = [d["doc_id"] for d in canon.get("retrieved_docs", [])]
    allowed_verdicts = {"supports", "partially supports", "irrelevant"}
    allowed_sq = {"high", "low"}
    any_support = False
    any_partial_support = False

    arr, err, _ = json_array_from_block(block)
    if err:
        pred_doc_verdicts = text_doc_verdicts_from_block(block)
        if not pred_doc_verdicts:
            problems.append(err)
            return problems
        if list(pred_doc_verdicts.keys()) != exp_docs:
            problems.append("doc_id_order_or_membership_mismatch")
        for did, v in pred_doc_verdicts.items():
            if v not in allowed_verdicts:
                problems.append(f"bad_verdict_{did}")
            if v == "supports":
                any_support = True
            elif v == "partially supports":
                any_partial_support = True
    else:
        if [o.get("doc_id") for o in arr] != exp_docs:
            problems.append("doc_id_order_or_membership_mismatch")
        for i, o in enumerate(arr, 1):
            if not isinstance(o, dict):
                problems.append(f"array_item_[{i}]_not_object")
                continue
            if DOC_RANGE.search(json.dumps(o, ensure_ascii=False)):
                problems.append("doc_range_in_array")
            if o.get("doc_id") != f"d{i}":
                problems.append(f"doc_id_not_d{i}")
            v = (o.get("verdict", "") or "").strip().lower()
            if v not in allowed_verdicts:
                problems.append(f"bad_verdict_{o.get('doc_id')}")
            if v == "supports":
                any_support = True
            elif v == "partially supports":
                any_partial_support = True
            vr = (o.get("verdict_reason", "") or "").strip()
            if max_verdict_reason_words > 0 and len(vr.split()) > max_verdict_reason_words:
                problems.append(f"verdict_reason_too_long_{o.get('doc_id')}")
            sq = (o.get("source_quality", "") or "").strip().lower()
            if sq not in allowed_sq:
                problems.append(f"bad_source_quality_{o.get('doc_id')}")

    _, err = conflict_line_from_block(block, max_conflict_reason_words=max_conflict_reason_words)
    if err:
        problems.append(err)

    end = THINK_CLOSE.search(text)
    tail = text[end.end() :] if end else ""
    abstaining = is_abstain_tail(tail)

    if abstaining and any_support:
        problems.append("abstain_violation_support_present")
    gold_abstain = gold_abstain_from_canon(canon)
    if gold_abstain is not None and abstaining != gold_abstain:
        if abstaining:
            problems.append("abstain_gold_mismatch:false_positive")
        else:
            problems.append("abstain_gold_mismatch:false_negative")

    if not abstaining:
        tail_clean = ZERO_WIDTH.sub("", tail.replace("[[END-OF-ANSWER]]", "")).strip()
        lines = [ln for ln in tail_clean.splitlines() if ln.strip() != ""]
        if not lines:
            problems.append("final_answer_missing")
            return problems
        final = " ".join(lines)
        sents = sentence_split(final)
        max_id = len(exp_docs)
        for m in CITE.finditer(final):
            dnum = int(m.group(1))
            if dnum < 1 or dnum > max_id:
                problems.append("citation_out_of_bounds")
                break
    else:
        tail2 = tail.replace("[[END-OF-ANSWER]]", "").strip()
        if not tail2:
            problems.append("final_answer_missing")

    return problems


def macro_f1_from_conf(conf: List[List[int]]) -> Tuple[float, List[float], List[float], List[float]]:
    n = len(conf)
    precs, recs, f1s = [], [], []
    for c in range(n):
        tp = conf[c][c]
        fp = sum(conf[r][c] for r in range(n)) - tp
        fn = sum(conf[c][r] for r in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    macro = sum(f1s) / n if n > 0 else 0.0
    return macro, precs, recs, f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon_jsonl", required=True)
    ap.add_argument("--gens_jsonl", required=True)
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--max_words_verdict_reason", type=int, default=80)
    ap.add_argument("--max_words_conflict_reason", type=int, default=50,  # v5: 50 words
                    help="Max words allowed in conflict_reason (v5: 50)")
    args = ap.parse_args()

    canon_by_id = {ex["id"]: ex for ex in read_jsonl(args.canon_jsonl)}
    total = 0
    ok = 0
    ok_ignore_abstain_support = 0
    ok_ignore_abstain_evidence = 0
    abstain = 0
    gold_abstain_total = 0
    gold_abstain_correct = 0
    false_abstain = []
    missed_abstain = []
    gold_abstain_with_support = []
    gold_abstain_with_partial = []
    pred_abstain_with_support = []
    pred_abstain_with_partial = []
    problems_log = []
    abstain_support_problem = "abstain_violation_support_present"
    abstain_evidence_problems = {abstain_support_problem}
    citation_eval_count = 0
    citation_pass_count = 0
    citation_coverage_sum = 0.0
    citation_low_ids = []

    # text-mode label accuracy
    conf = [[0] * len(CONFLICT_TYPES) for _ in range(len(CONFLICT_TYPES))]
    label_pairs = 0

    for rec in read_jsonl(args.gens_jsonl):
        total += 1
        cid = rec.get("id")
        if cid not in canon_by_id:
            problems_log.append({"id": cid, "problems": ["id_missing_in_canon"]})
            continue

        canon = canon_by_id[cid]
        probs = validate_example(
            rec,
            canon,
            max_verdict_reason_words=args.max_words_verdict_reason,
            max_conflict_reason_words=args.max_words_conflict_reason,
        )
        if not probs:
            ok += 1
        else:
            problems_log.append({"id": cid, "problems": probs})
        if not [p for p in probs if p != abstain_support_problem]:
            ok_ignore_abstain_support += 1
        if not [p for p in probs if p not in abstain_evidence_problems]:
            ok_ignore_abstain_evidence += 1

        # abstain stats
        raw = rec.get("raw", "")
        end = THINK_CLOSE.search(raw)
        tail = raw[end.end() :] if end else ""
        pred_abstain = is_abstain_tail(tail)
        if pred_abstain:
            abstain += 1
            if canon_has_supporting_doc(canon):
                pred_abstain_with_support.append(cid)
            if canon_has_partial_doc(canon) and not canon_has_supporting_doc(canon):
                pred_abstain_with_partial.append(cid)
        gold_abstain = gold_abstain_from_canon(canon)
        if gold_abstain is not None:
            gold_abstain_total += 1
            if pred_abstain == gold_abstain:
                gold_abstain_correct += 1
            elif pred_abstain:
                false_abstain.append(cid)
            else:
                missed_abstain.append(cid)
            if gold_abstain and canon_has_supporting_doc(canon):
                gold_abstain_with_support.append(cid)
            if gold_abstain and canon_has_partial_doc(canon):
                gold_abstain_with_partial.append(cid)

        if not pred_abstain:
            tail_clean = ZERO_WIDTH.sub("", tail.replace("[[END-OF-ANSWER]]", "")).strip()
            lines = [ln for ln in tail_clean.splitlines() if ln.strip() != ""]
            if lines:
                final = " ".join(lines)
                sents = sentence_split(final)
                if sents:
                    max_id = len(canon.get("retrieved_docs", []))
                    cited = 0
                    for sent in sents:
                        has_valid_cite = False
                        for m in CITE.finditer(sent):
                            dnum = int(m.group(1))
                            if 1 <= dnum <= max_id:
                                has_valid_cite = True
                                break
                        cited += int(has_valid_cite)
                    cov = cited / len(sents)
                    citation_eval_count += 1
                    citation_coverage_sum += cov
                    if cov >= MIN_CITATION_SENTENCE_COVERAGE:
                        citation_pass_count += 1
                    else:
                        citation_low_ids.append({"id": cid, "coverage": round(cov, 4)})

        # text-mode label F1: predicted conflict type vs gold conflict_type (if present)
        block, err = extract_think_block(raw)
        gold = canon.get("conflict_type", None)
        if block and not err and isinstance(gold, str) and gold in TYPE_TO_IDX:
            pred_tuple, _ = conflict_line_from_block(
                block, max_conflict_reason_words=args.max_words_conflict_reason
            )
            if pred_tuple:
                pred = pred_tuple[0]
                if pred in TYPE_TO_IDX:
                    conf[TYPE_TO_IDX[gold]][TYPE_TO_IDX[pred]] += 1
                    label_pairs += 1

    summary: Dict[str, Any] = {
        "total": total,
        "ok_all_checks": ok,
        "ok_rate_pct": round(100 * ok / max(1, total), 1),
        "ok_ignoring_abstain_support_violation": ok_ignore_abstain_support,
        "ok_ignoring_abstain_support_violation_rate_pct": round(
            100 * ok_ignore_abstain_support / max(1, total), 1
        ),
        "ok_ignoring_abstain_evidence_violation": ok_ignore_abstain_evidence,
        "ok_ignoring_abstain_evidence_violation_rate_pct": round(
            100 * ok_ignore_abstain_evidence / max(1, total), 1
        ),
        "abstain_count": abstain,
        "citation_coverage": {
            "definition": (
                "Separate citation-discipline metric over non-abstaining final answers. "
                "Sentence coverage is the fraction of final-answer sentences that contain "
                "at least one in-range citation. This does not affect contract_ok."
            ),
            "threshold": MIN_CITATION_SENTENCE_COVERAGE,
            "evaluated_non_abstain_count": citation_eval_count,
            "pass_count": citation_pass_count,
            "pass_rate_pct": round(100 * citation_pass_count / max(1, citation_eval_count), 1),
            "avg_sentence_coverage": round(citation_coverage_sum / max(1, citation_eval_count), 4),
            "below_threshold_examples": citation_low_ids[:50],
        },
        "abstain_gold": {
            "total_with_gold": gold_abstain_total,
            "correct": gold_abstain_correct,
            "accuracy_pct": round(100 * gold_abstain_correct / max(1, gold_abstain_total), 1),
            "false_abstain_ids": false_abstain,
            "missed_abstain_ids": missed_abstain,
            "gold_abstain_with_supporting_doc_count": len(gold_abstain_with_support),
            "gold_abstain_with_supporting_doc_ids": gold_abstain_with_support,
            "gold_abstain_with_partial_doc_count": len(gold_abstain_with_partial),
            "gold_abstain_with_partial_doc_ids": gold_abstain_with_partial,
        },
        "abstain_diagnostics": {
            "pred_abstain_with_support_count": len(pred_abstain_with_support),
            "pred_abstain_with_support_ids": pred_abstain_with_support,
            "pred_abstain_with_partial_only_count": len(pred_abstain_with_partial),
            "pred_abstain_with_partial_only_ids": pred_abstain_with_partial,
        },
        "problems": problems_log[:50],
    }

    if label_pairs > 0:
        macro, precs, recs, f1s = macro_f1_from_conf(conf)
        summary["label_f1"] = {
            "pairs_evaluated": label_pairs,
            "macro_f1": round(macro, 4),
            "per_class": {
                IDX_TO_TYPE[i]: {
                    "precision": round(precs[i], 4),
                    "recall": round(recs[i], 4),
                    "f1": round(f1s[i], 4),
                }
                for i in range(len(CONFLICT_TYPES))
            },
            "confusion_matrix": {
                IDX_TO_TYPE[i]: {IDX_TO_TYPE[j]: conf[i][j] for j in range(len(CONFLICT_TYPES))}
                for i in range(len(CONFLICT_TYPES))
            },
        }

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
