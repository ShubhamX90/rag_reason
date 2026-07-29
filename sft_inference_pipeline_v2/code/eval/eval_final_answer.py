#!/usr/bin/env python3
"""
Evaluate final-answer behavior for runs where a trace is not required.

This is intentionally a lightweight local diagnostic, not a replacement for a
semantic/judge-based evaluation. It is useful for true-minimal prompts where
base models and SFT models may answer directly without <think> blocks.
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
SENTINEL = "[[END-OF-ANSWER]]"
CITE = re.compile(r"\[d(\d+)\]", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)

ABSTAIN_CANON = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
ABSTAIN_PATTERNS = [
    re.compile(r"\bcannot\s+answer\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+evidence\b", re.IGNORECASE),
    re.compile(r"\bnot\s+enough\s+(?:information|evidence)\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\s+(?:answer|determine)\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+be\s+determined\b", re.IGNORECASE),
]
BENCHMARK_GOLD_ANSWER_PLACEHOLDERS = {
    "the answer is supported by the retrieved evidence.",
}


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def generation_text(row: Dict[str, Any]) -> str:
    for key in ("raw", "output", "prediction", "response", "generated", "text", "completion"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def strip_trace_and_sentinel(text: str) -> str:
    text = (text or "").replace(SENTINEL, " ").strip()
    matches = list(THINK_CLOSE.finditer(text))
    if matches:
        text = text[matches[-1].end() :].strip()
    return re.sub(r"\s+", " ", text).strip()


def expected_answer(canon: Dict[str, Any]) -> str:
    expected = canon.get("expected_response") or {}
    answer = expected.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    answer = canon.get("gold_answer")
    if isinstance(answer, str):
        return answer.strip()
    return ""


def gold_abstain(canon: Dict[str, Any]) -> Optional[bool]:
    expected = canon.get("expected_response") or {}
    value = expected.get("abstain")
    if isinstance(value, bool):
        return value
    answerable = canon.get("answerable_under_evidence")
    if isinstance(answerable, bool):
        return not answerable
    return None


def gold_answer_usable(canon: Dict[str, Any], answer: str, gold_abs: Optional[bool]) -> bool:
    if gold_abs is not False:
        return False
    if not answer.strip():
        return False
    return answer.strip().lower() not in BENCHMARK_GOLD_ANSWER_PLACEHOLDERS


def predicts_abstain(answer: str) -> bool:
    clean = strip_trace_and_sentinel(answer)
    if not clean:
        return False
    first = clean.splitlines()[0].strip() if "\n" in clean else clean
    if first.upper() == ABSTAIN_CANON:
        return True
    return any(pattern.search(clean) for pattern in ABSTAIN_PATTERNS)


def normalize_for_overlap(text: str) -> List[str]:
    text = CITE.sub(" ", text or "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return WORD.findall(text.lower())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_for_overlap(prediction)
    gold_tokens = normalize_for_overlap(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_for_overlap(prediction)
    gold_tokens = normalize_for_overlap(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def citation_stats(answer: str, valid_doc_count: int) -> Dict[str, Any]:
    citations = [int(x) for x in CITE.findall(answer or "")]
    invalid = [f"d{x}" for x in citations if x < 1 or x > valid_doc_count]
    sentences = sentence_split(answer)
    if not sentences:
        coverage = 0.0
    else:
        cited_sentences = sum(1 for sentence in sentences if CITE.search(sentence))
        coverage = cited_sentences / len(sentences)
    return {
        "citation_count": len(citations),
        "unique_citations": len(set(citations)),
        "invalid_citations": invalid,
        "citation_sentence_coverage": coverage,
    }


def evaluate(canon_jsonl: str, gens_jsonl: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    canon_by_id = {row.get("id"): row for row in read_jsonl(canon_jsonl)}
    rows: List[Dict[str, Any]] = []

    for gen in read_jsonl(gens_jsonl):
        example_id = gen.get("id")
        canon = canon_by_id.get(example_id)
        if not canon:
            rows.append({"id": example_id, "error": "id_not_in_canon"})
            continue

        raw = generation_text(gen)
        final = strip_trace_and_sentinel(raw)
        gold_answer = expected_answer(canon)
        gold_abs = gold_abstain(canon)
        gold_answer_ok = gold_answer_usable(canon, gold_answer, gold_abs)
        pred_abs = predicts_abstain(final)
        valid_doc_count = len(canon.get("retrieved_docs") or [])
        citations = citation_stats(final, valid_doc_count)
        non_abstain_pair = gold_answer_ok and not pred_abs

        row = {
            "id": example_id,
            "gold_abstain": gold_abs,
            "gold_answer_usable": gold_answer_ok,
            "pred_abstain": pred_abs,
            "abstain_match": (gold_abs == pred_abs) if gold_abs is not None else None,
            "word_count": len(normalize_for_overlap(final)),
            "has_think": "<think>" in (raw or "").lower() and "</think>" in (raw or "").lower(),
            "has_sentinel": SENTINEL in (raw or ""),
            "token_f1": token_f1(final, gold_answer) if non_abstain_pair else None,
            "rouge_l_f1": rouge_l_f1(final, gold_answer) if non_abstain_pair else None,
            **citations,
            "gold_answer": gold_answer,
            "final_answer": final,
        }
        rows.append(row)

    valid_rows = [row for row in rows if not row.get("error")]
    total = len(valid_rows)
    abstain_labeled = [row for row in valid_rows if row["gold_abstain"] is not None]
    abstain_matches = [row for row in abstain_labeled if row["abstain_match"]]
    gold_non_abstain = [row for row in valid_rows if row["gold_abstain"] is False]
    gold_non_abstain_usable = [row for row in valid_rows if row["gold_answer_usable"]]
    final_scored = [row for row in valid_rows if row["token_f1"] is not None]
    invalid_citation_rows = [row for row in valid_rows if row["invalid_citations"]]

    def avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    false_positive_abstain = [
        row["id"] for row in valid_rows if row["gold_abstain"] is False and row["pred_abstain"]
    ]
    false_negative_abstain = [
        row["id"] for row in valid_rows if row["gold_abstain"] is True and not row["pred_abstain"]
    ]
    true_positive_abstain = [
        row["id"] for row in valid_rows if row["gold_abstain"] is True and row["pred_abstain"]
    ]
    true_negative_abstain = [
        row["id"] for row in valid_rows if row["gold_abstain"] is False and not row["pred_abstain"]
    ]
    low_overlap = sorted(
        [row for row in final_scored if row["token_f1"] is not None],
        key=lambda row: row["token_f1"],
    )[:12]

    tp = len(true_positive_abstain)
    fp = len(false_positive_abstain)
    fn = len(false_negative_abstain)
    tn = len(true_negative_abstain)
    refusal_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    refusal_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    refusal_f1 = (
        2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
        if (refusal_precision + refusal_recall) > 0
        else 0.0
    )
    non_refusal_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    report = {
        "total": total,
        "errors": Counter(row.get("error") for row in rows if row.get("error")),
        "trace_presence": {
            "think_count": sum(1 for row in valid_rows if row["has_think"]),
            "sentinel_count": sum(1 for row in valid_rows if row["has_sentinel"]),
        },
        "abstain": {
            "gold_abstain_count": sum(1 for row in valid_rows if row["gold_abstain"] is True),
            "pred_abstain_count": sum(1 for row in valid_rows if row["pred_abstain"]),
            "accuracy_pct": round(100 * len(abstain_matches) / len(abstain_labeled), 2)
            if abstain_labeled
            else 0.0,
            "confusion": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            },
            "refusal_metrics": {
                "precision": round(refusal_precision, 4),
                "recall": round(refusal_recall, 4),
                "f1": round(refusal_f1, 4),
                "specificity": round(non_refusal_specificity, 4),
            },
            "false_positive_ids": false_positive_abstain,
            "false_negative_ids": false_negative_abstain,
        },
        "citations": {
            "avg_citation_count": avg([row["citation_count"] for row in valid_rows]),
            "avg_unique_citations": avg([row["unique_citations"] for row in valid_rows]),
            "avg_sentence_coverage": avg([row["citation_sentence_coverage"] for row in valid_rows]),
            "rows_with_invalid_citations": len(invalid_citation_rows),
        },
        "lexical_overlap_non_abstain": {
            "scored_pairs": len(final_scored),
            "gold_non_abstain_count": len(gold_non_abstain),
            "gold_non_abstain_usable_answer_count": len(gold_non_abstain_usable),
            "avg_token_f1": avg([row["token_f1"] for row in final_scored if row["token_f1"] is not None]),
            "avg_rouge_l_f1": avg([row["rouge_l_f1"] for row in final_scored if row["rouge_l_f1"] is not None]),
            "low_overlap_sample": [
                {
                    "id": row["id"],
                    "token_f1": round(row["token_f1"], 4),
                    "gold_answer": row["gold_answer"],
                    "final_answer": row["final_answer"][:500],
                }
                for row in low_overlap
            ],
        },
        "notes": {
            "description": "Final-answer proxy metrics for prompt settings where traces may not be required.",
            "gold_abstain_source": "expected_response.abstain when present, otherwise answerable_under_evidence",
            "gold_answer_filter": "Rows with empty or placeholder gold_answer are excluded from lexical overlap scoring.",
            "warning": "Lexical overlap is not a semantic quality judge. Use it for triage only.",
        },
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canon_jsonl", required=True)
    parser.add_argument("--gens_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--per_id_jsonl")
    args = parser.parse_args()

    report, rows = evaluate(args.canon_jsonl, args.gens_jsonl)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.per_id_jsonl:
        per_id_path = Path(args.per_id_jsonl)
        per_id_path.parent.mkdir(parents=True, exist_ok=True)
        with per_id_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
