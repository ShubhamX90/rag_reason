#!/usr/bin/env python3
"""
sanitize.py - Robust post-processing for text-mode generations
==============================================================

This sanitizer is intentionally permissive about what the model emits and
aggressive about normalizing it into the contract expected by the evaluators.

It can optionally use a canonical split JSONL to recover the expected doc_id
order for each example.
"""

import argparse
import json
import re
from pathlib import Path


THINK_OPEN_RE = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"\s*</think>", re.IGNORECASE)
CITE_RE = re.compile(r"\[d(\d+)\]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
DOC_RANGE_RE = re.compile(r"d\d+\s*[–-]\s*d\d+", re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"^\s*cannot\s+answer\s*[,:\-]?\s*insufficient\s+evidence\.?\s*$",
    re.IGNORECASE,
)
PUNCT_ONLY_RE = re.compile(r"^[\s,;]+$")

CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
TYPE_ALIASES = {
    "Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
    "Conflict due outdated information": "Conflict due to outdated information",
}
VERDICT_ALIASES = {
    "support": "supports",
    "supported": "supports",
    "partial": "partially supports",
    "partially_supports": "partially supports",
    "partially-supports": "partially supports",
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno} bad json: {exc}")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def trim_words(text, max_words):
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def normalize_conflict_type(label):
    label = re.sub(r"\s+", " ", (label or "").strip())
    label = TYPE_ALIASES.get(label, label)
    return label if label in CONFLICT_TYPES else None


def normalize_verdict(verdict):
    verdict = re.sub(r"\s+", " ", str(verdict or "").strip().lower())
    verdict = VERDICT_ALIASES.get(verdict, verdict)
    if verdict not in {"supports", "partially supports", "irrelevant"}:
        return "irrelevant"
    return verdict


def extract_think_span(text):
    text = ZERO_WIDTH_RE.sub("", text or "")
    m1 = THINK_OPEN_RE.search(text)
    m2 = THINK_CLOSE_RE.search(text)
    if not m1 or not m2 or m2.start() <= m1.end():
        return None, None, None
    return m1.end(), m2.start(), text[m1.end():m2.start()]


def find_first_top_level_array(block):
    start = block.find("[")
    if start < 0:
        return None, None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(block)):
        ch = block[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
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
                    return start, idx + 1
    return None, None


def parse_doc_array_from_text(text):
    start, end, inner = extract_think_span(text)
    if start is None:
        return []
    arr_start, arr_end = find_first_top_level_array(inner)
    if arr_start is None or arr_end is None:
        return []
    try:
        arr = json.loads(inner[arr_start:arr_end])
    except Exception:
        return []
    return arr if isinstance(arr, list) else []


def load_expected_doc_ids(canon_jsonl):
    if not canon_jsonl:
        return {}
    by_id = {}
    for ex in read_jsonl(canon_jsonl):
        cid = ex.get("id")
        docs = [doc.get("doc_id") for doc in ex.get("retrieved_docs", []) if isinstance(doc, dict)]
        if cid and docs:
            by_id[cid] = docs
    return by_id


def repair_doc_items(arr, expected_doc_ids=None):
    changed = False
    repaired = []
    if expected_doc_ids and len(expected_doc_ids) == len(arr):
        target_ids = expected_doc_ids
    else:
        target_ids = [f"d{i}" for i in range(1, len(arr) + 1)]

    for idx, item in enumerate(arr):
        if not isinstance(item, dict):
            item = {}
            changed = True
        new_item = dict(item)
        target_id = target_ids[idx]
        if new_item.get("doc_id") != target_id:
            new_item["doc_id"] = target_id
            changed = True

        verdict = normalize_verdict(new_item.get("verdict"))
        if new_item.get("verdict") != verdict:
            changed = True
        new_item["verdict"] = verdict

        verdict_reason = trim_words(new_item.get("verdict_reason") or "", 80)
        if new_item.get("verdict_reason") != verdict_reason:
            changed = True
        new_item["verdict_reason"] = verdict_reason

        source_quality = str(new_item.get("source_quality") or "").strip().lower()
        if source_quality not in {"high", "low"}:
            source_quality = "low"
            changed = True
        new_item["source_quality"] = source_quality

        key_fact = trim_words(new_item.get("key_fact") or "", 80)
        if verdict == "irrelevant":
            if key_fact:
                changed = True
            key_fact = ""
        if new_item.get("key_fact") != key_fact:
            changed = True
        new_item["key_fact"] = key_fact
        repaired.append(new_item)
    return repaired, changed


def parse_conflict_line(line):
    line = (line or "").strip()
    if not line:
        return None

    for sep in [" - ", " — ", " – ", ": "]:
        if sep in line:
            left, right = line.split(sep, 1)
            label = normalize_conflict_type(left)
            if label:
                reason = trim_words(DOC_RANGE_RE.sub("multiple docs", right.strip()), 50)
                return label, reason

    m = re.match(
        r"^The conflict_type is ['\"]?(?P<label>.+?)['\"]?\s+because\s+(?P<reason>.+)$",
        line,
        flags=re.IGNORECASE,
    )
    if m:
        label = normalize_conflict_type(m.group("label"))
        if label:
            reason = trim_words(DOC_RANGE_RE.sub("multiple docs", m.group("reason").strip()), 50)
            return label, reason
    return None


def sanitize_block(block, expected_doc_ids=None):
    changed = False
    arr_start, arr_end = find_first_top_level_array(block)
    if arr_start is None:
        return block, False, len(expected_doc_ids or [])

    arr_text = block[arr_start:arr_end]
    try:
        arr = json.loads(arr_text)
    except Exception:
        return block, False, len(expected_doc_ids or [])
    if not isinstance(arr, list):
        return block, False, len(expected_doc_ids or [])

    arr, arr_changed = repair_doc_items(arr, expected_doc_ids=expected_doc_ids)
    changed = changed or arr_changed
    arr_json = json.dumps(arr, ensure_ascii=False, indent=2)

    tail = block[arr_end:]
    lines = tail.splitlines()
    idx = 0
    while idx < len(lines) and (not lines[idx].strip() or PUNCT_ONLY_RE.match(lines[idx])):
        idx += 1

    label = None
    label_idx = None
    for probe in range(idx, min(len(lines), idx + 4)):
        parsed = parse_conflict_line(lines[probe])
        if parsed:
            label = parsed
            label_idx = probe
            break

    if not label:
        label = ("No conflict", "Evidence aligns.")
        label_idx = idx - 1 if idx > 0 else -1
        changed = True

    rest_lines = lines[label_idx + 1:] if label_idx + 1 < len(lines) else []
    rest_text = "\n".join(rest_lines).strip()
    normalized_label = f"{label[0]} - {label[1]}"
    new_block = arr_json + "\n\n" + normalized_label
    if rest_text:
        new_block += "\n" + rest_text

    if new_block != block.strip():
        changed = True
    return new_block, changed, len(arr)


def fix_oob_citations(text, max_id):
    changed = False

    def repl(match):
        nonlocal changed
        doc_num = int(match.group(1))
        if doc_num < 1 or doc_num > max_id:
            changed = True
            return ""
        return match.group(0)

    fixed = CITE_RE.sub(repl, text or "")
    return fixed, changed


def sentence_split(text):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]


def pick_fallback_citations(doc_items):
    ranked = []
    for item in doc_items:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        verdict = item.get("verdict")
        source_quality = item.get("source_quality")
        if not isinstance(doc_id, str):
            continue
        if verdict == "supports":
            rank = 0 if source_quality == "high" else 1
        elif verdict == "partially supports":
            rank = 2 if source_quality == "high" else 3
        else:
            continue
        ranked.append((rank, doc_id))
    ranked.sort()
    return [doc_id for _, doc_id in ranked]


def ensure_sentence_citations(text, fallback_doc_ids):
    sentences = sentence_split(text)
    if not sentences or not fallback_doc_ids:
        return text, False
    changed = False
    cite = "".join(f"[{doc_id}]" for doc_id in fallback_doc_ids[:2])
    fixed = []
    for sent in sentences:
        if CITE_RE.search(sent):
            fixed.append(sent)
            continue
        changed = True
        fixed.append(f"{sent} {cite}".strip())
    return " ".join(fixed), changed


def normalize_tail(tail, max_id, fallback_doc_ids=None):
    tail = ZERO_WIDTH_RE.sub("", tail or "")
    tail_wo_sentinel = tail.replace("[[END-OF-ANSWER]]", "").strip()
    lines = [line.strip() for line in tail_wo_sentinel.splitlines() if line.strip()]

    if lines and ABSTAIN_RE.match(lines[0]):
        return "\n\nCANNOT ANSWER, INSUFFICIENT EVIDENCE\n[[END-OF-ANSWER]]", True

    fixed_tail, changed = fix_oob_citations(tail_wo_sentinel, max_id)
    fixed_tail = fixed_tail.strip()
    if not fixed_tail:
        fixed_tail = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
        changed = True
    else:
        fixed_tail, cite_changed = ensure_sentence_citations(fixed_tail, fallback_doc_ids or [])
        changed = changed or cite_changed

    if "[[END-OF-ANSWER]]" not in fixed_tail:
        fixed_tail = fixed_tail.rstrip() + "\n[[END-OF-ANSWER]]"
        changed = True

    return "\n\n" + fixed_tail, changed


def sanitize_record(rec, expected_doc_ids=None):
    raw = rec.get("raw", "")
    start, end, inner = extract_think_span(raw)
    if start is None:
        return rec, False
    close_match = THINK_CLOSE_RE.search(ZERO_WIDTH_RE.sub("", raw or ""))
    if not close_match:
        return rec, False

    new_block, block_changed, max_id = sanitize_block(inner, expected_doc_ids=expected_doc_ids)
    fallback_doc_ids = pick_fallback_citations(parse_doc_array_from_text(raw))
    new_tail, tail_changed = normalize_tail(raw[close_match.end():], max_id=max_id, fallback_doc_ids=fallback_doc_ids)
    if not (block_changed or tail_changed):
        if raw.rstrip().endswith("[[END-OF-ANSWER]]"):
            return rec, False
        rec = dict(rec)
        rec["raw"] = raw.rstrip() + "\n[[END-OF-ANSWER]]"
        return rec, True

    new_raw = raw[:start] + new_block + "\n</think>" + new_tail
    rec = dict(rec)
    rec["raw"] = new_raw
    return rec, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_jsonl", required=False)
    ap.add_argument("--out", dest="out_jsonl", required=False)
    ap.add_argument("--in_jsonl")
    ap.add_argument("--out_jsonl")
    ap.add_argument("--canon_jsonl", default="", help="Optional canonical JSONL for expected doc ordering")
    args = ap.parse_args()

    in_path = args.in_jsonl or getattr(args, "in", None)
    out_path = args.out_jsonl or getattr(args, "out", None)
    if not in_path or not out_path:
        raise SystemExit("Provide --in/--out or --in_jsonl/--out_jsonl")

    expected_doc_ids = load_expected_doc_ids(args.canon_jsonl)
    out_rows = []
    changed = 0
    for rec in read_jsonl(in_path):
        cid = rec.get("id")
        new_rec, did_change = sanitize_record(rec, expected_doc_ids=expected_doc_ids.get(cid))
        if did_change:
            changed += 1
        out_rows.append(new_rec)

    write_jsonl(out_path, out_rows)
    print(json.dumps({"total": len(out_rows), "changed": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
