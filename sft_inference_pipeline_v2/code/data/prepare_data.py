#!/usr/bin/env python3
"""
prepare_data.py - Split-aware data preparation for SFT Pipeline v2
==================================================================

This version assumes train/val splits already exist and treats them as the
source of truth. It:
  1. Audits and normalizes the provided split files
  2. Writes canonical cleaned JSONL files
  3. Builds chat-message JSONL files for:
       - e2e training/inference
       - oracle_conflict inference
       - oracle_notes inference
       - oracle_both inference

Training policy:
  - Train only on e2e messages
  - Fine-tune stagewise and monolithic separately
  - Oracle variants are generated for inference / diagnostic use
  - Assistant target style is configurable so experiments can cleanly ablate
    target formatting without changing the rest of the pipeline

Usage:
  python3 code/data/prepare_data.py \
    --stagewise_train_jsonl data/splits/stagewise_multi/train/stage3_final.jsonl \
    --stagewise_val_jsonl data/splits/stagewise_multi/val/stage3_final.jsonl \
    --monolithic_train_jsonl data/splits/monolithic_multi/train/monolithic_final.jsonl \
    --monolithic_val_jsonl data/splits/monolithic_multi/val/monolithic_final.jsonl \
    --out_dir data \
    --prompts_dir prompts
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SENTINEL = "[[END-OF-ANSWER]]"
EM_DASH = " - "
META_ARTIFACT_RE = re.compile(
    r"\b(?:gold_answer|gold answer|provided[_ ]gold_answer|refusal-required sample|"
    r"refusal required sample|sample because)\b|could not parse json|fallback:",
    re.IGNORECASE,
)
CANON_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
CANON_TYPES_SET = set(CANON_TYPES)
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
ALLOWED_VERDICTS = {"supports", "partially supports", "irrelevant"}
ALLOWED_SOURCE_QUALITY = {"high", "low"}
DOC_ID_RE = re.compile(r"\bd(\d+)\b")
DOC_RANGE_RE = re.compile(r"\bd(\d+)\s*[–-]\s*d(\d+)\b")
LABELISH_RE = re.compile(
    r"^\s*(?P<label>No conflict|Complementary information|"
    r"Conflicting opinions (?:and|or) research outcomes|"
    r"Conflict due(?: to)? outdated information|"
    r"Conflict due to misinformation)\s*(?:[:\-–—]\s*|\s+because\s+)(?P<reason>.+?)\s*$",
    re.IGNORECASE,
)
THINK_OPEN_RE = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"\s*</think>", re.IGNORECASE)
PROMPT_FILE_MAPS = {
    "default": {
        "e2e": ("system_e2e.txt", "user_e2e.txt"),
        "oracle_conflict": ("system_oracle.txt", "user_oracle.txt"),
        "oracle_notes": ("system_oracle_notes.txt", "user_oracle_notes.txt"),
        "oracle_both": ("system_oracle_both.txt", "user_oracle_both.txt"),
    },
    "minimal": {
        "e2e": ("system_e2e_minimal.txt", "user_e2e_minimal.txt"),
        "oracle_conflict": ("system_oracle_minimal.txt", "user_oracle_minimal.txt"),
        "oracle_notes": ("system_oracle_notes_minimal.txt", "user_oracle_notes_minimal.txt"),
        "oracle_both": ("system_oracle_both_minimal.txt", "user_oracle_both_minimal.txt"),
    },
    "legacy_text_contract": {
        "e2e": ("system_e2e_legacy_text_contract.txt", "user_e2e_legacy_text_contract.txt"),
        "oracle_conflict": ("system_oracle_legacy_text_contract.txt", "user_oracle_legacy_text_contract.txt"),
        "oracle_notes": ("system_oracle_notes_legacy_text_contract.txt", "user_oracle_notes_legacy_text_contract.txt"),
        "oracle_both": ("system_oracle_both_legacy_text_contract.txt", "user_oracle_both_legacy_text_contract.txt"),
    },
    "runtime": {
        "e2e": ("system_e2e_runtime.txt", "user_e2e_runtime.txt"),
        "oracle_conflict": ("system_oracle_runtime.txt", "user_oracle_runtime.txt"),
        "oracle_notes": ("system_oracle_notes_runtime.txt", "user_oracle_notes_runtime.txt"),
        "oracle_both": ("system_oracle_both_runtime.txt", "user_oracle_both_runtime.txt"),
    },
    "final_only": {
        "e2e": ("system_e2e_final_only.txt", "user_e2e_final_only.txt"),
        "oracle_conflict": ("system_oracle_final_only.txt", "user_oracle_final_only.txt"),
        "oracle_notes": ("system_oracle_notes_final_only.txt", "user_oracle_notes_final_only.txt"),
        "oracle_both": ("system_oracle_both_final_only.txt", "user_oracle_both_final_only.txt"),
    },
}
TRAIN_TASKS = ("e2e_trace", "doc_verdict", "conflict_type", "answer_only")
TASK_USER_PREFIX = {
    "doc_verdict": (
        "Training subtask: produce ONLY the Stage 1 evidence assessment inside a "
        "<think>...</think> block, then end with [[END-OF-ANSWER]]."
    ),
    "conflict_type": (
        "Training subtask: produce ONLY the Stage 2 conflict assessment inside a "
        "<think>...</think> block, then end with [[END-OF-ANSWER]]."
    ),
    "answer_only": (
        "Training subtask: produce ONLY the final grounded answer. Do not include a "
        "<think> block. End with [[END-OF-ANSWER]]."
    ),
}


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


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def message_tag_suffix(message_tag):
    return f"_{message_tag}" if message_tag else ""


def task_suffix(task):
    return "" if task == "e2e_trace" else f"_{task}"


def message_filename(split_name, mode, message_tag="", task="e2e_trace"):
    return f"{split_name}_{mode}{task_suffix(task)}{message_tag_suffix(message_tag)}_messages.jsonl"


def trim_words(text, max_words):
    words = (text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def normalize_conflict_type(label):
    label = re.sub(r"\s+", " ", (label or "").strip())
    return TYPE_ALIASES.get(label, label)


def normalize_verdict(verdict):
    verdict = re.sub(r"\s+", " ", str(verdict or "").strip().lower())
    verdict = VERDICT_ALIASES.get(verdict, verdict)
    return verdict if verdict in ALLOWED_VERDICTS else "irrelevant"


def normalize_source_quality(source_quality):
    sq = str(source_quality or "").strip().lower()
    return sq if sq in ALLOWED_SOURCE_QUALITY else "low"


def remap_doc_ids_in_text(text, mapping):
    if not text or not mapping:
        return text or ""
    out = str(text)
    for old_id in sorted(mapping, key=len, reverse=True):
        new_id = mapping[old_id]
        out = re.sub(rf"\b{re.escape(old_id)}\b", new_id, out)
    return out


def sanitize_doc_ranges(text):
    return DOC_RANGE_RE.sub("multiple docs", text or "")


def extract_think_inner(think_text):
    think_text = think_text or ""
    m1 = THINK_OPEN_RE.search(think_text)
    m2 = THINK_CLOSE_RE.search(think_text)
    if m1 and m2 and m2.start() > m1.end():
        return think_text[m1.end():m2.start()]
    return think_text


def find_first_top_level_array(text):
    start = text.find("[")
    if start < 0:
        return None, None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
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


def parse_label_line(line):
    if not line:
        return None, None
    m = LABELISH_RE.match(line.strip())
    if not m:
        return None, None
    label = normalize_conflict_type(m.group("label"))
    reason = m.group("reason").strip()
    if label not in CANON_TYPES_SET:
        return None, None
    return label, trim_words(sanitize_doc_ranges(reason), 50)


def heuristic_conflict_reasoning(doc_array):
    supports = [d["doc_id"] for d in doc_array if d["verdict"] == "supports"]
    partial = [d["doc_id"] for d in doc_array if d["verdict"] == "partially supports"]
    irrelevant = [d["doc_id"] for d in doc_array if d["verdict"] == "irrelevant"]
    supporting = supports + partial
    if supporting and irrelevant:
        return (
            f"Docs {', '.join(supporting)} contribute evidence, while docs "
            f"{', '.join(irrelevant)} do not help answer the query."
        )
    if supporting:
        return f"Docs {', '.join(supporting)} provide the evidence used for the final answer."
    return "All retrieved documents are irrelevant or insufficient for answering the query."


def doc_ids_by_verdict(doc_array, verdicts):
    wanted = set(verdicts)
    return [d["doc_id"] for d in doc_array if d.get("verdict") in wanted]


def default_conflict_reason(label):
    defaults = {
        "No conflict": "Non-irrelevant docs agree on the same answer.",
        "Complementary information": "Docs contribute different non-contradictory facets of the answer.",
        "Conflicting opinions or research outcomes": "Docs make incompatible claims within the same scope or time window.",
        "Conflict due to outdated information": "Older and newer snapshots disagree, and newer evidence supersedes older claims.",
        "Conflict due to misinformation": "A weaker claim conflicts with stronger evidence and is likely false or misleading.",
    }
    return defaults.get(label, "Evidence aligns.")


def canonical_conflict_reasoning(label, doc_array):
    support_ids = doc_ids_by_verdict(doc_array, {"supports"})
    partial_ids = doc_ids_by_verdict(doc_array, {"partially supports"})
    relevant_ids = support_ids + partial_ids
    irrelevant_ids = doc_ids_by_verdict(doc_array, {"irrelevant"})
    lead = ", ".join(relevant_ids[:4]) if relevant_ids else "no relevant docs"
    if label == "No conflict":
        return f"Docs {lead} support the same core answer, with only detail-level differences."
    if label == "Complementary information":
        return f"Docs {lead} contribute different scopes or facets that fit together without contradiction."
    if label == "Conflicting opinions or research outcomes":
        return f"Docs {lead} contain incompatible claims within the same scope or time window."
    if label == "Conflict due to outdated information":
        return f"Docs {lead} reflect different temporal snapshots, so older claims conflict with newer ones."
    if label == "Conflict due to misinformation":
        return f"Docs {lead} conflict, and the weaker claim is outweighed by stronger evidence."
    if irrelevant_ids:
        return f"Docs {', '.join(irrelevant_ids[:4])} do not provide enough relevant evidence to answer the query."
    return heuristic_conflict_reasoning(doc_array)


def heuristic_explanation(ex, doc_array):
    expected = ex.get("expected_response") or {}
    abstain = bool(expected.get("abstain"))
    evidence = [d for d in (expected.get("evidence") or []) if isinstance(d, str)]
    if abstain:
        return (
            "The retrieved evidence does not justify a reliable answer. "
            "The final response therefore abstains."
        )
    if evidence:
        cited = ", ".join(evidence)
        return (
            f"Docs {cited} provide the strongest support available in the retrieved set. "
            "The final answer should synthesize only those grounded snippets."
        )
    supporting = [d["doc_id"] for d in doc_array if d["verdict"] != "irrelevant"]
    if supporting:
        cited = ", ".join(supporting)
        return (
            f"Docs {cited} contain the best available evidence. "
            "The final answer should stay within those retrieved snippets."
        )
    return (
        "The retrieved evidence is insufficient to support a grounded answer. "
        "The final response should abstain."
    )


def clean_reasoning_text(text):
    text = (text or "").replace("<think>", " ").replace("</think>", " ")
    text = text.replace(SENTINEL, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_meta_artifacts(text):
    return bool(META_ARTIFACT_RE.search(text or ""))


def clean_target_fragment(text, max_words):
    text = trim_words(clean_reasoning_text(sanitize_doc_ranges(text or "")), max_words)
    return "" if has_meta_artifacts(text) else text


def doc_sentence_with_citation(text, doc_id):
    text = clean_reasoning_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\[d\d+\]\s*", " ", text).strip()
    text = text.rstrip(" .")
    if not text:
        return ""
    return f"{text} [{doc_id}]."


def default_doc_verdict_reason(verdict, doc):
    snippet = trim_words(clean_reasoning_text((doc or {}).get("snippet") or ""), 22)
    if snippet and verdict != "irrelevant":
        return f"The snippet contains query-relevant evidence: {snippet}"
    if verdict == "irrelevant":
        return "The snippet does not provide query-specific support."
    return "The snippet provides some query-relevant support."


def build_doc_array(ex):
    doc_order = [doc.get("doc_id", "") for doc in ex.get("retrieved_docs", [])]
    docs_by_id = {doc.get("doc_id", ""): doc for doc in ex.get("retrieved_docs", [])}
    notes_by_id = {}
    for note in ex.get("per_doc_notes") or []:
        if isinstance(note, dict):
            notes_by_id[note.get("doc_id", "")] = note
    arr = []
    for doc_id in doc_order:
        note = notes_by_id.get(doc_id, {})
        verdict = normalize_verdict(note.get("verdict"))
        verdict_reason = clean_target_fragment(note.get("verdict_reason") or "", 80)
        if not verdict_reason:
            verdict_reason = default_doc_verdict_reason(verdict, docs_by_id.get(doc_id, {}))
        key_fact = clean_target_fragment(note.get("key_fact") or "", 80)
        if verdict == "irrelevant":
            key_fact = ""
        arr.append({
            "doc_id": doc_id,
            "verdict": verdict,
            "verdict_reason": trim_words(verdict_reason, 80),
            "key_fact": trim_words(key_fact, 80) if key_fact else "",
            "source_quality": normalize_source_quality(note.get("source_quality")),
        })
    return arr


def parse_reasoning_from_think(think_text, fallback_type, fallback_reason):
    inner = extract_think_inner(think_text)
    arr_start, arr_end = find_first_top_level_array(inner)
    if arr_end is None:
        return (
            fallback_type,
            trim_words(clean_reasoning_text(sanitize_doc_ranges(fallback_reason)), 50),
            "",
            "",
        )
    tail = inner[arr_end:].strip()
    if not tail:
        return (
            fallback_type,
            trim_words(clean_reasoning_text(sanitize_doc_ranges(fallback_reason)), 50),
            "",
            "",
        )

    lines = [ln.rstrip() for ln in tail.splitlines()]
    non_empty = [(i, ln.strip()) for i, ln in enumerate(lines) if ln.strip()]
    label = fallback_type
    reason = trim_words(sanitize_doc_ranges(fallback_reason), 50)
    label_idx = None
    for idx, line in non_empty[:3]:
        parsed_label, parsed_reason = parse_label_line(line)
        if not parsed_label:
            continue
        label = parsed_label
        reason = clean_reasoning_text(parsed_reason) or reason
        label_idx = idx
        break

    if label_idx is not None:
        remaining = "\n".join(lines[label_idx + 1:]).strip()
    else:
        remaining = "\n".join(lines).strip()

    parts = [part.strip() for part in re.split(r"\n\s*\n", remaining) if part.strip()]
    conflict_reasoning = trim_words(clean_reasoning_text(parts[0]), 45) if parts else ""
    explanation = trim_words(clean_reasoning_text(parts[1]), 60) if len(parts) > 1 else ""
    return label, reason, conflict_reasoning, explanation


def ensure_answer_citations(answer, evidence_ids):
    answer = (answer or "").strip()
    if not answer or not evidence_ids:
        return answer
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return answer
    cite = f"[{evidence_ids[0]}]"
    fixed = []
    for sent in sentences:
        if re.search(r"\[d\d+\]", sent):
            fixed.append(sent)
        else:
            fixed.append(f"{sent} {cite}")
    return " ".join(fixed)


def build_final_answer(ex):
    expected = ex.get("expected_response") or {}
    abstain = bool(expected.get("abstain"))
    if abstain:
        return "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
    evidence_ids = [doc_id for doc_id in (expected.get("evidence") or []) if isinstance(doc_id, str)]
    answer_text = clean_expected_answer_text(expected.get("answer") or "")
    if has_meta_artifacts(answer_text):
        answer_text = fallback_final_answer(ex, evidence_ids)
    return ensure_answer_citations(answer_text, evidence_ids)


def fallback_final_answer(ex, evidence_ids):
    doc_array = build_doc_array(ex)
    notes_by_id = {item["doc_id"]: item for item in doc_array}
    sentences = []
    for doc_id in evidence_ids:
        note = notes_by_id.get(doc_id) or {}
        fact = note.get("key_fact") or note.get("verdict_reason") or ""
        fact = clean_target_fragment(fact, 32)
        sent = doc_sentence_with_citation(fact, doc_id)
        if sent:
            sentences.append(sent)
        if len(sentences) >= 2:
            break
    if sentences:
        return " ".join(sentences)

    docs_by_id = {doc.get("doc_id", ""): doc for doc in ex.get("retrieved_docs") or []}
    for doc_id in evidence_ids:
        snippet = trim_words((docs_by_id.get(doc_id) or {}).get("snippet") or "", 32)
        sent = doc_sentence_with_citation(snippet, doc_id)
        if sent:
            return sent
    return "CANNOT ANSWER, INSUFFICIENT EVIDENCE"


def build_reasoning_parts(ex, doc_array, target_style):
    conflict_type = normalize_conflict_type(ex.get("conflict_type"))
    if conflict_type not in CANON_TYPES_SET:
        conflict_type = "No conflict"
    conflict_reason = ex.get("conflict_reason") or ""
    label, parsed_reason, parsed_conflict_reasoning, parsed_explanation = parse_reasoning_from_think(
        ex.get("think") or "",
        conflict_type,
        conflict_reason,
    )
    if label not in CANON_TYPES_SET:
        label = conflict_type
    reason = parsed_reason or trim_words(clean_reasoning_text(sanitize_doc_ranges(conflict_reason)), 50)
    if not reason:
        reason = default_conflict_reason(label)
    if target_style == "canonical":
        conflict_reasoning = canonical_conflict_reasoning(label, doc_array)
        explanation = heuristic_explanation(ex, doc_array)
    else:
        if parsed_conflict_reasoning and not has_meta_artifacts(parsed_conflict_reasoning):
            conflict_reasoning = parsed_conflict_reasoning
        else:
            conflict_reasoning = heuristic_conflict_reasoning(doc_array)
        conflict_reasoning = trim_words(clean_reasoning_text(conflict_reasoning), 45)
        if parsed_explanation and not has_meta_artifacts(parsed_explanation):
            explanation = parsed_explanation
        else:
            explanation = heuristic_explanation(ex, doc_array)
        explanation = trim_words(clean_reasoning_text(explanation), 60)
    return label, reason, conflict_reasoning, explanation


def build_stage1_lines(doc_array):
    lines = []
    for item in doc_array:
        doc_id = item.get("doc_id") or ""
        verdict = item.get("verdict") or "irrelevant"
        verdict_reason = trim_words(item.get("verdict_reason") or "", 55)
        key_fact = trim_words(item.get("key_fact") or "", 45)
        source_quality = item.get("source_quality") or "low"
        parts = [verdict_reason or "No useful query-specific evidence is present."]
        if key_fact:
            parts.append(f"Key fact: {key_fact}")
        parts.append(f"Source quality: {source_quality}.")
        lines.append(f"- {doc_id}: {verdict} - {' '.join(parts)}")
    return lines


def build_trace_text_assistant(ex, task="e2e_trace", target_style="trace_text"):
    doc_array = build_doc_array(ex)
    label, reason, conflict_reasoning, explanation = build_reasoning_parts(ex, doc_array, target_style)
    final = build_final_answer(ex)
    stage1 = "\n".join(build_stage1_lines(doc_array))

    if task == "doc_verdict":
        return (
            "<think>\n"
            "Stage 1 - Evidence assessment:\n"
            f"{stage1}\n"
            "</think>\n"
            f"{SENTINEL}"
        )

    if task == "conflict_type":
        return (
            "<think>\n"
            "Stage 2 - Conflict assessment:\n"
            f"Conflict type: {label}\n"
            f"Reason: {reason}\n"
            f"Evidence pattern: {conflict_reasoning}\n"
            "</think>\n"
            f"{SENTINEL}"
        )

    if task == "answer_only":
        return f"{final}\n{SENTINEL}"

    return (
        "<think>\n"
        "Stage 1 - Evidence assessment:\n"
        f"{stage1}\n\n"
        "Stage 2 - Conflict assessment:\n"
        f"Conflict type: {label}\n"
        f"Reason: {reason}\n"
        f"Evidence pattern: {conflict_reasoning}\n\n"
        "Stage 3 - Answer plan:\n"
        f"{explanation}\n"
        "</think>\n\n"
        f"{final}\n"
        f"{SENTINEL}"
    )


def clean_expected_answer_text(answer):
    answer = (answer or "").strip()
    if not answer:
        return ""
    m_close = THINK_CLOSE_RE.search(answer)
    if m_close:
        answer = answer[m_close.end():].strip()
    answer = answer.replace(SENTINEL, " ").strip()
    if THINK_OPEN_RE.search(answer):
        answer = THINK_OPEN_RE.sub(" ", answer)
    answer = THINK_CLOSE_RE.sub(" ", answer)
    return re.sub(r"\s+", " ", answer).strip()


def build_assistant_text(ex, target_style="parsed", task="e2e_trace"):
    if task != "e2e_trace" or target_style == "trace_text":
        return build_trace_text_assistant(ex, task=task, target_style=target_style)

    doc_array = build_doc_array(ex)
    label, reason, conflict_reasoning, explanation = build_reasoning_parts(ex, doc_array, target_style)
    final = build_final_answer(ex)

    arr_json = json.dumps(doc_array, ensure_ascii=False, indent=2)
    assistant = (
        "<think>\n"
        f"{arr_json}\n\n"
        f"{label}{EM_DASH}{reason}\n"
        f"{conflict_reasoning}\n\n"
        f"{explanation}\n"
        "</think>\n\n"
        f"{final}\n"
        f"{SENTINEL}"
    )
    return assistant


def build_user_payload(ex, mode):
    payload = {
        "query": ex.get("query", ""),
        "retrieved_docs_json": json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
    }
    if mode in {"oracle_conflict", "oracle_both"}:
        payload["conflict_type"] = normalize_conflict_type(ex.get("conflict_type", ""))
    if mode in {"oracle_notes", "oracle_both"}:
        payload["per_doc_notes_json"] = json.dumps(build_doc_array(ex), ensure_ascii=False, indent=2)
    return payload


def build_messages(ex, mode, system_prompt, user_template, assistant_target_style, task="e2e_trace"):
    user_payload = build_user_payload(ex, mode)
    user_msg = user_template.format(**user_payload)
    if task in TASK_USER_PREFIX:
        user_msg = f"{TASK_USER_PREFIX[task]}\n\n{user_msg}"
    assistant = build_assistant_text(ex, target_style=assistant_target_style, task=task)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant},
    ]


def validate_messages(messages):
    if not isinstance(messages, list) or len(messages) != 3:
        return False
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        return False
    assistant = messages[2].get("content", "")
    if not assistant.rstrip().endswith(SENTINEL):
        return False
    think_opens = assistant.count("<think>")
    think_closes = assistant.count("</think>")
    if think_opens == 0 and think_closes == 0:
        return bool(assistant.replace(SENTINEL, "").strip())
    return think_opens == 1 and think_closes == 1 and assistant.find("<think>") < assistant.find("</think>")


def normalize_example(ex):
    ex = json.loads(json.dumps(ex, ensure_ascii=False))
    findings = Counter()

    old_doc_ids = [doc.get("doc_id", "") for doc in ex.get("retrieved_docs") or []]
    new_doc_ids = [f"d{i}" for i in range(1, len(old_doc_ids) + 1)]
    mapping = {old: new for old, new in zip(old_doc_ids, new_doc_ids) if old and old != new}
    if mapping:
        findings["doc_ids_reindexed"] += 1

    for idx, doc in enumerate(ex.get("retrieved_docs") or [], 1):
        doc["doc_id"] = f"d{idx}"

    notes = []
    notes_by_old = {}
    for note in ex.get("per_doc_notes") or []:
        if isinstance(note, dict):
            notes_by_old[note.get("doc_id", "")] = note
    for old_doc_id, new_doc_id in zip(old_doc_ids, new_doc_ids):
        note = json.loads(json.dumps(notes_by_old.get(old_doc_id, {"doc_id": old_doc_id}), ensure_ascii=False))
        note["doc_id"] = new_doc_id
        note["verdict"] = normalize_verdict(note.get("verdict"))
        note["verdict_reason"] = trim_words(note.get("verdict_reason") or "", 80)
        note["source_quality"] = normalize_source_quality(note.get("source_quality"))
        if note["verdict"] == "irrelevant":
            note["key_fact"] = ""
        else:
            note["key_fact"] = trim_words(note.get("key_fact") or "", 80)
        notes.append(note)
    ex["per_doc_notes"] = notes

    ex["conflict_type"] = normalize_conflict_type(ex.get("conflict_type"))
    if ex["conflict_type"] not in CANON_TYPES_SET:
        findings["unknown_conflict_type"] += 1
        ex["conflict_type"] = "No conflict"

    expected = ex.get("expected_response") or {}
    if mapping:
        expected["answer"] = remap_doc_ids_in_text(expected.get("answer") or "", mapping)
        expected["abstain_reason"] = remap_doc_ids_in_text(expected.get("abstain_reason") or "", mapping)
        expected["evidence"] = [mapping.get(doc_id, doc_id) for doc_id in (expected.get("evidence") or [])]
        ex["conflict_reason"] = remap_doc_ids_in_text(ex.get("conflict_reason") or "", mapping)
        ex["gold_answer"] = remap_doc_ids_in_text(ex.get("gold_answer") or "", mapping)
        ex["think"] = remap_doc_ids_in_text(ex.get("think") or "", mapping)

    expected_evidence = []
    valid_doc_ids = set(new_doc_ids)
    for doc_id in expected.get("evidence") or []:
        if isinstance(doc_id, str) and doc_id in valid_doc_ids:
            expected_evidence.append(doc_id)
        else:
            findings["evidence_doc_id_dropped"] += 1
    expected["evidence"] = expected_evidence

    abstain = bool(expected.get("abstain"))
    expected["abstain"] = abstain
    if abstain:
        expected["answer"] = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
    else:
        expected["answer"] = (expected.get("answer") or "").strip()
    ex["expected_response"] = expected

    answerable = ex.get("answerable_under_evidence")
    desired_answerable = not abstain
    if answerable is None or bool(answerable) != desired_answerable:
        findings["answerable_under_evidence_normalized"] += 1
        ex["answerable_under_evidence"] = desired_answerable

    ex["conflict_reason"] = trim_words(sanitize_doc_ranges(ex.get("conflict_reason") or ""), 50)
    return ex, findings


def audit_examples(split_name, examples):
    counters = Counter()
    conflict_types = Counter()
    docs_per_example = []
    abstain_counter = Counter()
    examples_with_findings = []

    for ex in examples:
        conflict_types[ex.get("conflict_type")] += 1
        docs_per_example.append(len(ex.get("retrieved_docs") or []))
        abstain_counter[(ex.get("expected_response") or {}).get("abstain")] += 1

        doc_ids = [doc.get("doc_id") for doc in ex.get("retrieved_docs") or []]
        note_ids = [note.get("doc_id") for note in ex.get("per_doc_notes") or []]
        if doc_ids != [f"d{i}" for i in range(1, len(doc_ids) + 1)]:
            counters["noncanonical_doc_ids"] += 1
            examples_with_findings.append((ex.get("id"), "noncanonical_doc_ids"))
        if note_ids != doc_ids:
            counters["note_doc_alignment_issues"] += 1
            examples_with_findings.append((ex.get("id"), "note_doc_alignment_issues"))
        if ex.get("conflict_type") not in CANON_TYPES_SET:
            counters["noncanonical_conflict_types"] += 1
            examples_with_findings.append((ex.get("id"), "noncanonical_conflict_types"))
        if bool((ex.get("expected_response") or {}).get("abstain")) == bool(ex.get("answerable_under_evidence")):
            counters["abstain_answerable_same_value"] += 1
            examples_with_findings.append((ex.get("id"), "abstain_answerable_same_value"))

    print(f"[Audit] {split_name}: rows={len(examples)}")
    if docs_per_example:
        print(
            f"[Audit] {split_name}: docs/example min={min(docs_per_example)} "
            f"max={max(docs_per_example)} avg={sum(docs_per_example)/len(docs_per_example):.2f}"
        )
    print(f"[Audit] {split_name}: conflict_types={dict(sorted(conflict_types.items()))}")
    print(f"[Audit] {split_name}: abstain={dict(abstain_counter)}")
    if counters:
        print(f"[Audit] {split_name}: findings={dict(counters)}")
        print(f"[Audit] {split_name}: sample_ids={examples_with_findings[:10]}")


def load_and_normalize(path, split_name):
    normalized = []
    findings = Counter()
    for ex in read_jsonl(path):
        new_ex, row_findings = normalize_example(ex)
        normalized.append(new_ex)
        findings.update(row_findings)
    audit_examples(split_name, normalized)
    if findings:
        print(f"[Normalize] {split_name}: applied={dict(findings)}")
    return normalized


def prompt_paths(prompts_dir, prompt_profile="default"):
    prompts_dir = Path(prompts_dir)
    file_map = PROMPT_FILE_MAPS[prompt_profile]
    prompts = {}
    for mode, (system_name, user_name) in file_map.items():
        prompts[mode] = (
            load_text(prompts_dir / system_name),
            load_text(prompts_dir / user_name),
        )
    return prompts


def build_message_files(split_name, examples, modes, out_dir, prompts, assistant_target_style, message_tag="", tasks=None):
    messages_dir = Path(out_dir) / "messages"
    tasks = tasks or ["e2e_trace"]
    written = []
    for mode in modes:
        system_prompt, user_template = prompts[mode]
        for task in tasks:
            rows = []
            dropped = 0
            for ex in examples:
                try:
                    messages = build_messages(
                        ex,
                        mode,
                        system_prompt,
                        user_template,
                        assistant_target_style=assistant_target_style,
                        task=task,
                    )
                except Exception as exc:
                    print(f"[WARN] {split_name}/{mode}/{task} {ex.get('id')}: {exc}")
                    dropped += 1
                    continue
                if not validate_messages(messages):
                    dropped += 1
                    continue
                rows.append({"id": ex.get("id"), "task": task, "messages": messages})

            out_path = messages_dir / message_filename(split_name, mode, message_tag, task=task)
            write_jsonl(out_path, rows)
            written.append(out_path.name)
            print(f"[Messages] {split_name}/{mode}/{task}: kept={len(rows)}, dropped={dropped} -> {out_path}")
    return written


def combine_message_files(out_dir, destination_name, source_names):
    messages_dir = Path(out_dir) / "messages"
    combined = []
    for source_name in source_names:
        source_path = messages_dir / source_name
        if not source_path.exists():
            continue
        combined.extend(list(read_jsonl(source_path)))
    if not combined:
        return
    out_path = messages_dir / destination_name
    write_jsonl(out_path, combined)
    print(f"[Messages] combined: kept={len(combined)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Prepare train/val split data and build message files.")
    ap.add_argument("--train_jsonl", default="", help="Legacy alias for stagewise train split JSONL")
    ap.add_argument("--val_jsonl", default="", help="Legacy alias for stagewise val split JSONL")
    ap.add_argument("--stagewise_train_jsonl", default="", help="Stagewise train split JSONL")
    ap.add_argument("--stagewise_val_jsonl", default="", help="Stagewise val split JSONL")
    ap.add_argument("--monolithic_train_jsonl", default="", help="Monolithic train split JSONL")
    ap.add_argument("--monolithic_val_jsonl", default="", help="Monolithic val split JSONL")
    ap.add_argument("--out_dir", default="data", help="Base output directory")
    ap.add_argument("--prompts_dir", default="prompts", help="Prompt templates directory")
    ap.add_argument(
        "--prompt_profile",
        default="default",
        choices=sorted(PROMPT_FILE_MAPS),
        help=(
            "Prompt family to use when building message files. "
            "'default' keeps the current strict/upper-bound prompts; "
            "'minimal' uses a bare inference prompt to test SFT-internalized behavior; "
            "'legacy_text_contract' keeps the old minimal strict JSON/text contract; "
            "'runtime' builds the recommended lightweight trace prompt; "
            "'final_only' builds final-answer-only baseline prompts."
        ),
    )
    ap.add_argument(
        "--message_tag",
        default="",
        help=(
            "Optional suffix inserted before '_messages.jsonl'. "
            "If omitted, non-default prompt profiles automatically use the profile name."
        ),
    )
    ap.add_argument(
        "--assistant_target_style",
        default="parsed",
        choices=["parsed", "canonical", "trace_text"],
        help=(
            "How to construct assistant training targets. "
            "'parsed' preserves reasoning/explanation structure closer to the source think traces; "
            "'canonical' rewrites those prose sections into more normalized templates; "
            "'trace_text' emits a human-readable 3-stage public evidence trace instead of JSON-first traces."
        ),
    )
    ap.add_argument("--train_modes", nargs="+", default=["e2e"], choices=["e2e", "oracle_conflict", "oracle_notes", "oracle_both"])
    ap.add_argument(
        "--val_modes",
        nargs="+",
        default=["e2e", "oracle_conflict", "oracle_notes", "oracle_both"],
        choices=["e2e", "oracle_conflict", "oracle_notes", "oracle_both"],
    )
    ap.add_argument(
        "--train_tasks",
        nargs="+",
        default=["e2e_trace"],
        choices=TRAIN_TASKS,
        help="Derived training tasks to build from each source example.",
    )
    ap.add_argument(
        "--val_tasks",
        nargs="+",
        default=["e2e_trace"],
        choices=TRAIN_TASKS,
        help="Derived validation tasks to build from each source example.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    message_tag = args.message_tag.strip()
    if not message_tag and args.prompt_profile != "default":
        message_tag = args.prompt_profile

    prompts = prompt_paths(args.prompts_dir, prompt_profile=args.prompt_profile)

    stagewise_train_path = args.stagewise_train_jsonl or args.train_jsonl
    stagewise_val_path = args.stagewise_val_jsonl or args.val_jsonl
    monolithic_train_path = args.monolithic_train_jsonl
    monolithic_val_path = args.monolithic_val_jsonl

    if not stagewise_train_path or not stagewise_val_path:
        raise SystemExit("Provide stagewise train/val JSONL paths.")

    stagewise_train = load_and_normalize(stagewise_train_path, "train_stagewise")
    stagewise_val = load_and_normalize(stagewise_val_path, "val_stagewise")
    write_jsonl(splits_dir / "train.jsonl", stagewise_train)
    write_jsonl(splits_dir / "val.jsonl", stagewise_val)
    write_jsonl(splits_dir / "train_stagewise.jsonl", stagewise_train)
    write_jsonl(splits_dir / "val_stagewise.jsonl", stagewise_val)
    print(f"[Write] canonical train_stagewise -> {splits_dir / 'train_stagewise.jsonl'}")
    print(f"[Write] canonical val_stagewise   -> {splits_dir / 'val_stagewise.jsonl'}")
    print(f"[Write] compatibility train       -> {splits_dir / 'train.jsonl'}")
    print(f"[Write] compatibility val         -> {splits_dir / 'val.jsonl'}")

    stagewise_train_files = build_message_files(
        "train_stagewise",
        stagewise_train,
        args.train_modes,
        out_dir,
        prompts,
        assistant_target_style=args.assistant_target_style,
        message_tag=message_tag,
        tasks=args.train_tasks,
    )
    build_message_files(
        "val_stagewise",
        stagewise_val,
        args.val_modes,
        out_dir,
        prompts,
        assistant_target_style=args.assistant_target_style,
        message_tag=message_tag,
        tasks=args.val_tasks,
    )
    if len(args.train_tasks) > 1:
        combine_message_files(
            out_dir,
            f"train_stagewise_multitask{message_tag_suffix(message_tag)}_messages.jsonl",
            stagewise_train_files,
        )
    combine_message_files(
        out_dir,
        f"val_e2e{message_tag_suffix(message_tag)}_messages.jsonl",
        [message_filename("val_stagewise", "e2e", message_tag, task="e2e_trace")],
    )
    combine_message_files(
        out_dir,
        f"val_oracle_conflict{message_tag_suffix(message_tag)}_messages.jsonl",
        [message_filename("val_stagewise", "oracle_conflict", message_tag, task="e2e_trace")],
    )
    combine_message_files(
        out_dir,
        f"val_oracle_notes{message_tag_suffix(message_tag)}_messages.jsonl",
        [message_filename("val_stagewise", "oracle_notes", message_tag, task="e2e_trace")],
    )
    combine_message_files(
        out_dir,
        f"val_oracle_both{message_tag_suffix(message_tag)}_messages.jsonl",
        [message_filename("val_stagewise", "oracle_both", message_tag, task="e2e_trace")],
    )

    if not message_tag:
        combined_train_alias = (Path(out_dir) / "messages" / "train_e2e_messages.jsonl")
        if combined_train_alias.exists():
            combined_train_alias.unlink()
            print(f"[Messages] removed stale combined training alias -> {combined_train_alias}")

    if monolithic_train_path and monolithic_val_path:
        monolithic_train = load_and_normalize(monolithic_train_path, "train_monolithic")
        monolithic_val = load_and_normalize(monolithic_val_path, "val_monolithic")
        write_jsonl(splits_dir / "train_monolithic.jsonl", monolithic_train)
        write_jsonl(splits_dir / "val_monolithic.jsonl", monolithic_val)
        print(f"[Write] canonical train_monolithic -> {splits_dir / 'train_monolithic.jsonl'}")
        print(f"[Write] canonical val_monolithic   -> {splits_dir / 'val_monolithic.jsonl'}")

        monolithic_train_files = build_message_files(
            "train_monolithic",
            monolithic_train,
            args.train_modes,
            out_dir,
            prompts,
            assistant_target_style=args.assistant_target_style,
            message_tag=message_tag,
            tasks=args.train_tasks,
        )
        build_message_files(
            "val_monolithic",
            monolithic_val,
            args.val_modes,
            out_dir,
            prompts,
            assistant_target_style=args.assistant_target_style,
            message_tag=message_tag,
            tasks=args.val_tasks,
        )
        if len(args.train_tasks) > 1:
            combine_message_files(
                out_dir,
                f"train_monolithic_multitask{message_tag_suffix(message_tag)}_messages.jsonl",
                monolithic_train_files,
            )

    print("\nPrepared split-aware datasets successfully.")
    print(f"Prompt profile: {args.prompt_profile}")
    print(f"Message tag: {message_tag or '(default)'}")
    print(f"Assistant target style: {args.assistant_target_style}")
    print(f"Train tasks: {args.train_tasks}")
    print(f"Val tasks: {args.val_tasks}")
    print("Training recommendation: fine-tune stagewise and monolithic separately.")
    if len(args.train_tasks) > 1:
        stagewise_train_name = f"train_stagewise_multitask{message_tag_suffix(message_tag)}_messages.jsonl"
        monolithic_train_name = f"train_monolithic_multitask{message_tag_suffix(message_tag)}_messages.jsonl"
    else:
        stagewise_train_name = message_filename("train_stagewise", "e2e", message_tag, task=args.train_tasks[0])
        monolithic_train_name = message_filename("train_monolithic", "e2e", message_tag, task=args.train_tasks[0])
    print(f"Use data/messages/{stagewise_train_name} or data/messages/{monolithic_train_name} explicitly.")
    print("Inference recommendation: use val_stagewise_* or val_monolithic_* files explicitly.")


if __name__ == "__main__":
    main()
