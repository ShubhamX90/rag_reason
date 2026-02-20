#!/usr/bin/env python3
"""
prepare_data.py  –  Data Preparation for SFT Pipeline v2
=========================================================
Takes the raw Stage-3 annotated JSONL and produces:
  1. Train / Val / Test splits  (canonical schema)
  2. Chat-message JSONL files   (system + user + assistant) ready for
     QLoRA fine-tuning and baseline/SFT inference.

Two prompt modes are produced:
  • e2e     – model must predict per-doc notes + conflict type + answer
  • oracle  – gold conflict type is provided; model predicts per-doc notes + answer

Usage:
  python code/data/prepare_data.py \
    --raw_jsonl  data/raw/stage3_final.jsonl \
    --out_dir    data \
    --prompts_dir prompts \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --seed 42
"""

import json, argparse, re, os, random, hashlib
from pathlib import Path
from collections import Counter

# ────────── Constants ──────────
SENTINEL = "[[END-OF-ANSWER]]"
EM_DASH  = " — "
ALLOWED_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}

# Common aliases from raw data
CONFLICT_TYPE_ALIASES = {
    "Conflicting opinions and research outcomes": "Conflicting opinions or research outcomes",
}

def normalize_conflict_type(ct):
    """Normalize conflict type to canonical form."""
    ct = (ct or "").strip()
    return CONFLICT_TYPE_ALIASES.get(ct, ct)

# ────────── IO helpers ──────────
def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{p}:{ln} bad json: {e}")

def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()

def truncate_words(s, maxw):
    w = s.strip().split()
    return " ".join(w[:maxw])

def list_ids(ids):
    return ", ".join(ids)

def sanitize_no_ranges(text):
    return re.sub(r"\bd(\d+)\s*[–-]\s*d(\d+)\b", r"d\1, d\2", text)


# ────────── Build gold assistant response (TEXT-MODE) ──────────

def build_doc_array(ex):
    """Build the per-doc JSON array from gold per_doc_notes."""
    rd = ex.get("retrieved_docs") or []
    notes = {(n.get("doc_id") or ""): n for n in (ex.get("per_doc_notes") or [])}
    arr = []
    for d in rd:
        did = d.get("doc_id") or ""
        n = notes.get(did, {})
        verdict = (n.get("verdict") or "irrelevant").strip()
        vr = (n.get("verdict_reason") or "").strip()
        kf = (n.get("key_fact") or "").strip()
        sq = (n.get("source_quality") or "low").strip().lower()
        if verdict == "irrelevant":
            kf = ""
        if sq not in ("high", "low"):
            sq = "low"
        arr.append({
            "doc_id": did,
            "verdict": verdict,
            "verdict_reason": truncate_words(vr, 80),
            "key_fact": kf,
            "source_quality": sq,
        })
    return arr

def ids_by_verdict(arr):
    irr, sup, psup = [], [], []
    for x in arr:
        did = x["doc_id"]
        v = (x.get("verdict") or "").strip()
        if v == "supports":
            sup.append(did)
        elif v == "partially supports":
            psup.append(did)
        else:
            irr.append(did)
    return irr, sup, psup

def build_reasoning(arr):
    irr, sup, psup = ids_by_verdict(arr)
    if sup or psup:
        ok = sup + psup
        if irr:
            return sanitize_no_ranges(
                f"Docs {list_ids(ok)} {'partially support or support' if psup else 'support'} "
                f"the answer, while docs {list_ids(irr)} are irrelevant; mechanism: contextual-scope."
            )
        else:
            return sanitize_no_ranges(
                f"Docs {list_ids(ok)} align on the answer; mechanism: none (agreement)."
            )
    else:
        return sanitize_no_ranges(
            f"All docs {list_ids(irr)} are irrelevant to the query; mechanism: contextual-scope."
        )

def build_label_line(conflict_type, conflict_reason):
    ctype = conflict_type if conflict_type in ALLOWED_TYPES else "No conflict"
    reason = truncate_words(conflict_reason or "", 50)
    return sanitize_no_ranges(f"{ctype}{EM_DASH}{reason}")

def build_explanation(arr, abstain):
    if abstain:
        return "Given the above evidence is insufficient to answer the query, abstention is required."
    irr, sup, psup = ids_by_verdict(arr)
    ok = sup + psup
    if ok:
        return f"The identified sources ({list_ids(ok)}) provide the basis for the final answer."
    return "Proceeding with the best supported conclusion from the available evidence."

def ensure_answer_citations(ans, evidence_ids):
    if not evidence_ids:
        return ans
    sents = re.split(r"(?<=[.!?])\s+", ans.strip())
    for i, s in enumerate(sents):
        if not re.search(r"\[d\d+\]", s):
            sents[i] = s + f" [{evidence_ids[0]}]"
    return " ".join(sents)

def build_assistant_text(ex):
    """Build the full TEXT-MODE assistant response from gold annotations."""
    arr = build_doc_array(ex)
    arr_json = json.dumps(arr, ensure_ascii=False, indent=2)

    ctype = normalize_conflict_type(ex.get("conflict_type") or "")
    creason = (ex.get("conflict_reason") or "").strip()
    abstain = bool(ex.get("expected_response", {}).get("abstain", False))

    reasoning = build_reasoning(arr)
    label_line = build_label_line(ctype, creason)
    explanation = build_explanation(arr, abstain)

    if abstain:
        final = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
    else:
        ans = (ex.get("expected_response", {}).get("answer") or "").strip()
        ev = ex.get("expected_response", {}).get("evidence") or []
        valid_ids = [i for i in ev if any(d.get("doc_id") == i for d in (ex.get("retrieved_docs") or []))]
        final = ensure_answer_citations(ans, valid_ids)

    assistant = (
        "<think>\n"
        f"{arr_json}\n"
        f"{sanitize_no_ranges(reasoning)}\n\n"
        f"{sanitize_no_ranges(label_line)}\n\n"
        f"{sanitize_no_ranges(explanation)}\n"
        "</think>\n\n"
        f"{final}\n{SENTINEL}"
    )
    return assistant


# ────────── Build messages for different modes ──────────

def build_messages_e2e(ex, system_prompt, user_template):
    """E2E: model predicts everything."""
    user_msg = user_template.format(
        query=ex.get("query", ""),
        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
    )
    assistant = build_assistant_text(ex)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant},
    ]

def build_messages_oracle(ex, system_prompt, user_template):
    """Oracle: gold conflict type is provided."""
    user_msg = user_template.format(
        query=ex.get("query", ""),
        conflict_type=normalize_conflict_type(ex.get("conflict_type", "")),
        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
    )
    assistant = build_assistant_text(ex)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant},
    ]


# ────────── Validation ──────────

def validate_messages(msgs):
    """Basic structural checks on a message triple."""
    if not (isinstance(msgs, list) and len(msgs) == 3):
        return False
    if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[2]["role"] != "assistant":
        return False
    assistant = msgs[2]["content"]
    if assistant.count("<think>") != 1 or assistant.count("</think>") != 1:
        return False
    if SENTINEL not in assistant:
        return False
    return True


# ────────── Splitting ──────────

def deterministic_split(examples, train_r, val_r, test_r, seed=42):
    """Stratified split by conflict_type with deterministic shuffling."""
    rng = random.Random(seed)

    # Group by conflict type
    by_type = {}
    for ex in examples:
        ct = ex.get("conflict_type", "Unknown")
        by_type.setdefault(ct, []).append(ex)

    train, val, test = [], [], []
    for ct, items in sorted(by_type.items()):
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_r))
        n_val = max(1, round(n * val_r))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Very small class: at least 1 train
            n_train = max(1, n - 2)
            n_val = min(1, n - n_train)
            n_test = n - n_train - n_val

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ────────── Main ──────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_jsonl", required=True, help="Raw Stage-3 annotated JSONL")
    ap.add_argument("--out_dir", default="data", help="Output base directory")
    ap.add_argument("--prompts_dir", default="prompts", help="Prompts directory")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_split", action="store_true",
                    help="Skip splitting; use existing splits in out_dir/splits/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    splits_dir = out_dir / "splits"
    msgs_dir = out_dir / "messages"
    splits_dir.mkdir(parents=True, exist_ok=True)
    msgs_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts_dir = Path(args.prompts_dir)
    sys_e2e = load_text(prompts_dir / "system_e2e.txt")
    usr_e2e = load_text(prompts_dir / "user_e2e.txt")
    sys_oracle = load_text(prompts_dir / "system_oracle.txt")
    usr_oracle = load_text(prompts_dir / "user_oracle.txt")

    # ── Step 1: Split ──
    if args.skip_split:
        print("[Split] Skipping split; loading existing splits...")
        train = list(read_jsonl(splits_dir / "train.jsonl"))
        val = list(read_jsonl(splits_dir / "val.jsonl"))
        test = list(read_jsonl(splits_dir / "test.jsonl"))
    else:
        print(f"[Split] Loading raw data from {args.raw_jsonl}...")
        all_data = list(read_jsonl(args.raw_jsonl))
        # Normalize conflict types
        for ex in all_data:
            if "conflict_type" in ex:
                ex["conflict_type"] = normalize_conflict_type(ex["conflict_type"])
        print(f"[Split] Total examples: {len(all_data)}")

        train, val, test = deterministic_split(
            all_data, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

        write_jsonl(splits_dir / "train.jsonl", train)
        write_jsonl(splits_dir / "val.jsonl", val)
        write_jsonl(splits_dir / "test.jsonl", test)
        print(f"[Split] Train={len(train)}, Val={len(val)}, Test={len(test)}")

        # Distribution
        for name, split in [("Train", train), ("Val", val), ("Test", test)]:
            dist = Counter(ex.get("conflict_type", "?") for ex in split)
            print(f"  {name}: {dict(sorted(dist.items()))}")

    # ── Step 2: Build messages ──
    for mode in ["e2e", "oracle"]:
        sys_prompt = sys_e2e if mode == "e2e" else sys_oracle
        usr_tmpl = usr_e2e if mode == "e2e" else usr_oracle
        builder = build_messages_e2e if mode == "e2e" else build_messages_oracle

        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            rows = []
            kept = dropped = 0
            for ex in split_data:
                try:
                    msgs = builder(ex, sys_prompt, usr_tmpl)
                except Exception as e:
                    print(f"  [WARN] {ex.get('id')}: {e}")
                    dropped += 1
                    continue

                if not validate_messages(msgs):
                    dropped += 1
                    continue

                rows.append({"id": ex.get("id"), "messages": msgs})
                kept += 1

            out_path = msgs_dir / f"{split_name}_{mode}_messages.jsonl"
            write_jsonl(out_path, rows)
            print(f"[Messages] {mode}/{split_name}: kept={kept}, dropped={dropped} → {out_path}")

    print("\n✓ Data preparation complete!")
    print(f"  Splits:   {splits_dir}/")
    print(f"  Messages: {msgs_dir}/")


if __name__ == "__main__":
    main()
