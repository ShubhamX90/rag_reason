#!/usr/bin/env python3
"""
Convert Main Stagewise Outputs to Monolithic Format - CORRECTED
---------------------------------------------------------------
Handles different call structures:

E2E/Oracle1 (3 calls):
  Call1 = doc adjudications (array)
  Call2 = conflict analysis (object)
  Call3 = final answer (text)
  
Oracle2/Oracle3 (2 calls):
  Call1 = conflict analysis (object)
  Call2 = final answer (text)
  Need to get adjudications from gold data

Output format (monolithic TEXT-MODE):
<think>
[doc adjudications array]
conflict line
</think>

final answer
[[END-OF-ANSWER]]

Usage:
  python convert_main_stagewise.py \
    --stagewise_dir outputs/main_stagewise/e2e/test \
    --canon_jsonl data/splits/test_v5.jsonl \
    --oracle_level e2e \
    --out_jsonl outputs/main_stagewise/e2e/test/combined.raw.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def read_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except:
                continue


def write_jsonl(p: str, rows: List[Dict]):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert_to_monolithic_format(adjudications, conflict_obj, answer_raw: str) -> str:
    """
    Combine outputs into monolithic TEXT-MODE format:
    <think>
    [adjudications array]
    conflict line
    </think>
    
    final answer
    [[END-OF-ANSWER]]
    """
    # Adjudications array
    if not isinstance(adjudications, list):
        adjudications = []
    adjudications_json = json.dumps(adjudications, ensure_ascii=False, indent=2)
    
    # Conflict line
    if not isinstance(conflict_obj, dict):
        conflict_obj = {"conflict_type": "No conflict", "conflict_reason": ""}
    
    conflict_type = conflict_obj.get("conflict_type", "No conflict")
    conflict_reason = conflict_obj.get("conflict_reason", "")
    conflict_line = f"{conflict_type} — {conflict_reason}"
    
    # Final answer (remove sentinel, we'll add it back)
    answer_text = answer_raw.replace("[[END-OF-ANSWER]]", "").strip()
    
    # Assemble
    monolithic = (
        "<think>\n"
        f"{adjudications_json}\n"
        f"{conflict_line}\n"
        "</think>\n\n"
        f"{answer_text}\n"
        "[[END-OF-ANSWER]]"
    )
    
    return monolithic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stagewise_dir", required=True)
    ap.add_argument("--canon_jsonl", required=True)
    ap.add_argument("--oracle_level", required=True, choices=["e2e", "oracle1", "oracle2", "oracle3"])
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()
    
    stagewise_dir = Path(args.stagewise_dir)
    has_call3 = args.oracle_level in ["e2e", "oracle1"]
    
    # Load canonical data (for gold fallback)
    canon_by_id = {ex["id"]: ex for ex in read_jsonl(args.canon_jsonl)}
    
    # Load call outputs
    call1_file = stagewise_dir / "call1_outputs.jsonl"
    call2_file = stagewise_dir / "call2_outputs.jsonl"
    call3_file = stagewise_dir / "call3_outputs.jsonl" if has_call3 else None
    
    call1_by_id = {r["id"]: r for r in read_jsonl(str(call1_file))} if call1_file.exists() else {}
    call2_by_id = {r["id"]: r for r in read_jsonl(str(call2_file))} if call2_file.exists() else {}
    call3_by_id = {r["id"]: r for r in read_jsonl(str(call3_file))} if (call3_file and call3_file.exists()) else {}
    
    print(f"[Convert] Oracle level: {args.oracle_level}")
    print(f"[Convert] Call structure: {'3 calls (Call1=adj, Call2=conflict, Call3=answer)' if has_call3 else '2 calls (Call1=conflict, Call2=answer)'}")
    print(f"[Convert] Call1: {len(call1_by_id)} examples")
    print(f"[Convert] Call2: {len(call2_by_id)} examples")
    if has_call3:
        print(f"[Convert] Call3: {len(call3_by_id)} examples")
    
    # Find examples that have final answer (Call3 for E2E/Oracle1, Call2 for Oracle2/3)
    if has_call3:
        # E2E/Oracle1: need Call3 for final answer
        final_answer_by_id = call3_by_id
        common_ids = set(final_answer_by_id.keys())
    else:
        # Oracle2/Oracle3: Call2 is final answer
        final_answer_by_id = call2_by_id
        common_ids = set(final_answer_by_id.keys())
    
    print(f"[Convert] Examples with final answer: {len(common_ids)}")
    
    # Convert
    out_rows = []
    for ex_id in sorted(common_ids):
        # Get adjudications
        if has_call3:
            # E2E/Oracle1: Call1 is adjudications
            if ex_id in call1_by_id and call1_by_id[ex_id].get("valid", False):
                adjudications = call1_by_id[ex_id]["parsed"]
            else:
                adjudications = []  # Parse failed
        else:
            # Oracle2/Oracle3: Use gold per_doc_notes as adjudications
            adjudications = canon_by_id[ex_id].get("per_doc_notes", [])
        
        # Get conflict analysis
        if has_call3:
            # E2E/Oracle1: Call2 is conflict analysis
            if ex_id in call2_by_id and call2_by_id[ex_id].get("valid", False):
                conflict_obj = call2_by_id[ex_id]["parsed"]
            else:
                conflict_obj = {"conflict_type": "No conflict", "conflict_reason": ""}
        else:
            # Oracle2/Oracle3: Call1 is conflict analysis
            if ex_id in call1_by_id and call1_by_id[ex_id].get("valid", False):
                conflict_obj = call1_by_id[ex_id]["parsed"]
            else:
                conflict_obj = {"conflict_type": "No conflict", "conflict_reason": ""}
        
        # Get final answer
        answer_raw = final_answer_by_id[ex_id]["raw"]
        
        # Convert to monolithic format
        monolithic_text = convert_to_monolithic_format(adjudications, conflict_obj, answer_raw)
        
        out_rows.append({
            "id": ex_id,
            "raw": monolithic_text,
        })
    
    write_jsonl(args.out_jsonl, out_rows)
    print(f"[Convert] Wrote {len(out_rows)} examples → {args.out_jsonl}")


if __name__ == "__main__":
    main()