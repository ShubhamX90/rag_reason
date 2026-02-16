#!/usr/bin/env python3
"""
Convert Ablation Type1 Outputs to Monolithic Format
---------------------------------------------------
Type1 skips document adjudication, so outputs:
- Empty adjudications array []
- Conflict line from Call1
- Answer from Call2

Output format:
<think>
[]
conflict_type — conflict_reason
</think>

final answer
[[END-OF-ANSWER]]

Usage:
  python convert_ablation_type1.py \
    --ablation_dir outputs/ablations/type1_e2e/test \
    --canon_jsonl data/splits/test_v5.jsonl \
    --oracle_level e2e \
    --out_jsonl outputs/ablations/type1_e2e/test/combined.raw.jsonl
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


def convert_to_monolithic_format(conflict_obj: Dict, answer_raw: str) -> str:
    """
    Type1 ablations don't generate per-doc adjudications,
    so we output an empty array.
    """
    # Empty adjudications array (Type1 doesn't do document adjudication)
    adjudications_json = "[]"
    
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
    ap.add_argument("--ablation_dir", required=True, help="Directory with call1_outputs.jsonl and call2_outputs.jsonl")
    ap.add_argument("--canon_jsonl", required=True, help="Canonical test split")
    ap.add_argument("--oracle_level", required=True, choices=["e2e", "oracle"])
    ap.add_argument("--out_jsonl", required=True, help="Output combined.raw.jsonl")
    args = ap.parse_args()
    
    ablation_dir = Path(args.ablation_dir)
    
    # Load results
    call1_results = {r["id"]: r for r in read_jsonl(str(ablation_dir / "call1_outputs.jsonl"))}
    call2_results = {r["id"]: r for r in read_jsonl(str(ablation_dir / "call2_outputs.jsonl"))}
    
    # Load canonical data
    canon_data = {ex["id"]: ex for ex in read_jsonl(args.canon_jsonl)}
    
    # Convert
    combined = []
    skipped_count = 0
    
    for ex_id in sorted(canon_data.keys()):
        if ex_id not in call1_results or not call1_results[ex_id].get("valid", False):
            skipped_count += 1
            continue
        
        if ex_id not in call2_results or not call2_results[ex_id].get("has_sentinel", False):
            skipped_count += 1
            continue
        
        call1_data = call1_results[ex_id]
        call2_data = call2_results[ex_id]
        
        conflict_obj = call1_data.get("parsed", {})
        answer_raw = call2_data.get("raw", "")
        
        monolithic_text = convert_to_monolithic_format(conflict_obj, answer_raw)
        
        combined.append({
            "id": ex_id,
            "generated_text": monolithic_text
        })
    
    # Write output
    write_jsonl(args.out_jsonl, combined)
    
    print(f"Converted {len(combined)} examples to {args.out_jsonl}")
    print(f"Skipped {skipped_count} examples (invalid or missing)")


if __name__ == "__main__":
    main()