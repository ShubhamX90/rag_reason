#!/usr/bin/env python3
"""
Convert Ablation Type2 Outputs to Monolithic Format
---------------------------------------------------
Type2 generates per-doc adjudications but skips explicit conflict analysis.

For E2E:
- Adjudications from Call1
- Empty conflict line (no explicit conflict label)
- Answer from Call2

For Oracle:
- Adjudications from gold data (per_doc_notes)
- Empty conflict line
- Answer from Call1

Output format:
<think>
[adjudications array]

</think>

final answer
[[END-OF-ANSWER]]

Usage:
  python convert_ablation_type2.py \
    --ablation_dir outputs/ablations/type2_e2e/test \
    --canon_jsonl data/splits/test_v5.jsonl \
    --oracle_level e2e \
    --out_jsonl outputs/ablations/type2_e2e/test/combined.raw.jsonl
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


def convert_to_monolithic_format(adjudications: List[Dict], answer_raw: str) -> str:
    """
    Type2 ablations generate adjudications but no explicit conflict label.
    """
    # Adjudications array
    if not isinstance(adjudications, list):
        adjudications = []
    adjudications_json = json.dumps(adjudications, ensure_ascii=False, indent=2)
    
    # Empty conflict line (Type2 doesn't generate explicit conflict label)
    # Just a blank line where the conflict line would be
    
    # Final answer (remove sentinel, we'll add it back)
    answer_text = answer_raw.replace("[[END-OF-ANSWER]]", "").strip()
    
    # Assemble (note: no conflict line, just blank line)
    monolithic = (
        "<think>\n"
        f"{adjudications_json}\n"
        "\n"  # Empty line where conflict line would be
        "</think>\n\n"
        f"{answer_text}\n"
        "[[END-OF-ANSWER]]"
    )
    
    return monolithic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_dir", required=True, help="Directory with outputs")
    ap.add_argument("--canon_jsonl", required=True, help="Canonical test split")
    ap.add_argument("--oracle_level", required=True, choices=["e2e", "oracle"])
    ap.add_argument("--out_jsonl", required=True, help="Output combined.raw.jsonl")
    args = ap.parse_args()
    
    ablation_dir = Path(args.ablation_dir)
    
    # Load canonical data
    canon_data = {ex["id"]: ex for ex in read_jsonl(args.canon_jsonl)}
    
    combined = []
    skipped_count = 0
    
    if args.oracle_level == "e2e":
        # Load results
        call1_results = {r["id"]: r for r in read_jsonl(str(ablation_dir / "call1_outputs.jsonl"))}
        call2_results = {r["id"]: r for r in read_jsonl(str(ablation_dir / "call2_outputs.jsonl"))}
        
        for ex_id in sorted(canon_data.keys()):
            if ex_id not in call1_results or not call1_results[ex_id].get("valid", False):
                skipped_count += 1
                continue
            
            if ex_id not in call2_results or not call2_results[ex_id].get("has_sentinel", False):
                skipped_count += 1
                continue
            
            call1_data = call1_results[ex_id]
            call2_data = call2_results[ex_id]
            
            adjudications = call1_data.get("parsed", [])
            answer_raw = call2_data.get("raw", "")
            
            monolithic_text = convert_to_monolithic_format(adjudications, answer_raw)
            
            combined.append({
                "id": ex_id,
                "generated_text": monolithic_text
            })
    
    else:  # oracle
        # Oracle: use gold per_doc_notes as adjudications, answer from Call1
        call1_results = {r["id"]: r for r in read_jsonl(str(ablation_dir / "call1_outputs.jsonl"))}
        
        for ex_id in sorted(canon_data.keys()):
            if ex_id not in call1_results or not call1_results[ex_id].get("has_sentinel", False):
                skipped_count += 1
                continue
            
            ex = canon_data[ex_id]
            call1_data = call1_results[ex_id]
            
            adjudications = ex.get("per_doc_notes", [])
            answer_raw = call1_data.get("raw", "")
            
            monolithic_text = convert_to_monolithic_format(adjudications, answer_raw)
            
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