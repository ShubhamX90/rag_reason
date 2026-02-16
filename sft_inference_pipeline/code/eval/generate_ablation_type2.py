#!/usr/bin/env python3
"""
Ablation Type 2 Generator - Document Adjudication Without Conflict Label
-------------------------------------------------------------------------
Type2 generates per-document adjudications (with quotes) but skips explicit
conflict analysis. The final answer must infer conflict behavior from the 
adjudications.

Type2 E2E (2 calls):
  Call1: Document adjudication with quotes → adjudications JSON array
  Call2: Final answer (infer conflict behavior) → text with sentinel

Type2 Oracle (1 call):
  Call1: Final answer using GIVEN adjudications → text with sentinel

Usage:
  python generate_ablation_type2.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --in_jsonl data/splits/test_v5.jsonl \
    --oracle_level e2e \
    --prompts_dir prompts/ablations \
    --output_dir outputs/ablations/type2_e2e/test \
    --auto_length \
    --load_in_4bit \
    --save_every 25 \
    --resume
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

SENTINEL = "[[END-OF-ANSWER]]"

# ============================================================
# IO Utilities
# ============================================================

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


def load_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_already_generated(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    generated_ids = set()
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        generated_ids.add(obj["id"])
                except:
                    continue
    except:
        return set()
    return generated_ids


# ============================================================
# Stopping Criteria
# ============================================================

class SentinelStopper(StoppingCriteria):
    def __init__(self, tokenizer, sentinel: str):
        super().__init__()
        self.sentinel_ids = tokenizer.encode(sentinel, add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[0] == 0 or len(self.sentinel_ids) == 0:
            return False
        seq = input_ids[0].tolist()
        n = len(self.sentinel_ids)
        return len(seq) >= n and seq[-n:] == self.sentinel_ids


# ============================================================
# JSON Parsing
# ============================================================

def parse_json_array(text: str) -> Optional[List[Dict]]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return arr
        except:
            pass
    
    start = text.find("[")
    if start == -1:
        return None
    
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
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
                    try:
                        arr = json.loads(text[start:i+1])
                        if isinstance(arr, list):
                            return arr
                    except:
                        pass
                    break
    return None


def validate_call1_adjudications(arr: Optional[List[Dict]], expected_n: int) -> tuple:
    if not arr or not isinstance(arr, list):
        return False, "not_array"
    
    if len(arr) != expected_n:
        return False, f"wrong_count_{len(arr)}_expected_{expected_n}"
    
    required_fields = ["doc_id", "verdict", "verdict_reason", "key_fact", "quote", "source_quality"]
    for i, obj in enumerate(arr, 1):
        if not isinstance(obj, dict):
            return False, f"doc{i}_not_dict"
        for field in required_fields:
            if field not in obj:
                return False, f"doc{i}_missing_{field}"
        
        expected_doc_id = f"d{i}"
        if obj.get("doc_id") != expected_doc_id:
            return False, f"doc{i}_wrong_id"
    
    return True, None


# ============================================================
# Generation
# ============================================================

def estimate_doc_count(ex: Dict) -> int:
    docs = ex.get("retrieved_docs", [])
    return len(docs) if isinstance(docs, list) else 0


def estimate_max_new_tokens(n_docs: int, base: int, cap: int, task: str) -> int:
    if task == "adjudications":
        # Type2 adjudications include quotes, need more tokens
        target = 200 + n_docs * 120
        return min(target, cap)
    else:  # answer
        target = base + n_docs * 30
        return min(target, cap)


def build_prompt(tokenizer, sys_txt: str, user_txt: str) -> str:
    messages = [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": user_txt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one(model, tokenizer, prompt: str, max_new: int, temp: float, top_p: float, max_ctx: int, stopper=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    if input_ids.shape[1] + max_new > max_ctx:
        overflow = input_ids.shape[1] + max_new - max_ctx
        input_ids = input_ids[:, overflow:]
    
    stopping_criteria = StoppingCriteriaList([stopper]) if stopper else None
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            do_sample=(temp > 0),
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ============================================================
# Progress Tracker
# ============================================================

class ProgressTracker:
    def __init__(self, total: int, label: str):
        self.total = total
        self.label = label
        self.current = 0
        self.start_time = time.time()
    
    def update(self):
        self.current += 1
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        print(f"\r{self.label}: {self.current}/{self.total} | {rate:.2f} ex/s | ETA: {eta/60:.1f}m", end="", flush=True)
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\r{self.label}: {self.total}/{self.total} | Done in {elapsed/60:.1f}m")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--oracle_level", required=True, choices=["e2e", "oracle"])
    ap.add_argument("--prompts_dir", required=True)
    ap.add_argument("--max_new_tokens_base", type=int, default=600)
    ap.add_argument("--max_new_tokens_cap", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--auto_length", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts_dir = Path(args.prompts_dir)
    
    # Load model
    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    
    sentinel_stopper = SentinelStopper(tokenizer, SENTINEL)
    
    # Load data
    items = list(read_jsonl(args.in_jsonl))
    print(f"Loaded {len(items)} examples from {args.in_jsonl}")
    
    if args.oracle_level == "e2e":
        # Two calls: adjudication + answer
        call1_out = output_dir / "call1_outputs.jsonl"
        call2_out = output_dir / "call2_outputs.jsonl"
        
        # Load prompts
        call1_sys = load_text(prompts_dir / "system_call1_type2_e2e.txt")
        call1_usr_template = load_text(prompts_dir / "user_call1_type2_e2e.txt")
        call2_sys = load_text(prompts_dir / "system_call2_type2_e2e.txt")
        call2_usr_template = load_text(prompts_dir / "user_call2_type2_e2e.txt")
        
        # ========================================================
        # CALL 1: Document Adjudication
        # ========================================================
        print(f"\n{'='*60}")
        print("CALL 1: Document Adjudication (with quotes)")
        print(f"{'='*60}")
        
        already_done_call1 = load_already_generated(call1_out) if args.resume else set()
        items_call1 = [ex for ex in items if ex.get("id") not in already_done_call1]
        print(f"Already done: {len(already_done_call1)}, To generate: {len(items_call1)}")
        
        if items_call1:
            progress_call1 = ProgressTracker(len(items_call1), "Call1")
            file_mode = "a" if (args.resume and call1_out.exists()) else "w"
            
            with open(call1_out, file_mode, encoding="utf-8") as f1:
                for idx, ex in enumerate(items_call1):
                    ex_id = ex.get("id")
                    n_docs = estimate_doc_count(ex)
                    
                    user_msg = call1_usr_template.format(
                        query=ex.get("query", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2)
                    )
                    
                    prompt = build_prompt(tokenizer, call1_sys, user_msg)
                    max_new = estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap, "adjudications") if args.auto_length else args.max_new_tokens_cap
                    raw_output = generate_one(model, tokenizer, prompt, max_new, args.temperature, args.top_p, max_ctx)
                    
                    parsed = parse_json_array(raw_output)
                    is_valid, err_msg = validate_call1_adjudications(parsed, n_docs) if parsed else (False, "parse_failed")
                    
                    f1.write(json.dumps({"id": ex_id, "raw": raw_output, "parsed": parsed, "valid": is_valid, "error": err_msg if not is_valid else None}, ensure_ascii=False) + "\n")
                    
                    if (idx + 1) % args.save_every == 0:
                        f1.flush()
                    progress_call1.update()
            progress_call1.finish()
        
        call1_results = {r["id"]: r for r in read_jsonl(str(call1_out))}
        
        # ========================================================
        # CALL 2: Final Answer
        # ========================================================
        print(f"\n{'='*60}")
        print("CALL 2: Final Answer Generation (inferred conflict)")
        print(f"{'='*60}")
        
        items_call2_candidates = [ex for ex in items if ex.get("id") in call1_results and call1_results[ex["id"]].get("valid", False)]
        already_done_call2 = load_already_generated(call2_out) if args.resume else set()
        items_call2 = [ex for ex in items_call2_candidates if ex.get("id") not in already_done_call2]
        print(f"Valid from Call1: {len(items_call2_candidates)}, Already done: {len(already_done_call2)}, To generate: {len(items_call2)}")
        
        if items_call2:
            progress_call2 = ProgressTracker(len(items_call2), "Call2")
            file_mode = "a" if (args.resume and call2_out.exists()) else "w"
            
            with open(call2_out, file_mode, encoding="utf-8") as f2:
                for idx, ex in enumerate(items_call2):
                    ex_id = ex.get("id")
                    n_docs = estimate_doc_count(ex)
                    call1_data = call1_results[ex_id]
                    adjudications = call1_data["parsed"]
                    
                    user_msg = call2_usr_template.format(
                        query=ex.get("query", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        s1_adjudications_json=json.dumps(adjudications, ensure_ascii=False, indent=2)
                    )
                    
                    prompt = build_prompt(tokenizer, call2_sys, user_msg)
                    max_new = estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap, "answer") if args.auto_length else args.max_new_tokens_base
                    
                    raw_output = generate_one(model, tokenizer, prompt, max_new, args.temperature, args.top_p, max_ctx, stopper=sentinel_stopper)
                    has_sentinel = SENTINEL in raw_output
                    
                    f2.write(json.dumps({"id": ex_id, "raw": raw_output, "has_sentinel": has_sentinel}, ensure_ascii=False) + "\n")
                    
                    if (idx + 1) % args.save_every == 0:
                        f2.flush()
                    progress_call2.update()
            progress_call2.finish()
        
        # Summary
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Outputs: {output_dir}")
        
        call1_total = len(list(read_jsonl(str(call1_out))))
        call1_valid = sum(1 for r in read_jsonl(str(call1_out)) if r.get("valid", False))
        print(f"  Call 1: {call1_valid}/{call1_total} valid")
        
        call2_total = len(list(read_jsonl(str(call2_out))))
        call2_sentinel = sum(1 for r in read_jsonl(str(call2_out)) if r.get("has_sentinel", False))
        print(f"  Call 2: {call2_sentinel}/{call2_total} with sentinel")
        
    else:  # oracle
        # One call: answer using GIVEN adjudications
        call1_out = output_dir / "call1_outputs.jsonl"
        
        # Load prompts
        call1_sys = load_text(prompts_dir / "system_call1_type2_oracle.txt")
        call1_usr_template = load_text(prompts_dir / "user_call1_type2_oracle.txt")
        
        # ========================================================
        # CALL 1: Final Answer (Oracle)
        # ========================================================
        print(f"\n{'='*60}")
        print("CALL 1: Final Answer Generation (Oracle - given adjudications)")
        print(f"{'='*60}")
        
        already_done_call1 = load_already_generated(call1_out) if args.resume else set()
        items_call1 = [ex for ex in items if ex.get("id") not in already_done_call1]
        print(f"Already done: {len(already_done_call1)}, To generate: {len(items_call1)}")
        
        if items_call1:
            progress_call1 = ProgressTracker(len(items_call1), "Call1")
            file_mode = "a" if (args.resume and call1_out.exists()) else "w"
            
            with open(call1_out, file_mode, encoding="utf-8") as f1:
                for idx, ex in enumerate(items_call1):
                    ex_id = ex.get("id")
                    n_docs = estimate_doc_count(ex)
                    
                    user_msg = call1_usr_template.format(
                        query=ex.get("query", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        s1_adjudications_json=json.dumps(ex.get("per_doc_notes", []), ensure_ascii=False, indent=2)
                    )
                    
                    prompt = build_prompt(tokenizer, call1_sys, user_msg)
                    max_new = estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap, "answer") if args.auto_length else args.max_new_tokens_base
                    
                    raw_output = generate_one(model, tokenizer, prompt, max_new, args.temperature, args.top_p, max_ctx, stopper=sentinel_stopper)
                    has_sentinel = SENTINEL in raw_output
                    
                    f1.write(json.dumps({"id": ex_id, "raw": raw_output, "has_sentinel": has_sentinel}, ensure_ascii=False) + "\n")
                    
                    if (idx + 1) % args.save_every == 0:
                        f1.flush()
                    progress_call1.update()
            progress_call1.finish()
        
        # Summary
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Outputs: {output_dir}")
        
        call1_total = len(list(read_jsonl(str(call1_out))))
        call1_sentinel = sum(1 for r in read_jsonl(str(call1_out)) if r.get("has_sentinel", False))
        print(f"  Call 1: {call1_sentinel}/{call1_total} with sentinel")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()