#!/usr/bin/env python3
"""
Stagewise Sequential Generator - CORRECTED VERSION
--------------------------------------------------
Handles the actual call structure from prompts:

E2E (3 calls):
  Call1: Generate doc adjudications
  Call2: Generate conflict analysis using Call1
  Call3: Generate final answer using Call1 + Call2

Oracle1 (3 calls):
  Call1: Generate doc adjudications  
  Call2: Generate conflict reasoning using Call1 + GIVEN conflict_type
  Call3: Generate final answer using Call1 + Call2

Oracle2 (2 calls):
  Call1: Generate conflict analysis using GIVEN per_doc_notes
  Call2: Generate final answer using Call1 + GIVEN per_doc_notes
  
Oracle3 (2 calls):
  Call1: Copy GIVEN conflict_type + reasoning using GIVEN per_doc_notes
  Call2: Generate final answer using Call1 + GIVEN per_doc_notes

Usage:
  python generate_main_stagewise.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --in_jsonl data/splits/test_v5.jsonl \
    --oracle_level e2e \
    --prompts_dir prompts/main_stagewise \
    --output_dir outputs/main_stagewise/e2e/test \
    --auto_length \
    --load_in_4bit \
    --save_every 25 \
    --resume
"""

import os
import json
from robust_json_parser import parse_json_robust
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
# JSON Parsing & Validation
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


def parse_json_object(text: str) -> Optional[Dict]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except:
            pass
    
    start = text.find("{")
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
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i+1])
                        if isinstance(obj, dict):
                            return obj
                    except:
                        pass
                    break
    return None


def validate_call1_adjudications(arr: List[Dict], expected_doc_count: int) -> Tuple[bool, str]:
    """Validate Call1 output for E2E/Oracle1 (doc adjudications)"""
    if not isinstance(arr, list):
        return False, "not_a_list"
    if len(arr) != expected_doc_count:
        return False, f"wrong_length"
    for i, obj in enumerate(arr, 1):
        if not isinstance(obj, dict):
            return False, f"item_{i}_not_dict"
        if obj.get("doc_id") != f"d{i}":
            return False, f"wrong_doc_id"
    return True, "ok"


def validate_conflict_analysis(obj: Dict) -> Tuple[bool, str]:
    """Validate conflict analysis output (Call2 for E2E/Oracle1, Call1 for Oracle2/3)"""
    if not isinstance(obj, dict):
        return False, "not_a_dict"
    if "conflict_type" not in obj or "conflict_reason" not in obj:
        return False, "missing_keys"
    return True, "ok"


# ============================================================
# Progress Tracker
# ============================================================

class ProgressTracker:
    def __init__(self, total: int, stage_name: str):
        self.total = total
        self.completed = 0
        self.stage_name = stage_name
        self.start_time = time.time()
        self.last_print_time = time.time()
    
    def update(self, increment: int = 1):
        self.completed += increment
        current_time = time.time()
        if (current_time - self.last_print_time >= 5.0) or (self.completed % 10 == 0):
            self._print_progress()
            self.last_print_time = current_time
    
    def _print_progress(self):
        elapsed = time.time() - self.start_time
        pct = 100.0 * self.completed / max(1, self.total)
        if self.completed > 0:
            avg_time = elapsed / self.completed
            remaining = self.total - self.completed
            eta_mins = (avg_time * remaining) / 60.0
            print(f"[{self.stage_name}] {self.completed}/{self.total} ({pct:.1f}%) | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_mins:.1f}m | Speed: {avg_time:.1f}s/ex")
        else:
            print(f"[{self.stage_name}] {self.completed}/{self.total} ({pct:.1f}%)")
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"[{self.stage_name} Complete] {self.completed}/{self.total} in {elapsed/60:.1f} minutes")


# ============================================================
# Generation Helpers
# ============================================================

def estimate_doc_count(ex: Dict) -> int:
    rd = ex.get("retrieved_docs", [])
    return len(rd) if isinstance(rd, list) else 0


def estimate_max_new_tokens(n_docs: int, base: int, cap: int, call_type: str) -> int:
    if call_type == "adjudications":
        est = int(100 * max(1, n_docs) + 50)
    elif call_type == "conflict":
        est = 250
    elif call_type == "answer":
        est = int(200 + 20 * n_docs)
    else:
        est = base
    return max(base, min(est, cap))


def build_prompt(tokenizer, system_txt: str, user_txt: str) -> str:
    msgs = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_one(model, tokenizer, prompt: str, max_new: int, temperature: float,
                 top_p: float, max_ctx: int, stopper=None) -> str:
    safety = 32
    max_inp = max(512, int(max_ctx) - int(max_new) - safety)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_inp).to(model.device)
    gen = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=(temperature > 0.0),
        temperature=temperature, top_p=top_p, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, stopping_criteria=stopper,
    )
    return tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


# ============================================================
# Main Pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--attn_impl", choices=["eager", "sdpa"], default="sdpa")
    
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--oracle_level", required=True, choices=["e2e", "oracle1", "oracle2", "oracle3"])
    ap.add_argument("--prompts_dir", default="prompts/main_stagewise")
    
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--auto_length", action="store_true")
    ap.add_argument("--max_new_tokens_base", type=int, default=800)
    ap.add_argument("--max_new_tokens_cap", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of calls for this oracle level
    has_call3 = args.oracle_level in ["e2e", "oracle1"]
    
    call1_out = output_dir / "call1_outputs.jsonl"
    call2_out = output_dir / "call2_outputs.jsonl"
    call3_out = output_dir / "call3_outputs.jsonl" if has_call3 else None
    
    items_all = list(read_jsonl(args.in_jsonl))
    items = items_all[:args.limit] if args.limit > 0 else items_all
    
    print(f"\n{'='*60}")
    print(f"MAIN STAGEWISE PIPELINE: {args.oracle_level.upper()}")
    print(f"{'='*60}")
    print(f"Total examples: {len(items)}")
    print(f"Calls for {args.oracle_level}: {'3 (Call1, Call2, Call3)' if has_call3 else '2 (Call1, Call2)'}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load prompts
    prompts_dir = Path(args.prompts_dir)
    call1_sys = load_text(prompts_dir / f"system_call1_stagewise_{args.oracle_level}.txt")
    call1_usr_template = load_text(prompts_dir / f"user_call1_stagewise_{args.oracle_level}.txt")
    call2_sys = load_text(prompts_dir / f"system_call2_stagewise_{args.oracle_level}.txt")
    call2_usr_template = load_text(prompts_dir / f"user_call2_stagewise_{args.oracle_level}.txt")
    
    if has_call3:
        call3_sys = load_text(prompts_dir / f"system_call3_stagewise_{args.oracle_level}.txt")
        call3_usr_template = load_text(prompts_dir / f"user_call3_stagewise_{args.oracle_level}.txt")
    
    # Load model
    print(f"[Loading] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    quant_cfg = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[Loading] Model in 4-bit mode...")
    
    print(f"[Loading] Model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=args.attn_impl, quantization_config=quant_cfg,
        local_files_only=args.local_files_only,
    )
    model.eval()
    print("[Loading] Model loaded successfully\n")
    
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    tokenizer.model_max_length = max_ctx
    tokenizer.truncation_side = "left"
    sentinel_stopper = StoppingCriteriaList([SentinelStopper(tokenizer, SENTINEL)])
    
    # ========================================================
    # CALL 1
    # ========================================================
    print(f"\n{'='*60}")
    if args.oracle_level in ["e2e", "oracle1"]:
        print("CALL 1: Document-level Adjudication")
        call1_task = "adjudications"
    else:  # oracle2, oracle3
        print("CALL 1: Conflict Analysis")
        call1_task = "conflict"
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
                
                # Build user message based on oracle level
                if args.oracle_level == "e2e":
                    user_msg = call1_usr_template.format(
                        query=ex.get("query", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                    )
                elif args.oracle_level == "oracle1":
                    user_msg = call1_usr_template.format(
                        query=ex.get("query", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                    )
                elif args.oracle_level == "oracle2":
                    user_msg = call1_usr_template.format(
                        query=ex.get("query", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        per_doc_notes_json=json.dumps(ex.get("per_doc_notes", []), ensure_ascii=False, indent=2),
                    )
                else:  # oracle3
                    user_msg = call1_usr_template.format(
                        query=ex.get("query", ""),
                        conflict_type=ex.get("conflict_type", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        per_doc_notes_json=json.dumps(ex.get("per_doc_notes", []), ensure_ascii=False, indent=2),
                    )
                
                prompt = build_prompt(tokenizer, call1_sys, user_msg)
                max_new = estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap, call1_task) if args.auto_length else args.max_new_tokens_base
                raw_output = generate_one(model, tokenizer, prompt, max_new, args.temperature, args.top_p, max_ctx)
                
                # Parse based on task
                if call1_task == "adjudications":
                    parsed = parse_json_array(raw_output)
                    is_valid, err_msg = validate_call1_adjudications(parsed, n_docs) if parsed else (False, "parse_failed")
                else:  # conflict
                    parsed = parse_json_object(raw_output)
                    is_valid, err_msg = validate_conflict_analysis(parsed) if parsed else (False, "parse_failed")
                
                f1.write(json.dumps({"id": ex_id, "raw": raw_output, "parsed": parsed, "valid": is_valid, "error": err_msg if not is_valid else None}, ensure_ascii=False) + "\n")
                
                if (idx + 1) % args.save_every == 0:
                    f1.flush()
                progress_call1.update()
        progress_call1.finish()
    
    call1_results = {r["id"]: r for r in read_jsonl(str(call1_out))}
    
    # ========================================================
    # CALL 2
    # ========================================================
    print(f"\n{'='*60}")
    if args.oracle_level in ["e2e", "oracle1"]:
        print("CALL 2: Conflict Analysis")
        call2_task = "conflict"
    else:  # oracle2, oracle3
        print("CALL 2: Final Answer Generation")
        call2_task = "answer"
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
                
                # Build user message based on oracle level
                if args.oracle_level == "e2e":
                    # Call2 for E2E: conflict analysis using Call1 adjudications
                    s1_adjudications = call1_data["parsed"]
                    user_msg = call2_usr_template.format(
                        query=ex.get("query", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        s1_adjudications_json=json.dumps(s1_adjudications, ensure_ascii=False, indent=2),
                    )
                elif args.oracle_level == "oracle1":
                    # Call2 for Oracle1: conflict reasoning with GIVEN conflict_type
                    s1_adjudications = call1_data["parsed"]
                    user_msg = call2_usr_template.format(
                        query=ex.get("query", ""),
                        conflict_type=ex.get("conflict_type", ""),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        s1_adjudications_json=json.dumps(s1_adjudications, ensure_ascii=False, indent=2),
                    )
                elif args.oracle_level == "oracle2":
                    # Call2 for Oracle2: final answer using Call1 conflict + GIVEN per_doc_notes
                    s2_conflict = call1_data["parsed"]  # Call1 was conflict analysis for oracle2
                    user_msg = call2_usr_template.format(
                        query=ex.get("query", ""),
                        s2_conflict_json=json.dumps(s2_conflict, ensure_ascii=False, indent=2),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        per_doc_notes_json=json.dumps(ex.get("per_doc_notes", []), ensure_ascii=False, indent=2),
                    )
                else:  # oracle3
                    # Call2 for Oracle3: final answer using Call1 conflict + GIVEN per_doc_notes
                    s2_conflict = call1_data["parsed"]  # Call1 was conflict analysis for oracle3
                    user_msg = call2_usr_template.format(
                        query=ex.get("query", ""),
                        s2_conflict_json=json.dumps(s2_conflict, ensure_ascii=False, indent=2),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        per_doc_notes_json=json.dumps(ex.get("per_doc_notes", []), ensure_ascii=False, indent=2),
                    )
                
                prompt = build_prompt(tokenizer, call2_sys, user_msg)
                max_new = estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap, call2_task) if args.auto_length else (300 if call2_task == "conflict" else args.max_new_tokens_base)
                
                stopper_to_use = sentinel_stopper if call2_task == "answer" else None
                raw_output = generate_one(model, tokenizer, prompt, max_new, args.temperature, args.top_p, max_ctx, stopper=stopper_to_use)
                
                # Validate based on task
                if call2_task == "conflict":
                    parsed = parse_json_object(raw_output)
                    is_valid, err_msg = validate_conflict_analysis(parsed) if parsed else (False, "parse_failed")
                else:  # answer
                    parsed = None
                    is_valid = SENTINEL in raw_output
                    err_msg = None if is_valid else "no_sentinel"
                
                f2.write(json.dumps({"id": ex_id, "raw": raw_output, "parsed": parsed, "valid": is_valid, "error": err_msg}, ensure_ascii=False) + "\n")
                
                if (idx + 1) % args.save_every == 0:
                    f2.flush()
                progress_call2.update()
        progress_call2.finish()
    
    call2_results = {r["id"]: r for r in read_jsonl(str(call2_out))}
    
    # ========================================================
    # CALL 3 (only for E2E and Oracle1)
    # ========================================================
    if has_call3:
        print(f"\n{'='*60}")
        print("CALL 3: Final Answer Generation")
        print(f"{'='*60}")
        
        items_call3_candidates = [
            ex for ex in items
            if (ex.get("id") in call1_results and call1_results[ex["id"]].get("valid", False) and
                ex.get("id") in call2_results and call2_results[ex["id"]].get("valid", False))
        ]
        already_done_call3 = load_already_generated(call3_out) if args.resume else set()
        items_call3 = [ex for ex in items_call3_candidates if ex.get("id") not in already_done_call3]
        print(f"Valid from Call1+2: {len(items_call3_candidates)}, Already done: {len(already_done_call3)}, To generate: {len(items_call3)}")
        
        if items_call3:
            progress_call3 = ProgressTracker(len(items_call3), "Call3")
            file_mode = "a" if (args.resume and call3_out.exists()) else "w"
            
            with open(call3_out, file_mode, encoding="utf-8") as f3:
                for idx, ex in enumerate(items_call3):
                    ex_id = ex.get("id")
                    n_docs = estimate_doc_count(ex)
                    call1_data = call1_results[ex_id]
                    call2_data = call2_results[ex_id]
                    s1_adjudications = call1_data["parsed"]
                    s2_conflict = call2_data["parsed"]
                    
                    user_msg = call3_usr_template.format(
                        query=ex.get("query", ""),
                        s2_conflict_json=json.dumps(s2_conflict, ensure_ascii=False, indent=2),
                        retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
                        s1_adjudications_json=json.dumps(s1_adjudications, ensure_ascii=False, indent=2),
                    )
                    
                    prompt = build_prompt(tokenizer, call3_sys, user_msg)
                    max_new = estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap, "answer") if args.auto_length else args.max_new_tokens_base
                    raw_output = generate_one(model, tokenizer, prompt, max_new, args.temperature, args.top_p, max_ctx, stopper=sentinel_stopper)
                    has_sentinel = SENTINEL in raw_output
                    
                    f3.write(json.dumps({"id": ex_id, "raw": raw_output, "has_sentinel": has_sentinel}, ensure_ascii=False) + "\n")
                    
                    if (idx + 1) % args.save_every == 0:
                        f3.flush()
                    progress_call3.update()
            progress_call3.finish()
    
    # ========================================================
    # Summary
    # ========================================================
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Outputs: {output_dir}")
    
    call1_total = len(list(read_jsonl(str(call1_out))))
    call1_valid = sum(1 for r in read_jsonl(str(call1_out)) if r.get("valid", False))
    print(f"  Call 1: {call1_valid}/{call1_total} valid")
    
    call2_total = len(list(read_jsonl(str(call2_out))))
    call2_valid = sum(1 for r in read_jsonl(str(call2_out)) if r.get("valid", False))
    print(f"  Call 2: {call2_valid}/{call2_total} valid")
    
    if has_call3:
        call3_total = len(list(read_jsonl(str(call3_out))))
        call3_sentinel = sum(1 for r in read_jsonl(str(call3_out)) if r.get("has_sentinel", False))
        print(f"  Call 3: {call3_sentinel}/{call3_total} with sentinel")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()