#!/usr/bin/env python3
"""
Simple RAG Baseline Generator
------------------------------
Single-call generation: query + docs → final answer directly.
No multi-stage reasoning, no structured outputs.

Usage:
  python generate_simple_rag.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --in_jsonl data/splits/test_v5.jsonl \
    --output_jsonl outputs/baselines/simple_rag_test.jsonl \
    --system_prompt prompts/baselines/system_simplerag.txt \
    --user_prompt prompts/baselines/user_simplerag.txt \
    --load_in_4bit \
    --resume
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

SENTINEL = "[[END-OF-ANSWER]]"


# ============================================================
# IO & Helper Functions
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
# Progress Tracker
# ============================================================

class ProgressTracker:
    def __init__(self, total: int, name: str = "Generation"):
        self.total = total
        self.completed = 0
        self.name = name
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
            print(f"[{self.name}] {self.completed}/{self.total} ({pct:.1f}%) | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_mins:.1f}m | Speed: {avg_time:.1f}s/ex")
        else:
            print(f"[{self.name}] {self.completed}/{self.total} ({pct:.1f}%)")
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"[{self.name} Complete] {self.completed}/{self.total} in {elapsed/60:.1f} minutes")


# ============================================================
# Generation Functions
# ============================================================

def estimate_doc_count(ex: dict) -> int:
    rd = ex.get("retrieved_docs", [])
    return len(rd) if isinstance(rd, list) else 0


def estimate_max_new_tokens(n_docs: int, base: int, cap: int) -> int:
    """Estimate tokens needed based on number of docs."""
    est = int(200 + 20 * n_docs)  # ~200 base + 20 per doc
    return max(base, min(est, cap))


def build_prompt(tokenizer, system_txt: str, user_txt: str) -> str:
    msgs = [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_one(model, tokenizer, prompt: str, max_new: int, temperature: float,
                 top_p: float, max_ctx: int, stopper) -> str:
    safety = 32
    max_inp = max(512, int(max_ctx) - int(max_new) - safety)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_inp).to(model.device)
    
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopper,
    )
    
    return tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


# ============================================================
# Main Pipeline
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    
    # Model
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--attn_impl", choices=["eager", "sdpa"], default="sdpa")
    
    # Data & prompts
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--system_prompt", required=True, help="Path to system prompt file")
    ap.add_argument("--user_prompt", required=True, help="Path to user prompt template file")
    
    # Generation params
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--auto_length", action="store_true", help="Auto-adjust max_new_tokens by doc count")
    ap.add_argument("--max_new_tokens_base", type=int, default=800)
    ap.add_argument("--max_new_tokens_cap", type=int, default=2000)
    
    # Control
    ap.add_argument("--limit", type=int, default=0, help="Limit to N examples (0=all)")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    
    args = ap.parse_args()
    
    # Setup output
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    items_all = list(read_jsonl(args.in_jsonl))
    items = items_all[:args.limit] if args.limit > 0 else items_all
    
    print(f"\n{'='*60}")
    print(f"SIMPLE RAG BASELINE GENERATION")
    print(f"{'='*60}")
    print(f"Model: {args.base_model}")
    print(f"Input: {args.in_jsonl}")
    print(f"Output: {args.output_jsonl}")
    print(f"Total examples: {len(items)}")
    print(f"{'='*60}\n")
    
    # Load prompts
    system_txt = load_text(args.system_prompt)
    user_template = load_text(args.user_prompt)
    
    print(f"[Setup] System prompt: {args.system_prompt}")
    print(f"[Setup] User prompt: {args.user_prompt}")
    print()
    
    # Load model
    print(f"[Loading] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    quant_cfg = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[Loading] Model in 4-bit mode...")
    
    print(f"[Loading] Model from {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_impl,
        quantization_config=quant_cfg,
        local_files_only=args.local_files_only,
    )
    model.eval()
    print("[Loading] Model loaded successfully\n")
    
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    tokenizer.model_max_length = max_ctx
    tokenizer.truncation_side = "left"
    sentinel_stopper = StoppingCriteriaList([SentinelStopper(tokenizer, SENTINEL)])
    
    # Resume capability
    already_done = load_already_generated(output_path) if args.resume else set()
    items_to_gen = [ex for ex in items if ex.get("id") not in already_done]
    
    print(f"[Generation] Already done: {len(already_done)}")
    print(f"[Generation] To generate: {len(items_to_gen)}")
    print()
    
    if not items_to_gen:
        print("✓ All examples already generated!")
        return
    
    # Generate
    progress = ProgressTracker(len(items_to_gen), "Simple RAG")
    file_mode = "a" if (args.resume and output_path.exists()) else "w"
    
    with open(output_path, file_mode, encoding="utf-8") as f:
        for idx, ex in enumerate(items_to_gen):
            ex_id = ex.get("id")
            n_docs = estimate_doc_count(ex)
            
            # Build user message
            user_msg = user_template.format(
                query=ex.get("query", ""),
                retrieved_docs_json=json.dumps(ex.get("retrieved_docs", []), ensure_ascii=False, indent=2),
            )
            
            # Build full prompt
            prompt = build_prompt(tokenizer, system_txt, user_msg)
            
            # Determine max_new_tokens
            if args.auto_length:
                max_new = estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap)
            else:
                max_new = args.max_new_tokens_base
            
            # Generate
            raw_output = generate_one(
                model, tokenizer, prompt, max_new,
                args.temperature, args.top_p, max_ctx, sentinel_stopper
            )
            
            # Check if has sentinel
            has_sentinel = SENTINEL in raw_output
            
            # Save
            f.write(json.dumps({
                "id": ex_id,
                "raw": raw_output,
                "has_sentinel": has_sentinel,
            }, ensure_ascii=False) + "\n")
            
            if (idx + 1) % args.save_every == 0:
                f.flush()
            
            progress.update()
    
    progress.finish()
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {args.output_jsonl}")
    
    # Count sentinel
    with_sentinel = sum(1 for r in read_jsonl(str(output_path)) if r.get('has_sentinel', False))
    total = sum(1 for _ in read_jsonl(str(output_path)))
    print(f"With sentinel: {with_sentinel}/{total} ({100*with_sentinel/total:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()