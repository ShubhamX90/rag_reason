#!/usr/bin/env python3
"""
TEXT-MODE baseline generator (v6, enhanced with resume + progress)
-------------------------------------------------------------------
Features:
- Resume from checkpoint (skip already generated IDs)
- Progress bar with ETA
- Periodic saving (every N examples)
- All original v6 features (sentinel stopping, auto-length, 4-bit, etc.)

Usage:
  python generate_textmode_baseline_v6_enhanced.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --input_jsonl data/splits/test_v5_oracle1_messages.jsonl \
    --system_prompt_path prompts/monolithic/system_monolithic_oracle1.txt \
    --out_jsonl outputs/baseline_oracle1_test.raw.jsonl \
    --auto_length \
    --load_in_4bit \
    --save_every 50  # Save checkpoint every 50 examples
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

SENTINEL = "[[END-OF-ANSWER]]"


# ---------------- IO ----------------
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
        return f.read()


def load_already_generated(out_path: Path) -> set:
    """Load IDs that were already generated (for resume)."""
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
    except Exception as e:
        print(f"[WARN] Could not load existing outputs: {e}")
        return set()
    
    return generated_ids


# ---------------- Prompt helpers ----------------
def _extract_user_content(msgs: List[Dict[str, str]]) -> str:
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def build_messages_from_final(
    system_txt: str,
    ex: Dict[str, Any],
) -> List[Dict[str, str]]:
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or not msgs:
        user_txt = ""
    else:
        user_txt = _extract_user_content(msgs)

    return [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": user_txt},
    ]


def estimate_doc_count_from_user(user_txt: str) -> int:
    return max(0, user_txt.count('"doc_id"'))


# ---------------- Stopping on sentinel ----------------
class SentinelStopper(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, sentinel: str):
        super().__init__()
        self.sentinel_ids = tokenizer.encode(sentinel, add_special_tokens=False)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        if input_ids.shape[0] == 0 or len(self.sentinel_ids) == 0:
            return False
        seq = input_ids[0].tolist()
        n = len(self.sentinel_ids)
        return len(seq) >= n and seq[-n:] == self.sentinel_ids


# ---------------- Length heuristic ----------------
def estimate_max_new_tokens(n_docs: int, base: int, cap: int) -> int:
    est = int(240 + 70 * max(1, n_docs))
    if n_docs >= 12:
        est = max(est, 1600)
    if n_docs >= 16:
        est = max(est, 2000)
    return max(base, min(est, cap))


# ---------------- Progress tracking ----------------
class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
    
    def update(self, increment: int = 1):
        self.completed += increment
        current_time = time.time()
        
        # Print every 5 seconds or every 10 items
        if (current_time - self.last_print_time >= 5.0) or (self.completed % 10 == 0):
            self._print_progress()
            self.last_print_time = current_time
    
    def _print_progress(self):
        elapsed = time.time() - self.start_time
        pct = 100.0 * self.completed / max(1, self.total)
        
        if self.completed > 0:
            avg_time_per_item = elapsed / self.completed
            remaining_items = self.total - self.completed
            eta_seconds = avg_time_per_item * remaining_items
            eta_mins = eta_seconds / 60.0
            
            print(f"[Progress] {self.completed}/{self.total} ({pct:.1f}%) | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_mins:.1f}m | "
                  f"Speed: {avg_time_per_item:.1f}s/example")
        else:
            print(f"[Progress] {self.completed}/{self.total} ({pct:.1f}%)")
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\n[Complete] {self.completed}/{self.total} in {elapsed/60:.1f} minutes "
              f"({elapsed/max(1, self.completed):.1f}s per example)")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--system_prompt_path", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--attn_impl", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--auto_length", action="store_true")
    ap.add_argument("--max_new_tokens_base", type=int, default=1200)
    ap.add_argument("--max_new_tokens_cap", type=int, default=2200)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    
    # Enhanced features
    ap.add_argument("--save_every", type=int, default=50,
                    help="Save checkpoint every N examples (0 = only at end)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing output file (skip already generated IDs)")

    args = ap.parse_args()

    # ---------------- Load data ----------------
    items_all = list(read_jsonl(args.input_jsonl))
    if args.limit > 0:
        items = items_all[: args.limit]
    else:
        items = items_all

    # ---------------- Resume logic ----------------
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    already_generated = set()
    if args.resume and out_path.exists():
        already_generated = load_already_generated(out_path)
        print(f"[Resume] Found {len(already_generated)} already generated IDs")
    
    # Filter out already generated
    items_to_generate = [ex for ex in items if ex.get("id") not in already_generated]
    
    if not items_to_generate:
        print("[Done] All examples already generated!")
        return
    
    print(f"[Info] Total: {len(items)} | Already done: {len(already_generated)} | "
          f"To generate: {len(items_to_generate)}")

    # ---------------- Preflight check ----------------
    with_msgs = sum(
        1 for x in items_to_generate
        if isinstance(x.get("messages"), list)
        and any(m.get("role") == "user" for m in x["messages"])
    )
    
    if with_msgs == 0:
        print("[FATAL] No usable messages with user role")
        return

    # ---------------- Load model ----------------
    print(f"[Loading] Tokenizer from {args.base_model}")
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    use_int4 = args.load_in_4bit or (os.environ.get("GEN_INT4", "0") == "1")
    quant_cfg = None
    if use_int4:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("[Loading] Model in 4-bit mode")

    print(f"[Loading] Model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_impl,
        quantization_config=quant_cfg,
        local_files_only=args.local_files_only,
    )
    model.eval()
    print("[Loading] Model loaded successfully")

    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    tok.model_max_length = max_ctx
    tok.truncation_side = "left"
    safety = 32

    sys_txt = load_text(args.system_prompt_path)
    stopper = StoppingCriteriaList([SentinelStopper(tok, SENTINEL)])

    # ---------------- Generation loop with progress ----------------
    # Open in append mode if resuming, write mode otherwise
    file_mode = "a" if (args.resume and out_path.exists()) else "w"
    
    progress = ProgressTracker(len(items_to_generate))
    
    with open(out_path, file_mode, encoding="utf-8") as wf:
        for idx, ex in enumerate(items_to_generate):
            cid = ex.get("id")
            msgs = ex.get("messages", [])
            user_txt = _extract_user_content(msgs) if isinstance(msgs, list) else ""

            chat_msgs = build_messages_from_final(sys_txt, ex)

            prompt = tok.apply_chat_template(
                chat_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

            n_docs = estimate_doc_count_from_user(user_txt)
            if args.auto_length:
                max_new = estimate_max_new_tokens(
                    n_docs,
                    args.max_new_tokens_base,
                    args.max_new_tokens_cap,
                )
            else:
                max_new = args.max_new_tokens_base

            max_inp = max(512, int(max_ctx) - int(max_new) - safety)
            inputs = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_inp,
            ).to(model.device)

            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=(args.temperature > 0.0),
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                stopping_criteria=stopper,
            )
            out_text = tok.decode(
                gen[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )

            wf.write(
                json.dumps({"id": cid, "raw": out_text}, ensure_ascii=False) + "\n"
            )
            
            # Periodic flush for safety
            if args.save_every > 0 and (idx + 1) % args.save_every == 0:
                wf.flush()
                print(f"[Checkpoint] Saved at {idx + 1} examples")
            
            progress.update()
    
    progress.finish()
    print(f"[Complete] Wrote → {out_path}")


if __name__ == "__main__":
    main()