#!/usr/bin/env python3
"""
generate.py  –  Unified Generation for SFT and Baseline Models (v2)
====================================================================
Single script for both fine-tuned (LoRA) and untuned (baseline) inference.
Uses the SAME prompts as training for consistency.

Modes:
  • SFT inference:      --lora_dir checkpoints/sft_e2e_run1/best_dev_f1
  • Baseline inference:  (omit --lora_dir)

Usage:
  # SFT model
  python code/eval/generate.py \
    --base_model /path/to/Qwen-8B-Instruct \
    --lora_dir   checkpoints/qwen_stagewise_e2e_run1/best_dev_f1 \
    --input_jsonl data/messages/val_stagewise_e2e_messages.jsonl \
    --out_jsonl   outputs/sft_qwen_e2e_val_stagewise.raw.jsonl \
    --auto_length --dtype bf16

  # Baseline (untuned) model
  python code/eval/generate.py \
    --base_model /path/to/Qwen-8B-Instruct \
    --input_jsonl data/messages/val_stagewise_oracle_conflict_messages.jsonl \
    --out_jsonl   outputs/baseline_qwen_oracle_conflict_val_stagewise.raw.jsonl \
    --auto_length --load_in_4bit
"""

import os, json, argparse, time, importlib
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    StoppingCriteria, StoppingCriteriaList,
)
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # Older Transformers builds do not expose this auto class.
    AutoModelForImageTextToText = None

SENTINEL = "[[END-OF-ANSWER]]"
CONFLICT_TYPES = {
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}


# ────────── IO ──────────
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

def load_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _content_text(msg):
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _as_text_part_messages(msgs):
    converted = []
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({**msg, "content": content})
    return converted


def _manual_inst_chat_template(tok, msgs, *, tokenize=False, add_generation_prompt=True):
    """Fallback for local model repos that omit tokenizer.chat_template."""
    system_txt = ""
    user_parts = []
    for msg in msgs:
        role = msg.get("role")
        text = _content_text(msg).strip()
        if not text:
            continue
        if role == "system":
            system_txt = text
        elif role == "user":
            user_parts.append(text)
        elif role == "assistant":
            user_parts.append(text)
    prompt_body = "\n\n".join([p for p in [system_txt, "\n\n".join(user_parts)] if p]).strip()
    bos = tok.bos_token or ""
    rendered = f"{bos}[INST] {prompt_body} [/INST]"
    if add_generation_prompt:
        rendered += " "
    if tokenize:
        return tok(rendered, add_special_tokens=False)["input_ids"]
    return rendered


def apply_chat_template_compat(tok, msgs, *, tokenize=False, add_generation_prompt=True):
    attempts = [
        ("native", msgs),
        ("text-parts", _as_text_part_messages(msgs)),
    ]
    if len(msgs) == 2 and msgs[0].get("role") == "system" and msgs[1].get("role") == "user":
        folded = [
            {
                "role": "user",
                "content": (_content_text(msgs[0]).strip() + "\n\n" + _content_text(msgs[1])).strip(),
            }
        ]
        attempts.extend([
            ("folded-system-user", folded),
            ("folded-system-user-text-parts", _as_text_part_messages(folded)),
        ])

    errors = []
    for mode, candidate in attempts:
        try:
            if mode != "native":
                print(f"[Tokenizer] Using chat-template compatibility mode: {mode}")
            return tok.apply_chat_template(
                candidate,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:
            errors.append(f"{mode}: {exc}")
    print(
        "[Tokenizer] tokenizer.chat_template unavailable or incompatible; "
        "using manual [INST] fallback. Errors: " + " | ".join(errors)
    )
    return _manual_inst_chat_template(
        tok,
        msgs,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
    )


def _load_mistral_common_tokenizer(model_path, *, local_files_only=True):
    candidates = []
    for module_name in (
        "transformers",
        "transformers.models.mistral3",
        "transformers.models.mistral3.tokenization_mistral_common",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr_name in ("MistralCommonTokenizer", "MistralCommonTokenizerFast", "MistralCommonBackend"):
            candidate = getattr(module, attr_name, None)
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)

    errors = []
    for candidate in candidates:
        for kwargs in (
            {"local_files_only": local_files_only},
            {},
        ):
            try:
                return candidate.from_pretrained(model_path, **kwargs)
            except Exception as exc:
                errors.append(f"{candidate.__name__}({kwargs}): {exc}")

    if candidates:
        raise RuntimeError("failed to load mistral common tokenizer: " + " | ".join(errors))
    raise RuntimeError("mistral3 detected but no Mistral common tokenizer class is importable in this transformers build")


def load_tokenizer_compat(model_path, *, use_fast=True, local_files_only=True):
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if getattr(config, "model_type", None) == "mistral3":
        print("[Tokenizer] Using Mistral common tokenizer path for mistral3 compatibility")
        return _load_mistral_common_tokenizer(model_path, local_files_only=local_files_only)

    kwargs = dict(
        use_fast=use_fast,
        local_files_only=local_files_only,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(model_path, **kwargs)


def pick_inference_dtype(dtype_name: str):
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return mapping[dtype_name]


def _from_pretrained_with_compat(loader, base_model, kwargs, loader_name):
    try:
        model = loader.from_pretrained(base_model, **kwargs)
        print(f"[Load] Loaded with {loader_name}")
        return model
    except (TypeError, ValueError) as e:
        err = str(e).lower()
        if "dtype" in kwargs and "dtype" in err:
            retry_kwargs = dict(kwargs)
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
            model = loader.from_pretrained(base_model, **retry_kwargs)
            print(f"[Load] Loaded with {loader_name} using torch_dtype compatibility fallback")
            return model
        if "attn_implementation" in kwargs and "attn" in err:
            print(f"[Load] attn_implementation={kwargs.get('attn_implementation')!r} is unsupported by {loader_name}; retrying without it.")
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("attn_implementation", None)
            model = loader.from_pretrained(base_model, **retry_kwargs)
            print(f"[Load] Loaded with {loader_name} without attn_implementation")
            return model
        raise


def load_causal_lm(base_model: str, torch_dtype, quant_cfg, attn_impl: Optional[str], local_files_only: bool):
    kwargs = dict(
        dtype=torch_dtype,
        device_map="auto",
        quantization_config=quant_cfg,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    try:
        return _from_pretrained_with_compat(
            AutoModelForCausalLM,
            base_model,
            kwargs,
            "AutoModelForCausalLM",
        )
    except (TypeError, ValueError, RuntimeError) as causal_exc:
        if AutoModelForImageTextToText is None:
            raise
        print(f"[Load] AutoModelForCausalLM failed; trying AutoModelForImageTextToText: {causal_exc}")
        return _from_pretrained_with_compat(
            AutoModelForImageTextToText,
            base_model,
            kwargs,
            "AutoModelForImageTextToText",
        )


# ────────── Prompt building ──────────
def extract_user_content(msgs):
    for m in msgs:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def extract_system_content(msgs):
    for m in msgs:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""

def build_chat_messages(ex, system_override=None):
    """Build [system, user] messages for generation from a messages JSONL example."""
    msgs = ex.get("messages", [])
    sys_txt = system_override or extract_system_content(msgs)
    user_txt = extract_user_content(msgs)
    return [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": user_txt},
    ]

def estimate_doc_count(user_txt):
    return max(0, user_txt.count('"doc_id"'))


# ────────── Sentinel stopping ──────────
class SentinelStopper(StoppingCriteria):
    def __init__(self, tokenizer, sentinel):
        super().__init__()
        self.sentinel_ids = tokenizer.encode(sentinel, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] == 0 or not self.sentinel_ids:
            return False
        seq = input_ids[0].tolist()
        n = len(self.sentinel_ids)
        return len(seq) >= n and seq[-n:] == self.sentinel_ids


# ────────── Length heuristic ──────────
def estimate_max_new_tokens(n_docs, base, cap):
    est = int(320 + 95 * max(1, n_docs))
    if n_docs >= 8:
        est += 160
    if n_docs >= 12:
        est += 320
    if n_docs >= 16:
        est += 480
    est = max(est, base)
    est = min(est, cap)
    return est


def has_single_think_block(text):
    return text.count("<think>") == 1 and text.count("</think>") == 1 and text.find("<think>") < text.find("</think>")


def extract_conflict_label(text):
    start = text.find("[")
    scan_text = text
    if start >= 0:
        probe = start + 1
        while probe < len(text) and text[probe].isspace():
            probe += 1
        if probe >= len(text) or text[probe] not in "{]":
            start = -1
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        end_idx = None
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
                        end_idx = i + 1
                        break
        if end_idx is not None:
            scan_text = text[end_idx:]
    for line in scan_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("conflict type:"):
            label = line.split(":", 1)[1].strip()
            if label in CONFLICT_TYPES:
                return label
        for sep in (" - ", " — ", " – "):
            if sep in line:
                label = line.split(sep, 1)[0].strip()
                if label in CONFLICT_TYPES:
                    return label
    return None


def is_minimal_contract_ok(text):
    return has_single_think_block(text) and SENTINEL in text and extract_conflict_label(text) is not None


def is_final_answer_contract_ok(text):
    text = (text or "").strip()
    return SENTINEL in text and bool(text.replace(SENTINEL, "").strip())


def is_generation_ok(text, contract_mode):
    if contract_mode == "none":
        return True
    if contract_mode == "final":
        return is_final_answer_contract_ok(text)
    return is_minimal_contract_ok(text)


# ────────── Resume support ──────────
def load_done_ids(out_path):
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "id" in obj:
                    done.add(obj["id"])
            except:
                pass
    return done


# ────────── Main ──────────
def main():
    ap = argparse.ArgumentParser(description="Unified generation for SFT and baseline models")
    ap.add_argument("--base_model", required=True, help="HF model path")
    ap.add_argument("--lora_dir", default=None,
                    help="LoRA adapter directory (omit for baseline inference)")
    ap.add_argument("--input_jsonl", required=True,
                    help="Message JSONL from prepare_data.py")
    ap.add_argument("--system_prompt_path", default=None,
                    help="Override system prompt (default: use prompt from messages)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--attn_impl", choices=["eager", "sdpa"], default="sdpa")

    ap.add_argument("--auto_length", action="store_true",
                    help="Adjust max_new_tokens based on doc count")
    ap.add_argument("--max_new_tokens_base", type=int, default=1200)
    ap.add_argument("--max_new_tokens_cap", type=int, default=2200)
    ap.add_argument("--retry_attempts", type=int, default=1,
                    help="Number of extra deterministic retries when the draft is structurally invalid.")
    ap.add_argument("--retry_scale", type=float, default=1.6,
                    help="Per-retry max_new_tokens multiplier.")
    ap.add_argument("--retry_max_new_cap", type=int, default=3200,
                    help="Upper bound on max_new_tokens across retries.")
    ap.add_argument("--contract_mode", choices=["trace", "final", "none"], default="trace",
                    help="Retry success criterion: full trace, final-answer-only, or no contract check.")

    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="Ignored when --load_in_4bit is used.")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-generated IDs and append")

    args = ap.parse_args()

    # Determine mode
    is_sft = args.lora_dir is not None
    mode_str = "SFT" if is_sft else "BASELINE"
    print(f"[Mode] {mode_str} inference")
    if is_sft and not Path(args.lora_dir).is_dir():
        raise FileNotFoundError(
            f"LoRA adapter directory not found: {args.lora_dir}\n"
            "Expected a local adapter directory containing adapter_config.json."
        )

    # Load input
    items = list(read_jsonl(args.input_jsonl))
    if args.limit > 0:
        items = items[:args.limit]
    print(f"[Data] {len(items)} examples from {args.input_jsonl}")

    # System prompt override
    sys_override = None
    if args.system_prompt_path:
        sys_override = load_text(args.system_prompt_path)

    # Tokenizer
    tok = load_tokenizer_compat(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Model loading
    quant_cfg = None
    torch_dtype = pick_inference_dtype(args.dtype)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        )
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"[Load] dtype={str(torch_dtype).replace('torch.', '')} quantized={'yes' if args.load_in_4bit else 'no'}")

    base = load_causal_lm(
        args.base_model,
        torch_dtype=torch_dtype,
        quant_cfg=quant_cfg,
        attn_impl=args.attn_impl,
        local_files_only=args.local_files_only,
    )

    if is_sft:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.lora_dir, is_trainable=False)
        print(f"[Model] Loaded LoRA adapter from {args.lora_dir}")
    else:
        model = base
        print(f"[Model] Using base model (no LoRA)")

    model.eval()

    # Context management
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)
    tok.model_max_length = max_ctx
    tok.truncation_side = "left"
    safety = 32

    # Sentinel stopper
    stopper = SentinelStopper(tok, SENTINEL)
    stops = StoppingCriteriaList([stopper])

    # Resume
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    file_mode = "w"
    if args.resume and out_path.exists():
        done_ids = load_done_ids(out_path)
        file_mode = "a"
        print(f"[Resume] {len(done_ids)} existing generations found")

    # Generate
    total = skipped = generated = 0
    t0 = time.time()

    with open(out_path, file_mode, encoding="utf-8") as wf:
        for i, ex in enumerate(items):
            cid = ex.get("id")
            total += 1

            if cid in done_ids:
                skipped += 1
                continue

            msgs = ex.get("messages", [])
            user_txt = extract_user_content(msgs)
            chat_msgs = build_chat_messages(ex, sys_override)
            prompt = apply_chat_template_compat(tok, chat_msgs, tokenize=False, add_generation_prompt=True)

            n_docs = estimate_doc_count(user_txt)
            max_new = (estimate_max_new_tokens(n_docs, args.max_new_tokens_base, args.max_new_tokens_cap)
                       if args.auto_length else args.max_new_tokens_base)

            model_device = next(model.parameters()).device
            out = ""
            for attempt in range(max(0, args.retry_attempts) + 1):
                attempt_max_new = max_new if attempt == 0 else min(
                    int(round(max_new * (args.retry_scale ** attempt))),
                    args.retry_max_new_cap,
                )
                max_inp = max(512, int(max_ctx) - int(attempt_max_new) - safety)
                inputs = tok(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_inp,
                    verbose=False,
                ).to(model_device)

                gen_kwargs = dict(
                    max_new_tokens=attempt_max_new,
                    do_sample=(args.temperature > 0.0),
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                    stopping_criteria=stops,
                )
                if args.temperature > 0.0:
                    gen_kwargs["temperature"] = args.temperature
                    gen_kwargs["top_p"] = args.top_p
                with torch.inference_mode():
                    gen = model.generate(**inputs, **gen_kwargs)
                out = tok.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                if is_generation_ok(out, args.contract_mode):
                    break
                if attempt < args.retry_attempts:
                    print(f"  [retry:{attempt + 1}] id={cid} max_new={attempt_max_new} contract_incomplete")
            wf.write(json.dumps({"id": cid, "raw": out}, ensure_ascii=False) + "\n")
            wf.flush()
            generated += 1

            if generated % 10 == 0:
                elapsed = time.time() - t0
                rate = generated / elapsed if elapsed > 0 else 0
                remaining = (len(items) - total) / rate if rate > 0 else 0
                print(f"  [{generated}/{len(items) - len(done_ids)}] "
                      f"{rate:.1f} ex/s, ~{remaining/60:.0f}m remaining")

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed/60:.1f}m → {out_path}")
    print(f"  total={total}, skipped={skipped}, generated={generated}")


if __name__ == "__main__":
    main()
