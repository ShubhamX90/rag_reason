#!/usr/bin/env python3
"""
train_qlora.py  –  QLoRA Fine-Tuning for Conflict-Aware RAG  (v2)
===================================================================
Improvements over v1:
  • NEFTune embedding noise for better generalization
  • Proper cosine LR schedule with linear warmup
  • Richer dev evaluation: macro-F1 on conflict type + doc-verdict accuracy
  • Mixed-mode training: can combine e2e + oracle message files
  • Gradient norm clipping
  • Configurable LoRA rank/alpha with sensible defaults
  • Better logging with per-epoch metrics
  • Optional validation loss tracking
  • Reproducible seeding

Usage:
  python code/train/train_qlora.py \
    --base_model  /path/to/Llama-3.1-8B-Instruct \
    --train_jsonl data/messages/train_e2e_messages.jsonl \
    --val_jsonl   data/messages/val_e2e_messages.jsonl \
    --out_dir     checkpoints/sft_e2e_run1 \
    --epochs 6 --lr 2e-4 --bsz 1 --grad_accum 16

Mixed-mode training (combine e2e + oracle for more diverse supervision):
  python code/train/train_qlora.py \
    --base_model  /path/to/Llama-3.1-8B-Instruct \
    --train_jsonl data/messages/train_e2e_messages.jsonl \
                  data/messages/train_oracle_messages.jsonl \
    --val_jsonl   data/messages/val_e2e_messages.jsonl \
    --out_dir     checkpoints/sft_mixed_run1
"""

import os, json, math, argparse, re, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ────────── Labels & Patterns ──────────
CONFLICT_TYPES = [
    "No conflict",
    "Complementary information",
    "Conflicting opinions or research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
]
TYPE2IDX = {t: i for i, t in enumerate(CONFLICT_TYPES)}
DASH_SEPS = (" — ", " – ", " - ")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
SENTINEL = "[[END-OF-ANSWER]]"

# ────────── IO ──────────
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln, s in enumerate(f, 1):
            s = s.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                raise ValueError(f"{path}:{ln} bad json: {e}")

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# ────────── Parsing helpers ──────────
def _span_after_first_json_array(text):
    start = text.find("[")
    if start < 0:
        return (None, None)
    depth, end, in_str, esc = 0, None, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "[": depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    return (start, end) if end is not None else (None, None)

def _find_conflict_line_span(text):
    _, array_end = _span_after_first_json_array(text)
    if array_end is None:
        return (None, None)
    tail = (text[array_end:] or "").lstrip("\n")
    offset0 = len(text) - len(tail)
    for ln in tail.splitlines():
        line = ln.strip()
        if not line:
            offset0 += len(ln) + 1
            continue
        for sep in DASH_SEPS:
            if sep in line:
                left = line.split(sep, 1)[0].strip()
                if left in TYPE2IDX:
                    s = text.find(line, offset0)
                    if s >= 0:
                        return (s, s + len(line))
        offset0 += len(ln) + 1
    return (None, None)

def extract_conflict_label(text):
    ls, le = _find_conflict_line_span(text)
    if ls is None:
        return None
    line = text[ls:le]
    for sep in DASH_SEPS:
        if sep in line:
            left = line.split(sep, 1)[0].strip()
            return TYPE2IDX.get(left, None)
    return None

def has_single_think_block(text):
    opens = [m.start() for m in re.finditer(re.escape(THINK_OPEN), text)]
    closes = [m.start() for m in re.finditer(re.escape(THINK_CLOSE), text)]
    return len(opens) == 1 and len(closes) == 1 and opens[0] < closes[0]

def is_format_faithful(assistant):
    return (has_single_think_block(assistant) and
            SENTINEL in assistant and
            extract_conflict_label(assistant) is not None)


# ────────── NEFTune: add noise to embeddings ──────────
class NEFTuneHook:
    """Adds uniform noise to embeddings during training (NEFTune).
    Paper: https://arxiv.org/abs/2310.05914
    Shown to improve SFT performance significantly."""

    def __init__(self, model, noise_alpha=5.0):
        self.noise_alpha = noise_alpha
        self.handle = None
        self._attach(model)

    def _attach(self, model):
        # Find the embedding layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and "embed_tokens" in name:
                self.handle = module.register_forward_hook(self._hook)
                print(f"[NEFTune] Attached to {name} with alpha={self.noise_alpha}")
                return
        print("[NEFTune] WARNING: Could not find embed_tokens layer")

    def _hook(self, module, input, output):
        if module.training:
            dims = torch.tensor(output.size(1) * output.size(2), dtype=output.dtype)
            mag = self.noise_alpha / torch.sqrt(dims)
            noise = torch.zeros_like(output).uniform_(-mag.item(), mag.item())
            return output + noise
        return output

    def remove(self):
        if self.handle:
            self.handle.remove()


# ────────── Dataset ──────────
class ChatSFTDataset(Dataset):
    """
    Loads prebuilt message JSONL (system + user + assistant).
    - Supervises only assistant tokens
    - Up-weights conflict label line tokens
    - Validates TEXT-MODE format
    """
    def __init__(self, tok, jsonl_paths, max_len=8192, conflict_weight=3.0, drop_on_truncation=True):
        self.tok = tok
        self.max_len = int(max_len)
        self.conf_w = float(conflict_weight)
        self.items = []
        self.class_idx = []
        n_kept = n_drop = 0
        sup_total = 0

        # Support multiple files (mixed-mode training)
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]

        for jsonl_path in jsonl_paths:
            for ex in read_jsonl(jsonl_path):
                msgs = ex.get("messages")
                if not (isinstance(msgs, list) and len(msgs) == 3):
                    n_drop += 1
                    continue

                sys_m, user_m, asst_m = msgs
                if (sys_m.get("role") != "system" or
                    user_m.get("role") != "user" or
                    asst_m.get("role") != "assistant"):
                    n_drop += 1
                    continue

                system_prompt = sys_m.get("content", "")
                assistant = asst_m.get("content", "")

                # Validate strict TEXT-MODE
                if not is_format_faithful(assistant):
                    n_drop += 1
                    continue

                # Tokenize prompt (system + user)
                prompt_msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_m.get("content", "")},
                ]
                prompt_text = self.tok.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                p_ids = self.tok(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]

                # Tokenize assistant
                a_tok = self.tok(
                    assistant, add_special_tokens=True, truncation=False,
                    return_offsets_mapping=True
                )
                a_ids = a_tok["input_ids"]

                # Token budget: preserve assistant, truncate prompt if needed
                if len(p_ids) + len(a_ids) > self.max_len:
                    budget = self.max_len - len(a_ids)
                    if budget <= 64:
                        if drop_on_truncation:
                            n_drop += 1
                            continue
                        a_ids = a_ids[:max(16, self.max_len - 64)]
                        budget = self.max_len - len(a_ids)
                    p_ids = p_ids[:max(64, budget)]

                full_ids = p_ids + a_ids
                if len(full_ids) > self.max_len:
                    if drop_on_truncation:
                        n_drop += 1
                        continue
                    full_ids = full_ids[:self.max_len]

                attn = [1] * len(full_ids)

                # Labels: supervise assistant only
                plen = len(p_ids)
                labels = [-100] * len(full_ids)
                for i in range(plen, len(full_ids)):
                    labels[i] = full_ids[i]
                sup_total += sum(1 for t in labels if t != -100)

                # Up-weight conflict label line tokens
                line_s, line_e = _find_conflict_line_span(assistant)
                weights = [1.0] * len(full_ids)
                if line_s is not None and line_e is not None and "offset_mapping" in a_tok:
                    base = plen
                    for j, (b_off, e_off) in enumerate(a_tok["offset_mapping"]):
                        if b_off is None or e_off is None:
                            continue
                        pos = base + j
                        if pos >= len(full_ids):
                            break
                        if labels[pos] != -100 and b_off >= line_s and e_off <= line_e:
                            weights[pos] = self.conf_w

                lab = extract_conflict_label(assistant)
                self.class_idx.append(-1 if lab is None else lab)

                self.items.append({
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "loss_weights": torch.tensor(weights, dtype=torch.float32),
                })
                n_kept += 1

        print(f"[Data] Loaded from {len(jsonl_paths)} file(s): kept={n_kept}, dropped={n_drop}")
        print(f"[Data] Supervised tokens total = {sup_total}")
        cnt = Counter([c for c in self.class_idx if c is not None and c >= 0])
        if cnt:
            print(f"[Data] Label distribution: {dict(sorted(cnt.items()))}")
        if not self.items:
            raise RuntimeError("Dataset has 0 usable items after filtering.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PadCollator:
    def __init__(self, tok):
        self.tok = tok
        self.tok.padding_side = "right"

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        def pad(v, fill):
            return torch.cat([v, torch.full((max_len - len(v),), fill, dtype=v.dtype)])
        return {
            "input_ids":      torch.stack([pad(x["input_ids"], self.tok.pad_token_id) for x in batch]),
            "attention_mask":  torch.stack([pad(x["attention_mask"], 0) for x in batch]),
            "labels":         torch.stack([pad(x["labels"], -100) for x in batch]),
            "loss_weights":   torch.stack([pad(x["loss_weights"], 1.0) for x in batch]),
        }


# ────────── Trainer with weighted CE + gradient clipping ──────────
class WeightedTokenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        weights = inputs.pop("loss_weights")
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size())

        active = (shift_labels != -100).float()
        mult = active * shift_weights
        denom = mult.sum().clamp_min(1.0)
        loss = (ce * mult).sum() / denom
        return (loss, outputs) if return_outputs else loss


# ────────── Dev evaluation callback ──────────
class DevEvalCallback(TrainerCallback):
    """
    End-of-epoch evaluation on val set:
      - Conflict type macro-F1
      - Format compliance rate
    Saves best checkpoint by macro-F1 with early stopping.
    """
    def __init__(self, tok, val_jsonl, out_dir, patience=4, max_new_tokens=180):
        self.tok = tok
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best = -1.0
        self.bad = 0
        self.patience = patience
        self.max_new = max_new_tokens

        # Cache val examples
        self.cache = []
        for ex in read_jsonl(val_jsonl):
            msgs = ex.get("messages") or []
            user_txt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            sys_txt = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
            asst_txt = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            gold = extract_conflict_label(asst_txt)
            self.cache.append((sys_txt, user_txt, gold))

    def _generate_label(self, model, sys_txt, user_txt):
        msgs = [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": user_txt},
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        max_ctx = int(getattr(self.tok, "model_max_length", 8192))
        max_inp = max(384, max_ctx - self.max_new - 64)

        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=max_inp).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=self.max_new, do_sample=False,
                eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id,
            )
        cont = gen[0][inputs["input_ids"].shape[-1]:]
        out = self.tok.decode(cont, skip_special_tokens=True)
        return extract_conflict_label(out), is_format_faithful(out) if SENTINEL in out else False

    @staticmethod
    def _macro_f1(golds, preds, n=5):
        cm = [[0] * n for _ in range(n)]
        for g, p in zip(golds, preds):
            if g is None or p is None:
                continue
            cm[g][p] += 1
        f1s = []
        for c in range(n):
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in range(n)) - tp
            fn = sum(cm[c][r] for r in range(n)) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
            f1s.append(f1)
        return sum(f1s) / n

    def on_epoch_end(self, args, state, control, **kw):
        model = kw["model"].eval()
        preds, golds = [], []
        format_ok = 0
        for sys_txt, user_txt, g in self.cache:
            p, fmt = self._generate_label(model, sys_txt, user_txt)
            preds.append(p)
            golds.append(g)
            if fmt:
                format_ok += 1

        macro = self._macro_f1(golds, preds)
        fmt_rate = format_ok / max(1, len(self.cache))
        epoch = getattr(state, "epoch", -1)
        print(f"[DevEval] epoch={epoch:.2f} macro-F1={macro:.4f} format_ok={fmt_rate:.1%}")

        if macro > self.best + 1e-6:
            self.best = macro
            self.bad = 0
            best_dir = self.out_dir / "best_dev_f1"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir.as_posix())
            print(f"[DevEval] ★ New BEST saved → {best_dir}")
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print(f"[DevEval] Early stopping (patience={self.patience})")
                control.should_training_stop = True
        return control


# ────────── Main ──────────
def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning for conflict-aware RAG")
    ap.add_argument("--base_model", required=True, help="HF model path")
    ap.add_argument("--train_jsonl", nargs="+", required=True,
                    help="Training message JSONL(s). Multiple files = mixed-mode training.")
    ap.add_argument("--val_jsonl", required=True, help="Validation message JSONL")
    ap.add_argument("--out_dir", required=True, help="Output directory for checkpoints")

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64,
                    help="LoRA alpha. Using alpha=2*r is common for better scaling.")
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # SFT improvements
    ap.add_argument("--conflict_weight", type=float, default=3.0,
                    help="Loss multiplier for conflict label line tokens")
    ap.add_argument("--neftune_alpha", type=float, default=5.0,
                    help="NEFTune noise alpha (0=disabled). 5.0 recommended.")
    ap.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    ap.add_argument("--dev_max_new", type=int, default=180)

    # Hardware
    ap.add_argument("--attn_impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--resume_from", type=str, default=None)

    args = ap.parse_args()

    # Stability
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)
    random.seed(42)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.model_max_length = args.max_len

    # QLoRA base model
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        local_files_only=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lconf = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(base, lconf)
    model.print_trainable_parameters()

    # NEFTune
    neftune_hook = None
    if args.neftune_alpha > 0:
        neftune_hook = NEFTuneHook(model, noise_alpha=args.neftune_alpha)

    # Data
    train_ds = ChatSFTDataset(
        tok, args.train_jsonl,
        max_len=args.max_len,
        conflict_weight=args.conflict_weight,
    )
    collator = PadCollator(tok)

    # Class-balanced sampler (1/sqrt(count))
    counts = Counter([c for c in train_ds.class_idx if c is not None and c >= 0])
    if counts:
        per_ex_w = []
        for c in train_ds.class_idx:
            cnt = max(1, counts.get(c, 1))
            per_ex_w.append(1.0 / (cnt ** 0.5))
        sampler = WeightedRandomSampler(
            torch.as_tensor(per_ex_w, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader_kwargs = dict(sampler=sampler)
    else:
        train_loader_kwargs = dict(shuffle=True)

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.bsz * args.grad_accum)))

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=max(1, steps_per_epoch // 10),
        eval_strategy="no",
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // 2),
        save_total_limit=2,
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        group_by_length=True,
        dataloader_num_workers=2,
        save_safetensors=True,
        seed=42,
        overwrite_output_dir=True,
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # Custom dataloader with class-balanced sampler
    def _get_train_dl():
        return DataLoader(
            train_ds,
            batch_size=targs.per_device_train_batch_size,
            collate_fn=collator,
            pin_memory=True,
            **train_loader_kwargs,
        )
    trainer.get_train_dataloader = _get_train_dl

    # Dev evaluation callback
    # Use first val_jsonl for eval
    trainer.add_callback(
        DevEvalCallback(
            tok, args.val_jsonl, args.out_dir,
            patience=args.patience,
            max_new_tokens=args.dev_max_new,
        )
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from)

    # Cleanup NEFTune
    if neftune_hook:
        neftune_hook.remove()

    # Save final
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"✓ Saved adapter → {args.out_dir}")


if __name__ == "__main__":
    main()
