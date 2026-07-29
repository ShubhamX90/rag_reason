#!/usr/bin/env python3
"""
train_qlora.py  –  High-rigor QLoRA SFT for Conflict-Aware RAG
===============================================================

Key properties:
  • Assistant-only supervision on prebuilt chat message JSONL
  • Prompt/assistant tokenization aligned with instruct chat templates
  • Prompt-tail preservation for long-context samples
  • Realistic end-of-epoch generation-based dev evaluation
  • QLoRA + NEFTune + weighted conflict-label loss
  • Safer defaults for separate stagewise/monolithic training runs
"""

import os, json, math, argparse, re, random, inspect
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback,
    StoppingCriteria, StoppingCriteriaList,
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
ABSTAIN_ANSWER = "CANNOT ANSWER, INSUFFICIENT EVIDENCE"
CITATION_RE = re.compile(r"\[d\d+\]")
VERDICT_LABELS = ("supports", "partially supports", "irrelevant")
VERDICT_ALIASES = {
    "support": "supports",
    "supported": "supports",
    "partial": "partially supports",
    "partially_supports": "partially supports",
    "partially-supports": "partially supports",
}

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


def count_docs_in_user_message(text):
    return max(0, text.count('"doc_id"'))


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


def pick_compute_dtype():
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA fine-tuning requires CUDA. No CUDA device is available.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_dist_env():
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    return {
        "local_rank": local_rank,
        "world_size": world_size,
        "rank": rank,
        "is_distributed": world_size > 1,
        "is_main_process": rank == 0,
    }


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def maybe_destroy_process_group(dist_env):
    if dist_env["is_distributed"] and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_causal_lm(base_model, quantization_config, torch_dtype, attn_impl, local_files_only=True, device_map="auto"):
    kwargs = dict(
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    try:
        return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    except (TypeError, ValueError) as e:
        err = str(e).lower()
        if "dtype" in kwargs and "dtype" in err:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if "attn_implementation" in kwargs and "attn" in err:
            print(f"[Load] attn_implementation={attn_impl!r} is unsupported here; retrying without it.")
            kwargs.pop("attn_implementation", None)
            return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        raise


def select_lora_target_modules(model):
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    chosen = [name for name in preferred if name in available]
    if not chosen:
        raise RuntimeError(
            "Could not find any expected LoRA target modules. "
            "Expected a subset of q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj."
        )
    print(f"[LoRA] Target modules: {chosen}")
    return chosen


def training_args_with_compat(**kwargs):
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    unsupported = sorted(set(kwargs) - set(params))
    for key in unsupported:
        kwargs.pop(key, None)
    return TrainingArguments(**kwargs)


def write_run_metadata(out_dir, args, train_ds, compute_dtype, lora_targets, dist_env):
    if not dist_env["is_main_process"]:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_model": args.base_model,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "out_dir": args.out_dir,
        "training_args": vars(args),
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "lora_target_modules": lora_targets,
        "train_examples": len(train_ds),
        "label_distribution": dict(sorted(Counter(c for c in train_ds.class_idx if c is not None and c >= 0).items())),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

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
            if left in TYPE2IDX:
                return TYPE2IDX[left]
    return None

def has_single_think_block(text):
    opens = [m.start() for m in re.finditer(re.escape(THINK_OPEN), text)]
    closes = [m.start() for m in re.finditer(re.escape(THINK_CLOSE), text)]
    return len(opens) == 1 and len(closes) == 1 and opens[0] < closes[0]

def is_format_faithful(assistant):
    return (has_single_think_block(assistant) and
            SENTINEL in assistant and
            extract_conflict_label(assistant) is not None)


def normalize_verdict_label(verdict):
    verdict = re.sub(r"\s+", " ", str(verdict or "").strip().lower())
    verdict = VERDICT_ALIASES.get(verdict, verdict)
    return verdict if verdict in VERDICT_LABELS else None


def extract_doc_verdict_map(text):
    array_s, array_e = _span_after_first_json_array(text or "")
    if array_s is None or array_e is None:
        return {}
    try:
        arr = json.loads((text or "")[array_s:array_e])
    except Exception:
        return {}
    if not isinstance(arr, list):
        return {}
    verdicts = {}
    for idx, item in enumerate(arr, 1):
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            doc_id = f"d{idx}"
        verdict = normalize_verdict_label(item.get("verdict"))
        if verdict is not None:
            verdicts[doc_id] = verdict
    return verdicts


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


def stratified_subset(examples, subset_size, seed):
    if subset_size <= 0 or len(examples) <= subset_size:
        return examples
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for ex in examples:
        buckets[ex["gold"]].append(ex)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = sorted(buckets, key=lambda gold: (len(buckets[gold]), -1 if gold is None else gold))
    selected = []
    while len(selected) < subset_size:
        made_progress = False
        for gold in order:
            bucket = buckets[gold]
            if bucket and len(selected) < subset_size:
                selected.append(bucket.pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def cache_bucket_key(example):
    gold = example.get("gold")
    abstain = bool(example.get("gold_abstain"))
    return (gold, abstain)


def _find_literal_spans(text, literal):
    spans = []
    start = 0
    while literal:
        idx = text.find(literal, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(literal)))
        start = idx + len(literal)
    return spans


def _find_citation_spans(text):
    return [(m.start(), m.end()) for m in CITATION_RE.finditer(text or "")]


def _boost_char_span(weights, labels, offsets, global_s, global_e, value):
    if global_s is None or global_e is None or global_s >= global_e:
        return
    for pos, (b_off, e_off) in enumerate(offsets):
        if b_off is None or e_off is None or labels[pos] == -100:
            continue
        if b_off < global_e and e_off > global_s:
            weights[pos] = max(weights[pos], value)


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
    def __init__(
        self,
        tok,
        jsonl_paths,
        max_len=8192,
        conflict_weight=3.0,
        contract_weight=2.5,
        array_weight=1.35,
        citation_weight=1.75,
        drop_on_truncation=True,
    ):
        self.tok = tok
        self.max_len = int(max_len)
        self.conf_w = float(conflict_weight)
        self.contract_w = float(contract_weight)
        self.array_w = float(array_weight)
        self.citation_w = float(citation_weight)
        self.items = []
        self.class_idx = []
        self.drop_reasons = Counter()
        n_kept = n_drop = 0
        sup_total = 0

        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]

        for jsonl_path in jsonl_paths:
            for ex in read_jsonl(jsonl_path):
                msgs = ex.get("messages")
                if not (isinstance(msgs, list) and len(msgs) == 3):
                    self.drop_reasons["bad_messages_shape"] += 1
                    n_drop += 1
                    continue

                sys_m, user_m, asst_m = msgs
                if (sys_m.get("role") != "system" or
                    user_m.get("role") != "user" or
                    asst_m.get("role") != "assistant"):
                    self.drop_reasons["bad_message_roles"] += 1
                    n_drop += 1
                    continue

                system_prompt = sys_m.get("content", "")
                assistant = asst_m.get("content", "")

                # Validate strict TEXT-MODE
                if not is_format_faithful(assistant):
                    self.drop_reasons["assistant_not_format_faithful"] += 1
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
                full_text = prompt_text + assistant
                full_tok = self.tok(
                    full_text,
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                    verbose=False,
                )
                full_ids = full_tok["input_ids"]
                offsets = list(full_tok["offset_mapping"])
                assistant_char_start = len(prompt_text)

                assistant_tok_start = next(
                    (i for i, (_, e_off) in enumerate(offsets) if (e_off or 0) > assistant_char_start),
                    len(full_ids),
                )
                assistant_token_count = len(full_ids) - assistant_tok_start

                # Token budget: preserve assistant and keep the prompt tail.
                if len(full_ids) > self.max_len:
                    budget = self.max_len - assistant_token_count
                    if budget <= 64:
                        if drop_on_truncation:
                            self.drop_reasons["assistant_too_long_for_budget"] += 1
                            n_drop += 1
                            continue
                        prompt_keep = min(64, assistant_tok_start)
                        assistant_keep = max(16, self.max_len - prompt_keep)
                        start = max(0, assistant_tok_start - prompt_keep)
                        end = min(len(full_ids), assistant_tok_start + assistant_keep)
                        full_ids = full_ids[start:end]
                        offsets = offsets[start:end]
                        assistant_tok_start = prompt_keep
                    else:
                        keep_from = max(0, assistant_tok_start - max(64, budget))
                        full_ids = full_ids[keep_from:]
                        offsets = offsets[keep_from:]
                        assistant_tok_start -= keep_from

                attn = [1] * len(full_ids)

                # Labels: supervise assistant only
                labels = [-100] * len(full_ids)
                for i in range(assistant_tok_start, len(full_ids)):
                    labels[i] = full_ids[i]
                sup_total += sum(1 for t in labels if t != -100)

                # Up-weight conflict label line tokens
                line_s, line_e = _find_conflict_line_span(assistant)
                weights = [1.0] * len(full_ids)
                array_s, array_e = _span_after_first_json_array(assistant)
                if array_s is not None and array_e is not None:
                    _boost_char_span(
                        weights, labels, offsets,
                        assistant_char_start + array_s,
                        assistant_char_start + array_e,
                        self.array_w,
                    )
                if line_s is not None and line_e is not None:
                    _boost_char_span(
                        weights, labels, offsets,
                        assistant_char_start + line_s,
                        assistant_char_start + line_e,
                        self.conf_w,
                    )
                for literal in (THINK_OPEN, THINK_CLOSE, SENTINEL, ABSTAIN_ANSWER):
                    for span_s, span_e in _find_literal_spans(assistant, literal):
                        _boost_char_span(
                            weights, labels, offsets,
                            assistant_char_start + span_s,
                            assistant_char_start + span_e,
                            self.contract_w,
                        )
                for cite_s, cite_e in _find_citation_spans(assistant):
                    _boost_char_span(
                        weights, labels, offsets,
                        assistant_char_start + cite_s,
                        assistant_char_start + cite_e,
                        self.citation_w,
                    )

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
        if self.drop_reasons:
            print(f"[Data] Drop reasons: {dict(sorted(self.drop_reasons.items()))}")
        if not self.items:
            raise RuntimeError("Dataset has 0 usable items after filtering.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class DistributedWeightedSampler(Sampler):
    """Weighted sampling that shards sampled indices across DDP ranks."""

    def __init__(self, weights, num_replicas, rank, replacement=True, seed=42):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


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
      - Doc-verdict micro accuracy
      - Format compliance rate
      - Abstain accuracy
    Saves best checkpoint by a blended selection score with early stopping.
    """
    def __init__(
        self,
        tok,
        val_jsonl,
        out_dir,
        dist_env,
        patience=4,
        max_new_tokens_base=1200,
        max_new_tokens_cap=2200,
        dev_subset=0,
        selection_doc_verdict_weight=0.0,
        selection_format_weight=0.35,
        selection_abstain_weight=0.1,
        retry_attempts=1,
        retry_scale=1.6,
        retry_cap=2400,
        seed=42,
    ):
        self.tok = tok
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best = -1.0
        self.bad = 0
        self.patience = patience
        self.dist_env = dist_env
        self.max_new_base = max_new_tokens_base
        self.max_new_cap = max_new_tokens_cap
        self.selection_doc_verdict_weight = float(selection_doc_verdict_weight)
        self.selection_format_weight = float(selection_format_weight)
        self.selection_abstain_weight = float(selection_abstain_weight)
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_scale = max(1.0, float(retry_scale))
        self.retry_cap = int(retry_cap)
        self.history_path = self.out_dir / "dev_eval_history.jsonl"
        self.best_score = -1.0
        self.stopper = SentinelStopper(tok, SENTINEL)

        # Cache val examples
        self.cache = []
        for ex in read_jsonl(val_jsonl):
            msgs = ex.get("messages") or []
            user_txt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            sys_txt = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
            asst_txt = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            gold = extract_conflict_label(asst_txt)
            gold_abstain = ABSTAIN_ANSWER in asst_txt
            self.cache.append({
                "sys_txt": sys_txt,
                "user_txt": user_txt,
                "gold": gold,
                "gold_abstain": gold_abstain,
                "gold_doc_verdicts": extract_doc_verdict_map(asst_txt),
                "n_docs": count_docs_in_user_message(user_txt),
            })
        if dev_subset and dev_subset > 0 and len(self.cache) > dev_subset:
            rng = random.Random(seed)
            buckets = defaultdict(list)
            for ex in self.cache:
                buckets[cache_bucket_key(ex)].append(ex)
            for bucket in buckets.values():
                rng.shuffle(bucket)
            order = sorted(
                buckets,
                key=lambda key: (
                    -1 if key[0] is None else key[0],
                    1 if key[1] else 0,
                    len(buckets[key]),
                ),
            )
            selected = []
            while len(selected) < dev_subset:
                progressed = False
                for key in order:
                    bucket = buckets[key]
                    if bucket and len(selected) < dev_subset:
                        selected.append(bucket.pop())
                        progressed = True
                if not progressed:
                    break
            self.cache = selected
            subset_counts = Counter((ex["gold"], ex["gold_abstain"]) for ex in self.cache)
            pretty_counts = {
                f"{'UNKNOWN' if gold is None else CONFLICT_TYPES[gold]}|abstain={abstain}": count
                for (gold, abstain), count in sorted(
                    subset_counts.items(),
                    key=lambda item: (-1 if item[0][0] is None else item[0][0], 1 if item[0][1] else 0),
                )
            }
            print(
                f"[DevEval] Using a fixed stratified subset of {len(self.cache)} validation examples "
                f"with labels={pretty_counts}"
            )

    def _dist_barrier(self):
        if self.dist_env["is_distributed"] and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _generate_label(self, model, sys_txt, user_txt, n_docs):
        model = unwrap_model(model)
        msgs = [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": user_txt},
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        max_new = estimate_max_new_tokens(n_docs, self.max_new_base, self.max_new_cap)
        max_ctx = int(getattr(model.config, "max_position_embeddings", getattr(self.tok, "model_max_length", 8192)))
        model_device = next(model.parameters()).device
        out = ""
        for attempt in range(self.retry_attempts + 1):
            attempt_max_new = max_new if attempt == 0 else min(
                int(math.ceil(max_new * (self.retry_scale ** attempt))),
                self.retry_cap,
            )
            max_inp = max(512, max_ctx - attempt_max_new - 64)
            inputs = self.tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_inp,
                verbose=False,
            ).to(model_device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=attempt_max_new,
                    do_sample=False,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([self.stopper]),
                )
            cont = gen[0][inputs["input_ids"].shape[-1]:]
            out = self.tok.decode(cont, skip_special_tokens=True)
            if is_format_faithful(out):
                break
        return (
            extract_conflict_label(out),
            is_format_faithful(out),
            (ABSTAIN_ANSWER in out),
            extract_doc_verdict_map(out),
        )

    @staticmethod
    def _macro_f1(golds, preds):
        present = sorted({g for g in golds if g is not None})
        if not present:
            return 0.0
        n = len(CONFLICT_TYPES)
        cm = [[0] * n for _ in range(n)]
        for g, p in zip(golds, preds):
            if g is None or p is None:
                continue
            cm[g][p] += 1
        f1s = []
        for c in present:
            tp = cm[c][c]
            fp = sum(cm[r][c] for r in range(n)) - tp
            fn = sum(cm[c][r] for r in range(n)) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
            f1s.append(f1)
        return sum(f1s) / len(f1s)

    def on_epoch_end(self, args, state, control, **kw):
        self._dist_barrier()
        try:
            if not self.dist_env["is_main_process"]:
                return control
            model = kw["model"].eval()
            save_model = unwrap_model(model)
            preds, golds = [], []
            format_ok = 0
            abstain_correct = 0
            doc_verdict_correct = 0
            doc_verdict_total = 0
            for ex in self.cache:
                p, fmt, pred_abstain, pred_doc_verdicts = self._generate_label(
                    model, ex["sys_txt"], ex["user_txt"], ex["n_docs"]
                )
                preds.append(p)
                golds.append(ex["gold"])
                if fmt:
                    format_ok += 1
                if pred_abstain == ex["gold_abstain"]:
                    abstain_correct += 1
                for doc_id, gold_verdict in ex["gold_doc_verdicts"].items():
                    doc_verdict_total += 1
                    if pred_doc_verdicts.get(doc_id) == gold_verdict:
                        doc_verdict_correct += 1

            macro = self._macro_f1(golds, preds)
            fmt_rate = format_ok / max(1, len(self.cache))
            abstain_acc = abstain_correct / max(1, len(self.cache))
            doc_verdict_acc = doc_verdict_correct / max(1, doc_verdict_total)
            macro_weight = max(
                0.0,
                1.0
                - self.selection_doc_verdict_weight
                - self.selection_format_weight
                - self.selection_abstain_weight,
            )
            selection_score = (
                (macro_weight * macro)
                + (self.selection_doc_verdict_weight * doc_verdict_acc)
                + (self.selection_format_weight * fmt_rate)
                + (self.selection_abstain_weight * abstain_acc)
            )
            epoch = getattr(state, "epoch", -1)
            print(
                f"[DevEval] epoch={epoch:.2f} macro-F1={macro:.4f} "
                f"doc_verdict_acc={doc_verdict_acc:.1%} format_ok={fmt_rate:.1%} "
                f"abstain_acc={abstain_acc:.1%} selection_score={selection_score:.4f}"
            )
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "macro_f1_present_classes": macro,
                    "doc_verdict_accuracy": doc_verdict_acc,
                    "format_ok_rate": fmt_rate,
                    "abstain_accuracy": abstain_acc,
                    "selection_score": selection_score,
                    "num_examples": len(self.cache),
                }) + "\n")

            if selection_score > self.best_score + 1e-6:
                self.best_score = selection_score
                self.best = macro
                self.bad = 0
                best_dir = self.out_dir / "best_dev_f1"
                best_dir.mkdir(parents=True, exist_ok=True)
                save_model.save_pretrained(best_dir.as_posix())
                self.tok.save_pretrained(best_dir.as_posix())
                with open(best_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "epoch": epoch,
                        "macro_f1_present_classes": macro,
                        "doc_verdict_accuracy": doc_verdict_acc,
                        "format_ok_rate": fmt_rate,
                        "abstain_accuracy": abstain_acc,
                        "selection_score": selection_score,
                        "num_examples": len(self.cache),
                    }, f, ensure_ascii=False, indent=2)
                print(f"[DevEval] ★ New BEST saved → {best_dir}")
            else:
                self.bad += 1
                if self.bad >= self.patience:
                    print(f"[DevEval] Early stopping (patience={self.patience})")
                    control.should_training_stop = True
            return control
        finally:
            self._dist_barrier()


# ────────── Main ──────────
def main():
    ap = argparse.ArgumentParser(description="QLoRA fine-tuning for conflict-aware RAG")
    ap.add_argument("--base_model", required=True, help="HF model path")
    ap.add_argument("--train_jsonl", nargs="+", required=True,
                    help="Training message JSONL file(s). For this repo, separate stagewise/monolithic runs are recommended.")
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
    ap.add_argument("--contract_weight", type=float, default=2.5,
                    help="Loss multiplier for exact contract markers such as think tags, sentinel, and abstain line.")
    ap.add_argument("--array_weight", type=float, default=1.35,
                    help="Loss multiplier for the per-document JSON array inside the think block.")
    ap.add_argument("--citation_weight", type=float, default=1.75,
                    help="Loss multiplier for [dX] citation tokens.")
    ap.add_argument("--class_balance_power", type=float, default=0.5,
                    help="Exponent for inverse-frequency example sampling. 0.5 reproduces the prior 1/sqrt(count) behavior.")
    ap.add_argument("--neftune_alpha", type=float, default=5.0,
                    help="NEFTune noise alpha (0=disabled). 5.0 recommended.")
    ap.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    ap.add_argument("--dev_max_new_base", type=int, default=1200)
    ap.add_argument("--dev_max_new_cap", type=int, default=2200)
    ap.add_argument("--dev_subset", type=int, default=0, help="Optional fixed-size subset for faster dev eval")
    ap.add_argument("--dev_doc_verdict_weight", type=float, default=0.0,
                    help="Blend weight for doc-verdict micro accuracy when selecting the best dev checkpoint.")
    ap.add_argument("--dev_format_weight", type=float, default=0.35,
                    help="Blend weight for format compliance when selecting the best dev checkpoint.")
    ap.add_argument("--dev_abstain_weight", type=float, default=0.1,
                    help="Blend weight for abstain accuracy when selecting the best dev checkpoint. Set to 0.0 for semantics-first ablations.")
    ap.add_argument("--dev_retry_attempts", type=int, default=1,
                    help="Number of extra greedy retries for dev eval when the draft is structurally invalid.")
    ap.add_argument("--dev_retry_scale", type=float, default=1.6,
                    help="Per-retry max_new_tokens multiplier for dev eval retries.")
    ap.add_argument("--dev_retry_cap", type=int, default=2400,
                    help="Upper bound on max_new_tokens during dev eval retries.")

    # Hardware
    ap.add_argument("--attn_impl", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--ddp_timeout_sec", type=int, default=10800,
                    help="Distributed process-group timeout in seconds. Increase this when rank-0 dev eval is long.")
    ap.add_argument("--overwrite_output_dir", action="store_true")

    args = ap.parse_args()

    # Stability
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    dist_env = get_dist_env()
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        if dist_env["is_distributed"]:
            torch.cuda.set_device(dist_env["local_rank"])

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume_from and not args.overwrite_output_dir:
        raise RuntimeError(
            f"Output directory already exists and is not empty: {out_dir}\n"
            "Choose a new run directory, pass --resume_from, or explicitly allow overwrite."
        )

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.model_max_length = args.max_len

    # QLoRA base model
    compute_dtype = pick_compute_dtype()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype if compute_dtype in (torch.bfloat16, torch.float16) else torch.float16,
    )
    device_map = {"": dist_env["local_rank"]} if dist_env["is_distributed"] else "auto"
    base = load_causal_lm(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=compute_dtype,
        attn_impl=args.attn_impl,
        local_files_only=True,
        device_map=device_map,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.config.use_cache = False
    lora_targets = select_lora_target_modules(base)

    lconf = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
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
        contract_weight=args.contract_weight,
        array_weight=args.array_weight,
        citation_weight=args.citation_weight,
    )
    collator = PadCollator(tok)
    write_run_metadata(out_dir, args, train_ds, compute_dtype, lora_targets, dist_env)

    # Class-balanced sampler (inverse frequency raised to a configurable power)
    counts = Counter([c for c in train_ds.class_idx if c is not None and c >= 0])
    sampler = None
    if counts:
        per_ex_w = []
        for c in train_ds.class_idx:
            cnt = max(1, counts.get(c, 1))
            per_ex_w.append(1.0 / (cnt ** args.class_balance_power))
        if dist_env["is_distributed"]:
            sampler = DistributedWeightedSampler(
                per_ex_w,
                num_replicas=dist_env["world_size"],
                rank=dist_env["rank"],
                replacement=True,
                seed=42,
            )
        else:
            sampler = WeightedRandomSampler(
                torch.as_tensor(per_ex_w, dtype=torch.double),
                num_samples=len(train_ds),
                replacement=True,
            )
        train_loader_kwargs = dict(sampler=sampler)
    else:
        if dist_env["is_distributed"]:
            sampler = DistributedSampler(
                train_ds,
                num_replicas=dist_env["world_size"],
                rank=dist_env["rank"],
                shuffle=True,
                seed=42,
            )
            train_loader_kwargs = dict(sampler=sampler)
        else:
            train_loader_kwargs = dict(shuffle=True)

    samples_per_rank = len(sampler) if sampler is not None else len(train_ds)
    steps_per_epoch = max(1, math.ceil(samples_per_rank / (args.bsz * args.grad_accum)))

    train_arg_kwargs = dict(
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
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=torch.cuda.is_available(),
        save_safetensors=True,
        seed=42,
    )
    train_arg_params = inspect.signature(TrainingArguments.__init__).parameters
    if "group_by_length" in train_arg_params:
        train_arg_kwargs["group_by_length"] = False
    if compute_dtype == torch.bfloat16:
        train_arg_kwargs["bf16"] = True
    else:
        train_arg_kwargs["fp16"] = True
    if dist_env["is_distributed"]:
        train_arg_kwargs["ddp_find_unused_parameters"] = False
        train_arg_kwargs["ddp_timeout"] = args.ddp_timeout_sec
    if args.overwrite_output_dir:
        train_arg_kwargs["overwrite_output_dir"] = True
    targs = training_args_with_compat(**train_arg_kwargs)

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
            pin_memory=targs.dataloader_pin_memory,
            num_workers=targs.dataloader_num_workers,
            **train_loader_kwargs,
        )
    trainer.get_train_dataloader = _get_train_dl

    # Dev evaluation callback
    # Use first val_jsonl for eval
    trainer.add_callback(
        DevEvalCallback(
            tok, args.val_jsonl, args.out_dir, dist_env,
            patience=args.patience,
            max_new_tokens_base=args.dev_max_new_base,
            max_new_tokens_cap=args.dev_max_new_cap,
            dev_subset=args.dev_subset,
            selection_doc_verdict_weight=args.dev_doc_verdict_weight,
            selection_format_weight=args.dev_format_weight,
            selection_abstain_weight=args.dev_abstain_weight,
            retry_attempts=args.dev_retry_attempts,
            retry_scale=args.dev_retry_scale,
            retry_cap=args.dev_retry_cap,
        )
    )

    # Train
    try:
        try:
            trainer.train(resume_from_checkpoint=args.resume_from)
        finally:
            if neftune_hook:
                neftune_hook.remove()

        # Save final
        if dist_env["is_main_process"]:
            unwrap_model(model).save_pretrained(args.out_dir)
            tok.save_pretrained(args.out_dir)
            print(f"✓ Saved adapter → {args.out_dir}")
    finally:
        maybe_destroy_process_group(dist_env)


if __name__ == "__main__":
    main()
