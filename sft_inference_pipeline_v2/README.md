# SFT & Inference Pipeline v2 — Conflict-Aware RAG

## Overview

This pipeline handles **QLoRA fine-tuning** and **inference** for conflict-aware Retrieval-Augmented Generation (RAG) models. The model learns to:
1. Analyze retrieved documents and produce per-doc verdicts (supports / partially supports / irrelevant)
2. Classify the conflict type across documents (5 classes)
3. Generate a conflict-aware final answer with proper citations

### What Changed from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Prompting strategies | Stage-wise (3 calls) + Monolithic + Ablations + Simple RAG | Single unified prompt (formerly "monolithic") |
| Oracle levels | oracle1 (conflict type), oracle2 (per-doc notes), oracle3 (both) | oracle (conflict type only) |
| Training/inference prompts | Separate prompt sets | **Same prompts** for training, SFT inference, and baseline inference |
| Dataset annotation source | Part of this pipeline | Done externally (foundation model); pipeline receives annotated data |
| Fine-tuning | Basic QLoRA | QLoRA + NEFTune + better LR schedule + mixed-mode training + improved class weighting |

### Prompt Modes

- **E2E (end-to-end):** Model receives query + retrieved docs → must predict per-doc verdicts, conflict type, and answer
- **Oracle:** Model receives query + retrieved docs + **gold conflict type** → must predict per-doc verdicts and answer

## Directory Structure

```
sft_inference_pipeline_v2/
├── run.sh                          # Main interactive entry point
├── README.md                       # This file
├── prompts/
│   ├── system_e2e.txt              # E2E system prompt
│   ├── user_e2e.txt                # E2E user template
│   ├── system_oracle.txt           # Oracle system prompt
│   └── user_oracle.txt             # Oracle user template
├── code/
│   ├── data/
│   │   └── prepare_data.py         # Split + build training messages
│   ├── train/
│   │   └── train_qlora.py          # QLoRA fine-tuning
│   └── eval/
│       ├── generate.py             # Unified inference (SFT + baseline)
│       ├── sanitize.py             # Post-process outputs
│       ├── eval_conflict_type.py   # Conflict type evaluation
│       ├── eval_doc_verdicts.py    # Per-doc verdict evaluation
│       └── eval_contract.py        # Output format contract checks
├── scripts/
│   ├── prepare_data.sh             # Interactive data prep
│   ├── train.sh                    # Interactive training launcher
│   ├── generate.sh                 # Interactive generation launcher
│   └── evaluate.sh                 # Interactive evaluation runner
├── data/
│   ├── raw/                        # stage3_final.jsonl goes here
│   ├── splits/                     # Train/val/test canonical splits
│   └── messages/                   # Formatted message JSONL files
├── outputs/                        # Generated outputs & reports
└── checkpoints/                    # Saved model adapters (created during training)
```

## Quick Start

### Prerequisites

```bash
pip install torch transformers peft bitsandbytes accelerate
```

### Option A: Interactive Mode

```bash
bash run.sh
```

This presents a menu to run any pipeline stage interactively.

### Option B: Step-by-Step

```bash
# 1. Prepare data (split + build messages)
bash scripts/prepare_data.sh

# 2. Train (QLoRA fine-tuning)
bash scripts/train.sh

# 3. Generate (inference)
bash scripts/generate.sh

# 4. Evaluate
bash scripts/evaluate.sh
```

## Detailed Usage

### 1. Data Preparation

Place your `stage3_final.jsonl` in `data/raw/`, then:

```bash
bash scripts/prepare_data.sh
```

Or directly:

```bash
python code/data/prepare_data.py \
    --raw_jsonl data/raw/stage3_final.jsonl \
    --out_dir data \
    --prompts_dir prompts \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

This produces:
- `data/splits/{train,val,test}.jsonl` — canonical splits (stratified by conflict type)
- `data/messages/{train,val,test}_{e2e,oracle}_messages.jsonl` — formatted for training/inference

### 2. Fine-Tuning

Interactive:
```bash
bash scripts/train.sh
```

Direct:
```bash
# E2E training
python code/train/train_qlora.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --train_jsonl data/messages/train_e2e_messages.jsonl \
    --val_jsonl data/messages/val_e2e_messages.jsonl \
    --out_dir checkpoints/sft_e2e_run1

# Mixed-mode training (recommended — uses both E2E + Oracle data)
python code/train/train_qlora.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --train_jsonl data/messages/train_e2e_messages.jsonl \
                  data/messages/train_oracle_messages.jsonl \
    --val_jsonl data/messages/val_e2e_messages.jsonl \
    --out_dir checkpoints/sft_mixed_run1
```

### 3. Inference

Interactive:
```bash
bash scripts/generate.sh
```

Direct:
```bash
# SFT model inference
python code/eval/generate.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --lora_dir checkpoints/sft_e2e_run1/best_dev_f1 \
    --input_jsonl data/messages/test_e2e_messages.jsonl \
    --out_jsonl outputs/sft_e2e_test.raw.jsonl \
    --auto_length --load_in_4bit

# Baseline (untuned) inference — same command without --lora_dir
python code/eval/generate.py \
    --base_model /path/to/Llama-3.1-8B-Instruct \
    --input_jsonl data/messages/test_e2e_messages.jsonl \
    --out_jsonl outputs/baseline_e2e_test.raw.jsonl \
    --auto_length --load_in_4bit

# Post-process (always run after generation)
python code/eval/sanitize.py \
    --in_jsonl outputs/sft_e2e_test.raw.jsonl \
    --out_jsonl outputs/sft_e2e_test.sanitized.jsonl
```

### 4. Evaluation

Interactive:
```bash
bash scripts/evaluate.sh
```

Direct:
```bash
CANON=data/splits/test.jsonl
GENS=outputs/sft_e2e_test.sanitized.jsonl
OUT=outputs/reports/sft_e2e_test

python code/eval/eval_contract.py      --canon_jsonl $CANON --gens_jsonl $GENS --report_json $OUT/contract.json
python code/eval/eval_doc_verdicts.py  --canon_jsonl $CANON --gens_jsonl $GENS --report_json $OUT/doc_verdicts.json
python code/eval/eval_conflict_type.py --canon_jsonl $CANON --gens_jsonl $GENS --report_json $OUT/conflict_type.json
```

## Fine-Tuning Improvements (v2)

### NEFTune (Noisy Embeddings)
Adds small uniform noise to token embeddings during training. Empirically shown to improve SFT performance by 2-8% across benchmarks. Controlled by `--neftune_alpha` (default: 5.0, set to 0 to disable).

### Better LoRA Configuration
- Default `lora_alpha = 2 * lora_r` (64 vs 32) for better scaling
- Targets all linear layers including gate/up/down projections

### Mixed-Mode Training
Train on **both** E2E and Oracle message files simultaneously. This doubles the effective dataset size and teaches the model to handle both scenarios, improving generalization.

### Improved Class Balancing
- Weighted random sampler with `1/sqrt(count)` per class
- Conflict label line tokens up-weighted by `conflict_weight` (default: 3.0)

### Better Learning Rate Schedule
- Cosine decay with 6% linear warmup
- Weight decay (0.01) for regularization
- Gradient norm clipping (max 1.0)

### Richer Validation
End-of-epoch evaluation reports:
- Conflict type macro-F1 (primary metric for early stopping)
- Format compliance rate

## Training Tips

1. **Start with mixed-mode training** — it effectively doubles your training data
2. **NEFTune alpha=5** is a good default; try 10 for very small datasets
3. **conflict_weight=3.0** helps with minority classes; increase to 5.0 if F1 is low on rare types
4. **6 epochs with patience=4** is usually sufficient; the model often peaks at epoch 3-4
5. **LoRA r=32, alpha=64** is a solid default; increase r to 64 for larger datasets
6. **Monitor val macro-F1** — if it plateaus early, try increasing learning rate slightly

## Output Format (TEXT-MODE)

The model outputs a structured response:

```
<think>
[JSON array of per-doc verdicts]
<ConflictType> — <concise reason>
<Conflict reasoning referencing specific doc IDs>

<Bridge to final answer>
</think>

<Final answer with [dX] citations>
[[END-OF-ANSWER]]
```

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Contract compliance % | Format adherence (think block, JSON array, citations, sentinel) |
| Doc verdict micro-acc | Per-document verdict accuracy (supports/partially/irrelevant) |
| Doc verdict macro-F1 | Class-balanced F1 across verdict types |
| Conflict type accuracy | 5-class classification accuracy |
| Conflict type macro-F1 | Class-balanced F1 across conflict types |

## Data Schema

### Raw Input (`stage3_final.jsonl`)
```json
{
    "id": "#0102",
    "query": "...",
    "conflict_type": "No conflict",
    "conflict_reason": "...",
    "retrieved_docs": [{"doc_id": "d1", "source_url": "...", "snippet": "...", "timestamp": "..."}],
    "per_doc_notes": [{"doc_id": "d1", "verdict": "supports", "key_fact": "...", "verdict_reason": "...", "source_quality": "high"}],
    "expected_response": {"answer": "...", "evidence": ["d1", "d3"], "abstain": false}
}
```

### Message Files (`*_messages.jsonl`)
```json
{
    "id": "#0102",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<think>...</think>\n\n...\n[[END-OF-ANSWER]]"}
    ]
}
```
