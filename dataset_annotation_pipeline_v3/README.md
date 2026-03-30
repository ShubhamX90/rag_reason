# RAG Dataset Annotation Pipeline (v3)

A multi-stage annotation pipeline for conflict-aware Retrieval-Augmented Generation (RAG) datasets.
Supports both **single-LLM** and **multi-LLM committee** annotation for the two main strategies: **3-stage** and **monolithic**.

---

## Overview

The pipeline produces fully-annotated JSONL records containing:

| Field | Stage | Description |
|---|---|---|
| `per_doc_notes` | Stage 1 | Per-document verdict (supports / partially supports / irrelevant), key fact, quote, source quality |
| `conflict_reason` | Stage 2 | Why these documents collectively exhibit the given conflict type |
| `answerable_under_evidence` | Stage 2 | Whether the query can be answered from the retrieved evidence |
| `expected_response` | Stage 3 | Evidence-grounded answer with `[dX]` citations, abstain flag |
| `think` | Stage 3 | Internal `<think>…</think>` reasoning trace |

---

## Annotation Strategies

### Strategy A: 3-Stage

```
Normalized JSONL
    │
    ▼ Stage 1  (N calls per query, one per retrieved doc)
    │  Per-document verdict + evidence extraction
    │
    ▼ Stage 2  (1 call per query)
    │  Conflict-type reasoning + answerability
    │
    ▼ Stage 3  (1 call per query)
    │  Grounded expected response + think trace
    │
    ▼ Annotated JSONL
```

**Best for:** Maximum traceability; stage-by-stage debugging; explicit evidence adjudication.

### Strategy B: Monolithic

```
Normalized JSONL
    │
    ▼ One call per query
    │  All stages combined: per-doc verdicts + conflict reason + response
    │
    ▼ Annotated JSONL (same schema)
```

**Best for:** Fewer round trips; faster end-to-end annotation; cross-strategy comparison.

Both strategies produce **identical output schemas** — fully interchangeable downstream.

## LLM Approaches

Each main strategy can be run in one of two ways:

| Approach | Works with 3-Stage | Works with Monolithic | Notes |
|---|---|---|---|
| **Single-LLM** | yes | yes | One model runs the entire chosen strategy |
| **Multi-LLM committee** | yes | yes | Committee voting across OpenRouter-backed models |

---

## LLM Providers

| Provider | Models | Modes | Use Case |
|---|---|---|---|
| **Anthropic** | `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` | async, batch | Single-LLM runs |
| **OpenAI** | `gpt-4o` and compatible chat models | async, batch | Single-LLM runs |
| **OpenRouter** | `qwen/...`, `anthropic/...`, `openai/...`, and committee models | async only | Single-LLM reproducibility and all multi-LLM committee runs |

### Model recommendations

- **Single-LLM**: `claude-sonnet-4-6` is the strongest general default.
- **Single-LLM reproducibility**: an OpenRouter model such as `qwen/qwen-2.5-72b-instruct` is a good fit.
- **Multi-LLM committee**: the committee is defined centrally in [src/voting.py](/Users/shubhammishra/Desktop/rag_reason/dataset_annotation_pipeline_v3/src/voting.py).

---

## Execution Modes

| Mode | Description | Cost | Speed |
|---|---|---|---|
| **Async** | Concurrent API calls via asyncio + semaphore | Standard pricing | Fast (configurable concurrency) |
| **Batch** | Anthropic/OpenAI batch APIs | **discounted where supported** | Single-LLM only |

> **Multi-LLM committee runs are async-only.** Batch mode applies only to the legacy single-LLM path.
> Multi-LLM runs and OpenRouter monolithic async runs also save a JSON cost sidecar by default: `<output>_cost_report.json`.

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
# Multi-LLM committee:
export OPENROUTER_API_KEY="sk-or-..."

# Optional for single-LLM runs:
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Option B: key files (more convenient for local dev)
echo "sk-or-..."  > ~/.openrouter_key
echo "sk-ant-..." > ~/.anthropic_key
echo "sk-..."     > ~/.openai_key
```

### 3. Normalize raw data (if needed)

```bash
python scripts/normalize_raw_dataset.py
```

### 4. Run interactively

```bash
bash run_pipeline.sh
```

The shell script now walks through:

1. strategy: `3-stage` or `Monolithic`
2. approach: `Single-LLM` or `Multi-LLM`
3. provider/mode when relevant
4. paths and concurrency

---

## Manual Script Usage

### 3-Stage Pipeline (Single-LLM, Async)

```bash
# Stage 1: per-document evidence adjudication
python scripts/run_stage1_async.py \
    --input  data/normalized/conflicts_normalized.jsonl \
    --output data/stage1_outputs/stage1_out.jsonl \
    --provider anthropic \
    --model claude-sonnet-4-6 \
    --concurrency 10

# Stage 2: conflict reasoning
python scripts/run_stage2_async.py \
    --input  data/stage1_outputs/stage1_out.jsonl \
    --output data/stage2_outputs/stage2_out.jsonl \
    --concurrency 12

# Stage 3: expected response
python scripts/run_stage3_async.py \
    --input  data/stage2_outputs/stage2_out.jsonl \
    --output data/stage3_outputs/stage3_out.jsonl \
    --concurrency 8
```

### 3-Stage Pipeline (Single-LLM, Batch)

```bash
python scripts/run_stage1_batch.py \
    --input  data/normalized/conflicts_normalized.jsonl \
    --output data/stage1_outputs/stage1_batch.jsonl \
    --batch-id-file data/.batch_ids/s1.txt

python scripts/run_stage2_batch.py \
    --input  data/stage1_outputs/stage1_batch.jsonl \
    --output data/stage2_outputs/stage2_batch.jsonl \
    --batch-id-file data/.batch_ids/s2.txt

python scripts/run_stage3_batch.py \
    --input  data/stage2_outputs/stage2_batch.jsonl \
    --output data/stage3_outputs/stage3_batch.jsonl \
    --batch-id-file data/.batch_ids/s3.txt
```

> **Tip:** `--batch-id-file` saves the batch ID to a file. If the script is interrupted, rerun with the same `--batch-id-file` to resume polling without resubmitting.

### Monolithic Pipeline (Single-LLM, Async)

```bash
python scripts/run_monolithic_async.py \
    --input  data/normalized/conflicts_normalized.jsonl \
    --output data/monolithic_outputs/mono_out.jsonl \
    --concurrency 8
```

### Monolithic Pipeline (Single-LLM, OpenRouter / Qwen)

```bash
python scripts/run_monolithic_async.py \
    --input    data/normalized/conflicts_normalized.jsonl \
    --output   data/monolithic_outputs/mono_qwen.jsonl \
    --provider openrouter \
    --model    qwen/qwen2.5-72b-instruct \
    --concurrency 6
```

### Monolithic Pipeline (Single-LLM, Batch)

```bash
python scripts/run_monolithic_batch.py \
    --input  data/normalized/conflicts_normalized.jsonl \
    --output data/monolithic_outputs/mono_batch.jsonl \
    --batch-id-file data/.batch_ids/mono.txt
```

### 3-Stage Pipeline (Multi-LLM Committee)

```bash
python scripts/run_stage1_multi_async.py \
    --input  data/normalized/conflicts_normalized.jsonl \
    --output data/stage1_outputs/stage1_multi.jsonl

python scripts/run_stage2_multi_async.py \
    --input  data/stage1_outputs/stage1_multi.jsonl \
    --output data/stage2_outputs/stage2_multi.jsonl

python scripts/run_stage3_multi_async.py \
    --input  data/stage2_outputs/stage2_multi.jsonl \
    --output data/stage3_outputs/stage3_multi.jsonl
```

Each run will also save a cost report JSON next to the output file, for example:

```text
data/stage1_outputs/stage1_multi_cost_report.json
data/stage2_outputs/stage2_multi_cost_report.json
data/stage3_outputs/stage3_multi_cost_report.json
```

For the refusals dataset, Stage 2 uses:

```bash
python scripts/run_stage2_multi_async.py \
    --input  data/stage1_outputs/refusals_stage1_multi.jsonl \
    --output data/stage2_outputs/refusals_stage2_multi.jsonl \
    --refusal-mode
```

### Monolithic Pipeline (Multi-LLM Committee)

```bash
python scripts/run_monolithic_multi_async.py \
    --input  data/normalized/conflicts_normalized.jsonl \
    --output data/monolithic_outputs/monolithic_multi.jsonl
```

You can override the default cost-report path with:

```bash
--cost-report path/to/custom_cost_report.json
```

---

## CLI Reference

Single-LLM scripts share these common flags:

| Flag | Description | Default |
|---|---|---|
| `--input` | Path to input JSONL | required |
| `--output` | Path to output JSONL | required |
| `--provider` | `anthropic` or `openrouter` | `anthropic` |
| `--model` | Model name | Provider default |
| `--temperature` | Generation temperature | `0.0` |
| `--concurrency` | Max simultaneous API calls (async only) | varies |
| `--limit` | Process only first N records | all |
| `--max-retries` | Retries per failed call | `3` |
| `--system-prompt` | Override system prompt path | built-in |
| `--user-prompt` | Override user prompt path | built-in |

Batch-specific flags:

| Flag | Description |
|---|---|
| `--batch-id` | Existing batch ID (skip submission, resume polling) |
| `--batch-id-file` | File to read/write batch ID (for crash recovery) |
| `--poll-interval` | Seconds between status checks (default: 30) |

---

## Output Schema

Each annotated record in the output JSONL has this structure:

```jsonc
{
  "id":            "#0001",
  "query":         "What is the maximum human lifespan?",
  "conflict_type": "Conflicting opinions or research outcomes",
  "gold_answer":   "...",
  "retrieved_docs": [
    {"doc_id": "d1", "source_url": "...", "snippet": "...", "timestamp": "..."},
    ...
  ],

  // Stage 1 output (per_doc_notes):
  "per_doc_notes": [
    {
      "doc_id":         "d1",
      "verdict":        "supports",           // supports | partially supports | irrelevant
      "key_fact":       "One paraphrased sentence strictly entailed by the quote.",
      "quote":          "≤50 word verbatim contiguous span from snippet.",
      "verdict_reason": "≤50 word justification.",
      "source_quality": "high"                // high | low
    },
    ...
  ],

  // Stage 2 output:
  "conflict_reason":            "≤50 word explanation referencing d1–dN.",
  "answerable_under_evidence":  true,

  // Stage 3 output:
  "expected_response": {
    "answer":        "Evidence-grounded answer with [d1] [d3] citations.",
    "evidence":      ["d1", "d3"],            // high-cred first
    "abstain":       false,
    "abstain_reason": null
  },
  "think": "<think>…</think>",

  // Strategy tag (monolithic only):
  "_annotation_strategy": "monolithic"
}
```

---

## Resumption

All async scripts support **automatic resumption**: if `--output` already contains processed records, those IDs are skipped. Simply rerun the same command.

All batch scripts support resumption via `--batch-id-file` (saves batch ID to disk so polling can continue after a crash).

---

## Validation

```bash
# Validate Stage-1 output
python scripts/validate_stage1.py

# Validate Stage-2 output
python scripts/validate_stage2.py

# Validate Stage-3 output
python scripts/validate_stage3.py
```

---

## Project Structure

```
dataset_annotation_pipeline/
│
├── run_pipeline.sh             ← Interactive runner: strategy → single/multi LLM
├── requirements.txt
├── .env.example
│
├── prompts/
│   ├── system_stage1.txt       ← Per-doc evidence adjudication
│   ├── user_stage1.txt
│   ├── system_stage2.txt       ← Conflict macro reasoning
│   ├── user_stage2.txt
│   ├── system_stage3.txt       ← Grounded response synthesis
│   ├── user_stage3.txt
│   ├── system_monolithic.txt   ← All-in-one monolithic prompt
│   └── user_monolithic.txt
│
├── src/
│   ├── llm_client.py           ← Unified client for single-LLM and OpenRouter committee runs
│   ├── voting.py               ← Weighted voting for committee outputs
│   ├── parsers.py              ← Robust output parsers for all stages
│   └── utils.py                ← Shared utilities (from original pipeline)
│
├── scripts/
│   │   ── Async runners ──────────────────────────────
│   ├── run_stage1_async.py
│   ├── run_stage2_async.py
│   ├── run_stage3_async.py
│   ├── run_monolithic_async.py
│   │   ── Multi-LLM committee runners ────────────────
│   ├── run_stage1_multi_async.py
│   ├── run_stage2_multi_async.py
│   ├── run_stage3_multi_async.py
│   ├── run_monolithic_multi_async.py
│   │   ── Batch runners (single-LLM only) ─────────────
│   ├── run_stage1_batch.py
│   ├── run_stage2_batch.py
│   ├── run_stage3_batch.py
│   ├── run_monolithic_batch.py
│   │   ── Validation & utilities ───────────────────────
│   ├── validate_stage1.py
│   ├── validate_stage2.py
│   ├── validate_stage3.py
│   ├── normalize_raw_dataset.py
│   └── pretty_stage3.py
│
└── data/
    ├── raw/                    ← Raw dataset files
    ├── normalized/             ← Normalized JSONL (input to Stage 1)
    ├── stage1_outputs/
    ├── stage2_outputs/
    ├── stage3_outputs/
    └── monolithic_outputs/     ← Monolithic strategy outputs
```

---

## Design Notes

### Why the committee path?
The multi-LLM committee path lets the repo vote on verdicts, answerability, abstention, and related fields instead of trusting one model call. The merge logic lives in [src/voting.py](/Users/shubhammishra/Desktop/rag_reason/dataset_annotation_pipeline_v3/src/voting.py), and the shell runner now exposes that choice directly.

### JSON parsing robustness
Both `src/parsers.py` (for all stages) and the original `src/utils.py` use multi-layered extraction:
1. Direct `json.loads()`
2. Strip markdown fences, retry
3. Find balanced `{…}` or `[…]` block using bracket-counting parser
4. Fix trailing commas, retry
5. Graceful fallback with `_parse_error` flag

### Batch crash recovery
Batch scripts save the Anthropic batch ID to `--batch-id-file`. If interrupted, rerun with the same file path — the script skips submission and resumes polling. Batch results are valid for 29 days after creation.
