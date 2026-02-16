# CATS RAG Evaluation Pipeline

This repository contains an evaluation pipeline for **Retrieval-Augmented Generation (RAG)** systems with a focus on:

- **Trust-Score** style aggregate metrics
- **Conflict-aware behavior** (Dragged-into-Conflicts taxonomy)
- **Grounded factuality** via NLI
- **Single-truth recall** using LLM-as-a-judge

It is designed to evaluate models on datasets where retrieved documents may **disagree, be outdated, complementary, or contain misinformation**, and to check whether the model **behaves correctly** given the conflict type.

---

## Features

- **Trust-Score components** (hooks in place for F1-GR, answer correctness, grounded citation).
- **Conflict-aware evaluation**:
  - Behavior adherence (per conflict type 1–5).
  - Factual grounding ratio (claims vs. supporting docs via NLI).
  - Single-truth answer recall (LLM judge for gold answers).
- **LLM-as-a-Judge** via OpenAI API *or* local HF models:
  - Behavior judge (JSON-only rubric-based).
  - NLI judge for grounding.
  - Single-truth recall judge.
- **Markdown report generation** summarizing:
  - Overall conflict-aware metrics.
  - Per-type breakdown (1–5).

---

## Repository Structure

A typical layout:

```text
.
├─ run_eval.py                 # Main entry-point to run evaluation
├─ requirements.txt            # Python dependencies
├─ finalparse/
│   └─ sample_eval.jsonl       # Small sample dataset in Stage-3 format (for demo)
├─ rag_eval/
│   ├─ __init__.py
│   ├─ config.py               # EvaluationConfig dataclass and parsing
│   ├─ llm.py                  # LLM wrapper (OpenAI / vLLM / HF local)
│   ├─ evaluator.py            # Orchestrates trustscore + conflict-aware evaluation
│   ├─ conflict_eval.py        # Behavior adherence, factual grounding, single-truth recall
│   ├─ judge_prompts.py        # Strict JSON prompts for behavior/NLI/single-truth judges
│   ├─ metrics.py              # Answered flags, F1_GR, claim extraction, Trust-Score
│   ├─ data.py                 # Record utilities: doc index, support IDs, gold answer, etc.
│   ├─ utils.py                # Helper functions (loading models, vLLM, misc)
│   ├─ logging_config.py       # Logger setup
│   └─ (optional) other helpers: searcher.py, post_hoc_cite.py, auto_ais_loader.py, ...
