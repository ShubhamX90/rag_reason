# RAG Reasoning SFT + Inference Pipeline v2

This repository contains the current reproducibility surface for a conflict-aware Retrieval-Augmented Generation (RAG) reasoning project. It supports supervised fine-tuning (SFT), benchmark inference, and post-inference analysis for models that must use only retrieved evidence to answer, abstain when evidence is insufficient, identify conflicts among sources, and cite the documents used.

The repository has two supervised families:

- **Trace-text SFT (main method):** the model produces a public evidence trace: document-level judgments, conflict assessment, an answer plan, and a cited final answer or controlled refusal.
- **Answer-only SFT (comparison family):** the model produces only the cited final answer or controlled refusal, without exposing the intermediate trace.

The root holds the current recipes, direct inputs, launchers, evaluation code, documentation, and stored result artifacts. Historical prototypes, stale variants, old logs, and superseded copies are preserved under [legacies/](legacies/README.md). They are retained for provenance but are not required to inspect or run the current workflows.

## What problem does the system solve?

Each example contains a query and a set of retrieved document snippets. The model must make an evidence-grounded decision:

| Evidence state | Required behavior |
| --- | --- |
| Retrieved evidence is sufficient | Give a grounded answer with document citations. |
| No individual document is decisive, but compatible partial documents jointly resolve the query | Synthesize the partial evidence and answer. |
| Retrieved evidence is genuinely insufficient | Return the controlled refusal `CANNOT ANSWER, INSUFFICIENT EVIDENCE`. |
| Retrieved sources disagree | Identify the conflict structure and frame the answer accordingly rather than refusing merely because disagreement exists. |

The trace-text system makes its intermediate decisions inspectable:

```text
Retrieved documents
        ↓
Stage 1: supports / partially supports / irrelevant for each document
        ↓
Stage 2: no conflict / complementary / conflicting / outdated / misinformation
        ↓
Stage 3: grounded answer plan or evidence-based refusal
        ↓
Cited final answer + [[END-OF-ANSWER]]
```

This is not a generic chain-of-thought benchmark. The public trace is a structured, evaluated evidence record whose document judgments, conflict label, response contract, citations, refusal decision, and final answer can be inspected separately.

## Repository status at a glance

The current canonical data backbone is **862 training examples**, **81 validation examples**, and a disjoint **736-example final benchmark**.

| Result family | Latest retained recipe | Latest stored complete matrix | Stored result location |
| --- | --- | --- | --- |
| Qwen 2.5 7B | Run L boundary-rebalanced recipe | Run K short-context-targeted | [final_model_outputs/qwen7b/](final_model_outputs/qwen7b/) |
| Qwen 2.5 32B | Run L boundary-rebalanced recipe | Run K short-context-targeted | [final_model_outputs/qwen32b/](final_model_outputs/qwen32b/) |
| Llama 3.1 8B | Run L boundary-rebalanced | Run L | [final_model_outputs/llama8b/](final_model_outputs/llama8b/) |
| Mistral 7B | Run L boundary-rebalanced | Run L | [final_model_outputs/mistral7b/](final_model_outputs/mistral7b/) |
| Answer-only: Qwen 7B, Llama 8B, Mistral 7B | Updated 862/81 final-only + minimal mixture | Updated answer-only results | [final_model_outputs/answer_only_sft/](final_model_outputs/answer_only_sft/) |
| Clean basic answer-only comparison | One final-only target for each canonical 862 training example | No result matrix yet | [answer_only_sft_export/](answer_only_sft_export/) |

“Latest retained recipe” and “latest stored complete matrix” are intentionally separate columns. The repository contains Qwen Run L training/evaluation recipes but no completed Qwen Run L matrix. Likewise, the clean 862-example answer-only baseline is ready to run but has no completed results. Neither should be reported as a completed result family.

## Quick start for reviewers

Start with the [reviewer reproduction guide](docs/REPRODUCIBILITY.md). It states exactly what is available locally and what requires an external base model or adapter.

The repository contains current code, prompts, data, generated messages, SLURM launchers, evaluation code, and stored raw/sanitized outputs. It does **not** include the base-model weights or the latest LoRA adapters used to generate the reported matrices. Thus:

| Task | Status |
| --- | --- |
| Inspect prompts, methods, data audits, output artifacts, and evaluations | Fully available locally |
| Rebuild canonical training and benchmark message files | Available locally |
| Re-evaluate the stored sanitized outputs | Available locally |
| Run base-model inference | Available after obtaining the named base model |
| Re-run a stored SFT matrix exactly | Requires the matching current LoRA adapter |
| Reproduce training-to-checkpoint byte-for-byte | Not fully available: adapter, model revision, runtime, and distributed-training provenance matter |

For a SLURM/Conda environment, configure paths and install the pinned environment from the repository root:

```bash
export PROJECT_ROOT="$(pwd)"
export WORK_ROOT="/path/to/writable/workspace"
export MODEL_ROOT="/path/to/downloaded-models"
export SCRATCH="/path/to/scratch"
INSTALL_MODE=frozen bash scripts/bootstrap_sharanga_env.sh
```

The frozen environment path is the reproducibility-oriented choice. Cluster paths, scheduler account, partitions, base-model locations, and GPU availability remain environment-specific and should be set explicitly outside the original Sharanga environment.

## Current workflows

### 1. Trace-text SFT

The main method teaches a model to produce a cited public evidence trace under three prompt strengths:

| Prompt family | Role |
| --- | --- |
| Strict/default | Detailed teacher-style instruction that fully states evidence, conflict, citation, and output rules |
| Runtime | Compact, deployment-like trace instruction |
| Minimal | Sparse instruction that tests whether the SFT internalized the response protocol |

The SFT mixture also contains targeted document-verdict, conflict-type, answer-only, boundary, and partial-synthesis training views. This gives direct supervision to component decisions while retaining end-to-end examples.

Run K and Run L are the current trace recipes:

| Recipe | Source training examples | Weighted message rows | Central design change |
| --- | ---: | ---: | --- |
| Run K: short-context targeted | 889 | 12,659 | Adds 27 derived five-document answerable examples, document-boundary teaching, and stronger partial-synthesis drills |
| Run L: boundary-rebalanced | 910 | 13,349 | Adds short answerable no-conflict coverage and rebalances no-conflict/misinformation boundaries |

The message rows are repeated supervised views, not independent questions. The design motivation, exact mixtures, historical progression, and scientific limitations are documented in [SFT_DESCRIPTION.md](docs/SFT_DESCRIPTION.md) and [ABLATION_STUDY.md](docs/ABLATION_STUDY.md).

Rebuild the current trace message products:

```bash
bash slurm/examples/rebuild_messages_prompt_robust_k_short_context_targeted.sh
bash slurm/examples/rebuild_messages_prompt_robust_l_boundary_rebalanced.sh
```

Run L training launchers are retained for Qwen, Llama, and Mistral. Current stored Llama/Mistral matrices can be regenerated from a matching adapter with:

```bash
SFT_LORA_DIR=/absolute/path/to/the-matching-adapter/best_dev_f1 \
  bash slurm/sharanga/examples/launch_llama_run_l_benchmark_matrix.sh
```

Use the corresponding model-specific launcher for Mistral or Qwen. Qwen Run K matrix launchers are also retained, but the original dedicated Run K Qwen training-submission wrapper was not present locally; the reconstructed training path must not be described as byte-for-byte identical to the original job.

### 2. Answer-only SFT

The answer-only family trains the same evidence-grounded answer/refusal decision without visible document notes, conflict labels, or a public `<think>` block. The established updated-split recipe uses the canonical 862/81 split and a 2:1 final-only/minimal prompt mixture:

| Prompt family | Copies per source example | Message rows |
| --- | ---: | ---: |
| Final-only | 8 | 6,896 |
| Minimal | 4 | 3,448 |
| **Total** | **12** | **10,344** |

The active standalone release is [answer_only_sft_export/](answer_only_sft_export/), with its own README, runners, validation, retained artifacts, and a portable [ZIP archive](answer_only_sft_export.zip). It also includes a deliberately simple **basic 862-example answer-only baseline**—one final-only target per canonical training example, no message duplication, weights, drills, trace targets, or derived K/L rows. That baseline is designed for a transparent future recipe-level comparison and currently has no stored evaluation results.

The established answer-only checkpoint-selection behavior is abstention-focused rather than semantic-F1-focused despite a historical directory name. This is a material disclosure; see [METHOD_LIMITATIONS.md](docs/METHOD_LIMITATIONS.md).

## Inference and benchmark matrix

The held-out benchmark matrix crosses four information conditions with three trace prompt profiles:

| Prompt mode | What the model receives | Scientific role |
| --- | --- | --- |
| `e2e` | Query and retrieved documents only | Main end-to-end, deployment-faithful condition |
| `oracle_conflict` | Gold conflict label in addition to evidence | Conditional Stage-2 diagnostic |
| `oracle_notes` | Gold document notes in addition to evidence | Conditional Stage-1 diagnostic |
| `oracle_both` | Gold document notes and conflict label | Conditional upper bound for later stages |

The full trace matrix contains 12 prompt conditions per model variant—4 modes × 3 profiles—and therefore 24 baseline/SFT rows per model family. Oracle conditions must be reported as diagnostic interventions, never as ordinary end-to-end performance.

Inference uses deterministic decoding by default (temperature 0), context-aware continuation budgets, sentinel-based stopping, and contract-aware retries that increase only the token budget for structurally incomplete outputs. Every generation produces a separate raw model-output JSONL and sanitized derivative JSONL; the raw artifact is preserved for audit. See [INFERENCE_WORKFLOW.md](docs/INFERENCE_WORKFLOW.md) for the complete workflow.

Rebuild the held-out message matrix:

```bash
bash slurm/examples/rebuild_benchmark_messages_holdout_736_matrix.sh
```

## Post-inference evaluation

The repository evaluates a chain of behaviors rather than relying on one score:

| Evaluation layer | Main measures |
| --- | --- |
| Response contract | Trace structure, document coverage/order, allowed labels, sentinel, in-range citations |
| Document evidence judgments | Micro accuracy, macro F1, per-class precision/recall/F1, confusion matrix for supports / partially supports / irrelevant |
| Conflict-type prediction | Accuracy, valid-label support, per-class metrics, distributions, and five-way confusion matrix |
| Abstention calibration | Accuracy, false abstentions, missed refusals, refusal precision/recall/F1, specificity |
| Citation discipline | Citation validity, counts, unique citations, and sentence-level coverage |
| Final answer | Token F1 and Rouge-L lexical proxy metrics, plus per-example records for manual analysis |

Coverage is essential when reading results: conflict accuracy uses valid parsed labels, document metrics use matched document pairs, and lexical overlap is scored only on usable gold-answer/non-abstaining pairs. Contract completion and support counts should therefore accompany downstream metrics. Full definitions, denominators, and limitations are in [POST_INFERENCE_EVALUATION.md](docs/POST_INFERENCE_EVALUATION.md).

The standard evaluator can be invoked through the scheduler runner after generation. Stored matrices already include report JSON files and per-ID records under `final_model_outputs/`.

## Documentation map

The root README is the navigation layer. Detailed methodology belongs in the documents below.

| Document | Use it for |
| --- | --- |
| [SFT_DESCRIPTION.md](docs/SFT_DESCRIPTION.md) | Full SFT history, current recipes, optimization configuration, rationale, model-specific status, and limitations |
| [PROMPT_DESIGN_AND_ABLATIONS.md](docs/PROMPT_DESIGN_AND_ABLATIONS.md) | Strict/runtime/minimal/final-only prompts, oracle modes, and prompt-robustness rationale |
| [ABLATION_TYPES.md](docs/ABLATION_TYPES.md) | Taxonomy of the different ablation families and their appropriate causal interpretation |
| [ABLATION_STUDY.md](docs/ABLATION_STUDY.md) | D–L ablation chronology, observed trade-offs, and paper-safe conclusions |
| [INFERENCE_WORKFLOW.md](docs/INFERENCE_WORKFLOW.md) | Matrix construction, baseline/SFT generation, retries, raw/sanitized outputs, and adapter-backed replay |
| [POST_INFERENCE_EVALUATION.md](docs/POST_INFERENCE_EVALUATION.md) | Contract, document, conflict, abstention, citation, and final-answer metrics |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Reviewer-oriented reproduction instructions and missing-adapter disclosure |
| [METHOD_LIMITATIONS.md](docs/METHOD_LIMITATIONS.md) | Checkpoint-selection and provenance limitations that must accompany claims |
| [dataset audits](docs/dataset_audits/) | Exact local train/validation and benchmark distribution audits |
| [current experiment records](docs/experiments/) | Run K/L recipe details, mixture tables, and Qwen matrix audit |
| [release checklist](docs/RELEASE_CHECKLIST.md) | Author-owned actions required before a public archival release |

## Repository layout

```text
code/
  data/                 Canonicalization and chat-message preparation
  train/                QLoRA SFT implementation
  eval/                 Generation, sanitization, and post-inference evaluators
data/
  splits/               Canonical 862/81 data, Run J/K/L provenance, and 736 holdout
  messages/             Rebuilt supervised and benchmark message JSONL files
prompts/                Strict, runtime, minimal, final-only, and oracle templates
scripts/                Split construction, message mixing, weighting, and validation utilities
slurm/                  Cluster training, generation, evaluation, and matrix launchers
final_model_outputs/    Stored raw/sanitized outputs, reports, and per-example artifacts
answer_only_sft_export/ Standalone answer-only SFT release surface
docs/                   Method, ablation, inference, evaluation, and reproducibility documentation
legacies/               Preserved historical assets; not needed for the current workflow
```

## Verification

Before a training or reproduction run, validate the current message products:

```bash
python3 scripts/check_trace_text_messages.py \
  data/messages/train_stagewise_prompt_robust_trace_text_k_short_context_targeted_messages.jsonl \
  data/messages/train_stagewise_prompt_robust_trace_text_l_boundary_rebalanced_messages.jsonl

python3 scripts/check_trace_text_messages.py --forbid_think --forbid_task_prefix \
  data/messages/train_stagewise_answer_only_matched_f_messages.jsonl \
  data/messages/val_stagewise_answer_only_final_only_messages.jsonl \
  data/messages/val_stagewise_answer_only_minimal_messages.jsonl
```

Expected message-row counts are 12,659 for Run K, 13,349 for Run L, and 10,344 for the established weighted answer-only mixture. The final benchmark contains 736 examples. Rebuilding K/L temporarily restores the canonical 862/81 compatibility split files on exit; this behavior is intentional.

## Stored artifacts and release limitations

The repository preserves complete stored evaluation artifacts—raw outputs, sanitized outputs, report JSON, and per-example records—but not the current adapters required for exact adapter-only re-inference. Historical local adapters under `legacies/` must not be substituted for current reported result families.

Before public release or ACL/ARR archival submission, authors should complete the author-owned items in [RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md), including licensing, citation metadata, redistribution rights, adapter release/checksums if exact inference is required, model revision information, and cluster/runtime details.

## Legacy archive

[legacies/](legacies/README.md) preserves relocated historical files and their SHA-256 manifest. Removing that one directory yields the intended clean current-facing submission surface while retaining a recoverable provenance archive in the working copy.
