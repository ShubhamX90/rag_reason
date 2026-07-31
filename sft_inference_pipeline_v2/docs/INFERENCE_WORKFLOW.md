# Inference Workflow and Reproduction Guide

## Purpose and scope

This document explains how the repository produces model outputs for the trace-text and answer-only RAG systems. It covers the logical inference workflow, the benchmark matrix, generation policy, raw-versus-sanitized artifacts, and the conditions needed to reproduce the current stored outputs.

It does not restate the SFT recipe or define every evaluation metric in detail. Those are documented in [SFT_DESCRIPTION.md](SFT_DESCRIPTION.md) and [POST_INFERENCE_EVALUATION.md](POST_INFERENCE_EVALUATION.md). It also does not imply that current reported adapters are included: the repository contains the exact prompt/data/evaluation contracts and stored generations, but the latest LoRA adapters are absent. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [METHOD_LIMITATIONS.md](METHOD_LIMITATIONS.md).

## 1. What inference means in this project

At inference time, the system receives a prebuilt chat message containing:

- a system instruction that specifies the evidence policy and response contract;
- a user message with the query and retrieved document snippets; and
- in oracle diagnostic modes only, gold intermediate annotations.

The model generates a continuation. It is either:

- a **baseline** continuation from the unchanged instruction-tuned base model; or
- an **SFT** continuation from the same base model with a separately loaded LoRA adapter.

This shared generation path is crucial for a fair base-versus-SFT comparison. The model family, benchmark messages, generation settings, post-processing, and evaluation pathway are held aligned; the presence of the SFT adapter is the main model-state difference.

The high-level lifecycle is:

```text
canonical benchmark example
        ↓
prompt/profile-specific message construction
        ↓
baseline model OR base model + LoRA adapter
        ↓
deterministic generation with contract-aware length handling
        ↓
raw generation JSONL (never overwritten)
        ↓
sanitized generation JSONL (separate auditable derivative)
        ↓
post-inference reports and per-example records
```

## 2. Canonical inputs and invariants

### Benchmark

The current final benchmark is `data/splits/benchmark_final_v2_holdout_clean_736.jsonl`, containing 736 held-out examples. Its document IDs are normalized as `d1`, `d2`, and so on. The benchmark must not be replaced with a training or validation split when reproducing final numbers.

The benchmark message preparation step constructs system-plus-user records only; it does not expose an assistant target to the generator. Each prompt condition writes its own message JSONL under `data/messages/`.

### Model state

| Variant | Required model state | Interpretation |
| --- | --- | --- |
| Baseline | The matching base instruction model | Untuned reference behavior |
| SFT | The same base instruction model plus the named LoRA adapter | Fine-tuned behavior for the corresponding run |

The adapter must match the base model and the run being reproduced. For example, a Qwen 7B Run K adapter cannot be substituted with a historical answer-only adapter, a Qwen 32B adapter, or a Run L recipe adapter. The directory names in older artifacts are not sufficient provenance by themselves; use the stored result-family name, model name, and recipe documentation together.

### Prompt condition

Each output is identified by both a **prompt mode** and an **instruction-strength profile**. Their definitions are in [PROMPT_DESIGN_AND_ABLATIONS.md](PROMPT_DESIGN_AND_ABLATIONS.md).

| Dimension | Values | Role at inference |
| --- | --- | --- |
| Prompt mode | `e2e`, `oracle_conflict`, `oracle_notes`, `oracle_both` | Determines whether the model must infer intermediate information or receives gold diagnostic inputs |
| Trace profile | strict/default, runtime, minimal | Controls the amount of procedural prompt scaffolding |
| Answer-only profile | final-only, minimal | Controls whether final-answer-only behavior is explicitly requested or tested under sparse instruction |

For the main trace benchmark, all four prompt modes are crossed with strict, runtime, and minimal profiles. That makes 12 conditions for the base model and the same 12 for SFT: a 24-row matrix per model family. The primary deployable result is always an `e2e` row. Oracle rows are conditional diagnostics, not ordinary inference settings.

## 3. Preparing the inference matrix

The current matrix builder reconstructs messages for every held-out example and every trace prompt condition. It produces the following tag convention:

| Profile | Message tag | Why a separate tag exists |
| --- | --- | --- |
| Strict/default | `strict` | Detailed trace contract and teacher-style rules |
| Runtime | `trace_text` | Compact practical trace contract |
| Minimal | `minimal` | Sparse prompt used to test internalization |

For each tag, the builder produces four files, one for each prompt mode. The expected file pattern is:

```text
data/messages/<benchmark>_<prompt-mode>_<tag>_messages.jsonl
```

The held-out matrix is rebuilt with:

```bash
bash slurm/examples/rebuild_benchmark_messages_holdout_736_matrix.sh
```

This is a prompt/data preparation action, not model inference. Rebuilding messages verifies that the exact prompt templates, document normalization, and oracle annotations are available before a GPU job is submitted.

## 4. Generation policy

### Deterministic default

All current matrix launchers use temperature `0.0` by default. This disables sampling: the model follows deterministic decoding subject to the runtime stack and hardware. The default top-p value is `1.0`, but it has no effect while sampling is disabled.

The purpose is reproducible comparison, not creative diversity. A stochastic run should be named and reported as a different inference condition rather than silently replacing the stored deterministic matrix.

### Chat formatting compatibility

The generator renders the system and user records using the base model's native chat template. It contains compatibility paths for local model repositories whose templates are missing or require text-part messages, and a final instruction-style fallback for models that cannot apply the native template. Mistral-family loading also has a compatible tokenizer path for versions that use the Mistral common tokenizer.

These compatibility measures make heterogeneous model families runnable from the same workflow. They are not intended to make prompts identical at the token level across architectures; each base model's own chat format remains part of the inference condition. A fair comparison therefore holds the model's native template fixed across its baseline and SFT variants.

### Context and output-length management

Trace output length depends strongly on the number of retrieved documents because Stage 1 must enumerate every document. Rather than giving all examples the same fixed continuation allowance, the generator can estimate a budget from the number of document IDs in the user message.

The current policy uses a base budget with a capped document-count adjustment. The standard matrix settings are:

| Profile | Base continuation budget | Maximum continuation cap | Rationale |
| --- | ---: | ---: | --- |
| Strict/default | 1,400 | 3,200 | Detailed document-level trace needs the largest allowance |
| Runtime | 1,200 | 2,200 | Compact trace still needs per-document coverage |
| Minimal | 900 | 1,800 | Sparse-prompt condition has a smaller nominal budget |
| Final-only answer-only | configured by its launcher | configured by its launcher | No public trace, so response is normally shorter |

The input is left-truncated only when necessary to fit the model context after reserving continuation space. This is a practical safety mechanism, not a preprocessing transformation of the benchmark. Since truncation can affect groundedness, runs should preserve their configuration and raw output artifacts so a reviewer can audit any unexpected behavior.

### Sentinel stopping

The trace and answer-only target formats end with `[[END-OF-ANSWER]]`. Generation uses this sentinel as a stopping criterion when it is emitted. This provides an explicit end condition beyond an arbitrary token count and aligns training, generation, sanitization, and evaluation around the same response boundary.

## 5. Contract-aware retries

Generation can retry an example if the output is structurally incomplete. Retries are deterministic: they do not resample a different answer. They only allocate more continuation tokens to the same input when the first attempt did not satisfy the configured contract.

| Contract mode | Success requirement used during generation | Current use |
| --- | --- | --- |
| `trace` | One complete `<think>...</think>` block, a recognized conflict label, and the end sentinel | Strict/default and runtime trace conditions |
| `final` | Nonempty final answer plus the end sentinel | Final-only answer-only conditions |
| `none` | No structural retry gate | Minimal trace profile by default |

The usual policy permits one extra attempt, scales the continuation budget by 1.6, and caps retries at 3,200 new tokens. The minimal profile intentionally defaults to no retry gate because the prompt does not itself request the full trace structure. This prevents the generation wrapper from turning the minimal condition into a hidden strict-prompt condition.

Retries should not be interpreted as a model-quality metric by themselves. They are an execution safeguard against avoidable token-budget truncation. Contract completion, retry behavior, and raw outputs remain auditable in the result artifacts.

## 6. Raw outputs, sanitization, and auditability

Each generation job writes two separate JSONL artifacts:

| Artifact | Content | Role |
| --- | --- | --- |
| `*.raw.jsonl` | Direct decoded continuation for each benchmark ID | Primary model output; never overwritten |
| `*.sanitized.jsonl` | A separately written normalized derivative | Input to the standard evaluators |

Maintaining both artifacts is scientifically important. The sanitized file does not erase the raw model behavior; it makes narrowly defined contract repairs visible and allows the evaluation code to operate on a stable representation. Papers and analyses should not imply that a sanitized response is an unmodified model continuation. Where sanitization could materially affect a conclusion, report raw and sanitized behavior separately.

### What trace sanitization does

For outputs with a parseable `<think>...</think>` span, the sanitizer may:

- normalize document-array order and canonical document IDs using the canonical benchmark;
- normalize verdict and conflict-label aliases;
- clear key facts for documents marked irrelevant;
- trim overlong diagnostic fields;
- remove out-of-range citations;
- add fallback citations to uncited final-answer sentences when grounded document items are available;
- canonicalize the refusal text; and
- append the end sentinel when the trace is otherwise present but lacks it.

It is intentionally conservative about absent traces: it does **not** invent a trace when the model never produced one. A record without a usable think block passes through unchanged. In particular, final-only answer-only outputs do not contain a think block and are not rewritten into trace outputs by the sanitizer.

The standardized evaluation workflow uses the sanitized file, while the raw file remains available for inspection. This dual-artifact design lets reviewers distinguish model generation failures from format-level repair and detect whether a result depends on sanitization.

## 7. Running a trace-text benchmark matrix

The model-specific matrix launchers submit a generation job and then an evaluation job with an `afterok` dependency for each cell. Each cell has a distinct output name composed from:

```text
<baseline-or-sft>_<model>_<strategy>_<run-name>_<mode>_<profile-tag>_<benchmark>
```

This naming preserves the factors needed to interpret an artifact later: model variant, base family, training strategy, recipe/run, information condition, prompt profile, and dataset.

| Stored result family | Current matrix launcher | Recipe represented by stored matrix |
| --- | --- | --- |
| Qwen 2.5 7B | [launch_qwen7b_run_k_benchmark_matrix.sh](../slurm/sharanga/examples/launch_qwen7b_run_k_benchmark_matrix.sh) | Run K |
| Qwen 2.5 32B | [launch_qwen32b_run_k_benchmark_matrix.sh](../slurm/sharanga/examples/launch_qwen32b_run_k_benchmark_matrix.sh) | Run K |
| Llama 3.1 8B | [launch_llama_run_l_benchmark_matrix.sh](../slurm/sharanga/examples/launch_llama_run_l_benchmark_matrix.sh) | Run L |
| Mistral 7B | [launch_mistral_run_l_benchmark_matrix.sh](../slurm/sharanga/examples/launch_mistral_run_l_benchmark_matrix.sh) | Run L |

For Qwen Run K, the release retains exact split and message construction plus the matrix/evaluation launchers, but not the original dedicated training-submission wrapper. For Run L, training and matrix launchers are retained for Qwen, Llama, and Mistral; only Llama and Mistral have completed Run L matrices stored locally.

An adapter-backed reproduction should explicitly provide `SFT_LORA_DIR`, for example:

```bash
SFT_LORA_DIR=/absolute/path/to/the-matching-run-k-adapter/best_dev_f1 \
  bash slurm/sharanga/examples/launch_qwen7b_run_k_benchmark_matrix.sh
```

The launcher will generate baseline and SFT outputs using the same matrix message files unless one of those variants is intentionally disabled. It does not treat an archived adapter as a substitute for the current reported adapter.

## 8. Answer-only inference

Answer-only inference is logically simpler but must not be mixed with trace evaluation claims.

| Property | Trace-text inference | Answer-only inference |
| --- | --- | --- |
| Expected visible output | Three-stage evidence trace plus cited answer/refusal | Cited final answer or controlled refusal only |
| Normal contract mode | `trace` for strict/runtime, `none` for minimal | `final` for final-only |
| Relevant evaluator | Contract, document verdict, conflict type, final answer | Primarily final answer; trace evaluators are not meaningful |
| Current stored profiles | strict/runtime/minimal across trace modes | final-only and minimal end-to-end outputs |

The established updated answer-only outputs are stored in `final_model_outputs/answer_only_sft/` for Qwen 7B, Llama 8B, and Mistral 7B. Their exact recipe and limitations are in the standalone [answer-only reproduction guide](../answer_only_sft_export/docs/REPRODUCE_LATEST.md). The clean basic 862-example baseline has its own benchmark messages and launchers but no completed result artifacts yet; it must not be represented as an evaluated result.

## 9. Cluster execution environment

The supplied inference launchers target a SLURM GPU environment. They expect:

- a Conda environment with the pinned project dependencies;
- accessible local directories for the base model and, for SFT, the LoRA adapter;
- writable working, output, cache, and log directories; and
- a GPU partition suitable for the selected model size.

The Sharanga environment helper resolves paths such as `WORK_ROOT`, `MODEL_ROOT`, `OUTPUT_ROOT`, and Hugging Face caches from the user's scratch area unless explicitly overridden. The matrix launchers choose their documented default partitions, but hardware availability is a scheduler concern rather than part of the experimental claim. Reproduction on a different cluster should set these locations explicitly instead of relying on the original filesystem layout.

The standard trace matrix generation uses BF16 and normally loads the base model without 4-bit inference quantization. The underlying generator also supports 4-bit loading, which is useful for constrained baseline experiments; it should be treated as a named inference configuration, not silently mixed with a BF16 result family.

## 10. Reproducing and verifying a stored result family

### What can be reproduced from the release

| Stage | Available locally? | Notes |
| --- | --- | --- |
| Benchmark and prompt-message construction | Yes | Deterministic file-level recreation of the matrix messages |
| Base-model generation | Yes, after obtaining the named base weights | Depends on the model revision and runtime environment |
| SFT generation of the reported matrices | Yes, after supplying the matching current adapter | Latest adapters are not bundled |
| Evaluation of supplied sanitized outputs | Yes | Reports and per-example artifacts are stored |
| Exact end-to-end training-to-generation replay | Not fully | Depends on missing current adapters and external training environment |

### Minimal verification sequence

1. Configure `PROJECT_ROOT`, `WORK_ROOT`, and `MODEL_ROOT`; install the pinned environment.
2. Check the held-out benchmark and reconstruct the matrix messages.
3. Run a small `LIMIT` smoke generation with a named base model to validate the model/template/environment path.
4. If the matching adapter is available, run the intended baseline/SFT matrix launcher with a new run name or separate output root.
5. Preserve both raw and sanitized outputs, then compare the generated reports and per-ID coverage with the stored artifacts.

Do not overwrite the checked-in `final_model_outputs/` during reproduction. Use a separate output root or run name. This avoids confusing supplied reference artifacts with newly generated outputs and makes numerical differences auditable.

## 11. Interpretation and reproducibility limits

Inference comparisons are only meaningful when the following factors are named:

- base model and revision;
- whether a LoRA adapter was loaded, and which one;
- prompt mode and profile;
- benchmark version;
- temperature and sampling policy;
- continuation/retry configuration;
- raw versus sanitized artifact; and
- evaluation support and contract coverage.

The pipeline makes prompt and message reconstruction deterministic at the file level. Exact training and adapter-level reproduction remains subject to unavailable current adapters, external base-model revisions, CUDA/driver versions, and distributed-training nondeterminism. The stored matrices are therefore complete evaluation artifacts, while a newly trained adapter should be reported as a new replication unless its provenance can be verified against the original run.

## 12. Artifact map

| Artifact | Role |
| --- | --- |
| [rebuild_benchmark_messages_holdout_736_matrix.sh](../slurm/examples/rebuild_benchmark_messages_holdout_736_matrix.sh) | Builds the 4-mode × 3-profile held-out message matrix |
| [generate_experiment.sh](../slurm/sharanga/generate_experiment.sh) | Shared baseline/SFT generation and raw/sanitized output writing |
| [evaluate_experiment.sh](../slurm/sharanga/evaluate_experiment.sh) | Standard post-generation evaluation submission |
| [final_model_outputs/](../final_model_outputs/) | Stored current matrices, raw/sanitized JSONL, reports, and per-ID artifacts |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, adapter availability, and replay limitations |
| [PROMPT_DESIGN_AND_ABLATIONS.md](PROMPT_DESIGN_AND_ABLATIONS.md) | Definition and interpretation of prompt matrix conditions |
| [answer_only_sft_export/](../answer_only_sft_export/) | Standalone answer-only runners, inputs, and retained artifacts |

## 13. Experimental controls at inference time

An inference comparison is not fair merely because two output files have the same number of lines. For a baseline-versus-SFT result, the following are aligned within a matrix cell:

| Factor | Baseline | SFT |
| --- | --- | --- |
| Base architecture and instruction checkpoint | Same named base model | Same named base model |
| Benchmark examples and document order | Same message JSONL | Same message JSONL |
| Prompt mode and profile | Same cell | Same cell |
| Decoding policy and token budget | Same cell configuration | Same cell configuration |
| Raw/sanitized processing and evaluator | Same shared path | Same shared path |
| Additional learned state | None | The named LoRA adapter only |

Thus the within-cell comparison is an adapter intervention under fixed base model, retrieval context, prompt, decoding, and post-processing conditions. Comparisons across model families, prompt profiles, oracle modes, or training-recipe names answer different questions and must not be compressed into one unnamed “best model” score.

### No hidden target exposure in normal inference

The benchmark generation records contain system and user turns only. They do not include the assistant training target, expected final answer, abstention label, or gold evidence list in normal `e2e` inference. Oracle modes are the deliberate exception: they inject a specific gold intermediate annotation so that the remaining stages can be studied conditionally. This is why an oracle score must never be reported as an ordinary deployment score.

## 14. Length, retry, and sanitization boundaries

### Why generation budgets vary by document count

A fifteen-document trace has a larger minimum valid output than a five-document trace because Stage 1 must cover every document. A fixed continuation limit would selectively truncate long-context examples and confound reasoning quality with output-space availability. The automatic budget therefore uses the number of document IDs in the actual user message, with a profile-specific base and a cap. It does not change the retrieved content or supply additional reasoning help; it only reserves a plausible amount of continuation space.

This policy does not guarantee a complete response. It makes a complete response possible, while contract completion remains an independently measured outcome. Input text is left-truncated only if needed to fit the model context after that continuation space is reserved, so raw artifacts and run configuration remain essential for auditing any unexpected failure.

### What a retry can and cannot change

When the configured contract is unmet, the generator can re-run the *same deterministic prompt* with a larger new-token budget. It does not sample a new answer, append corrective feedback, change the retrieved documents, substitute a prompt profile, or alter the model state.

| Retry policy | Permitted | Not permitted |
| --- | --- | --- |
| Trace/runtime/strict | Increase completion budget for the same input | Semantic rewriting, prompt repair, evidence changes, stochastic resampling |
| Final-only answer-only | Increase completion budget for the same input | Adding a trace or intermediate labels |
| Minimal trace | No structural retry gate by default | Turning a sparse-prompt condition into a hidden strict condition |

Retries are consequently an execution safeguard against preventable token-limit truncation, not a semantic self-correction method. It would be invalid to retry only failed SFT rows with a more permissive policy while retaining baseline outputs from the original budget.

### Sanitization: explicit repair boundary

Sanitization is a deterministic representation-level operation applied after raw generation. When a usable trace exists, it can canonicalize content the model already emitted: document order/IDs, label aliases, irrelevant-document key facts, overlong diagnostic fields, out-of-range citations, refusal spelling, or a missing terminal sentinel. It may add a fallback citation only from document items that occur in the model-generated trace.

It cannot invent a missing trace, construct absent document judgments from the canonical benchmark, write a new conflict analysis, supply a missing answer, or convert a final-only answer into trace text. A record with no usable think block remains unchanged. The sanitized file retains the same JSONL schema as raw generation and stores its normalized continuation in the `raw` field; the separate filename identifies it as a derivative.

This boundary is central to result interpretation. The sanitized artifact is the standard evaluator input, but it is not an untouched continuation. The raw artifact remains the primary record of model behavior. Any conclusion sensitive to a repair category should be checked and, where material, reported for both raw and sanitized outputs.

## 15. Completion, failure analysis, and paper reporting

### Completion criterion for one matrix cell

Completion means more than a generation job finishing successfully.

| Lifecycle stage | Required artifact | What it establishes |
| --- | --- | --- |
| Input preparation | Prompt/mode-specific message JSONL | Correct benchmark, prompt profile, and oracle condition were instantiated |
| Generation | `*.raw.jsonl` | A direct continuation exists for each generated ID |
| Normalization | `*.sanitized.jsonl` | Standard evaluator input exists without discarding raw output |
| Structural report | `contract.json` | Contract compliance and evaluation support are quantified |
| Intermediate reports | `doc_verdicts.json`, `conflict_type.json` | Stage-level behavior is measured where output contains those stages |
| Final report | `final_answer.json`, `final_answer_per_id.jsonl` | Answer/refusal behavior and per-example audit trail are present |

A score calculated on a reduced valid subset is a useful diagnostic, but not a substitute for a complete 736-example end-to-end result. The trace matrices contain 24 cells per model family: 12 baseline and 12 SFT cells. The answer-only family contains final-only and minimal end-to-end conditions rather than this 12-condition trace diagnostic grid, because it intentionally omits intermediate trace stages.

### Failure taxonomy

| Symptom | Primary layer to inspect | Correct reading |
| --- | --- | --- |
| Missing think block or sentinel | Prompt following, length, or contract | Inspect raw output and generation budget before attributing it to reasoning |
| High abstention accuracy but many false abstains | Evidence-sufficiency calibration | The model is too cautious on answerable examples |
| High conflict score with low support | Parsing/selective validity | Do not compare it to a full-support row as if equally reliable |
| Near-perfect document score in oracle-notes mode | Gold notes were supplied | Conditional upper bound, not end-to-end Stage-1 competence |
| Strong answer overlap with invalid citations | Grounding/contract | The response is not fully evidence-grounded under this project’s standard |
| Material raw/sanitized difference | Post-processing sensitivity | Audit the repair scope and avoid calling the sanitized text raw generation |

The recommended diagnostic order is: structural validity first; answer/refusal calibration second; document and conflict reasoning third; answer-overlap measures last. This avoids misreading an unparsable output as a low-quality conflict decision or treating a selective high score as general performance.

### Paper-ready description

For the trace-text matrices, the following statement is accurate:

> We evaluate baseline and LoRA-adapted models on a fixed 736-example held-out retrieval benchmark. For each model family, we cross four information conditions—end-to-end, oracle conflict label, oracle document notes, and both oracles—with three instruction strengths—strict, runtime, and minimal. Generation uses temperature-zero decoding, profile-specific continuation budgets, sentinel-aware stopping, and deterministic length-only retries where the contract requests them. Raw continuations are preserved, and a separately sanitized derivative is evaluated with contract validity and evaluation support reported alongside reasoning and answer metrics.

For answer-only inference, report only the final-only/minimal conditions and final-answer-focused evaluation. A trace evaluator’s expected lack of support on answer-only output is not an answer-only performance result.
