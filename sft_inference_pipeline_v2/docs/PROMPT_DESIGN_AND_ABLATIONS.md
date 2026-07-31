# Prompt Design, Prompt-Robust Training, and Oracle Ablations

## Purpose and scope

This document explains the prompt design used by the current conflict-aware RAG pipeline. It is intended to support a paper methods section, an ablation section, and reproducible interpretation of the stored benchmark matrices. It explains the *logical roles* of the prompt conditions rather than duplicating every template verbatim.

The central design question is not simply, “which instruction makes a model answer?” The project asks whether a model can use retrieved evidence to make a sequence of linked decisions:

1. assess the contribution of every retrieved document;
2. distinguish agreement, complementary evidence, and several kinds of conflict;
3. decide whether the evidence is sufficient to answer;
4. synthesize a grounded answer or a controlled refusal; and
5. reliably expose that behavior in a machine-checkable public response format.

The prompt suite deliberately varies how much of this procedure is restated at inference time. Consequently, a strong result under a sparse prompt is evidence that more of the protocol has been internalized by SFT; a strong result under an oracle prompt instead measures a conditional upper bound for a particular reasoning stage. Those are different claims and must not be conflated.

For the full SFT history and current recipe, see [SFT_DESCRIPTION.md](SFT_DESCRIPTION.md). For the actual template text, see [prompts/](../prompts/). For current Run K/L recipe details, see [Run K](experiments/sft_run_k_short_context_targeted_strategy_and_results.md), [Run L](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md), and the [K/L mixture tables](experiments/run_k_l_training_recipe_tables.md).

## 1. The two independent prompt axes

The evaluation setup varies two axes. Keeping them separate is essential when reading the results.

| Axis | Values | What it changes | What it is for |
| --- | --- | --- | --- |
| **Information condition / prompt mode** | `e2e`, `oracle_conflict`, `oracle_notes`, `oracle_both` | Whether the model must infer intermediate evidence judgments and/or conflict type itself | Localize where end-to-end errors originate |
| **Instruction strength / prompt profile** | strict/default, runtime, minimal | How much of the desired procedure and output contract is restated in the prompt | Measure prompt robustness and deployment realism |

This creates a 4-by-3 diagnostic grid. For each model variant (base or SFT), the standard current matrix has 12 prompt conditions. Comparing base and SFT across the same grid yields 24 rows per model family.

| Prompt mode | Strict/default | Runtime | Minimal |
| --- | --- | --- | --- |
| `e2e` | Full end-to-end reasoning with detailed instructions | End-to-end reasoning with a compact deployment-like instruction | End-to-end reasoning under sparse instruction |
| `oracle_conflict` | Stage-2 label supplied; evidence judgments still inferred | Same information condition with a compact contract | Sparse conditional probe |
| `oracle_notes` | Stage-1 document notes supplied; conflict label still inferred | Same information condition with a compact contract | Sparse conditional probe |
| `oracle_both` | Both Stage 1 and Stage 2 inputs supplied | Same information condition with a compact contract | Sparse conditional probe |

The main reported system condition is always **`e2e`**. The remaining three modes are diagnostic interventions. They answer questions such as: “If the conflict label were already correct, could the model produce a good answer?” They do not show that the end-to-end model can infer that label in normal use.

## 2. Shared behavioral contract

Across trace-oriented profiles, the system asks for a public three-stage evidence trace followed by a final answer.

| Stage | Required decision | Why it is exposed |
| --- | --- | --- |
| Stage 1: evidence assessment | For each retrieved document, classify it as `supports`, `partially supports`, or `irrelevant`; give a brief snippet-grounded rationale, an entailed key fact when applicable, and source quality | Makes document use auditable and allows document-level evaluation |
| Stage 2: conflict assessment | Choose one of five conflict types and give a concise reason and evidence pattern | Separates disagreement structure from answerability |
| Stage 3: answer plan | State briefly how the evidence supports an answer or why the evidence requires refusal | Connects intermediate assessment to the final decision |
| Final answer | Provide a grounded answer with document citations, or the exact controlled refusal | Makes the externally useful behavior explicit and evaluable |

The trace must use one `<think>...</think>` block and end with `[[END-OF-ANSWER]]`. Non-refusal answers use in-range end-of-sentence document citations such as `[d2]`; the strict contract requests citation coverage for at least 80% of answer sentences. Refusals use the exact string `CANNOT ANSWER, INSUFFICIENT EVIDENCE` and do not carry citations.

These requirements are not cosmetic. The trace turns a vague request to “reason over retrieved documents” into observable, separately measurable decisions. The sentinel makes it possible to distinguish an incomplete generation from a completed one. Canonical document IDs make citation checking mechanical. The strict refusal string removes ambiguity between a cautious answer and an evidence-based abstention.

### The evidence policy

All current profiles encode the same high-level epistemic policy:

- retrieved snippets are the only permitted evidence; the model must not fill gaps with parametric or external knowledge;
- a document *supports* only if its snippet directly and decisively answers the needed aspect of the query;
- a document *partially supports* if it is relevant but incomplete, indirect, hedged, or missing a necessary detail;
- evidence is insufficient only when the retrieved set, considered jointly, leaves a necessary gap;
- the presence of disagreement is not itself grounds for refusal; and
- multiple partial documents can jointly make an answer possible.

The last two rules are especially important. A retrieval set may contain conflicting sources yet still support a properly framed answer, and no single snippet may resolve a question while several compatible partial snippets do. Without these guards, models can learn an overly conservative shortcut: “uncertainty, conflict, or partiality means refuse.” Runs J, K, and L were explicitly designed to counter that kind of false abstention.

### The conflict taxonomy

Stage 2 always uses exactly one of the following labels:

| Label | Decision rule | Intended final-answer behavior |
| --- | --- | --- |
| No conflict | Relevant documents support the same core answer; differences are only detail or granularity | Answer directly from the strongest consistent evidence |
| Complementary information | Relevant documents provide distinct, compatible facets such as scope, date, subgroup, or mechanism | Combine the facets into one coherent answer |
| Conflicting opinions or research outcomes | Relevant documents make incompatible claims within the same scope and time window | Present the competing positions neutrally unless the evidence itself resolves them |
| Conflict due to outdated information | Older and newer factual claims conflict and newer evidence supersedes the older statement | Prioritize current evidence and acknowledge superseded information when useful |
| Conflict due to misinformation | The retrieved set itself establishes that a weaker claim is false or misleading relative to stronger evidence | Correct the weaker claim using the stronger retrieved evidence |

The taxonomy deliberately distinguishes **complementarity** from **conflict**. Two snippets can contribute different necessary pieces without disagreeing. It also distinguishes general disagreement from disagreements caused by temporal change or a demonstrably weaker claim. These are not merely semantic labels: they prescribe different evidence aggregation and answer framing.

## 3. Instruction-strength profiles

The profile names describe instruction strength, not different source datasets or different labels. Current trace-text SFT deliberately trains across strict/default, runtime, and minimal forms, so that performance is not wholly dependent on one unusually detailed instruction.

| Profile | What the prompt explicitly supplies | Main scientific role | Current use |
| --- | --- | --- | --- |
| **Strict/default** | Detailed definitions, decision thresholds, source guidance, common-confusion guards, final self-check, and the exact three-stage response contract | High-control teacher condition and strongest contract check | Trace SFT supervision and benchmark evaluation (`strict` tag) |
| **Runtime** | Compact evidence-only policy, concise three-stage schema, conflict labels, abstention/composition rules, citations, and sentinel | Practical deployment-like condition | Multi-task SFT supervision and benchmark evaluation (`trace_text` tag) |
| **Minimal** | Only the basic evidence-only answering and evidence-sufficiency policy | Stress test of whether SFT internalized the behavior rather than merely echoing instructions | End-to-end SFT exposure and benchmark evaluation (`minimal` tag) |
| **Final-only** | Evidence-only final answer, citations, joint-partial-evidence rule, exact refusal, and sentinel; no public trace | Answer-only baseline family | Answer-only SFT and its inference comparisons, not the main trace matrix |

### 3.1 Strict/default: a detailed behavioral specification

The strict/default profile is intentionally explicit. It specifies the trace format, the five conflict definitions, the three document verdicts, citation rules, abstention policy, and a set of edge-case guards. Examples include threshold questions, date-sensitive questions, exact counts, “latest” questions, ambiguous scope, negative evidence, and source-credibility ordering.

Its design serves three purposes.

First, it makes the intended behavior unambiguous during supervised learning. A model is not asked to infer what “grounded” means from a label alone: it receives an operational definition of how to treat snippets, documents, conflicts, and citations.

Second, it makes format evaluation meaningful. If a model fails the strict condition, the failure could reflect either reasoning or failure to follow a highly specified contract. This is why strict results should be reported with contract coverage, not only answer overlap.

Third, the profile encodes known failure boundaries directly. For example, it prohibits importing background knowledge, prevents treating a partial list as an exact count, and forbids calling historical context outdated unless it actually competes with current evidence. These are high-risk locations for superficially plausible but unsupported answers.

Strict/default is therefore a useful *upper-information prompt condition*, but it is not by itself proof that the model has learned robust reasoning. A model can depend on repeated instructions. The runtime and minimal profiles test whether the learned mapping persists as that scaffolding is removed.

### 3.2 Runtime: the compact operational prompt

The runtime profile retains the conceptual skeleton of the strict prompt while removing detailed edge-case instructions and the long self-check. It still asks the model to use only retrieved documents, output the three-stage trace, choose from the same label set, cite document IDs, abstain only for genuine evidence insufficiency, and combine jointly sufficient partial evidence.

This is the most deployment-like trace condition in the repository. It is compact enough not to dominate the context window or repeatedly teach every rule at generation time, while still declaring the public contract expected by the evaluator.

The central methodological value of runtime prompting is that it separates **learned behavior** from **prompt-carried behavior**. If strict performance is high but runtime performance drops sharply, the model may be relying on the detailed prompt rather than the SFT. If runtime remains strong after SFT, the adapter has likely internalized more of the evidence and formatting policy.

Runtime is also the vehicle for direct component supervision during trace SFT. The model sees runtime variants of the end-to-end trace, the document-verdict task, the conflict-type task, and the answer-only task. This concentrates short, clear supervision on individual decisions while preserving end-to-end training.

### 3.3 Minimal: a test of internalization, not a weaker task

The minimal profile only says, in effect: answer using retrieved documents, combine partial evidence when it jointly resolves the query, and abstain when a necessary gap remains. It does not restate the three-stage trace schema, citations, taxonomy, or sentinel.

The task is deliberately not simplified. The expected trace-text target and the underlying factual problem remain the same; only the instructional scaffolding is withdrawn. A model that produces the intended trace and grounded final answer under this condition has learned a more prompt-invariant mapping from retrieval context to behavior.

This makes minimal prompting scientifically useful but operationally demanding. A base instruction-tuned model may answer in a reasonable natural-language style yet fail the project's formal trace contract. That is a format mismatch, not necessarily evidence of zero factual competence. Conversely, high minimal contract compliance after SFT is evidence of internalized response structure, but must still be read alongside grounding, conflict, and abstention metrics.

Minimal results should therefore never be interpreted from a single headline number. In the stored matrices, rows with a `trace gap`, low contract coverage, or reduced evaluation support must not be promoted as clean end-to-end wins even if one metric is high.

### 3.4 Final-only: the controlled answer-only comparison

The final-only profile tells the model to reveal no reasoning trace and to emit only the final grounded answer with citations or the exact refusal. It retains the evidence-only rule, the joint-partial-evidence instruction, the sentinel, and the same refusal criterion.

This family is intentionally separate from trace-text SFT. It asks whether the final answer behavior can be learned without public intermediate supervision. It is not a shortened version of the main trace system, and its outputs should not be described as evidence traces. The documented answer-only results use their own training recipe and checkpoint-selection disclosure; see [SFT_DESCRIPTION.md](SFT_DESCRIPTION.md) and [METHOD_LIMITATIONS.md](METHOD_LIMITATIONS.md).

## 4. Prompt modes: end-to-end and oracle interventions

Prompt mode changes the information available to the model, while preserving the query and retrieved documents. This is an intervention-style ablation: supplying a gold intermediate variable asks how well later stages would work if that earlier decision did not need to be predicted.

| Mode | Supplied inputs beyond query and retrieved documents | Model must still infer | Appropriate interpretation |
| --- | --- | --- | --- |
| `e2e` | None | Document verdicts, conflict type, sufficiency, answer | The actual end-to-end system condition |
| `oracle_conflict` | Gold conflict type | Document verdicts, sufficiency, answer | Conditional probe of the impact of Stage-2 errors |
| `oracle_notes` | Gold per-document notes | Conflict type, sufficiency, answer | Conditional probe of the impact of Stage-1 errors |
| `oracle_both` | Gold document notes and gold conflict type | Sufficiency and answer synthesis | Conditional upper bound when both intermediate stages are correct |

### 4.1 `e2e`: the only deployment-faithful condition

The end-to-end prompt supplies only the query and the retrieved documents. It is the correct condition for reporting the system's actual capability because it requires the model to perform every upstream decision itself. Any main result, headline comparison, or claimed improvement in RAG reasoning should come from this mode.

### 4.2 `oracle_conflict`: isolate conflict-classification error

This mode gives the gold Stage-2 conflict label and requires the model to copy it exactly. The model still evaluates documents and produces the answer. If this intervention substantially improves final-answer behavior relative to `e2e`, then inaccurate conflict categorization is plausibly a meaningful bottleneck. If it changes little, the limiting factor is more likely evidence judgment, sufficiency calibration, or answer synthesis.

It is not a deployment setting: real retrieval systems do not receive the gold conflict type. It is a conditional diagnostic and should be labelled as such in a paper figure or table.

### 4.3 `oracle_notes`: isolate document-assessment error

This mode provides the per-document ground-truth notes: document order, verdict, key fact, and source quality. The model must preserve these Stage-1 facts in the trace but still infer the conflict type and synthesize the answer.

It measures how much downstream behavior benefits from perfect local evidence assessment. Strong document metrics in an oracle-notes condition do not show that the model can generate those notes independently. They show the behavior that is reachable when the first stage is correct.

### 4.4 `oracle_both`: an upper bound for the remaining decision

This supplies both Stage-1 notes and the Stage-2 label. It tests evidence sufficiency, answer planning, answer synthesis, citations, and contract compliance given correct intermediate structure. The remaining gap relative to an ideal answer estimates errors in the final decision and generation stages; the gap from `e2e` to this condition indicates the value of the supplied intermediate information.

Again, it is an upper-bound diagnostic, not a replacement for end-to-end results. In particular, high oracle conflict accuracy can be numerically misleading if only a small subset of outputs is valid enough to evaluate. Contract coverage and evaluation support must be reported alongside the metric.

## 5. How prompt robustness is trained

The current trace recipes do not train on one prompt and evaluate on unrelated prompts. They deliberately expose the same source examples through several instruction strengths and several task views.

| Supervision view | Prompt profile normally used | What it teaches |
| --- | --- | --- |
| Strict end-to-end trace | Strict/default | The fully specified target behavior and contract |
| Runtime end-to-end trace | Runtime | The same reasoning process under compact instructions |
| Minimal end-to-end trace | Minimal | Trace and evidence policy internalization under sparse instructions |
| Document-verdict task | Runtime | Local evidence assessment independent of full answer generation |
| Conflict-type task | Runtime | Taxonomy decision independent of a long response |
| Answer-only task | Runtime | Direct final-answer/refusal behavior within the trace training mixture |
| Boundary drills | Runtime plus short instructional prefixes | Difficult label or verdict distinctions |
| Partial-synthesis drills | Strict, runtime, and minimal variants | Combining several incomplete but jointly sufficient snippets |

Runs J, K, and L keep this three-profile logic while changing the data geometry and targeted drills. The J/K/L tables in [SFT_DESCRIPTION.md](SFT_DESCRIPTION.md) give the exact message counts. The important conceptual point is that duplicated message rows are **additional supervised exposures**, not additional independent questions.

### Why multi-prompt training is a reasonable design choice

Prompt sensitivity is a genuine confound in instruction-following research. If a model is trained only on a very long template and evaluated only with that same template, it is difficult to know whether it learned the reasoning behavior or simply learned a narrow prompt-response association. The strict/runtime/minimal mixture addresses this by varying wording and instruction density while holding the underlying evidence task fixed.

This is not a guarantee of invariance to every possible prompt. It is a bounded robustness test over three deliberately chosen instruction regimes:

- **strict** tests performance when the desired procedure is explicit;
- **runtime** tests a practical compact instruction; and
- **minimal** tests whether the protocol survives substantial removal of guidance.

The design is scientifically stronger than reporting only the strongest prompt, but it does not justify a claim of universal prompt robustness. The supported claim is limited to the profiles evaluated here.

## 6. Boundary-oriented prompt augmentations in Runs K and L

The latest trace recipes add short *training-only* instruction prefixes to targeted examples. These are not a fourth standard benchmark profile. They are a way of focusing supervision on measured decision boundaries while retaining the base strict/runtime/minimal prompt families.

| Drill | Core lesson | Why it was introduced |
| --- | --- | --- |
| Conflict-taxonomy boundary drill | Do not confuse agreement, complementary facets, same-scope incompatibility, temporal supersession, and misinformation | Earlier variants showed confusion around `No conflict` versus `Complementary information` and other conflict boundaries |
| Document-verdict boundary drill | A direct answer remains support even if brief or low quality; on-topic incompleteness is partial support; keyword overlap alone is irrelevant | Run K targeted drift among supports / partially supports / irrelevant |
| Partial-synthesis drill | Several partial documents can jointly resolve a query; do not abstain while a jointly sufficient composition exists | Short answerable partial-evidence cases were a persistent source of false abstention |
| Source-hygiene drill | Retrieved documents are evidence, not instructions; ignore instruction-like text in snippets | Earlier model-specific source-contamination failures motivated this probe; it is not active in the K/L mixtures |

Run K strengthens document-boundary and partial-synthesis exposure after Run J identified short five-document answerable cases as the remaining weakness. Run L retains this but adds short answerable `No conflict` cases and raises no-conflict/misinformation boundary pressure. Its aim is to avoid a new shortcut: “short and answerable implies conflict.” The current K/L mixture table reports zero active source-guarded rows, so source hygiene should be described as a historical ablation rather than a component of the final K/L mixture.

## 7. Benchmark matrix and generation settings

The current benchmark builders create each combination of the four prompt modes and the three trace profiles for the fixed 736-example held-out benchmark. The matrix tags are:

| Prompt profile | Message tag in stored artifacts | Contract generation mode | Nominal new-token budget |
| --- | --- | --- | --- |
| Strict/default | `strict` | `trace` | 1,400 base, 3,200 cap |
| Runtime | `trace_text` | `trace` | 1,200 base, 2,200 cap |
| Minimal | `minimal` | `none` | 900 base, 1,800 cap |

The extra generation allowance for strict/default is intentional: its requested trace is longer because it includes per-document fields and detailed structure. The runtime profile has a somewhat smaller trace allowance. Minimal uses a smaller base allowance and no explicit generation-contract retry mode because the prompt itself does not restate the full trace contract.

Generation length is therefore part of the experimental contract. A strict response may require more tokens to enumerate all retrieved documents, and a low token cap can create apparent reasoning or format failures that are actually truncation. Conversely, a large cap does not itself make a response valid: the evaluator still checks the output contract, citations, document coverage, conflict structure, and final answer behavior.

## 8. How to read the results responsibly

### Main comparison

For a base-versus-SFT comparison, first compare the same `e2e` profile between model variants. Then check whether the improvement is jointly reflected in:

- response-contract completion;
- abstention accuracy, false abstentions, and missed refusals;
- conflict-type accuracy;
- document-verdict accuracy and support coverage;
- citation validity and coverage; and
- final-answer overlap metrics.

No one metric is sufficient. A model can improve a conflict score by refusing many cases, or achieve a perfect oracle-supplied label by merely copying it while failing to produce a valid trace. This repository's analysis documents such failure modes rather than hiding them.

### What the stored matrices already suggest

The retained Qwen Run K audit shows why prompt condition matters. Qwen 7B SFT end-to-end behavior is generally strong under the three profiles, while base-model rows can have zero contract coverage or heavy over-abstention under some prompt settings. The same audit also shows that oracle conditions can sharply improve component-specific metrics, but this is conditional on gold information and should be interpreted as a diagnostic upper bound. See [qwen7b_run_k_benchmark_matrix_audit.md](experiments/qwen7b_run_k_benchmark_matrix_audit.md).

For Llama 3.1 8B and Mistral 7B, the latest stored full matrices use Run L. Their results should be read using the same hierarchy: `e2e` rows establish end-to-end behavior; the three oracle modes explain where errors remain. The model-specific summaries are [Llama](../final_model_outputs/llama8b_benchmark_matrix_analysis.md) and [Mistral](../final_model_outputs/mistral7b_benchmark_matrix_analysis.md).

### A compact reporting template

For an ACL-style paper, a clear presentation is:

1. Report the principal base-versus-SFT result under `e2e` for each of strict, runtime, and minimal prompts.
2. State explicitly that strict/runtime/minimal vary instruction strength, not the benchmark examples.
3. Report oracle conditions in an ablation table or error-localization figure, labelled “gold intermediate information supplied.”
4. Include contract coverage and number of evaluable outputs near any conditional metric.
5. Avoid presenting the best cell from a 12-condition matrix as the sole model result without naming its prompt mode and profile.

## 9. Limits and defensible claims

The prompt suite supports several careful conclusions:

- Multi-profile SFT tests whether trace behavior survives a controlled reduction in prompt scaffolding.
- Oracle interventions identify whether document assessment and conflict classification are plausible bottlenecks for later-stage answer quality.
- Explicit boundary drills are targeted responses to observed error slices, rather than indiscriminate prompt elaboration.
- The final-only profile is a controlled answer-only comparison, not a trace-reasoning system.

It does **not** establish that the method is robust to arbitrary paraphrases, adversarial prompt injections, languages, retrieval depths, or external retrieval domains. The source-hygiene study is relevant historical evidence, but active K/L training does not contain source-guarded rows. Likewise, oracle conditions do not demonstrate deployment performance because their intermediate labels and notes are not available in an ordinary retrieval pipeline.

The strongest concise claim is: **under the three specified instruction strengths, the trace-text SFT recipe is evaluated for prompt robustness, while oracle variants are used only to diagnose conditional stage-level limits.**

## 10. Artifact map

| Artifact | Role |
| --- | --- |
| [prompts/](../prompts/) | Source prompt templates for every profile and mode |
| [SFT_DESCRIPTION.md](SFT_DESCRIPTION.md) | Full SFT evolution, current mixtures, optimization, and results interpretation |
| [Run K strategy](experiments/sft_run_k_short_context_targeted_strategy_and_results.md) | Short-context targeted recipe and stored Qwen results |
| [Run L strategy](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md) | Boundary-rebalanced recipe and current result status |
| [K/L recipe tables](experiments/run_k_l_training_recipe_tables.md) | Exact message-mixture arithmetic and boundary pressure |
| [Qwen Run K audit](experiments/qwen7b_run_k_benchmark_matrix_audit.md) | Matrix-level example of responsible prompt-condition interpretation |
| [Method limitations](METHOD_LIMITATIONS.md) | Checkpoint-selection and missing-artifact disclosure |

The benchmark matrix is recreated from the held-out benchmark using [rebuild_benchmark_messages_holdout_736_matrix.sh](../slurm/examples/rebuild_benchmark_messages_holdout_736_matrix.sh). The current Qwen Run K and Llama/Mistral Run L matrix launchers specify the profile grid, prompt modes, and generation budgets. These are reproducibility anchors, not substitutes for reporting the scientific role of each condition described above.
