# Types of Ablations in the RAG Reasoning Project

## Why this document exists

This repository contains several different kinds of experiments that are all casually called “ablations.” They do not answer the same question, do not have the same causal strength, and should not be combined into one undifferentiated results table.

This document is a map of the ablation families used in the project. For each family, it explains:

- what changes;
- what is intended to remain fixed;
- the scientific question it answers;
- the relevant historical or current runs; and
- the correct way to report it.

The chronological evidence and observed outcomes are in [ABLATION_STUDY.md](ABLATION_STUDY.md). The prompt details are in [PROMPT_DESIGN_AND_ABLATIONS.md](PROMPT_DESIGN_AND_ABLATIONS.md). This document instead answers the simpler but crucial question: **what types of comparisons do we actually have?**

## 1. Overview: the ablation taxonomy

| Type | Unit being varied | Core question | Representative evidence | Causal strength |
| --- | --- | --- | --- | --- |
| A. Supervision-visibility | Assistant target: public trace versus final answer only | Do visible intermediate targets change learned behavior? | Trace-text SFT versus answer-only SFT | Recipe-level comparison; not yet a pure single-variable test |
| B. Prompt-robustness | Amount of instruction at inference | Does learned behavior survive reduced scaffolding? | Strict/default, runtime, minimal profiles | Strong diagnostic comparison when model/examples are fixed |
| C. Oracle-information | Gold intermediate evidence supplied at inference | Which reasoning stage is the bottleneck? | `e2e`, `oracle_conflict`, `oracle_notes`, `oracle_both` | Conditional diagnostic; not deployment performance |
| D. Multi-task decomposition | Which supervised task views are included | Do direct Stage-1/Stage-2/final-answer targets help the full task? | Run D backbone | Foundational mixture design; not fully isolated in stored results |
| E. Conflict-boundary teaching | Targeted taxonomy guidance and its exposure | Can models distinguish nearby conflict classes more reliably? | Runs F, K, L | Narrow training ablation / targeted revision |
| F. Document-verdict boundary teaching | Guidance for supports / partial / irrelevant | Can Stage-1 labeling be improved without losing downstream quality? | 32B Run G, K/L | Narrow training ablation / targeted revision |
| G. Partial-evidence composition | Extra jointly sufficient partial-evidence examples | Can the model answer when no single document is decisive? | K/L partial-synthesis drills | Targeted revision, coupled with other changes |
| H. Source-hygiene | Exposure to instruction-like text inside retrieved snippets | Can the model treat sources as evidence rather than commands? | 7B Run G | Localized robustness ablation |
| I. Data-geometry alignment | Training/validation coverage and effective sample weights | Does representation of the benchmark failure slice improve calibration? | Run J | Multi-component recipe revision |
| J. Short-context augmentation | Derived five-document answerable examples | Can residual short-context false abstention be reduced? | Run K | Targeted multi-component revision |
| K. Boundary rebalancing | Class balance within short-context augmentation | Does adding short no-conflict evidence prevent induced skew? | Run L | Targeted multi-component revision |
| L. Model-family transfer | Base architecture / scale | Does the same recipe generalize across models? | Qwen, Llama, Mistral K/L results | External-validity comparison, not a controlled ablation |
| M. Selection-policy comparison | Validation/checkpoint decision rule | Does a chosen checkpoint optimize the intended objective? | Trace versus answer-only selection disclosure | Methodological control; current answer-only limitation |

The word “causal” should be used most cautiously for Types A, D, I, J, K, L, and M, because more than one ingredient changes or because the compared systems are inherently different. The strongest contained contrasts are usually B and C, where the model and examples can be held fixed and the intervention is explicit.

## 2. Type A: supervision-visibility ablation

### What changes

The assistant training target either exposes the full public evidence trace or reveals only the final answer/refusal.

| Trace-text SFT | Answer-only SFT |
| --- | --- |
| Stage 1 document assessment, Stage 2 conflict assessment, Stage 3 answer plan, final cited answer | Final cited answer or controlled refusal only |
| Direct supervision for intermediate decisions | No direct visible supervision for document verdicts, conflict labels, or answer plan |
| Intended to make reasoning steps auditable | Intended to test final-answer behavior without public trace targets |

### Scientific question

Does supervising intermediate evidence and conflict structure lead to better grounded end-to-end behavior than supervising only the final response?

### What must be held fixed for a strong answer

A clean causal comparison would use the same base model, 862 training IDs, 81 validation IDs, prompt family, optimization budget, decoding policy, and checkpoint-selection rule. The only intended difference would be the visible assistant target.

### What exists now

The established answer-only family uses the current 862/81 source split but a 10,344-row weighted final-only/minimal mixture and an abstention-focused inherited selection scheme. The current trace recipe also differs in multi-task views, trace prompts, targeted drills, sample weights, short-context augmentations, and its composite development selection. Thus the stored comparison is a **recipe-level comparison**, not a pure trace-removal ablation.

The standalone export includes a clean basic answer-only baseline with exactly one final-only target per canonical training example. This is better suited to a future clean comparison, but it has no completed result artifacts yet.

### Paper-safe wording

> We compare the full trace-text and answer-only training recipes as two supervision paradigms. Because their established mixtures and selection procedures differ, this comparison should be interpreted at recipe level rather than as an isolated estimate of the public trace target.

Relevant records: [SFT description](SFT_DESCRIPTION.md), [answer-only recipe](../answer_only_sft_export/docs/LATEST_ANSWER_ONLY_SFT_RECIPE.md), [basic 862 baseline](../answer_only_sft_export/docs/BASIC_ANSWER_ONLY_862_BASELINE.md), and [method limitations](METHOD_LIMITATIONS.md).

## 3. Type B: prompt-robustness ablation

### What changes

Only the amount of task instruction supplied at inference changes.

| Profile | Instruction level | What it tests |
| --- | --- | --- |
| Strict/default | Detailed definitions, edge-case rules, format schema, and self-check | High-information teacher condition and contract compliance |
| Runtime | Compact trace protocol and evidence policy | Deployment-like guided operation |
| Minimal | Basic evidence-only answerability policy | Whether the SFT has internalized the protocol |

### What stays fixed

For a given matrix cell group, the base or SFT model, benchmark examples, prompt mode, benchmark labels, and evaluation procedures remain fixed. Only the textual scaffolding differs.

### Scientific question

Does a model need the long prompt to behave correctly, or has SFT made the evidence reasoning and response contract robust to substantial instruction removal?

### Interpretation

- Strong strict but weak minimal behavior indicates prompt dependence.
- Strong runtime behavior supports practical use of a compact prompt.
- Strong minimal trace behavior provides evidence of internalization, but still requires grounding and abstention checks.

This is not an ablation of task difficulty: the evidence problem and expected trace-text target remain the same. It is an ablation of **instructional scaffolding**.

### Important confound to disclose

The generation budgets are profile-specific because a strict trace is longer: strict/default uses a nominal 1,400-token base allowance (3,200 cap), runtime 1,200 (2,200 cap), and minimal 900 (1,800 cap). These budgets are part of the experimental contract. A profile comparison should report response validity and support counts so truncation or parsing differences are visible.

Relevant record: [prompt design](PROMPT_DESIGN_AND_ABLATIONS.md).

## 4. Type C: oracle-information ablations

### What changes

The model receives gold intermediate information at inference while the query and retrieved documents remain unchanged.

| Mode | Gold information supplied | Remaining work for the model |
| --- | --- | --- |
| `e2e` | None | Infer document verdicts, conflict type, sufficiency, and answer |
| `oracle_conflict` | Conflict type | Infer document verdicts, sufficiency, and answer |
| `oracle_notes` | Per-document notes | Infer conflict type, sufficiency, and answer |
| `oracle_both` | Document notes and conflict type | Infer sufficiency and synthesize final answer |

### Scientific question

Where is the limiting error located? For example, if `oracle_conflict` improves final behavior substantially over `e2e`, then conflict-type prediction is a plausible bottleneck. If `oracle_both` still fails, the remaining issue lies in evidence sufficiency, answer planning, synthesis, citations, or formatting.

### Why this is valuable

An ordinary aggregate score can hide the source of failure. Oracle ablations decompose the end-to-end mapping into conditional stages without changing the retrieval context itself. They are especially useful for deciding whether the next training intervention should target Stage 1, Stage 2, or final-answer calibration.

### What they are not

Oracle modes are not deployment settings. They must never replace `e2e` as the headline system result, because a real system does not receive gold document verdicts or a gold conflict label. Near-perfect document scores under `oracle_notes` are expected—those notes were supplied by design.

### Paper-safe wording

> Oracle conditions are diagnostic upper bounds in which selected gold intermediate annotations are supplied; all main system claims use the end-to-end condition.

Relevant record: [prompt design](PROMPT_DESIGN_AND_ABLATIONS.md) and the stored [Qwen 7B matrix audit](experiments/qwen7b_run_k_benchmark_matrix_audit.md).

## 5. Type D: supervision-decomposition ablation

### What changes

The training mixture includes not only a full end-to-end trace but direct task-specific targets:

| Supervised view | Decision being isolated |
| --- | --- |
| End-to-end trace | The complete chain from evidence assessment to answer |
| Document-verdict task | Whether each snippet supports, partially supports, or is irrelevant |
| Conflict-type task | Which relationship holds among non-irrelevant documents |
| Answer-only task | Final grounded answer versus controlled refusal |

### Scientific question

Can shorter, direct supervision for individual decisions help the model learn a long, structured end-to-end response more reliably than end-to-end traces alone?

### Evidence status

Run D established the multi-task prompt-robust backbone: strict end-to-end, runtime end-to-end, runtime component tasks, and minimal end-to-end targets. Later runs retain this structure. The repository does not currently contain a matched “end-to-end only” result set that changes only the component task views, so this is a foundational design choice rather than a completed isolated causal ablation.

### How to describe it

Call it **multi-task decomposition** or **component supervision**, not a completed causal proof that every individual subtask is necessary. Its rationale is strong—each intermediate decision is evaluated and receives direct signal—but a future removal study would be required to quantify each component's unique contribution.

Relevant record: [SFT description](SFT_DESCRIPTION.md) and [Run D strategy](../legacies/docs/experiments/sft_run_d_prompt_robust_strategy.md).

## 6. Type E: conflict-taxonomy boundary ablation

### What changes

Targeted conflict-type rows receive a concise instructional prefix explaining the boundary between:

- aligned evidence (`No conflict`);
- compatible distinct facets (`Complementary information`);
- same-scope incompatibility (`Conflicting opinions or research outcomes`);
- temporal supersession (`Conflict due to outdated information`); and
- a weaker or false claim relative to stronger evidence (`Conflict due to misinformation`).

### Scientific question

Does direct teaching of nearby semantic boundaries work better than broadly increasing conflict-label pressure?

### Key comparisons

| Comparison | Main variation | Lesson |
| --- | --- | --- |
| Run D -> E | Broad conflict calibration | Model-size-sensitive; helped 32B conflict but hurt 7B |
| Run D -> F | Add one conflict-boundary drill per source example | More targeted and generally more successful direction |
| Run K -> L | Rebalance boundary pressure, especially no-conflict and misinformation | Correct local skew induced by conflict-heavy short-context additions |

### Why this is a strong idea

The hard mistakes are not uniformly distributed among five labels. The recurrent error is often `No conflict` versus `Complementary information`: evidence can be jointly useful without being contradictory. A boundary drill describes this distinction explicitly, allowing supervision to concentrate on the model's actual decision surface rather than merely making rare labels more frequent.

### Causal caution

Run F is the narrowest historical test, although it also modestly changes conflict loss emphasis. K/L include boundary pressure as one component of broader data revisions and should be reported as targeted recipe changes rather than isolated prompt effects.

Relevant records: [Run F](../legacies/docs/experiments/sft_run_f_boundary_guarded_strategy.md), [Run K](experiments/sft_run_k_short_context_targeted_strategy_and_results.md), [Run L](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md).

## 7. Type F: document-verdict boundary ablation

### What changes

The model receives extra direct document-verdict views and/or a short prefix clarifying three Stage-1 boundaries:

- direct query-relevant evidence is `supports`, even when brief or from a lower-quality source;
- on-topic but incomplete evidence is `partially supports`; and
- keyword overlap, generic background, wrong-domain senses, and tangents are `irrelevant`.

### Scientific question

Can Stage-1 document accuracy be recovered or improved without degrading conflict reasoning and final answer behavior?

### Evidence

Qwen 32B Run G is the clearest historical test. Compared with Run F, it doubles the direct runtime document-verdict task exposure and adds a document-boundary drill. It improves document micro accuracy, but loses significant strict/minimal conflict and contract-adjusted quality. This is an instructive trade-off rather than a final recipe win.

Run K and L retain one document-boundary drill per runtime document-verdict example because later error analysis identified document judgment as part of the residual short-context problem. However, because they also change data coverage, partial-synthesis exposure, and weights, those later runs are not standalone document-verdict ablations.

### Paper-safe conclusion

> Extra Stage-1 supervision can improve document-label fidelity, but the stored 32B ablation shows that this does not automatically preserve conflict and contract quality; selection must remain multi-objective.

Relevant records: [32B Run G](../legacies/docs/experiments/sft_run_g_doc_stabilized_strategy.md) and [K/L mixture tables](experiments/run_k_l_training_recipe_tables.md).

## 8. Type G: partial-evidence composition ablation

### What changes

The mixture adds copies of examples where no document individually has a `supports` verdict, but several `partially supports` documents jointly determine the answer. The added instruction says, in effect: combine compatible partial evidence; abstain only if a necessary gap remains after composition.

### Scientific question

Can the model avoid the incorrect shortcut “no single decisive snippet means refusal”?

### Why it matters

The held-out benchmark contains answerable partial-only cases, many in the dominant five-document context. Run J analysis showed a high false-abstention rate for that slice, including especially hard misinformation examples. Partial-synthesis training targets the decision policy directly rather than simply reducing all refusals.

### Evidence status

Run K strengthens strict, runtime, answer-only, and minimal partial-synthesis exposures; Run L retains them while adding no-conflict partial-only counterexamples. Their results are part of broader short-context revisions, so the repository supports a targeted-design argument but not a clean one-factor estimate for the drill alone.

### What to report

Report answerable partial-only performance separately from overall abstention whenever possible. Overall refusal accuracy alone can conceal a system that handles easy direct-support cases but still refuses jointly sufficient evidence.

Relevant records: [Run J deep audit](../legacies/docs/experiments/archive/run_j_deep_audit_2026_06_25.md), [Run K](experiments/sft_run_k_short_context_targeted_strategy_and_results.md), and [Run L](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md).

## 9. Type H: source-hygiene ablation

### What changes

A targeted training prefix teaches that retrieved text is evidence, not an instruction to execute. The model should ignore prompt-like commands, roleplay, refusals, or foreign directives within snippets while still assessing whether those snippets contain factual evidence.

### Scientific question

Can targeted evidence-versus-instruction supervision prevent retrieved text from contaminating the response contract?

### Evidence

The Qwen 7B G variant introduces one source-hygiene end-to-end drill per runtime end-to-end source example to repair a specific minimal-prompt failure. It repaired the known malformed/source-contaminated example and restored complete minimal trace structure, but the broader recipe overcorrected other metrics. The result shows a local robustness benefit, not a universal gain from adding source-hygiene rows.

### Current status

Source-hygiene rows are not active in the Run K/L mixtures. It is a retained historical ablation and should be described as such. Do not imply that the latest L recipe includes this intervention.

Relevant record: [7B Run G](../legacies/docs/experiments/sft_run_g_7b_source_guarded_strategy.md).

## 10. Type I: data-geometry alignment ablation

### What changes

Run J changes which evidence regimes are represented in training and validation, and changes their effective exposure through weighting. It augments the prior split with benchmark-like short answerable examples while preserving the 736-example final holdout.

### Scientific question

Is persistent over-abstention caused partly by a mismatch between the training/validation distribution and the final benchmark's dominant short-context answerable cases?

### Why it differs from a prompt ablation

No wording change can fully compensate if the model rarely sees the evidence configuration it must handle at evaluation. The final benchmark is dominated by five-document contexts, and the post-J error analysis becomes much more localized after short answerable coverage is added.

### Evidence and limit

Run J substantially reduces broad false abstention and provides the current 862/81 backbone. It changes several ingredients together—training coverage, validation coverage, and sample weights—so it is a **data-geometry recipe revision**, not a single-factor ablation. The source summary also contains an archived prose discrepancy: 862 final rows minus 168 training augmentations implies 694 base rows, whereas one legacy note calls the older setup 692/56. The machine-readable summary supports the arithmetic-backed 694/56 interpretation.

### Paper-safe conclusion

> Aligning training and validation exposure with the measured short-context answerable failure slice was associated with a large calibration improvement; the individual contribution of selection, weighting, and validation coverage is not separately identified.

Relevant records: [Run J record](../legacies/docs/experiments/sft_run_j_benchmark_aug_strategy_and_results.md) and [split audit](dataset_audits/train_val_split_audit.md).

## 11. Type J: short-context augmentation ablation

### What changes

Run K adds 27 derived five-document answerable variants from existing Run J training examples only. It also strengthens partial-synthesis, document-boundary, and targeted conflict-boundary exposure.

### Scientific question

After broad false abstention is repaired, can a small amount of five-document answerable supervision improve the remaining concentrated errors without disturbing the full recipe?

### Evidence

For Qwen 32B, K improves abstention calibration, conflict accuracy, document accuracy, answer overlap, and citation measures relative to J. For Qwen 7B, it reduces false abstentions but increases missed refusals and reduces conflict/answer-overlap measures. This is evidence of a capacity-sensitive trade-off, not a universal improvement.

### Correct terminology

Call K a **targeted short-context recipe revision** or **derived-data augmentation study**. It is not solely a data ablation because it changes several drill and weighting choices at the same time.

Relevant record: [Run K](experiments/sft_run_k_short_context_targeted_strategy_and_results.md).

## 12. Type K: boundary-rebalancing ablation

### What changes

Run L keeps K's useful short answerable support but adds 21 short answerable `No conflict` examples—18 direct support and 3 partial-only. It also changes local no-conflict and misinformation exposure.

### Scientific question

Does K create a local shortcut by showing that short answerable cases are usually conflict-bearing, and can balanced short no-conflict counterexamples correct it?

### Why this is scientifically meaningful

L does not merely increase the total data volume. It changes the *conditional composition* of the added data. The intended lesson is: a short retrieval context can be answerable without conflict, and partial evidence does not automatically mean either abstention or conflict. This is a decision-boundary correction.

### Evidence and limit

Run L produces a strong stored Llama 3.1 8B result but leaves Mistral 7B overly abstaining; Qwen L launch artifacts exist without a stored completed matrix. Thus L is evidence for a carefully motivated balance intervention whose benefit still depends on the model family.

Relevant records: [Run L](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md) and [K/L tables](experiments/run_k_l_training_recipe_tables.md).

## 13. Type L: model-family and capacity comparison

### What changes

The base architecture and scale change: Qwen 7B/32B, Llama 3.1 8B, and Mistral 7B are trained or evaluated with related trace recipes.

### Scientific question

Does the recipe behave consistently across model families, or does capacity/pretraining change how targeted supervision is absorbed?

### What it shows

This is not a controlled ablation in the strict sense: architectures differ in pretraining data, instruction tuning, tokenizer, and many latent properties. It is nevertheless important external-validity evidence. The retained results show that Qwen 32B absorbs Run K's targeted short-context signal more cleanly than Qwen 7B, and that Run L is strong for Llama 8B but not operationally reliable for Mistral 7B due to remaining false abstention.

### Paper-safe wording

> We evaluate transfer across several instruction-tuned model families. The results demonstrate that the direction of benefit is not architecture-invariant, so we report model-specific outcomes rather than a single universal best recipe.

Relevant records: [model status](SFT_DESCRIPTION.md), [Qwen 7B](../final_model_outputs/qwen7b_benchmark_matrix_analysis.md), [Qwen 32B](../final_model_outputs/qwen32b_benchmark_matrix_analysis.md), [Llama](../final_model_outputs/llama8b_benchmark_matrix_analysis.md), and [Mistral](../final_model_outputs/mistral7b_benchmark_matrix_analysis.md).

## 14. Type M: checkpoint-selection and optimization-policy comparison

### What changes

The validation objective or checkpoint selection policy changes, rather than the raw supervised examples alone.

### Why it matters

Checkpoint selection can reverse the apparent conclusion of a training ablation. A checkpoint optimized only for abstention may not be best for semantic answer quality; a checkpoint optimized only for conflict labels may not be best for contract reliability or false-abstention control.

The trace recipes use a composite validation criterion that balances document verdicts, response format, abstention, and false-abstention penalties. The established answer-only recipe inherits a trace-oriented callback with document/format weights set to zero, so its historical `best_dev_f1` directory is in effect selected by abstention accuracy. This is preserved for reproducibility but cannot be described as semantics-first selection.

### Evidence status

This is a methodological limitation and a prospective ablation opportunity, not a completed fair comparison. The clean basic answer-only baseline transparently uses the fixed final epoch to avoid making an unsupported answer-quality selection claim.

Relevant records: [method limitations](METHOD_LIMITATIONS.md) and [answer-only limitations](../answer_only_sft_export/docs/KNOWN_LIMITATIONS.md).

## 15. Which ablations belong in the paper?

| Priority | Ablation family | Recommended treatment |
| --- | --- | --- |
| Primary | Prompt robustness (B) | Main table or figure: `e2e` strict/runtime/minimal with contract and calibration gates |
| Primary | Oracle diagnostics (C) | Ablation table, clearly marked as gold-intermediate upper bounds |
| Primary | J/K/L data/coverage evolution (I/J/K) | Main method-development table with exact source counts and limitations |
| Secondary but valuable | D/E/F/G targeted supervision studies (D/E/F/G/H) | Historical ablation table showing why broad pressure was rejected and targeted drills adopted |
| Secondary / conditional | Trace versus answer-only (A) | Recipe-level comparison; promote clean 862 baseline only after results exist |
| Essential qualifier | Model-family transfer (L) | Model-specific result table and discussion, not a pooled universal claim |
| Required disclosure | Selection policy (M) | Limitations/supplementary methods; especially answer-only checkpoint selection |

## 16. A concise reporting rule

For every result, name all three parts of the experimental condition:

```text
training recipe + inference information condition + prompt profile
```

For example:

```text
Run K trace-text SFT + end-to-end inference + minimal prompt
Run L trace-text SFT + oracle-notes inference + runtime prompt
Established answer-only SFT + end-to-end inference + final-only prompt
```

This naming convention prevents three common mistakes: treating an oracle row as end-to-end performance, treating a strict prompt result as minimal-prompt internalization, or treating answer-only and trace-text recipes as identical except for one output field.

## 17. Related documents

- [Chronological ablation evidence](ABLATION_STUDY.md)
- [Prompt design and oracle ablations](PROMPT_DESIGN_AND_ABLATIONS.md)
- [SFT recipe and current result status](SFT_DESCRIPTION.md)
- [Run K/L exact mixtures](experiments/run_k_l_training_recipe_tables.md)
- [Reproducibility and missing-artifact disclosure](REPRODUCIBILITY.md)
