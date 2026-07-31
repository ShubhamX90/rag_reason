# Supervised fine-tuning: recipe history, current methods, and scientific rationale

## Purpose and scope

This document describes supervised fine-tuning (SFT) at the methodological level: the learning objectives, recipe evolution, model-specific configurations, and limits of the resulting claims. It is written for paper preparation and review. Prompt wording, inference behavior, and metric definitions belong in their own documents.

Two SFT families are maintained:

1. **Trace-text SFT** teaches a model to assess retrieved evidence, identify conflict structure, decide whether the evidence suffices, and produce a cited answer or refusal in a structured public trace.
2. **Answer-only SFT** teaches only the grounded final answer or controlled refusal. It deliberately omits the public trace and intermediate labels.

The trace-text family is the main structured-reasoning method. The answer-only family is both a practical alternative and a comparison condition; its results should never be described as trace-text results.

## Executive summary

The current trace-text method emerged through a documented, error-driven sequence rather than one undifferentiated training run.

- Early prompt-robust SFT established that a structured reasoning response could be internalized under strict, runtime, and genuinely minimal instructions.
- Boundary and source-hygiene variants addressed concrete failures in conflict categorization and instruction-like text inside retrieved sources.
- A data audit showed that persistent over-abstention was partly a training-geometry problem: the final benchmark is dominated by short, answerable retrieval contexts that earlier training underrepresented.
- Run J repaired the training and validation geometry. Run K targeted remaining short-context failures. Run L rebalanced that intervention with short answerable no-conflict cases.

The latest **stored Qwen results** use the Run K short-context-targeted recipe. The latest **stored Llama 3.1 8B and Mistral 7B results** use the Run L boundary-rebalanced recipe. Run L Qwen launch recipes are retained, but no completed Qwen Run L matrix was present in this release. An available recipe is not the same as a completed, reportable result.

The established answer-only result family uses an 862/81 split with a weighted final-only/minimal mixture. A newer clean 862-example answer-only baseline is included for future comparison. It is a new experiment, not a replacement for historical answer-only results.

## 1. Learning problem

Each example contains a question and retrieved documents. The core supervised decision is simple:

| Evidence state | Desired behavior |
| --- | --- |
| Sufficient evidence | Produce a grounded answer with citations. |
| Insufficient evidence | Produce a controlled refusal rather than inventing an answer. |

In trace-text SFT, that decision is decomposed into linked public steps:

1. Judge each document as supporting, partially supporting, or irrelevant.
2. Identify the applicable conflict category, where relevant.
3. Assess evidence sufficiency and abstention.
4. Synthesize a final answer or refusal.
5. Obey a stable response contract, including citations and an end marker.

Answer-only SFT keeps the final decision but hides the intermediate public representation. In both families, abstention is supervised as an evidence decision, not a generic safety response.

## 2. Data basis and evaluation separation

### Current canonical split

| Role | Examples | Use |
| --- | ---: | --- |
| Training backbone | 862 | SFT and the clean answer-only comparison baseline |
| Validation set | 81 | Development diagnostics and checkpoint selection |
| Final held-out benchmark | 736 | Final generation and evaluation only |

The final benchmark is strongly concentrated around five-document contexts: 631 of 736 examples have five retrieved documents. This fact became central to later recipe design. Full split, schema, answerability, and document-count evidence is available in the [train/validation audit](dataset_audits/train_val_split_audit.md) and [benchmark audit](dataset_audits/benchmark_736_audit.md).

### Run J split redesign

The detailed historical trace record begins with a smaller 609-example training setup. Early experiments exposed over-abstention on the short, answerable contexts common in the benchmark. Run J corrected this at the data level rather than only changing inference instructions.

From a larger candidate pool, Run J selected 193 eligible benchmark-like answerable examples: 168 entered training and 25 entered validation. Blank answers, duplicate queries, and overlaps were excluded. This created the current 862/81 backbone while retaining the final 736-example holdout. The complete selection record is in the [Run J split summary](../data/splits/run_j/run_j_split_summary.json).

This is stronger than a decoding-only fix because the failure was measured, the relevant training and validation coverage changed, and a final holdout remained. It is not an independent external domain: all splits arise from the same broader project data ecosystem. Paper claims should remain within that scope.

## 3. Trace-text supervision

Trace-text SFT teaches more than one long answer. The end-to-end target coordinates document judgments, a conflict class, evidence sufficiency, answer synthesis, citations, and an end marker. The mixture also includes direct document-verdict, conflict-type, and answer-only supervision, so component decisions receive explicit signal.

Three instruction strengths are present during training:

| Prompt family | Methodological role |
| --- | --- |
| Strict/default | Detailed teacher-style instruction for the complete response contract. |
| Runtime | Shorter practical instruction for guided deployment. |
| Minimal | Sparse instruction testing whether the learned structure is internalized. |

Training across these conditions makes prompt robustness part of the objective. Oracle prompt conditions are inference ablations, not a substitute for end-to-end reasoning.

## 4. Recipe evolution

### Evidence policy for historical runs

The repository preserves many historical artifacts, but not every prototype has a complete narrative, launch record, and result set. The timeline makes detailed scientific claims only for runs with retained evidence. A legacy filename or checkpoint is not treated as proof of a reproducible result.

### Run D: prompt-robust trace internalization

Run D is the earliest detailed retained trace-text recipe. It asked whether a model could retain the structured trace under a truly minimal prompt, rather than only when a detailed runtime prompt restated the format. It mixed strict, runtime, and minimal end-to-end prompts with document-verdict, conflict-type, and answer-only sub-tasks.

The 609 source examples produced a 6,699-row message mixture. Run D established useful minimal-prompt trace behavior and became the reference backbone. Its main unresolved problem was confusion at conflict-taxonomy boundaries.

### Run E: broad conflict calibration as a negative lesson

Run E applied broad conflict-label pressure. It improved some 32B conflict outcomes but degraded 7B conflict behavior and reduced 32B minimal-prompt document-verdict quality. The lesson was that indiscriminate conflict oversampling is model-size-sensitive and can trade one reasoning stage against another. Later recipes therefore did not simply keep raising a global conflict weight.

### Run F: targeted conflict-boundary teaching

Run F retained the prompt-robust backbone and added one explicit conflict-boundary teaching slice. It clarified no conflict, complementary information, incompatible research claims, outdated information, and misinformation. The mixture grew to 7,308 rows, mainly through one boundary-guided conflict example per source row.

This was a narrower and more interpretable intervention than Run E. It improved the large-model conflict and contract trade-off but exposed different remaining weaknesses by scale: 32B showed document-verdict drift, whereas 7B showed a localized source-instruction contamination failure and fragile complementary-information recall.

### Run G: model-specific probes

Run G deliberately diverged by model size.

- The 32B doc-stabilized variant added document-verdict boundary teaching to reduce overuse of the partially-supports label. It recovered document quality but did not retain enough Run F conflict and contract improvement to become the default large-model strategy.
- The 7B source-guarded variant increased minimal-prompt exposure, targeted complementary-information recall, and taught the model that retrieved text is evidence rather than executable instructions.

These are useful ablations because they show that failure modes were not interchangeable across architectures. They are not the current reported recipe.

### Historical H/I artifacts

The archive includes H/I-era exploratory assets. The retained Run I summary records a 734/74 augmented split, including 40 added training rows and 18 calibration-validation rows. It does not have the same complete method-and-result record as Runs D, F, J, K, and L. It should be described as historical exploration, not central paper evidence without recovered provenance.

### Run J: repair the measured geometry mismatch

Run J changed both data coverage and effective training exposure:

- built the current 862/81 train/validation backbone;
- included benchmark-like short answerable cases in both training and validation;
- increased short-answerable and partial-evidence exposure;
- reduced relative short-refusal pressure without deleting refusal supervision; and
- retained substantial minimal end-to-end trace supervision.

Run J substantially reduced broad over-abstention. Residual errors became local and diagnosable: short answerable cases, especially partial-evidence cases, and the no-conflict versus complementary-information boundary. Its detailed historical record is preserved in the legacy [Run J note](../legacies/docs/experiments/sft_run_j_benchmark_aug_strategy_and_results.md).

### Run K: short-context targeted correction

Run K started from the Run J 862/81 backbone and added 27 derived five-document answerable examples derived from training rows only. The final training split contained 889 examples. It strengthened short-answerable exposure, partial-synthesis drills, document-verdict boundary teaching, and non-no-conflict class-boundary pressure.

The mixture contains 12,659 message rows. These are supervised views, not 12,659 independent source questions. Its weighted five-document geometry assigns 6,235.53 effective units to answerable examples and 1,653.69 to refusal examples.

Run K improved important abstention, contract, citation, and document-verdict measures, particularly for Qwen 32B. Qwen 7B showed a real trade-off: fewer false abstentions but weaker answer-overlap and some conflict-classification measures. The experiment therefore motivated rebalancing rather than a claim of universal improvement. See the [Run K record](experiments/sft_run_k_short_context_targeted_strategy_and_results.md).

### Run L: boundary rebalancing

Run L kept the useful short-context pressure from K but corrected its one-sidedness. K added conflict-bearing short answerables but no short, answerable no-conflict cases. L added 48 derived examples to the Run J backbone, yielding 910 training examples:

- 27 short answerable support examples;
- 18 short no-conflict answerable support examples; and
- 3 short no-conflict partial-evidence examples.

It also strengthened no-conflict and misinformation boundary pressure. The 13,349-row mixture gives 7,305.60 weighted units to five-document answerable examples and 1,719.96 to five-document refusals.

Run L is the logically strongest current trace recipe: it follows a measured failure pattern, corrects the specific missing boundary coverage, and keeps core optimization settings stable relative to J/K. It is not an isolated causal estimate of one feature because data, weights, and drill mixture change together. See the [Run L record](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md) and the [K/L mixture tables](experiments/run_k_l_training_recipe_tables.md).

## 5. Current trace-text recipe

### Run J supervision mixture

| Component | Message rows | Purpose |
| --- | ---: | --- |
| Strict end-to-end trace | 1,724 | Fully specified teacher signal. |
| Strict partial-synthesis trace | 66 | Combine incomplete but jointly sufficient evidence. |
| Runtime end-to-end trace | 862 | Practical guided reasoning behavior. |
| Runtime partial-synthesis trace | 66 | Apply composition under a shorter prompt. |
| Document-verdict task | 862 | Direct evidence-assessment signal. |
| Conflict-type task | 1,724 | Direct conflict-taxonomy supervision. |
| Conflict-boundary drill | 862 | Clarify difficult conflict classes. |
| Answer-only task | 862 | Preserve direct final-response behavior. |
| Runtime partial-synthesis answer-only task | 66 | Apply synthesis without requiring the full trace. |
| Minimal end-to-end trace | 3,448 | Internalization under sparse instruction. |
| Minimal partial-synthesis trace | 132 | Composition under minimal prompting. |
| **Total** | **10,674** | **Run J benchmark-augmented multi-prompt, multi-task SFT.** |

Run J is the first current-backbone mixture. Its key change is not only the row count: it pairs the 862/81 redesigned split with more effective exposure to short answerable cases while retaining the strict/runtime/minimal teaching structure.

### Run K supervision mixture

| Component | Message rows | Purpose |
| --- | ---: | --- |
| Strict end-to-end trace | 1,778 | Fully specified teacher signal. |
| Strict partial-synthesis trace | 66 | Combine incomplete but jointly sufficient evidence. |
| Runtime end-to-end trace | 889 | Practical guided reasoning behavior. |
| Runtime partial-synthesis trace | 132 | Stronger compositional pressure than Run J. |
| Document-verdict task | 889 | Direct evidence-assessment signal. |
| Document-boundary drill | 889 | Clarify supports / partial / irrelevant decisions. |
| Conflict-type task | 1,778 | Direct conflict-taxonomy supervision. |
| Conflict-boundary drill | 1,463 | Stronger pressure on the targeted taxonomy boundary. |
| Answer-only task | 889 | Preserve direct final-response behavior. |
| Runtime partial-synthesis answer-only task | 132 | Apply synthesis without requiring the full trace. |
| Minimal end-to-end trace | 3,556 | Internalization under sparse instruction. |
| Minimal partial-synthesis trace | 198 | Strongest minimal composition exposure so far. |
| **Total** | **12,659** | **Run K short-context-targeted multi-prompt, multi-task SFT.** |

Relative to J, K increases the source split from 862 to 889 examples and adds 27 targeted short answerable variants. Its message growth is therefore purposeful: it comes from both those variants and stronger document-boundary and partial-synthesis supervision.

### Run L supervision mixture


| Component | Message rows | Purpose |
| --- | ---: | --- |
| Strict end-to-end trace | 1,820 | Fully specified teacher signal. |
| Strict partial-synthesis trace | 69 | Combine incomplete but jointly sufficient evidence. |
| Runtime end-to-end trace | 910 | Practical guided behavior. |
| Runtime partial-synthesis trace | 138 | Stronger composition under the runtime prompt. |
| Document-verdict task | 910 | Direct evidence-assessment signal. |
| Document-boundary drill | 910 | Supports / partial / irrelevant distinctions. |
| Conflict-type task | 1,858 | Direct taxonomy supervision. |
| Conflict-boundary drill | 1,839 | Difficult class distinctions. |
| Answer-only task | 910 | Direct final-response behavior. |
| Runtime partial-synthesis answer-only task | 138 | Apply synthesis without requiring the full trace. |
| Minimal end-to-end trace | 3,640 | Internalization under sparse instruction. |
| Minimal partial-synthesis trace | 207 | Composition under minimal prompting. |
| **Total** | **13,349** | **Run L boundary-rebalanced multi-prompt, multi-task SFT.** |

Relative to K, L increases the source split from 889 to 910 examples. Its additional message mass comes from the 21 new short answerable no-conflict examples, modestly stronger conflict-boundary coverage, and corresponding partial-synthesis variants. Exact arithmetic is in the [K/L tables](experiments/run_k_l_training_recipe_tables.md).

### QLoRA optimization configuration

All current trace recipes use parameter-efficient QLoRA: the base model is loaded in quantized form and low-rank adapters are optimized. The standard Run L configuration is:

| Setting | Qwen 7B / 32B | Llama 3.1 8B / Mistral 7B |
| --- | ---: | ---: |
| Epochs | 2 | 2 |
| Learning rate | 2e-4 | 2e-4 |
| Per-device batch size | 1 | 1 |
| Gradient accumulation | 8 | 16 |
| Maximum sequence length | 12,288 | 12,288 |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| LoRA dropout | 0.05 | 0.05 |
| NEFTune alpha | 5.0 | 5.0 |
| Attention implementation | SDPA | SDPA |
| Conflict emphasis | 3.55 | 3.55 |
| Response-contract emphasis | 3.0 | 3.0 |
| Citation emphasis | 1.7 | 1.7 |
| Document-array emphasis | 1.25 | 1.25 |
| Class-balance power | 0.4 | 0.4 |

Qwen Run L uses two-GPU distributed training with accumulation of 8. Llama and Mistral Run L use one GPU with accumulation of 16. The nominal accumulated batch is comparable, but exact global batch depends on the number of processes. NEFTune supplies controlled embedding noise as a regularizer; it does not replace data curation.

### Development selection

Trace checkpoints are selected on the 81-example validation set using a composite development criterion. It balances conflict behavior, document judgment, format reliability, abstention, and explicit false-abstention penalties. In Run L the development weights are 0.18 for document verdicts, 0.25 for response format, and 0.22 for abstention, with additional penalties for false abstentions on answerable evidence.

This is deliberately multi-objective. A high conflict score is not accepted as useful if outputs are malformed, support coverage collapses, or the model refuses answerable cases.

### Why the latest three strategies were designed this way

Runs J, K, and L are best understood as successive attempts to increase the **right kind of data exposure**, not simply to make the training file larger.

#### Run J: align supervision with the deployment failure distribution

The benchmark audit showed that five-document contexts dominate final evaluation, while the early training configuration had a much less favorable balance of short answerable and short refusal examples. A model trained on that geometry can learn a locally rational but globally harmful heuristic: short evidence contexts are often safer to refuse.

Run J changes three linked mechanisms:

1. **Coverage:** it inserts benchmark-like short answerable cases into training. This gives the model direct examples contradicting the short-context-equals-refusal shortcut.
2. **Selection alignment:** it inserts such cases into validation as well. Consequently, checkpoint selection can see the failure mode that mattered on final evaluation instead of optimizing only the old distribution.
3. **Effective exposure:** it gives short answerable and partial-evidence cases more training mass while retaining refusal examples. This changes the loss geometry without misrepresenting duplicated messages as new independent data.

The expected effect is better calibrated answering on short contexts, not merely lower refusal rate. A successful model must reduce false abstentions without creating a symmetric failure of answering when evidence is insufficient. The Run J result record reports that broad over-abstention was substantially reduced, after which the residual errors became narrow enough to target.

#### Run K: teach the remaining hard decision boundary

After J, the main residual cases were not random. They clustered in short answerable and partial-evidence examples, particularly around misinformation and the no-conflict versus complementary-information boundary. Global reweighting was no longer a sufficiently precise tool.

Run K therefore adds only 27 derived five-document answerable examples and reinforces the corresponding reasoning operations. The design has three complementary mechanisms:

1. **Counterfactual context shortening:** derived training variants preserve the underlying question and evidence relation while exposing the model to the difficult five-document retrieval regime.
2. **Compositional evidence training:** partial-synthesis drills reward combining individually incomplete but jointly sufficient evidence, directly countering the tendency to refuse whenever no single document is decisive.
3. **Boundary sharpening:** document and conflict drills provide direct supervision where end-to-end examples alone may leave taxonomy distinctions underdetermined.

This small, targeted expansion is important scientifically. It tests an explicit error hypothesis rather than relying on a large unstructured increase in examples. Its outcome was architecture-dependent, which is exactly the type of result an ACL paper should report rather than smooth over.

#### Run L: correct a bias introduced by K

K's short-context additions were intentionally conflict-bearing. That improves coverage of difficult conflict cases, but can inadvertently teach a new shortcut: short answerable contexts tend to contain conflict. Run L addresses this by adding 21 short answerable no-conflict examples, including three partial-evidence cases, while retaining the K short-answerable support set.

This is more than class balancing in the ordinary sense. It changes the local decision boundary: the model sees that short evidence can be answerable and non-conflicting, and that partial evidence does not automatically imply either abstention or conflict. The moderate extra no-conflict and misinformation boundary pressure is meant to stabilize both sides of this decision.

Thus L is scientifically motivated by a failed-hypothesis correction: it does not assume K was wrong; it preserves K's useful signal and adds the missing counterexamples. This kind of sequential, diagnostic design is stronger than presenting a long list of opaque hyperparameter searches.

### ACL-style discussion of the available results

#### Direct J-to-K comparison: Qwen

The strongest direct evidence comes from the Qwen J/K comparison because the target failure, data addition, and held-out benchmark are explicitly documented.

| Metric on the 736-example benchmark | Qwen 7B: J -> K | Qwen 32B: J -> K | Interpretation |
| --- | --- | --- | --- |
| Abstention accuracy | 94.29 -> 95.79 | 94.97 -> 96.60 | Better overall refusal calibration at both scales. |
| False abstentions | 39 -> 18 | 29 -> 25 | The short-answerable intervention reduces the principal J failure. |
| Missed refusals | 3 -> 13 | 8 -> 0 | 7B pays a refusal-recall cost; 32B improves both sides. |
| Conflict accuracy | 62.07 -> 60.14 | 62.55 -> 67.30 | The same data pressure is not scale-invariant. |
| Token F1 | 0.3555 -> 0.3296 | 0.3945 -> 0.4068 | 32B turns the added signal into better answer overlap; 7B does not. |
| Citation pass | 66.3 -> 73.3 | 55.1 -> 75.3 | Citation discipline improves at both scales. |

For Qwen 32B, K is a coherent improvement: abstention calibration, conflict recognition, document judgment, answer overlap, and citation behavior all move in the desired direction. This supports the hypothesis that additional targeted short-context and compositional examples improve the relevant representation when model capacity is adequate.

For Qwen 7B, K is a trade-off rather than a clean win. The model refuses fewer answerable cases and is more structurally disciplined, but it also misses more true refusals and loses some answer-overlap and conflict accuracy. A plausible mechanism is that the added answerable pressure shifts the smaller model's decision boundary too far toward answering. This is why L adds short no-conflict counterexamples rather than simply increasing the K weights again.

#### Run L behavior across model families

Llama 3.1 8B Run L provides a strong result in a second model family. Its best end-to-end minimal row has 93.2% contract validity, 97.55% abstention accuracy, 17 false abstentions, one missed refusal, 79.81% document micro accuracy, and token F1 of 0.3516. Across the full matrix, SFT improves average contract validity from 26.47% to 86.7%, abstention accuracy from 79.31% to 96.98%, and reduces average false abstentions from 57.08 to 17.83.

Mistral 7B demonstrates the necessary counterexample. Run L substantially improves response structure and answer-overlap relative to its base model, but it remains over-abstaining: its best available end-to-end row has 49 false abstentions, and no matrix row meets the repository's operational-trust threshold. This should be discussed as a model-family limitation, not omitted because it complicates the narrative.

Taken together, the results support a nuanced conclusion: targeted data and structured supervision can substantially improve calibration and response reliability, but their benefit depends on model capacity, pretraining, and the balance of answerable versus refusal pressure. The remaining dominant confusion is consistently no-conflict being predicted as complementary information, so the taxonomy boundary is not solved.

### Data-exposure scaling: hypothesis, observed trend, and limits

It is reasonable to expect that exposing an SFT model to more **relevant, diverse, and correctly balanced** supervision improves overall behavior. The repository contains evidence consistent with that expectation, but it does **not** establish a universal scaling law of the form “more rows always yields better results.”

| Stage | Source training examples | Message rows | What increased | Evidence-supported reading |
| --- | ---: | ---: | --- | --- |
| Run D | 609 | 6,699 | Prompt-family and multi-task supervision | Established prompt-robust trace internalization. |
| Run J | 862 | 10,674 raw rows | Short benchmark-like answerables and validation coverage | Broad over-abstention decreased. |
| Run K | 889 | 12,659 | 27 targeted short answerables plus compositional and boundary drills | Broadly beneficial for Qwen 32B; mixed trade-off for Qwen 7B. |
| Run L | 910 | 13,349 | 21 additional short no-conflict examples and balanced boundary exposure | Strong Llama result; Mistral remains over-abstaining; no stored Qwen L matrix. |

Three distinctions are essential:

1. **Raw message count is not independent data scale.** Duplicated prompt views and sample weights increase optimization exposure, not the number of independent questions.
2. **Coverage matters more than volume alone.** J/K/L add examples from specific underrepresented regions: short answerable, partial-evidence, and short no-conflict cases. Their value comes from correcting a decision boundary, not merely from increasing token count.
3. **Scaling is conditional on model and balance.** The J-to-K Qwen comparison and the Llama/Mistral contrast show non-monotonic outcomes. Extra answerable pressure helps one model while causing another to over-answer or over-refuse.

The defensible ACL-style claim is therefore: **in this RAG setting, progressively broader and better-balanced targeted supervision is associated with improved overall behavior when it covers measured failure slices and is matched to model capacity.** It is not defensible to claim a general law that any additional SFT data or any larger mixture must improve every metric.

Future work can make this a stronger scaling study by holding the prompt family, optimization budget, and class balance fixed while varying only the number of independently curated short-context examples. The current J/K/L sequence supplies the motivation and the empirical failure slices for that controlled ablation.

## 6. Model-specific status

| Base model | Latest stored result family | Latest retained trace recipe | Interpretation |
| --- | --- | --- | --- |
| Qwen2.5 7B Instruct | Run K | Run L | Complete Run K matrix is stored. Run L training and evaluation recipes remain, but no completed Qwen Run L matrix is stored locally. |
| Qwen2.5 32B Instruct | Run K | Run L | Complete Run K matrix is stored. Run K is the strongest locally documented Qwen 32B family; Run L is a principled follow-up without a retained Qwen matrix. |
| Llama 3.1 8B Instruct | Run L | Run L | Complete Run L matrix is stored. |
| Mistral 7B Instruct v0.3 | Run L | Run L | Complete Run L matrix is stored. |

Matrix analyses are available for [Qwen 7B](../final_model_outputs/qwen7b_benchmark_matrix_analysis.md), [Qwen 32B](../final_model_outputs/qwen32b_benchmark_matrix_analysis.md), [Llama 3.1 8B](../final_model_outputs/llama8b_benchmark_matrix_analysis.md), and [Mistral 7B](../final_model_outputs/mistral7b_benchmark_matrix_analysis.md).

The original Run K Qwen submission wrapper is not present. Its split, message-recipe, evaluation, and final-output evidence are retained, but the training job should be described as reconstructed rather than byte-for-byte replayed from one original submission file.

## 7. Answer-only SFT

### Established updated-split recipe

The established answer-only recipe uses the same current 862/81 split but removes the public trace from the assistant target. For each source example it creates two training prompt families:

| Prompt family | Copies per source | Rows |
| --- | ---: | ---: |
| Final-only | 8 | 6,896 |
| Minimal | 4 | 3,448 |
| **Total** | **12** | **10,344** |

The 2:1 final-only/minimal ratio makes the output contract explicit while testing whether answer-only behavior survives reduced scaffolding. Targets contain a grounded answer or canonical refusal, citations when appropriate, and an end marker. They do not include document-verdict arrays, conflict labels, or the public trace.

The recorded QLoRA configuration is two epochs, learning rate 2e-4, batch size 1, accumulation 8, maximum length 12,288, LoRA rank 32, alpha 64, dropout 0.05, and NEFTune alpha 5.0. Stored result artifacts cover Qwen 7B, Llama 3.1 8B, and Mistral 7B under final-only and minimal prompts.

One critical limitation applies. The inherited development callback was built for trace outputs. In the reported answer-only recipe, document and format selection weights are zero and the adapter is effectively selected by abstention accuracy. The historical directory name best_dev_f1 must not be described as semantic-F1 checkpoint selection. See [method limitations](METHOD_LIMITATIONS.md).

### Clean 862-example answer-only baseline

The active standalone export contains a deliberately simple comparison baseline: one final-only target per canonical training example, for 862 training rows and 81 validation rows. It has no prompt duplication, sample weights, document-verdict or conflict tasks, trace targets, injected drills, or Run K/L derived rows.

This is appropriate for a transparent recipe-level comparison against the full trace method. It is **not** an isolated causal ablation of trace supervision, because the trace recipe also differs in augmentation, prompt mixture, drills, weighting, and checkpoint selection. The scope is stated in the [baseline note](../answer_only_sft_export/docs/BASIC_ANSWER_ONLY_862_BASELINE.md).

The clean baseline uses the fixed final-epoch adapter rather than the inherited trace-oriented development-selected adapter. Its future results must be reported separately from stored updated-split answer-only artifacts.

## 8. Why the strategy is scientifically defensible

The repository supports methodological claims, not a claim of universal model superiority.

1. **Measured error slices drive changes.** J addressed short-answerable coverage; K targeted residual short-context failures; L corrected K's missing short no-conflict coverage.
2. **Source examples, augmented examples, and weighted views are separated.** The 862, 889, and 910 source-example counts are not presented as the 10,344, 12,659, and 13,349 message-row counts. Repetition is an optimization choice, not new independent data.
3. **A final holdout is retained.** K/L derived examples originate from training rows rather than moving final held-out rows into training.
4. **Evaluation is multi-metric.** Contract validity and support coverage are gates before conflict, document, citation, abstention, and answer scores are interpreted.
5. **Model size is treated empirically.** The D/F/G history shows that a change helping one architecture need not help another.

## 9. Limits that must accompany paper claims

- J/K/L change several ingredients together. They are principled recipe comparisons, not single-factor causal ablations.
- Latest K/L and updated answer-only adapters are absent from this checkout. The recipes and evaluated outputs remain, but exact adapter-only inference requires those adapters or a new training run.
- The original Run K Qwen training wrapper is missing.
- Reported answer-only checkpoint selection is abstention-focused, not semantics-first.
- The held-out benchmark is separate from current training but is not an independent external data domain.
- The clean 862-example answer-only baseline is a new experiment until generation and evaluation artifacts are complete.

These disclosures delimit the reproducible claim; they do not justify hiding design choices or overstating results.

## 10. Concise paper-ready description

> We fine-tune instruction-following language models with parameter-efficient QLoRA on evidence-grounded retrieval examples. Our trace-text objective jointly supervises document-level evidence judgments, conflict categorization, answerability, and a cited final response across strict, runtime, and minimal instruction conditions. The final recipe was developed through held-out error analysis: benchmark-like short answerable cases were incorporated into the training/validation geometry, followed by targeted short-context and class-boundary balancing. Models are evaluated on a disjoint 736-example holdout using response validity, abstention calibration, evidence/conflict behavior, citations, and answer overlap.

Use this wording together with the model-specific status and limitations above; do not imply one universally best checkpoint across all architectures.

## Related records

- [Current experiment index](experiments/README.md)
- [Run K strategy and results](experiments/sft_run_k_short_context_targeted_strategy_and_results.md)
- [Run L strategy and status](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md)
- [Run K/L mixture tables](experiments/run_k_l_training_recipe_tables.md)
- [Dataset audits](dataset_audits/README.md)
- [Method and checkpoint-selection limitations](METHOD_LIMITATIONS.md)
- [Reviewer reproduction guide](REPRODUCIBILITY.md)
- [Standalone answer-only recipe](../answer_only_sft_export/docs/LATEST_ANSWER_ONLY_SFT_RECIPE.md)
