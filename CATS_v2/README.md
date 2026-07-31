# CATS v2

CATS v2 is a research-oriented evaluation framework for retrieval-augmented generation under evidence conflict. It is designed for settings where a model is asked to answer using retrieved documents, but the retrieved evidence may be incomplete, contradictory, temporally stale, differently reliable, or opinion-bearing. In such settings, conventional answer evaluation is not enough. A response can be fluent, superficially correct, and even partially supported, while still failing in a deeper way: it may answer when it should abstain, collapse a disagreement into false certainty, ignore recency, misuse evidence, or omit the core target answer.

This repository contributes a structured answer-evaluation methodology for that setting. Its core claim is that trustworthy RAG evaluation should not be reduced to a single notion of correctness. Instead, it should decompose trustworthiness into several complementary properties:

1. whether the system should have answered at all,
2. whether it followed the correct conflict-handling behavior,
3. whether its claims were grounded in the available evidence,
4. and whether it contained the target answer when the task admits a single truth target.

These dimensions are combined into the Conflict-Aware Trust Score, or CATS.

This README is the top-level research, artifact, and reproduction guide. It
connects the conceptual design to the exact current code paths, prompts,
configuration files, datasets, outputs, human-evaluation package, local judge
serving workflow, audit commands, and ACL-paper reporting requirements. The
specialist documents linked below provide deeper detail for individual areas;
this file defines how those documents fit together.

For the curated submission-oriented repo layout and canonical current paths, see `CURRENT_REPO_MAP.md`.

## Research Motivation

Many RAG evaluations treat retrieved evidence as if it were either cleanly sufficient or cleanly irrelevant. Real retrieval settings are rarely so simple. Retrieved sets often contain:

- temporally outdated documents mixed with newer ones,
- high-quality sources mixed with weaker or derivative sources,
- partial evidence distributed across multiple documents,
- documents that support different subclaims,
- genuine conflict in the evidence,
- or opinion-bearing material where a single canonical answer is not appropriate.

In such cases, a model can fail in ways that ordinary exact-match or semantic-similarity evaluation will not capture.

For example:

- A model may answer confidently when the retrieved evidence is insufficient.
- A model may refuse to answer even though the evidence is sufficient.
- A model may produce a plausible answer that is not actually grounded in the retrieved evidence.
- A model may cite evidence but cite the wrong evidence.
- A model may answer a time-sensitive question using stale documents.
- A model may present one side of a disputed issue as settled fact.
- A model may behave carefully but still omit the actual target answer.

The central research purpose of CATS v2 is to evaluate these failure modes explicitly rather than collapsing them into a single correctness label.

## What This Repository Contributes

This repository contributes a conflict-aware evaluation protocol for RAG answers. Conceptually, it provides four things:

1. A decomposition of answer trustworthiness into distinct measurable properties.
2. A conflict-aware taxonomy that determines what kind of answering behavior is appropriate for a given sample.
3. A grounding protocol that checks not only whether a claim is supportable, but whether the answer cites supporting retrieved evidence.
4. A multi-judge evaluation design that uses explicit committee lanes to assess behavior, grounding, and answer recall. The current production local benchmark lane uses three judges.

The repository therefore supports research questions such as:

- How often do RAG systems answer when they should abstain?
- How robust are systems to different conflict types?
- Do systems use retrieved evidence faithfully, or merely sound plausible?
- When a task has a single target answer, does the answer actually include it?
- How should these dimensions be integrated into one trust-oriented score?

## Central Evaluation Question

At the highest level, CATS v2 asks:

`Given the retrieved evidence and the conflict structure of the sample, how trustworthy is this answer?`

That question is decomposed into four sub-questions:

1. Was answering appropriate under the evidence?
2. Did the system follow the right conflict-handling behavior?
3. Were the answer’s claims grounded in eligible retrieved evidence?
4. If the sample had a single truth target, did the answer contain it?

The CATS score is built from those sub-questions.

## Current Judge Committees

This curated repository preserves three committee lanes, but they are not the
same experimental condition:

- Default remote code path: `claude-haiku-4-5` + `gpt-5.4` + `deepseek-v3.2`
- Current mixed val-ceiling path: `Codex CLI (gpt-5.4)` + `deepseek-v4-flash`
- Current local benchmark path: `qwen3.5-397b-a17b` (priority 6) + `mistral-small-4` (priority 3) + `deepseek-r1-distill-32b` (priority 2)

These committees are used for the judgment-heavy parts of the evaluation:

- Behavior Adherence
- Factual Grounding
- Single-Truth Recall

Grounded Refusal is evaluated deterministically from annotations and answer form rather than by committee judgment.

The purpose of the committee design is to reduce evaluator brittleness. Conflict-aware behavior, evidence sufficiency, and answer containment are judgment-sensitive; multi-judge evaluation provides a more stable signal than a single free-form evaluator. The committee is still an evaluator, not human gold truth; human agreement analyses and judge-sensitivity checks are therefore important for paper claims.

## Conceptual Data Inputs

Although this repository operates over JSONL records internally, the research object is better understood in terms of information roles rather than file fields. Each evaluation sample contains some or all of the following conceptual inputs:

- A user query or question.
- A retrieved evidence set.
- Document-level notes describing how each retrieved document relates to the answer.
- A conflict label or conflict category.
- A model answer to evaluate.
- In some tasks, a gold target answer.
- In some tasks, an explicit answerability label under the retrieved evidence.

These inputs serve different methodological purposes.

### Query

The query defines the answer task. It determines what the system is being asked to resolve and frames the relevance of the retrieved evidence.

### Retrieved Evidence

The retrieved evidence is the support context against which the answer is evaluated. CATS v2 does not treat correctness as free-floating. An answer may be factually true in the world and still fail CATS grounding if it is not supported by the retrieved evidence made available to the model.

### Per-document Notes

Per-document notes supply structured supervision about how each retrieved document relates to the answer task. These notes are crucial because they distinguish:

- documents that support the answer,
- documents that partially support it,
- documents that conflict,
- documents that are irrelevant,
- or documents that contribute in other conflict-specific ways.

These notes anchor the grounding metric and help determine whether the retrieved evidence is sufficient to support answering.

### Conflict Category

The conflict label identifies what kind of evidence situation the sample represents. This matters because the correct behavior is not constant across all tasks. A system should not behave the same way on a straightforward factual question, a temporal conflict, and a conflicting-opinion case.

Conflict awareness is therefore not an auxiliary feature of the framework. It is a central design principle.

### Model Answer

The model answer is the primary object of evaluation. CATS v2 evaluates the answer as a behaviorally situated artifact:

- Did it answer or abstain appropriately?
- Did it treat the evidence structure correctly?
- Did it make supportable claims?
- Did it include the answer the user needed?

### Gold Answer

A gold answer is used only for samples where a single-truth target exists. Not every conflict setting has a single canonical answer. For that reason, gold-answer use is conditional rather than universal.

## Conflict Awareness As A Research Principle

The framework assumes that evidence conflict is not noise to be ignored, but a meaningful property of the task. The same answer can be good in one conflict setting and poor in another.

For example:

- In a single-truth factual task, the system should identify and state the correct target answer.
- In a temporal conflict task, the system should privilege the most current and appropriately supported information.
- In a source-quality conflict task, the system should not flatten all documents into equal authority.
- In an opinion-bearing conflict task, the system may need to represent disagreement rather than force a false single answer.

This means the evaluation target is not just factual correctness. It is conflict-appropriate answer behavior.

## The Four Metrics

## 1. Grounded Refusal

### Purpose

Grounded Refusal evaluates whether the system made the correct high-level decision about answering versus abstaining.

This metric addresses a foundational question:

`Given the retrieved evidence, should the system have answered at all?`

The motivation is simple. In trust-sensitive settings, a wrong decision to answer can be as damaging as a wrong answer. Likewise, unnecessary refusal can make the system unhelpful even when the evidence is sufficient.

### Inputs Needed

Grounded Refusal uses:

- the model answer,
- and a notion of whether the question is answerable under the retrieved evidence.

That answerability signal may come from an explicit answerability annotation or be derived from the structured evidence notes.

### Metric Logic

The logic has two sides:

1. determine whether the evidence was sufficient to support answering,
2. determine whether the model actually answered or refused.

The sample receives credit when those two decisions align.

This yields four conceptual cases:

- answerable and answered,
- answerable and refused,
- unanswerable and answered,
- unanswerable and refused.

Only the first and fourth are desirable.

### What It Measures

Grounded Refusal isolates abstention quality. It does not ask whether the answer content was good. It asks whether the system crossed the threshold into answering at the right time.

### Why It Matters

A system that answers everything may look helpful but is unsafe under uncertainty. A system that refuses too often may look cautious but is practically unusable. Grounded Refusal measures this balance directly.

### Research Significance

This metric is especially important in evidence-bound evaluation because it defines the first stage of trustworthiness: the system should only attempt an answer when the available evidence supports doing so.

## 2. Behavior Adherence

### Purpose

Behavior Adherence evaluates whether the answer followed the correct strategy for the sample’s conflict type.

It asks:

`Did the system handle this evidence situation in the right way?`

This metric is what makes the framework conflict-aware rather than merely correctness-aware.

### Inputs Needed

Behavior Adherence uses:

- the query,
- the model answer,
- the conflict category,
- and, when needed, the retrieved evidence.

### Metric Logic

The metric first identifies the conflict setting, then judges the answer against the behavior that setting requires.

The notion of “correct behavior” depends on the sample. Examples include:

- answering directly when the evidence is straightforward,
- expressing uncertainty when evidence is insufficient,
- acknowledging dispute when evidence reflects genuine disagreement,
- preferring more current evidence in temporal conflict,
- or not overclaiming when support is partial and distributed.

The committee judges whether the answer adhered to the appropriate behavior for that type of evidence landscape.

### What It Measures

Behavior Adherence does not measure whether the answer was supported claim by claim. That is the job of Factual Grounding. Instead, it measures whether the answer took the right stance toward the evidence.

Two answers may share similar factual content while differing in behavior:

- one may overstate confidence,
- another may correctly surface uncertainty,
- one may erase conflict,
- another may correctly preserve it.

Behavior Adherence captures that difference.

### Why It Matters

A response can be factually grounded and still be methodologically wrong. For example, it may present one source as definitive when the evidence landscape requires a qualified or comparative answer. In research terms, this metric evaluates epistemic conduct rather than isolated factuality.

### Research Significance

This metric operationalizes the core thesis that trustworthy RAG must be sensitive to evidence structure, not only answer content.

## 3. Factual Grounding

### Purpose

Factual Grounding evaluates whether the answer’s claims are supported by eligible retrieved evidence and whether the answer points to that evidence through citations or explicit document references.

It asks:

`Are the answer’s claims actually grounded in the evidence the model had available?`

### Inputs Needed

Factual Grounding uses:

- the model answer,
- the retrieved documents,
- and the per-document notes that indicate which documents can serve as eligible support.

### Metric Logic

The metric proceeds at the level of claims rather than whole-answer impression.

The answer is decomposed into claim-sized units. For each claim, the evaluator:

1. identifies cited or referenced support documents,
2. determines which retrieved documents are eligible support documents,
3. judges whether one or more eligible documents support the claim,
4. checks whether the answer cited at least one supporting document,
5. and counts the claim as grounded only when supported evidence and citation discipline align.

The current framework also allows support to be distributed across documents. This matters because many answers are not fully supported by any one document alone. Some claims become justified only when multiple pieces of evidence are combined.

### Citation Sensitivity

This metric is intentionally stricter than a generic semantic factuality score. It cares not only about whether support exists, but whether the answer identifies supporting evidence. This reflects the research view that trustworthy evidence use is partly observable through citation behavior.

The framework therefore recognizes a broad family of inline document-reference styles, including bracketed, parenthetical, and source-attributed references.

### What It Measures

Factual Grounding measures evidence faithfulness. It does not ask whether the answer sounds correct in the abstract. It asks whether the answer is justified by the retrieved evidence set.

### Why It Matters

In RAG research, hallucination is not the only problem. A system can produce non-hallucinatory but still ungrounded responses by relying on unsupported inference, selective evidence use, or unstated priors. Factual Grounding is designed to expose those failures.

### Research Significance

This is the repository’s main evidence-faithfulness metric. It is especially important for evaluating whether retrieval is actually being used rather than merely coexisting with a plausible answer.

## 4. Single-Truth Recall

### Purpose

Single-Truth Recall evaluates whether the answer contains the target answer when the task admits a single truth target.

It asks:

`When there is a canonical answer to recover, did the system actually include it?`

### Inputs Needed

Single-Truth Recall uses:

- the model answer,
- the gold target answer,
- and the conflict type, because the metric is only meaningful in tasks where a single target answer exists.

### Metric Logic

The metric compares the answer against the gold target at the level of answer containment rather than stylistic surface form. It asks whether the target answer is present in the answer, either directly or in a way that preserves the correct substantive content.

This metric is applied only to task types where a single canonical target is conceptually appropriate. It is not universally imposed across all conflict categories, because some evidence settings are genuinely non-single-truth.

### What It Measures

Single-Truth Recall measures answer completeness with respect to the core target fact.

An answer can be well-behaved and reasonably grounded but still fail to deliver the information the user needed. This metric isolates that failure.

### Why It Matters

Trustworthy answering is not just about avoiding unsupported claims. It is also about recovering the answer that the task calls for, when such an answer exists.

### Research Significance

This metric ensures that the framework does not over-reward caution, hedging, or evidence discussion at the expense of actually giving the answer.

## How The Metrics Work Together

The four metrics are designed to capture complementary rather than redundant properties.

- Grounded Refusal captures whether the system should have answered.
- Behavior Adherence captures whether it answered in the right conflict-aware way.
- Factual Grounding captures whether its claims were justified by the evidence.
- Single-Truth Recall captures whether it contained the target answer when one exists.

Together, they form a layered view of trustworthiness.

This layered design matters because failures can occur at different levels:

- A system may fail before answering by refusing incorrectly.
- It may answer when it should not.
- It may answer in the wrong conflict-sensitive style.
- It may use evidence badly even while sounding cautious.
- It may be grounded but incomplete.

A single score is still useful, but only when it is built from these distinct components rather than treated as a primitive.

## The Role Of CATS

The Conflict-Aware Trust Score is the summary score built from the applicable sub-metrics.

Its role is not to replace the individual metrics conceptually. Rather, it provides a compact overall view once the underlying dimensions have been evaluated. In research use, the component metrics remain essential because they explain why systems differ.

For example, two systems may have similar aggregate trust scores while differing sharply in:

- abstention behavior,
- conflict-handling behavior,
- evidence faithfulness,
- or answer recall.

For that reason, the research contribution of the repository is not just the final score, but the decomposition that gives the score meaning.

## Committee Rationale In The Research Design

The current evaluation framework uses a judge committee for judgment-based metrics. The remote default is OpenRouter-backed, while the local path can use OpenAI-compatible servers for locally hosted models such as Qwen, DeepSeek distill, Gemma, and Mistral. See [LOCAL_COMMITTEE_GUIDE.md](LOCAL_COMMITTEE_GUIDE.md) for the local committee configs, staged response cache, and Sharanga deployment plan.

The purpose of this design is methodological robustness.

Conflict-aware evaluation often involves:

- nuanced interpretation of the answer,
- comparison of evidence and claim structure,
- recognition of whether uncertainty was expressed appropriately,
- and assessment of whether a target answer is substantively present.

These are not always well served by a single automatic scoring rule. The committee structure reduces dependence on one evaluator’s phrasing sensitivity or local bias. The combined signal is intended to be more stable than a single-judge decision.

## Annotation Philosophy

A key assumption of this repository is that evidence evaluation should be anchored in structured annotations, especially per-document notes and conflict categories.

This matters because “support” is not a raw lexical property of text. Whether a document supports a claim may depend on:

- recency,
- scope,
- source reliability,
- whether the support is partial or complete,
- whether support is distributed across documents,
- and whether the question is asking for a single fact or a representation of disagreement.

The framework therefore uses annotations not as incidental metadata, but as part of the evaluation theory.

## Problem Formulation

The repository can be understood as defining the following evaluation problem.

Given:

- a query `q`,
- a retrieved document set `D = {d1, d2, ..., dn}`,
- structured annotations over those documents,
- a conflict category `c`,
- and a model answer `a`,

estimate how trustworthy `a` is as an evidence-conditioned response to `q`.

The key point is that trustworthiness is not treated as a primitive scalar label. Instead, it is factorized into four dimensions:

- answerability-aligned response behavior,
- conflict-appropriate answering strategy,
- evidence-grounded claim support,
- and target-answer recovery when a single-truth target exists.

This formulation reflects a broader methodological stance: in conflict-rich RAG, answer quality is not exhausted by semantic similarity to a reference answer. It must also account for whether the answer was warranted, how it handled disagreement, and whether it used evidence responsibly.

## Evaluation Assumptions

The framework rests on several explicit assumptions.

### Evidence-bounded evaluation

CATS v2 evaluates answers relative to the retrieved evidence, not relative to unrestricted world knowledge. This means an answer may be globally correct and still score poorly if it is not justified by the evidence available to the model.

This assumption is intentional. The framework is designed to measure retrieval-conditioned trustworthiness rather than unconstrained factual competence.

### Structured conflict is meaningful

The framework assumes that conflict categories encode real differences in what good answering behavior should look like. It therefore rejects the idea that one universal response style is adequate for all evidence settings.

### Support is annotation-guided

The framework assumes that per-document notes are meaningful supervision for determining eligible support. It does not infer the entire support structure from raw text alone.

### Citation behavior is part of trustworthiness

The framework assumes that evidence use should be at least partly visible in the answer. For that reason, factual grounding is intentionally citation-sensitive rather than purely latent-semantic.

### Not every task has a single canonical answer

The framework assumes that single-truth recall is appropriate only for some task types. It therefore treats answer containment as conditional rather than universal.

## Why These Metrics Are Separated Instead Of Merged Early

An important design decision in CATS v2 is to keep the metrics separate until the final aggregation stage.

This is methodologically important for at least three reasons.

### Different failure modes require different interventions

If a system has low Grounded Refusal, the likely problem is calibration under evidence insufficiency. If it has low Factual Grounding, the likely problem is evidence faithfulness. If it has low Behavior Adherence, the likely problem is conflict-sensitive reasoning. These are not the same failure.

### Aggregate scores can hide scientifically meaningful differences

Two systems can have similar overall trust scores while failing in very different ways. A decomposed framework preserves that information.

### Trustworthiness is multidimensional by nature

The repository is built on the premise that trustworthy answering is not a single latent variable that can be directly observed. It is better modeled as an organized set of partially independent properties.

## Intended Research Use Cases

This repository is especially well suited for work that compares systems along dimensions other than raw answer correctness.

Examples include:

- comparing baseline and fine-tuned RAG systems under conflicting evidence,
- measuring whether prompt strategies improve abstention behavior,
- evaluating whether models become more grounded when given stronger evidence-formatting cues,
- analyzing whether systems overfit to answer production while failing to cite support,
- testing whether models distinguish conflict categories in a behaviorally meaningful way,
- and studying how evidence-aware reasoning changes when the retrieval set contains disagreement or partial support.

It is also useful for ablation-style research where one wants to isolate which aspect of trustworthiness improved and which did not.

## Intended Non-Use Cases

The framework is not designed for every possible evaluation setting.

It is not primarily intended for:

- pure open-book QA evaluation without conflict structure,
- tasks where retrieved evidence is unavailable or irrelevant,
- settings where citation-sensitive grounding is not meaningful,
- or scenarios where one wants only a surface-level semantic similarity score.

The framework may still be adaptable to some of these settings, but that is not the conceptual center of the repository.

## Threats To Validity

Like any research evaluation framework, CATS v2 has validity limits. Making them explicit is part of using the framework responsibly.

### Annotation dependence

The quality of evaluation depends heavily on the quality of the per-document notes, conflict labels, and answerability supervision. If those are noisy or inconsistent, the downstream metrics inherit that noise.

### Judge dependence

Behavior Adherence, Factual Grounding, and Single-Truth Recall rely on LLM judges. Even with the current three-judge local committee, these judgments are not perfectly objective. The committee improves robustness, but it does not remove evaluator dependence entirely.

### Evidence-boundedness as both strength and limitation

Because the framework is evidence-bounded, it may penalize answers that are true but unsupported by the retrieved set. This is a feature for retrieval-faithfulness research, but a limitation if the research question is unrestricted factual knowledge.

### Citation-style sensitivity

Grounding is sensitive to whether evidence use is visible in the answer. This is a deliberate design choice, but it means that stylistic differences in citation behavior can affect measured grounding.

### Conflict taxonomy scope

The usefulness of Behavior Adherence depends on whether the conflict taxonomy adequately captures the evidence phenomena that matter for the task distribution being studied.

## Practical Interpretation In A Paper

When reporting results from this repository in a paper, the strongest use of the framework is usually not just to present the final CATS score, but to explain the profile behind it.

Examples of paper-style interpretation:

- A model may improve CATS mainly by improving Grounded Refusal, indicating better abstention calibration.
- A model may improve Factual Grounding without improving Behavior Adherence, suggesting better evidence use but weak conflict-sensitive strategy.
- A model may improve Single-Truth Recall while degrading Grounded Refusal, indicating greater answer aggressiveness at the expense of caution.
- A model may score strongly on straightforward tasks but poorly on disagreement-heavy ones, revealing limited conflict awareness.

The framework therefore supports both benchmarking and analysis.

## What Kind Of Research This Supports

This repository is suitable for research on:

- retrieval-augmented generation under conflicting evidence,
- abstention behavior and answerability,
- evidence-grounded answer evaluation,
- conflict-aware answer strategy,
- trust-oriented model comparison,
- citation-sensitive grounding analysis,
- and judge-based evaluation of RAG outputs.

It is particularly useful where the goal is not merely to test whether a model can produce the right answer, but whether it can do so responsibly under imperfect evidence conditions.

## Intended Interpretation Of Results

Results from this framework should be interpreted dimensionally.

Examples:

- Low Grounded Refusal suggests poor abstention calibration.
- Low Behavior Adherence suggests weak conflict awareness.
- Low Factual Grounding suggests poor evidence faithfulness or weak evidence citation.
- Low Single-Truth Recall suggests answer omission even when other dimensions are acceptable.

The framework is therefore diagnostic, not merely ranking-oriented.

## Scope And Supporting Documents

This README is the top-level scientific and reproduction index, not a
replacement for every specialized document. It gives enough context to
understand the whole repository and points to the detailed sources that should
be consulted for exact procedures:

- `CATS_METRICS_METHODOLOGY.md`: implementation-locked metric definitions, formulas, denominators, edge cases, and paper-ready metric language.
- `CATS_AGGREGATE_LOGIC.md`: current hierarchical CATS aggregate, rationale, reviewer objections, and balanced/prevalence definitions.
- `LOCAL_COMMITTEE_GUIDE.md`: local judge models, prompts, serving, GPU placement, cache staging, Slurm orchestration, failure recovery, and ACL methods language.
- `prompts/`: latest local committee task prompts and the shared JSON-only system instruction.
- `CURRENT_REPO_MAP.md`: curated canonical paths and main/legacy boundary.
- `exports/cats_human_eval_cli/README.md`: human-evaluation CLI package and reviewer workflow.
- `exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_full_receipts/agreement_analysis/agreement_report.md`: current human-human, human-committee, and committee-internal agreement analysis.
- `legacies/README.md`: what was archived and how to produce a clean submission copy.

Operational details are included here at a high level and linked in full from
`LOCAL_COMMITTEE_GUIDE.md` so the root README can serve both as a research
overview and as the entry point for reproduction.

## Summary

CATS v2 is a research framework for evaluating trustworthiness in retrieval-augmented generation under evidence conflict. Its central contribution is a decomposition of answer quality into four complementary dimensions:

- answer-vs-refusal appropriateness,
- conflict-aware behavior,
- evidence-grounded factuality,
- and target-answer recovery when a single truth target exists.

By combining these dimensions, the repository supports a richer notion of RAG evaluation than standard correctness-based metrics alone. It is designed for work where the central question is not simply whether an answer is correct, but whether it is trustworthy relative to the evidence it used and the conflict structure of the task.

## Repository-Wide Research and Reproduction Specification

## 1. Current Production Status

This section is the version-lock for the curated repository.

### 1.1 Current benchmark facts

| Item | Current canonical value |
| --- | --- |
| Standard local benchmark | 736 examples |
| Conflict regimes | 5 |
| Current paper-facing experiment matrix | Exactly 108 rows |
| Current local benchmark committee | Qwen3.5-397B-A17B, Mistral Small 4, DeepSeek-R1-Distill-32B |
| Local committee priorities | 6, 3, 2 respectively |
| Committee voting | Weighted majority |
| Active judge tasks | Behavior Adherence, committee Factual Grounding v2, Single-Truth Recall |
| Deterministic task | Grounded Refusal |
| Active aggregate | cats_h_gated_harmonic_v1 |
| Primary aggregate display | CATS-Balanced and CATS-Prevalence as secondary summaries |
| Human-evaluation package | exports/cats_human_eval_cli/ |
| Main legacy archive | legacies/ |

The exact four categories of the 108-row matrix are:

- 96 standard benchmark rows;
- 6 answer-only SFT rows;
- 2 Llama comparison rows;
- 4 latest fixed Mistral/Qwen CoT few-shot and CoN rows.

The 108 rows are the experiment scope for the paper-facing master results. Do
not expand that number by counting historical, unfixed, staged, or duplicate
detailed-results files. Some legacy tools still enumerate a broader 114-file
universe; their six extra paths are intentionally outside the current matrix.

### 1.2 Canonical output files

The latest local-committee master artifacts are:

~~~text
outputs/benchmark_local_committee_3judge/master_results/
  cats_master_results_20260731_hierarchical.csv
  cats_master_results_20260731_hierarchical.json
  cats_master_results_20260731_hierarchical.md
  cats_master_results_20260731_hierarchical_audit.json
  cats_master_results_20260731_hierarchical_audit.md
~~~

The current workbook is:

~~~text
outputs/master_results_20260731_hierarchical.xlsx
~~~

The workbook is a presentation artifact generated from audited source results.
For any scientific claim, trace the cell back to the source row, then to the
run-local final/detailed_results.json, and finally to per_sample records.

### 1.3 Canonical dataset facts

The repository's current data map is:

- train split: 609 examples;
- validation split: 49 examples;
- benchmark holdout: 736 examples;
- validation/gold-ceiling pilot: 49 examples;
- current human evaluation: a separate sampled study package under exports/.

The 609/49 train-validation split is not the 736-example benchmark holdout.
They serve different experimental purposes and must not be presented as one
dataset count.

## 2. How To Read This Repository

A paper author or reviewer should use the following reading order:

1. This README for the end-to-end scientific and artifact map.
2. CURRENT_REPO_MAP.md for canonical paths and legacy boundaries.
3. CATS_METRICS_METHODOLOGY.md for every metric definition, formula, denominator, and limitation.
4. CATS_AGGREGATE_LOGIC.md for the hierarchical CATS design and hostile-reviewer defenses.
5. LOCAL_COMMITTEE_GUIDE.md for local judge prompts, serving, caching, Slurm, and reproduction.
6. prompts/ for the current BA, FG-v2, STR, rubric, and JSON-system prompt copies.
7. outputs/benchmark_local_committee_3judge/README.md for the benchmark output layout.
8. exports/cats_human_eval_cli/README.md and REVIEWER_USER_MANUAL.md for the human-evaluation study package.

The specialist files are complementary, not competing descriptions. If a
formula or implementation detail differs between a historical report and the
current source code, the active source code, current config, and audited current
result artifacts take precedence.

## 3. Repository Map

### 3.1 Root directories

| Path | Role | Paper/reproduction status |
| --- | --- | --- |
| rag_eval/ | Core evaluator, prompt generators, committee client, metrics, data normalization | Active source |
| run_evaluation.py | Main evaluation entry point | Active source |
| configs/ | YAML experiment and committee configurations | Active; select exact config |
| data/ | Canonical train/validation/benchmark data | Active canonical data |
| final_model_outputs/ | Raw model outputs used to prepare evaluation inputs | Active provenance |
| inputs/ | Prepared evaluator inputs, including benchmark variants | Active evaluator inputs |
| outputs/ | Local committee results, reports, caches, audits, workbook sources | Active result artifacts |
| prompts/ | Paper-facing copies of current committee prompts | Active documentation artifact |
| exports/cats_human_eval_cli/ | Standalone human-evaluation pipeline and current study returns | Active human-eval package |
| slurm/sharanga/local_committee/ | Server and orchestration assets for local judges | Active deployment assets |
| scripts/ | Preparation, orchestration, audit, merge, and workbook utilities | Active utilities |
| legacies/ | Superseded material retained without deletion | Exclude from clean paper submission unless provenance is needed |
| logs/ | Runtime logs | Reproduction/debugging artifact; do not use as sole scientific source |

### 3.2 Active source files

The minimum evaluator source set is:

- rag_eval/evaluator.py: per-example orchestration and aggregate construction;
- rag_eval/conflict_eval.py: BA, FG-v2, STR execution paths;
- rag_eval/judge_prompts.py: executable prompt generators;
- rag_eval/judge_committee.py: provider transport, parsing, caching, voting;
- rag_eval/metrics.py: refusal detection, GR formulas, claim extraction;
- rag_eval/data.py: answerability and gold-answer normalization;
- rag_eval/config.py: dataclasses and committee construction;
- run_evaluation.py: YAML overlay and CLI entry point.

### 3.3 Active configurations

The principal configurations are:

~~~text
configs/benchmark_local_openai_3judge_qwen397.yaml
configs/local_staged/benchmark_local_stage_qwen397_collect.yaml
configs/local_staged/benchmark_local_stage_mistral4_collect.yaml
configs/local_staged/benchmark_local_stage_deepseek32_collect.yaml
configs/local_staged/benchmark_local_stage_final_readonly.yaml
configs/val_tier2_local_openai.yaml
configs/val_tier2_local_openai_2xh200_fallback.yaml
configs/local_staged/
configs/local_staged_gold_ceiling/
~~~

The benchmark configuration is the one to use for current 736-example local
committee evaluations. The val-tier configurations are separate validation or
ceiling lanes and must be named as such in paper tables.

## 4. End-to-End Scientific Workflow

### 4.1 Research pipeline

The repository implements this sequence:

1. Construct or obtain a dataset with query, retrieved documents, annotations, conflict type, answerability, and optional gold answer.
2. Generate model outputs under a named variant, model family, training regime, prompt mode, and split.
3. Prepare evaluator inputs while preserving sample IDs and source provenance.
4. Evaluate each output with deterministic GR and applicable committee tasks.
5. Store per-sample judgments and all intermediate evidence needed to audit them.
6. Aggregate component metrics with their own applicability denominators.
7. Construct per-example Answer Quality and hierarchical CATS summaries.
8. Audit current results against source files and regenerate the master workbook.
9. Run human-agreement analysis where human review exists.
10. Report component metrics first, aggregates second, and uncertainty/limitations explicitly.

### 4.2 Unit of analysis

The fundamental unit is one model answer on one benchmark sample. The
evaluator does not begin by averaging dataset columns and then combining those
averages. It first computes example-level decisions, claims, judge outcomes,
and applicability, then aggregates.

This ordering is essential because:

- FG is claim-sensitive inside each answer;
- STR can be unavailable for conflict Type 3;
- correct refusals have no answer content to judge;
- committee disagreement is retained continuously for CATS;
- and a wrong answer/refusal decision gates the example score to zero.

### 4.3 Separation of evaluated model and judge models

The evaluated model produces the answer under study. The judge committee sees
that answer and relevant context but does not alter it. Judge model IDs, serving
versions, priorities, and prompts are evaluation conditions, not properties of the
evaluated model.

Any comparison must hold the judge protocol fixed unless the experiment is
explicitly a judge-sensitivity or evaluator-ablation study.

## 5. Exact Metric Stack

### 5.1 Grounded Refusal

Let A_i be gold answerability and predicted_A_i be the deterministic parser's
answer decision:

~~~text
A_i = 1 when evidence supports answering
A_i = 0 when the correct action is refusal
predicted_A_i = 1 when the model answered
predicted_A_i = 0 when the model refused
g_i = 1[predicted_A_i == A_i]
~~~

Gold answerability precedence is:

1. expected_response.abstain, if present;
2. answerable_under_evidence, if present;
3. historical support-verdict fallback.

The answer/refusal parser operates after think-trace removal. Empty output is
treated as refusal. The parser is start-oriented and recognizes canonical
insufficient-evidence and inability-to-answer openings.

The answer-positive confusion matrix is:

~~~text
TP = answered and gold-answerable
FP = answered and refusal-required
FN = refused and gold-answerable
TN = refused and refusal-required
~~~

The reported metrics are:

~~~text
answer_precision = TP / (TP + FP)
answer_recall    = TP / (TP + FN)
answer_f1        = 2 * precision * recall / (precision + recall)
gr_accuracy      = (TP + TN) / N
~~~

A zero is used when a denominator is zero. Refusal-positive metrics reorient
the same matrix:

~~~text
refusal_TP = TN
refusal_FP = FN
refusal_FN = FP
refusal_TN = TP

refusal_precision = TN / (TN + FN)
refusal_recall    = TN / (TN + FP)
refusal_f1        = harmonic mean of refusal precision and recall
~~~

GR is deterministic and is not a local committee vote.

### 5.2 Behavior Adherence

BA asks whether the answer follows the policy appropriate to one conflict type:

| Type | Conflict regime | Required policy |
| ---: | --- | --- |
| 1 | No Conflict | Answer directly without inventing alternatives or uncertainty. |
| 2 | Complementary Information | Reconcile compatible partial answers into one coherent answer. |
| 3 | Conflicting Opinions or Research Outcomes | Represent disagreement neutrally rather than declaring one side uncontested. |
| 4 | Outdated Information | Prioritize current evidence and optionally acknowledge superseded information. |
| 5 | Misinformation | Reject inaccurate sources and rely on reliable, verified evidence. |

BA must not secretly re-score answerability, factual entailment, citation
validity, unsupported-claim detection, or STR. Those are separate constructs.

For each valid judge:

~~~text
v_j = binary adherent decision
c_j = confidence in that decision
p_j = configured priority
w_j = p_j * max(c_j, 0.01)
~~~

The weighted committee decision is:

~~~text
W_plus  = sum(w_j for adherent judges)
W_minus = sum(w_j for non-adherent judges)
binary_BA_i = 1[W_plus > W_minus]
consensus_i = W_plus / (W_plus + W_minus)
~~~

A tie is non-adherent. The directly reported BA component is the applicable
example mean of binary_BA_i. CATS uses consensus_i so a split committee and a
unanimous committee are not treated identically.

### 5.3 Committee Factual Grounding v2

FG begins with deterministic claim/citation extraction. The active benchmark
configuration uses a maximum of 8 claims per answer. The extractor protects
internal sentence periods, detects bracketed/parenthetical/bare document IDs,
removes citation text, drops citation-only and source-meta fragments, filters
very short fragments, supports concise cited-answer fallback, and retains
claim-level citation details.

Only documents with normalized verdicts equivalent to supports or partially
supports are eligible. For each claim, the committee identifies documents whose
text conveys the claim and optionally a two-document combination when no single
document suffices.

A claim is grounded if:

~~~text
(single-document support exists AND the answer cites a supporting document)
OR
(cross-document support exists AND the answer cites a combination document)
~~~

For three valid judges, FG support requires:

- supporting priority mass strictly greater than half of valid raw priority mass;
- at least two valid judges naming the document when multiple judges are available.

The one-valid-judge case permits one positive judge as a transparent degraded
fallback. FG does not use judge confidence in document-support weights, does not
award graded partial credit for a partially supporting document, and does not add
a contradiction penalty in the active FG-v2 path.

Example FG:

~~~text
FG_i = supported_extracted_claims_i / extracted_claims_i
~~~

Dataset FG is the macro-average of applicable example FG values. It is not a
pooled claim micro-average.

### 5.4 Single-Truth Recall

STR applies only when:

~~~text
gold answer exists AND conflict_type in (1, 2, 4, 5)
~~~

Type 3 is excluded because conflicting opinions do not necessarily have one
canonical truth to assert.

For each gold answer item, the prompt asks whether the model asserts the target
as its own conclusion. Merely quoting a document, listing the answer as one
possibility, or rejecting it does not count. Paraphrases and logically equivalent
formulations can count.

If any exact semantic match exists, STR_i=1. If there is no exact match but a
negative judgment has positive-side minority confidence at least 0.30, it may
qualify as a partial match:

~~~text
STR_i = min(1, 0.5 * partial_matches / gold_answer_count)
~~~

Otherwise STR_i=0. Dataset STR is the applicable-example macro-average.

### 5.5 Applicability counts

Every result must retain:

~~~text
behavior_n
fg_n
str_n
answer_quality_n
~~~

A zero score with a positive count is measured performance. An unavailable score
with a zero count is a denominator condition. They must never be conflated in
paper tables.

Correct required refusals are excluded from BA, FG, STR, and Answer Quality.
They contribute GR correctness and decision-only CATS under the current policy.

### 5.6 Answer Quality and CATS

Answer Quality is computed per example:

~~~text
q_i = sqrt(FG_i * STR_i) if both apply
q_i = FG_i               if only FG applies
q_i = unavailable        if FG is unavailable
~~~

The active aggregate version is cats_h_gated_harmonic_v1. For answerable or
otherwise non-correct-refusal examples:

~~~text
s_i = g_i * harmonic_mean(behavior_consensus_i, q_i)
~~~

When only one of the two continuous inputs exists, the available input is used
after the same GR gate. For correct required refusals:

~~~text
s_i = g_i
~~~

The harmonic fusion avoids cube-root inflation of weak content values and the GR
gate prevents downstream quality from rescuing an incorrect answer/refusal
decision.

CATS-Prevalence is the arithmetic mean of complete per-example scores and
preserves the empirical benchmark distribution. CATS-Balanced averages
decision-balanced conflict-type scores, giving equal weight to conflict regimes
and balancing answerable/refusal-required subgroups within each type where both
exist. Both are secondary summaries. The primary scientific evidence is the
component metric profile.

For the full mathematical specification, see
CATS_METRICS_METHODOLOGY.md and CATS_AGGREGATE_LOGIC.md.

### 5.7 Executable edge-case contract

The following implementation details are part of the current benchmark
protocol, not optional conveniences:

- `configs/benchmark_local_openai_3judge_qwen397.yaml` sets
  `max_claims_per_answer: 8`; the evaluator dataclass default is 5, so
  reproductions must use the benchmark YAML rather than relying on defaults.
- Correct required refusals are excluded from BA, FG, STR, and Answer Quality.
  Their CATS contribution is `g_i`, which is 1 for a correct refusal and 0 for
  an incorrect decision. This is decision-only credit, not a claim that the
  refusal wording or rationale has passed a separate quality rubric.
- An answered example with no extracted claims has applicable FG and receives
  FG=0; it is not silently removed from the denominator. A correct refusal is
  the distinct exception and has those answer-content metrics marked
  inapplicable.
- Committee BA and STR use only valid, parseable judge responses. Their
  continuous consensus uses configured priority multiplied by confidence with
  a 0.01 confidence floor; a weighted tie is non-adherent for the binary
  component. FG support uses raw judge priority and its explicit corroboration
  threshold, not judge confidence.
- Active FG uses gold-eligible supporting or partially supporting documents,
  citation linkage, and committee support votes. It does not use NLI confidence
  weighting, graded partial document credit, or a contradiction penalty in the
  production FG-v2 path.
- Type 3 is excluded from STR by design. Its zero-valued STR field must not be
  interpreted as a measured recall failure, and its zero `str_n` must be
  preserved in tables.
- The legacy flat average may remain in serialized diagnostics for historical
  comparison, but it is not the paper-facing CATS aggregate and must not be
  substituted for `cats_prevalence_score` or `cats_balanced_score`.

## 6. Dataset, Variant, and Output Organization

### 6.1 Canonical data paths

~~~text
data/splits/92p5_7p5/
  stagewise_multi/train/stage3_final.jsonl
  stagewise_multi/val/stage3_final.jsonl
  monolithic_multi/train/monolithic_final.jsonl
  monolithic_multi/val/monolithic_final.jsonl
  train_ids.json
  val_ids.json
  test_ids.json
  split_manifest.json

data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl
data/ceiling_pilots/val_stage3_gold_expected_as_model_output.jsonl
~~~

The benchmark holdout is the central comparison set for the local committee
pipeline. The train and validation files belong to the annotation/training
workflow and should not be substituted for the holdout in benchmark tables.

### 6.2 Model-output paths

~~~text
final_model_outputs/
  llama8b/
  mistral7b/
  qwen7b/
  qwen32b/
  answer_only_sft/
~~~

Each model family has multiple evaluation families and prompt/training modes,
including e2e, oracle variants, baseline/SFT, minimal/runtime/strict, and
answer-only or technique-specific comparison outputs. The variant identity is
part of the result key. Never merge outputs solely by model name.

### 6.3 Prepared evaluator inputs

~~~text
inputs/prepped_model_eval_inputs/benchmark_set_all_modes/
inputs/prepped_model_eval_inputs/other_techniques/
inputs/prepped_model_eval_inputs/other_techniques_fixed/
~~~

Prepared inputs are the boundary between raw model-generation provenance and the
committee evaluator. They must preserve sample IDs, conflict labels, retrieved
documents, document notes, answerability, and gold answers where applicable.

### 6.4 Result layout

~~~text
outputs/benchmark_local_committee_3judge/
  benchmark_set_all_modes/
  other_techniques/
  other_techniques_fixed/
  citation_quality_posthoc/
  master_results/
  response_cache/
  run_outputs/
~~~

A run directory normally contains detailed_results.json, eval_report.md, and
run_config.yaml. The result file, not only the Markdown report, should be kept
for independent recalculation.

### 6.5 Fixed comparison provenance

The current fixed comparison rows are the latest fixed results:

- fixed CoT Mistral: latest applicability count retained;
- fixed CoT Qwen: latest applicability count retained;
- fixed CoN Mistral: latest applicability count retained;
- fixed CoN Qwen: latest applicability count retained.

Unfixed comparison results and staged collection files remain provenance
artifacts but are excluded from the 108-row master scope.

## 7. Local Committee Reproduction

The complete local deployment and operational guide is
LOCAL_COMMITTEE_GUIDE.md. The essential production protocol is summarized here.

### 7.1 All-at-once run

Use the exact benchmark config:

~~~bash
python run_evaluation.py \
  --input inputs/prepped_model_eval_inputs/benchmark_set_all_modes/<variant>/input.jsonl \
  --config configs/benchmark_local_openai_3judge_qwen397.yaml \
  --committee local
~~~

All three judge endpoints must be available and the output/cache paths must be
unique to the input variant.

### 7.2 Staged run

Use one collector per judge with the same response_cache_dir:

~~~text
configs/local_staged/benchmark_local_stage_qwen397_collect.yaml
configs/local_staged/benchmark_local_stage_mistral4_collect.yaml
configs/local_staged/benchmark_local_stage_deepseek32_collect.yaml
configs/local_staged/benchmark_local_stage_final_readonly.yaml
~~~

The collectors write per-judge responses. The final read-only configuration
combines those cached responses with the complete three-judge committee. A
collection output is not a final score.

### 7.3 Endpoint validation

Always test a real chat completion, not just model listing:

~~~bash
python slurm/sharanga/local_committee/probe_openai_endpoint.py \
  --base-url http://<host>:<port>/v1 \
  --model local/qwen3.5-397b-a17b \
  --timeout 180
~~~

The same must be done for Mistral and DeepSeek. Check JSON output, model ID,
latency, and absence of unexpected think-only output.

### 7.4 Hardware placement

The validated benchmark placement is:

~~~text
Qwen397      -> 2x H200
Mistral4     -> 2x H100
DeepSeek32   -> 1x A100
~~~

Mistral's A100 path previously failed during a real completion despite serving
/v1/models. Use H100/H200 unless the A100 route is revalidated.

### 7.5 Cache discipline

Cache entries are keyed by evaluation mode, sanitized model ID, and SHA-256 of
the fully rendered prompt. Do not reuse caches across prompt versions, input
variants, committee priorities, or model-output sources.

For final results:

- no unintended cache misses;
- all three model subdirectories populated;
- no all-judge failures;
- final run in read_only mode;
- source config and cache root preserved.

### 7.6 Prompt provenance

The current paper-facing prompt bundle is `prompts/`. It contains exactly the
latest local-committee task materials:

~~~text
prompts/behavior_adherence_prompt.template.txt
prompts/behavior_rubric.md
prompts/factual_grounding_prompt.template.txt
prompts/single_truth_recall_prompt.template.txt
prompts/committee_json_system_prompt.txt
~~~

These are inspection/reproduction copies with explicit placeholders. The
executable generators remain in `rag_eval/judge_prompts.py`, where runtime
values such as query, answer, conflict type, eligible documents, citations,
and provenance are rendered. Historical/NLI prompt files are not part of the
current local-committee prompt bundle; the presence of legacy NLI code should
not be read as evidence that NLI was used for the active benchmark results.

## 8. Human Evaluation

The human-evaluation package is intentionally kept separate from the local
committee and the main CATS result matrix.

### 8.1 Package location

~~~text
exports/cats_human_eval_cli/
~~~

It contains:

- standalone study initialization;
- balanced reviewer assignments;
- interactive review capture;
- draft/submitted state persistence;
- raw and enriched judgment exports;
- consolidation and agreement-analysis scripts.

The package preserves the same conceptual metric framing:

- GR/refusal applicability remains deterministic;
- BA uses the current conflict-type rubric;
- FG uses the same deterministic claim extraction;
- STR uses the same assertion-of-gold framing.

### 8.2 Current human-evaluation study

The current study artifacts are under:

~~~text
exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/
~~~

Reviewer return folders are kept under the study's reviewer_returns directory.
Consolidated outputs and agreement analysis are kept under consolidated
full-receipt or partial-receipt directories. Do not mix partial and full
consolidations when reporting final results.

### 8.3 Human-review workflow

The normal reviewer sequence is:

1. Initialize or open the assigned study bundle.
2. Use the stable reviewer copy and never switch to a fresh unzip midway.
3. Review only assigned samples.
4. Complete BA, FG, and STR when applicable.
5. Save/autosave and submit only after all applicable fields are complete.
6. Export raw/enriched judgments.
7. Consolidate reviewer returns using the study scripts.
8. Run coverage, duplicate, missing-review, and agreement audits.
9. Compare human judgments with local committee outputs through aligned sample/claim units.
10. Produce the detailed human results analysis and discussion report.

The reviewer manual remains the operational authority for reviewers. It should be
read before launching any review session.

### 8.4 Human metrics and interpretation

Human analysis should report:

- reviewer coverage and per-reviewer counts;
- human-human agreement;
- binary and continuous agreement where applicable;
- Cohen's kappa only with its unit, prevalence, and missingness defined;
- agreement by metric and conflict type;
- human-consensus versus local-committee agreement;
- coverage differences caused by the nonuniform final reviewer sample;
- disagreement examples and adjudication policy.

Human agreement is a validation analysis for the local committee, not a silent
replacement of the committee scores in the 108-row master matrix.

## 9. Paper-Ready Reporting

### 9.1 Primary table

The primary evaluation table should include:

- GR answer precision, recall, F1;
- GR refusal precision, recall, F1;
- GR accuracy;
- BA and behavior_n;
- FG and fg_n;
- STR and str_n;
- Answer Quality and answer_quality_n;
- per-type values or a supplemental table.

### 9.2 Secondary aggregate table

CATS-Balanced should be listed before CATS-Prevalence when a scalar summary is
needed, because it protects against one conflict regime dominating through
prevalence. Neither aggregate should replace the components.

Report:

- aggregate version;
- prevalence/balanced definition;
- five type scores;
- answerable/refusal-required subgroup scores;
- complete/unscorable status;
- sensitivity to alternative aggregation;
- confidence intervals or paired uncertainty estimates.

### 9.3 Committee methods paragraph

A faithful methods description is:

> We evaluate generated RAG responses with a locally hosted committee of three
> OpenAI-compatible judge models: Qwen3.5-397B-A17B, Mistral Small 4, and
> DeepSeek-R1-Distill-32B, assigned priorities 6, 3, and 2. The committee
> evaluates conflict-policy behavior, citation-linked factual grounding, and
> single-truth answer recovery, while grounded refusal is computed
> deterministically from benchmark answerability labels and the model's final
> answer. Behavior and single-truth judgments are aggregated with
> priority-by-confidence weighted majority; factual-grounding support uses raw
> priority mass plus a corroboration requirement. We preserve individual
> responses, confidence, rationales, weighted totals, prompt hashes, and
> applicability flags. When concurrent serving is unavailable, per-judge
> responses are collected into a shared cache and combined by a final read-only
> aggregation. We report component metrics and denominators before the
> secondary hierarchical CATS summaries.

### 9.4 Claims that require caution

Do not claim that:

- committee agreement equals human gold truth;
- CATS proves universal model trustworthiness;
- a high FG score proves external factual correctness;
- a correct refusal has been evaluated for refusal wording quality;
- a Type 3 STR value exists in the active protocol;
- an intermediate staged output is a final three-judge result;
- the legacy 114-file verifier changes the current 108-row experiment count;
- a zero metric with zero applicability means measured failure.

## 10. Audit and Quality Control

### 10.1 Core commands

From the repository root:

~~~bash
python3 -m unittest discover -s tests -q
python3 -m py_compile \
  rag_eval/metrics.py \
  rag_eval/data.py \
  rag_eval/conflict_eval.py \
  rag_eval/judge_committee.py \
  rag_eval/judge_prompts.py \
  rag_eval/evaluator.py \
  run_evaluation.py
python3 scripts/audit_cats_master_results.py
python3 scripts/update_master_results_workbook.py --help
~~~

Do not run a rewrite/update command against the master workbook unless the
intended source scope and output path are explicitly confirmed.

The current workbook is `outputs/master_results_20260731_hierarchical.xlsx`.
The older `scripts/audit_master_results_excel.py` and
`scripts/audit_master_results_excel_against_sources.py` target retired workbook
and schema names; they are not the current workbook audit entry points. Use the
audited master JSON/CSV/Markdown files and their accompanying audit artifacts
as the authoritative source checks. A current source audit may still list the
four fixed comparison files without sibling `run_config.yaml` files as
provenance warnings; that does not invalidate the 108-row source, uniqueness,
completeness, or CSV/JSON/Markdown consistency checks.

### 10.2 Expected current source audit

The authoritative current audit should establish:

~~~text
108 source rows
108 complete rows
0 unscorable examples
0 CSV mismatches
0 JSON mismatches
0 Markdown mismatches
~~~

The audit may separately report six ignored out-of-scope detailed-results files
and the four fixed-comparison provenance warnings described above. These are
expected scope/provenance conditions, not extra paper-facing experiments and
not a reason to change the 108-row master. The warnings must be explained rather
than silently ignored.

### 10.3 Audit invariants

At minimum verify:

- n equals the length of per_sample;
- every conflict type is in 1 through 5;
- TP+FP+FN+TN equals n;
- all unit-interval scores lie in [0,1];
- applicability counts equal true applicability flags;
- answer_quality_n is no greater than fg_n;
- STR excludes Type 3;
- cats_unscorable_n is zero for publishable complete runs;
- every represented type has a complete score before balanced aggregation;
- source rows are unique by source_relpath;
- fixed comparison rows use the latest fixed source files;
- workbook non-CATS values remain equal to the audited source matrix when that is the declared preservation requirement.

### 10.4 Verification hierarchy

When artifacts disagree, investigate in this order:

1. per-sample detailed_results.json;
2. run-local summary and report;
3. audited master CSV/JSON;
4. workbook;
5. historical reports and legacy outputs.

Never fix an apparent workbook discrepancy by editing the workbook manually
without identifying the source-level cause.

## 11. Reproducibility and Change Control

Treat any of the following as a new evaluation version:

- prompt wording or rubric changes;
- judge model, checkpoint, serving template, or quantization changes;
- priority or voting-strategy changes;
- refusal parser or answer normalization changes;
- claim extraction or claim-cap changes;
- FG eligibility, citation-linkage, corroboration, or cross-document rule changes;
- STR applicability or partial-match threshold changes;
- timeout, cache, or failure-exclusion policy changes;
- CATS aggregate formula changes.

Preserve old outputs under legacies/ or an explicitly named provenance directory,
create a new versioned output directory, rerun audits, and update the relevant
prompt, guide, methodology, and aggregate documents together. Never silently
overwrite a result while retaining an old run/config name.

## 12. Scientific Assumptions and Limitations

### 12.1 Evidence-bounded evaluation

CATS evaluates the answer relative to the retrieved evidence, not unrestricted
world knowledge. An answer may be true in the world but fail grounding if the
available evidence does not support it. This is a deliberate RAG-faithfulness
target.

### 12.2 Annotation dependence

Answerability, document verdicts, key facts, quotes, conflict categories, and gold
answers are supervision. Their quality and consistency constrain metric validity.

### 12.3 Judge dependence

BA, FG, and STR depend on the local committee. Multiple judges and stored
disagreement improve auditability but do not create human ground truth. Human
evaluation and sensitivity analysis should support strong claims.

### 12.4 Citation sensitivity

Grounding rewards visible evidence linkage. Citation syntax, claim segmentation,
and citation inheritance can affect scores. This is a deliberate design choice
that should be disclosed.

### 12.5 Conflict-type scope

Behavior rubrics and STR applicability assume the five-type taxonomy is
meaningful for the benchmark. Results should be reported by type because a
single overall average can hide small or difficult regimes.

### 12.6 Aggregate status

CATS is a designed secondary summary. It should not be presented as a universal
utility function or a replacement for GR, BA, FG, and STR. Scientific conclusions
should be anchored in the components, denominators, confidence intervals, and
error analysis.

## 13. Submission and Artifact Packaging

For an ACL artifact or reviewer-facing repository, include:

- this README;
- CURRENT_REPO_MAP.md;
- CATS_METRICS_METHODOLOGY.md;
- CATS_AGGREGATE_LOGIC.md;
- LOCAL_COMMITTEE_GUIDE.md;
- prompts/;
- exact active configs;
- canonical benchmark data or documented access instructions;
- prepared inputs where redistribution permits;
- final audited result CSV/JSON/Markdown;
- master workbook if useful for manual review;
- source code and test suite;
- human-evaluation package and its agreement reports, if included in the paper;
- a short artifact README naming the exact command and environment.

For a clean submission, remove only the legacies/ directory if the submission
policy permits it and the active artifacts are self-contained. Do not delete
legacy material from the working repository merely to make the submitted copy
clean.

## 14. Final Researcher Checklist

Before final paper submission, confirm:

- The paper uses the exact 736-example benchmark count where appropriate.
- The experiment matrix is exactly 108 rows.
- The four fixed comparison rows are the latest fixed versions.
- The local committee is identified as Qwen397/Mistral4/DeepSeek32 with priorities 6/3/2.
- The prompt bundle contains BA, FG-v2, STR, rubric, and system prompt only.
- The local committee guide and metric methodology match the source code.
- GR, BA, FG, STR, and Answer Quality denominators are reported.
- CATS-Balanced and CATS-Prevalence are clearly labeled secondary summaries.
- Human review coverage and agreement are reported separately from committee scores.
- All final numbers trace to audited source JSON and per-sample records.
- Every local judge endpoint was tested with a real chat completion.
- Any staged cache run used complete read-only final aggregation.
- Failed, timed-out, malformed, or missing judge responses are disclosed.
- The legacy 114-file verifier warning is not confused with the current scope.
- The clean ACL submission can be reproduced without the legacies/ directory.

## 15. Specialist Documentation Index

| Document | Use |
| --- | --- |
| CURRENT_REPO_MAP.md | Canonical current paths, counts, and legacy boundary |
| CATS_METRICS_METHODOLOGY.md | Full formula-level metric specification and scientific defense |
| CATS_AGGREGATE_LOGIC.md | CATS aggregate design, rationale, alternatives, and hostile-reviewer response |
| LOCAL_COMMITTEE_GUIDE.md | Local judge deployment, serving, cache, Slurm, exact call graph, and ACL methods |
| prompts/README.md | Current local committee prompt bundle and template variables |
| outputs/benchmark_local_committee_3judge/README.md | Benchmark output/cache layout |
| exports/cats_human_eval_cli/README.md | Human-evaluation package capabilities and commands |
| exports/cats_human_eval_cli/REVIEWER_USER_MANUAL.md | Reviewer-facing human-evaluation operation |
| exports/cats_human_eval_cli/studies/.../agreement_analysis/agreement_report.md | Human/committee agreement analysis and interpretation |
| final_model_outputs/*_benchmark_matrix_analysis.md | Model-family benchmark matrix analyses |
| slurm/sharanga/local_committee/README.md | Sharanga server scripts and hardware-specific notes |

## 16. Closing Perspective

CATS v2 is best understood as an auditable evaluation system, not just a scalar
score. Its contribution is the explicit chain from evidence conflict, to
answer/refusal decision, to conflict-aware behavior, to cited claim grounding,
to target-answer recovery, and finally to carefully qualified aggregate
summaries.

A result is scientifically meaningful only when a reader can identify:

- what evidence and annotations were supplied;
- what model answer was evaluated;
- which prompt and judge committee produced each judgment;
- which examples each metric applied to;
- how failures and disagreement were handled;
- how the aggregate was derived;
- and how the final number can be traced back to stored per-sample evidence.

That traceability is the central repository design goal and the basis for a
defensible ACL submission.
