# CATS v2

CATS v2 is a research-oriented evaluation framework for retrieval-augmented generation under evidence conflict. It is designed for settings where a model is asked to answer using retrieved documents, but the retrieved evidence may be incomplete, contradictory, temporally stale, differently reliable, or opinion-bearing. In such settings, conventional answer evaluation is not enough. A response can be fluent, superficially correct, and even partially supported, while still failing in a deeper way: it may answer when it should abstain, collapse a disagreement into false certainty, ignore recency, misuse evidence, or omit the core target answer.

This repository contributes a structured answer-evaluation methodology for that setting. Its core claim is that trustworthy RAG evaluation should not be reduced to a single notion of correctness. Instead, it should decompose trustworthiness into several complementary properties:

1. whether the system should have answered at all,
2. whether it followed the correct conflict-handling behavior,
3. whether its claims were grounded in the available evidence,
4. and whether it contained the target answer when the task admits a single truth target.

These dimensions are combined into the Conflict-Aware Trust Score, or CATS.

This README is written as a research-facing description of the framework. It focuses on the conceptual design, metric definitions, inputs, assumptions, and methodological significance of the system. It intentionally avoids implementation-heavy operational detail.

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
4. A multi-judge evaluation design that uses a current two-member committee to assess behavior, grounding, and answer recall.

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

This curated repository preserves three current committee lanes:

- Default remote code path: `claude-haiku-4-5` + `gpt-5.4` + `deepseek-v3.2`
- Current mixed val-ceiling path: `Codex CLI (gpt-5.4)` + `deepseek-v4-flash`
- Current local benchmark path: `qwen3.5-397b-a17b` + `mistral-small-4` + `deepseek-r1-distill-32b`

These committees are used for the judgment-heavy parts of the evaluation:

- Behavior Adherence
- Factual Grounding
- Single-Truth Recall

Grounded Refusal is evaluated deterministically from annotations and answer form rather than by committee judgment.

The purpose of the committee design is to reduce evaluator brittleness. Conflict-aware behavior, evidence sufficiency, and answer containment are judgment-sensitive; multi-judge evaluation provides a more stable signal than a single free-form evaluator.

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

Behavior Adherence, Factual Grounding, and Single-Truth Recall rely on LLM judges. Even with a two-judge committee, these judgments are not perfectly objective. The committee improves robustness, but it does not remove evaluator dependence entirely.

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

## Scope And Deliberate Exclusions

This README intentionally does not focus on operational matters such as runtime workflows, batch-launch procedures, local environment setup, or engineering infrastructure. Those details are secondary to the scientific role of the repository.

The main purpose of this document is to describe:

- the research problem,
- the evaluation framework,
- the meaning of each metric,
- the assumptions behind the annotations,
- and the logic by which the repository contributes to a conflict-aware trust evaluation methodology for RAG.

## Summary

CATS v2 is a research framework for evaluating trustworthiness in retrieval-augmented generation under evidence conflict. Its central contribution is a decomposition of answer quality into four complementary dimensions:

- answer-vs-refusal appropriateness,
- conflict-aware behavior,
- evidence-grounded factuality,
- and target-answer recovery when a single truth target exists.

By combining these dimensions, the repository supports a richer notion of RAG evaluation than standard correctness-based metrics alone. It is designed for work where the central question is not simply whether an answer is correct, but whether it is trustworthy relative to the evidence it used and the conflict structure of the task.
