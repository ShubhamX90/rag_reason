# Conflict-Aware RAG Dataset Annotation Pipeline

## Clean repository layout

The top-level tree contains only the retained current workflow and its latest
artifacts. Legacy runners, superseded datasets/splits, pilot outputs, duplicate
export packs, unused external-repository files, and generated metadata are kept
reversibly under [`legacies/`](legacies/); they are not required for reproducing
the current results and can be omitted from a submission archive.

The main paths are:

- `src/` and `scripts/`: current annotation, voting, validation, benchmark-build,
  and human-review logic
- `configs/local_committee/`, `prompts/`, and `slurm/`: current local committee
  configuration, prompts, and launchers
- `data/releases/`: canonical reviewer-facing released datasets (`train=862`,
  `val=81`, benchmark holdout=`736`)
- `data/final_annotations/`: current 658-example training annotation outputs
- `data/splits/92p5_7p5/`: current internal annotation-pool split artifacts
  (`609/49/0`) used by retained committee-validation workflows
- `data/benchmarks/final_benchmark_2026-06-22/`: current internal benchmark-build
  artifact (`1,000` rows) retained for pipeline reproducibility
- `human_reviews/training/` and `human_reviews/benchmark/`: separate current
  human-review populations
- `outputs/`: retained latest local committee runs for training validation,
  benchmark non-refusals, and benchmark refusals

The benchmark construction workflow is documented in
[`docs/conflicts_benchmark_build.md`](docs/conflicts_benchmark_build.md).

If you want the final released datasets first, start with
[`data/releases/README.md`](data/releases/README.md). That directory is the clean
entry point for the `862/81` training split and the `736`-example benchmark
holdout that should ship with a reviewer-facing copy of the repository.

## Overview

This repository is centered on the creation of a conflict-aware dataset for retrieval-augmented generation (RAG). Its purpose is not simply to label whether an answer is correct, but to capture the reasoning conditions under which a RAG system should trust, combine, contest, downgrade, or refuse retrieved evidence.

The core research motivation is straightforward: many RAG failures do not come from retrieval alone, and they do not come from answer generation alone. They arise from the interaction between the two. A system may retrieve documents that:

- fully support an answer
- partially support an answer
- appear relevant but omit the needed detail
- disagree because they refer to different scopes
- disagree because one source is outdated
- disagree because one source is misleading
- fail to answer the question at all

Most standard QA-style datasets collapse these cases into a single answer target. This repository instead treats them as distinct supervision problems. The dataset therefore models RAG as an evidence interpretation task, not only as an answer production task.

At a high level, the repository produces annotations for:

- document-level evidential relevance
- conflict type across retrieved evidence
- answerability under the retrieved evidence
- final grounded response behavior
- abstention when the evidence does not justify an answer

This makes the dataset suitable for research on evidence-grounded generation, retrieval error analysis, answerability prediction, abstention modeling, conflict-aware reasoning, and modular supervision for RAG systems.

## Research Objective

The central objective of the repository is to create supervision for a model that must behave correctly under imperfect retrieval.

The intended downstream system is not rewarded merely for producing a plausible answer. It is expected to:

1. inspect each retrieved document separately
2. determine which documents materially help answer the query
3. identify the relation among the useful documents
4. decide whether the evidence actually supports answering
5. produce a response whose behavior matches the evidence pattern

This is especially important for queries where the retrieved set is not internally uniform. In realistic web retrieval, the evidence can be mixed in several ways:

- one document gives the exact answer while others are noisy
- multiple documents each contribute a different piece of a complete answer
- older pages conflict with more recent ones
- low-quality pages contradict stronger sources
- different documents report incompatible claims within the same scope

The dataset is therefore designed to supervise not only correctness, but evidence-sensitive behavior.

## What The Dataset Represents

Each example begins with a user query and a set of retrieved snippets. The retrieved set is the object of study. The dataset asks: what should a careful RAG system do with this evidence?

The final annotation does not reduce that question to a single answer string. Instead, each example is represented through a layered annotation structure:

1. document-level evidence notes
2. record-level conflict reasoning
3. record-level answerability judgment
4. final evidence-grounded response

This layering reflects the underlying research view that good RAG behavior can be decomposed into interpretable sub-decisions.

## Dataset Statistics

The normalized annotation pool currently included in this repository contains **658 query-level examples**:

- **458** conflict-oriented examples
- **200** refusal-oriented examples

Across these 658 examples, the repository contains **5,189 retrieved snippets** in total:

- **4,194** snippets in the conflict-oriented pool
- **995** snippets in the refusal-oriented pool

The average retrieval set size is therefore approximately:

- **9.2 documents per query** in the conflict-oriented pool
- **5.0 documents per query** in the refusal-oriented pool
- **7.9 documents per query** overall

The normalized conflict-type distribution across the combined 658-example pool is:

| Conflict type | Count |
|---|---:|
| `No conflict` | 281 |
| `Complementary information` | 186 |
| `Conflicting opinions or research outcomes` | 117 |
| `Conflict due to outdated information` | 68 |
| `Conflict due to misinformation` | 6 |

These numbers matter because the dataset is intentionally not dominated by one single reasoning pattern. The distribution includes both clean agreement cases and more difficult cases involving contextual scope, contradiction, temporal drift, and misinformation.

## What One Example Contains

At the input level, each example contains:

- a stable example ID
- a user query
- a conflict-type label
- an optional gold answer field
- a list of retrieved documents, each with a document ID, source URL, snippet, and timestamp when available

At the annotation level, each example contains:

- a note for each retrieved document
- a conflict explanation across the set
- an answerability judgment
- a final grounded response object

Conceptually, the dataset asks not only "what is the answer?" but also:

- which document supports which part of that answer?
- which documents are irrelevant?
- is the disagreement real, temporal, contextual, or misleading?
- does the evidence justify answering at all?

## Annotation Ontology

### 1. Document-level evidence verdicts

Each retrieved snippet is assigned one of three verdicts:

| Verdict | Meaning |
|---|---|
| `supports` | The snippet directly provides decisive evidence for answering the query. |
| `partially supports` | The snippet is relevant but incomplete, indirect, scoped, hedged, or missing a required detail. |
| `irrelevant` | The snippet does not provide meaningful evidence for the query. |

This distinction is one of the most important features of the dataset. It prevents retrieval supervision from collapsing into a binary "useful vs useless" view. Many real retrieval results are neither fully decisive nor entirely irrelevant. `partially supports` is used to capture these middle cases explicitly.

Examples of why a document may be `partially supports` rather than `supports` include:

- it identifies the right entity but omits the requested date
- it gives the lower-48 answer when the query is about the entire United States
- it reports a general tendency while the query asks for an exact count
- it contains relevant but hedged language rather than a decisive claim
- it contributes only one facet of a multi-document answer

This label is especially important for research because it separates insufficiency from irrelevance.

### 2. Source quality

Each retrieved document also receives a coarse source-quality label:

- `high`
- `low`

This is not intended to be a complete theory of factual reliability. Rather, it provides a lightweight signal about source credibility that the final response can use when ordering or prioritizing evidence. The role of this label is to encourage the dataset to reflect how evidence-grounded systems should treat official, educational, institutional, and major reference sources differently from weak or miscellaneous sources.

### 3. Conflict taxonomy

Each example is labeled with one of five conflict types:

| Label | Interpretation |
|---|---|
| `No conflict` | The retrieved evidence agrees on the core answer. |
| `Complementary information` | Different documents add distinct, non-contradictory pieces that together complete the answer. |
| `Conflicting opinions or research outcomes` | The retrieved set contains genuinely incompatible claims, findings, or interpretations. |
| `Conflict due to outdated information` | The disagreement is explained by temporal change, update, or supersession. |
| `Conflict due to misinformation` | Some retrieved material is false or misleading relative to stronger evidence in the same retrieved set. |

These labels are not decorative metadata. They define how the final response should behave.

For example:

- under `No conflict`, the answer should be direct and stable
- under `Complementary information`, the answer should synthesize non-overlapping valid details
- under `Conflicting opinions or research outcomes`, the answer should represent disagreement rather than force false consensus
- under `Conflict due to outdated information`, the answer should privilege newer evidence while acknowledging older claims when relevant
- under `Conflict due to misinformation`, the answer should identify and correct misleading claims using better evidence from the retrieval set

### 4. Answerability and abstention

Two distinct fields capture evidence sufficiency:

- `answerable_under_evidence`
- `expected_response.abstain`

These fields are related but not interchangeable.

`answerable_under_evidence` is a judgment about the retrieved set itself. It answers:

> Does this evidence set contain enough relevant material to support an evidence-grounded response?

`expected_response.abstain` is a judgment about the final response behavior. It answers:

> Given the evidence and the task constraints, should the model refuse to answer?

This distinction is important for research because it separates retrieval adequacy from response behavior.

#### Answerable cases

Typical answerable cases include:

- one or more snippets directly answer the query
- multiple partially supporting snippets can be combined into a complete answer
- the evidence disagrees, but the disagreement itself can be described faithfully
- the answer depends on choosing newer evidence over older evidence
- the final response can meaningfully explain why one part of the retrieval set is misleading

#### Non-answerable cases

Typical non-answerable cases include:

- all retrieved documents are off-topic
- the documents discuss the right topic but omit the required detail
- the snippets are too vague, truncated, or indirect to support a grounded answer
- the evidence is topically adjacent but never reaches the target proposition

#### Why both fields are needed

The dataset keeps both answerability and abstention because they diagnose different failure modes:

- a retriever may fail to bring usable evidence at all
- a generator may receive usable evidence but still answer badly
- a model may abstain too often even when the retrieval set is usable
- a model may over-answer when the evidence is only weakly suggestive

This makes the dataset useful for studying calibration and evidence-sensitive refusal behavior, not only answer generation.

## Annotation Logic

The annotation process is structured around three conceptual stages. These stages are not merely operational conveniences. They reflect the intended decomposition of evidence-grounded reasoning.

### Stage 1: evidence adjudication

The first stage judges each retrieved document independently against the query.

The purpose of this stage is to answer:

- does this document help?
- how strongly does it help?
- what exactly is the key fact it contributes?

This stage is deliberately strict. The annotation requires the key fact to be grounded in a short verbatim quote from the snippet. The goal is to prevent the annotator from smuggling in outside knowledge or silently repairing a weak snippet with parametric memory.

This creates a form of local faithfulness supervision: each document-level note must remain anchored to what the retrieval system actually surfaced.

Research-wise, this is valuable because it distinguishes:

- failure of retrieval
- failure of evidence interpretation
- failure of final synthesis

### Stage 2: conflict reasoning

The second stage reasons across the set of document-level notes.

Its job is not to answer the original question directly. Instead, it must explain the structure of the evidence:

- do the documents agree?
- are they adding different facets?
- are they talking about different scopes?
- are they genuinely contradictory?
- is the apparent contradiction explained by time?
- is one part of the set misleading?

This stage is central to the dataset's research value. It transforms the retrieval set from a flat bag of snippets into an interpretable evidence configuration.

The conflict explanation is expected to identify the mechanism behind the pattern. This is important because superficially similar disagreements may differ substantially in cause. A model that cannot distinguish contradiction from complementarity, or temporal drift from misinformation, will behave poorly even if it has strong language modeling ability.

### Stage 3: final grounded response

The final stage produces the response that a RAG system should ideally give.

This response is grounded in:

- the retrieved documents
- the document-level evidence notes
- the conflict label
- the conflict explanation
- the answerability decision

The answer is therefore conditioned not only on what the documents say, but on how the retrieved set is structurally interpreted.

This stage is where the dataset enforces behavioral alignment between evidence pattern and answer style. The system should not respond the same way to all evidence configurations. A correct response under `Complementary information` is different in form from a correct response under `Conflicting opinions or research outcomes`, even if both are technically answerable.

## Why The Dataset Uses A Layered Design

The layered design reflects a specific research belief: evidence-grounded generation is easier to analyze when its sub-decisions are explicit.

A single end-to-end answer target hides too much:

- whether the model identified the useful documents
- whether it understood the source of disagreement
- whether it answered despite insufficient evidence
- whether it relied on one decisive document or many partial ones

By contrast, the current design exposes these intermediate decisions directly. This enables:

- more interpretable evaluation
- targeted supervision of failure points
- modular training objectives
- richer qualitative error analysis

The clean repository keeps the retained stagewise annotation path only. Older
one-shot strategy assets were quarantined under `legacies/` so the current
workflow reads as one coherent stagewise pipeline.

## Multi-LLM Committee Annotation

One of the central design choices in this repository is the use of a multi-LLM committee rather than a single annotator model.

The reason is methodological. When the annotation task involves nuanced evidence judgments, a single model can be brittle:

- it may over-trust weak snippets
- it may flatten complementary evidence into contradiction
- it may over-answer or over-abstain
- it may be idiosyncratic in borderline cases

The committee is used to stabilize these judgments.

### Committee purpose

The committee is intended to:

- reduce dependence on a single model's quirks
- improve stability of categorical decisions
- preserve diversity of reasoning signals
- keep a traceable audit trail for final labels

This is not a simple majority-vote setup. It is a weighted committee.

### Current committee

The current default committee is:

| Model | Weight |
|---|---:|
| `anthropic/claude-haiku-4.5` | 0.35 |
| `openai/gpt-5.4` | 0.30 |
| `deepseek/deepseek-v3.2` | 0.20 |
| `mistralai/mistral-small-2603` | 0.15 |

These weights reflect influence rather than a flat one-model-one-vote assumption.

### Committee logic

The committee does not vote on every field independently. That would produce incoherent records. Instead, it separates:

- decision fields
- supporting explanation fields

Decision fields include the core structured labels the repository wants to stabilize, such as:

- document-level verdict
- answerability
- conflict type in re-annotation settings
- abstention

Supporting fields include the text that must stay internally coherent, such as:

- key fact
- quote
- verdict explanation
- conflict explanation
- final answer
- evidence list
- abstain reason

The principle is:

1. vote on the key decision
2. identify the winning side
3. take the supporting explanation bundle from the highest-weight model on that winning side

This preserves coherence. It avoids pathological outputs in which the label comes from one reasoning path and the explanation comes from another.

### Stage-specific committee behavior

The committee behaves differently across the annotation stages.

#### Stage 1

At the document level, the committee votes on the document verdict:

- `supports`
- `partially supports`
- `irrelevant`

Once the verdict is chosen, the associated explanation fields are taken from the strongest model that voted for that verdict. This means the final note reflects both consensus on the key label and coherence in the explanation.

#### Stage 2

At the conflict-reasoning level, the committee primarily votes on `answerable_under_evidence`. In settings where conflict type itself is being re-annotated, it can also vote on `conflict_type`.

The important design point is that the conflict explanation should come from a model whose interpretation matches the selected conflict pattern. This keeps the explanatory text aligned with the chosen label.

#### Stage 3

At the final response level, the committee votes on whether the example should be answered or abstained from. After that decision is made, the final response package is taken from the strongest model that agreed with the winning abstention decision.

This means the dataset does not average answers across models. It uses collective agreement to choose the behavioral decision, then preserves one coherent answer from the winning side.

### Why this matters for research

The committee design supports a stronger annotation signal than a one-model pipeline in at least three ways:

1. it reduces label volatility in borderline evidence cases
2. it preserves disagreement traces that can be studied later
3. it yields more interpretable provenance for final labels

This is especially useful for evidence-sensitive tasks where the difficulty lies not in linguistic generation, but in judging sufficiency, contradiction, and source trust.

## What Makes The Dataset Strict

Several design choices make the resulting dataset more useful for research than a standard synthetic QA target.

### 1. It forbids silent use of outside knowledge

The annotation is grounded in the retrieved snippets and their derived notes. This matters because many RAG systems appear to answer well only by falling back to parametric memory. The dataset is designed to supervise evidence-grounded behavior instead.

### 2. It makes document-level grounding inspectable

The key-fact and quote pairing keeps evidence supervision local and auditable. This is important for studying whether a model's evidence use is real or merely rhetorical.

### 3. It distinguishes insufficiency from irrelevance

Many retrieval outputs are not fully helpful but are also not meaningless. The dataset explicitly models this intermediate regime.

### 4. It conditions answer behavior on evidence structure

The correct response is not only a factually right answer. It is an answer whose style and caution level match the evidence regime.

### 5. It separates retrieval adequacy from final refusal behavior

This enables research on over-answering, under-answering, and miscalibrated abstention.

### 6. It preserves annotation provenance

Committee vote metadata allows later study of consensus, disagreement, and uncertainty patterns within the annotation process itself.

## Why This Repository Exists

At its core, this repository exists to support research on a more realistic form of RAG behavior.

In realistic settings, retrieval does not hand the model a perfectly clean evidence packet. It hands the model a mixed set of snippets with varying relevance, varying credibility, and varying agreement structure. A robust RAG system should know how to:

- identify decisive evidence
- integrate complementary evidence
- describe disagreement faithfully
- privilege newer evidence when time matters
- resist misleading evidence
- abstain when the retrieval set does not justify answering

This repository is built around supervising exactly those behaviors.

## Summary

This repository creates a conflict-aware RAG dataset whose target is not only the final answer, but the full logic of evidence interpretation.

The dataset is designed to teach and analyze:

- what retrieved documents are actually useful
- how multiple retrieved documents relate to one another
- when evidence is sufficient to answer
- when evidence is insufficient and abstention is warranted
- how a final response should change under agreement, complementarity, contradiction, temporal drift, or misinformation

The multi-stage annotation structure and the weighted multi-LLM committee are both in service of that research goal: producing supervision that is faithful to retrieved evidence, structurally interpretable, and useful for studying evidence-grounded reasoning in RAG systems.
