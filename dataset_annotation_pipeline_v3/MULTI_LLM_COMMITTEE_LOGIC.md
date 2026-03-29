# Multi-LLM Committee Logic

This note explains the core idea behind the repository's multi-LLM annotation approach in plain language.

## Big Picture

The pipeline supports two main annotation strategies:

1. `3-stage`
2. `Monolithic`

For either strategy, the multi-LLM version does not trust a single model run.
Instead, it sends the same task to a small committee of models and then merges their outputs in a structured way.

The goal is:

- make key labeling decisions more stable
- reduce dependence on one model's quirks
- keep a clear audit trail of how the final annotation was chosen

## Committee Strategy

The committee is a fixed set of models defined centrally in `src/voting.py`.

Each model has a weight.

The weights reflect how much influence that model should have in a vote.
So this is not a simple one-model-one-vote system.
It is a weighted committee.

Conceptually:

- stronger / more trusted models get more influence
- cheaper or more diverse models still contribute signal
- the final decision is made by combining all their votes

## Core Design Principle

The committee does **not** vote on every field.

Instead, the system separates fields into two broad groups:

1. decision fields
2. supporting text fields

### Decision fields

These are the fields where the pipeline wants agreement across models.
Examples:

- `verdict`
- `answerable_under_evidence`
- `conflict_type` in the refusal-mode Stage 2 path
- `abstain`

These are the fields that get voted on.

### Supporting text fields

These are explanation or evidence fields that belong together as one model's coherent output.
Examples:

- `verdict_reason`
- `key_fact`
- `quote`
- `source_quality`
- `conflict_reason`
- final answer text
- evidence list
- `abstain_reason`
- `think`

These are **not blended** across models.
Once the winning decision is known, the pipeline takes the associated text fields from the most influential model that voted for that winning decision.

This is important.
It keeps the final record internally coherent.

The system avoids making stitched-together explanations like:

- verdict from model A
- quote from model B
- key fact from model C

unless that specific stage explicitly does per-item voting.

## How Voting Works

For a given decision field:

1. each committee model produces its own output
2. the pipeline extracts the value to vote on
3. each vote contributes that model's weight
4. weights are summed per candidate value
5. the value with the highest total weight wins

So if two lighter models disagree with one heavier model, the heavier model can still win if its total weight is larger.

If there is a tie, the implementation uses a deterministic tie-break so runs are stable.
In plain terms: ties are resolved consistently instead of randomly.

## Stagewise Multi-LLM Logic

The 3-stage strategy applies the committee independently at each stage.

That means each stage has its own voting question.

## Stage 1: Per-document Evidence Adjudication

Input:

- one query
- one retrieved document

Each model produces a Stage 1 note for that document.

The main field being voted on is:

- `verdict`

Possible verdicts:

- `supports`
- `partially supports`
- `irrelevant`

### What gets voted on

- `verdict`

### How the final Stage 1 note is chosen

After the winning `verdict` is selected, the pipeline takes the rest of the Stage 1 note from the highest-weight model that voted for that winning verdict.

That means the following fields come from one winning model output:

- `key_fact`
- `quote`
- `verdict_reason`
- `source_quality`

So the final Stage 1 note is:

- consensus on the label
- one coherent explanation bundle from the strongest model on the winning side

## Stage 2: Conflict Reasoning

Input:

- the query
- the per-document notes from Stage 1
- usually a `conflict_type`

This stage behaves differently depending on the dataset mode.

### Normal conflicts mode

In the regular conflicts dataset, `conflict_type` is treated as already given.

The committee mainly votes on:

- `answerable_under_evidence`

Then:

- the winning boolean is used as the final answerability label
- `conflict_reason` is taken from the highest-weight model that voted for that winning answerability value

So here:

- the label is voted
- the explanation comes from the strongest model on the winning side

### Refusal-mode Stage 2

In refusal mode, the committee can also re-annotate the conflict type from evidence.

In that path, the committee votes on:

- `conflict_type`
- `answerable_under_evidence`

These are independent votes.

Then:

- final `answerable_under_evidence` comes from the answerability vote
- final `conflict_type` comes from the conflict-type vote
- final `conflict_reason` is taken from the highest-weight model that voted for the winning `conflict_type`

Why do that?

Because the explanation of the conflict should match the conflict type that actually won.

So if the committee decides the true pattern is "outdated information", the conflict reason should come from a model that argued for that same interpretation.

## Stage 3: Final Expected Response

Input:

- query
- retrieved docs
- per-doc notes
- conflict reasoning
- answerability

Each model produces a final response object.

The main field being voted on is:

- `expected_response.abstain`

So the committee is deciding:

- should the final annotation answer the question?
- or should it abstain?

### What gets voted on

- `abstain`

### How the final Stage 3 output is chosen

After the winning abstain value is chosen, the pipeline takes the full response package from the highest-weight model that voted for that winning abstain decision.

That includes:

- final answer text
- evidence list
- `abstain_reason`
- `think`

So again, the pattern is:

- vote on the key decision
- keep one coherent narrative from the strongest model on the winning side

## Monolithic Multi-LLM Logic

The monolithic strategy asks each model to do everything in one shot:

- per-doc notes
- conflict reasoning
- final answer

Then the committee merges those full outputs.

Because the output contains several layers, the merge happens in pieces.

## Monolithic: Per-document Notes

For each `doc_id`, the pipeline compares the document-level note from each model.

It votes on:

- that document's `verdict`

Then for that document, it takes the explanation fields from the highest-weight model that voted for the winning verdict.

So per document, the final fields come from one winning-side model:

- `key_fact`
- `quote`
- `verdict_reason`
- `source_quality`

This makes monolithic per-doc voting behave similarly to Stage 1 voting.

## Monolithic: Final Answerability Direction

For the final response side of the monolithic output, the pipeline votes on:

- `expected_response.abstain`

Then it adopts the broader answer package from the highest-weight model that voted for the winning abstain value.

That means fields like these come from that winning-side model:

- `conflict_reason`
- final answer text
- final evidence list
- `abstain_reason`
- `think`

The per-doc notes are still replaced with the separately voted consensus notes described above.

So monolithic merging is:

1. vote document-by-document on `verdict`
2. vote globally on `abstain`
3. keep the strongest winning-side answer package
4. replace its per-doc notes with the voted consensus per-doc notes

## Why Explanations Are Not Blended

The pipeline avoids averaging or stitching explanation text across models.

This is deliberate.

If you tried to vote independently on every explanation field, you could easily end up with a broken annotation like:

- a verdict from one reasoning path
- a quote that does not support that verdict
- a key fact that belongs to a different quote
- a conflict reason that does not match the selected conflict type
- an abstain reason that does not match the abstain decision

So the pipeline instead uses:

- voting for key categorical / boolean decisions
- winner-side adoption for explanation bundles

This keeps the output readable and internally aligned.

## How the Final Annotated Entry Is Decided

The final record is built from the committee in a stage-aware way.

In plain language:

- the committee first agrees on the key label for that step
- once that label is decided, the system picks the strongest model that supported that label
- the explanation fields from that model become the final explanation

So the final entry is not "the output of one best model".
It is also not "an average of all models".

It is:

- a committee decision on the important label
- plus a coherent explanation bundle from the strongest model on the winning side

## Audit Trail

The pipeline also records voting metadata in the output.

Examples include:

- vote tallies
- which model won
- what each model voted for in some stages

This makes it possible to inspect:

- whether the committee was unanimous
- whether there was disagreement
- which model supplied the final explanation text

So the output is not just an annotation.
It is also an auditable committee decision.

## Summary in One Sentence

The multi-LLM paradigm is:

"Use weighted voting to choose the important structured decision, then take the full supporting explanation from the strongest model that agreed with that winning decision."

That is the core logic repeated across both the stagewise and monolithic strategies.
