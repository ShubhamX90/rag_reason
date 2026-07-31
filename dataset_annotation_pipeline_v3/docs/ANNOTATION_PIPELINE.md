# Stagewise LLM Annotation and Committee Pipeline

## Scope, status, and terminology

This document describes the retained **stagewise multi-LLM annotation method**:
the committees that were used, their weights, the three reasoning stages, the
mode-specific logic for ordinary, benchmark, and refusal examples, and the
provenance of the retained annotation artifacts. It is the methodology
companion to the dataset cards:
[`TRAINING_DATASET_DESCRIPTION.md`](TRAINING_DATASET_DESCRIPTION.md) explains
how the training release was assembled, while
[`BENCHMARK_DATASET_DESCRIPTION.md`](BENCHMARK_DATASET_DESCRIPTION.md) explains
the benchmark release and its human-selection process.

The repository intentionally keeps only the stagewise strategy in the active
tree. One-shot/monolithic runners are quarantined in `legacies/code/legacy_runners/`
and are not part of the current workflow or reproduction instructions.

Two meanings of “current” must be kept separate:

1. **Current executable behavior** is defined by the current `src/`, `scripts/`,
   `prompts/`, and `configs/local_committee/` files.
2. **Retained output provenance** is defined by the model names, vote metadata,
   cost reports, and run-specific local configurations recorded with an output.

The second point matters because the 658-record historical training annotation
artifact was produced with a different OpenRouter committee from the current
default configuration. Re-running the current method is therefore a
methodological reproduction, not an assertion that it regenerates
byte-identical historical model generations.

## Committee configurations: models, weights, and use

The committee is a weighted ensemble, not a flat vote. Models independently
produce structured annotations; their weights determine the outcome of each
categorical decision. The exact committee must always be reported alongside the
artifact it produced.

### Historical 658-record training annotation committee

The retained three-stage annotation artifact at
`data/final_annotations/stagewise_multi/` was finalized with the following
four-model OpenRouter committee:

| Model | Weight | Role in the recorded committee |
|---|---:|---|
| Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6`) | 0.35 | Highest-weight general evidence and reasoning judge. |
| GPT-5.4 (`openai/gpt-5.4`) | 0.30 | Independent high-capability instruction-following judge. |
| DeepSeek V3.2 (`deepseek/deepseek-v3.2`) | 0.20 | Diverse lower-cost reasoning signal. |
| Mistral Small 2603 (`mistralai/mistral-small-2603`) | 0.15 | Fourth, diverse committee signal. |
| **Total** | **1.00** | |

This is the committee relevant when discussing the historical 658-record
stagewise training artifact and its retained cost reports. It should not be
replaced in a paper with a later live-code configuration.

### Local benchmark-construction committee: 800 non-refusals and 200 refusals

The retained local Stage-1/2 benchmark artifacts use a separate three-model
committee. The priorities 6, 2, and 3 are normalized by their total of 11:

| Model | Priority | Normalized weight |
|---|---:|---:|
| Qwen 3.5 397B-A17B (`local/qwen3.5-397b-a17b`) | 6 | 0.5455 (6/11) |
| DeepSeek R1 Distill 32B (`local/deepseek-r1-distill-32b`) | 2 | 0.1818 (2/11) |
| Mistral Small 4 (`local/mistral-small-4`) | 3 | 0.2727 (3/11) |
| **Total** | **11** | **1.0000** |

This committee produced the retained 800-row non-refusal and 200-row refusal
local Stage-1/2 artifacts. It is an internal benchmark-construction committee;
neither artifact is identical to the current 736-example benchmark holdout or
its 128 current refusal examples.

### Four-model local validation committee

The retained 49-record local validation workflow uses the following separate
four-model configuration:

| Model | Priority | Normalized weight |
|---|---:|---:|
| Qwen 3.5 397B-A17B | 4 | 0.500 |
| DeepSeek R1 Distill 32B | 2 | 0.250 |
| Gemma 4 31B (`local/gemma-4-31b`) | 1 | 0.125 |
| Mistral Small 4 | 1 | 0.125 |
| **Total** | **8** | **1.000** |

This is a retained validation/serving workflow, not an evaluation score or a
definition of the released datasets.

### Current default method configuration

The active default OpenRouter configuration retains the same weight pattern as
the historical four-model artifact but now uses Claude Haiku 4.5 at the
highest-weight slot:

| Model | Weight |
|---|---:|
| Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`) | 0.35 |
| GPT-5.4 (`openai/gpt-5.4`) | 0.30 |
| DeepSeek V3.2 (`deepseek/deepseek-v3.2`) | 0.20 |
| Mistral Small 2603 (`mistralai/mistral-small-2603`) | 0.15 |
| **Total** | **1.00** |

This table describes the live implementation's default, not retrospectively the
models that generated the historical training artifact. A configurable local
committee can supersede these defaults for a specific run.

### Core decision principle

At every stage, all judges see the same structured input and independently
return a structured output. The committee votes only on the central categorical
or boolean decision for that stage. It then adopts the associated natural-
language explanation from the highest-weight model that supported the winning
decision. The method therefore combines **committee agreement on decisions**
with **one coherent reasoning bundle**, never an averaged or stitched answer.

## Record contract and stage flow

Each input row represents one query and contains `id`, `query`,
`retrieved_docs`, a `conflict_type`, optional `gold_answer`, and related
provenance fields. A retrieved document provides at least `doc_id`, `snippet`,
`source_url`, and `timestamp`; benchmark preparation also preserves `title`,
`url`, and `date` where available.

```text
query + retrieved_docs
        │
        ▼
Stage 1: document-level evidence adjudication
        │  adds per_doc_notes
        ▼
Stage 2: set-level conflict reasoning and answerability
        │  adds/replaces conflict_reason and answerable_under_evidence
        ▼
Stage 3: grounded response synthesis
           adds expected_response and think
```

The stages are deliberately not a single end-to-end prompt. Each preserves the
input fields from the preceding stage and appends or updates only the fields it
is responsible for. This makes the evidence interpretation, conflict judgment,
and response behavior separately inspectable.

### Input adapters and exact field lineage

Source-specific input records are normalized into the common stagewise contract
before annotation. This is part of the methodology because it defines which
source fields are visible to the judges.

| Input family | Method-level transformation |
|---|---|
| Raw CONFLICTS-style records | Maps `question` to `query` and `correct_answer` to `gold_answer`; merges title/snippet/short text, normalizes dates, deduplicates documents by URL-plus-normalized text, and assigns stable record/document IDs. |
| Raw non-refusal benchmark records | Maps `url`/`date` to `source_url`/`timestamp`, normalizes the label spelling, and preserves source label/reason as provenance. |
| Curated refusal benchmark records | Normalizes document fields, retains refusal provenance, and marks the record as a refusal-required benchmark item. |
| Historical base training pool | Retains all three stage outputs for the 658-record committee artifact. It is an annotation lineage artifact, not the canonical 862/81 release. |

At Stage 1 the runner adds `per_doc_notes` without replacing the query or
retrieved documents. At Stage 2 it overwrites `conflict_reason` and
`answerable_under_evidence`; in benchmark/refusal modes it also saves the
incoming `conflict_type` to `_gold_conflict_type` before writing the voted
label. At Stage 3 it overwrites/adds only `expected_response` and `think` plus
the abstention audit fields. The benchmark preparation fields beginning with
an underscore remain provenance aids and are not generation targets.

### Stage 1 — per-document evidence adjudication

Stage 1 submits every `(query, document)` pair to each committee member. The
prompt requires one JSON note per document:

| Field | Contract |
|---|---|
| `doc_id` | Identifier of the retrieved document being judged. |
| `verdict` | `supports`, `partially supports`, or `irrelevant`. |
| `key_fact` | A one-sentence paraphrase anchored to the selected quote; blank only for `irrelevant`. |
| `quote` | A contiguous verbatim snippet span of at most 50 words; blank only for `irrelevant`. |
| `verdict_reason` | Concise evidence-grounded justification. |
| `source_quality` | Coarse `high`/`low` URL-derived credibility signal. |

The ordinary Stage-1 prompt is intentionally strict about using only the given
snippet, quote–fact coupling, and source-quality rules. The benchmark-specific
prompt changes one important decision boundary: older, contradictory, or
otherwise conflict-bearing evidence about the same target is usually retained
as `partially supports` rather than discarded as `irrelevant`. This lets Stage
2 see the evidence needed to diagnose temporal and scientific conflicts.

For each document, the merged note includes `_vote_tally`, `_winner_model`, and
`_all_verdicts`. If a winner note has missing or malformed supporting fields,
the merge procedure supplies conservative default text, clears
`key_fact`/`quote` for `irrelevant`, and derives `source_quality` from the URL
when necessary.

The standard prompt has several deliberately conservative decision rules that
are easy to lose in a high-level description: an on-topic but incomplete,
hedged, subset-specific, or inconclusive snippet is `partially supports`; a
threshold query requires a decisive bound or categorical claim for `supports`;
and a date/current/next query requires the requisite explicit temporal detail.
Source quality is a coarse URL-only rule, not an independent fact-check: `.gov`
and `.edu` domains plus a defined list of institutional, journal, reference,
and major-news hosts are `high`; every other or missing host is `low`.

### Stage 2 — conflict reasoning and answerability

Stage 2 consumes the Stage-1 record. Its behavior depends on an explicit mode;
the modes are not interchangeable.

| Invocation mode | Prompt family | Is `conflict_type` voted? | Expected use |
|---|---|---:|---|
| Default / conflicts | `system_stage2.txt`, `user_stage2.txt` | No | A supplied conflict label is preserved; the committee votes answerability and writes an evidence-grounded reason. |
| `--benchmark-mode` | `*_stage2_benchmark.txt` | Yes | Committee independently classifies the evidence set and answerability; input label is retained as `_gold_conflict_type`. |
| `--refusal-mode` | `*_stage2_refusal.txt` | Yes | Refusal-required examples; the prompt directs `answerable_under_evidence=false` while the committee still classifies evidence conflict type. Input label is retained as `_gold_conflict_type`. |

The five normalized conflict labels are `No conflict`, `Complementary
information`, `Conflicting opinions or research outcomes`, `Conflict due to
outdated information`, and `Conflict due to misinformation`. The parser also
accepts the historical spelling `Conflicting opinions and research outcomes`
for compatibility.

The Stage-2 reason must refer to document IDs and articulate a mechanism
(temporal, factual, contextual, methodological, or linguistic). The normal
conflicts prompt asks the model to preserve the supplied label exactly; the
benchmark/refusal prompts make the label an output. The runner records
`_ans_vote_tally` and `_ans_winner_model`; label-voting modes also record
`_ct_vote_tally` and `_ct_winner_model`.

The benchmark and refusal Stage-2 user prompts apply an explicit label
decision order: direct contradiction, temporal supersession, demonstrably
misleading claim relative to stronger in-set evidence, compatible distinct
facets, then aligned evidence. This is prompt guidance rather than a
hard-coded classifier. The normal conflicts prompt instead receives the label
as a fixed target and is evaluated on explaining the supplied evidence pattern
without reinterpreting the label.

### Stage 3 — grounded expected response

Stage 3 receives the query, documents, Stage-1 notes, Stage-2 reasoning,
answerability, and gold answer when present. It adds:

```json
{
  "expected_response": {
    "answer": "...",
    "evidence": ["d1", "d3"],
    "abstain": false,
    "abstain_reason": null
  },
  "think": "<think>...</think>"
}
```

The normal prompt requires an evidence-grounded, cited response when the set
is answerable. If it abstains, the answer string must be exactly `CANNOT
ANSWER, INSUFFICIENT EVIDENCE`, with an empty evidence list and an
evidence-grounded reason. The refusal prompt directs those abstention fields
for every refusal-required input.

Stage 3 independently votes `expected_response.abstain`; it is not a
programmatic copy of Stage 2's `answerable_under_evidence`. This is a useful
diagnostic separation, but it means a complete analysis must examine both
fields. The retained 658-record artifact, for example, has 201 Stage-2
non-answerable decisions and 202 Stage-3 abstentions.

For a non-abstaining response, the normal prompt requests 2–3 cited sentences
(3–4 for simple, fully consistent fact retrieval), high-credibility evidence
first, and inclusion of all non-irrelevant document IDs in the evidence array.
It also asks the `think` trace to account for every retrieved document. These
are model instructions, not assertions proven by the structural validator; a
paper should report any separate citation- or trace-quality audit rather than
assuming prompt compliance.

## Weighted committee decision logic

Once the committee for a run has been fixed, the same merge rule is applied
independently at every stage. The model/weight tables above specify the
available recorded configurations; they are not interchangeable across runs.

For a decision field with candidate values (v), every successfully represented
judge vote contributes its configured weight. The system selects

\[
v^* = \arg\max_v \sum_{m \in M} w_m\,\mathbf{1}[y_m=v].
\]

Exact equal tallies are resolved deterministically by the implementation's
lexicographic value ordering, rather than at random. Once (v^*) is selected,
the supporting text is adopted intact from the highest-weight model that voted
for (v^*). Thus the committee does not average prose or splice explanations
from different judges.

| Stage | Voted decision | Text adopted from the winning side |
|---|---|---|
| 1 | Document `verdict` | `key_fact`, `quote`, `verdict_reason`, `source_quality`. |
| 2, conflicts mode | `answerable_under_evidence` | `conflict_reason` from the answerability winner; input label remains untouched. |
| 2, benchmark/refusal modes | `answerable_under_evidence` and `conflict_type` independently | `conflict_reason` from the conflict-type winner. |
| 3 | `expected_response.abstain` | Entire `expected_response` object and `think` trace. |

This policy gives categorical decisions committee support while retaining a
single internally coherent quote/reason/answer bundle. It should not be
described as an ensemble that blends natural-language explanations.

### Audit fields and how to read them

The output JSONL exposes the decision trace at the level actually voted by the
committee. The numeric tally values are normalized committee weight, rounded
to four decimal places, not counts of judges.

| Field | Written at | Meaning |
|---|---|---|
| `_vote_tally` | Each Stage-1 note | Weight assigned to each document verdict. |
| `_all_verdicts` | Each Stage-1 note | Per-model verdict values before merge; a useful disagreement trace. |
| `_winner_model` | Each Stage-1 note | Highest-weight judge on the chosen verdict side; its note bundle supplies the prose. |
| `_ans_vote_tally` / `_ans_winner_model` | Stage-2 row | Weighted answerability decision and winner-side judge. |
| `_ct_vote_tally` / `_ct_winner_model` | Benchmark/refusal Stage-2 row | Independent weighted conflict-type decision and judge supplying `conflict_reason`. |
| `_abstain_vote_tally` / `_abstain_winner_model` | Stage-3 row | Weighted final abstention decision and judge supplying the complete final-response bundle. |
| `_validation_errors`, `_stage3_errors`, `_parse_error`, `_error` | Any stage, when applicable | Parser/schema diagnostics or call-level error context. These fields must be included in quality auditing. |
| `_all_models_failed` | Any merged stage, when applicable | Explicit conservative all-failure fallback, not a valid consensus. |

For example, a three-judge local tally of `{"partially supports": 0.5455,
"irrelevant": 0.4545}` means the former won by normalized priority, even if
two lower-priority judges produced the latter. It is therefore incorrect to
infer one-model-one-vote majority from a tally. Conversely, a one-value tally
such as `{"supports": 1.0}` indicates a unanimous decision among the records
available to that merge.

## Reproduction infrastructure (secondary to the methodology)

The runners default to OpenRouter and require `OPENROUTER_API_KEY`. The
`local_openai` backend instead targets OpenAI-compatible local servers (for
example, vLLM or SGLang) using JSON configurations. A local judge configuration
contains a model ID, endpoint (or endpoint environment-variable override), a
positive priority, timeout, and optional request body. Priorities are
normalized to weights before a run.

For local serving, judges can be collected one at a time into a shared response
cache and then aggregated in read-only mode. A cache miss during aggregation is
a hard error, preventing an accidental mixture of old cached and fresh live
responses. The repository's configuration guide contains exact server settings
and commands; the model identities and weights that matter scientifically are
given at the start of this document.

The cache key hashes the cache format version, provider, model, system prompt,
user prompt, maximum tokens, temperature, and request options. It deliberately
does not include a transient local compute-node URL. Cache entries are useful
for exact replay on the same prompt/model setup, but are excluded from version
control; an archive intended for byte-level replay must preserve them securely
outside the Git repository.

The local collector uses `response_format={"type":"json_object"}` for every
checked-in judge configuration. Qwen configurations additionally request
`chat_template_kwargs={"enable_thinking": false}`. Nevertheless, the client
can preserve local-server reasoning content when a server exposes it: it
normalizes a separate reasoning field into a leading `<think>...</think>` block
before Stage-3 parsing. Endpoint environment variables such as
`LOCAL_QWEN_BASE_URL` override the fallback localhost URL, allowing a Slurm
allocation to advertise its transient node address without changing the cache
identity.

## Operational reproduction notes

All stage runners use temperature 0.0 by default and asynchronous global
semaphores. Default maximum output budgets are 512 tokens in Stage 1, 400 in
Stage 2, and 6000 in Stage 3. Default total concurrent requests are 25, 20,
and 15 respectively; a local configuration can reduce/override that limit.
Transient API failures are retried with exponential backoff, while 400, 401,
403, and 404 responses fail immediately.

Each runner reads its existing output path before work and skips IDs already
present. This supports interrupted-run resumption, but it also means a changed
prompt, committee, or input must use a new output path (or an intentionally
managed empty output), not silently append to an old file.

For OpenRouter calls, the client records response generation IDs and fetches
exact USD cost from OpenRouter's generation endpoint. Each stage can write a
per-run JSON report, append-only JSONL ledger, and deduplicated cumulative
summary. Local OpenAI-compatible calls do not provide those OpenRouter
generation IDs, so this mechanism is not a local-serving cost estimator.

### Run controls and mode safeguards

All three runners require `--input` and `--output`. The shared controls are
`--temperature`, `--concurrency`, `--limit`, `--max-retries`,
`--system-prompt`, `--user-prompt`, `--cost-report`, `--cost-ledger`,
`--cumulative-cost-report`, `--use-cache`, `--committee-backend`,
`--committee-config`, `--cache-mode`, and `--cache-dir`.

| Stage | Required stage-specific control | Default behavior and safeguard |
|---|---|---|
| Stage 1 | No mode selection | Uses the ordinary evidence prompt unless its prompt family is explicitly changed. Every configured judge assesses every document. |
| Stage 2 | At most one of benchmark mode or refusal mode | Default preserves the input conflict label. Benchmark mode independently votes label and answerability; refusal mode uses refusal-specific instructions. The two modes cannot be combined. |
| Stage 3 | Optional refusal mode | Uses the refusal-specific response contract only for refusal-required records; normal Stage 3 consumes the preceding Stage-2 record. |
| Benchmark orchestration | Prepared input or an explicit prepare-only mode | Validates the benchmark schema before and after its retained annotation stages; local single-judge collection must be completed stage by stage before aggregation. |

Cache modes have distinct operational meanings:

| Mode | Read cache | Write cache | Appropriate use |
|---|---:|---:|---|
| `off` | No | No | Fresh live experiment without replay support. |
| `read_write` | Yes | Yes | Normal cache-backed collection; cache misses invoke the model. |
| `read_only` | Yes | No | Deterministic aggregation/replay; a miss terminates the run. |
| `write_only` | No | Yes | Force fresh calls while refreshing a cache namespace. |

`--use-cache` is a compatibility shorthand for `read_write` when an explicit
`--cache-mode` is absent. The cache namespace is stage-specific (`stage1_multi`,
`stage2_multi`, `stage2_multi_benchmark`, `stage2_multi_refusal`,
`stage3_multi`, or `stage3_multi_refusal`), preventing an identical prompt in
two semantically different stage modes from being reused accidentally.

Exact command-line examples and endpoint settings are intentionally kept in
[`configs/local_committee/README.md`](../configs/local_committee/README.md).
They are operational aids, whereas this document specifies the annotation
logic and the scientific provenance of the retained artifacts.

### Safe local collection sequence

For a multi-server local committee, do not send one request to every endpoint
from a single transient allocation unless all judges are actually live. The
retained design is deliberately two-phase:

1. Prepare one immutable input JSONL and record its ID order.
2. Run Stage 1 once per judge using a matching single-judge collection
   configuration and `read_write` cache mode.
3. Aggregate Stage 1 using the corresponding full read-only committee
   configuration and `read_only` cache mode.
4. Run Stage 2 collection from that full aggregated Stage-1 file, never from a
   single-judge Stage-1 output.
5. Aggregate Stage 2, then separate ordinary and refusal records where the
   Stage-3 prompt families differ.
6. Run Stage 3 with the correct normal or refusal prompt family and validate
   the resulting JSONL.

Aggregation preserves a fixed base-record order and joins other judges by
record ID. It is deterministic only when every per-model collection covers the
same record IDs and document IDs; that condition should be verified before a
costly Stage-2 run.

## Parsing, failure handling, and validation boundaries

The parsers accept common model formatting deviations: JSON code fences,
leading prose, balanced embedded JSON objects, trailing commas, literal control
characters inside strings, and a leading `<think>...</think>` block. They attach
validation diagnostics such as `_validation_errors` rather than silently
discarding the parsed content.

If a model call or parse fails, the runner emits a conservative structured
fallback that still participates in the vote: Stage 1 defaults to
`irrelevant`, Stage 2 defaults to non-answerable, and Stage 3 defaults to a
safe abstention. If every model fails, the merge writes an explicit
`_all_models_failed` record with the same conservative behavior. This prevents
malformed model text from crashing a long batch, but it is not a substitute for
reviewing `_error`, `_parse_error`, `_validation_errors`, and
`_all_models_failed` fields in a quality audit.

The bundled validators are intentionally structural:

- `validate_benchmark_gold.py` checks required fields, valid label inventories,
  document/note alignment, and boolean answerability.
- `validate_stage3.py` checks the abstention string, response field types, and
  presence of a trace for non-abstentions.

They do **not** prove quote contiguity, quote entailment, factual correctness,
temporal ordering, or that a Stage-3 answer semantically follows Stage 2.
Those are prompt-level objectives and require manual or separate programmatic
quality audits. The benchmark and training human-review pipelines are described
separately and should not be conflated with these structural checks.

## Retained runs and provenance limits

| Artifact | Scope and retained stages | Interpretation |
|---|---|---|
| `data/final_annotations/stagewise_multi/{stage1,stage2,stage3_final}.jsonl` | 658 rows, 5,204 retrieved documents; all three stages. | Historical base training annotation artifact, not the final 862/81 release by itself. The adjacent Stage-1/2/3 cost reports retain one recorded 190-record/950-document OpenRouter run (3,800/760/760 calls and $22.85199/$3.59977/$7.99391); they are not aggregate cost totals for all 658 rows. |
| `outputs/local_committee_benchmark800_3model_rerun1/final/` | 800 non-refusal records, 4,000 documents; Stage 1 and 2. | Three-judge local benchmark construction artifact. It precedes human quality selection and is not the final 736-record holdout. |
| `outputs/local_committee_refusals200_3model/final/` | 200 refusal records, 1,000 documents; Stage 1 and 2. | Historical internal refusal construction artifact; it is distinct from the current 128-record benchmark refusal release. |
| `outputs/local_committee_val49/final/stage3_final_combined.jsonl` | 49-record local validation experiment. | Retained validation output, not a release file and not a benchmark evaluation result. |

The historical 658-row cost reports name `anthropic/claude-sonnet-4.6`,
`openai/gpt-5.4`, `deepseek/deepseek-v3.2`, and
`mistralai/mistral-small-2603`, with 0.35/0.30/0.20/0.15 decision weights in
the recorded run context. The active default code now names Claude Haiku 4.5
at the 0.35 slot. Therefore, publication text must attribute historical
outputs to their recorded model metadata, and describe the code default
separately.

The release datasets are assembled and quality-controlled through additional
normalization, split construction, and human review steps. In particular,
neither the 658-row annotation artifact nor the 800/200 internal local
artifacts should be represented as identical to the final training or benchmark
releases. Consult the dataset cards before reporting final dataset statistics.

### Recorded committee-agreement diagnostics

The following counts are computed directly from the retained audit fields. They
are **committee-decision diagnostics**, not human inter-annotator agreement and
not final dataset quality metrics. A contested decision simply means that the
weight tally has more than one candidate value; it does not by itself indicate
an error.

| Artifact and decision | Unanimous | Contested | Interpretation |
|---|---:|---:|---|
| Historical 658-row Stage-1 document verdicts (5,204 notes) | 2,548 (48.96%) | 2,656 (51.04%) | Document relevance is the most granular and correspondingly most disputed committee decision in this artifact. |
| Historical 658-row Stage-2 answerability | 588 (89.36%) | 70 (10.64%) | Set-level answerability was usually unanimous under the recorded committee. |
| Historical refusal-subset Stage-2 conflict type (200 rows) | 88 (44.00%) | 112 (56.00%) | Conflict taxonomy was independently voted only for the refusal-mode rows; the normal conflicts-mode rows preserved their supplied label. |
| Historical 658-row Stage-3 abstention | 205 (31.16%) | 453 (68.84%) | Final response behavior remained a deliberately independent, frequently contested decision. |
| Local 800 non-refusal Stage-1 document verdicts (4,000 notes) | 2,401 (60.02%) | 1,599 (39.98%) | Three-judge local benchmark construction artifact. |
| Local 800 non-refusal Stage-2 answerability | 738 (92.25%) | 62 (7.75%) | No conflict-type vote is present because these rows used normal conflicts mode. |
| Local 200 refusal Stage-1 document verdicts (1,000 notes) | 692 (69.20%) | 308 (30.80%) | Three-judge historical refusal construction artifact. |
| Local 200 refusal Stage-2 answerability | 200 (100.00%) | 0 | Consistent with the refusal-mode task contract. |
| Local 200 refusal Stage-2 conflict type | 114 (57.00%) | 86 (43.00%) | Independent conflict-type vote; this does not define the current 128-row benchmark refusal release. |

None of the four inspected artifacts contain a record-level
`_all_models_failed=true` flag. This is reassuring as an operational check, but
it does not establish semantic correctness: a committee can unanimously produce
an unsupported quote or a poor conflict explanation. The human-review reports
and targeted audits remain the appropriate evidence for those claims.

The 49-row local validation output is intentionally not used in the table as a
quality headline. It is a retained experiment with 15 unanimous and 34
contested Stage-3 abstention votes, and all 49 final responses abstain. Its
purpose is validating the local workflow, not establishing performance of the
released benchmark.

## Reproducibility checklist

1. Start from the relevant canonical input and record its checksum and IDs.
2. Pin the repository commit, prompt files, runner arguments, backend, and
   committee configuration; do not rely on a model nickname alone.
3. Use a fresh output directory for any changed condition; verify resume IDs
   before allowing a resumed job.
4. Preserve the raw-response cache if exact replay is required; otherwise call
   the result a methodological rerun, not a byte-identical reproduction.
5. Retain the stage cost reports/ledgers for OpenRouter runs and inspect parse
   or API-error audit fields.
6. Run the appropriate structural validator and then perform targeted semantic
   inspection of quotations, citations, temporal claims, and abstentions.
7. Report the exact historical committee from the artifact when discussing a
   completed dataset build, and the current code committee only when describing
   the live implementation.

## Related documentation

- [`TRAINING_DATASET_DESCRIPTION.md`](TRAINING_DATASET_DESCRIPTION.md): source
  composition, final training release, and human-review provenance.
- [`BENCHMARK_DATASET_DESCRIPTION.md`](BENCHMARK_DATASET_DESCRIPTION.md):
  benchmark source/retrieval path, human preselection, refusals, and releases.
- [`TAVILY_RETRIEVAL_METHODOLOGY.md`](TAVILY_RETRIEVAL_METHODOLOGY.md):
  non-refusal benchmark evidence retrieval.
- [`MULTI_LLM_COMMITTEE_LOGIC.md`](../MULTI_LLM_COMMITTEE_LOGIC.md): concise
  conceptual summary of the merge rule.
