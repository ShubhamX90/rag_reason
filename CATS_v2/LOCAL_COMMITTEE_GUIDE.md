# Local Judge Committee Guide

> **Current production specification:** The paper-facing local benchmark uses the
> three-judge committee in `configs/benchmark_local_openai_3judge_qwen397.yaml`
> and the staged equivalents under `configs/local_staged/benchmark_local_stage_*`.
> Its judges are Qwen3.5-397B-A17B, Mistral Small 4, and
> DeepSeek-R1-Distill-32B. The active judge tasks are Behavior Adherence,
> committee Factual Grounding v2, and Single-Truth Recall. The root-level
> [`prompts/`](prompts/) folder contains the latest paper-facing copies of those
> prompts. Older committee lanes and legacy scoring paths are not part of the
> current 108-experiment master scope.

This repo can run the LLM judge committee against locally hosted
OpenAI-compatible model servers. The intended servers are vLLM, SGLang,
LMDeploy, or any adapter that exposes:

```text
POST /v1/chat/completions
```

No OpenRouter key is needed for this mode.

## CATS Configuration

Use one of the local configs:

```bash
python run_evaluation.py \
  --input data/splits/92p5_7p5/stagewise_multi/val/stage3_final.jsonl \
  --config configs/val_tier2_local_openai.yaml \
  --committee local
```

The ideal config is:

```text
configs/val_tier2_local_openai.yaml
```

It expects:

```text
local/qwen3.5-397b-a17b        http://127.0.0.1:8001/v1  priority 4
local/deepseek-r1-distill-32b  http://127.0.0.1:8002/v1  priority 2
local/gemma-4-31b              http://127.0.0.1:8003/v1  priority 1
local/mistral-small-4          http://127.0.0.1:8004/v1  priority 1
```

The 2x H200 fallback config is:

```text
configs/val_tier2_local_openai_2xh200_fallback.yaml
```

It replaces the 4x H200 Qwen anchor with `local/qwen3.5-122b`. This is a
continuity fallback, not a same-quality replacement.

## Benchmark 3-Judge Variant

For the 736-sample benchmark workflow in this repo, there is now a separate
3-judge local committee that removes Gemma and uses:

```text
local/qwen3.5-397b-a17b        priority 6
local/mistral-small-4          priority 3
local/deepseek-r1-distill-32b  priority 2
```

Primary benchmark config:

```text
configs/benchmark_local_openai_3judge_qwen397.yaml
```

Benchmark staged configs:

```text
configs/local_staged/benchmark_local_stage_qwen397_collect.yaml
configs/local_staged/benchmark_local_stage_mistral4_collect.yaml
configs/local_staged/benchmark_local_stage_deepseek32_collect.yaml
configs/local_staged/benchmark_local_stage_final_readonly.yaml
```

Benchmark prepared inputs live under:

```text
inputs/prepped_model_eval_inputs/benchmark_set_all_modes/.../input.jsonl
```

Benchmark committee outputs/cache live under:

```text
/scratch/$USER/rag-reason/cats_outputs/benchmark_local_committee_3judge/
```

Validated benchmark GPU placement at the moment:

```text
Qwen397      -> 2x H200
Mistral4     -> 2x H100
DeepSeek32   -> 1x A100
```

Important:
- the current `mistral_small4_a100_tp2_server.sbatch` path is intentionally not
  a default benchmark lane anymore.
- on a real July 4, 2026 probe run, that A100 path answered `/v1/models` but
  then crashed on `/v1/chat/completions` with
  `RuntimeError: unsupported 'a' scalar_type`.
- for clean benchmark launches, always treat a tiny successful
  `/v1/chat/completions` JSON probe as mandatory, not just server startup.

Benchmark controller job placement in the current curated repo is intentionally
simple:

```text
partition=compute
qos=cpulimit
```

That behavior is implemented by `scripts/select_controller_partition.sh`.

For dynamic benchmark orchestration, use:

```text
scripts/submit_benchmark_file_pipeline_dynamic.sh
```

This watcher-style launcher polls Slurm, submits the three collect jobs only
when prior stage dependencies are satisfied, with health-gating support for the
judge endpoints.

You can still override the controller partition manually at submit time with:

```bash
export PRIMARY_CONTROLLER_PARTITION=compute
export FORCE_CONTROLLER_PARTITION=gpu_a100_8   # optional manual override
```

## Local Server Shape

The evaluator sends the same judge prompts used by the existing committee to
each configured endpoint:

```yaml
conflict_eval:
  committee:
    type: "local_openai"
    voting_strategy: "weighted_majority"
    max_concurrent_requests: 4
    timeout_seconds: 900
    response_cache_dir: "outputs/local_committee_cache/val_tier2_4xh200"
    cache_mode: "read_write"
    judges:
      - model_id: "local/qwen3.5-397b-a17b"
        base_url: "http://127.0.0.1:8001/v1"
        priority: 4
        max_tokens: 1600
        request_timeout: 900
```

`model_id` must match the model name served by your local backend, or a name
accepted by that backend. Local calls are marked unmetered in cost summaries,
but token usage is still tracked when the server returns `usage`.

## Staged Committee Runs

If Sharanga cannot host every judge at the same time, keep
`response_cache_dir` fixed and use the response cache.

Recommended staged flow:

1. Start exactly one local judge server.
2. Run the matching single-model collection config with `cache_mode:
   "read_write"`.
3. Stop that server and repeat for the next local judge.
4. After all intended judges have populated the shared cache, run the final
   read-only aggregation config.

The staged 2x H200 fallback configs are:

```text
configs/local_staged/val_local_stage_qwen122_collect.yaml
configs/local_staged/val_local_stage_deepseek32_collect.yaml
configs/local_staged/val_local_stage_gemma31_collect.yaml
configs/local_staged/val_local_stage_mistral4_collect.yaml
configs/local_staged/val_local_stage_final_readonly.yaml
```

All five configs use the same cache:

```text
outputs/local_committee_cache/val_tier2_2xh200_fallback
```

The final score is only comparable to the intended committee when all judge
responses are present in the cache. Intermediate staged outputs are only cache
collection runs.

The 2x H200 fallback final aggregation uses weighted majority:

```text
local/qwen3.5-122b               priority 3
local/deepseek-r1-distill-32b    priority 2
local/gemma-4-31b                priority 1
local/mistral-small-4            priority 1
```

This distribution is defensible as a fallback because Qwen is the anchor but
cannot decide alone: Qwen plus any smaller judge reaches 4/7 priority mass, and
DeepSeek plus Gemma plus Mistral also reaches 4/7 and can overrule Qwen.

## Sharanga Setup Plan

Sync this repo to Sharanga under the requested project directory:

```bash
rsync -av \
  --exclude .git \
  --exclude venv \
  --exclude outputs \
  --exclude logs \
  /Users/shubhammishra/Desktop/rag_reason/CATS_v2/ \
  sharanga:~/rag-reason/CATS_v2/
```

On Sharanga, first inspect GPU availability and QoS:

```bash
sinfo -o "%P %G %D %t %m %c"
scontrol show partition
sacctmgr show qos format=Name,Priority,MaxTRESPU,MaxTRESPerUser,MaxJobsPU,MaxSubmitJobsPU
squeue -u "$USER"
```

Questions to answer before launching servers:

```text
1. Can the account get 4 concurrent H200 GPUs for Qwen3.5-397B-A17B?
2. If not, can it get 2 concurrent H200 GPUs for Qwen3.5-122B?
3. Can Gemma 4 31B and Mistral Small 4 run concurrently on H100/A100 partitions?
4. Is there a spare H200/H100/A100 allocation for DeepSeek-R1-Distill-32B Q4?
5. Can the CATS evaluator run on the same node as the servers, or does it need
   node-hostname base URLs or SSH port forwarding?
```

## Example Server Commands

Adjust paths to Sharanga's scratch model locations.

Qwen3.5-397B-A17B, ideal anchor:

```bash
vllm serve "$SCRATCH/models/Qwen3.5-397B-A17B" \
  --served-model-name local/qwen3.5-397b-a17b \
  --tensor-parallel-size 4 \
  --dtype auto \
  --quantization fp8 \
  --host 0.0.0.0 \
  --port 8001
```

Qwen3.5-122B, 2x H200 fallback anchor:

```bash
vllm serve "$SCRATCH/models/Qwen3.5-122B" \
  --served-model-name local/qwen3.5-122b \
  --tensor-parallel-size 2 \
  --dtype auto \
  --quantization fp8 \
  --host 0.0.0.0 \
  --port 8001
```

DeepSeek-R1-Distill-32B Q4:

```bash
vllm serve "$SCRATCH/models/DeepSeek-R1-Distill-32B" \
  --served-model-name local/deepseek-r1-distill-32b \
  --dtype auto \
  --quantization bitsandbytes \
  --host 0.0.0.0 \
  --port 8002
```

Gemma 4 31B:

```bash
vllm serve "$SCRATCH/models/Gemma-4-31B" \
  --served-model-name local/gemma-4-31b \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8003
```

Mistral Small 4:

```bash
vllm serve "$SCRATCH/models/Mistral-Small-4" \
  --served-model-name local/mistral-small-4 \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8004
```

These commands are templates. Confirm the actual model paths, supported
quantization flags, and vLLM/SGLang version on Sharanga before submitting long
jobs.

## 1. Authoritative Scope and Version Lock

This section is the paper-facing specification for the local committee. It is
more authoritative than historical job notes or older result directories.

### 1.1 Production committee identity

The current benchmark committee is:

| Rank/order | Served model ID | Role | Priority |
| ---: | --- | --- | ---: |
| 1 | local/qwen3.5-397b-a17b | High-capacity anchor judge | 6 |
| 2 | local/mistral-small-4 | Independent local judge | 3 |
| 3 | local/deepseek-r1-distill-32b | Independent local judge | 2 |

Gemma is not in the current benchmark committee. Qwen, Mistral, and DeepSeek
are the three models combined for current benchmark results. Their priorities
are part of the method and must not be changed during reproduction without a
new, explicitly versioned result set.

The active judge tasks are:

1. Behavior Adherence (BA), one conflict-type-specific policy judgment per applicable answer.
2. Committee Factual Grounding v2, one claim-level evidence-linkage judgment per extracted claim.
3. Single-Truth Recall (STR), one semantic assertion judgment per applicable gold-answer item.

Grounded Refusal (GR) is deterministic. It is computed from benchmark
answerability labels and the model's answer/refusal form, not by asking the
local committee to make another GR judgment.

The current prompt bundle is:

- prompts/behavior_adherence_prompt.template.txt
- prompts/behavior_rubric.md
- prompts/factual_grounding_prompt.template.txt
- prompts/single_truth_recall_prompt.template.txt
- prompts/committee_json_system_prompt.txt

The executable prompt generators remain in rag_eval/judge_prompts.py. The
files under prompts/ are faithful parameterized copies for inspection and paper
writing; the Python generator performs runtime substitution and document
formatting.

### 1.2 Benchmark and master-results scope

The standard local committee benchmark contains 736 examples across five conflict
types. The paper-facing hierarchical master matrix contains exactly 108
experiment rows:

- 96 standard benchmark rows;
- 6 answer-only SFT rows;
- 2 comparison rows for the available Llama technique results;
- 4 latest fixed comparison rows for Mistral/Qwen CoT few-shot and CoN.

The current guide primarily describes how these benchmark outputs are produced.
It does not redefine the 108-row scope. The authoritative master artifacts are
under:

~~~text
outputs/benchmark_local_committee_3judge/master_results/
~~~

The latest hierarchical source files are:

~~~text
cats_master_results_20260731_hierarchical.csv
cats_master_results_20260731_hierarchical.json
cats_master_results_20260731_hierarchical.md
cats_master_results_20260731_hierarchical_audit.json
cats_master_results_20260731_hierarchical_audit.md
~~~

The workbook presentation artifact is:

~~~text
outputs/master_results_20260731_hierarchical.xlsx
~~~

The source of truth for a single experiment is its run-local
final/detailed_results.json, especially the per_sample list and stored
committee details. The workbook and master CSV are derived presentation
products, not substitutes for per-sample evidence.

### 1.3 What is and is not comparable

The following are not interchangeable:

- A three-judge Qwen/Mistral/DeepSeek result is not directly the same protocol as a four-judge validation committee.
- A staged cache-collection output is not a final committee evaluation.
- A final read-only aggregation is comparable to an all-at-once run only when it uses the same input file, prompt generator, judge identities, priorities, and complete cache.
- A fallback Qwen122/Gemma/Mistral/DeepSeek committee is a continuity baseline, not the same committee as the current benchmark committee.
- Historical NLI-based grounding code is not the active committee FG-v2 path.
- The older flat CATS average is not final_cats_prevalence or final_cats_balanced.

Every paper table must identify the committee lane, benchmark slice, prompt/config
version, and whether the run was all-at-once or staged.

## 2. System Architecture

### 2.1 Component graph

~~~text
prepared model-output JSONL
        |
        v
run_evaluation.py + YAML overlay
        |
        v
EnhancedEvaluator
        |
        +--> normalize gold answerability
        +--> strip think trace / detect answer vs refusal
        +--> GR decision signal
        |
        +--> BA prompt ------------------+
        +--> FG claim extraction/prompt -+--> local OpenAI-compatible judges
        +--> STR prompt -----------------+          |
                                                   v
                                        JudgeResponse / FG response parsing
                                                   |
                                                   v
                                          committee aggregation and audit fields
                                                   |
                                                   v
                                  per_sample + summary + eval_report.md
                                                   |
                                                   v
                                    master audit / workbook aggregation
~~~

The evaluated model and judge models are separate roles. The local committee
does not regenerate the evaluated model answer. It receives the already
generated answer plus the query, conflict context, and relevant evidence fields,
then emits structured judgments.

### 2.2 Runtime layers

There are four runtime layers:

1. Model-output preparation converts model generations and annotations into prepared JSONL.
2. Evaluation orchestration loads YAML, constructs the committee, evaluates examples asynchronously, and writes results.
3. Judge serving exposes OpenAI-compatible chat completions for each local model.
4. Slurm orchestration manages GPU allocation, health gates, cache collectors, and final read-only merges.

The evaluation process can run all at once when all three judge servers are
available. The staged process separates model inference from committee voting by
persisting each judge response in a shared response cache.

## 3. Evaluation Data Contract

### 3.1 Prepared input record

Each prepared JSONL record conceptually contains:

| Field/group | Purpose |
| --- | --- |
| Query/question | User task presented to the evaluated model and judge prompts. |
| Model final answer | Candidate response to score; may include a think trace that is stripped before judging. |
| Retrieved documents | Document IDs, snippets, dates, sources, and retrieval context. |
| Per-document notes | Gold verdict, key fact, quote, and support/relevance annotations. |
| Conflict category | One of the five conflict regimes. |
| Expected response / answerability | Gold answer-versus-refusal target. |
| Gold answer(s) | Target answer items for STR-applicable examples. |
| Sample ID | Stable identity used in outputs, caches, and audits. |

Historical field names vary by preparation stage. The evaluator's normalization
functions are the executable schema boundary. Do not manually rename fields in a
final result file without preserving the source record and rerunning evaluation.

### 3.2 Gold answerability precedence

The evaluator resolves answerability in this order:

1. expected_response.abstain, when an explicit boolean is present. Gold answerability is its negation.
2. answerable_under_evidence, when present.
3. Historical fallback from annotated support IDs or verdicts.

An explicit abstention label wins over an inference from document verdicts. This
prevents a partially supportive document from automatically making a complex
question answerable.

### 3.3 Document eligibility for active FG

For committee FG-v2, documents are eligible only when their normalized gold
verdict is one of:

~~~text
supports
support
partially supports
partial support
partially_supports
~~~

Irrelevant or otherwise non-positive documents are not passed to the FG prompt.
The prompt includes each eligible document's ID, gold verdict, key fact, and a
passage assembled from quote/snippet fields. The gold verdict is relevance
supervision; the committee checks whether the document conveys the specific
model claim.

### 3.4 Model-answer preprocessing

Before local judging:

1. Obtain the model's final output.
2. Strip think traces using repository answer normalization.
3. Detect answer versus refusal using the normalized final answer.
4. Use the same normalized answer for BA, FG extraction, and STR.

For reasoning models such as DeepSeek, a visible think block must not become a
judged answer claim, citation, or refusal decision.

## 4. Exact Metric Call Graph

### 4.1 Per-example sequence

For each sample, the active evaluator follows this order:

1. Resolve gold_answerable.
2. Obtain and normalize the model answer.
3. Compute pred_answered using the refusal detector.
4. Compute gr_accuracy = 1[pred_answered == gold_answerable].
5. Identify a correct required refusal as not gold_answerable and not pred_answered.
6. For a correct required refusal, exclude BA, FG, STR, and Answer Quality from applicability; CATS receives decision-only credit.
7. Otherwise, extract up to the configured max_claims_per_answer claims with citations.
8. Call the BA committee once for the example.
9. Call committee FG once for every extracted claim.
10. Call STR once for each normalized gold answer when a gold answer exists and the type is in (1, 2, 4, 5).
11. Serialize component scores, applicability flags, vote details, claims, and errors.
12. Aggregate example values into type and overall summaries.

The benchmark YAML sets max_claims_per_answer to 8. The dataclass default is 5;
the YAML overlay is therefore part of the benchmark protocol and must be
preserved when reproducing benchmark results.

### 4.2 Correct refusal semantics

The active evaluator treats a correct required refusal as:

~~~text
gr_accuracy = 1
behavior_applicable = false
factual_grounding_applicable = false
single_truth_applicable = false
answer_quality = not computable
cats_example_score = gr_accuracy
~~~

This is intentional: a required refusal has no answer claims whose grounding or
conflict behavior can be evaluated. The YAML key correct_refusal_full_credit
documents this policy, but the evaluator branch is the executable authority and
implements the rule directly.

### 4.3 Wrong refusals and wrong answers

Outputs that are not correct required refusals go through the judgment path. A
wrong refusal on an answerable item can produce zero extracted claims and
therefore zero FG, while its GR gate is zero. An answer on a refusal-required
item is also GR-incorrect; any downstream judgment is diagnostic and cannot
rescue its CATS score because g_i=0 gates the example score to zero.

## 5. Prompt and Transport Specification

### 5.1 Task prompt roles

Behavior Adherence supplies the query, model answer, one conflict type, exactly
one rubric, and optional Type 4/5 date/source provenance. It asks for adherent,
a short rationale, and confidence.

Single-Truth Recall supplies one gold answer item and the full model answer. It
asks whether the model asserts that gold answer as its own conclusion, not
whether it merely quotes or attributes it.

Committee Factual Grounding supplies the query, bounded model-answer context,
one extracted claim, eligible annotated documents, and valid document IDs. It
asks for individually supporting IDs and, only when no single document supports
the claim, a two-document cross-support combination.

The exact paper-facing copies are in prompts/. The active source functions are
behavior_judge_prompt, single_truth_recall_prompt, and fg_committee_prompt in
rag_eval/judge_prompts.py.

### 5.2 System message and JSON contract

For local OpenAI-compatible, DeepSeek, and OpenRouter chat-completion calls, the
client sends:

~~~text
You are an evaluation judge. Respond ONLY with JSON.
~~~

The task prompt repeats the JSON-only requirement. The local config also sends:

~~~json
{"response_format": {"type": "json_object"}}
~~~

Expected Behavior/STR response:

~~~json
{
  "adherent": true,
  "rationale": "short explanation",
  "confidence": 0.0
}
~~~

Expected FG response:

~~~json
{
  "supporting_docs": ["d2"],
  "cross_doc_support": false,
  "cross_doc_combo": []
}
~~~

The parser tolerates markdown fences and extracts the first balanced JSON object.
Behavior parse failures are surfaced as error and excluded from valid committee
voting. FG parsing filters returned IDs to the valid document ID set and has a
limited recovery path for mildly malformed key-value output.

### 5.3 Thinking-model handling

Qwen's benchmark config passes:

~~~yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
~~~

This is intended to make Qwen emit the short JSON judgment directly. DeepSeek
may return a reasoning block before JSON; the client removes content before the
first closing think boundary when present. The evaluator also strips think traces
from model answers before judging. These are two separate protections:

- server request configuration controls judge-output format;
- evaluator normalization controls the answer being evaluated.

### 5.4 Local endpoint contract

Each server must expose:

~~~text
POST http://<host>:<port>/v1/chat/completions
GET  http://<host>:<port>/v1/models
~~~

The YAML base_url must end at /v1; the client appends the endpoint path. model_id
must match the server's accepted model name, especially the served-model-name
value used by vLLM.

## 6. Committee Mathematics

### 6.1 Behavior and STR vote objects

For each valid judge j, let:

~~~text
v_j = 1 if the judge returns adherent=true, otherwise 0
c_j = parsed confidence clipped to [0, 1]
p_j = configured integer priority
~~~

The effective weighted vote is:

~~~text
w_j = p_j * max(c_j, 0.01)
~~~

The 0.01 floor prevents a valid zero-confidence response from having exactly
zero influence. A timeout, API error, or parse failure is not valid and is
removed before this calculation.

Define:

~~~text
W_plus  = sum(w_j for v_j = 1)
W_minus = sum(w_j for v_j = 0)
~~~

The binary committee decision is:

~~~text
b_i_binary = 1[W_plus > W_minus]
~~~

The strict inequality means a weighted tie is non-adherent. The committee
confidence fields are:

~~~text
majority_confidence = max(W_plus, W_minus) / (W_plus + W_minus)
minority_confidence = min(W_plus, W_minus) / (W_plus + W_minus)
~~~

The continuous support used by CATS is:

~~~text
b_i = W_plus / (W_plus + W_minus)
~~~

For the full benchmark committee, raw priorities sum to 11. BA and STR use
confidence-scaled weights, so effective totals change per judgment. Priorities
must not be interpreted as probabilities or empirical judge-accuracy estimates;
they are configured influence weights.

### 6.2 Binary Behavior Adherence report

The workbook BA component is the applicable-example mean:

~~~text
BA = sum(b_i_binary over applicable examples) / behavior_n
~~~

Correct required refusals are excluded from behavior_n. Source artifacts also
retain behavior_consensus_score, raw votes, weighted totals, individual judge
responses, rationales, latency, cost, and valid/error status. Binary BA and
continuous CATS support are deliberately different quantities.

### 6.3 Factual Grounding vote aggregation

FG does not use judge confidence in document support votes. For each claim, let
J be the valid FG judges and let:

~~~text
priority_mass = sum(max(1, p_j) for j in J)
threshold = priority_mass / 2
~~~

If more than one judge is valid, at least two valid judges must name a document
or cross-document support. If only one judge is valid, one vote is sufficient:

~~~text
minimum_positive_judges = 2 if len(J) > 1 else 1
~~~

For document d, with priority mass P_d and naming count V_d:

~~~text
d is supporting iff P_d > threshold and V_d >= minimum_positive_judges
~~~

For cross-document support, if P_cross is the priority mass of judges setting
cross_doc_support=true and V_cross is their count:

~~~text
cross_support = 1[P_cross > threshold and V_cross >= minimum_positive_judges]
~~~

With all three production judges valid, Qwen alone has priority 6 and crosses
the raw threshold 5.5 but fails the two-judge corroboration requirement; Qwen
plus either Mistral or DeepSeek can pass. Mistral plus DeepSeek have priority 5
and cannot pass the raw threshold when all three judges are valid.

### 6.4 Committee failure semantics

For BA/STR, valid responses have error equal to None. If all judges fail, the
fallback stores all_failed=true, zero confidence, and no positive support. For
FG, if no judge response is valid, support sets are empty and the claim is not
grounded. Failure counts and per-claim reasons remain in output and must be
reported during audit.

## 7. Metric-Specific Semantics

### 7.1 Behavior Adherence

The exact active rubric is:

| Type | Required behavior |
| ---: | --- |
| 1 | Direct, clear answer; do not invent alternatives or uncertainty. |
| 2 | Reconcile complementary partial answers into one coherent response. |
| 3 | Represent conflicting opinions neutrally rather than collapsing disagreement. |
| 4 | Prioritize up-to-date information and optionally acknowledge superseded information. |
| 5 | Reject misinformation and rely on reliable, verified evidence. |

BA is orthogonal to answer/refusal correctness, factual entailment, citation
validity, unsupported-claim detection, and gold-answer recovery. The judge
assesses conflict-policy behavior, not generic writing quality. For Types 4 and 5,
retrieved date/source metadata is added when available so prioritization can be
judged against evidence rather than wording alone.

### 7.2 Factual Grounding

The deterministic claim extractor runs before FG. It protects internal periods,
extracts bracketed/parenthetical/bare document references, strips citation text,
drops citation-only and meta-reference fragments, filters very short claims,
inherits citations for an eligible concise lead claim when the next sentence is
cited, applies a terse-answer fallback, and caps claims at 8 for the benchmark
YAML.

For claim k, let C_k be model-cited documents, S_k committee-supported documents,
and X_k the accepted cross-document combination. The claim is grounded iff:

~~~text
y_k = 1[(C_k intersects S_k is nonempty)
     OR (cross_support AND C_k intersects X_k is nonempty)]
~~~

The example score is:

~~~text
FG_i = sum(y_k) / number_of_extracted_claims_i
~~~

An example with no extracted claims receives FG_i=0 when FG is applicable. The
dataset FG is an example-macro mean, not a pooled claim micro-average. Detailed
output retains supporting IDs, cross-document fields, cited IDs, reason, and
committee errors for every claim.

### 7.3 Single-Truth Recall

STR applies only when a gold answer exists and the conflict type is in:

~~~text
(1, 2, 4, 5)
~~~

Type 3 is excluded because it does not necessarily have one canonical truth to
assert. The prompt distinguishes assertion from mention, quotation, or
attribution. Paraphrases and equivalent formulations can count as matches.

For each gold answer item, a semantic committee acceptance yields an exact match.
If the committee rejects it but the positive side has minority confidence at
least 0.30, it is a qualified partial match. If any exact match exists, example
STR is 1. Otherwise:

~~~text
STR_i = min(1, 0.5 * partial_matches / gold_answer_count)
~~~

If there are no exact or qualified partial matches, STR is zero. Dataset STR is
the applicable-example macro-average.

### 7.4 Grounded Refusal

Let A_i be gold answerability and predicted A_i be the parser's answer decision:

~~~text
g_i = 1[predicted_A_i == A_i]
~~~

The dataset reports answer-positive and refusal-positive precision, recall, and
F1, plus accuracy. The local committee does not vote on this metric. The refusal
parser recognizes canonical inability/insufficient-evidence openings, empty
output, and wrapped refusal forms after think-trace removal. It is start-oriented
to avoid classifying a substantive answer as a refusal merely because a later
clause mentions insufficient evidence.

## 8. Applicability and CATS Interaction

### 8.1 Applicability counts

Every run must report:

~~~text
behavior_n
fg_n
str_n
answer_quality_n
~~~

These are counts of examples with the corresponding applicability flag, not
counts of successful judgments. A score of zero with a positive denominator is
different from an unavailable score with a zero denominator.

### 8.2 Answer Quality

For example FG score f_i and, where applicable, STR score r_i:

~~~text
q_i = sqrt(f_i * r_i)  if both apply
q_i = f_i               if only FG applies
q_i = unavailable       if FG is unavailable
~~~

Answer Quality is calculated per example before dataset averaging. It is not the
geometric mean of already averaged FG and STR columns.

### 8.3 Hierarchical CATS

For a correct required refusal:

~~~text
s_i = g_i
~~~

For other examples, with continuous BA consensus b_i and Answer Quality q_i:

~~~text
s_i = g_i * 2*b_i*q_i/(b_i+q_i)  if b_i and q_i are available
s_i = g_i * b_i                  if only b_i is available
s_i = g_i * q_i                  if only q_i is available
s_i = g_i                        if neither is available
~~~

Because g_i is a gate, an incorrect answer/refusal decision cannot be rescued
by a favorable downstream judge result. CATS-Prevalence is the arithmetic mean
of complete example scores over the empirical benchmark distribution.
CATS-Balanced averages complete conflict-type scores after balancing answerable
and refusal-required subgroups where both exist. Both are secondary summaries;
primary claims should use GR, BA, FG, and STR with denominators.

## 9. Cache and Staged Evaluation Protocol

### 9.1 Cache key and layout

The committee cache is configured with one shared response_cache_dir, for example:

~~~text
outputs/benchmark_local_committee_3judge/response_cache/
~~~

The client stores prompt responses by:

~~~text
<cache_dir>/<mode>/<sanitized_model_id>/<sha256(prompt)>.json
~~~

The mode is behavior or fg. Each cache payload contains mode, model ID,
provider, prompt SHA-256, write timestamp, and the response payload. Changing
any rendered prompt input creates a distinct cache key.

### 9.2 Cache modes

| Mode | Behavior |
| --- | --- |
| off | No cache reads or writes. |
| read_write | Read existing entries and write successful new responses. |
| read_only | Read existing entries; misses become explicit errors. |
| write_only | Do not read; write successful responses. |

Staged collectors use read_write with one judge configured. The final merge uses
read_only with all three judges configured. This prevents the final pass from
silently contacting a server or producing a mixed partial run.

### 9.3 Required staged sequence

1. Use the exact prepared input JSONL and final cache directory.
2. Start Qwen397 and run configs/local_staged/benchmark_local_stage_qwen397_collect.yaml.
3. Start Mistral Small 4 and run configs/local_staged/benchmark_local_stage_mistral4_collect.yaml.
4. Start DeepSeek32 and run configs/local_staged/benchmark_local_stage_deepseek32_collect.yaml.
5. Verify cache coverage by model, mode, and prompt count.
6. Run configs/local_staged/benchmark_local_stage_final_readonly.yaml.
7. Audit the final detailed_results.json and confirm all expected judges were valid.

The stage output directories named judge1_collect, judge2_collect, and
judge3_collect are collection artifacts. They are not final results to cite in
a paper. The final read-only output is the authoritative staged evaluation.

### 9.4 Cache integrity checks

Before final aggregation, verify:

- all three model subdirectories are present;
- behavior entries exist for every BA-applicable sample;
- FG entries exist for every extracted claim and judge;
- prompt hashes were generated from the same prompt source and input;
- no cache was mixed between benchmark variants or committee configurations;
- final read-only logs contain no cache misses or all-judge failures.

Never copy a cache between prompt versions merely because model IDs match.
Prompt wording and input context are part of the cache condition.

## 10. GPU Serving and Health Validation

### 10.1 Validated benchmark placement

The curated placement is:

~~~text
Qwen3.5-397B-A17B  -> 2x H200
Mistral Small 4    -> 2x H100
DeepSeek32         -> 1x A100
~~~

This reflects observed serving behavior, not only theoretical memory fit. The
Mistral A100 route has previously answered /v1/models but failed on a real chat
completion with an FP8/Marlin scalar-type error. A server being alive is not
sufficient evidence that a benchmark lane is valid.

### 10.2 Mandatory endpoint probe

After every server launch and before collection:

~~~bash
source /scratch/pabitra/rag-reason/envs/local-judge-serving/bin/activate
python slurm/sharanga/local_committee/probe_openai_endpoint.py \
  --base-url http://<host>:<port>/v1 \
  --model <served-model-id> \
  --timeout 180
~~~

The probe must exercise /v1/chat/completions, not only /v1/models. Confirm HTTP
success, accepted model ID, nonempty response, parseable JSON, no unexpected
think-only output, and acceptable latency.

### 10.3 Server-specific constraints

Qwen397/Qwen122 must use validated tensor-parallel, quantization, and backend
settings from the matching Slurm script. Qwen122 currently requires disabled
DeepGEMM, the validated backend flags, and enable_thinking=false.

DeepSeek32 can take approximately 25--30 minutes to load from scratch. Do not
classify a long load as failure without checking Slurm state and logs.

Gemma may require the repository chat template. Gemma is not part of the current
three-judge benchmark committee.

Mistral Small 4 uses the repository chat template and text-only settings. H100
or H200 is preferred over the previously failing A100 route unless revalidated.

### 10.4 Network topology

The evaluator must reach judge hostnames from the node where the evaluation job
runs. If evaluation runs on a different node, use server-advertised hostnames
or approved forwarding. Do not use 127.0.0.1 in a Slurm job unless evaluator
and server are on the same node.

## 11. Slurm and Orchestration Protocol

### 11.1 Preflight

Before long jobs:

~~~bash
sinfo -o "%P %G %D %t %m %c"
scontrol show partition
sacctmgr show qos format=Name,Priority,MaxTRESPU,MaxTRESPerUser,MaxJobsPU,MaxSubmitJobsPU
squeue -u "$USER"
~~~

Confirm GPU type, capacity, CPU quota, QoS limits, network access, and controller
placement. The observed CPU QoS limit can prevent multiple eight-CPU H100 jobs
from running concurrently even when GPUs appear available.

### 11.2 Health-gated dynamic pipeline

The watcher-oriented path is:

~~~text
scripts/submit_benchmark_file_pipeline_dynamic.sh
scripts/watch_benchmark_file_pipeline.py
slurm/sharanga/local_committee/benchmark_endpoint_health_gate.sbatch
slurm/sharanga/local_committee/benchmark_collect_eval.sbatch
slurm/sharanga/local_committee/benchmark_final_merge.sbatch
~~~

The controller should submit collection jobs only after endpoint health checks
pass, then submit final merge only after required collection jobs succeed. A
Slurm job ID is not proof that an endpoint is usable.

### 11.3 Controller placement

The curated controller defaults to partition=compute and qos=cpulimit. Use
scripts/select_controller_partition.sh for current selection logic. Manual
overrides should be recorded:

~~~bash
export PRIMARY_CONTROLLER_PARTITION=compute
export FORCE_CONTROLLER_PARTITION=gpu_a100_8
~~~

### 11.4 One-file pipeline invariants

For each prepared input file, keep one stable:

- input JSONL path;
- output run directory;
- shared response-cache directory;
- committee YAML/configuration;
- source model-output provenance;
- final detailed_results.json.

Do not reuse an output or cache directory across different input files or prompt
versions. Prompt hashes prevent some accidental reuse, but directory identity is
still necessary for human auditability.

## 12. Output and Provenance Schema

### 12.1 Run-local files

A completed run should contain:

~~~text
<run_dir>/
  detailed_results.json
  eval_report.md
  run_config.yaml
  logs/                    # if configured
~~~

run_config.yaml is part of the scientific result. It records input/output paths,
committee type, model IDs, endpoints, priorities, token budgets, timeout, cache
mode, and metric settings. If a final result lacks a run config, numerical use
may still be possible, but reproducibility provenance is incomplete and must be
disclosed or reconstructed from contemporaneous records.

### 12.2 Per-sample fields

The active evaluator serializes fields including:

~~~text
sample_id
conflict_type
pred_answered
gold_answerable
correct_refusal
gr_accuracy
behavior_score
behavior_applicable
behavior_consensus_score
behavior_details
factual_grounding_score
factual_grounding_applicable
factual_grounding_details
single_truth_recall_score
single_truth_recall_details
single_truth_applicable
~~~

Detailed fields also contain normalized claims, cited IDs, supporting IDs,
cross-document combinations, individual judge responses, weighted vote totals,
confidence, rationale, latency, cost, and error strings when available.

### 12.3 Summary fields

The summary retains:

- GR answer/refusal precision, recall, F1, and accuracy;
- BA, FG, STR, and Answer Quality with applicability counts;
- cats_aggregate_version;
- CATS-Prevalence and CATS-Balanced;
- per-type summaries;
- correct-refusal counts;
- CATS completeness and unscorable counts;
- legacy flat CATS only for historical comparison.

The master workbook maps these to columns J:AA as documented in
CATS_METRICS_METHODOLOGY.md. Always preserve component counts when exporting.

## 13. Reproducibility Checklist

### 13.1 Before a run

- Confirm the input file and expected row count.
- Confirm the committee is the intended three-judge production committee.
- Confirm priorities 6/3/2 and weighted-majority voting.
- Confirm model IDs match server served names.
- Confirm prompt source and prompts/ bundle are unchanged.
- Confirm max_claims_per_answer=8 for benchmark config.
- Confirm cache mode and cache directory.
- Confirm all endpoints pass a chat-completion probe.
- Confirm Slurm GPU, CPU, QoS, and network constraints.

### 13.2 During a run

- Monitor each server log and endpoint health.
- Monitor collection job exit status, not only submission status.
- Watch for timeout, cache miss, parse error, and all-judge-failed messages.
- Keep each judge cache collection isolated by model and shared only through the intended cache root.
- Do not alter prompts, priorities, or serving settings mid-run.

### 13.3 After a run

- Confirm n equals expected benchmark sample count.
- Confirm all three judge identities appear in final committee details.
- Confirm no unexpected cache misses or failed judges.
- Confirm behavior_n, fg_n, str_n, and answer_quality_n.
- Confirm per-type counts and all five conflict types.
- Confirm cats_complete=true and cats_unscorable_n=0 for a publishable run.
- Preserve final config, source input, report, detailed results, and cache provenance.
- Run independent master and workbook audits before citing results.

## 14. Audit Commands

From the repository root:

~~~bash
python3 -m unittest discover -s tests -q
python3 -m py_compile \
  rag_eval/judge_prompts.py \
  rag_eval/judge_committee.py \
  rag_eval/conflict_eval.py \
  rag_eval/evaluator.py
python3 scripts/audit_cats_master_results.py
python3 scripts/audit_master_results_excel.py
~~~

The current authoritative source audit should report:

~~~text
source rows: 108
complete rows: 108
unscorable examples: 0
CSV mismatches: 0
JSON mismatches: 0
Markdown mismatches: 0
~~~

The older verify_master_gr_metrics.py historically enumerated a 114-file
universe. If it reports six missing rows, reconcile that output against four
unfixed comparison finals and two staged artifacts deliberately outside current
108-row scope. Do not interpret that legacy warning as evidence that current
master has 114 experiments or that a current row is missing.

## 15. Failure Modes and Recovery

### 15.1 Server is up but completion fails

Re-run the real chat probe. Inspect quantization, tensor parallelism, model
template, and backend compatibility. /v1/models alone is insufficient.

### 15.2 Cache miss in final read-only pass

Do not switch the final pass to read_write merely to make it complete. Identify
which model, mode, or prompt hash is missing, rerun that model's collection stage
with the exact same input and cache root, then rerun final read-only.

### 15.3 One judge times out

The committee excludes the failed response from that vote. This can alter BA
weighted-majority results and FG corroboration thresholds. Report valid-judge
counts and do not describe the sample as a full three-judge decision.

### 15.4 Malformed JSON

Behavior parsing records an error and excludes the response. FG has a bounded
recovery parser for simple key-value responses, but recovered responses must be
visible in output/logs. Do not manually edit parsed results.

### 15.5 Unexpected thinking output

For Qwen, verify enable_thinking=false. For DeepSeek, verify client removal of
the think block before JSON parsing. If the evaluated model emits a think trace,
verify answer normalization before rerunning.

### 15.6 Partial staged cache

Collection runs are not final scores. A final read-only merge must use all
intended judge responses. If one judge is absent, the result is a reduced-
committee diagnostic and must not enter the main comparable matrix.

## 16. ACL-Level Methods Description

The following is a faithful starting point for a paper methods section:

> We evaluate generated RAG responses with a locally hosted committee of three
> OpenAI-compatible judge models: Qwen3.5-397B-A17B, Mistral Small 4, and
> DeepSeek-R1-Distill-32B. The committee is used for conflict-policy behavior,
> citation-linked factual grounding, and single-truth answer recovery; grounded
> refusal is computed deterministically from benchmark answerability labels and
> the model's answer/refusal form. For Behavior Adherence and Single-Truth
> Recall, each valid judge emits a binary decision, rationale, and confidence.
> We aggregate these votes with priority- and confidence-weighted majority:
> w_j = priority_j * max(confidence_j, 0.01), and select the positive decision
> only when positive weighted mass strictly exceeds negative mass. For the
> secondary CATS aggregate, we retain the positive weighted support fraction
> rather than collapsing committee disagreement to a binary value. For Factual
> Grounding, judges identify eligible documents supporting each extracted claim;
> document support is accepted only when it exceeds half of valid judges' raw
> priority mass and satisfies the corroboration requirement. A claim counts as
> grounded only when the model cites an identified supporting document, either
> individually or through an accepted cross-document combination. The committee
> runs through local OpenAI-compatible endpoints with fixed prompts, model
> identities, priorities, response-format constraints, and per-judge caching.
> Staged cache collection and final read-only aggregation are used when models
> cannot be served concurrently. We report committee composition, applicability
> counts, valid-judge counts, and component metrics before any secondary
> aggregate.

### 16.1 Required accompanying details

An ACL submission should also provide:

- the five BA rubrics or a pointer to prompts/behavior_rubric.md;
- complete BA, FG, and STR prompts in an appendix or artifact repository;
- model IDs, checkpoints, hardware, quantization, and serving framework versions;
- priorities and vote aggregation equations;
- cache/staged-run policy and treatment of missing/failed judges;
- benchmark size, conflict-type distribution, and metric applicability counts;
- human-versus-committee agreement analysis where available;
- confidence intervals or a pre-specified uncertainty procedure;
- a statement that committee judgments are evaluator outputs, not human gold truth.

## 17. Scientific Defensibility and Limitations

### 17.1 Why a committee is used

Conflict behavior and semantic answer containment are judgment-sensitive. A
single judge can be unstable under wording, model-specific priors, or ambiguous
conflict cases. Multiple judges provide evaluator redundancy and preserve a
measurable disagreement signal. This does not establish objective truth: the
committee remains an evaluation instrument whose validity should be checked
against human judgments and sensitivity analyses.

### 17.2 Why priorities are disclosed

Priority weighting gives a deterministic, auditable aggregation rule. It is not a
claim that Qwen is six times as accurate as Mistral or DeepSeek. The paper should
state how priorities were selected, avoid tuning them on reported test outcomes,
and include unweighted or leave-one-judge-out sensitivity where practical.

### 17.3 Why FG uses raw priority and corroboration

FG responses do not expose per-document confidence in their JSON schema. Using
raw priority avoids inventing a confidence scale that the FG prompt does not
produce. Requiring two valid judges when multiple judges are available prevents
one high-priority judge from unilaterally certifying a document, while allowing a
transparent one-judge fallback when others fail.

### 17.4 What the committee does not prove

The local committee does not prove:

- universal factual correctness outside provided evidence;
- retrieval completeness;
- human-level validity without agreement evidence;
- that a model answer is optimal under every conflict interpretation;
- that CATS is a universal utility function;
- that a missing or failed judge is equivalent to a negative judgment.

These boundaries belong in the paper limitations section rather than being hidden
behind the aggregate score.

## 18. Change-Control Rules

Treat any of the following as a new evaluation version:

- changing any prompt wording or rubric;
- changing committee model IDs, checkpoints, serving templates, or quantization;
- changing priority values or voting strategy;
- changing max_claims_per_answer, answer normalization, or refusal parsing;
- changing cache namespace or reusing a cache under a different prompt/config;
- changing FG eligibility, corroboration, or citation-linkage rules;
- changing STR applicability or partial-match threshold;
- changing timeout, retry, or failure-exclusion policy.

When one changes, preserve old outputs under the repository legacy/provenance
structure, create a versioned new output directory, rerun audits, and update the
prompt bundle, this guide, metric methodology, and aggregate logic documentation
together. Never silently overwrite a final result while retaining the old run
directory name.

## 19. Final Reproduction Checklist

Before calling a local-committee result final, a second researcher should be
able to answer yes to every item:

- Is the exact prepared input file identified?
- Is the evaluated model-output source identified?
- Are all three judge model IDs and priorities recorded?
- Are prompt files and executable prompt source identified?
- Are all endpoint probes successful for chat completions?
- Are hardware, serving framework, quantization, and chat-template settings recorded?
- Is the cache directory unique to this input/configuration?
- Is final aggregation read-only after complete staged collection?
- Are all expected judge responses present and parseable?
- Are timeout, error, and parse-failure counts zero or explicitly disclosed?
- Are BA, FG, STR, and GR denominators present?
- Are all five conflict types represented?
- Are CATS completeness and unscorable counts verified?
- Does the 108-row master audit distinguish current scope from legacy 114-file tools?
- Can every paper-facing number be traced to a stored per-sample record?

If any answer is no, the result is not yet ready to be presented as a fully
reproducible ACL evaluation.
