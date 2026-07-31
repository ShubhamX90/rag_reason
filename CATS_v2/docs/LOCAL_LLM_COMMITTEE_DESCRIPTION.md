# Local LLM Committee: Logical Description and Scientific Rationale

**Status:** Current paper-facing description of the production local committee
for the CATS v2 benchmark.

**Scope:** This document explains the committee as a scientific evaluation
instrument: why it exists, what models and tasks it contains, how judgments are
combined, how those judgments enter each CATS metric, what makes the design
reproducible, and what limitations must be disclosed in an ACL paper.

**Level of detail:** The discussion is intentionally methodological and logical.
It does not reproduce the evaluator's function-by-function implementation. For
deployment commands, configuration overlays, cache layouts, endpoint probing,
Slurm orchestration, and failure-recovery procedures, see
[`LOCAL_COMMITTEE_GUIDE.md`](LOCAL_COMMITTEE_GUIDE.md). For the complete metric
definitions and aggregate formulas, see
[`CATS_METRICS_METHODOLOGY.md`](CATS_METRICS_METHODOLOGY.md) and
[`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md).

## 1. Executive Summary

CATS v2 evaluates retrieval-augmented generation under evidence conflict. The
evaluation target is not only whether a model produces a plausible answer. It
also asks whether the model:

1. answers or refuses in accordance with what the retrieved evidence permits;
2. follows the response policy appropriate to the conflict regime;
3. links its factual claims to supporting retrieved evidence; and
4. recovers a target answer when the benchmark defines one stable truth.

The local LLM committee is the judge layer for the judgment-sensitive parts of
this evaluation. The committee receives an already generated model answer and
the relevant benchmark context. It does not generate the answer under study,
does not retrieve documents, and does not replace the benchmark annotations.
Its role is to apply structured rubrics consistently across a large collection
of model outputs while preserving the individual judgments needed for audit.

The current production local benchmark committee contains three locally hosted,
OpenAI-compatible judge models:

| Served judge identity | Committee role in the current protocol | Priority |
| --- | --- | ---: |
| `local/qwen3.5-397b-a17b` | High-capacity anchor judge | 6 |
| `local/mistral-small-4` | Independent local judge | 3 |
| `local/deepseek-r1-distill-32b` | Independent local judge | 2 |

The committee evaluates three judgment-sensitive tasks:

- **Behavior Adherence (BA):** whether the answer follows the policy required by
  the conflict type;
- **Committee Factual Grounding v2 (FG-v2):** whether each extracted answer
  claim is supported by eligible retrieved evidence and properly cited; and
- **Single-Truth Recall (STR):** whether the answer asserts the benchmark's
  target truth when a single truth target is defined.

Grounded Refusal (GR) is deliberately outside the committee. It is computed
deterministically from the benchmark's answerability annotation and the model's
answer-versus-refusal behavior. This separation prevents the committee from
being asked to judge a construct that has a direct gold decision label.

The committee is therefore best understood as a structured, multi-judge
measurement layer, not as an oracle and not as a substitute for human ground
truth. Its scientific value comes from explicit task boundaries, multiple
judges, preserved disagreement, a fixed protocol, reproducible local serving,
and direct comparison with human evaluation where available.

## 2. What Problem the Committee Solves

### 2.1 Why ordinary automatic metrics are insufficient

The CATS benchmark contains examples where retrieved documents can be:

- mutually consistent;
- complementary but incomplete;
- genuinely conflicting in their conclusions;
- temporally outdated relative to newer evidence; or
- factually misleading or unreliable.

The appropriate response depends on this evidence regime. A response that
confidently selects one side may be acceptable in a no-conflict example and
unacceptable in a conflicting-opinions example. A response that refuses may be
responsible when evidence is insufficient and unhelpful when the evidence is
sufficient. A response can also be factually plausible while failing to cite
the retrieved evidence that supports it.

These distinctions are not fully captured by exact-match answer accuracy,
lexical overlap, or a single pooled factuality score. They require judgments
about response policy, evidence linkage, and semantic assertion. The committee
provides a repeatable way to make those judgments at scale.

### 2.2 Why use multiple judges

A single generative judge can be sensitive to wording, prompt interpretation,
answer length, or idiosyncratic reasoning. A committee reduces the dependence
of a result on one model's individual interpretation. It also exposes
disagreement instead of hiding it inside one opaque score.

The committee does not guarantee independence. All three judges are language
models and may share broad training-data patterns, evaluation biases, or
failure modes. The defensible claim is therefore not that the committee creates
objective truth. The defensible claim is that it provides a more auditable and
potentially more stable evaluator than a single unexamined judge, subject to
human agreement and sensitivity checks.

### 2.3 Why local hosting matters

Local hosting is a methodological choice as well as an infrastructure choice.
It provides:

- **Protocol control:** model identifiers, serving templates, decoding settings,
  system instructions, and output formats can be fixed and recorded;
- **Reproducibility:** the same judge checkpoints can be served again without
  relying on changing external API behavior or model aliases;
- **Data governance:** benchmark queries, retrieved documents, and model answers
  can remain within the controlled evaluation environment;
- **Cost and throughput control:** a large benchmark can be evaluated with a
  known hardware allocation and parallel judge services;
- **Auditability:** individual responses, parsing outcomes, latencies, cache
  keys, and errors can be retained as artifacts; and
- **Staged execution:** each judge can be run separately and combined later in a
  read-only aggregation step without changing the mathematical protocol.

Local hosting does not by itself establish judge quality. It makes the judge
condition explicit and controllable, which is necessary for a reproducible
paper result.

## 3. Protocol Scope and Version Lock

### 3.1 Current production scope

The current local committee protocol is the one used for the standard
736-example benchmark and the paper-facing hierarchical master matrix. The
master matrix contains exactly 108 experiment rows, comprising:

- 96 standard benchmark rows;
- 6 answer-only SFT rows;
- 2 Llama comparison rows; and
- 4 latest fixed Mistral/Qwen comparison rows covering CoT few-shot and CoN.

The committee identity, priorities, prompts, task rubrics, claim cap, voting
rules, and refusal semantics are all part of this protocol. Changing any of
them creates a new evaluation condition and requires a new versioned result set.

### 3.2 Active committee versus historical lanes

The repository contains historical and auxiliary committee lanes, including
validation configurations and older multi-judge combinations. They are not
interchangeable with the current production benchmark committee.

In particular:

- the current benchmark lane is Qwen397/Mistral4/DeepSeek32 with priorities
  6/3/2;
- a validation or fallback lane with another judge composition is a different
  evaluator condition;
- a staged collection output is not itself a final committee result;
- a final read-only merge is comparable to an all-at-once run only when the
  inputs, prompts, judge identities, priorities, and complete response cache
  are identical; and
- historical NLI-based grounding code is not the active local committee FG-v2
  protocol.

The current paper-facing prompt copies are kept under [`prompts/`](../prompts/).
That bundle contains the BA prompt, BA rubric, FG-v2 prompt, STR prompt, and
shared JSON-only system instruction. It intentionally does not present a
historical NLI prompt as part of the active committee protocol.

### 3.3 Configuration as part of the scientific method

The active benchmark configuration fixes more than model names. It specifies:

- the three served model IDs and their priorities;
- the local OpenAI-compatible provider;
- weighted-majority voting for BA and STR;
- raw-priority support voting for FG;
- the maximum of eight extracted claims per answer;
- the timeout and concurrency policy;
- the shared response-cache location and mode;
- the thinking-output handling required by the served models; and
- the refusal, grounding, citation, and aggregate flags.

The claim cap is especially important. The evaluator's general dataclass
default is five, while the benchmark YAML sets `max_claims_per_answer` to
eight. Reproduction must use the benchmark configuration rather than silently
falling back to a default.

## 4. Committee Composition

### 4.1 Model identities

The active committee uses three different locally served model families:

1. `local/qwen3.5-397b-a17b`;
2. `local/mistral-small-4`; and
3. `local/deepseek-r1-distill-32b`.

The names identify the served judge condition. They should be reported exactly,
including the served-model identity, rather than being reduced to a vague
description such as "an ensemble of open models." Quantization, serving
template, context handling, and endpoint configuration can affect the judge
behavior and therefore belong to the evaluation provenance.

### 4.2 Functional roles

The current configuration gives the judges different operational roles:

- **Qwen397 is the high-capacity anchor.** Its priority of 6 makes it the
  strongest single influence in the current protocol, while the FG corroboration
  rule prevents it from unilaterally satisfying multi-judge support.
- **Mistral Small 4 is an independent local judge.** Its priority of 3 gives it
  substantial influence while requiring it to agree with another judge for an
  FG document-support decision.
- **DeepSeek-R1-Distill-32B is an additional independent local judge.** Its
  priority of 2 supplies a third perspective and can corroborate Qwen or Mistral
  in FG, but it cannot pass the full three-judge FG threshold alone or with
  Mistral when all three judges are valid.

These are protocol roles, not claims that one model is universally more
accurate. The repository does not establish that the priority values are
empirical estimates of judge accuracy. They are configured influence weights
that must be interpreted and reported as design choices.

### 4.3 Why three judges are useful for this benchmark

Three judges create a practical balance among:

- evaluator diversity;
- the need to observe disagreement;
- serving and storage cost;
- throughput over 736 examples and many experiment variants; and
- a decision rule that can require corroboration without requiring a very large
  panel.

The number three is not a universal optimum. It is the fixed panel size for the
current benchmark protocol. Claims about robustness should be supported by the
stored vote distribution, committee-internal agreement analysis, human
agreement, and sensitivity to alternate aggregation where those analyses are
available.

### 4.4 Why the committee excludes Gemma in the active benchmark lane

Earlier repository lanes contain Gemma-based committee configurations. Gemma is
not in the current production benchmark committee. This exclusion is a version
boundary, not evidence that Gemma is intrinsically unsuitable as a judge. A
Gemma-containing result must be labeled as a different committee condition and
must not be merged into the current 3-judge master results.

## 5. Priority Weighting: Meaning and Boundaries

### 5.1 Configured priorities

Let the configured priority of judge (j) be (p_j). The active priorities are:

\[
p_{Qwen}=6, \qquad p_{Mistral}=3, \qquad p_{DeepSeek}=2.
\]

The raw priority total is therefore:

\[
P_{total}=6+3+2=11.
\]

These numbers are not probabilities, confidence intervals, sample counts, or
measured judge accuracies. They determine relative influence inside the
pre-specified voting rules.

### 5.2 Why unequal priorities can be defensible

An equal vote assumes that every judge should have identical influence under
the chosen protocol. The current design instead represents a role hierarchy:
the high-capacity anchor has more influence, while the other two judges supply
independent corroboration and disagreement information.

This can be defensible when the authors transparently state that:

- the priority scheme was fixed as part of the evaluator design;
- it is not estimated from the test outputs or tuned to improve a preferred
  model's rank;
- it is not a direct claim of calibrated judge accuracy;
- all individual judgments remain available; and
- equal-priority and alternative-priority sensitivity analyses are reported or
  acknowledged as important robustness checks.

The current repository documents the priorities and their operational roles. It
does not contain a separate calibration experiment proving that 6/3/2 is the
statistically optimal weighting. The paper should not make that stronger claim.

### 5.3 Confidence-scaled voting for BA and STR

For BA and STR, each valid judge returns a binary decision and a confidence
value. Let:

- (v_j \in \{0,1\}) be the judge's positive decision;
- (c_j \in [0,1]) be the parsed confidence;
- (p_j) be the configured priority; and
- (w_j) be the effective vote weight.

The effective vote weight is:

\[
w_j=p_j\max(c_j,0.01).
\]

The confidence floor ensures that a valid response with reported zero
confidence is not silently assigned exactly zero influence. It does not rescue
timeouts, API failures, malformed outputs, or parse failures; invalid responses
are removed before weights are calculated.

The positive and negative weighted masses are:

\[
W_+=\sum_{j:v_j=1}w_j,
\qquad
W_-=\sum_{j:v_j=0}w_j.
\]

The binary committee decision is:

\[
b_i^{bin}=\mathbf{1}[W_+>W_-].
\]

A strict inequality is used. A weighted tie is non-adherent rather than being
silently resolved in favor of the positive side.

The continuous positive support retained for CATS is:

\[
b_i=\frac{W_+}{W_++W_-},
\]

when at least one valid judge contributes. The corresponding majority and
minority support fractions are:

\[
q_i^{majority}=\frac{\max(W_+,W_-)}{W_++W_-},
\qquad
q_i^{minority}=\frac{\min(W_+,W_-)}{W_++W_-}.
\]

The binary BA/STR result and continuous (b_i) are deliberately retained as
different quantities. The binary result is easy to interpret as adherence or
non-adherence. The continuous value preserves disagreement for the secondary
CATS aggregate instead of converting a split vote and a unanimous vote into
the same number.

### 5.4 Practical consequences of 6/3/2

With all three judges valid:

- Qwen alone has priority mass 6, which is greater than half of 11, but this
  does not make Qwen an automatic FG decision-maker because FG additionally
  requires corroboration from at least two valid judges;
- Qwen plus Mistral has priority mass 9 and can pass the raw-priority FG
  threshold;
- Qwen plus DeepSeek has priority mass 8 and can pass the raw-priority FG
  threshold; and
- Mistral plus DeepSeek have priority mass 5, which is below half of 11 and
  cannot pass the full-valid-judge FG threshold.

This asymmetry is intentional and must be disclosed. A reader should be able to
understand not only the model names but also which coalitions can satisfy each
decision rule.

## 6. What the Committee Judges

### 6.1 Common input boundary

The committee evaluates a fixed model answer against benchmark context. It does
not regenerate the answer or decide which documents should have been retrieved.
Conceptually, a judged record contains:

- the user query;
- the model's final answer after think-trace removal;
- the five-way conflict type;
- retrieved document IDs and passages;
- document dates and sources when relevant;
- gold document verdicts and key facts for grounding;
- the gold answer item for STR when applicable; and
- the stable sample identity used for audit and caching.

The same normalized final answer is used for answer/refusal detection, BA, FG
claim extraction, and STR. This prevents one metric from judging a hidden
reasoning trace while another judges only the visible answer.

### 6.2 Structured output contract

The local chat-completion judges are instructed to return JSON only. BA and STR
return a binary decision, a short rationale, and a confidence value. FG returns
supporting document IDs and, where needed, a cross-document support indication
and document combination.

The structured contract is important because it makes the judgment auditable:
the pipeline can distinguish a positive decision from a rationale, preserve
confidence separately, validate document IDs, and record malformed or failed
responses instead of treating arbitrary text as a valid score.

### 6.3 Thinking-output handling

The committee protocol has two distinct normalization boundaries:

1. judge-output control: serving/configuration settings encourage the local
   judges to return compact JSON, and reasoning blocks are removed where the
   client receives them; and
2. evaluated-answer control: any visible think trace in the model answer under
   evaluation is stripped before answer/refusal parsing and before the answer is
   sent to the judgment prompts.

This prevents hidden or intermediate reasoning text from being counted as a
claim, a citation, or a refusal.

## 7. Task 1: Behavior Adherence

### 7.1 Construct definition

Behavior Adherence asks whether the response follows the policy appropriate to
the evidence-conflict regime. It is not a generic style score and is not a
second factuality score.

The active five-type rubric is:

| Type | Conflict regime | Required response policy |
| ---: | --- | --- |
| 1 | No Conflict | Answer directly and clearly without inventing alternatives or unnecessary uncertainty. |
| 2 | Complementary Information | Reconcile compatible partial information into one coherent answer. |
| 3 | Conflicting Opinions or Research Outcomes | Represent disagreement neutrally rather than collapsing it into one uncontested conclusion. |
| 4 | Outdated Information | Prefer current evidence and, when useful, acknowledge superseded information. |
| 5 | Misinformation | Reject inaccurate or unreliable evidence and rely on reliable, verified information. |

The BA judge sees the query, answer, conflict type, selected rubric instruction,
and optional Type 4/5 date/source provenance. The provenance block makes
recency and reliability behavior judgeable from the evidence context rather than
from generic wording preferences.

### 7.2 Orthogonality requirements

BA should not independently reward or penalize:

- whether the model should have answered or refused;
- whether a factual claim is entailed by a document;
- whether a citation points to a supporting document;
- whether the gold target answer was recovered; or
- whether the answer contains a claim that is unsupported under FG.

Those properties are represented by GR, FG, and STR. Keeping BA policy-focused
reduces double counting in both component interpretation and CATS aggregation.

### 7.3 BA aggregation

For the applicable example set (I_{BA}), the reported binary component is:

\[
BA=\frac{1}{|I_{BA}|}\sum_{i\in I_{BA}}b_i^{bin}.
\]

Correct required refusals are not in (I_{BA}), because this protocol treats
their answer-content behavior as inapplicable. The output retains the applicable
count `behavior_n` and the continuous committee support separately.

### 7.4 Why BA is useful

BA captures a failure mode that ordinary answer correctness can miss. A model
may produce a factually plausible statement while mishandling disagreement,
presenting stale information as current, or accepting misinformation. BA makes
the response policy itself measurable and supports per-conflict-type analysis.

## 8. Task 2: Committee Factual Grounding v2

### 8.1 Construct definition

FG-v2 asks whether the model's claims are supported by the retrieved evidence
and whether the answer cites the supporting evidence. It is therefore a
claim-level evidence-linkage metric, not a generic external fact-checking
metric.

The benchmark supplies document annotations. Only documents with normalized
verdicts equivalent to `supports` or `partially supports` are eligible for the
active FG prompt. The committee then determines whether an eligible document
conveys the specific extracted claim.

This division of labor is important:

- benchmark annotation defines which retrieved documents are eligible evidence;
- deterministic preprocessing defines the claims and visible citations; and
- the committee judges whether the eligible document text supports the specific
  claim and whether a combination of documents is needed.

The committee is not asked to infer document relevance from an unrestricted
corpus, and it is not used as an NLI system in the active production path.

### 8.2 Claim extraction boundary

Before FG judging, the answer is converted into candidate claims. The current
benchmark protocol:

- protects periods inside initials, decimals, domains, and common abbreviations;
- recognizes bracketed, parenthetical, and bare document references;
- removes citation text from the claim content while retaining cited IDs;
- removes citation-only and meta-reference fragments;
- filters very short fragments;
- supports concise cited-answer fallback;
- can inherit citations for an eligible concise lead statement when the evidence
  follows immediately; and
- limits the evaluated answer to at most eight extracted claims.

These rules are part of the measurement instrument. Changing claim segmentation
or the claim cap changes FG's denominator and must be versioned.

### 8.3 Single-document and cross-document support

For a claim (k), the committee can identify:

- one or more documents that individually support the claim; or
- a pair of documents whose information must be combined to support the claim.

The claim is grounded only when the answer's cited document set links to the
committee-supported evidence:

\[
y_{ik}=1
\quad\text{iff}\quad
(C_{ik}\cap S_{ik}\ne\varnothing)
\;\text{or}\;
(X_{ik}=1\;\text{and}\;C_{ik}\cap D^{cross}_{ik}\ne\varnothing),
\]

where:

- (C_{ik}) is the set of documents cited for claim (k);
- (S_{ik}) is the committee-supported single-document set;
- (X_{ik}) indicates accepted cross-document support; and
- (D^{cross}_{ik}) is the accepted document combination.

The example-level FG score is:

\[
FG_i=\frac{\sum_k y_{ik}}{K_i},
\]

where (K_i) is the number of extracted claims. An answer with no extracted
claims receives FG=0 when FG is applicable; it is not silently removed from the
FG denominator. Correct required refusals are the separate inapplicable case.

The dataset metric is an example macro-average:

\[
FG=\frac{1}{n_{FG}}\sum_{i\in I_{FG}}FG_i.
\]

Long answers do not receive automatic extra weight simply because they contain
more claims.

### 8.4 FG voting rule

FG uses raw priority rather than confidence-scaled priority. For the valid judge
set (J), define:

\[
P_J=\sum_{j\in J}\max(1,p_j),
\qquad
T_J=\frac{P_J}{2}.
\]

For a candidate supporting document (d), let (P_d) be the priority mass of
judges naming (d), and (V_d) be the number of valid judges naming it. The
document is accepted as committee support iff:

\[
P_d>T_J
\quad\text{and}\quad
V_d\ge m_J,
\]

where:

\[
m_J=
\begin{cases}
2,& |J|>1,\\
1,& |J|=1.
\end{cases}
\]

Cross-document support uses the analogous priority-mass and corroboration rule.

With all three judges valid, (P_J=11), so the strict priority threshold is
5.5 and at least two valid judges must support the document or combination.

The active FG-v2 path does not:

- multiply document votes by self-reported judge confidence;
- award graded partial credit for a partially supporting document;
- add a separate contradiction penalty; or
- use the historical NLI prompt as the production decision mechanism.

These exclusions make the active rule simpler to audit, but they also define
its limitations and must be stated in the paper.

### 8.5 Why FG is useful

FG separates "the answer contains a plausible statement" from "the answer
connects its statement to the evidence it was supposed to use." This is central
to retrieval-augmented generation: a response can be externally true but fail
the evidence-bounded task if it does not ground its claims in the retrieved
documents.

## 9. Task 3: Single-Truth Recall

### 9.1 Construct definition

STR measures whether the model asserts a benchmark-provided target answer as
its own conclusion when a stable single truth is defined. It is not token
overlap, lexical recall, or citation recall.

STR applies when:

\[
\text{gold answer exists}
\quad\text{and}\quad
\text{conflict type}\in\{1,2,4,5\}.
\]

Type 3 is excluded because a conflicting-opinions example may not have one
canonical proposition that every responsible answer should assert.

The STR judge distinguishes asserting a target from:

- merely quoting a document;
- listing the target as one possibility;
- attributing the target to a source without adopting it; or
- explicitly rejecting the target.

Paraphrases and logically equivalent formulations may count as semantic matches.

### 9.2 Per-item and example aggregation

For each gold answer item, the committee returns a binary semantic acceptance
decision. If any gold item has an exact semantic match, the example receives:

\[
STR_i=1.
\]

If there is no exact match, a negative decision can qualify as a partial match
when the positive "gold target is present" side has minority support of at least
0.30. In that case:

\[
STR_i=\min\left(1,\frac{0.5P_i}{G_i}\right),
\]

where (P_i) is the number of qualified partial matches and (G_i) is the
number of gold answer items. If there is no exact or qualified partial match,
STR_i=0.

The dataset metric is:

\[
STR=\frac{1}{n_{STR}}\sum_{i\in I_{STR}}STR_i.
\]

The applicable count `str_n` is essential because Type 3 and examples without
gold answers are not in the STR denominator.

### 9.3 Why STR is useful

Grounding and behavior do not guarantee that the response recovers the desired
answer. A model might cite true evidence while omitting the target conclusion,
or discuss the evidence without making an answer. STR exposes this failure mode
while respecting the benchmark's boundary that some conflict regimes do not
have a single truth to assert.

## 10. Grounded Refusal Is Separate from the Committee

### 10.1 Deterministic GR decision

Let (A_i) denote gold answerability under the retrieved evidence and
(\widehat A_i) denote whether the model answered rather than refused. Then:

\[
g_i=\mathbf{1}[\widehat A_i=A_i].
\]

The gold answerability resolution gives precedence to an explicit expected
response abstention label, then to `answerable_under_evidence`, and finally to
the benchmark's historical support-verdict fallback. The model answer is
normalized after think-trace removal. Empty output is treated as refusal, and
the parser recognizes the repository's canonical refusal forms.

### 10.2 Why GR should not be committee-judged

The benchmark has a direct answerability target. Having the same LLM committee
judge whether the answer should have been a refusal would introduce an
unnecessary evaluator layer and could blur the distinction between benchmark
supervision and judge opinion.

GR therefore supplies the decision gate for CATS and is reported through answer
precision, answer recall, answer F1, refusal precision, refusal recall, refusal
F1, and accuracy. The committee's role begins after this deterministic decision
boundary for non-correct-refusal examples.

### 10.3 Correct refusal semantics

When the evidence requires refusal and the model refuses:

\[
g_i=1,
\quad
BA_i,FG_i,STR_i\text{ are inapplicable},
\quad
AQ_i\text{ is inapplicable}.
\]

The current CATS policy gives such an example decision-only score:

\[
s_i=g_i=1.
\]

This does not claim that the refusal wording, explanation, or absence of
unsupported content has been independently judged. It is a deliberate protocol
choice. A paper must state it plainly and should report refusal-required and
answerable subsets separately where possible.

## 11. From Committee Judgments to CATS

### 11.1 Primary metrics remain primary

The committee-derived components should be reported directly:

- BA with `behavior_n`;
- FG with `fg_n`;
- STR with `str_n`; and
- GR metrics over the complete benchmark decision set.

These component metrics are the primary scientific evidence. CATS is a
secondary structured summary and must not hide which dimension produced a
model's result.

### 11.2 Answer Quality

For an applicable example, Answer Quality combines FG and STR at the example
level:

\[
AQ_i=
\begin{cases}
\sqrt{FG_i\,STR_i},& FG_i\text{ and }STR_i\text{ apply},\\
FG_i,& FG_i\text{ applies but }STR_i\text{ does not},\\
\text{unavailable},& FG_i\text{ does not apply}.
\end{cases}
\]

The geometric mean is used here because grounding and target recovery are both
content requirements when both apply. A low value in one should not be fully
masked by a high value in the other.

### 11.3 Example-level CATS score

For a non-correct-refusal example, let (b_i) be continuous BA support and
(AQ_i) be Answer Quality. The current active aggregate uses GR as a gate and
harmonic fusion for the two downstream quality dimensions:

\[
s_i=
g_i\cdot H(b_i,AQ_i),
\]

where:

\[
H(x,y)=\frac{2xy}{x+y}
\]

when both inputs are positive and the score is zero when either input is zero.
If only one downstream input is available, the available input is used after
the GR gate. For a correct required refusal, the decision-only rule above
applies.

This ordering matters. The system first evaluates each example, then combines
its applicable dimensions, then aggregates across examples. It does not first
average BA, FG, STR, and GR over different denominators and then take a flat
mean.

### 11.4 Balanced and prevalence summaries

The prevalence summary is:

\[
CATS\text{-}Prevalence=\frac{1}{N}\sum_{i=1}^{N}s_i.
\]

It reflects the benchmark's observed mixture of conflict types and answerability
regimes.

The balanced summary first computes type-level example scores and balances
answerable versus refusal-required subgroups within each type when both exist.
It then gives equal weight to the five conflict types:

\[
CATS\text{-}Balanced=\frac{1}{5}\sum_{t=1}^{5}T_t,
\]

where (T_t) is the decision-balanced score for conflict type (t). The exact
subgroup rule and completeness conditions are specified in
[`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md).

Both summaries are secondary. The balanced form protects against one prevalent
conflict regime dominating the aggregate; the prevalence form reflects the
benchmark's actual composition. Neither should replace the component table.

## 12. Committee Failure and Missingness Semantics

### 12.1 Valid versus invalid judge responses

A judge response is valid only when it is received, parseable, and contains the
fields needed for the relevant task. Timeouts, API errors, malformed JSON, and
unrecoverable output are invalid responses. They must not be converted into a
semantic negative judgment.

This distinction prevents infrastructure failure from being misreported as
model behavior. It also means every result should preserve:

- the number of valid judges;
- the individual response or error status;
- the weighted or raw vote totals used;
- the final decision;
- the rationale and confidence when available; and
- the affected claim or example.

### 12.2 BA and STR all-failure case

If every judge fails for BA or STR, the committee cannot establish a valid
positive or negative semantic judgment. The result must record an all-failed or
equivalent status, zero usable support, and the failure details. The affected
metric denominator and CATS completeness state must be auditable.

### 12.3 FG all-failure case

If no judge response is valid for an FG claim, the claim has no committee support
and is not grounded for the active result. The claim-level output must retain
the committee-error reason. Aggregate audits should distinguish this from a
valid committee judgment that found no supporting document.

### 12.4 Applicability is not performance

An applicable metric with score zero is measured poor performance. An
inapplicable metric with count zero is a denominator condition. Examples:

- a refusal-required example can have `behavior_n=0` for that example because
  BA is inapplicable;
- an answer with no extracted claims can have `fg_n=1` and FG=0;
- a Type 3 example can have `str_n=0` because STR is not defined there; and
- a complete benchmark can still have different BA, FG, and STR denominators.

The paper should show these counts rather than presenting zeros without their
denominators.

## 13. All-at-Once and Staged Evaluation

### 13.1 All-at-once mode

When all three local judge services are available, the evaluator can send the
required prompts to the committee services during one evaluation run. The
result stores the per-judge responses and the final committee aggregation.

### 13.2 Staged collection mode

The staged protocol separates response collection from final aggregation:

1. each judge collects responses into a shared cache;
2. cache entries are tied to the task prompt and judge identity;
3. collection jobs may run independently or at different times;
4. a final read-only process loads the complete cache; and
5. the final process recomputes committee decisions without generating new
   responses.

This is useful when the three models require different GPU placements or when
the evaluator node cannot serve all judges simultaneously. It is scientifically
equivalent to all-at-once evaluation only when the effective protocol is the
same: same prepared input, answer normalization, prompt rendering, model IDs,
priorities, voting rules, and complete valid cache.

A partial collector output is not a final score. Missing judges must not be
silently treated as a complete committee.

### 13.3 Cache identity and contamination control

The response cache is part of the audit trail. A cached response must not be
reused across:

- different evaluated-model variants;
- different prompt wording or rubric versions;
- different committee membership or priorities;
- different claim extraction or claim-cap rules; or
- different benchmark records.

The current cache design uses model identity and a hash of the rendered prompt
to distinguish responses. A reproducible run should preserve the source
configuration, cache mode, cache root, and final read-only status.

## 14. Serving and Operational Preconditions

The logical serving contract is simple: each judge exposes an
OpenAI-compatible `/v1/chat/completions` endpoint and a model identity that
matches the configured served name. Operational details are in
`LOCAL_COMMITTEE_GUIDE.md`, but the scientific preconditions are:

- the endpoint has passed a real chat-completion probe, not merely a model-list
  probe;
- the returned model identity is the intended judge;
- JSON output is actually parseable under the active client;
- model-specific reasoning or think blocks are handled consistently;
- the evaluator can reach the endpoint from its execution node; and
- the hardware/serving configuration is recorded with the result.

For the current validated benchmark placement, the repository documents Qwen397
on two H200 GPUs, Mistral Small 4 on two H100 GPUs, and DeepSeek32 on one A100
GPU. These are operational placements, not claims about model quality. A
different placement is acceptable only after the actual service and output
contract are revalidated.

## 15. Scientific Defensibility

### 15.1 Strengths of the design

The current committee design has several defensible features:

1. **Construct separation.** GR is deterministic, while BA, FG, and STR are
   judged only where semantic interpretation is needed.
2. **Task-specific prompts.** The committee does not use one generic "is this
   answer good?" prompt for all metrics.
3. **Explicit conflict conditioning.** BA is conditioned on a five-type policy
   rubric rather than treating all answers as ordinary QA answers.
4. **Evidence-bounded grounding.** FG evaluates citation-linked claims against
   eligible retrieved evidence rather than unrestricted world knowledge.
5. **Claim-level accounting.** FG is computed within each answer before the
   dataset macro-average, avoiding automatic length weighting.
6. **Corroboration for grounding.** A single high-priority judge cannot satisfy
   the full-valid-judge FG support rule alone.
7. **Preserved disagreement.** Individual votes, confidence, priority totals,
   rationales, and validity state remain available for audit.
8. **Explicit missingness.** Failed or inapplicable judgments are not silently
   reinterpreted as semantic labels.
9. **Protocol versioning.** Model identities, prompts, priorities, claim cap,
   and aggregation logic are tied to named configurations and result artifacts.
10. **Human validation pathway.** Human review can assess agreement among humans,
    agreement between humans and the committee, and committee-internal
    agreement.

### 15.2 What the committee does not prove

The committee does not prove that:

- a response is objectively true in the external world;
- the benchmark annotations are error-free;
- the judge models are statistically independent;
- the confidence values are calibrated probabilities;
- the 6/3/2 priorities are optimal or universally valid;
- high committee agreement is equivalent to human gold truth;
- high FG means unrestricted factual correctness beyond retrieved evidence; or
- a high CATS value establishes universal model trustworthiness.

The committee is an evaluation instrument with a defined construct and known
limitations. Those limitations should be included in the paper rather than
hidden behind the word "ensemble."

### 15.3 Main hostile-reviewer concerns and mitigations

#### Concern 1: "The judges may be correlated."

**Risk:** Three language models may share biases, and majority agreement may
reflect common model behavior rather than independent validation.

**Mitigation:** Report committee composition, per-judge outputs, pairwise
agreement where available, human agreement, and sensitivity to the committee
aggregation. Describe the committee as multi-model structured judging, not as
independent human-like annotators.

#### Concern 2: "The priority weights are arbitrary."

**Risk:** A 6/3/2 scheme can look like post-hoc tuning or a mechanism to make
the preferred judge dominate.

**Mitigation:** State the weights as configured design choices, do not call them
accuracy estimates, freeze them before comparing model variants, preserve
individual votes, and report equal-weight or alternate-weight sensitivity when
making strong claims. Do not tune priorities against the test results.

#### Concern 3: "Self-reported confidence is not calibrated."

**Risk:** Confidence may be a stylistic output rather than a probability of
correctness.

**Mitigation:** Use confidence only in the declared BA/STR support weighting,
retain the binary majority result separately, report confidence/support
distributions, and avoid interpreting confidence as calibrated probability. FG
deliberately uses raw priority rather than confidence-scaled support.

#### Concern 4: "One judge can dominate."

**Risk:** Qwen's priority exceeds half of the raw priority mass.

**Mitigation:** The FG rule additionally requires two valid judges when more than
one judge is available, so Qwen cannot independently establish document support
there. For BA/STR, report continuous support and individual judgments so the
anchor's influence is visible rather than hidden.

#### Concern 5: "The judge is evaluating its own interpretation of the rubric."

**Risk:** A model judge may reward fluency or generic helpfulness instead of the
target construct.

**Mitigation:** Use metric-specific prompts, explicit type-specific BA criteria,
structured outputs, orthogonality requirements, and human comparison. Keep
prompts fixed and provide the exact prompt bundle as an artifact.

#### Concern 6: "FG is circular because documents have gold verdicts."

**Risk:** The committee receives gold-annotated document eligibility, which may
look like evaluation leakage.

**Mitigation:** State the construct precisely: FG measures whether the answer
uses annotated retrieved evidence and cites it. Gold verdicts define the
benchmark's evidence labels; the committee still judges claim-specific support.
This is not an unrestricted fact-checking metric. Report the dependence on
document annotations as a limitation.

#### Concern 7: "Correct refusals get easy full credit."

**Risk:** A correct refusal receives decision-only CATS credit while answerable
examples face downstream behavior and content evaluation.

**Mitigation:** Disclose the policy, report answerable and refusal-required
subgroups, keep GR as a primary component, and do not claim that refusal quality
has been evaluated. If refusal wording quality becomes a research question,
add a separately specified refusal-quality rubric rather than silently changing
the current score.

#### Concern 8: "A flat aggregate hides denominator differences."

**Risk:** BA, FG, and STR do not apply to exactly the same examples.

**Mitigation:** Report `behavior_n`, `fg_n`, `str_n`, and `answer_quality_n`; build
CATS at the example level; preserve balanced and prevalence summaries; and keep
the four component metrics primary.

## 16. Human Evaluation and Committee Validation

Human evaluation is not folded silently into the local committee score. It is a
separate validation layer.

### 16.1 Human-human agreement

Human reviewers assess the same kinds of constructs under the human-evaluation
package. Analysis should report:

- reviewer coverage and overlap;
- agreement by metric and conflict type;
- the unit of agreement, such as example-level decision or claim-level label;
- Cohen's kappa or other agreement statistics with their missingness and
  prevalence assumptions;
- disagreements and adjudication policy; and
- the number of reviewers contributing to each sample.

### 16.2 Human-committee agreement

Human judgments can be aligned with committee outputs by stable sample IDs and,
for FG, by stable claim/document units where available. Agreement should be
reported separately for:

- GR decision agreement;
- BA agreement;
- FG claim support and citation-linkage agreement;
- STR target-assertion agreement; and
- any composite or overall behavior agreement.

Human-committee agreement supports the claim that the committee's judgments are
reasonably aligned with the study's human evaluation. It does not prove that the
committee is equivalent to a population of human experts, especially when human
coverage is incomplete or reviewer overlap is nonuniform.

### 16.3 Committee-internal agreement

The three local judges provide a further diagnostic: pairwise agreement,
unanimity, positive support mass, minority support, and disagreement by conflict
type. This is useful for identifying difficult regimes and for explaining why a
continuous CATS support value differs from a binary BA value.

The committee-internal analysis should not be described as independent human
inter-annotator agreement. It is agreement among model judges under the active
prompt and serving protocol.

## 17. Reproducibility and Artifact Chain

A committee result should be traceable through the following chain:

1. evaluated model output and its variant identity;
2. prepared benchmark input containing the query, documents, annotations, and
   stable sample ID;
3. exact committee configuration and model identities;
4. rendered prompt and, when cache artifacts are retained, its prompt hash for
   each judge call;
5. individual judge response, confidence, rationale, latency, and error state;
6. parsed per-task decision and vote totals;
7. per-example BA, FG, STR, GR, applicability, and CATS fields;
8. run-local detailed results and report;
9. audited master CSV/JSON/Markdown; and
10. presentation workbook, when used.

The workbook is not the primary scientific source. A reviewer should be able to
move from a displayed metric to the master row, from the master row to the
source `detailed_results.json`, and from there to per-example and per-judge
evidence.

### 17.1 What changes require a new committee version

Create a new versioned committee result if any of the following changes:

- model checkpoint, served model ID, quantization, or serving template;
- prompt wording, rubric, system instruction, or output contract;
- judge priorities or voting rule;
- confidence handling or validity filtering;
- claim extraction or maximum claim count;
- eligible-document rule or cross-document rule;
- STR partial-match threshold or assertion semantics;
- answer/refusal normalization;
- timeout, cache, or failed-response policy; or
- CATS aggregate formula.

Old outputs should remain available under the repository's legacy/provenance
boundary, but they must not be relabeled as results from the current protocol.

### 17.2 Minimum final-run checks

Before treating a run as paper-ready, verify:

- all three intended judge identities are present;
- every expected prompt call has either a valid response or an explicitly stored
  failure;
- no unintended cache misses or cross-variant cache reuse occurred;
- the claim cap and active YAML flags are correct;
- the final staged aggregation is read-only;
- BA, FG, and STR denominators are present;
- no all-judge failure is hidden inside a numeric metric;
- per-sample records and committee details are retained; and
- the summary is derived from the same result files audited into the master
  matrix.

## 18. ACL-Ready Methods Description

The following paragraph is a concise paper-facing description consistent with
the current protocol:

> We evaluated generated responses with a locally hosted committee of three
> OpenAI-compatible judge models: Qwen3.5-397B-A17B, Mistral Small 4, and
> DeepSeek-R1-Distill-32B, assigned fixed priorities of 6, 3, and 2. The
> committee judged conflict-conditioned Behavior Adherence, citation-linked
> Factual Grounding, and Single-Truth Recall using task-specific JSON-constrained
> prompts. Behavior and single-truth decisions used priority-by-confidence
> weighted majority, while factual-grounding document support used raw priority
> mass together with a corroboration requirement of at least two valid judges
> when multiple judges were available. Grounded Refusal was computed
> deterministically from benchmark answerability labels and the model's
> answer/refusal form rather than by the committee. We stored individual judge
> responses, rationales, confidence values, vote totals, validity states, cache
> prompt hashes where available, and per-example applicability fields. We report
> the component metrics
> and their denominators as primary results, with example-level balanced and
> prevalence CATS summaries as secondary analyses.

The paper should add the exact benchmark size, committee version, model-serving
details, prompt version, and human-agreement results appropriate to the final
experiment.

## 19. Reporting Checklist

### Committee identity

- [ ] Exact served model IDs are reported.
- [ ] The Qwen/Mistral/DeepSeek composition is distinguished from historical
  validation or fallback committees.
- [ ] Priorities 6/3/2 are reported as design weights, not judge accuracies.
- [ ] The committee size and valid-judge handling are stated.

### Prompt and metric scope

- [ ] BA, FG-v2, and STR are described separately.
- [ ] GR is identified as deterministic and outside the committee.
- [ ] The active prompt bundle is archived or linked.
- [ ] The eight-claim benchmark cap is disclosed.
- [ ] The absence of NLI from the active local committee path is clear.

### Aggregation

- [ ] BA and STR use the declared confidence-scaled weighted rule.
- [ ] FG uses raw priority and corroboration, not confidence weights.
- [ ] Weighted ties are identified as non-adherent for BA/STR.
- [ ] Individual votes and continuous support are retained.
- [ ] Applicability counts are reported.
- [ ] CATS is described as a secondary example-level aggregate.

### Validity and limitations

- [ ] Failed and malformed judge responses are distinguished from negative
  judgments.
- [ ] Committee-internal agreement is not called human IAA.
- [ ] Human-committee agreement is reported as validation, not proof of truth.
- [ ] Judge correlation, weighting, confidence calibration, annotation
  dependence, and refusal asymmetry are acknowledged.
- [ ] Any sensitivity analysis or its absence is stated.

## 20. Related Repository Documents

| Document | Purpose |
| --- | --- |
| `LOCAL_COMMITTEE_GUIDE.md` | Operational deployment, serving, staged collection, cache discipline, orchestration, and detailed implementation contract |
| `CATS_METRICS_METHODOLOGY.md` | Full definitions, formulas, denominators, applicability, and scientific defense for every metric |
| `CATS_AGGREGATE_LOGIC.md` | Current hierarchical CATS design, alternatives, and hostile-reviewer analysis |
| `HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md` | Human-study design, reviewer logic, consolidation, agreement metrics, committee validation, and paper-facing limitations |
| `../prompts/README.md` | Active prompt bundle and template-variable documentation |
| `../configs/benchmark_local_openai_3judge_qwen397.yaml` | Current benchmark committee identity, priorities, and evaluator flags |
| `../outputs/benchmark_local_committee_3judge/master_results/` | Audited master source artifacts for the 108-row experiment matrix |
| `../exports/cats_human_eval_cli/` | Separate human-evaluation pipeline and reviewer study artifacts |
| `../README.md` | Top-level repository map, end-to-end workflow, and paper-facing navigation |

## Closing Statement

The local committee is strongest when presented as a transparent measurement
instrument rather than as an unquestionable source of truth. Its contribution
to CATS v2 is to make conflict-aware behavior, citation-linked grounding, and
semantic target recovery measurable at scale under a fixed and auditable
protocol. The design is scientifically defensible when its construct boundaries,
priorities, prompts, missingness rules, human validation, and limitations are
reported with the same clarity as its headline scores.
