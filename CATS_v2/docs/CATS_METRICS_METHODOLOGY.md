# CATS v2 Metrics: Mathematical Definitions, Design Logic, and Scientific Defense

This document is the companion metric-methodology note for
[`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md).
That document focuses on the hierarchical CATS aggregate. This document explains
the complete metric stack that produces the values shown in the master results
workbook, including the grounded-refusal metrics, Behavior Adherence, Factual
Grounding, Single-Truth Recall, Answer Quality, applicability counts, and the two
CATS summaries.

The goal is not merely to list formulas. A metric is scientifically useful only
when its target construct, unit of analysis, denominator, annotation procedure,
failure modes, and interpretation are explicit. Every metric below is therefore
described at four levels:

1. What capability or failure mode it is intended to measure.
2. How the implementation computes it.
3. Why the construction is defensible for this RAG conflict-resolution task.
4. What the metric does not establish and how it should be reported.

## 1. Current Status and Scope

The active aggregate implementation is `cats_h_gated_harmonic_v1`. The
paper-facing hierarchical master matrix contains exactly 108 experiment rows:

- 96 standard benchmark runs;
- 6 answer-only SFT runs;
- 2 comparison runs for the available Llama technique results;
- 4 redone comparison runs from `other_techniques_fixed` for Mistral and Qwen across CoT few-shot and CoN.

The standard local committee benchmark contains 736 examples divided into five
conflict regimes. The metric definitions operate on one example at a time and
then aggregate over the appropriate applicable examples. The master workbook is
a presentation artifact; the authoritative per-example evidence is stored in
each run's `final/detailed_results.json`, especially its `per_sample` records.

The four fixed comparison runs are the latest versions used in the master
results. Their latest applicability counts are retained: fixed CoT Mistral has
651 Behavior/FG-applicable examples, fixed CoT Qwen has 698, and fixed CoN
Mistral and Qwen have 736 each. Older unfixed Mistral/Qwen comparison files and
staged collection files remain local for provenance but are outside the 108-row
master scope.

There is an important audit-tool scope distinction. The current
`scripts/audit_cats_master_results.py` is the authoritative 108-row inventory
check. The older `scripts/verify_master_gr_metrics.py` still enumerates a
historical 114-file universe; its six "missing" rows are the four unfixed
comparison finals and two staged collection artifacts deliberately excluded from
the current master. Its legacy 114-row warning must not be interpreted as a
missing row or a change to the current 108-experiment result set.

## 2. Master Workbook Map

The current `cats_master_results` sheet uses the following semantic columns.
The letter mapping matters because the workbook was extended from the older
flat-CATS layout.

| Columns | Current field | Meaning | Unit |
| --- | --- | --- | --- |
| J | `gr_answering_precision` | Precision when answering is the positive class | Dataset |
| K | `gr_answering_recall` | Recall of answerable examples | Dataset |
| L | `gr_answering_f1` | F1 for the answer-positive GR task | Dataset |
| M | `gr_refusal_precision` | Precision when refusal is the positive class | Dataset |
| N | `gr_refusal_recall` | Recall of refusal-required examples | Dataset |
| O | `gr_refusal_f1` | F1 for the refusal-positive GR task | Dataset |
| P | `gr_accuracy` | Correct answer-versus-refusal decisions | Dataset |
| Q | `single_truth_recall` | Recovery of the gold answer when STR applies | Applicable examples |
| R | `factual_grounding` | Citation-supported claim fraction | Applicable examples |
| S | `behavioral_adherence` | Conflict-policy adherence | Applicable examples |
| T | `answer_quality` | Per-example FG/STR fusion, then averaged | Applicable examples |
| U | `final_cats_prevalence` | CATS-Prevalence | Dataset |
| V | `final_cats_balanced` | CATS-Balanced | Dataset |
| W | `n` | Total examples | Count |
| X | `behavior_n` | Behavior-applicable examples | Count |
| Y | `fg_n` | Grounding-applicable examples | Count |
| Z | `str_n` | STR-applicable examples | Count |
| AA | `answer_quality_n` | Examples with computable Answer Quality | Count |

The workbook does not expose every diagnostic field. The source JSON and master
CSV/JSON also retain fields such as committee consensus, correct-refusal count,
`cats_complete`, and the legacy flat average for audit and compatibility.

## 3. Evaluation Objects and Notation

Let the evaluation set for one experiment contain examples
\(i=1,\ldots,N\). For the standard benchmark, \(N=736\).

For each example:

- \(A_i \in \{0,1\}\): gold answerability. `1` means the evidence supports an answer; `0` means the correct action is refusal.
- \(\hat A_i \in \{0,1\}\): predicted answerability. `1` means the model produced an answer; `0` means the output is detected as a refusal.
- \(g_i \in \{0,1\}\): grounded-refusal decision correctness.
- \(b_i^{bin} \in \{0,1\}\): final committee binary Behavior Adherence decision when applicable.
- \(b_i \in [0,1]\): continuous committee support for Behavior Adherence used by CATS.
- \(f_i \in [0,1]\): per-example Factual Grounding score when applicable.
- \(r_i \in [0,1]\): per-example Single-Truth Recall score when applicable.
- \(q_i \in [0,1]\): per-example Answer Quality score when computable.
- \(s_i \in [0,1]\): per-example hierarchical CATS score.

The central design distinction is between:

1. **Decision correctness:** did the system answer or abstain appropriately?
2. **Conflict behavior:** did it handle the conflict regime appropriately?
3. **Evidence grounding:** are its cited claims supported by retrieved evidence?
4. **Target recovery:** did it assert the intended single truth when one exists?

Keeping these constructs separate prevents a model from appearing strong merely
because one broad score hides a specific failure mode.

## 4. End-to-End Metric Pipeline

For every run, the evaluator follows this conceptual sequence:

1. Load the model output and remove any think-trace section before scoring.
2. Detect whether the final output is an answer or a refusal.
3. Determine the gold answerability from the benchmark annotation.
4. Compute the grounded-refusal decision signal and the dataset confusion matrix.
5. For non-correct-refusal outputs, extract answer claims and citations.
6. Run the Behavior Adherence committee.
7. Run committee-based Factual Grounding over extracted claims.
8. Run Single-Truth Recall when a gold answer exists and the conflict type is configured for STR.
9. Aggregate each component over its own applicable denominator.
10. Construct Answer Quality and the hierarchical CATS summaries from per-example values.

The implementation deliberately does not treat a correct refusal as an answer
with zero grounding. Correct refusals have no answer content to ground and are
excluded from Behavior, FG, STR, and Answer Quality denominators. They contribute
to grounded-refusal decision metrics and to CATS through decision correctness.

## 5. Grounded-Refusal Decision Metrics

### 5.1 Construct and use case

Grounded Refusal (GR) measures whether the model chooses the correct high-level
action under the available evidence:

- answer when the retrieved evidence is sufficient;
- refuse when the evidence is insufficient, contradictory, unreliable, or otherwise not adequate for a supported answer.

This is the foundational RAG conflict capability. A system can produce fluent,
well-cited text and still fail the task if it answers when it should abstain. It
can also be over-conservative by refusing answerable questions. GR therefore
must report both sides of the decision problem rather than only measuring answer
quality on outputs that happened to be answered.

### 5.2 Predicted answer/refusal label

The evaluator normalizes the model output and applies a start-oriented refusal
detector. It recognizes canonical opening forms such as:

- `CANNOT ANSWER` or `CAN'T ANSWER`;
- `INSUFFICIENT EVIDENCE`, `INSUFFICIENT INFORMATION`, or `NOT ENOUGH EVIDENCE`;
- `UNABLE TO ANSWER`, `UNABLE TO DETERMINE`, or `CANNOT DETERMINE`;
- wrapped openings such as an evidence preface followed shortly by an explicit inability to answer.

The detector is intentionally start-oriented. A broad substring search would
misclassify a substantive answer that says, for example, "the evidence is
insufficient for source X, but the retrieved record supports Y". Empty output is
treated as a refusal. Think traces and explicit end-of-answer sentinels are
removed before this decision.

Define:

$$
\hat A_i =
\begin{cases}
1, & \text{if the normalized output is an answer},\\
0, & \text{if it is empty or matches the refusal detector}.
\end{cases}
$$

This is a behavioral parser, not a semantic judge. Its purpose is to establish
the answer/refusal decision variable consistently before evaluating content.

### 5.3 Gold answerability

Gold answerability is resolved with a precedence rule rather than inferred only
from the presence of supporting documents:

1. `expected_response.abstain`, when present, is authoritative; gold answerability is its negation.
2. `answerable_under_evidence`, when present, is used next.
3. Older schemas fall back to annotated per-document support verdicts.

This precedence matters because a document can partially support a proposition
without making the overall question answerable. Treating any partial support as
answerability would incorrectly penalize appropriate abstentions.

### 5.4 Per-example decision correctness

The per-example GR signal is a binary agreement indicator:

$$
g_i = \mathbf{1}[\hat A_i=A_i].
$$

Equivalently:

$$
g_i =
\begin{cases}
1, & \hat A_i=1 \text{ and } A_i=1,\\
1, & \hat A_i=0 \text{ and } A_i=0,\\
0, & \text{otherwise.}
\end{cases}
$$

The dataset GR accuracy reported in column P is:

$$
\mathrm{GR\ Accuracy}=\frac{1}{N}\sum_{i=1}^{N}g_i
=\frac{TP+TN}{TP+FP+FN+TN}.
$$

This per-example signal is also the decision gate used by the CATS aggregate.
Dataset-level F1 is not used as an example-level CATS primitive because F1 is a
non-decomposable function of the complete confusion matrix.

### 5.5 Answer-positive confusion matrix

For the primary GR orientation, **answered** is the positive class and
**refused** is the negative class:

| | Gold answerable \(A_i=1\) | Gold refusal-required \(A_i=0\) |
| --- | ---: | ---: |
| Predicted answer \(\hat A_i=1\) | TP | FP |
| Predicted refusal \(\hat A_i=0\) | FN | TN |

The formulas are:

$$
\mathrm{Precision}_{ans}=\frac{TP}{TP+FP},
$$

$$
\mathrm{Recall}_{ans}=\frac{TP}{TP+FN},
$$

$$
F1_{ans}=\frac{2\,\mathrm{Precision}_{ans}\,\mathrm{Recall}_{ans}}
{\mathrm{Precision}_{ans}+\mathrm{Recall}_{ans}}.
$$

When a denominator is zero, the implementation returns zero rather than
silently producing an undefined value. The answer-positive precision, recall,
and F1 appear in columns J, K, and L.

**Interpretation:**

- Answer precision penalizes answering questions that should have been refused.
- Answer recall penalizes refusing answerable questions.
- Answer F1 summarizes the balance between those two answer-side errors.

### 5.6 Refusal-positive diagnostics

The same confusion matrix is reoriented with **refusal** as the positive class:

$$
TP_{ref}=TN,\quad FP_{ref}=FN,\quad FN_{ref}=FP,\quad TN_{ref}=TP.
$$

Therefore:

$$
\mathrm{Precision}_{ref}=\frac{TN}{TN+FN},
$$

$$
\mathrm{Recall}_{ref}=\frac{TN}{TN+FP},
$$

$$
F1_{ref}=\frac{2\,\mathrm{Precision}_{ref}\,\mathrm{Recall}_{ref}}
{\mathrm{Precision}_{ref}+\mathrm{Recall}_{ref}}.
$$

These values appear in columns M, N, and O. They are not redundant with the
answer-positive metrics: a model can have strong answer recall while refusing
too rarely, or strong refusal recall while refusing too aggressively. Reporting
both orientations makes the trade-off visible.

The implementation also records an abstention-specificity diagnostic. Under the
refusal-positive orientation, this is:

$$
\mathrm{Specificity}_{ref}=\frac{TN_{ref}}{TN_{ref}+FP_{ref}}
=\frac{TP}{TP+FN},
$$

which is numerically the answer-positive recall because the two are the same
binary confusion matrix viewed from opposite positive-class orientations. It is
an audit diagnostic, not a separate master-workbook column.

### 5.7 Scientific defense and limitations

The GR family is defensible because it directly evaluates the decision the
benchmark is designed to test, uses a transparent confusion matrix, and reports
both answer and refusal perspectives. It avoids the common error of evaluating
only answered outputs, which would hide over-refusal and under-refusal.

Its limitations are equally important:

- It evaluates the answer/refusal action, not whether an answer is well-written or complete.
- It depends on the benchmark's gold answerability annotation.
- Refusal detection is a deterministic parser and can miss unusual refusal wording or misclassify an unusual opening.
- Accuracy can be affected by answerability prevalence, so the two F1 orientations should accompany it.

## 6. Behavior Adherence (BA)

### 6.1 Construct and use case

Behavior Adherence measures whether the response follows the expected policy for
the particular conflict regime. It is not a factual correctness metric and is
not a citation-grounding metric. It asks whether the model used the right
conflict-resolution behavior.

The five regime-specific rubrics are:

| Type | Conflict regime | Expected response behavior |
| ---: | --- | --- |
| 1 | No Conflict | Give a clear, direct answer without inventing alternatives or uncertainty. |
| 2 | Complementary Information | Consolidate partial answers into a coherent response rather than framing them as a debate. |
| 3 | Conflicting Opinions or Research Outcomes | Represent the disagreement neutrally and summarize the competing viewpoints. |
| 4 | Outdated Information | Prioritize the up-to-date information and optionally acknowledge outdated sources. |
| 5 | Misinformation | Reject inaccurate sources and answer from reliable, verified evidence. |

The construct is necessary because a response can be factually plausible while
handling a conflict incorrectly. For example, collapsing a genuine research
disagreement into a single unqualified claim is a behavioral failure even if
one cited source happens to support that claim.

### 6.2 Per-example committee decision

For an applicable answered example, each judge receives the query, model answer,
conflict type, and the applicable rubric. For Types 4 and 5, compact document
dates and source provenance are also surfaced so that temporal or reliability
prioritization can be judged against evidence rather than wording alone.

Each valid judge returns:

- `adherent`: a binary policy decision;
- `confidence`: confidence in that decision;
- rationale and audit metadata.

The committee can use majority, unanimous, or weighted-majority voting. In the
active weighted-majority design, judge \(j\) has configured priority \(p_j\) and
returns confidence \(c_j\). Its effective weight is:

$$
w_j=p_j\max(c_j,0.01).
$$

The weighted committee decision is:

$$
W^+=\sum_{j:y_j=1}w_j,\qquad
W^- =\sum_{j:y_j=0}w_j,
$$

$$
b_i^{bin}=\mathbf{1}[W^+>W^-].
$$

The final binary Behavior Adherence value stored for direct reporting is

$$
BA=\frac{1}{N_{BA}}\sum_{i\in I_{BA}}b_i^{bin},
$$

where \(I_{BA}\) is the set of behavior-applicable examples and
\(N_{BA}=|I_{BA}|\). Correct refusals are not in \(I_{BA}\), because they have
no answer behavior to evaluate in this version of CATS.

### 6.3 Continuous consensus used by CATS

The CATS aggregate preserves disagreement instead of treating every majority as
identical. The per-example consensus support is:

$$
b_i=\frac{W^+}{W^++W^-}.
$$

If weighted totals are unavailable, the implementation falls back to the raw
vote fraction:

$$
b_i=\frac{V^+}{V^++V^-}.
$$

If no vote details are available, the stored binary Behavior score is used as a
compatibility fallback. Thus, a unanimous 3--0 decision contributes 1.0, while
a 2--1 decision contributes approximately 2/3 under an equal-weight committee.

This continuous value is a CATS input, not a replacement for the primary binary
BA report. The distinction lets the paper report an interpretable adherence
rate while retaining committee uncertainty in the secondary scalar.

### 6.4 Orthogonality requirement

The Behavior judge is instructed to exclude:

- answerability correctness;
- factual entailment and citation validity;
- unsupported-claim detection as a hidden grounding criterion;
- Single-Truth Recall;
- whether the gold answer itself was recovered.

This separation is scientifically important. If BA independently rewarded
factual correctness and FG also rewarded factual support, the aggregate would
double-count the same property. BA should be read as conflict-policy adherence,
not as a general answer-quality score.

### 6.5 Scientific defense and limitations

BA has strong construct validity for CATS because the benchmark is explicitly
about handling different evidence-conflict regimes. A single undifferentiated
"helpfulness" score would not test whether the model consolidates complementary
evidence, represents disagreement, handles temporal supersession, and rejects
misinformation differently.

The main limitations are judge dependence, rubric interpretation, and the fact
that the committee itself is an evaluator rather than human ground truth. BA
should therefore be reported with committee composition, vote statistics, and
human-agreement analysis where available.

## 7. Factual Grounding (FG)

### 7.1 Construct and use case

Factual Grounding measures whether the claims made by the model are supported by
the retrieved evidence it cites. It is narrower than factual correctness in the
world: a claim may be true in reality but not grounded in the evidence available
to the model. Conversely, a retrieved document can support a claim even when the
claim is not independently verified outside the benchmark.

This distinction is central to RAG evaluation. The metric targets evidence use,
not merely the plausibility of generated text.

### 7.2 Claim and citation extraction

The evaluator operates on the model's final answer after think-trace removal.
It splits the answer into candidate claims and preserves inline document
citations such as `[d1]`, `[d2]`, or grouped citation forms. The claim extractor:

- protects periods in initials, decimals, domains, and common abbreviations;
- removes citation-only fragments;
- removes citation meta-statements that are not substantive claims;
- strips attribution wrappers so NLI sees the verifiable proposition;
- drops very short fragments;
- applies the configured maximum number of claims per answer.

The current evaluation configuration uses `max_claims_per_answer=5`. The exact
extracted claim list is retained in per-sample details for auditability.

### 7.3 Eligible evidence

The evaluator filters retrieved documents using gold per-document annotations.
Only positive support verdicts are eligible:

$$
\mathcal D_i^+ = \{d:\mathrm{verdict}(d)\in
\{\text{supports},\text{partially supports}\}\}.
$$

Documents annotated as irrelevant are not allowed to ground a claim. The gold
document verdicts provide relevance ground truth; the committee is not asked to
re-judge whether a document is relevant to the query. It is asked whether the
specific claim is conveyed by the eligible document's text.

### 7.4 Claim-level support rule

For each extracted claim \(c_{ik}\), the FG committee identifies documents that
semantically support the claim. Paraphrases, natural implications, and
annotated partial support can count when they establish the claim's core
assertion. Keyword overlap alone does not count.

A claim is grounded only if both conditions hold:

1. At least one eligible document, or an accepted two-document combination, supports the claim.
2. The model cited at least one of the supporting document IDs.

For single-document support:

$$
G_{ik}=\mathbf{1}\left[\mathcal S_{ik}\neq\varnothing
\;\land\;\mathcal C_{ik}\cap\mathcal S_{ik}\neq\varnothing\right],
$$

where \(\mathcal S_{ik}\) is the committee-supported document set and
\(\mathcal C_{ik}\) is the model's cited-document set.

For cross-document support, the committee can identify a pair whose combined
evidence establishes the claim. The model must still cite at least one document
from that pair. No confidence weighting or contradiction penalty is applied in
the current FG-v3 score; support is binary at the claim level.

### 7.5 Per-example FG formula

Let \(K_i\) be the number of extracted claims for an answerable/applicable
example. Then:

$$
f_i=\frac{\sum_{k=1}^{K_i}G_{ik}}{K_i}.
$$

Thus, a response with 4 extracted claims, 3 of which are supported and properly
cited, receives \(f_i=3/4=0.75\). If an answer has no extractable claims, its
FG result is 0.0 when FG is applicable. Correct refusals are handled earlier
and are excluded from FG rather than being interpreted as zero-grounding
answers.

### 7.6 Dataset-level FG

The workbook FG value is the arithmetic mean over applicable examples:

$$
FG=\frac{1}{N_{FG}}\sum_{i\in I_{FG}}f_i.
$$

This is an example-macro average, not a pooled claim micro-average. Therefore,
each answer contributes one example-level score regardless of how many claims it
contains. This avoids long answers automatically receiving more weight solely
because they contain more sentences, although the configured claim cap and
extraction policy still influence the metric.

### 7.7 Scientific defense and limitations

FG is defensible because it enforces the two conditions that define grounded
generation: semantic evidential support and citation linkage. It also permits
cross-document support, which is necessary for complementary-information cases.

The metric does not prove external factual truth, and it inherits the quality of
the gold document annotations and judge committee. The support prompt is
deliberately permissive for paraphrase and partial support, so FG should not be
interpreted as a strict entailment benchmark. Claim extraction errors can also
affect the denominator. These limitations should be disclosed and supported by
claim-level audit samples.

## 8. Single-Truth Recall (STR)

### 8.1 Construct and use case

STR measures whether the model's answer asserts the designated gold answer when
the conflict regime has a single target truth. It complements FG:

- FG asks whether what the model says is supported by retrieved evidence.
- STR asks whether the model recovered and committed to the intended target answer.

A response can be well grounded yet omit the central answer. Conversely, it can
mention the gold answer in a citation while rejecting it; that should not count
as recall.

### 8.2 Applicability

The active configuration enables STR for conflict types 1, 2, 4, and 5:

$$
I_{STR}=\{i:\text{gold answer exists and }t_i\in\{1,2,4,5\}\}.
$$

Type 3 is excluded because conflicting opinions or research outcomes do not
necessarily have one answer that should be asserted as the sole truth. Correct
refusals are also not STR-applicable. If a gold answer is absent, STR is not
computed for that example.

### 8.3 Committee judgment target

The STR committee is asked whether the model **asserts the gold answer as its own
conclusion**. The following count as a match:

- paraphrases and abbreviations;
- logically equivalent formulations;
- minor spelling, punctuation, casing, whitespace, and unit-formatting differences.

The following do not count:

- merely quoting a document that contains the gold answer;
- listing the gold as one possibility without endorsing it;
- asserting a conflicting answer;
- refusing to answer.

This assertion-versus-mention distinction is essential for a RAG system that may
quote evidence while reaching the wrong conclusion.

### 8.4 Per-example STR formula

Let there be \(M_i\) gold answer items for example \(i\). For each gold item,
the committee returns a binary adherent decision. Let \(E_i\) be the number of
exact/semantic matches and \(P_i\) the number of partial matches.

The implementation uses the following ordered rule:

$$
r_i=
\begin{cases}
1, & E_i>0,\\
\min\left(1,\;0.5\frac{P_i}{M_i}\right), & E_i=0 \text{ and } P_i>0,\\
0, & \text{otherwise.}
\end{cases}
$$

Partial credit is awarded only when the negative committee decision has
non-trivial minority support on the positive side, currently at least 0.30.
This avoids using the majority confidence incorrectly: a confident negative
decision with a tiny positive minority should not be treated as an uncertain
partial match.

For a single gold answer, the possible values are therefore 0, 0.5, and 1.0.
For multiple gold items, partial matches scale by the fraction of gold items
partially recovered, capped at 1.0.

### 8.5 Dataset-level STR

The workbook STR value is:

$$
STR=\frac{1}{N_{STR}}\sum_{i\in I_{STR}}r_i.
$$

It is not a token-overlap recall and should not be compared directly to lexical
recall metrics from ordinary QA benchmarks. It is a committee-judged semantic
target-recovery rate under the CATS answerability and conflict-type rules.

### 8.6 Scientific defense and limitations

STR is necessary because grounding alone can reward a response that cites true
facts but fails to answer the question. Its semantic assertion criterion is more
appropriate than exact string matching for paraphrastic model outputs.

Its limitations are the LLM committee's semantic judgment, the hard distinction
between exact and partial recovery, and the exclusion of Type 3. STR must be
reported with its applicable count and should not be interpreted as measuring
all factual content in the response.

## 9. Applicability Counts and Denominators

### 9.1 Why denominators are first-class results

An average without its denominator can be misleading. In CATS, different
properties apply to different examples by design. A correct refusal has no
answer claims, so scoring it as a zero for FG or STR would punish the model for
not producing content that the gold decision says it should not produce.

The workbook therefore reports:

- `n`: total examples;
- `behavior_n`: examples included in the BA mean;
- `fg_n`: examples included in the FG mean;
- `str_n`: examples included in the STR mean;
- `answer_quality_n`: examples for which Answer Quality was computable.

For every component metric \(m\), the denominator is its own applicable set:

$$
\bar m=\frac{1}{N_m}\sum_{i\in I_m}m_i.
$$

This is a missing-by-design distinction, not missing-data imputation. The
applicability rule is defined by the task and stored in per-sample fields such
as `behavior_applicable`, `factual_grounding_applicable`, and
`single_truth_applicable`.

### 9.2 Correct refusal handling

For a correct refusal:

- GR decision correctness is 1;
- Behavior Adherence is not applicable;
- Factual Grounding is not applicable;
- Single-Truth Recall is not applicable;
- Answer Quality is not applicable;
- CATS receives the decision-only value 1.

For a wrong refusal, the decision gate is zero. Answer-content metrics may be
computed if the output contains answer content, but no downstream quality can
rescue the incorrect answer/refusal decision in CATS.

### 9.3 Applicability is not a performance score

High `behavior_n` or `fg_n` does not mean high quality. It means more examples
were eligible for that construct. Counts must be reported beside the metric,
especially for the four fixed comparison runs whose latest applicability counts
differ from older workbook displays.

## 10. Answer Quality Pillar

Answer Quality is an intermediate per-example construct used by CATS. It is not
the same as the final aggregate and should be interpreted alongside FG and STR.

For an applicable answer:

$$
q_i=
\begin{cases}
\sqrt{f_i r_i}, & f_i \text{ and } r_i \text{ are both applicable},\\
f_i, & \text{only FG is applicable},\\
\text{not applicable}, & \text{FG is not applicable}.
\end{cases}
$$

The dataset-level workbook value is:

$$
AQ=\frac{1}{N_{AQ}}\sum_{i\in I_{AQ}}q_i.
$$

The geometric mean is used within Answer Quality because grounding and target
recovery are jointly important. It is zero-sensitive: if either component is
zero, the combined answer-quality score is zero. It also prevents a high STR
value from fully masking a low grounding value, and vice versa.

This geometric mean is deliberately not used as the top-level CATS fusion. A
cube-root fusion of decision, behavior, and answer quality would inflate weak
non-zero quality values. The top-level CATS design instead uses decision gating
and a harmonic mean for behavior plus Answer Quality.

## 11. Hierarchical CATS Summaries

The full rationale and reviewer-facing defense are in
[`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md).
The operational summary is included here so the workbook fields are
self-contained.

### 11.1 Per-example CATS score

For an answerable/applicable example:

$$
s_i=g_i H(b_i,q_i),
$$

where the harmonic mean is:

$$
H(b_i,q_i)=\frac{2b_iq_i}{b_i+q_i},
$$

with value zero if either input is zero. Since \(g_i\) is binary,

$$
g_i=0\Longrightarrow s_i=0.
$$

For a refusal-required example:

$$
s_i=g_i.
$$

Thus, the aggregate rewards the correct decision for a required refusal but does
not claim to measure refusal wording quality.

The implementation has compatibility fallbacks when a historical answerable
row lacks one intermediate component:

- both \(b_i\) and \(q_i\): gated harmonic fusion;
- only \(b_i\): \(s_i=g_i b_i\);
- only \(q_i\): \(s_i=g_i q_i\);
- neither: \(s_i=g_i\).

These fallbacks are auditable in the stored applicability counts. Standard
current benchmark runs normally use the full path.

### 11.2 Type-level aggregation

For conflict type \(t\), compute the mean of example scores:

$$
T_t^{A}=\frac{1}{|I_t^A|}\sum_{i\in I_t^A}s_i,
\qquad
T_t^{R}=\frac{1}{|I_t^R|}\sum_{i\in I_t^R}s_i.
$$

The decision-balanced type score is:

$$
T_t=
\begin{cases}
\frac{T_t^A+T_t^R}{2}, & \text{if both subgroups exist},\\
T_t^A, & \text{if only answerable examples exist},\\
T_t^R, & \text{if only refusal-required examples exist}.
\end{cases}
$$

This prevents a type's score from being dominated by whichever answerability
class is more prevalent when both classes exist.

### 11.3 CATS-Prevalence

CATS-Prevalence preserves the empirical example distribution:

$$
\mathrm{CATS\text{-}Prevalence}
=\frac{1}{N}\sum_{i=1}^{N}s_i.
$$

It is useful when the reported number should reflect the actual benchmark mix,
including type and answerability prevalence. It can be influenced by benchmark
composition, so it should never be reported without the type counts and
component metrics.

### 11.4 CATS-Balanced

For the five represented conflict types:

$$
\mathrm{CATS\text{-}Balanced}
=\frac{1}{5}\sum_{t=1}^{5}T_t.
$$

This gives each conflict regime equal top-level importance and balances
answerable/refusal-required subgroups within a type when both are present. It is
the preferred first aggregate column for regime-balanced comparison, but it
remains a secondary summary rather than the primary scientific criterion.

The implementation also retains `cats_type_balanced_score`, which averages raw
type means without the within-type answerability balancing. It is a diagnostic,
not the canonical workbook aggregate.

### 11.5 Legacy flat average

The implementation retains `cats_flat_legacy_score` for historical comparison.
It is an arithmetic average of whichever marginal components are available and
does not represent the current hierarchical design. It must not be used as the
paper-facing CATS headline.

## 12. Scientific Design Rationale

### 12.1 Construct separation

The metrics correspond to distinct failure modes:

| Failure mode | Metric that exposes it |
| --- | --- |
| Answers when evidence is insufficient | GR answer/refusal metrics |
| Refuses answerable questions | GR answer recall / refusal diagnostics |
| Mishandles the conflict regime | Behavior Adherence |
| Makes unsupported cited claims | Factual Grounding |
| Omits or rejects the target truth | Single-Truth Recall |
| Succeeds jointly across decision, behavior, and content | CATS summaries |

This decomposition improves construct validity. A single scalar cannot tell a
reviewer whether a low score came from over-refusal, hallucinated support,
failure to represent disagreement, or omission of the target answer.

### 12.2 Conditional scoring and non-applicability

The benchmark is not a conventional answer-only QA task. Refusal is sometimes
the correct output. Applying answer-content metrics to correct refusals would
create a design error: it would treat the absence of claims as a failure even
when the gold policy says not to answer. Separate applicability denominators are
therefore more defensible than forcing every metric onto every example.

### 12.3 Non-compensation

The decision gate enforces a hierarchy: a wrong answer/refusal decision cannot be
rescued by fluent prose or high apparent grounding. The harmonic mean then
requires both behavior and answer quality to be strong after the decision gate.
This is preferable to a flat average when the dimensions are jointly necessary.

### 12.4 Committee uncertainty

Direct component reports use binary committee decisions for interpretability.
CATS additionally retains continuous committee support so disagreement is not
erased. This is a transparent compromise: it does not pretend that judge
confidence is human certainty, but it preserves more information than a hard
majority label alone.

### 12.5 No claim of universal optimality

The weights and aggregation functions are design choices, not laws of nature.
They are defensible because they are pre-specified, interpretable, grounded in
the task hierarchy, and not fitted to improve a particular model ranking. The
paper should present component metrics first and use CATS as a structured
secondary summary with sensitivity analyses.

## 13. Recommended Reporting for the Paper

### Primary table

Report at minimum:

- GR answer precision, recall, and F1;
- GR refusal precision, recall, and F1;
- GR accuracy;
- Behavior Adherence with `behavior_n`;
- Factual Grounding with `fg_n`;
- Single-Truth Recall with `str_n`.

### Secondary columns

- Answer Quality with `answer_quality_n`;
- CATS-Balanced;
- CATS-Prevalence.

### Diagnostics and uncertainty

Report or place in an appendix:

- the five type-level CATS scores;
- answerable versus refusal-required subgroup scores;
- confusion-matrix counts TP/FP/FN/TN;
- committee consensus distribution and valid-judge counts;
- paired bootstrap confidence intervals or another pre-specified uncertainty procedure;
- sensitivity to CATS formula choice and to leaving out the small Type 5 regime;
- human-versus-committee agreement results where human evaluation is available.

The recommended interpretation is:

> CATS is a secondary structured summary. The primary conclusions are based on the grounded-refusal decision metrics, Behavior Adherence, Factual Grounding, and Single-Truth Recall. CATS-Balanced and CATS-Prevalence provide complementary views of joint performance under regime-balanced and empirical-prevalence weighting.

## 14. Reproducibility and Audit Procedure

For a result file to be accepted into the master matrix:

1. Confirm that `per_sample` is present and has the expected benchmark rows.
2. Recompute the GR confusion matrix directly from `pred_answered` and `gold_answerable`.
3. Recompute BA, FG, STR, and applicability counts from stored per-sample fields.
4. Recompute Answer Quality and both CATS summaries using the active evaluator.
5. Confirm that all five conflict types represented in the run have scored examples.
6. Confirm `cats_complete=True` and `cats_unscorable_n=0` for the final summary.
7. Verify that the source path belongs to the authoritative 108-row scope.
8. Preserve committee details, vote totals, judge identities, and applicability flags.

The repository's current checks include:

- `scripts/audit_cats_master_results.py` for the 108-row source matrix;
- `scripts/update_master_results_workbook.py` for workbook regeneration and cell verification;
- `scripts/verify_master_gr_metrics.py` for independent GR recomputation;
- `tests/test_cats_aggregate.py` for aggregate invariants.

## 15. Limitations and Responsible Claims

These metrics support claims about the evaluated benchmark and committee setup;
they do not establish universal model quality.

- GR depends on the benchmark's answerability labels and the refusal parser.
- BA, FG, and STR depend on LLM-judge validity and committee composition.
- FG depends on the quality and granularity of document-level gold annotations.
- STR is not lexical recall and is intentionally unavailable for Type 3 in the active configuration.
- Applicability denominators can differ substantially across runs and must be shown.
- CATS is a designed summary and should not replace the component metrics.
- Correct refusal wording quality is outside the current CATS aggregate.
- Small type-specific populations, especially Type 5, can produce unstable estimates.
- The latest fixed comparison runs have different applicability counts from older workbook displays; the master uses the latest fixed counts by design.

The scientifically defensible claim is therefore not "one scalar proves the
model is good." It is that the evaluation reports complementary, auditable
constructs and provides a transparent secondary summary of joint conflict-aware
performance under explicitly documented assumptions.

## 16. Implementation References

- [`rag_eval/metrics.py`](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/metrics.py): refusal detection, claim extraction, GR formulas.
- [`rag_eval/data.py`](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/data.py): gold answerability and gold-answer extraction.
- [`rag_eval/conflict_eval.py`](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/conflict_eval.py): Behavior, Factual Grounding, and STR committee paths.
- [`rag_eval/judge_committee.py`](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/judge_committee.py): judge voting and weighted support.
- [`rag_eval/judge_prompts.py`](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/judge_prompts.py): behavior, FG, and STR rubrics.
- [`rag_eval/evaluator.py`](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/evaluator.py): per-sample orchestration and aggregate calculations.
- [`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md): detailed CATS aggregate rationale and reviewer-facing defense.

## Appendix A. Implementation-Locked Formula Inventory

This appendix is intentionally more operational than the preceding sections. It
is a checklist against the active Python implementation, so that a paper,
reviewer response, or independent reimplementation does not silently omit a
branch of a metric. The formulas below use the same names as the serialized
results wherever possible. A value shown as a percentage in a spreadsheet is the
corresponding unit-interval value multiplied by 100 for display only.

### A.1 Complete metric inventory

| Metric or field | Unit of analysis | Exact source of the value | Denominator or domain | Active in master workbook |
| --- | --- | --- | --- | --- |
| `gr_accuracy` | Example, then dataset | Equality of predicted and gold answerability | All `N` examples; dataset denominator `max(1,N)` | Yes, column P |
| `gr_answering_precision` | Dataset | Answer-positive confusion matrix | Predicted-answer positives | Yes, column J |
| `gr_answering_recall` | Dataset | Answer-positive confusion matrix | Gold-answerable positives | Yes, column K |
| `gr_answering_f1` | Dataset | Harmonic mean of answer precision and recall | Both answer-positive metrics | Yes, column L |
| `gr_refusal_precision` | Dataset | Refusal-positive reorientation of the same matrix | Predicted-refusal positives | Yes, column M |
| `gr_refusal_recall` | Dataset | Refusal-positive reorientation of the same matrix | Gold-refusal positives | Yes, column N |
| `gr_refusal_f1` | Dataset | Harmonic mean of refusal precision and recall | Both refusal-positive metrics | Yes, column O |
| `behavioral_adherence` | Example, then applicable-example mean | Binary committee result | `behavior_n` | Yes, column S |
| `behavior_consensus_score` | Example, then diagnostic mean | Weighted support fraction for adherence | Valid committee responses | Retained in source artifacts; not a workbook column |
| `factual_grounding` | Example, then applicable-example mean | Supported cited claims divided by extracted claims | `fg_n` examples; claim denominator within each example | Yes, column R |
| `single_truth_recall` | Example, then applicable-example mean | Exact-or-qualified-partial recovery of each gold answer | `str_n` examples; gold-answer denominator within each example | Yes, column Q |
| `answer_quality` | Example, then computable-example mean | Geometric fusion of FG and STR where both apply | `answer_quality_n` | Yes, column T |
| `cats_example_score` | Example | GR gate plus harmonic behavior/content fusion | Complete per-example score | Source artifacts |
| `cats_prevalence_score` | Dataset | Arithmetic mean of example CATS scores | All examples only when complete | Yes, column U |
| `cats_decision_balanced_score` | Conflict type | Mean of answerable and refusal-required subgroup CATS | Existing subgroups; only published as complete type score | Source artifacts |
| `cats_balanced_score` | Dataset | Mean of type decision-balanced CATS scores | All represented conflict types | Yes, column V |
| `cats_type_balanced_score` | Dataset diagnostic | Mean of raw type CATS scores | All represented conflict types | Source artifacts |
| `cats_flat_legacy_score` | Dataset diagnostic | Conditional arithmetic mean of component metrics | Components with nonzero applicability | Source artifacts only; not the final metric |

The distinction between a score and its applicability count is essential. For
example, `factual_grounding=0.0` with `fg_n=0` would not mean that every answer
was ungrounded; it would mean that the metric had no applicable denominator and
the implementation's empty-list display value was used. The master audit must
therefore inspect both the score and its `*_n` field.

### A.2 Domains, clipping, and display conventions

The component scores are intended to lie in `[0,1]`. Continuous judge-derived
values are clipped when reconstructed for CATS:

$$
\operatorname{clip}(x;0,1)=\min(1,\max(0,x)).
$$

The evaluator stores unit-interval values in JSON and CSV. Spreadsheet displays
may format them as percentages or decimals, but no metric formula changes when
the display format changes. Counts are integers and must never be treated as
percentages. A paper table should state its scale once, for example "all scores
are percentages; counts are shown in parentheses."

Unless otherwise stated, an empty numerator is not interpreted as evidence of
success. The implementation uses zero for an undefined component ratio and
uses explicit applicability counts to distinguish "zero performance" from "not
applicable". CATS has an additional completeness gate described in Appendix F.

## Appendix B. Grounded-Refusal Formula Contract

### B.1 Per-example labels

Let `gold_answerable_i` be the benchmark label after the precedence rules in
Section 5. Let `pred_answered_i` be the parser output after think-trace removal.
Then:

$$
g_i = \mathbf{1}[\hat A_i=A_i]
     = \begin{cases}
       1 & \text{if } \hat A_i=A_i,\\
       0 & \text{otherwise.}
       \end{cases}
$$

This is exactly `gr_accuracy_from_flags`. A correct answer contributes one; a
wrong answer, an answer on a refusal-required item, or a refusal on an
answerable item contributes zero. The per-example value is not the dataset F1.

### B.2 Confusion-matrix equations

With `p_i=\hat A_i` and `a_i=A_i`, the answer-positive counts are:

$$
\begin{aligned}
TP &= \sum_{i=1}^{N}\mathbf{1}[p_i=1 \land a_i=1],\\
FP &= \sum_{i=1}^{N}\mathbf{1}[p_i=1 \land a_i=0],\\
FN &= \sum_{i=1}^{N}\mathbf{1}[p_i=0 \land a_i=1],\\
TN &= \sum_{i=1}^{N}\mathbf{1}[p_i=0 \land a_i=0].
\end{aligned}
$$

The four cells satisfy:

$$
TP+FP+FN+TN=N.
$$

The implementation uses zero when a precision or recall denominator is zero:

$$
P_{ans}=\begin{cases}\frac{TP}{TP+FP}&TP+FP>0\\0&\text{otherwise}\end{cases}
$$

$$
R_{ans}=\begin{cases}\frac{TP}{TP+FN}&TP+FN>0\\0&\text{otherwise}\end{cases}
$$

$$
F1_{ans}=\begin{cases}
\frac{2P_{ans}R_{ans}}{P_{ans}+R_{ans}}&P_{ans}+R_{ans}>0\\
0&\text{otherwise.}
\end{cases}
$$

Dataset GR accuracy is:

$$
Accuracy_{GR}=\frac{TP+TN}{\max(1,N)}.
$$

The `max(1,N)` guard prevents a crash on an empty input. It does not turn an
empty evaluation into a meaningful positive result; an empty run must be
reported as invalid or incomplete rather than interpreted as an accurate model.

### B.3 Refusal-positive equations

The same four outcomes are reoriented with refusal as the positive class:

$$
TP_{ref}=TN,\quad FP_{ref}=FN,\quad FN_{ref}=FP,\quad TN_{ref}=TP.
$$

Therefore:

$$
P_{ref}=\begin{cases}\frac{TN}{TN+FN}&TN+FN>0\\0&\text{otherwise}\end{cases}
$$

$$
R_{ref}=\begin{cases}\frac{TN}{TN+FP}&TN+FP>0\\0&\text{otherwise}\end{cases}
$$

$$
F1_{ref}=\begin{cases}
\frac{2P_{ref}R_{ref}}{P_{ref}+R_{ref}}&P_{ref}+R_{ref}>0\\
0&\text{otherwise.}
\end{cases}
$$

The implementation also records refusal specificity:

$$
Specificity_{ref}=\frac{TN_{ref}}{TN_{ref}+FN_{ref}}
                   =\frac{TP}{TP+FN}
                   =R_{ans},
$$

when the denominator is nonzero. This identity is a useful audit invariant, not
a fifth independent behavior dimension.

### B.4 Gold-label precedence and parser edge cases

The gold resolution function is applied before all GR equations:

$$
A_i=
\begin{cases}
1-\operatorname{abstain}(expected\_response_i), & \text{if that boolean exists},\\
answerable\_under\_evidence_i, & \text{otherwise if present},\\
\operatorname{fallback\_support\_rule}_i, & \text{otherwise.}
\end{cases}
$$

The fallback supports the historical benchmark schema and can accept partial
support IDs. It is not allowed to override an explicit `expected_response.abstain`
label. This is a data-contract decision: the benchmark's intended action is
more authoritative than a shallow inference from document verdict strings.

For the prediction label, the parser is deliberately conservative about
where refusal phrases may occur. It strips model think traces and recognized
end-of-answer wrappers, treats an empty final answer as refusal, and recognizes
canonical inability/insufficient-evidence openings. A refusal phrase embedded
after a substantive answer is not automatically enough to turn the whole output
into a refusal. Any parser modification changes all six GR workbook columns and
therefore requires a versioned audit.

### B.5 Scientific interpretation

Answer precision answers: "When the system answers, how often was answering
permitted?" Answer recall answers: "Of the answerable questions, how many did
the system answer?" Refusal precision answers: "When the system refuses, how
often was refusal warranted?" Refusal recall answers: "Of the refusal-required
questions, how many did the system correctly decline?" The two orientations are
not redundant in interpretation even though they derive from one matrix.

Accuracy is retained because it is an intuitive overall decision rate, but it is
not sufficient: it can be high when one answerability class dominates. The two
F1 orientations and the full confusion matrix expose conservative and
over-answering failure modes that accuracy can hide.

## Appendix C. Behavior Adherence Committee Contract

### C.1 Construct definition

Behavior Adherence asks whether the response follows the response policy
appropriate to the conflict regime. It is not a generic helpfulness score and
is not a second factual-grounding judge. The active five-type rubric is:

| Type | Conflict regime | Required behavior |
| --- | --- | --- |
| 1 | No Conflict | Give a direct, unambiguous answer without manufacturing alternatives or uncertainty. |
| 2 | Complementary | Reconcile partial information into one coherent answer and retain compatible details. |
| 3 | Conflicting Opinions | Represent disagreement neutrally, identify the competing positions, and avoid presenting one disputed view as uncontested fact. |
| 4 | Outdated Information | Prefer the current evidence, avoid allowing superseded material to control the answer, and acknowledge old information only when useful. |
| 5 | Misinformation | Reject or correct inaccurate evidence and rely on the reliable, verified material rather than repeating falsehoods as fact. |

The prompt explicitly instructs judges not to use answerability correctness,
factual entailment, citation validity, unsupported-claim detection, or STR as
hidden BA criteria. This orthogonality is necessary because otherwise a single
latent "good answer" judgment would be counted in multiple columns.

### C.2 Binary committee vote

Each valid judge returns an adherence label `v_j` and a confidence `c_j`:

$$
v_j=\begin{cases}1&\text{judge says adherent}\\0&\text{judge says not adherent.}\end{cases}
$$

Let `J` be the valid-judge set, excluding timeout, parse, and API-error
responses. For the unweighted majority strategy:

$$
V_1=\sum_{j\in J}v_j,\qquad V_0=|J|-V_1.
$$

$$
b_i^{bin}=\mathbf{1}[V_1>V_0].
$$

The strict `>` means a tie is non-adherent. The raw majority-side and
minority-side confidence fields are:

$$
c_i^{maj}=\frac{\max(V_1,V_0)}{|J|},\qquad
c_i^{min}=\frac{\min(V_1,V_0)}{|J|}.
$$

The unanimous strategy requires:

$$
b_i^{bin}=\mathbf{1}[V_1=|J|],
$$

but still reports majority-side and minority-side fractions so the degree of
disagreement is visible. A unanimous strategy is therefore intentionally
stricter than majority voting.

### C.3 Active weighted-majority equations

The production committee's default path is weighted majority. Each judge has a
configured priority `pi_j` and returns confidence `c_j`. The effective vote
weight is:

$$
w_j=\pi_j\max(c_j,0.01).
$$

The 0.01 floor prevents a valid judge that emits zero confidence from silently
having zero influence. It does not make an erroneous or timed-out response
valid; invalid responses are removed before weights are calculated.

The two weighted sides are:

$$
W_1=\sum_{j\in J:v_j=1}w_j,
\qquad
W_0=\sum_{j\in J:v_j=0}w_j,
\qquad
W=W_1+W_0.
$$

The binary decision and confidence fields are:

$$
b_i^{bin}=\mathbf{1}[W_1>W_0],
$$

$$
c_i^{maj}=\frac{\max(W_1,W_0)}{W},
\qquad
c_i^{min}=\frac{\min(W_1,W_0)}{W},
$$

when `W>0`. A weighted tie is also non-adherent. The rationale stored in the
result is selected from the highest-priority/highest-confidence response on the
winning side; it is explanatory metadata, not an extra score.

### C.4 Continuous committee consensus used by CATS

The binary BA column uses the committee decision. CATS uses the retained degree
of support so that a 2--1 split is not treated identically to a unanimous vote:

$$
b_i=\begin{cases}
\frac{W_1}{W_1+W_0}, & W_1+W_0>0,\\
\frac{V_1}{V_1+V_0}, & \text{otherwise if raw votes exist},\\
b_i^{bin}, & \text{otherwise if only the binary score exists},\\
- & \text{if no usable BA result exists}.
\end{cases}
$$

The implementation clips reconstructed values to `[0,1]`. The final `-` case
means the example cannot receive the complete CATS formula and is counted in
`cats_unscorable_n`; it is not silently imputed as zero.

### C.5 Committee failure behavior

If at least one judge returns a valid response, only valid responses participate
in the vote. If all judges fail, the fallback stores `all_failed=True`, zero
confidence, and no positive support. A caller must preserve this failure in the
audit trail. Treating "all judges failed" as a confident non-adherent judgment
would conflate infrastructure failure with model behavior.

### C.6 Scientific defense and risks

The policy-specific rubric gives BA construct validity: the score corresponds to
the intended conflict-handling behavior rather than generic stylistic quality.
The committee reduces dependence on one judge and the continuous consensus
retains disagreement information. Priority weighting is defensible only if the
paper states how priorities were chosen and does not present the weighted vote
as human ground truth. Sensitivity analysis should report the binary decision,
consensus distribution, valid-judge counts, and, where practical, unweighted
versus weighted results.

## Appendix D. Factual Grounding Contract

### D.1 Claim extraction is part of the metric

FG is not computed from every token. It is computed from the claims produced by
the deterministic extractor. Let `C_i` be the extracted claim list for example
`i`, with `N_i=|C_i|`. Each claim is a pair:

$$
c_{ik}=(text_{ik},D^{cite}_{ik}),
$$

where `D^cite` is the set/list of cited document IDs associated with the claim.

The extractor's operational sequence is:

1. Protect periods in initials, decimals, domains, and known abbreviations.
2. Split the answer into sentences using the safe tokenizer.
3. Extract bracketed/grouped citations, parenthetical document references, and bare document references before citation removal.
4. Deduplicate document IDs while preserving first-seen order.
5. Remove citations from the claim text.
6. Drop empty strings, citation-only fragments, and meta-reference statements about the answer or sources rather than substantive propositions.
7. Retain normal claims only when they have at least four tokens and at least one content-like token of length five or more.
8. Strip attribution wrappers and drop a claim if it becomes empty or shorter than four tokens.
9. Cap the output at `max_claims_per_answer=5`.
10. If a concise lead claim is immediately followed by a cited supporting sentence, inherit the neighboring citation when the deterministic lead-claim predicate allows it.
11. If no normal claim survived, promote deterministic terse candidates of at most eight tokens; uncited terse candidates longer than three tokens are not promoted.

The extractor therefore defines the observable FG target. Two systems that say
the same content with different sentence or citation formatting can receive
different claim units, which is a limitation that should be acknowledged.

### D.2 Eligible documents

The benchmark's document annotation is used as the eligibility filter. A
document is eligible if its normalized verdict is one of:

$$
E=\{supports, support, partially\ supports, partial\ support,
partially\_supports\}.
$$

Documents marked irrelevant or otherwise non-positive are not allowed to become
supporting evidence in the FG denominator. If `E` is empty, every extracted
claim receives zero support and the per-example FG is zero, while all claims
remain in the denominator.

### D.3 Per-claim committee support

For claim `k`, the committee returns a supporting-document set `D^sup_{ik}`,
a cross-document flag `x_{ik}`, and, when applicable, a cross-document
combination `D^combo_{ik}`. A claim is counted as grounded if either a cited
single supporting document exists or a cited cross-document combination exists:

$$
y_{ik}=\mathbf{1}\left[
\left(D^{cite}_{ik}\cap D^{sup}_{ik}\neq\varnothing\right)
\lor
\left(x_{ik}=1\land D^{cite}_{ik}\cap D^{combo}_{ik}\neq\varnothing\right)
\right].
$$

The claim-level reason taxonomy is:

| Reason | Meaning |
| --- | --- |
| `single_doc_cited` | At least one cited document was identified as supporting. |
| `cross_doc_cited` | The claim requires multiple documents and at least one cited combination document was identified. |
| `no_supporting_doc_found` | The committee found neither single-document nor cross-document support. |
| `supporting_doc_not_cited` | Support was found but the cited set misses it. |
| `cross_doc_not_cited` | Cross-document support was found but the answer cited none of the identified combination documents. |
| `not_supported` | No applicable support condition passed. |
| `committee_error` | The FG committee call failed for that claim. |
| `no_eligible_docs` | No positively annotated evidence document existed for the example. |

FG does not award partial credit for a partially supporting document, does not
weight a claim by judge confidence, and does not add a separate contradiction
penalty in the active v2 path. Positive and partial-positive gold verdicts are
eligible evidence; the committee's role is to determine whether the specific
claim is supported by that evidence and whether the model cited it.

### D.4 Committee support threshold

For each claim, let `J_i` be the valid FG judges and let `pi_j` be their integer
priority. The valid priority mass and threshold are:

$$
P_i=\sum_{j\in J_i}\max(1,\pi_j),
\qquad
\tau_i=\frac{P_i}{2}.
$$

If more than one judge is valid, a positive document must also be named by at
least two valid judges. If exactly one judge is valid, one positive judge is
enough:

$$
m_i=\begin{cases}2&|J_i|>1\\1&|J_i|=1.\end{cases}
$$

For document `d`, with `V_{id}` the number of valid judges naming it and
`P_{id}` their priority mass:

$$
d\in D^{sup}_{ik}\iff P_{id}>\tau_i\land V_{id}\ge m_i.
$$

Cross-document support uses the analogous condition. If `X_i` is the priority
mass of judges setting `cross_doc_support=true` and `U_i` is their count:

$$
x_{ik}=\mathbf{1}[X_i>\tau_i\land U_i\ge m_i].
$$

When `x_{ik}=1`, the returned combination IDs are the union of the combinations
reported by positive cross-document judges. This union identifies the evidence
that the model must have cited; it is not a claim that every document in the
union independently entails the claim.

### D.5 Per-example and dataset equations

For `N_i>0`:

$$
f_i=FG_i=\frac{\sum_{k=1}^{N_i}y_{ik}}{N_i}.
$$

If no claims were extracted, the implementation returns `supported_claims=0`,
`total_claims=0`, and `FG_i=0.0`. The example is still FG-applicable whenever
the evaluator's `factual_grounding_applicable` flag is true; this is a genuine
zero claim score, not an omitted denominator. The dataset metric is an
example-macro mean:

$$
FG_{dataset}=\frac{1}{N_{FG}}\sum_{i\in I_{FG}}f_i,
$$

where `I_FG` is the set of examples whose FG flag is true and `N_FG=fg_n`.
This is not a pooled claim micro-average such as
`sum supported claims / sum total claims`. The macro choice prevents examples
with many extracted sentences from dominating the run-level metric.

### D.6 Scientific defense and limitations

FG measures citation-linked evidential support, which is the operational RAG
property needed for claims about grounded generation. Requiring both semantic
support and a citation prevents an answer from receiving full credit merely by
being true in the abstract while failing to link its statement to retrieved
evidence. The positive-document filter prevents irrelevant retrievals from
being treated as evidence.

The metric does not establish that the retrieval corpus is complete, that every
claim is factually true outside the provided evidence, or that a citation is
human-readable. It is also sensitive to claim segmentation, citation syntax,
document annotation quality, and the judge committee. These limitations are
reasons to report `fg_n`, claim counts, error reasons, and component results,
not reasons to hide FG behind a single aggregate.

## Appendix E. Single-Truth Recall Contract

### E.1 Applicability and gold-answer normalization

STR is active only when both conditions hold:

$$
\operatorname{STRApplicable}_i=
\mathbf{1}[gold\_answers_i\neq\varnothing]
\cdot
\mathbf{1}[type_i\in\{1,2,4,5\}].
$$

Type 3 is excluded in the active configuration because conflicting opinions do
not have one stable single truth to recover. This is a construct boundary, not a
missing-data accident.

Gold answers are normalized into a list `G_i` as follows:

- `None` becomes an empty list;
- a nonempty string becomes a one-item list;
- a list/tuple contributes each nonempty item after string conversion;
- another scalar becomes one string item if nonempty.

If `G_i` is empty, or if the candidate answer is empty, STR returns zero with no
exact or partial matches. Such an example should normally be non-applicable
when the gold answer is absent; the explicit zero behavior is retained for
schema safety.

### E.2 Gold-answer judgment

For each gold answer `g` in `G_i`, the committee receives the gold answer and
the model answer and returns an adherence-like decision. Let `e_{ig}` be one
when the committee says the answer semantically contains the gold truth. Exact
match means committee acceptance, not string identity:

$$
e_{ig}=\mathbf{1}[\operatorname{committee\_accepts}(g,answer_i)].
$$

Paraphrases and semantically equivalent formulations are therefore eligible.
Mentioning a gold phrase while rejecting it should not count as recovery; the
prompt asks whether the answer actually asserts the target truth.

If the primary decision is negative, partial credit is available only when the
negative decision is uncertain enough. Let `m_{ig}` be the minority-side
confidence, specifically the support for the "gold truth is present" side:

$$
p_{ig}=\mathbf{1}[e_{ig}=0\land m_{ig}\ge 0.30].
$$

The threshold is applied to minority confidence, not majority confidence. Thus
a confident 3--1 negative vote does not receive partial credit merely because
the majority confidence is high; a genuinely split committee can qualify.

### E.3 Per-example STR formula

Let:

$$
E_i=\sum_{g\in G_i}e_{ig},\qquad
P_i=\sum_{g\in G_i}p_{ig},\qquad
M_i=|G_i|.
$$

The active implementation is:

$$
r_i=STR_i=\begin{cases}
0, & M_i=0\text{ or candidate is empty},\\
1, & E_i>0,\\
\min\left(1,\frac{0.5P_i}{M_i}\right), & E_i=0\land P_i>0,\\
0, & \text{otherwise.}
\end{cases}
$$

The `min(1,...)` guard is retained even though `0.5P_i/M_i` cannot exceed 0.5
when `P_i<=M_i`; it makes the intended bounded range explicit and protects the
formula if the match-count representation changes.

Examples:

| Gold answers | Exact matches | Qualified partial matches | STR |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 1.00 |
| 1 | 0 | 1 | 0.50 |
| 2 | 0 | 1 | 0.25 |
| 2 | 0 | 2 | 0.50 |
| 2 | 0 | 0 | 0.00 |
| 2 | 1 | any | 1.00 |

### E.4 Dataset equation and interpretation

For the applicable set `I_STR`:

$$
STR_{dataset}=\frac{1}{N_{STR}}\sum_{i\in I_{STR}}r_i,
\qquad N_{STR}=str_n.
$$

This is a macro-average over examples, so an example with several gold
answers does not automatically dominate the run. It is semantic target
recovery, not token overlap, BLEU-style matching, or citation grounding.

The metric is particularly useful for No Conflict, Complementary, Outdated, and
Misinformation cases where a stable intended answer exists. Its exclusion of
Type 3 avoids pretending that a disputed proposition has one canonical answer.
The paper must state this domain restriction and should not compare STR values
across runs with materially different `str_n` without showing the counts.

## Appendix F. Applicability, Completeness, and Aggregation Edge Cases

### F.1 Per-example applicability table

The active evaluator's intended applicability logic is:

| Example state | BA | FG | STR | Answer Quality | CATS content inputs |
| --- | --- | --- | --- | --- | --- |
| Correct refusal on refusal-required item | No | No | No | No | Decision only |
| Answer on answerable item | Yes | Yes | If gold answer and type in `(1,2,4,5)` | If FG exists; fuses STR if present | BA consensus plus Q when available |
| Wrong refusal on answerable item | Yes/diagnostic path | Yes path may yield zero claims | Depends on configured gold/type path | Usually zero/derived from available fields | GR gate forces zero |
| Wrong answer on refusal-required item | Yes | Yes path may yield zero or unsupported claims | Usually not a valid single-truth target | Derived only from applicable fields | GR gate forces zero |

The precise serialized applicability flags are authoritative for a stored run.
The table explains the intended construct boundaries; it should not be used to
overwrite a per-sample flag during post-processing.

### F.2 Count equations

For each metric `M` with applicability flag `app_{i,M}`:

$$
n_M=\sum_{i=1}^{N}\mathbf{1}[app_{i,M}=1].
$$

The workbook fields are:

$$
behavior_n=n_{BA},\quad fg_n=n_{FG},\quad str_n=n_{STR},
\quad answer\_quality_n=n_Q.
$$

If a component has no applicable values, its empty aggregate is represented as
zero in the in-memory summary for schema stability, while the count remains
zero. Paper tables should display `N/A` or an explicit dash when the count is
zero rather than presenting that zero as a measured failure rate.

### F.3 Answer Quality construction

For each example, the implementation first constructs answer quality from the
available content metrics:

$$
q_i=\begin{cases}
\sqrt{f_i r_i}, & f_i\text{ and }r_i\text{ are available},\\
f_i, & f_i\text{ is available and }r_i\text{ is not},\\
- & f_i\text{ is unavailable}.
\end{cases}
$$

The geometric mean is:

$$
G(f_i,r_i)=\exp\left(\frac{\log f_i+\log r_i}{2}\right)
           =\sqrt{f_ir_i},
$$

for strictly positive inputs, with the implementation's zero handling:

$$
G(x_1,\ldots,x_k)=0
\quad\text{if any }x_j\le0.
$$

This makes zero performance on either content pillar prevent the fused content
score from being positive. It does not mean STR is available for all conflict
types; when STR is structurally unavailable, FG is the available content
quality input and the denominator is recorded accordingly.

### F.4 CATS completeness gates

For a bucket `B` (overall or one conflict type), let `n_B` be its example count,
`u_B` its unscorable count, and `z_B` its empty/zero-scored count. The active
implementation defines:

$$
cats\_scored\_n_B=n_B-u_B,
$$

$$
cats\_complete_B=
\mathbf{1}[u_B=0\land cats\_scored\_n_B=n_B\land n_B>0].
$$

The arithmetic example mean is:

$$
C_B=\frac{1}{cats\_scored\_n_B}\sum_{i\in B,\ scored}s_i,
$$

with an implementation display value of zero if the score list is empty. The
published prevalence score is only exposed when complete:

$$
CATS\text{-}Prevalence_B=\begin{cases}C_B&cats\_complete_B=1\\N/A&\text{otherwise.}\end{cases}
$$

At the overall level, completeness additionally requires every represented
conflict type to have at least one scored example. This prevents a missing type
from silently disappearing from the balanced average.

### F.5 Refusal-required and answerable subgroup means

Within bucket `B`, define:

$$
C_B^A=\frac{1}{|I_B^A|}\sum_{i\in I_B^A}s_i,
\qquad
C_B^R=\frac{1}{|I_B^R|}\sum_{i\in I_B^R}s_i,
$$

where `I_B^A` and `I_B^R` are the gold-answerable and refusal-required scored
subsets. A subgroup with no examples has no score, not a zero score. The
decision-balanced type score is:

$$
C_B^{DB}=\begin{cases}
\frac{C_B^A+C_B^R}{2},&C_B^A\text{ and }C_B^R\text{ both exist},\\
C_B^A,&\text{only }C_B^A\text{ exists},\\
C_B^R,&\text{only }C_B^R\text{ exists},\\
N/A,&\text{neither exists}.
\end{cases}
$$

The active implementation publishes this type score only when the type is
complete, although it retains the subgroup values as diagnostics.

### F.6 Per-example hierarchical CATS formula

Decision correctness is a gate:

$$
s_i=\begin{cases}
g_i, & \text{correct refusal path},\\
g_i H(b_i,q_i), & b_i\text{ and }q_i\text{ exist},\\
g_i b_i, & b_i\text{ exists but }q_i\text{ does not},\\
g_i q_i, & q_i\text{ exists but }b_i\text{ does not},\\
g_i, & \text{neither content input exists}.
\end{cases}
$$

For positive inputs, the two-input harmonic mean is:

$$
H(b_i,q_i)=\frac{2}{1/b_i+1/q_i}
           =\frac{2b_iq_i}{b_i+q_i}.
$$

The implementation returns zero if either input is nonpositive. Therefore:

$$
g_i=0\Longrightarrow s_i=0,
$$

and a low behavior consensus cannot be fully rescued by high Answer Quality,
nor can high behavior consensus rescue low content quality. The gate avoids
cube-root inflation of weak content scores and reflects the task hierarchy: an
incorrect answer/refusal action is a fundamental failure before quality of the
chosen action is considered.

### F.7 Dataset-level CATS formulas

For the complete example set:

$$
CATS\text{-}Prevalence=\frac{1}{N}\sum_{i=1}^{N}s_i.
$$

This preserves the empirical distribution of conflict types and answerability.
For `T` conflict types, the decision-balanced summary is:

$$
CATS\text{-}Balanced=\frac{1}{T}\sum_{t=1}^{T}C_t^{DB},
$$

where every represented type must have a complete decision-balanced score. The
older raw equal-type diagnostic is:

$$
CATS\text{-}TypeBalanced=\frac{1}{T}\sum_{t=1}^{T}C_t,
$$

and is retained for audit comparison but is not the canonical workbook
`final_cats_balanced` value.

### F.8 Legacy flat average, exactly as retained

The historical flat diagnostic is conditional on metric availability. Let `x_GR`
be dataset GR F1 for the overall legacy summary and dataset GR accuracy for a
per-type legacy summary. Then:

$$
L_B=\operatorname{mean}\left(
\left[x_{GR}\right]
\cup
\left[BA_B\;\text{if }behavior_n>0\right]
\cup
\left[FG_B\;\text{if }fg_n>0\right]
\cup
\left[STR_B\;\text{if }str_n>0\right]
\right).
$$

The legacy result is retained only to make historical comparisons auditable. It
is not the final CATS metric because it mixes dataset-level and conditional
component summaries, gives each available column equal weight without a
construct-level justification, and can change its effective dimensionality when
applicability changes.

## Appendix G. Worked Numerical Examples

These examples are deliberately small and are intended to make the formulas
checkable by a reader without running the repository.

### G.1 GR example

Suppose the benchmark has 10 items and the model produces:

$$
TP=4,\quad FP=1,\quad FN=2,\quad TN=3.
$$

Then:

$$
P_{ans}=4/5=0.80,
\quad R_{ans}=4/6\approx0.667,
$$

$$
F1_{ans}=\frac{2(0.80)(0.667)}{0.80+0.667}\approx0.727,
$$

$$
Accuracy_{GR}=(4+3)/10=0.70.
$$

For refusal-positive scoring:

$$
P_{ref}=3/(3+2)=0.60,
\quad R_{ref}=3/(3+1)=0.75,
$$

$$
F1_{ref}=\frac{2(0.60)(0.75)}{0.60+0.75}\approx0.667.
$$

The two F1 values expose different failure directions despite sharing the same
10 examples.

### G.2 BA committee example

Assume three valid judges with priorities `(2,1,1)`, confidences
`(0.90,0.80,0.60)`, and votes `(adherent, adherent, not adherent)`. Their
weights are `(1.80,0.80,0.60)`, so:

$$
W_1=2.60,\quad W_0=0.60,\quad b_i=2.60/3.20=0.8125.
$$

The binary result is adherent because `2.60>0.60`, but CATS uses `0.8125`
rather than one. If the votes were `(adherent, not adherent, not adherent)`
with the same weights, the result would still be adherent because `1.80>1.40`,
but consensus would fall to `1.80/3.20=0.5625`. This illustrates why the
continuous value preserves uncertainty even when the binary decision remains
positive.

### G.3 FG example

Suppose an answer yields three extracted claims. The committee identifies:

- claim 1: cited document intersects supporting documents -> `y=1`;
- claim 2: support exists but the cited set misses it -> `y=0`;
- claim 3: two documents jointly support it and one of the combination documents is cited -> `y=1`.

Then:

$$
f_i=(1+0+1)/3=0.667.
$$

If the example is FG-applicable, it contributes `0.667` to the example-macro
dataset mean. A claim with a partially supporting document still contributes
only one binary outcome: support-and-citation passes or it does not.

### G.4 STR example

Suppose an example has two gold answers and no exact committee matches. One gold
has minority confidence `0.35` and the other `0.20`. Only the first qualifies:

$$
E_i=0,\quad P_i=1,\quad M_i=2,
\qquad r_i=0.5(1/2)=0.25.
$$

If either gold had an exact semantic match, `r_i` would be 1.0 rather than 0.25.

### G.5 Answer Quality and CATS example

Suppose:

$$
f_i=0.80,\quad r_i=0.50,\quad b_i=0.75,\quad g_i=1.
$$

First:

$$
q_i=\sqrt{0.80\cdot0.50}=\sqrt{0.40}\approx0.6325.
$$

Then:

$$
H(b_i,q_i)=\frac{2(0.75)(0.6325)}{0.75+0.6325}
\approx0.686.
$$

Therefore:

$$
s_i=1\cdot0.686=0.686.
$$

If the GR decision were wrong, `g_i=0` and the same BA/FG/STR values would
produce `s_i=0`. If the item were a correct required refusal, BA, FG, STR, and
Answer Quality would be non-applicable and the CATS contribution would be
`s_i=g_i=1` under the current decision-only refusal policy.

## Appendix H. Dependency Graph and Audit Invariants

### H.1 Metric dependency graph

The metric stack is not a collection of independent spreadsheet formulas:

```text
model final answer
  -> think-trace stripping and refusal parser
  -> predicted answerability ----------------------+
                                                   |
gold answerability precedence ---------------------+--> GR labels/confusion matrix
                                                   |
claims + citations -> BA committee ----------------+
                  -> FG claim/document committee --+--> applicable component means
gold answers + type -> STR committee --------------+       |
                                                          +--> Answer Quality
BA consensus + Answer Quality + GR gate -----------------+--> per-example CATS
per-example CATS + conflict type + decision class ----------> CATS summaries
```

This graph explains why changing refusal parsing, claim extraction, gold
answerability, or committee configuration can change multiple workbook columns.
It also explains why the paper should report component metrics before the
aggregate.

### H.2 Recommended invariants for every run

The following checks are suitable for an independent audit script:

1. `n == len(per_sample)` for the evaluated benchmark slice.
2. Every per-sample record has a conflict type in the expected set `{1,2,3,4,5}`.
3. `TP + FP + FN + TN == n` for the reconstructed GR confusion matrix.
4. `0 <=` every unit-interval metric `<= 1` after parsing.
5. `behavior_n`, `fg_n`, `str_n`, and `answer_quality_n` equal the number of true applicability flags.
6. `answer_quality_n <= fg_n` because Answer Quality requires FG.
7. `str_n` excludes Type 3 under the active configuration.
8. `cats_unscorable_n == 0` and `cats_complete == true` for a publishable complete run.
9. Every represented type has `cats_complete=true` before an overall balanced score is accepted.
10. `cats_prevalence_score` is null/N/A rather than a fabricated number when the overall completeness gate fails.
11. The source artifact records `cats_aggregate_version == cats_h_gated_harmonic_v1` for the current master.
12. The workbook row count is exactly 108 and the four fixed comparison rows use their latest fixed counts.
13. The authoritative scope audit is the 108-row audit; any historical 114-row verifier output is reconciled against the six documented out-of-scope artifacts before interpretation.

### H.3 Recalculation pseudocode

The following is a language-neutral specification for reproducing the active
summary without relying on spreadsheet cell formulas:

```text
for each sample:
    g = int(pred_answered == gold_answerable)
    if BA applicable:
        store binary BA and continuous committee support b
    if FG applicable:
        store f = supported_claims / total_claims
    if STR applicable:
        store r using exact-match/qualified-partial rules
    if f exists:
        q = sqrt(f * r) if r exists else f
    if correct_refusal:
        s = g
    else if b and q exist:
        s = g * harmonic_mean(b, q)
    else if b exists:
        s = g * b
    else if q exists:
        s = g * q
    else:
        s = g

per-example means -> per-type means
per-type answerable/refusal subgroup means -> decision-balanced type means
all-example mean -> CATS-Prevalence when complete
mean of complete type decision-balanced means -> CATS-Balanced when complete
```

This pseudocode is intentionally explicit about the order of operations. In
particular, fusing already averaged dataset metrics would not reproduce the
active CATS result.

## Appendix I. Non-Operative or Non-Active Options That Must Not Be Claimed

The repository contains historical or configurable pathways that should not be
described as active parts of the 108-row master metric unless a run explicitly
records and audits them:

- `require_inline_citations` is stored as a configuration option but is not an
  independent hard constraint in the active FG-v2 formula. Citation linkage is
  assessed claim by claim through the cited-document intersection rule.
- The older NLI-based `enhanced_factual_grounding` function has additional
  parameters such as contradiction handling, minimum entailment confidence,
  cross-document requirements, and optional partial credit. Those options are
  not the committee FG-v2 formula used for the current master matrix.
- The active FG-v2 formula does not award graded partial credit for a partially
  supporting document and does not add a separate contradiction penalty.
- Correct refusal wording quality is not scored in current CATS. A correct
  required refusal receives decision-only CATS contribution `g_i=1`. This is a
  deliberate current scope decision and must not be described as a refusal
  quality evaluation.
- The legacy flat average remains for provenance only. It must not be presented
  as `final_cats_prevalence` or `final_cats_balanced`.
- `cats_type_balanced_score` is an older equal-type diagnostic. The canonical
  balanced workbook value is the decision-balanced type mean
  `cats_balanced_score`.
- Human-evaluation agreement, Cohen's kappa, and LLM--human agreement are
  separate validation analyses. They do not silently replace the local
  committee's BA, FG, or STR values in the 108-row master workbook.

## Appendix J. Paper-Ready Methods Language

The following paragraph is a faithful concise description that can be expanded
in a paper's methods section:

> We evaluate each generated response along four separable dimensions. Grounded
> Refusal is a binary answer-versus-abstain decision evaluated against the
> benchmark's precedence-resolved answerability label; we report answer-positive
> and refusal-positive precision, recall, and F1, together with decision
> accuracy. Behavior Adherence is a conflict-type-specific policy judgment made
> by a multi-model committee, and Factual Grounding is the fraction of extracted
> claims for which the model cites a committee-identified supporting document,
> including eligible cross-document support. Single-Truth Recall is a semantic
> committee judgment of whether the response asserts each applicable gold answer,
> with partial credit only for genuinely split committee decisions. Component
> means use their own applicability denominators. For a secondary aggregate, we
> first combine Factual Grounding and Single-Truth Recall geometrically when both
> apply; we then combine continuous Behavior consensus and this Answer Quality
> score harmonically, and gate the result by grounded-refusal decision correctness.
> CATS-Prevalence averages complete per-example scores, whereas CATS-Balanced
> gives equal weight to the five conflict regimes after balancing answerable and
> refusal-required subgroups within each regime. We report both aggregates only
> as structured summaries; all substantive conclusions are anchored in the
> component metrics and their denominators.

## Appendix K. Formula Completeness Checklist

Before locking a paper table or regenerating the master workbook, verify that
the accompanying methods text or supplement contains all of the following:

- predicted answer/refusal parser definition and preprocessing;
- gold answerability precedence;
- per-example GR equality formula;
- TP, FP, FN, TN definitions;
- answer precision, recall, F1, and accuracy formulas;
- refusal-positive reorientation and F1 formulas;
- refusal specificity identity, if reported;
- all BA rubric definitions for Types 1--5;
- majority, weighted-majority, and unanimous vote equations;
- priority/confidence weight equation and tie rule;
- valid-judge filtering and all-failure behavior;
- binary BA versus continuous CATS consensus distinction;
- claim extraction, citation extraction, filtering, inheritance, and cap;
- eligible-document verdict set;
- single-document and cross-document FG support condition;
- FG committee priority threshold and corroboration threshold;
- FG reason taxonomy and no-claim/no-eligible-document behavior;
- per-example and dataset FG macro formulas;
- STR applicability set and gold-answer normalization;
- STR semantic exact-match and minority-confidence partial-match formulas;
- STR per-example and dataset formulas;
- every applicability count and its denominator;
- Answer Quality geometric formula and missing-STR branch;
- harmonic mean formula, GR gate, and CATS fallback branches;
- refusal-required decision-only CATS rule;
- type-level, prevalence, decision-balanced, and canonical balanced formulas;
- legacy flat formula and explicit non-primary status;
- completeness gate and missing-type handling;
- 108-row scope and latest fixed comparison provenance;
- limitations, uncertainty reporting, and non-operative configuration flags.

If any item above changes in code, the aggregate version, methodology note, and
master-results audit should be updated together. That coupling is part of the
reproducibility protocol, not merely documentation hygiene.
