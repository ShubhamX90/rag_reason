# CATS Aggregate Logic

This note defines the current paper-facing aggregate for the CATS v2 benchmark. It documents the mathematics, implementation contract, scientific rationale, denominator rules, reviewer-facing limitations, and the exact conditions under which an aggregate is complete.

The active implementation version is `cats_h_gated_harmonic_v1`. The paper-facing master results scope contains exactly 108 experiment rows; broader local result directories are not automatically part of this scope.

The aggregate is a secondary summary. The primary scientific results remain grounded-refusal performance, Behavior Adherence, Factual Grounding, and Single-Truth Recall. A scalar is useful for compact comparison, but it must not hide the dimensions that explain a system's behavior.

Implementation references:

- [rag_eval/evaluator.py](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/evaluator.py:120)
- [rag_eval/conflict_eval.py](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/conflict_eval.py:93)
- [rag_eval/judge_prompts.py](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/rag_eval/judge_prompts.py:40)
- [scripts/audit_cats_master_results.py](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/scripts/audit_cats_master_results.py:1)
- [scripts/update_master_results_workbook.py](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/scripts/update_master_results_workbook.py:1)

## 1. Executive Summary

The revised aggregate, called **CATS-Harmonized** in the implementation, follows this sequence:

1. Determine whether the model made the correct answer-versus-refusal decision.
2. For answerable examples, fuse factual grounding and single-truth recall into an answer-quality score.
3. Use the decision result as a hard gate.
4. Combine continuous committee behavior consensus with answer quality using a harmonic mean.
5. For refusal-required examples, use the grounded-refusal decision-correctness score only; answer-content metrics are not applicable.
6. Average complete example-level scores first within each conflict type.
7. Report both a prevalence-sensitive summary and a type-balanced summary.

The core answerable-example equation is:

$$
s_i = g_i \frac{2b_iq_i}{b_i+q_i},
$$

where $g_i$ is binary decision correctness, $b_i$ is continuous committee support for Behavior Adherence, and $q_i$ is answer quality. If $g_i=0$, then $s_i=0$ regardless of downstream quality.

For a refusal-required example, the current aggregate uses:

$$
s_i = g_i.
$$

This is an explicit scope choice: the aggregate measures whether the system made
the correct answer-versus-refusal decision, while refusal wording quality is not
included because the historical result files do not contain a consistently
audited refusal-quality judgment. Refusal-required and answerable subgroup
scores are still reported separately so this asymmetry remains visible.

## 2. Why the Revision Was Necessary

The former score was a flat arithmetic mean of available marginal metrics. That design was easy to compute, but it was vulnerable to several legitimate criticisms:

- It treated decision correctness, behavior, grounding, and recall as flat peers even though recall is a conditional answer-content property.
- It allowed strong performance on one dimension to compensate too easily for a critical failure on another.
- It did not make the refusal-quality claim: a correct refusal is intentionally scored by decision correctness only because that is the available reproducible signal.
- It used a cube-root geometric mean at the top level. When the binary decision and behavior terms were one, an answer-quality value of `0.25` became `0.63`, which can make weak content appear stronger than its underlying score.
- It converted committee disagreement into a hard binary behavior label, so a 2--1 vote and a unanimous vote could contribute identically.
- It did not make answerable-versus-refusal composition visible within each conflict type.

The revision addresses these issues while preserving the benchmark's conceptual hierarchy and the primary component metrics.

## 3. Plain-Language Interpretation

The score asks a strict but understandable question:

> On this particular example, did the system make the right decision, handle the conflict appropriately, and produce content of adequate quality for the evidence regime?

For an answerable example, the system must first pass the answer/refusal decision gate. After that, behavior and answer quality must both be good. The harmonic mean prevents one of those two dimensions from dominating the other.

For a refusal-required example, the answer text is not evaluated as if it should contain a fact. The current aggregate records only whether the required refusal decision was correct. Refusal wording quality is outside this version of CATS and must not be inferred from the decision score.

The score is not a claim that one number captures every aspect of RAG quality. It is a structured secondary summary of joint success.

## 4. Benchmark Structure

The benchmark contains five conflict regimes. In the 736-example benchmark used for the local committee evaluation, the type counts are:

| Conflict type | Regime | Count | Answerable | Refusal-required |
| --- | --- | ---: | ---: | ---: |
| 1 | No Conflict | 211 | 154 | 57 |
| 2 | Complementary Information | 221 | 176 | 45 |
| 3 | Conflicting Opinions or Research Outcomes | 109 | 96 | 13 |
| 4 | Outdated Information | 158 | 145 | 13 |
| 5 | Misinformation | 37 | 37 | 0 |
| **Total** |  | **736** | **608** | **128** |

The distribution is not perfectly answerability-balanced. Types 1 and 2 contain materially more refusal-required examples than Types 3--5, while Type 5 contains none. This is why the repository records answerable and refusal-required subgroup counts and why the balanced score must expose its handling of these subgroups rather than hiding it.

### 4.1 Master experiment scope

The paper-facing hierarchical master matrix contains exactly 108 experiment
rows:

- 96 standard benchmark runs;
- 6 answer-only SFT runs;
- 2 comparison runs from `other_techniques` for the available Llama results;
- 4 redone comparison runs from `other_techniques_fixed` for Mistral and Qwen across CoT few-shot and CoN.

The unfixed Mistral/Qwen comparison JSONs and staged collection JSONs remain
available locally for provenance, but are explicitly out of scope for the
108-row master matrix. The four fixed comparison runs are the authoritative
latest versions used in the master workbook. Their latest applicability counts
are retained, including 651 Behavior/FG examples for fixed CoT Mistral and 698
for fixed CoT Qwen; the fixed CoN Mistral and Qwen runs have 736 each.

## 5. Notation

For example $i$:

- $g_i \in \{0,1\}$: answer/refusal decision correctness;
- $b_i \in [0,1]$: continuous committee support for behavior adherence;
- $f_i \in [0,1]$: factual grounding score;
- $r_i \in [0,1]$: single-truth recall score;
- $q_i \in [0,1]$: fused answer-quality score;
- $s_i \in [0,1]$: final example-level CATS-Harmonized score.

The ordinary Behavior Adherence metric remains binary for direct reporting:

$$
\mathrm{BA}^{\mathrm{binary}}_i \in \{0,1\}.
$$

The aggregate additionally uses committee consensus. For three equally weighted judges, examples can therefore contribute values such as $0$, $1/3$, $2/3$, or $1$. With weighted voting, the stored weighted support fraction is used:

$$
b_i = \frac{W_i^{+}}{W_i^{+}+W_i^{-}},
$$

where $W_i^{+}$ and $W_i^{-}$ are the committee's weighted adherent and non-adherent vote totals. If weighted totals are unavailable, the raw vote fraction is used. If no vote details exist, the stored binary score is used only as a compatibility fallback and is auditable as such.

This distinction is deliberate: the paper can report binary Behavior Adherence as the easily interpretable primary component, while the secondary scalar retains information about committee disagreement.

## 6. Step 1: Decision Correctness

The first primitive is the per-example grounded-refusal decision signal:

$$
g_i=
\begin{cases}
1, & \text{if the model answered when answerable or refused when refusal was required},\\
0, & \text{otherwise.}
\end{cases}
$$

The composite uses per-example decision correctness rather than dataset-level F1 because the composite is built before averaging. Dataset-level grounded-refusal F1, precision, recall, and abstention diagnostics remain primary reported metrics.

This is important because F1 is not decomposable: a per-example F1 value is not a meaningful primitive that can be fused independently with the other dimensions.

## 7. Step 2: Answer Quality

Answer quality is hierarchical. Factual Grounding and Single-Truth Recall are not treated as four unrelated flat peers with decision and behavior.

$$
q_i=
\begin{cases}
\sqrt{f_i r_i}, & \text{when both Factual Grounding and STR apply},\\
f_i, & \text{when only Factual Grounding applies},\\
\text{not applicable}, & \text{when the example requires refusal}.
\end{cases}
$$

The geometric mean is retained inside answer quality because grounding and target recovery are jointly necessary aspects of answer content. A high recall answer that is unsupported should not be perfect, and a well-grounded answer that omits the target truth should not be perfect.

This is a different role from the top-level combination. The revision does not use the top-level cube-root geometric mean because that transformation was too forgiving for weak nonzero answer-quality values.

## 8. Step 3: Answerable-Example Score

For an answerable example, the revised score is:

$$
\boxed{s_i=g_i H(b_i,q_i)}
$$

where:

$$
H(b_i,q_i)=\frac{2b_iq_i}{b_i+q_i},
$$

and $H=0$ if either input is zero.

Because $g_i$ is binary, it is a gate:

$$
g_i=0 \Longrightarrow s_i=0.
$$

The score behaves as follows:

| $q_i$ | Old $q_i^{1/3}$ when $g=b=1$ | New $H(1,q_i)$ |
| ---: | ---: | ---: |
| 0.125 | 0.500 | 0.222 |
| 0.250 | 0.630 | 0.400 |
| 0.500 | 0.794 | 0.667 |
| 0.720 | 0.896 | 0.837 |

The new score remains high when both dimensions are high, but it does not cosmetically inflate a weak answer-quality score. It is also familiar to readers because it has the same balancing interpretation as an F-score: a high result requires both inputs to be high.

If a legacy row lacks answer quality but has an applicable behavior score, the implementation uses the available behavior path with the decision gate. This branch is mainly for compatibility with old or non-answer-content runs; standard answerable benchmark rows have answer-quality output.

## 9. Step 4: Refusal-Required-Example Score

For a gold refusal-required example, the current aggregate is deliberately
decision-only:

$$
\boxed{s_i=g_i}
$$

The model receives one point when it correctly refuses and zero when it answers
an unanswerable question. Factual Grounding, Single-Truth Recall, answer
quality, and Behavior Adherence are marked not applicable for a correct refusal
because there is no answer content to score. This keeps the denominator and the
reproducibility of the historical evaluation intact.

This choice has a known limitation: two correct refusals receive the same CATS
example score even if one is better justified or written. That limitation is
preferable to introducing a new judge-dependent metric into old runs without
uniform annotations. If refusal-quality annotations are added in a future
version, they should be reported as a separate diagnostic or used only in a
versioned successor aggregate, not silently mixed with these results.

## 10. Why Harmonic Mean Instead of a Top-Level Geometric Mean?

The earlier top-level formula was:

$$
(g_i b_i q_i)^{1/3}.
$$

When $g_i=b_i=1$, this reduces to $q_i^{1/3}$. That root transformation is mathematically valid, but it expands moderate and low scores. A response with answer quality `0.25` became `0.63`, which is difficult to defend as a faithful representation of answer quality.

The revised design separates two conceptual roles:

- decision correctness is foundational and therefore gates the score;
- behavior and answer quality are jointly necessary after the gate and therefore use the harmonic mean.

The decision-only refusal branch is documented separately rather than being
silently treated as an answer-quality score.

The harmonic mean is not claimed to be uniquely correct. It is selected because it is transparent, symmetric between behavior and answer quality, zero-sensitive, and less inflationary than the cube-root construction. The paper should call it a principled design choice, not a learned or universally optimal law.

## 11. Conflict-Type Aggregation

For each conflict type, first compute answerable and refusal-required subgroup means from the complete example scores:

$$
T_t^A=\frac{1}{|I_t^A|}\sum_{i\in I_t^A}s_i,
\qquad
T_t^R=\frac{1}{|I_t^R|}\sum_{i\in I_t^R}s_i.
$$

The decision-balanced type score is:

$$
T_t^{\mathrm{DB}}=
\begin{cases}
\frac{T_t^A+T_t^R}{2}, & \text{if both subgroups are present},\\
T_t^A, & \text{if only answerable examples are present},\\
T_t^R, & \text{if only refusal-required examples are present}.
\end{cases}
$$

This prevents a type's score from being dominated by whichever decision class happens to be more common. Type 5 currently has only answerable examples, so its available subgroup is used and the report states that no refusal subgroup exists.

The implementation also retains the raw equal-type diagnostic `cats_type_balanced_score`, which averages the unbalanced type means. The canonical `cats_balanced_score` uses the decision-balanced type means once all examples have complete CATS-Harmonized scores.

## 12. Dataset-Level Summaries

### 12.1 CATS-Prevalence

The prevalence-sensitive summary is:

$$
\mathrm{CATS\text{-}Prevalence}=\frac{1}{N}\sum_{i=1}^{N}s_i.
$$

Equivalently, it weights each type by its empirical count. It answers:

> How well does the system perform under the benchmark's observed mixture of conflict types and answerability regimes?

It is useful as a deployment-like summary when the benchmark distribution is intended to be representative.

### 12.2 CATS-Balanced

The canonical balanced summary is:

$$
\mathrm{CATS\text{-}Balanced}=\frac{1}{5}\sum_{t=1}^{5}T_t^{\mathrm{DB}}.
$$

It gives each conflict regime equal top-level importance and balances answerable/refusal-required examples within each regime when both are present.

It answers:

> How well does the system perform across the five conflict regimes when no regime is allowed to dominate because it has more examples?

### 12.3 Incomplete score status

The final summaries are valid when every example has a computable example-level
score and every conflict type represented in the run has at least one scored
example. Correct refusals satisfy this condition through their decision score;
they do not require a separate refusal-quality artifact.

This is preferable to reporting a numeric result whose denominator is silently different from the benchmark.

## 13. Applicability Rules

The applicability rules are gold-defined and output-independent wherever possible:

- Decision correctness applies to every example.
- Behavior Adherence applies to answered examples. Correct refusals have no answer content for the Behavior Adherence component and contribute to CATS through decision correctness only.
- Factual Grounding applies to answer-content cases; correct refusals have no answer claims to ground.
- Single-Truth Recall applies only when a single-truth target exists under the configured gold annotation.
- A wrong refusal is not a correct refusal and is not rescued by an inapplicable answer-quality path; its decision gate is zero.
- A malformed or empty answer is not allowed to escape evaluation by becoming conveniently not applicable.

The purpose is to distinguish genuinely undefined dimensions from missing evidence and from model failure.

## 14. Orthogonality of Behavior

Behavior Adherence must not double-count the other metrics. The behavior prompts now state that behavior judges must not independently reward or penalize:

- whether the model answered or refused correctly;
- factual entailment or citation validity;
- the presence of the gold answer;
- Single-Truth Recall;
- generic unsupported factual claims except where they are part of the refusal contract.

For answerable examples, behavior concerns conflict-conditioned response policy: reconciliation, neutral presentation of disagreement, temporal prioritization, misinformation handling, or directness in the no-conflict case.

Refusal wording quality is outside the current CATS aggregate. It must not be
implicitly folded into the ordinary Behavior Adherence score or reconstructed
from the binary decision label.

This separation is necessary before combining the terms. Otherwise the aggregate would count the same property twice under different names.

## 15. Why Not a Hand-Weighted Average?

A hand-weighted score would be:

$$
S=w_1x_1+w_2x_2+w_3x_3+w_4x_4,
\qquad \sum_jw_j=1.
$$

That design raises immediate questions:

- Why those weights?
- Were they selected before inspecting results?
- Do they represent human utility, task structure, or desired rankings?
- Are the metrics sufficiently independent to justify flat weighting?

CATS-Harmonized does not claim to eliminate normative choices. It makes them explicit and structural:

- STR is nested within answer quality;
- decision correctness is a gate;
- behavior and answer quality are jointly necessary;
- type balancing is reported separately from empirical prevalence;
- primary metrics remain visible.

This is more defensible than choosing four coefficients after observing model performance, while still acknowledging that the hierarchy itself is a design decision.

## 16. Worked Examples

### 16.1 Strong answer

Let $g_i=1$, $b_i=0.8$, $f_i=0.81$, and $r_i=0.64$. Then $q_i=\sqrt{0.81\cdot0.64}=0.72$, and:

$$
s_i=H(0.8,0.72)=\frac{2(0.8)(0.72)}{0.8+0.72}\approx0.759.
$$

The result is high because both behavior and answer quality are high, but it is below either dimension's maximum.

### 16.2 Weak answer quality despite correct decision

Let $g_i=1$, $b_i=1$, and $q_i=0.25$. Then:

$$
s_i=H(1,0.25)=0.4.
$$

The score does not turn weak answer content into `0.63` merely because the binary decision and behavior labels are correct.

### 16.3 Wrong decision

Let $g_i=0$, $b_i=1$, and $q_i=0.8$. Then:

$$
s_i=0\cdot H(1,0.8)=0.
$$

Good downstream text cannot rescue the foundational answer/refusal error.

### 16.4 Correct refusal under the current scope

Let $g_i=1$ for a gold refusal-required example. Then:

$$
s_i=g_i=1.
$$

The aggregate records correct abstention. The quality of the refusal wording is
not part of this version and should not be inferred from the perfect decision
score.

### 16.5 Committee disagreement

With three equally weighted judges, two adherent votes and one non-adherent vote produce $b_i=2/3$. This is distinct from a unanimous adherent result $b_i=1$. The binary Behavior Adherence component can still report the majority as adherent, while the aggregate preserves the disagreement.

## 17. Reviewer Attacks and Defensible Answers

### Attack: “Correct refusals get free perfect credit.”

Response: This is a real limitation and is not hidden. The current aggregate
uses decision correctness only for refusal-required examples because that signal
is available consistently across the historical runs. We do not claim that the
scalar measures refusal wording quality; the refusal-required subgroup and
grounded-refusal diagnostics expose the scope of the result.

### Attack: “The cube root inflates weak answer quality.”

Response: The top-level cube-root formula was replaced with a decision gate plus harmonic Behavior/Answer-Quality fusion. The paper should show the sensitivity analysis described in Section 19.

### Attack: “A single binary committee vote is brittle.”

Response: Binary Behavior Adherence remains a primary report for interpretability, but the aggregate uses the stored continuous committee support fraction and retains individual vote details for audit.

### Attack: “Answerability composition is hidden within each type.”

Response: Each type now reports answerable and refusal-required subgroup counts and means. The canonical balanced score averages those subgroup means when both exist.

### Attack: “The aggregate is still arbitrary.”

Response: It is a designed metric, not a law of nature. Its choices are pre-specified, hierarchical, transparent, and not tuned to model rankings. Component metrics remain primary, and both prevalence and balanced summaries are reported.

### Attack: “Why not use dataset-level GR F1 inside the scalar?”

Response: F1 is not decomposable at example level. It is therefore reported as a primary component, while per-example decision correctness is used as the composite gate.

## 18. Recommended Paper Reporting

The paper should use this hierarchy.

Primary results:

- grounded-refusal F1, precision, recall, and abstention diagnostics;
- binary Behavior Adherence;
- Factual Grounding;
- Single-Truth Recall.

Secondary summaries:

- CATS-Balanced;
- CATS-Prevalence.

Diagnostics:

- five conflict-type scores;
- answerable versus refusal-required subgroup scores;
- committee vote-consensus distribution;
- answerable versus refusal-required coverage and subgroup counts;
- paired bootstrap confidence intervals where model differences are interpreted.

Recommended wording:

> We treat CATS as a multidimensional evaluation framework and report its component metrics directly. As a secondary summary, we compute a hierarchical example-level aggregate. Answerable examples use decision-gated harmonic fusion of continuous committee behavior consensus and answer quality; refusal-required examples contribute grounded-refusal decision correctness only. We report both CATS-Balanced, which equalizes the five conflict regimes and balances answerability within each regime where possible, and CATS-Prevalence, which reflects the benchmark's empirical distribution.

The aggregate should not be described as the sole or primary performance criterion.

## 19. Validation and Sensitivity Checks

Before locking paper numbers, run and archive:

1. Completeness checks confirming that every example and represented conflict type has a computable score.
2. Per-type and per-answerability-class counts.
3. Paired bootstrap intervals over examples, with the type-balanced resampling scheme documented separately.
4. Rank comparison between CATS-Balanced, CATS-Prevalence, and each component metric.
5. A sensitivity appendix comparing the revised formula with the legacy flat average and the prior cube-root hierarchy.
6. A leave-Type-5-out diagnostic because Type 5 has only 37 examples.
7. A check that Behavior Adherence prompts remain orthogonal to factual and decision metrics.

The sensitivity results should be used to show robustness, not to choose whichever formula produces the preferred ranking.

## 20. Limitations

The revised aggregate is stronger, but it remains a designed summary:

- Harmonic fusion encodes joint competence and is not the only possible choice.
- The benchmark's type distribution and answerability composition remain empirical design choices.
- Type 5 is small, so tiny differences in its score should not be overinterpreted.
- LLM committee judgments are not equivalent to human gold labels; committee uncertainty is retained but does not solve evaluator validity by itself.
- Refusal wording quality is not represented in this CATS version. If it becomes a paper claim, it requires a separately designed and auditable annotation study.

These limitations should be stated plainly. They do not invalidate the aggregate; they define the scope of what it supports.

## 21. Implementation Contract

New per-sample results should contain:

- `gr_accuracy`;
- `behavior_score` and `behavior_applicable`;
- `behavior_consensus_score`;
- `factual_grounding_score` and applicability;
- `single_truth_recall_score` and applicability;
- `gold_answerable`, `pred_answered`, and `correct_refusal`;
- full committee details, including weighted vote totals and individual responses.

The aggregate exposes:

- `cats_prevalence_score`;
- `cats_balanced_score`;
- `cats_type_balanced_score` diagnostic;
- `cats_answerable_score`;
- `cats_refusal_required_score`;
- `cats_scored_n`;
- `cats_unscorable_n`;
- `cats_complete`.

`null` final CATS values mean that an example or represented conflict type had
no computable score, not that a correct refusal lacked a refusal-quality judge.
Reports and workbooks must render this distinction explicitly.

## 22. Final Recommendation

Use **CATS-Balanced** as the first secondary aggregate column, with
**CATS-Prevalence** alongside it. Keep the four component metrics primary and
visible, and state explicitly that refusal-required examples are decision-only
in this aggregate.

Do not use the legacy flat average as a paper-facing headline. Do not publish CATS-Harmonized values from result files whose `cats_complete` flag is false.
