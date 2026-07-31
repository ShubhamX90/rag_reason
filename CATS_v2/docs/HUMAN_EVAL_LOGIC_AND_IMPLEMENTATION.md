# Human Evaluation: Logic, Study Design, and Implementation

**Status:** Current logical and implementation-level description of the CATS v2
human-evaluation pipeline and active reviewer study.

**Purpose:** This document explains how human evaluation is constructed,
normalized, assigned, collected, sanitized, consolidated, analyzed, and
compared with the local LLM committee. It is intended for ACL-paper methods
writing, artifact review, future reviewer-file processing, and scientifically
defensible interpretation of agreement metrics.

**Implementation level:** The discussion describes data contracts, workflow
stages, metric units, formulas, and audit decisions without reproducing every
source-code function. The executable package is under
[`exports/cats_human_eval_cli/`](../exports/cats_human_eval_cli/), and the
reviewer-facing instructions are in
[`REVIEWER_USER_MANUAL.md`](../exports/cats_human_eval_cli/REVIEWER_USER_MANUAL.md).

## 1. Executive Summary

The CATS human-evaluation pipeline is a separate validation layer for the
judgment-sensitive parts of CATS. It allows human reviewers to assess the same
broad constructs used by the local LLM committee:

- conflict-conditioned Behavior Adherence (BA);
- citation-linked Factual Grounding (FG); and
- Single-Truth Recall (STR).

Grounded Refusal (GR) remains deterministic. The human package does not ask
reviewers to re-decide the benchmark's gold answerability target. It uses the
stored answerability annotation and normalized model answer to determine
whether a sample is a correct refusal and whether answer-content metrics apply.

The pipeline has five conceptual layers:

1. **Study construction:** choose a balanced, auditable sample of model-output
   variants and create a standalone study bundle.
2. **Record normalization:** freeze the query, answer, retrieved documents,
   annotations, extracted claims, citations, answerability, and applicability
   fields shown to reviewers.
3. **Reviewer collection:** assign each sample to reviewers, capture
   metric-specific judgments with autosave and revision history, and distinguish
   drafts from submitted returns.
4. **Sanitization and consolidation:** verify reviewer identity, assignment
   membership, submission status, revisions, duplicates, and coverage before
   counting any return.
5. **Agreement analysis:** compute human-human, human-committee,
   human-consensus, and committee-internal agreement on explicitly defined
   units and coverage subsets.

The current study contains:

- 350 selected sample-variant records;
- 700 target review slots;
- 650 accepted submitted human judgments;
- 300 samples with two submitted human reviews;
- 50 samples with one submitted review and incomplete second-review coverage;
- 9 additional drafts retained for audit but excluded from submitted counts; and
- four registered reviewers: Atharv, Manan, Parth, and Samyek.

The primary human-human analyses therefore use the complete 300-sample
double-reviewed subset. The 350-sample pool and 650 human-committee
alignments are reported separately as supplementary or coverage-qualified
analyses.

## 2. Role of Human Evaluation in CATS v2

### 2.1 Validation, not silent replacement

The local committee remains the evaluator used for the 108-row CATS master
matrix. Human judgments do not silently overwrite committee metrics or change
the CATS aggregate for those experiments.

Human evaluation asks:

- Do human reviewers interpret the conflict-conditioned behavior rubric
  consistently?
- Do humans agree on citation-linked claim support?
- Do humans agree with the local committee on aligned units?
- Which metric dimensions are more stable across evaluators?
- Are disagreements concentrated in particular conflict types, prompts, model
  variants, or training regimes?

The appropriate interpretation is validation of the measurement layer, not
proof that the committee is equivalent to human truth.

### 2.2 Human and committee roles

| Construct | Human role | Local committee role |
| --- | --- | --- |
| GR | Deterministic applicability and decision context; not a free-form human judgment | Deterministic benchmark computation; not a committee vote |
| BA | Binary adherence label, confidence category, and rationale | Binary adherence label, model confidence, rationale, weighted committee decision |
| FG | Select eligible documents supporting each extracted claim; optionally identify a two-document combination | Select supporting IDs and cross-document combinations using priority-based corroboration |
| STR | Binary judgment of whether the answer asserts the gold target | Semantic binary judgment per gold item, with committee-side partial-match sensitivity |
| CATS | Validation-oriented component and agreement analysis | Primary local evaluation components and secondary CATS aggregates |

The human interface does not ask reviewers to reproduce LLM priority weights.
Human reviewers are treated as equally important annotators in human-human
agreement analyses unless a separate, explicitly justified design states
otherwise.

## 3. Current Study Version and Provenance

### 3.1 Active study identity

The current study is:

```text
qwen_llama_e2e_sft_baseline_balanced_4reviewers
```

Its study directory is:

```text
exports/cats_human_eval_cli/studies/
  qwen_llama_e2e_sft_baseline_balanced_4reviewers/
```

The selected source-row manifest is:

```text
exports/cats_human_eval_cli/studies/
  qwen_llama_e2e_sft_baseline_balanced_4reviewers__selected_source_rows.jsonl
```

The builder records selection seed `20260715`. The study manifest records
metrics version `cats_human_eval_cli_v0_1` and the CATS v2 deterministic claim
extraction version.

### 3.2 Source family

The study samples from:

```text
inputs/prepped_model_eval_inputs/benchmark_set_all_modes/
  <model>/e2e/<prompt>/<train_type>/input.jsonl
```

The selected dimensions are:

- models: `qwen7b`, `llama8b`;
- prompt modes: `minimal`, `runtime`, `strict`;
- training types: `baseline`, `sft`; and
- evaluation family: `e2e` only.

This is a selected human-validation slice, not the complete 736-example
benchmark across all 108 experiment variants.

### 3.3 Version boundaries

The following changes require a new study version or explicit provenance label:

- source model-output files;
- selected cells or sampling seed;
- correct-refusal exclusion rule;
- claim extraction logic or human-study claim cap;
- displayed document pool or gold annotations;
- reviewer identities, capacities, or pair quotas;
- reviewer instructions or rubric wording;
- consolidation acceptance rules;
- human-consensus rule;
- agreement unit or missingness rule; or
- human-versus-committee alignment rule.

## 4. Study Selection Design

### 4.1 The 12 source cells

The selected pool is formed from:

\[
2\text{ models}\times3\text{ prompts}\times2\text{ train types}=12\text{ cells}.
\]

| Model | Prompt modes | Train types |
| --- | --- | --- |
| `qwen7b` | `minimal`, `runtime`, `strict` | `baseline`, `sft` |
| `llama8b` | `minimal`, `runtime`, `strict` | `baseline`, `sft` |

Oracle variants and other model families are not part of this human study.

### 4.2 Selected sample count and balancing

The target pool is 350 unique sample-variant records. Ten cells contribute 29
records and two contribute 30:

- `qwen7b|minimal|baseline`: 30;
- `llama8b|runtime|sft`: 30; and
- each other cell: 29.

The resulting distribution is:

| Dimension | Distribution |
| --- | --- |
| Model | `llama8b=175`, `qwen7b=175` |
| Train type | `baseline=175`, `sft=175` |
| Prompt | `minimal=117`, `runtime=117`, `strict=116` |
| Conflict type | 70 examples each for Types 1, 2, 3, 4, and 5 |

The one-example prompt difference is unavoidable when dividing 350 across
three prompt modes.

### 4.3 Correct-refusal exclusion

During pool construction, records identified as deterministic correct refusals
are excluded before selection and assignment. The current study manifest reports
zero correct refusals among the 350 selected records.

This means the current human study does not estimate the quality of correct
refusal wording or rationale. The exclusion must be disclosed because it makes
the human-study population different from the full benchmark.

### 4.4 Base-sample duplication control

The same underlying benchmark sample can appear in multiple model/prompt/train
variants. The builder stores both a composite sample-variant ID and the
underlying base ID.

Selection limits repeated base IDs within conflict strata where possible. Type 5
allows up to two occurrences because of pool availability and balancing
constraints. Assignment scoring also penalizes showing the same base ID to the
same reviewer more than once. The final assignment audit records residual
reuse.

## 5. Reviewer Assignment Design

### 5.1 Reviewers and capacities

| Reviewer | Assigned capacity | Submitted in current snapshot |
| --- | ---: | ---: |
| Atharv | 200 | 200 |
| Manan | 200 | 200 |
| Parth | 200 | 200 |
| Samyek | 100 | 50 |

The target assignment total is:

\[
200+200+200+100=700\text{ review slots}.
\]

### 5.2 Two assigned reviewers per selected sample

The design target is:

\[
350\text{ samples}\times2=700\text{ assignment slots}.
\]

The assignment audit records these pair quotas:

| Reviewer pair | Target |
| --- | ---: |
| Atharv / Manan | 83 |
| Atharv / Parth | 84 |
| Atharv / Samyek | 33 |
| Manan / Parth | 83 |
| Manan / Samyek | 34 |
| Parth / Samyek | 33 |

Actual pair counts match the recorded targets in the current assignment audit.

### 5.3 Assignment balancing dimensions

The builder allocates quotas over:

- conflict type;
- model family;
- train type; and
- prompt mode.

Largest-remainder allocation supplies per-reviewer targets. A deterministic seed
and a scoring function penalize quota overflow and repeated base IDs.

This is a balancing mechanism, not a random-effects statistical model. It
reduces obvious coverage imbalance but does not eliminate dependence among
variants derived from the same underlying benchmark question.

### 5.4 Sample identity

Each selected record receives:

```text
<model>__<prompt>__<train_type>__<base_sample_id>
```

The composite ID is the primary identity used for assignment, storage, return
files, consolidation, coverage, and committee alignment. The base ID is retained
for selection and duplication audits.

## 6. Study-Bundle Construction and Normalization

### 6.1 Standalone bundle principle

The human package can be copied or distributed independently. Its logical
bundle contains:

```text
study/
  study.yaml
  data/samples.jsonl
  assignments/assignments.json
  assignments/<reviewer>_sample_ids.txt
  state/judgments.sqlite3
  state/events.jsonl
  exports/
  admin/
  reviewer_returns/
  consolidated/
```

The packaged reviewer bundle can have a different outer folder name, but these
data roles remain the same.

### 6.2 Normalization order

For each source record, the study builder:

1. identifies the source record and stable base ID;
2. selects the canonical model-output field;
3. preserves a separate raw output when available;
4. splits and removes visible think traces for judging;
5. resolves gold answerability;
6. detects answer versus refusal;
7. marks deterministic correct refusal when applicable;
8. merges retrieved documents with per-document notes;
9. determines FG-eligible documents;
10. extracts claims and visible citations;
11. determines STR applicability; and
12. writes the frozen normalized record to `data/samples.jsonl`.

Reviewers judge this normalized representation. They do not rerun extraction or
choose alternative source files.

### 6.3 Gold answerability precedence

Answerability is resolved in this order:

1. explicit `expected_response.abstain`, if present;
2. `answerable_under_evidence`, if present; and
3. fallback from positively supporting document notes.

An explicit abstention label takes precedence over inference from document
verdicts, preserving benchmark supervision.

### 6.4 Model-answer normalization

The visible final answer is obtained after think-trace removal. The same answer
is used for answer/refusal detection, claim extraction, BA context, FG context,
and STR context.

Empty output is treated as refusal. The refusal detector recognizes the
package's canonical insufficient-evidence, inability-to-answer, and
inability-to-determine forms.

### 6.5 Human-study claim extraction cap

The human package uses the shared deterministic claim-extraction logic but calls
it with a maximum of **12 claims per answer**. The active local benchmark YAML
uses a maximum of **8 claims per answer**.

This is a material protocol difference:

- human FG can display up to 12 extracted claims;
- current local benchmark FG uses up to 8; and
- the human and committee denominators can differ for answers with more than
  eight extracted claims.

The difference must be disclosed in human-versus-committee comparisons. The
human package is aligned in extraction logic, but not strictly identical in
claim-cap behavior.

### 6.6 Document merging and FG eligibility

Retrieved document fields are merged with per-document notes using document ID.
The normalized document context retains document ID, title/snippet, source,
date/timestamp, verdict, key fact, and quote when present.

Only verdicts equivalent to `supports` or `partially supports` enter the
FG-eligible pool. Ineligible documents remain provenance/context but cannot be
selected as human FG support.

## 7. Human Judgment Logic

### 7.1 Common principles

Reviewers receive query, conflict label, conflict reason, model answer,
retrieved documents, per-document notes, and a target answer when applicable.
Metric pages are separated so factual correctness does not silently change a
BA label and behavior impressions do not silently change FG support.

### 7.2 Behavior Adherence

The reviewer supplies a binary `adherent` label, confidence category, and short
rationale. The policy is conflict-conditioned:

| Type | Required behavior |
| ---: | --- |
| 1 | Answer directly without inventing alternatives or unnecessary uncertainty. |
| 2 | Reconcile complementary partial answers into one coherent answer. |
| 3 | Represent conflicting opinions or outcomes neutrally. |
| 4 | Prioritize current information and optionally acknowledge outdated evidence. |
| 5 | Reject misinformation and rely on reliable, verified information. |

Human BA is not a factual-correctness judgment. A reviewer can mark BA
adherent while identifying an FG or STR problem separately.

### 7.3 Human confidence categories

| Level | Label | Stored value |
| ---: | --- | ---: |
| 1 | Low | 0.25 |
| 2 | Medium-low | 0.50 |
| 3 | Medium-high | 0.75 |
| 4 | High | 1.00 |

Confidence is descriptive in the current human-human IAA. It does not give one
reviewer more voting authority than another.

### 7.4 Human Factual Grounding

For every displayed deterministic claim, the reviewer:

1. inspects the claim and model-cited document IDs;
2. selects all eligible documents that actually support the claim;
3. leaves single-document support empty if no one eligible document is
   sufficient;
4. optionally marks exactly two eligible documents as jointly supporting it;
5. enters the two-document combination when applicable; and
6. may add a short note.

The reviewer selects all eligible supporting documents, not merely documents
cited by the model. Citation linkage is evaluated by intersection.

For claim (k), let (C_k) be model citations, (S_k) human-selected single
document support, (X_k) cross-document support, and (D_k^{cross}) the
selected combination:

\[
y_k=1
\quad\text{iff}\quad
(C_k\cap S_k\ne\varnothing)
\;\text{or}\;
(X_k=1\;\text{and}\;C_k\cap D_k^{cross}\ne\varnothing).
\]

The human sample-level FG ratio is:

\[
FG_i^{human}=\frac{\sum_k y_k}{K_i},
\]

where (K_i) is the displayed claim count. No extracted claims produces FG=0
when applicable, with an explicit reason. Correct refusals are inapplicable.

### 7.5 Human Single-Truth Recall

For STR-applicable samples, the reviewer labels whether the answer asserts the
gold target as its own conclusion, then supplies confidence and rationale.
Paraphrases and logical equivalents count; quotation, attribution, possibility
listing, a different answer, and refusal do not.

\[
STR_i^{human}=1[\text{reviewer marks target assertion}].
\]

Humans do not assign the committee's 0.5 partial-match value. That sensitivity
analysis applies only to the committee side.

### 7.6 Correct-refusal interface behavior

Correct refusals automatically receive skipped/inapplicable BA, FG, and STR
fields in the interface. The active study excluded such rows before assignment,
so this branch is present for protocol completeness but did not contribute to
the current selected sample.

## 8. Reviewer Interaction and Persistence

### 8.1 Reviewer-facing flow

The interactive session logically:

1. identifies the reviewer from the registered reviewer list;
2. loads that reviewer's assigned sample IDs in assignment order;
3. shows the sample overview and progress dashboard;
4. permits document browsing before metric judgment;
5. opens BA, FG, and STR pages as applicable;
6. autosaves after metric edits and navigation actions;
7. shows a review summary before submission; and
8. requires all applicable metrics before accepting submission.

The normal launcher is:

```bash
cd exports/cats_human_eval_cli
./run_reviewer.sh
```

The review mode for reopening submitted records is:

```bash
./run_reviewer.sh review
```

### 8.2 Draft versus submitted state

The package distinguishes:

- **draft:** work saved but not finally submitted;
- **submitted:** all applicable fields complete and explicitly submitted.

Viewing a page or saving a draft does not make a judgment count as submitted.
The reviewer must use the explicit submission action.

### 8.3 Revision and active-state semantics

Each reviewer/sample judgment is stored as a revisioned record. When a new
revision is saved, the previous active revision is marked inactive. The active
state represents the latest saved judgment while the revision number and event
log preserve update history.

This prevents stale drafts from competing with later submitted judgments while
retaining an audit trail of changes.

### 8.4 SQLite state and event log

The study state uses:

- SQLite for reviewers, assignments, and revisioned judgments; and
- an append-only JSONL event log for save, submit, and study events.

The database is operational resume state. The event log is the chronological
audit trail. Neither should be manually edited during normal review operation.

## 9. Export and Return-Package Contract

### 9.1 Local export

The package exports active judgments in two forms:

- `active_judgments.jsonl`: operational judgment records;
- `active_judgments_enriched.jsonl`: judgments joined with sample context.

The enriched record includes study name, reviewer ID, composite sample ID,
status, revision, query, conflict metadata, answerability, correct-refusal
status, extracted claims, and the nested judgment payload.

The enriched file is the expected reviewer-return artifact for consolidation.

### 9.2 Return identity checks

A reviewer return is accepted only when:

- the return folder's reviewer directory matches the embedded reviewer ID;
- the sample is assigned to that reviewer;
- status is `submitted` for final metric counts; and
- the row has a valid revision and judgment payload.

The raw return folder is retained as provenance. Sanitization does not overwrite
original reviewer material.

### 9.3 Return inventory

For every reviewer-return directory, consolidation records:

- whether the enriched file exists;
- rows seen;
- status mix;
- embedded reviewer IDs;
- accepted submitted rows;
- accepted drafts; and
- invalid rows.

This matters because a return can contain submitted records and drafts together.

## 10. Sanitization and Consolidation

### 10.1 Consolidation goals

The consolidator does not simply concatenate files. It:

1. loads the assignment map and normalized sample index;
2. enumerates reviewer-return directories;
3. reads each enriched judgment file;
4. attaches organizer metadata;
5. verifies reviewer identity and assignment membership;
6. separates accepted submissions, drafts, and invalid rows;
7. detects non-increasing revisions and cross-return duplicates;
8. retains the best accepted revision for duplicate keys;
9. writes consolidated submitted judgments;
10. writes audit-only drafts and invalid rows; and
11. generates coverage and consolidation summaries.

### 10.2 Acceptance predicate

For a submitted row to count toward final human metrics:

\[
accepted=1[status=\text{submitted}]
\land1[embedded\ reviewer=return\ folder\ reviewer]
\land1[reviewer\in assigned\ reviewers(sample)].
\]

Rows failing this predicate are not silently counted. They are retained in
invalid/audit artifacts where possible.

### 10.3 Duplicate policy

Judgments are keyed by reviewer ID and composite sample ID. If the same key
appears more than once, the higher revision is retained and the duplicate
condition is recorded. A lower or equal revision does not replace a higher
accepted revision.

Duplicate detection covers repeated keys within one file and the same accepted
submitted key across return folders.

### 10.4 Current consolidation snapshot

The current full-receipt consolidation reports:

| Quantity | Count |
| --- | ---: |
| Accepted submitted judgments | 650 |
| Accepted drafts retained for audit | 9 |
| Invalid rows | 0 |
| Duplicate issues | 0 |
| Fully complete samples | 300 |
| Partially complete samples | 50 |
| Unstarted samples | 0 |

Reviewer-specific status is:

- Atharv: 200 submitted of 200 assigned;
- Manan: 200 submitted of 200 assigned;
- Parth: 200 submitted of 200 assigned; and
- Samyek: 50 submitted, 9 drafts, and 50 missing submitted slots out of 100.

The 9 Samyek drafts remain audit material but do not count as final submitted
reviews. The 50 missing submissions create the 50 partial-coverage samples.

## 11. Coverage and Analysis Subsets

### 11.1 Full selected pool

The full pool contains 350 sample-variant records and is useful for selection,
assignment, submitted-coverage, and supplementary analyses. It is not
automatically valid for human-human IAA because 50 records lack two submitted
human judgments.

### 11.2 Complete 300-sample subset

The primary human-human IAA subset contains exactly the 300 samples where both
assigned reviewers submitted accepted judgments.

| Dimension | Distribution |
| --- | --- |
| Model | `llama8b=153`, `qwen7b=147` |
| Prompt | `minimal=99`, `runtime=101`, `strict=100` |
| Train type | `baseline=149`, `sft=151` |
| Conflict type | Type 1: 63, Type 2: 60, Type 3: 58, Type 4: 60, Type 5: 59 |

This frame remains close to balanced but is not perfectly balanced after
coverage loss.

### 11.3 Human-committee alignment frame

Human-versus-committee comparisons align accepted reviewer records to local
committee outputs by composite sample ID. A sample can contribute two
reviewer-level human-committee units.

The current report contains:

- 650 behavior units;
- 465 strict STR units;
- 465 soft STR sensitivity units; and
- 1,029 FG claim-level units.

These counts are not the same as the 300 sample-level human-human frame.

### 11.4 Human-consensus frame

Human-consensus versus committee analysis is restricted to samples where the two
human reviewers agree on the relevant unit. It asks whether the committee agrees
when humans have reached the same label.

Current consensus frames include 256 behavior samples, 194 strict STR samples,
and 428 FG claim units. Consensus filtering improves interpretability but can
retain easier or clearer cases disproportionately.

## 12. Agreement Metrics and Mathematical Definitions

### 12.1 Binary pair unit

For a binary metric, each complete sample contributes:

\[
(a_i,b_i),\qquad a_i,b_i\in\{0,1\}.
\]

The pair exists only when both assigned reviewers have accepted submitted
judgments and the metric is applicable to both. FG claim analysis uses aligned
claim pairs; FG ratio analysis uses paired continuous sample values.

### 12.2 Percent agreement

\[
P_o=\frac{n_{11}+n_{00}}{n},
\]

where (n_{11}) is positive-positive agreement and (n_{00}) is
negative-negative agreement. (n_{10}) and (n_{01}) are the two disagreement
directions.

Percent agreement is interpretable but does not correct for agreement expected
from marginal prevalence.

### 12.3 Cohen's kappa

With positive rates (p_{A1},p_{B1}), define (p_{A0}=1-p_{A1}) and
(p_{B0}=1-p_{B1}). Then:

\[
P_e=p_{A1}p_{B1}+p_{A0}p_{B0},
\]

\[
\kappa=\frac{P_o-P_e}{1-P_e}.
\]

When (1-P_e=0), the implementation returns null rather than inventing a
coefficient.

Kappa must be reported with metric, unit, subset, paired-unit count,
prevalence, and missingness. It is prevalence-sensitive, so raw agreement and
contingency counts should accompany it.

### 12.4 Positive and negative agreement

\[
P_{pos}=\frac{2n_{11}}{2n_{11}+n_{10}+n_{01}},
\qquad
P_{neg}=\frac{2n_{00}}{2n_{00}+n_{10}+n_{01}}.
\]

These supplement kappa when positive and negative labels have different rates
or importance.

### 12.5 Nominal Krippendorff's alpha

For multi-reviewer nominal items, the analysis computes:

\[
\alpha=1-\frac{D_o}{D_e},
\]

where (D_o) is observed nominal disagreement and (D_e) is expected
disagreement from pooled category counts. Null is returned when too few valid
multi-label items exist or expected disagreement is zero.

Alpha is useful as within-human or within-committee multirater context. It is
not an apples-to-apples substitute for pairwise kappa across a two-human and
three-LLM design.

### 12.6 Continuous FG-ratio agreement

For paired ratios (x_i,y_i\in[0,1]), the analysis reports:

\[
exact=\frac{1}{n}\sum_i1[x_i\approx y_i],
\]

\[
MAE=\frac{1}{n}\sum_i|x_i-y_i|,
\qquad
RMSE=\sqrt{\frac{1}{n}\sum_i(x_i-y_i)^2}.
\]

Pearson and Spearman correlations are also reported. Correlation measures
association, not calibration or agreement, so it must be read with MAE/RMSE and
the two means.

### 12.7 Human confidence is not an IAA weight

Reviewers record confidence categories, but the current human-human agreement
script uses binary labels directly. It does not treat a high-confidence
reviewer as more authoritative than a low-confidence reviewer.

## 13. Human-human agreement analysis

Human-human agreement is the primary direct check of whether the task can be
reliably judged by people. It is computed only on samples with the necessary
human observations. The denominator is metric-specific: a sample can have two
valid behavior labels but no applicable STR label, and an FG comparison can be
made at claim level even when the sample-level ratios differ.

### 13.1 The complete-pair frame

The current study assigned two human reviewers to each of 350 selected samples.
The submitted receipts contain two accepted human judgments for 300 samples.
Those 300 samples are the primary human-human agreement frame. The remaining 50
samples have only one submitted human judgment because Samyek completed 50 of
the 100 slots assigned to him. They remain useful for coverage and
human-committee comparisons, but they cannot support a two-human agreement
statistic without imputation, and no imputation is performed.

This distinction is essential:

* 350 is the selected human-evaluation study population;
* 300 is the complete two-human pair frame;
* 650 is the number of accepted submitted human rows in the current receipt
  consolidation;
* 700 is the intended assignment-slot count, not the number of completed
  judgments.

Any table or sentence describing human-human IAA must name its denominator.
Calling a 300-sample human-human result a 350-sample result would overstate the
evidence.

### 13.2 Behavior agreement

For each complete sample, the two human behavior labels are compared as nominal
binary outcomes. The analysis reports percent agreement, Cohen's kappa, and
Krippendorff's alpha. Agreement is also stratified by conflict type, model,
training condition, prompt family, and answerability when the stratum has a
non-empty denominator.

The stratified view matters because a single overall coefficient can conceal
systematic difficulty. In this study, Type 5 is the most important diagnostic
for behavior disagreement: it requires distinguishing evidence that is
individually plausible but collectively incompatible. Type 2 can also expose a
different ambiguity, namely whether a response has properly reconciled
complementary evidence or has merely repeated both sides.

### 13.3 STR agreement

STR agreement is calculated only where STR is applicable. The comparison is
binary: both reviewers either judge that the model asserted the target
single-truth resolution as its own conclusion, or they do not. A refusal,
attribution without endorsement, possibility listing, or a different answer is
not silently converted into partial credit.

The analysis should report the number of paired examples as well as the
coefficient. A high kappa on a small or highly homogeneous subset should not be
presented as equivalent evidence to a high kappa on the full applicable frame.

### 13.4 FG agreement at claim level

FG is evaluated at a finer grain than behavior and STR. For each displayed
claim, the two reviewers' support labels are paired. The claim-level table
reports the number of paired claims, percent agreement, kappa, and alpha. This
is the most direct IAA measure for evidence attribution because it does not
collapse a sample containing many claims into a single binary judgment.

Claim-level and sample-level FG agreement answer different questions:

* claim-level agreement asks whether reviewers agree on individual support
  decisions;
* sample-level ratio agreement asks whether they estimate a similar proportion
  of the displayed answer as supported.

Both are needed. A reviewer pair can agree on most claims while differing in
the set of extracted claims, which changes the denominator of the sample-level
ratio.

### 13.5 FG ratio agreement

For each paired sample, the human FG ratios are compared as continuous values.
Exact agreement uses the repository's configured tolerance rather than
requiring binary floating-point identity. MAE describes average absolute
disagreement in the [0,1] scale; RMSE gives larger disagreements greater
influence. Pearson correlation measures linear association and Spearman
correlation measures rank association. Neither correlation is sufficient by
itself to establish agreement, so it must be reported with error and mean
statistics.

The ratio analysis is intentionally complementary to claim-level kappa. It
captures calibration of the final sample score, while claim-level IAA captures
the underlying annotation decisions.

## 14. Human-committee agreement

The human-committee comparison asks whether the local LLM committee makes
judgments that are consistent with human judgments on the same model outputs.
It is not a test that humans are an absolute gold standard. It is an external
concordance analysis with asymmetric roles: humans provide independent
reference observations and the committee provides the automated evaluator whose
use is being assessed.

### 14.1 Alignment key and eligible rows

Human rows are joined to committee rows through the stable sample identity and
the evaluation context represented by the selected manifest. A join is valid
only when the model, prompt family, training condition, inference setting, and
sample identity refer to the same generated response. A filename match alone
is not sufficient because filenames can be copied across variants.

The analysis retains a row only when both sides have a valid value for the
metric being compared. It does not count missing committee judgments as
disagreements and does not duplicate a committee judgment merely because two
humans reviewed the same sample.

### 14.2 Behavior comparison

Human and committee behavior labels are compared using the same binary
agreement, kappa, and alpha vocabulary as the human-human analysis. The result
should be read with the task's prevalence and the confusion table. Kappa can
become modest when one category is common even when raw agreement is high;
therefore raw agreement and the positive/negative agreement decomposition are
important in the paper.

Behavior is the committee's weakest current agreement dimension. This is
scientifically informative rather than a reason to hide the metric: behavior
requires interpreting the conflict regime, and disagreements concentrate in
cases where evidence must be reconciled rather than merely supported.

### 14.3 STR strict and soft comparisons

The committee stores a strict and a soft STR interpretation. The strict view
requires a direct target assertion. The soft view also accepts an explicitly
equivalent paraphrase or logically equivalent conclusion. Human STR is treated
according to the human rubric, so both committee views are compared separately
to humans.

The paper should not average strict and soft STR into one unexplained score. The
strict result is the conservative operational definition; the soft result is a
sensitivity analysis showing how much agreement changes when equivalent
wording is accepted.

### 14.4 FG comparison

Human and committee FG are compared at both claim level and sample-ratio level.
Claim-level comparison requires a defined pairing of claims. Where the
committee and human claim inventories are not text-identical, the analysis
must document the claim alignment rule and avoid presenting an approximate
alignment as exact annotation identity. Ratio-level comparison is less
dependent on identical claim segmentation and is therefore an important
secondary view.

The current ratio analysis reports exact-with-tolerance agreement, MAE, RMSE,
Pearson correlation, Spearman correlation, and the two means. This combination
distinguishes a committee that tracks human rankings from one that is actually
calibrated to human proportions.

### 14.5 Human consensus versus committee

For a stricter comparison, the two human judgments in the complete 300-sample
frame are first combined using the prespecified human consensus rule, then
compared to the committee. This avoids treating four human-committee rows from
two humans as four independent samples. The consensus analysis is smaller than
the row-level analysis because it requires two accepted human judgments and a
valid committee value.

The paper should report both:

* row-level human-committee concordance, which shows how individual human
  judgments relate to the committee; and
* complete-pair human-consensus concordance, which shows how the committee
  compares with the combined human decision on the strongest common frame.

## 15. Committee-internal agreement

The committee-internal analysis measures whether the three local models agree
with one another on the same generated responses. This is distinct from
human-committee agreement and is necessary for interpreting a committee score
as a stable aggregation rather than a single-model opinion.

### 15.1 The correct frame is complete_300

The primary committee-internal comparison uses the 300 samples for which the
human study has complete two-human coverage. All three committee models have
judgments for this common frame, so the three-model agreement is comparable to
the human consensus analysis.

The all-350 committee result may be retained as a supplementary diagnostic, but
it must never be labeled as the 300-sample committee result. The two frames
answer different questions and can have different type or variant
distributions.

### 15.2 Reconstruction from stored outputs

Committee-internal agreement is computed from stored model judgments and cached
evaluation artifacts, not by rerunning models during report generation. The
reconstruction procedure identifies the model-specific judgment for each
sample, verifies that all three models are present, normalizes labels using the
same metric definitions, and then computes multirater alpha plus pairwise
kappa summaries.

No missing model judgment is filled with the committee majority. A sample enters
the complete_300 committee frame only when the three required model values are
available. This prevents the agreement statistic from becoming circular.

### 15.3 Interpretation

Committee-internal agreement is a reliability diagnostic, not evidence that the
committee is correct. High internal FG agreement with lower behavior agreement
means the models consistently recognize evidence support more readily than
they agree on conflict-conditioned response policy. Human-committee results
must therefore be interpreted metric by metric rather than reduced to one
committee reliability number.

## 16. Current audited results snapshot

The following values are the current consolidated study snapshot. They are
included here to make the analysis contract explicit; generated result files
remain the machine-readable source of truth.

### 16.1 Human-human, complete 300

| Metric | Valid comparison units | Agreement / error | Kappa | Alpha |
|---|---:|---:|---:|---:|
| Behavior | 300 samples | 0.853 agreement | 0.669 | 0.667 |
| STR | 215 samples | 0.902 agreement | 0.773 | 0.772 |
| FG claim support | 477 claims | 0.897 agreement | 0.780 | 0.780 |
| FG ratio | 300 paired samples | exact 0.860; MAE 0.096 | n/a | n/a |

FG-ratio Pearson correlation is 0.807 and Spearman correlation is 0.796 in
this frame.

### 16.2 Human versus committee

| Metric | Valid comparison units | Agreement / error | Kappa |
|---|---:|---:|---:|
| Behavior | 650 submitted rows | 0.745 agreement | 0.407 |
| STR strict | 465 rows | 0.908 agreement | 0.790 |
| STR soft | 465 rows | 0.912 agreement | 0.798 |
| FG claim support | 1,029 claims | 0.899 agreement | 0.783 |
| FG ratio | paired rows | exact 0.865; MAE 0.095 | n/a |

FG-ratio Pearson correlation is 0.805 and Spearman correlation is 0.798.

### 16.3 Human consensus versus committee

On the complete human-pair frame, human consensus versus committee produces:

| Metric | Valid comparison units | Agreement / error | Kappa |
|---|---:|---:|---:|
| Behavior | 256 samples | 0.789 agreement | 0.483 |
| STR strict | 194 samples | 0.948 agreement | 0.876 |
| STR soft | 194 samples | 0.954 agreement | 0.888 |
| FG claim support | 428 claims | 0.949 agreement | 0.888 |

The smaller behavior and STR denominators reflect metric applicability and
valid paired values, not dropped disagreements.

### 16.4 Committee-internal, complete 300

The three-model committee has nominal alpha 0.457 for behavior, 0.678 for
strict STR, and 0.871 for FG claim support. Mean pairwise model kappa is 0.470
for behavior, 0.687 for strict STR, and 0.871 for FG. These values support a
metric-specific interpretation: committee stability is strongest for evidence
grounding, intermediate for target recall, and weakest for nuanced behavior.

## 17. Scientific interpretation and defensible claims

The results support a bounded claim, not an unconditional claim that the local
committee is interchangeable with human judgment.

### 17.1 What the evidence supports

The current evidence supports saying that:

* human reviewers show substantial agreement on behavior and strong agreement
  on STR and FG in the audited common frames;
* the committee agrees strongly with humans on STR and FG under the reported
  operational definitions;
* the committee's behavior judgments have weaker, but still measurable,
  concordance and should be treated as the main residual reliability risk;
* committee-internal agreement mirrors this pattern, with lower stability on
  conflict-conditioned behavior than on evidence support;
* the committee can serve as a scalable evaluator for the reported metrics when
  paired with human calibration, transparent prompts, and metric-level
  diagnostics.

### 17.2 What the evidence does not support

It does not support claiming that:

* committee agreement proves metric validity or causal correctness;
* a single overall IAA value summarizes all CATS behavior;
* the 350 selected samples have complete two-human labels;
* Samyek's 50 submitted rows should be expanded to 100 by imputation;
* strict and soft STR are interchangeable;
* high FG agreement establishes high behavior agreement;
* human labels are error-free or independent of the committee's output.

### 17.3 Why behavior is harder

Behavior asks whether the response follows a conflict-conditioned policy. It can
require recognizing complementary evidence, identifying a true contradiction,
handling temporal supersession, rejecting misinformation, or refusing when
evidence is insufficient. These judgments are not reducible to whether a cited
document entails one sentence. Thus, lower behavior agreement is expected to be
diagnostically meaningful and should motivate rubric refinement and qualitative
review, not selective reporting.

## 18. Qualitative error analysis

Quantitative coefficients should be accompanied by a coded disagreement review.
The recommended procedure is to sample disagreements by metric and conflict
type, inspect the raw response, evidence set, human rationales, and committee
rationales, and assign one primary cause plus optional secondary causes.

Suggested behavior codes are:

* conflict-regime confusion;
* complementary evidence incorrectly treated as contradiction;
* contradiction incorrectly treated as complementarity;
* failure to detect temporal supersession;
* over-crediting a refusal;
* under-crediting a justified refusal;
* unsupported factual content mistakenly treated as behavior-only;
* ambiguous or under-specified rubric boundary.

Suggested FG codes are:

* claim segmentation difference;
* citation versus support-set difference;
* multi-document combination disagreement;
* partial-support threshold disagreement;
* unsupported claim omitted by one reviewer;
* normalization or extraction artifact.

Suggested STR codes are:

* direct assertion versus attribution;
* equivalent paraphrase;
* possibility listing;
* mixed conclusion;
* refusal or non-answer;
* target-answer ambiguity.

The report should present representative examples without exposing private
reviewer information and should distinguish genuine rubric ambiguity from a
mechanical extraction or file-format issue.

## 19. Human-results metric log

The consolidated human-results log should preserve one auditable record for
each reported statistic. At minimum, each record should include:

* study identifier and source manifest;
* receipt or consolidation snapshot identifier;
* reviewer population and accepted-row counts;
* frame definition, including complete-pair status;
* metric name and operational definition;
* unit of analysis: sample, claim, or ratio;
* applicability rule and valid denominator;
* missingness and exclusion counts;
* label normalization version;
* agreement or error statistics;
* confidence interval method and seed, if intervals are generated;
* stratification dimensions and minimum-cell policy;
* interpretation note and known limitation;
* source artifact paths and generation timestamp.

This log prevents a future report from copying a coefficient without its
denominator or from mixing a complete-300 statistic with an all-350 statistic.
It also makes later reviewer receipts appendable without overwriting the
original audit trail.

## 20. Human consensus policy

The current complete frame has exactly two accepted human judgments per sample.
Consensus must therefore be defined before it is used in any paper table.

For binary metrics, the default consensus is agreement when both reviewers
provide the same valid label. If they disagree, the sample is retained for IAA
and disagreement analysis but is not silently resolved by confidence, reviewer
identity, or committee output. A paper-facing consensus score should either
exclude unresolved disagreements with its denominator reported or use a
predeclared adjudication protocol.

For continuous FG ratios, the default consensus summary is the arithmetic mean
of the two human ratios for descriptive analysis, while the individual ratios
remain available for IAA. The mean is not a claim that either reviewer is
correct; it is a transparent summary of the pair.

If an adjudicator is introduced later, adjudicated labels must be stored as a
separate role and version, never substituted into the original reviewer rows.
The report should distinguish raw reviewer agreement from adjudicated final
labels.

## 21. Processing future reviewer files

When another reviewer returns a file, the safe processing sequence is:

1. Preserve the original file byte-for-byte in the receipt area and record its
   checksum.
2. Inspect the archive or JSONL structure without assuming the filename is
   authoritative.
3. Identify the embedded reviewer identity, study identifier, sample keys,
   metric fields, statuses, and revision numbers.
4. Normalize fields into the current study schema while retaining the original
   payload and an audit note for every transformation.
5. Verify that each row belongs to an assigned sample and the declared reviewer
   slot.
6. Accept only submitted rows for the primary results; retain drafts separately
   for audit and completion tracking.
7. Resolve multiple revisions by the explicit revision rule and record the
   discarded versions as duplicates, never by silent replacement.
8. Recompute reviewer inventory, sample coverage, pair coverage, and duplicate
   counts.
9. Recompute agreement on the affected frames and compare with the prior
   metric log.
10. Review changed denominators and any changed stratified cells before
    updating the report.

No future receipt should be merged by simple concatenation. The consolidation
must remain idempotent: processing the same receipt twice produces the same
active judgment set and records the repeated input as an audit event rather
than a new judgment.

## 22. Limitations and safeguards

The current human study has several limitations that should be disclosed.
First, the complete two-human frame is 300 rather than the 350 selected
samples. Second, reviewer completion is unbalanced because Samyek submitted 50
of 100 assigned slots. Third, human reviewers and the committee may share
interpretive conventions because both use the same task materials, so agreement
does not prove independence. Fourth, kappa and alpha depend on prevalence and
category distributions. Fifth, claim-level FG statistics depend on the claim
normalization and pairing policy. Sixth, the human package uses deterministic
claim extraction with `max_claims=12`, whereas the active local benchmark
configuration uses `max_claims_per_answer: 8`; the two pipelines must not be
described as byte-identical.

The main safeguards are explicit denominators, separate complete and incomplete
frames, preserved raw receipts, revision-aware consolidation, metric-specific
agreement statistics, strict versus soft STR sensitivity, committee-internal
checks, and qualitative disagreement analysis.

## 23. ACL-ready methods description

The following prose can be adapted for the paper:

> We evaluated the automated CATS judgments against independent human reviews
> using a preregistered selected-sample manifest and revision-aware receipt
> consolidation. The study selected 350 generated responses across two local
> models, two training conditions, three prompt families, and five conflict
> types. Two human reviewers were assigned to each response; 300 responses had
> two accepted submitted reviews and formed the complete human-human agreement
> frame, while all accepted submitted rows were retained for coverage and
> human-committee analyses. We report metric-specific percent agreement,
> Cohen's kappa, and nominal Krippendorff's alpha for categorical judgments,
> together with exact-with-tolerance agreement, MAE, RMSE, Pearson correlation,
> and Spearman correlation for continuous grounding ratios. STR is reported
> under strict and soft committee interpretations. We separately measure
> three-model committee-internal agreement on the common complete-300 frame.
> Missing judgments are not imputed, confidence is not used as an IAA weight,
> and the five conflict types are retained as explicit strata. Results are
> interpreted as evidence of metric-level concordance and evaluator reliability,
> not as proof that the committee is an infallible substitute for human
> judgment.

## 24. Related artifacts

The implementation and evidence for this protocol are distributed across the
following repository areas:

* `../exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/`
  contains the current study configuration, assignments, receipts, active
  judgments, consolidation, and agreement-analysis outputs.
* `../exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers__selected_source_rows.jsonl`
  is the selected-source manifest.
* `../exports/cats_human_eval_cli/cats_human_eval/` contains the reusable
  package logic for schema validation, normalization, consolidation, and
  analysis.
* `LOCAL_LLM_COMMITTEE_DESCRIPTION.md` explains the automated committee and
  its metric-level role.
* `CATS_METRICS_METHODOLOGY.md` defines the benchmark metrics that human and
  committee judgments are intended to validate.
* `CATS_AGGREGATE_LOGIC.md` documents the separate aggregate-score design and
  why component metrics remain primary.

This document is the logical bridge between the human-review artifacts and the
paper-facing reliability analysis. It should be updated only when the active
study protocol, normalization rules, or reported results change, and every
update should preserve the prior generated artifacts under their dated audit
directory.
