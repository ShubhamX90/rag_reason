# Post-Inference Evaluation: Metrics, Meaning, and Interpretation

## Purpose

This document explains how the repository evaluates generated outputs after inference. The evaluation suite is designed for evidence-grounded RAG reasoning, where a useful response must be more than linguistically plausible. It must also be structurally valid, calibrated about whether evidence suffices, faithful to document-level evidence, aware of conflicts among documents, cited, and useful as a final answer.

The repository therefore does **not** rely on a single aggregate score. It evaluates a sequence of linked properties:

```text
generation coverage
        ↓
response-contract validity
        ↓
answer / refusal calibration
        ↓
document-verdict and conflict-type decisions
        ↓
citation discipline and final-answer overlap
```

This order matters. A high conflict-type accuracy over a small parseable subset is not equivalent to high end-to-end reasoning performance. A good answer-overlap score without valid citations is not fully grounded behavior. Likewise, a model can look accurate on refusal detection by refusing many answerable questions. The reports preserve coverage, confusion matrices, and per-example records to make such trade-offs visible.

This document describes the evaluation logic, not the SFT recipe. See [SFT_DESCRIPTION.md](SFT_DESCRIPTION.md) for training history, [INFERENCE_WORKFLOW.md](INFERENCE_WORKFLOW.md) for generation and sanitization, and [PROMPT_DESIGN_AND_ABLATIONS.md](PROMPT_DESIGN_AND_ABLATIONS.md) for prompt/oracle conditions.

## 1. Evaluation inputs and outputs

Every standard trace-text evaluation compares two aligned artifacts:

| Input | Role |
| --- | --- |
| Canonical benchmark JSONL | Gold retrieved documents, per-document notes, conflict type, answerability/refusal state, and gold answer where available |
| Generated JSONL | One model continuation per benchmark ID, normally the sanitized derivative while raw generations remain preserved |

The standard post-inference evaluation job produces four report families plus a per-example final-answer record.

| Report | Main question | Typical output file |
| --- | --- | --- |
| Contract | Did the output obey the public trace and answer contract? | `contract.json` |
| Document verdicts | Did it assess each document correctly? | `doc_verdicts.json` |
| Conflict type | Did it classify the evidence relationship correctly? | `conflict_type.json` |
| Final answer | Did it refuse/answer appropriately, cite documents, and overlap with usable gold answers? | `final_answer.json` |
| Per-example final answer | Which individual predictions explain an aggregate result? | `final_answer_per_id.jsonl` |

The standard evaluator consumes the sanitized JSONL. This does not replace raw generations: raw and sanitized files are kept separately so that any effect of normalization can be audited. Sanitization can canonicalize a usable trace but does not create a missing trace or invent missing reasoning. See [INFERENCE_WORKFLOW.md](INFERENCE_WORKFLOW.md) for the exact boundary.

## 2. The evaluation hierarchy

| Layer | Core metrics | What it establishes | Why it must be read first or alongside later metrics |
| --- | --- | --- | --- |
| Coverage | Generated IDs; parse/evaluation support | How much of the benchmark contributed to a metric | Prevents selective subsets from looking like complete results |
| Contract validity | Contract OK rate; trace/sentinel counts; problem types | Whether outputs follow the response protocol | Stage metrics often cannot be computed without a valid trace |
| Evidence calibration | Abstention accuracy; false abstains; missed refusals; refusal precision/recall | Whether the model answers when evidence is sufficient and refuses when it is not | Prevents a refusal-biased model from appearing safe or accurate |
| Stage 1 | Document micro accuracy; macro F1; per-class scores | Whether the model judges evidence relevance and sufficiency per document | Explains whether a bad final decision began with local evidence assessment |
| Stage 2 | Conflict accuracy; macro F1; class confusion | Whether it recognizes agreement, complementarity, conflict, outdatedness, or misinformation | Explains the framing of the final answer |
| Grounded response | Citation validity/coverage; token F1; Rouge-L F1 | Whether final answers are cited and lexically align with usable references | Captures user-facing utility but is not a semantic judge |

The practical reading order is:

1. Check that the output and evaluation support are sufficiently complete.
2. Check false abstentions and missed refusals.
3. Diagnose document and conflict errors.
4. Read citation and final-answer quality measures.

## 3. Response-contract evaluation

### What the trace contract measures

The trace-text contract checks whether a response supplies the public structure required by the task. A valid trace-oriented response needs, among other things:

- exactly one well-aligned `<think>...</think>` block;
- a Stage-1 assessment covering the retrieved documents in canonical order;
- valid document verdict labels (`supports`, `partially supports`, `irrelevant`);
- a recognized Stage-2 conflict label;
- a final answer or canonical refusal after the trace;
- no out-of-range citations; and
- the `[[END-OF-ANSWER]]` sentinel.

The resulting `ok_rate_pct` (often summarized as *contract OK*) is the fraction of generated rows passing all applicable checks. It is a structural reliability metric, not a direct measure of factual correctness. A model can emit a perfectly formatted but incorrect trace; conversely, a model can answer correctly in natural language while failing the project's trace format.

### Why contract validity is a gate

Document-verdict and conflict-type evaluators need a parseable trace. If a model omits the trace, those later metrics may be unavailable or calculated on only the subset that happened to match the format. This is why matrix analyses treat contract completion and evaluation support as first-gate checks.

For example, a base model under the minimal prompt may provide a reasonable direct answer but no trace. Its conflict support can therefore be zero—not because every conflict decision was necessarily wrong, but because the output did not expose a parseable Stage 2 label. The correct conclusion is *the trace evaluation is unsupported in that condition*, not simply “conflict accuracy is zero.”

### Contract diagnostics

The report preserves error categories such as missing/duplicated think blocks, document-ID order mismatch, invalid conflict type, missing sentinel, out-of-range citation, or refusal violations. These categories are useful because they distinguish:

- a decoding/length/format problem;
- a trace-parsing problem;
- an evidence-decision problem; and
- a grounding/citation problem.

The contract report also records trace and sentinel presence in the final-answer report. These counts are especially useful under minimal prompting, where the model is not explicitly reminded of the full trace schema.

### Citation coverage within the contract suite

The contract evaluator separately measures citation discipline on non-abstaining final answers. Sentence-level coverage is the fraction of final-answer sentences containing at least one in-range `[dX]` citation. The current evaluator marks a citation pass at **75%** sentence coverage. The strict prompt itself asks for at least 80%; this small difference should be reported honestly: the prompt specification and the evaluator threshold are related but not identical.

Citation coverage is intentionally reported separately from contract OK. This avoids a single format statistic hiding whether a syntactically valid answer actually links its claims to retrieved evidence.

## 4. Abstention and answerability calibration

### The central decision

The final response must make one fundamental evidence decision:

| Gold evidence state | Desired behavior |
| --- | --- |
| The retrieved evidence can answer the query | Answer with grounded citations |
| The retrieved evidence leaves a necessary gap | Use the canonical controlled refusal |

The gold refusal state comes from `expected_response.abstain` when present, otherwise from the inverse of `answerable_under_evidence`. The evaluator identifies a predicted refusal using the canonical phrase and several semantically similar refusal patterns.

### Confusion matrix terminology

The report uses the refusal/abstention class as the positive class.

| Outcome | Meaning | Why it matters |
| --- | --- | --- |
| True positive | Gold requires refusal and model refuses | Correct caution |
| True negative | Gold is answerable and model answers | Correct grounded answering |
| False positive / **false abstain** | Gold is answerable but model refuses | Over-conservative behavior; a central failure in this project |
| False negative / **missed refusal** | Gold requires refusal but model answers | Unsupported answering / hallucination-risk behavior |

Abstention accuracy is the fraction of all gold-labeled rows where answer-versus-refusal matches. It is useful but insufficient on its own because the benchmark has more answerable than refusal cases. The report therefore also exposes false-abstain and missed-refusal counts, plus refusal precision, refusal recall, refusal F1, and non-refusal specificity.

### How to read the trade-off

- Fewer false abstains means the model is less likely to withhold an answer when the retrieved evidence is enough.
- Fewer missed refusals means it is less likely to answer without enough evidence.
- High refusal recall with weak refusal precision often signals over-abstention.
- High refusal precision with weak recall may signal risky over-answering.

Neither direction is universally preferable. The evidence policy requires good calibration on both classes. This is why a run is not accepted simply because abstention accuracy rises: the underlying confusion counts must move in a defensible direction.

The benchmark also contains difficult cases where gold refusal coexists with one or more supporting-looking documents, or where only partial evidence exists. The contract report preserves these diagnostics rather than assuming every support-marked document automatically makes an answer possible. This is important because evidence sufficiency is a set-level decision, not merely the presence of one positive local verdict.

## 5. Stage 1: per-document verdict evaluation

### What is predicted

For every retrieved document, the trace assigns one verdict:

| Verdict | Meaning |
| --- | --- |
| `supports` | The snippet directly and decisively answers the needed part of the query |
| `partially supports` | The snippet is relevant but incomplete, indirect, hedged, or lacks a necessary detail |
| `irrelevant` | The snippet does not help answer the query |

The evaluator compares predicted verdicts with the canonical `per_doc_notes` verdicts. It parses either the text-stage trace format or the older compatible JSON-array representation. It scores document IDs present in both the prediction and gold record; missing and extra document IDs are separately logged as errors.

### Metrics

| Metric | Definition | What it reveals |
| --- | --- | --- |
| Document micro accuracy | Correct verdicts divided by all evaluated document pairs | Overall local evidence-classification reliability, dominated by frequent classes |
| Document macro F1 | Unweighted mean of class F1 scores | Whether minority verdicts, especially `irrelevant`, are being neglected |
| Per-class precision | Of documents predicted as a class, how many were correct | Overuse of that label |
| Per-class recall | Of gold documents in a class, how many were found | Under-recognition of that label |
| Confusion matrix | Gold verdict versus predicted verdict counts | The specific direction of evidence-assessment errors |
| Evaluated document pairs | Number of gold/predicted document intersections | Whether a high score is based on near-complete coverage |

### Why micro and macro both matter

The held-out benchmark is imbalanced at the document level: `partially supports` is common, while `irrelevant` is relatively rare. A model can achieve a respectable micro score by predicting the common labels while performing poorly on irrelevant distractors. Macro F1 gives each class equal weight and makes that failure visible.

The confusion direction is often more useful than accuracy alone. For example:

- `supports -> partially supports` indicates unnecessary caution that can contribute to false abstention;
- `partially supports -> supports` indicates overconfidence in incomplete evidence;
- `irrelevant -> partially supports` suggests topical-keyword matching rather than query-specific relevance; and
- `irrelevant -> supports` is a more severe evidence-grounding error.

### Example of report interpretation

The stored Llama 3.1 8B Run L end-to-end minimal result evaluates 3,685 document pairs with 79.81% micro accuracy and 0.7639 macro F1. Its class-level report shows strong partial-support performance but lower `irrelevant` F1 (0.682), indicating that distractor handling remains the weaker Stage-1 component. This interpretation is more informative than saying only that document accuracy is approximately 80%.

## 6. Stage 2: conflict-type prediction

### What conflict-type prediction measures

After assessing documents individually, the model assigns one relationship label to the relevant evidence set:

| Conflict type | What the classifier must recognize |
| --- | --- |
| No conflict | Relevant documents agree on the core answer; differences are superficial or redundant |
| Complementary information | Documents contribute distinct compatible facets that should be combined |
| Conflicting opinions or research outcomes | Documents make incompatible claims within the same scope and time window |
| Conflict due to outdated information | Older and newer factual claims conflict, with newer evidence superseding the older |
| Conflict due to misinformation | The retrieved set establishes that a weaker claim is false or misleading relative to stronger evidence |

Conflict-type prediction is valuable because the same final answer can be superficially plausible under very different evidence relationships. A model that labels complementary facts as conflict may hedge or refuse unnecessarily. A model that labels same-scope contradictions as complementary may collapse disagreement into an overconfident answer. Correct Stage 2 classification therefore supports appropriate answer framing, not merely a label leaderboard.

### Metrics

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Conflict accuracy | Exact label matches divided by valid gold/prediction label pairs | Overall Stage-2 classification correctness on parseable rows |
| Conflict support | Number of valid pairs included in accuracy | Essential denominator; low support means the score is selective |
| Macro F1 | Unweighted mean class F1, when reported by the contract evaluator | Whether rare labels contribute fairly |
| Per-class precision/recall/F1 | One-vs-rest performance for each label | Which conflict relation is over- or under-predicted |
| Actual/predicted distributions | Gold and predicted label frequencies | Detects collapse into one favorite class |
| Confusion matrix and top confusions | Gold-to-prediction error directions | Converts aggregate error into a hypothesis for the next recipe revision |

The dedicated conflict report treats an absent or unparsable trace label as `PRED_MISSING` or `PRED_INVALID` in its per-ID diagnostics, but the classification accuracy denominator includes only valid canonical label pairs. Therefore **always report conflict support together with accuracy**. A 100% conflict score on a handful of valid rows is not comparable to 65% over nearly all 736 examples.

### What common confusion pairs mean

| Gold -> predicted | Likely reasoning error |
| --- | --- |
| No conflict -> Complementary information | Model treats aligned or redundant evidence as if it contains distinct necessary facets |
| Complementary information -> No conflict | Model fails to recognize that multiple compatible pieces must be combined |
| Conflicting opinions -> Complementary information | Model smooths genuine incompatibility into a synthesis |
| Outdated information -> No conflict or complementary | Model misses temporal supersession |
| Misinformation -> complementary/no conflict | Model does not use relative evidence strength to distinguish a misleading claim |

The persistent dominant error in the retained matrices is often `No conflict -> Complementary information`. This is a substantive taxonomy boundary, not merely an output-format issue, and it motivated targeted boundary drills in later SFT recipes.

### Example

For the stored Llama Run L end-to-end minimal output, conflict accuracy is 61.07% over 732 valid pairs. The most frequent error is 106 `No conflict -> Complementary information` confusions. Its outdated-information F1 is high (0.839), whereas misinformation F1 is low (0.163), reflecting both a rare class and a difficult relative-credibility judgment. The report supports a specific conclusion—no-conflict/complementarity and misinformation remain weak boundaries—not a generic claim that “conflict reasoning is poor.”

## 7. Citations and grounding discipline

### Citation validity

The evaluator checks citations in the final answer against the number of retrieved documents. A citation must reference an in-range ID such as `[d1]` through `[dN]`. It reports:

- average number of citations and unique citations;
- average sentence-level citation coverage; and
- count of rows containing invalid citations.

An in-range citation is necessary but not sufficient for entailment. The evaluator checks formal grounding discipline—not whether every cited snippet semantically proves every claim. Per-document verdicts, manual inspection, and future semantic evaluation remain important complements.

### Citation coverage

For each non-refusal answer, sentence coverage is:

```text
number of final-answer sentences containing at least one valid citation
---------------------------------------------------------------
total final-answer sentences
```

A long answer with one citation can therefore have low coverage even if the citation itself is valid. This prevents a model from appearing grounded simply because it attaches one document ID to an otherwise unsupported paragraph. The contract report records the proportion of answers that pass the 75% coverage threshold and the average coverage; the final-answer report records average coverage across the generated set.

## 8. Final-answer and lexical-overlap evaluation

### What is scored

The final-answer evaluator removes a trace and sentinel when present, then compares the remaining user-facing answer with the canonical expected answer or gold answer. It evaluates three related but different properties:

| Property | Metrics | Purpose |
| --- | --- | --- |
| Answer/refusal decision | Abstention confusion and refusal metrics | Evidence calibration |
| Citation behavior | Counts, validity, coverage | Formal grounding discipline |
| Surface answer similarity | Token F1 and Rouge-L F1 | Lightweight proxy for agreement with reference wording/content |

### Token F1

Token F1 normalizes text by removing citations and punctuation, lowercasing, and tokenizing words. It compares token multisets between prediction and reference:

```text
precision = overlapping token occurrences / predicted tokens
recall    = overlapping token occurrences / gold tokens
F1        = harmonic mean of precision and recall
```

It rewards inclusion of reference content while penalizing extra unsupported wording. It does **not** understand paraphrase, logical equivalence, date normalization, or whether the answer is grounded in the correct cited evidence.

### Rouge-L F1

Rouge-L uses the longest common subsequence of normalized tokens. It is more sensitive to retained sequence order than bag-style token overlap. Like Token F1, it is a cheap lexical proxy, not a semantic judge.

### Scored-pair denominator

Lexical overlap is computed only when all of the following hold:

1. the gold row is non-refusal;
2. the gold answer is nonempty and not a known placeholder; and
3. the model did not abstain.

This is necessary because there is no meaningful answer-string comparison for a required refusal, a missing/placeholder reference, or a model refusal. It also means the denominator can shrink precisely when false abstention is high. Always report `scored_pairs` next to Token F1 and Rouge-L.

On the 736-example benchmark, 608 examples are marked answerable under evidence but only 548 have usable nonblank gold answers. In the Llama Run L minimal example, 17 false abstentions reduce the lexical-overlap denominator further to 531. A model should not be credited for a good lexical score without also reporting its false-abstain count and scored-pair coverage.

### Limits of final-answer overlap

The repository explicitly treats lexical overlap as a triage metric. It can under-score a correct paraphrase and over-score a fluent answer that repeats reference words but reaches the wrong conclusion. It does not replace human assessment, a semantic judge, or entailment checking. Its strongest use is alongside the per-ID low-overlap examples, which identify rows needing manual inspection.

## 9. Trace-text and answer-only evaluation differ

| Dimension | Trace-text output | Answer-only output |
| --- | --- | --- |
| Visible intermediate stages | Yes | No |
| Contract focus | Think block, document order/verdicts, conflict label, citations, sentinel | Nonempty final answer/refusal, citations where applicable, sentinel |
| Meaningful stage metrics | Document verdict and conflict type | Not applicable: no exposed Stage 1/2 targets |
| Primary post-inference measures | Full suite | Final-answer/refusal and citation behavior |

The generic cluster evaluation wrapper can run every evaluator on any JSONL file, but trace-specific reports naturally have little or no evaluable support for answer-only outputs. Such a zero should not be interpreted as answer-only reasoning failure; it means the evaluator expects information the answer-only target intentionally omits.

The updated answer-only artifacts should be reported with final-answer metrics, final-only contract behavior, and their documented checkpoint-selection limitation. The clean 862-example answer-only baseline is not yet a result family because no completed evaluation artifacts are stored. See [the standalone answer-only limitations](../answer_only_sft_export/docs/KNOWN_LIMITATIONS.md).

## 10. Operational reading rules for result tables

The current matrix analyses use a practical “operationally trustworthy” screen:

| Screen | Current criterion | Reason |
| --- | ---: | --- |
| Contract OK | at least 80% | Enough outputs must obey the response representation |
| False abstains | at most 25 | The model must not refuse too many answerable cases |
| Conflict support | at least 700 of 736 | Conflict metrics should represent nearly the complete benchmark |

This is a repository-specific reading aid, not a universal research threshold or a composite metric to optimize blindly. It prevents clearly malformed, heavily over-abstaining, or selectively supported rows from being promoted based on one attractive statistic.

When comparing two rows, use this checklist:

1. Are they the same model variant, prompt mode/profile, and benchmark split?
2. Do both have adequate contract and metric support?
3. Did false abstentions and missed refusals move in an acceptable direction?
4. If document/conflict accuracy changed, which class confusion changed?
5. Did citation validity/coverage remain acceptable?
6. Are Token F1/Rouge-L computed over comparable numbers of scored answer pairs?
7. Is the row end-to-end, or does it receive oracle information?

## 11. Worked interpretation pattern

A sound matrix discussion should make a chain of claims rather than quote a single number. For example, the stored Llama 3.1 8B Run L `SFT + e2e + minimal` row:

- has 93.2% contract completion and 732 valid conflict pairs, so its trace-based metrics have broad support;
- achieves 97.55% abstention accuracy with 17 false abstentions and one missed refusal, indicating strong but not perfect evidence calibration;
- obtains 79.81% document micro accuracy, with `irrelevant` the weaker document class;
- obtains 61.07% conflict accuracy, with the dominant confusion `No conflict -> Complementary information`;
- has no invalid citations and average sentence coverage of 0.6906; and
- has Token F1 of 0.3516 over 531 eligible answer pairs.

This is a much stronger scientific description than “the model scored 0.3516.” It identifies what the model reliably does, what it still confuses, and which metrics are conditional on the output representation.

The corresponding Mistral Run L end-to-end row illustrates why all layers are needed: it has strong format completion (90.8%) but 49 false abstentions. It is therefore the best available Mistral row, yet does not meet the repository’s operationally trustworthy screen. Structural improvement alone is not sufficient.

## 12. Artifact map and reproducible evaluation

| Artifact | Role |
| --- | --- |
| [eval_contract.py](../code/eval/eval_contract.py) | Validates the trace contract, abstention diagnostics, citation coverage, and compatible conflict-label F1 |
| [eval_doc_verdicts.py](../code/eval/eval_doc_verdicts.py) | Scores Stage-1 document verdicts and reports class confusion |
| [eval_conflict_type.py](../code/eval/eval_conflict_type.py) | Scores Stage-2 conflict labels, support, distributions, and top confusions |
| [eval_final_answer.py](../code/eval/eval_final_answer.py) | Scores refusal behavior, citations, Token F1, Rouge-L, and per-ID answer records |
| [evaluate_experiment.sh](../slurm/sharanga/evaluate_experiment.sh) | Runs the standard post-inference report suite |
| [final_model_outputs/](../final_model_outputs/) | Stored raw/sanitized outputs and report artifacts for current result families |
| [Qwen Run K audit](experiments/qwen7b_run_k_benchmark_matrix_audit.md) | Example matrix-level interpretation of the metrics |
| [Llama matrix analysis](../final_model_outputs/llama8b_benchmark_matrix_analysis.md) | Stored Run L example with broad end-to-end support |
| [Mistral matrix analysis](../final_model_outputs/mistral7b_benchmark_matrix_analysis.md) | Stored Run L counterexample: good structure but excessive false abstention |

To reproduce an evaluation report for an existing sanitized output, use the canonical benchmark and the four evaluator programs through the supplied evaluation wrapper. Reproducing a new model output additionally requires the named base model and, for SFT, the matching current LoRA adapter; current adapters are not bundled. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete release-status disclosure.

## 13. Paper-safe summary

> We evaluate generated RAG responses with a layered post-inference suite. Contract checks establish whether a response exposes a complete, machine-checkable evidence trace; abstention metrics quantify evidence-sufficiency calibration; document-verdict and conflict-type metrics assess intermediate evidence reasoning; citation checks measure formal grounding discipline; and Token F1/Rouge-L provide lightweight final-answer overlap diagnostics on usable non-refusal references. We report evaluation support and confusion patterns alongside aggregate scores, and treat oracle-conditioned metrics as diagnostic upper bounds rather than end-to-end performance.
