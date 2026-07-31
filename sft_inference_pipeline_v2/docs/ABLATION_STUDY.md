# Ablation Study and Recipe-Development Analysis

## Purpose and evidence standard

This document records how the trace-text SFT recipe evolved, what each experimental change was intended to test, and what the retained evidence supports. It is written for a paper ablation section and deliberately distinguishes three things that are often blurred together:

1. **Controlled ablations:** a narrow intervention on a stable backbone, such as adding a conflict-boundary drill.
2. **Targeted recipe revisions:** coordinated changes to data coverage, message mixture, and weighting, motivated by a measured error slice.
3. **Diagnostic evaluation conditions:** prompt-profile and oracle interventions that reveal sensitivity or bottlenecks but are not new training recipes.

The project contains valuable historical assets, but a legacy filename is not itself evidence of a completed and reproducible result. This document makes detailed numerical claims only where a retained strategy record or matrix analysis supports them. It does not retroactively turn the full development sequence into a perfectly controlled one-factor study.

The central research question is: **what supervision design makes an instruction-following model more reliable at evidence-grounded answering, controlled refusal, conflict reasoning, and response-contract compliance?** The final answer is not “more training rows always help.” Rather, the retained experiments indicate that *error-directed, balanced exposure* is more useful than indiscriminate pressure on one label or one objective.

For the current SFT recipe, see [SFT_DESCRIPTION.md](SFT_DESCRIPTION.md). For prompt and oracle definitions, see [PROMPT_DESIGN_AND_ABLATIONS.md](PROMPT_DESIGN_AND_ABLATIONS.md). For exact Run K/L message arithmetic, see [run_k_l_training_recipe_tables.md](experiments/run_k_l_training_recipe_tables.md).

## 1. Experimental outcome hierarchy

The pipeline evaluates several linked outcomes. Any ablation should be judged on this hierarchy rather than a single number.

| Level | Question | Representative evidence |
| --- | --- | --- |
| Structural validity | Did the model produce an evaluable response in the required form? | Contract completion, trace/sentinel presence, parse support |
| Decision calibration | Did it answer when evidence sufficed and refuse when it did not? | Abstention accuracy, false abstentions, missed refusals |
| Intermediate reasoning | Did it assess documents and conflicts appropriately? | Document-verdict accuracy, conflict-type accuracy |
| Grounded response quality | Did it produce a relevant cited answer? | Citation validity/coverage, token F1, Rouge-L |

Structural validity is a gate. A high conflict score calculated on a small surviving subset is not evidence of a strong system. Similarly, an apparent abstention improvement is not a win if it comes from refusing a large fraction of answerable examples. The 736-example held-out benchmark and matrix audits report the associated support and contract statistics precisely to prevent these misleading comparisons.

## 2. Development timeline at a glance

| Run / comparison | Main intervention | Intended failure addressed | Status in the paper narrative |
| --- | --- | --- | --- |
| D | Strict + runtime + minimal prompt mixture with multi-task supervision | Prove that trace behavior can survive minimal prompting | Foundational historical baseline |
| E | Broad conflict calibration / heavier conflict pressure | Improve conflict labeling globally | Negative lesson: model-size-sensitive trade-off |
| F | One explicit conflict-taxonomy boundary drill per source example | Improve difficult class boundaries without global oversampling | Strong historical controlled ablation |
| G (32B) | Extra document-verdict supervision and a document-boundary drill | Recover F's Stage-1 degradation | Useful trade-off ablation, not default |
| G (7B) | Extra minimal exposure, complementary boundary emphasis, source-hygiene drill | Repair a minimal-prompt structure leak | Useful localized fix with overcorrection cost |
| J | Benchmark-aligned train/validation geometry and answerable/refusal weighting | Reduce short-context false abstention | Turning-point recipe revision |
| K | Small set of derived 5-document answerables plus stronger local drills | Repair residual short answerable / partial-synthesis failures | Latest stored Qwen recipe |
| L | Add short answerable no-conflict coverage and rebalance class boundaries | Correct K's one-sided short-context pressure | Latest stored Llama/Mistral recipe |
| Trace-text vs answer-only | Visible intermediate supervision versus final-answer-only supervision | Test the role of public trace targets | Comparison family; clean 862 baseline results are pending |

Runs D through G use the older 609-example source backbone. Run J establishes the current 862/81 train/validation backbone; K and L extend it to 889 and 910 source training examples respectively. Message rows are repeated supervised views and must not be reported as independent examples.

## 3. The foundational prompt-robust ablation: Run D

### Hypothesis

Early work could obtain a well-formed trace when a detailed prompt repeatedly described the desired process. That alone does not show that SFT taught the process. Run D tested the stronger hypothesis that a model can internalize a public evidence trace and emit it even when the inference prompt is minimal.

### Intervention

Run D kept the source set fixed at 609 examples and varied the supervised views:

| Component | Message rows | Role |
| --- | ---: | --- |
| Strict/default end-to-end trace | 1,218 | Detailed teacher-style instruction |
| Runtime end-to-end trace | 609 | Compact guided instruction |
| Runtime document-verdict task | 609 | Direct Stage-1 supervision |
| Runtime conflict-type task | 1,218 | Direct Stage-2 supervision |
| Runtime answer-only task | 609 | Direct final response/refusal supervision |
| Minimal end-to-end trace | 2,436 | Sparse-prompt internalization pressure |
| **Total** | **6,699** | **Prompt-robust multi-task mixture** |

The important causal contrast is not merely row count. The model sees the same underlying evidence problem through detailed, compact, and sparse instructions, while its desired target remains a public trace plus final answer. The minimal rows receive the largest message count (four copies per source example), making them an explicit training objective rather than a post-hoc stress test.

### What it established

Run D is the earliest retained strategy that reliably demonstrated minimal-prompt trace internalization. In the older 49-example evaluation used at that stage, both Qwen 7B and Qwen 32B completed all 49 strict, runtime, and minimal traces. It became the reference point because it showed that performance did not depend exclusively on the long teacher prompt.

### What it did not establish

Run D was not the final recipe. Its main remaining weakness was conflict-taxonomy discrimination. In particular, the development record identified ambiguity around `No conflict`, `Complementary information`, and other boundary cases. The next experiments therefore tested how to improve those decisions without sacrificing the newly obtained minimal-prompt robustness.

## 4. Broad versus targeted conflict calibration: Runs E and F

### Run E: useful negative evidence

Run E attempted broad conflict calibration: it increased general conflict pressure rather than targeting a particular error boundary. The result is important precisely because it was not uniformly successful.

| Model size | D conflict accuracy: strict / runtime / minimal | E conflict accuracy: strict / runtime / minimal | Reading |
| --- | --- | --- | --- |
| Qwen 7B | 66.67 / 68.75 / 73.47 | 61.22 / 57.14 / 58.33 | Conflict performance fell substantially despite cleaner contract/document behavior |
| Qwen 32B | 63.27 / 69.39 / 63.27 | 73.47 / 73.47 / 71.43 | Conflict improved, while minimal document quality weakened |

The lesson is not that conflict supervision is harmful. The more precise lesson is that **broad label pressure interacts with model capacity and can distort the decision distribution.** For 7B, the intervention appears to have shifted the classifier away from its prior useful boundary behavior. For 32B, it helped Stage 2, but at a cost to Stage 1 under the sparse prompt.

This is scientifically useful negative evidence: a global objective change did not yield a universally portable recipe. It motivated a narrower intervention that teaches the semantic distinction at the exact point of confusion.

### Run F: a targeted conflict-boundary ablation

Run F retains the Run D backbone and adds a single derived runtime conflict-type example per source row. The extra row contains a compact boundary guide explaining when evidence is aligned, complementary, same-scope incompatible, temporally superseded, or misinformation.

| Component | Run D | Run F | Change |
| --- | ---: | ---: | --- |
| Strict/default end-to-end | 1,218 | 1,218 | Held fixed |
| Runtime end-to-end | 609 | 609 | Held fixed |
| Runtime document-verdict | 609 | 609 | Held fixed |
| Runtime conflict-type | 1,218 | 1,218 | Held fixed |
| Conflict-boundary drill | 0 | 609 | One targeted copy per source example |
| Runtime answer-only | 609 | 609 | Held fixed |
| Minimal end-to-end | 2,436 | 2,436 | Held fixed |
| **Total** | **6,699** | **7,308** | **Only the boundary-drill family is added** |

F also raises the explicit conflict loss emphasis modestly (3.2 to 3.6) while leaving the prompt-robust structure intact. It is therefore not a perfectly one-variable experiment, but it is much narrower than E: the new supervised content directly encodes the taxonomy distinctions that the model was confusing.

### Outcome and interpretation

| Model size | F conflict accuracy: strict / runtime / minimal | F document accuracy: strict / runtime / minimal | Key conclusion |
| --- | --- | --- | --- |
| Qwen 7B | 77.55 / 71.43 / 72.92 | 79.43 / 77.12 / 76.94 | Best strict/runtime conflict performance among D/E/F; one minimal structural leak remained |
| Qwen 32B | 77.55 / 71.43–75.00 / 72.92–73.47 | 80.56 / 79.54 / 81.07 | Strongest conflict/contract historical setting, but below D on document judgments |

For both scales, F improved the core conflict task more cleanly than broad E calibration. This supports the methodological claim that **teaching a decision boundary is often preferable to globally duplicating a label**. The effect is still architecture-dependent: F was selected as the strongest historical conflict/contract recipe for Qwen 32B, whereas the 7B story retained a split preference between D's minimal robustness and F's strict/runtime conflict strength.

## 5. Model-size-specific probes: the two Run G variants

Run G is intentionally not one universal method. It comprises two different follow-ups because the observed failure was different by model size. This is an important empirical point: the same output metric can arise from different underlying errors, so applying the same remedy to every model is not necessarily justified.

### 5.1 Qwen 32B G: document-verdict stabilization

F improved conflict behavior for Qwen 32B but showed a Stage-1 regression, particularly an overuse of `partially supports`. The 32B G intervention therefore added two forms of document-focused pressure: one more runtime document-verdict view and one document-boundary drill.

| Component | F | 32B G | Rationale |
| --- | ---: | ---: | --- |
| Runtime document-verdict task | 609 | 1,218 | Increase direct Stage-1 signal |
| Document-boundary drill | 0 | 609 | Explain support / partial / irrelevant distinctions |
| Conflict-boundary drill | 609 | 609 | Preserve F's Stage-2 teaching |
| Other principal D/F components | unchanged | unchanged | Retain prompt-robust backbone |
| **Total message rows** | **7,308** | **8,526** | Focused Stage-1 expansion |

The observed result is a genuine trade-off:

| Profile | F: conflict / doc micro / contract-adjusted | 32B G: conflict / doc micro / contract-adjusted | Interpretation |
| --- | --- | --- | --- |
| Strict | 77.55 / 80.56 / 85.7 | 69.39 / 85.17 / 77.6 | Document recovery, substantial conflict/contract loss |
| Runtime | 75.00 / 79.54 / 77.6 | 73.47 / 81.59 / 75.5 | Modest document gain, small conflict/contract loss |
| Minimal | 73.47 / 81.07 / 81.6 | 69.39 / 83.63 / 71.4 | Document recovery, substantial minimal trade-off |

G 32B therefore falsifies the simple idea that adding Stage-1 supervision will retain all Stage-2 gains. It recovered document accuracy but made the model more likely to smooth genuine conflict into `No conflict` and sometimes to over-trigger outdated conflict. It is correctly retained as a document-focused ablation/fallback, not as the historical default large-model checkpoint.

### 5.2 Qwen 7B G: source hygiene and minimal-format repair

The 7B problem was different: F had strong strict/runtime conflict performance but one minimal-prompt output leaked instruction-like text from a retrieved source and malformed the trace. Complementary-information recall was also fragile. The 7B G intervention:

- raised minimal end-to-end exposure from four to five copies per source example;
- doubled only the `Complementary information` boundary pressure;
- added one source-hygiene end-to-end drill per runtime end-to-end source row; and
- did **not** import the 32B document-stabilization intervention.

| Component | F | 7B G | Rationale |
| --- | ---: | ---: | --- |
| Minimal end-to-end trace | 2,436 | 3,045 | Restore minimal-prompt robustness |
| Complementary boundary drill | 189 rows | 378 rows | Address the observed 7B confusion boundary |
| Source-hygiene drill | 0 | 609 | Teach that snippets are evidence, not commands |
| **Total message rows** | **7,308** | **8,715** | Model-specific repair mixture |

The source-hygiene goal succeeded structurally: the minimal trace and sentinel returned to 49/49 and the previously malformed example was repaired. But the broader score pattern did not make G the preferred 7B checkpoint.

| Profile | F conflict accuracy | 7B G conflict accuracy | 7B G contract-adjusted | Reading |
| --- | ---: | ---: | ---: | --- |
| Strict | 77.55 | 73.47 | 69.4 | Lost F's strict conflict edge |
| Runtime | 71.43 | 73.47 | 65.3 | Small conflict gain, weak contract-adjusted behavior |
| Minimal | 72.92 | 69.39 | 67.3 | Restored structure but did not preserve decision quality |

The ablation shows that source hygiene can repair a specific format-contamination failure, but that extra minimal and complementary pressure can overcorrect a smaller model. It should be cited as a targeted robustness lesson, not evidence that more prompt injections are always beneficial.

## 6. The geometry shift: Run J

### Why a new kind of intervention was necessary

By the Run F/G stage, the remaining issue was no longer only local classification. Evaluation showed broad false abstention on the final benchmark, which is heavily concentrated in five-document contexts. The original training/validation geometry did not sufficiently represent short answerable examples, particularly those requiring composition of partial evidence.

Run J changed the source data and selection geometry, not just the wording of a prompt or a task weight. It selected benchmark-like answerable examples into the training and validation backbone while preserving a separate 736-example final holdout.

| Split change | Count | Purpose |
| --- | ---: | --- |
| Base source training rows | 694 | Implied by the retained split summary: 862 final rows minus 168 training augmentations |
| Benchmark-like answerables moved to training | 168 | Expose short answerable contexts during SFT |
| Final Run J training rows | 862 | Current canonical training backbone |
| Base validation rows | 56 | Earlier selection set |
| Benchmark-like answerables moved to validation | 25 | Let checkpoint selection see the known failure slice |
| Final validation rows | 81 | Current canonical validation backbone |
| Final benchmark holdout | 736 | Remains excluded from training |

The selection was not a blind transfer of all benchmark data. Blank-answer cases, duplicate queries, and overlaps were excluded. The selected set included 155 five-document examples out of 193 selected benchmark-like rows, aligning training and validation more closely with the benchmark's dominant retrieval depth.

One archived Run J narrative refers to the old setup as `692/56`. The machine-readable retained split summary records 168 training augmentations and 862 final training rows, whose arithmetic implies 694 base training rows; the same summary gives 25 validation augmentations and 81 final validation rows, which implies 56 base validation rows. This document uses the arithmetic-backed value and preserves the discrepancy as a provenance limitation rather than treating the older prose label as authoritative.

### Why validation augmentation matters

If a validation set underrepresents the deployment failure mode, checkpoint selection may prefer a model that appears strong on the old distribution but still falsely refuses short answerable cases. Adding the relevant answerable slice to validation does not prove generalization by itself, but it makes selection responsive to the behavior the final benchmark measures. Run J's composite selection also penalizes false abstentions rather than rewarding only a generic classification score.

### Observed change

Run J was the turning point for abstention calibration.

| Model / split | Contract OK | Abstention accuracy | False abstains | Missed refusals | Conflict accuracy | Document micro | Token F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B benchmark | 92.5 | 94.29 | 39 | 3 | 62.07 | 77.96 | 0.3555 |
| Qwen 32B benchmark | 92.8 | 94.97 | 29 | 8 | 62.55 | 82.05 | 0.3945 |

The relevant result is qualitative as well as numerical: the benchmark no longer exhibited catastrophic refusal behavior. The remaining errors clustered in an interpretable slice—short answerable examples, especially partial-only evidence and misinformation—rather than being broadly distributed. This allowed the next runs to target a small error region instead of making another global recipe change.

### Causal caution

J is not a single-factor ablation. It changes training coverage, validation coverage, and effective weighting together. It supports the claim that **benchmark-aligned data geometry was associated with much better calibration**. It does not isolate the numerical contribution of each individual change. A future controlled study would hold the split fixed and vary one exposure mechanism at a time.

## 7. Residual-error targeting: Run K

### Hypothesis

Run J's residual false abstentions were concentrated in five-document answerable cases. The hardest subset had no individually decisive document but did have jointly sufficient partial evidence. Run K tests whether a small, targeted addition can improve this boundary without destabilizing the successful J backbone.

### Intervention

K starts from the 862/81 Run J split and adds 27 derived five-document answerable variants made only from existing training examples. The validation set and final holdout remain unchanged.

| Change from J | Run K design | Intended effect |
| --- | --- | --- |
| Source training examples | 862 -> 889 | Add short-context support without using held-out rows |
| Derived examples | 27 conflict-bearing five-document answerables | Match the residual failure regime |
| Document-boundary drill | Enabled at one copy per runtime document task | Improve Stage-1 distinctions |
| Partial-synthesis drills | Stronger in runtime and minimal views | Teach joint sufficiency explicitly |
| Conflict-boundary pressure | Higher for non-no-conflict labels | Address residual taxonomy errors |
| Short-answerable weights | Retained/strengthened from J | Preserve the anti-false-abstention direction |

The exact mixture rises from 10,674 Run J message rows to 12,659 Run K rows. This is not simply 1,985 unrelated new examples: it consists of 27 derived source rows plus targeted repeated views and drills.

### Observed Qwen comparison

| Metric | Qwen 7B J | Qwen 7B K | Qwen 32B J | Qwen 32B K |
| --- | ---: | ---: | ---: | ---: |
| Abstention accuracy | 94.29 | 95.79 | 94.97 | 96.60 |
| False abstains | 39 | 18 | 29 | 25 |
| Missed refusals | 3 | 13 | 8 | 0 |
| Conflict accuracy | 62.07 | 60.14 | 62.55 | 67.30 |
| Document micro | 77.96 | 80.44 | 82.05 | 83.27 |
| Token F1 | 0.3555 | 0.3296 | 0.3945 | 0.4068 |
| Citation pass | 66.3 | 73.3 | 55.1 | 75.3 |

K is a clean overall improvement for the stored Qwen 32B results: it improves abstention, eliminates missed refusals, and improves conflict, document, answer-overlap, and citation metrics. For Qwen 7B, it is a genuine trade-off: false abstentions and document/citation behavior improve, but missed refusals increase and conflict/answer overlap decline.

The defensible interpretation is capacity-sensitive absorption of targeted data. The larger model benefited across the measured components, while the smaller model became more cautious in a way that improved one type of calibration but weakened other reasoning behaviors. Neither result justifies claiming that K is universally superior for all models.

## 8. Boundary rebalancing: Run L

### Hypothesis

K's derived short answerable cases were all conflict-bearing. That improves exposure to hard answerable cases but may teach a new spurious association: a short answerable retrieval context often contains conflict. Run L tests whether adding short answerable `No conflict` cases corrects this missing boundary coverage.

### Intervention

L returns to the Run J backbone and adds 48 derived source rows: the 27 short support cases used in K, 18 short `No conflict` support cases, and 3 short `No conflict` partial-only cases. It also increases runtime/boundary pressure for no-conflict and misinformation.

| Change from K | Run L design | Why it matters |
| --- | --- | --- |
| Source training examples | 889 -> 910 | Add 21 short answerable no-conflict rows |
| Derived `No conflict` short answerables | 0 -> 21 | Break the short-answerable-implies-conflict shortcut |
| No-conflict boundary drill weight | 1 -> 2 | Make alignment versus complementarity explicit |
| Misinformation boundary drill weight | 2 -> 3 | Retain pressure on a rare but difficult class |
| Runtime misinformation duplication | 1 -> 2 | Increase direct exposure to that underrepresented class |
| Total message rows | 12,659 -> 13,349 | Expand balanced targeted views |

This is a more principled follow-up than simply adding more conflict examples. It preserves the hypothesis that short answerables require additional exposure while testing whether **class balance within that exposure** matters.

### Stored result status and interpretation

The locally complete Run L matrices are for Llama 3.1 8B and Mistral 7B. Qwen Run L launch and message artifacts are retained, but no completed Qwen Run L matrix is stored in this release.

| Best Run L end-to-end minimal result | Llama 3.1 8B | Mistral 7B |
| --- | ---: | ---: |
| Contract completion | 93.2 | 90.8 |
| Abstention accuracy | 97.55 | 93.34 |
| False abstains | 17 | 49 |
| Missed refusals | 1 | 0 |
| Document micro | 79.81 | 78.47 |
| Token F1 | 0.3516 | 0.3305 |

Llama Run L meets the repository's operationally trustworthy criteria in its main end-to-end rows and substantially improves the overall base-versus-SFT profile. Mistral Run L improves structure and some answer behavior but does not meet the repository's operational threshold because false abstentions remain high. The correct cross-model conclusion is therefore not “L works” in the abstract; it is that the same boundary-balanced recipe is adopted differently by different base models.

This is a central result for the paper discussion: error-directed data balancing is promising but not architecture-invariant. The model family, base instruction behavior, and capacity likely affect whether additional answerable exposure improves calibration or produces excessive caution.

## 9. Trace-text versus answer-only supervision

This comparison asks a broader question: are public intermediate targets useful relative to training only the final answer? It is important, but it must be framed accurately because it is not a single isolated intervention in the established artifact set.

| Family | Visible assistant target | Main supervision signal | Current status |
| --- | --- | --- | --- |
| Trace-text SFT | Document assessment, conflict assessment, answer plan, final answer | Intermediate reasoning decisions plus final response | Main method; stored K/L matrices available by model |
| Established answer-only SFT | Final cited answer or controlled refusal only | Final decision and output contract | Stored outputs for Qwen 7B, Llama 8B, and Mistral 7B |
| Clean basic answer-only 862 baseline | One final-only target per canonical source row | Transparent final-answer-only baseline | Implementation is ready; no completed result artifacts yet |

The established answer-only mixture uses the same current 862/81 split but trains 10,344 weighted messages: 8 final-only and 4 minimal copies per source example. It also preserves a historical checkpoint-selection scheme that is effectively abstention-focused rather than semantic-quality-focused. The trace recipe differs not only in whether a trace is visible, but also in task mixture, targeted drills, sample weighting, data augmentation, and selection logic.

Consequently, an established trace-versus-answer-only comparison should be described as a **recipe-level comparison**, not proof that any difference is caused solely by exposing the trace. The clean 862-example answer-only baseline is better suited to a transparent future comparison because it contains one final-only target per canonical training example and removes extra drills, weights, and derived rows. But it has no completed evaluations in the repository and must not be reported as a result yet. See [the answer-only recipe](../answer_only_sft_export/docs/LATEST_ANSWER_ONLY_SFT_RECIPE.md) and [the clean-baseline disclosure](../answer_only_sft_export/docs/BASIC_ANSWER_ONLY_862_BASELINE.md).

## 10. Prompt and oracle ablations as diagnostics

The 12-condition evaluation matrix is itself an ablation family, but it changes **inference conditions**, not training data. It should be reported separately from the D–L training-recipe sequence.

| Intervention | What is held fixed | What is changed | Legitimate inference |
| --- | --- | --- | --- |
| Strict vs runtime vs minimal | Model, examples, prompt mode | Amount of procedural instruction | Whether behavior survives reduced prompt scaffolding |
| `e2e` vs `oracle_conflict` | Model, examples, prompt profile | Gold conflict label is supplied | Value / bottleneck of Stage-2 correctness |
| `e2e` vs `oracle_notes` | Model, examples, prompt profile | Gold document notes are supplied | Value / bottleneck of Stage-1 correctness |
| `e2e` vs `oracle_both` | Model, examples, prompt profile | Both intermediate artifacts are supplied | Conditional upper bound for final decision and synthesis |

The Qwen Run K matrices illustrate why this matters. SFT Qwen 7B has a strong end-to-end row under strict prompting (93.6% contract completion, 96.74% abstention accuracy, 19 false abstentions, 62.43% conflict accuracy), while an oracle-conflict runtime row reaches 76.77% conflict accuracy. The latter result says that giving the correct Stage-2 label changes the conditional downstream behavior; it does not say end-to-end conflict accuracy is 76.77%.

Similarly, oracle-note and oracle-both document metrics can approach 100% because the document notes are supplied. That is expected by design and should be presented as an upper-bound diagnostic rather than an end-to-end document understanding result. Full definitions and reporting guidance are in [PROMPT_DESIGN_AND_ABLATIONS.md](PROMPT_DESIGN_AND_ABLATIONS.md).

## 11. What the ablations collectively support

The development evidence supports the following claims:

1. **Prompt robustness must be trained and measured.** Run D demonstrates that a public trace can be learned under sparse prompts, while the strict/runtime/minimal matrix makes the degree of dependence observable.
2. **Targeted boundary instruction is more reliable than broad label pressure.** The E-to-F transition shows why: broad conflict calibration helped 32B but harmed 7B, whereas a taxonomy-boundary drill produced a stronger shared direction.
3. **The relevant error may be in data geometry, not only prompt wording.** Run J's benchmark-aligned train/validation redesign transformed broad false abstention into localized, actionable slices.
4. **More exposure helps only when it covers the right missing region and is balanced.** K adds short answerable conflict cases; L adds missing short answerable no-conflict cases to correct the induced skew. This is a conditional coverage principle, not a universal row-count scaling law.
5. **Model-size and architecture matter.** F/G and L show that the same intervention can improve one model's trade-off and harm another's. A recipe should therefore be selected using the full measured outcome vector, not transferred blindly.

## 12. Claims that would overstate the evidence

The repository does **not** currently support the following claims:

- that each run isolates one causal variable;
- that any recipe is universally best across Qwen, Llama, and Mistral;
- that more message rows automatically improve all metrics;
- that oracle results are normal deployment performance;
- that the established answer-only results are a pure trace-removal ablation; or
- that the clean 862-example answer-only baseline has completed results.

The appropriate paper language is: *we iteratively refined the recipe using held-out error analysis; controlled sub-ablations identify useful and harmful directions, while later J/K/L revisions are targeted multi-component recipe changes motivated by persistent benchmark slices.*

## 13. Recommended paper presentation

An ACL-style paper can present this work without overclaiming by separating three exhibits:

1. **Historical controlled direction tests:** a compact D/E/F/G table showing prompt robustness, broad-vs-targeted conflict pressure, and model-specific trade-offs.
2. **Current recipe development:** a J/K/L table giving source-example counts, message-mixture counts, added short-context coverage, and the main benchmark outcomes for each available model.
3. **Diagnostic matrix:** end-to-end results as primary, with strict/runtime/minimal and oracle results used to explain sensitivity and stage-level bottlenecks.

Every table should include a contract/support gate near downstream metrics. Any missing adapter or result-family limitation should be disclosed using [METHOD_LIMITATIONS.md](METHOD_LIMITATIONS.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 14. Primary artifact map

| Evidence | Location |
| --- | --- |
| D/E/F and G comparative decision record | [sft_d_e_f_comparison_and_g_plan.md](../legacies/docs/experiments/sft_d_e_f_comparison_and_g_plan.md) |
| Run D retained strategy | [sft_run_d_prompt_robust_strategy.md](../legacies/docs/experiments/sft_run_d_prompt_robust_strategy.md) |
| Run F boundary ablation | [sft_run_f_boundary_guarded_strategy.md](../legacies/docs/experiments/sft_run_f_boundary_guarded_strategy.md) |
| Run G document-stabilized ablation | [sft_run_g_doc_stabilized_strategy.md](../legacies/docs/experiments/sft_run_g_doc_stabilized_strategy.md) |
| Run G source-hygiene ablation | [sft_run_g_7b_source_guarded_strategy.md](../legacies/docs/experiments/sft_run_g_7b_source_guarded_strategy.md) |
| Run J split and result record | [sft_run_j_benchmark_aug_strategy_and_results.md](../legacies/docs/experiments/sft_run_j_benchmark_aug_strategy_and_results.md) |
| Run K current record | [sft_run_k_short_context_targeted_strategy_and_results.md](experiments/sft_run_k_short_context_targeted_strategy_and_results.md) |
| Run L current record | [sft_run_l_boundary_rebalanced_strategy_and_status.md](experiments/sft_run_l_boundary_rebalanced_strategy_and_status.md) |
| Qwen Run K matrices | [Qwen 7B](../final_model_outputs/qwen7b_benchmark_matrix_analysis.md), [Qwen 32B](../final_model_outputs/qwen32b_benchmark_matrix_analysis.md) |
| Llama/Mistral Run L matrices | [Llama 3.1 8B](../final_model_outputs/llama8b_benchmark_matrix_analysis.md), [Mistral 7B](../final_model_outputs/mistral7b_benchmark_matrix_analysis.md) |
| Dataset geometry audits | [train/validation](dataset_audits/train_val_split_audit.md), [benchmark](dataset_audits/benchmark_736_audit.md) |
