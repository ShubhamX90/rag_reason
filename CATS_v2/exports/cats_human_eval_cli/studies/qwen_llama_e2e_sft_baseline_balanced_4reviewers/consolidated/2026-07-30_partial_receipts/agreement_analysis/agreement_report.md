# Human Eval Agreement Report

## Scope

This report summarizes the current human-eval consolidation and agreement analysis snapshot for the CATS v2 human study as of **July 30, 2026**, using the consolidated partial-receipts bundle:

- Consolidated snapshot: `2026-07-30_partial_receipts`
- Counted human reviews: submitted judgments only
- Current intake status:
  - `atharv`: complete
  - `parth`: complete
  - `samyek`: accepted partial return
  - `manan`: not yet received in this snapshot

The numbers below are therefore valid for the **current available human-eval pool**, but they should still be described in the paper as a **partial-intake agreement analysis** unless and until the remaining reviewer return is merged.

## Conflict Type Legend

- Type `1`: No conflict
- Type `2`: Complementary information
- Type `3`: Conflicting opinions or research outcomes
- Type `4`: Conflict due to outdated information
- Type `5`: Conflict due to misinformation

## Why These Metrics Are the Right Ones

The human-eval setup has three distinct judgment structures, so a single agreement statistic would be too blunt:

- **Behavior** is binary at the sample level, so we report raw agreement, Cohen's kappa, positive agreement, negative agreement, and Krippendorff's alpha.
- **STR** is also binary at the sample level, but coverage is lower because it only applies where the STR field is defined, so it is analyzed separately from behavior.
- **Faithfulness / Grounding (FG)** is inherently two-level:
  - claim-level pass/fail agreement
  - sample-level grounding-ratio agreement

This split is scientifically important. Claim-level FG asks whether humans and committee agree on individual citation checks. Ratio-level FG asks whether they agree on the overall amount of grounded content in the full answer. Reporting both makes the analysis substantially more defensible than a single collapsed faithfulness number.

We also include a **strict** and **soft** STR committee comparison:

- **Strict STR** counts only committee score `1.0` as positive.
- **Soft STR** counts committee score `0.5` or `1.0` as positive.

Because the gap between strict and soft results is very small, the current findings are not being driven by a fragile thresholding choice.

## Coverage

- Submitted human reviews consolidated: `450`
- Committee-matched human reviews: `450`
- Missing committee matches: `0`
- Double-reviewed samples for human-human IAA: `115`
- Behavior double-review units: `115`
- STR double-review units: `81`
- FG ratio double-review units: `115`
- FG claim double-review units: `182`
- Human-vs-committee behavior units: `450`
- Human-vs-committee STR units, strict: `319`
- Human-vs-committee STR units, soft: `319`
- Human-vs-committee FG ratio units: `450`
- Human-vs-committee FG claim units: `710`
- Unanimous-human subset for behavior consensus: `99`
- Unanimous-human subset for STR consensus: `72`
- Exact-human-match subset for FG ratio consensus: `103`
- Claim alignment issues found during audit: `0`

This coverage profile is strong enough to support meaningful conclusions, especially for human-vs-committee alignment, but the human-human IAA section should still be interpreted with the current overlap size in mind.

## Headline Results

### Human-Human Reliability

- **Behavior**: `n=115`, agreement `0.861`, Cohen's kappa `0.674`, Krippendorff alpha `0.673`
- **STR**: `n=81`, agreement `0.889`, Cohen's kappa `0.734`, Krippendorff alpha `0.732`
- **FG claim-level**: `n=182`, agreement `0.918`, Cohen's kappa `0.831`, Krippendorff alpha `0.831`
- **FG sample-level ratio**: `n=115`, exact-match `0.896`, MAE `0.074`, Pearson `0.854`, Spearman `0.853`

### Human vs Committee

- **Behavior**: `n=450`, agreement `0.767`, Cohen's kappa `0.441`
- **STR strict**: `n=319`, agreement `0.903`, Cohen's kappa `0.774`
- **STR soft**: `n=319`, agreement `0.906`, Cohen's kappa `0.779`
- **FG claim-level**: `n=710`, agreement `0.882`, Cohen's kappa `0.752`
- **FG sample-level ratio**: `n=450`, exact-match `0.842`, MAE `0.115`, Pearson `0.767`, Spearman `0.765`

### Human Consensus vs Committee

- **Behavior, unanimous-human subset**: `n=99`, agreement `0.818`, Cohen's kappa `0.552`
- **STR strict, unanimous-human subset**: `n=72`, agreement `0.931`, Cohen's kappa `0.818`
- **STR soft, unanimous-human subset**: `n=72`, agreement `0.931`, Cohen's kappa `0.818`
- **FG claim-level, unanimous-human subset**: `n=167`, agreement `0.928`, Cohen's kappa `0.850`
- **FG ratio, exact-human-match subset**: `n=103`, exact-match `0.903`, MAE `0.066`

### Individual Committee Judges vs Human Behavior Labels

- **deepseek-r1-distill-32b**: `n=450`, agreement `0.767`, Cohen's kappa `0.387`
- **mistral-small-4**: `n=450`, agreement `0.638`, Cohen's kappa `0.202`
- **qwen3.5-397b-a17b**: `n=449`, agreement `0.764`, Cohen's kappa `0.436`

## Scientific Interpretation

### 1. Human labels are reliable enough to function as a serious reference set

The human-human agreement profile is strong across all three axes, with the cleanest reliability on FG, then STR, then behavior:

- FG claim-level kappa `0.831` is the strongest result in the study and indicates that humans are highly consistent when judging concrete citation support at the claim level.
- STR kappa `0.734` is also strong, suggesting that sentence-targeted retrieval judgments are not overly subjective under the current rubric.
- Behavior kappa `0.674` is lower than FG and STR, but still substantial enough to support meaningful downstream analysis.

This ordering makes conceptual sense. FG claim checks are the most local and operationalized judgments. STR is more contextual but still structurally constrained. Behavior is the most global and interpretive axis, so some extra variance is expected even under a good rubric.

### 2. The local committee aligns very well with humans on STR and FG

The strongest committee-alignment evidence is on STR and FG:

- STR strict kappa `0.774`
- STR soft kappa `0.779`
- FG claim-level kappa `0.752`
- FG ratio exact-match `0.842`, MAE `0.115`, Pearson `0.767`

This is the core evidence supporting the claim that the local LLM committee can serve as a credible large-scale evaluator for retrieval grounding and support-sensitive reasoning dimensions. In particular:

- The near-identical STR strict and soft results show that committee-human alignment is not being propped up by the ambiguous `0.5` bucket.
- FG is strong at both granularities, which matters because it means the committee is not only flagging individual grounded claims similarly to humans, but also arriving at similar holistic grounding proportions at the answer level.

For an ACL paper, this is likely the safest and strongest take-away from the present analysis.

### 3. Behavior is the hardest axis, and the committee is directionally aligned but not yet as human-like as it is on STR/FG

Behavior agreement is the main caution area:

- Human-human behavior kappa: `0.674`
- Human-vs-committee behavior kappa: `0.441`
- Human-consensus-vs-committee behavior kappa: `0.552`

This pattern is still useful, but it must be described carefully. The committee is **not random** on behavior and is clearly capturing meaningful signal, but it is materially less aligned with humans here than on STR or FG.

The consensus analysis is especially informative. When humans fully agree, committee behavior alignment rises from kappa `0.441` to `0.552`. That suggests a meaningful portion of the committee-human gap is concentrated in inherently ambiguous or rubric-sensitive cases, rather than reflecting pure evaluator failure everywhere.

### 4. The committee is stronger as an aggregate than the weakest individual judge, but behavior remains ensemble-limited

For behavior:

- Committee pooled kappa vs humans: `0.441`
- Best single judge observed: `qwen3.5-397b-a17b` at `0.436`
- DeepSeek judge: `0.387`
- Mistral judge: `0.202`

This supports an important paper claim: the committee aggregation is doing useful work and substantially improves over the weakest constituent judge. At the same time, it is only slightly better than the strongest individual judge on this axis, which implies that behavior performance is currently constrained more by rubric difficulty and model interpretation mismatch than by simple vote-aggregation failure.

## Difficulty-Specific Patterns

The conflict-category breakdown is one of the most informative parts of the audit.

### Behavior

Human-human behavior agreement by conflict category:

- Type `1`: kappa `0.625`
- Type `2`: kappa `0.451`
- Type `3`: kappa `0.712`
- Type `4`: kappa `0.596`
- Type `5`: kappa `0.904`

Human-vs-committee behavior agreement by conflict category:

- Type `1`: kappa `0.357`
- Type `2`: kappa `0.357`
- Type `3`: kappa `0.637`
- Type `4`: kappa `0.450`
- Type `5`: kappa `0.141`

The most important finding here is **Type 5**:

- Humans agree with each other extremely strongly on Type `5` behavior cases.
- The committee aligns poorly with humans on that same category.

That is a classic signature of a **systematic evaluator mismatch**, not mere human noise. In other words, Type `5` should be treated as a likely rubric-interpretation failure mode for the committee rather than a generally ambiguous problem class.

By contrast, Type `2` is difficult for both humans and committee, which suggests genuine task ambiguity rather than a uniquely committee-specific weakness.

### STR

Human-human STR agreement by conflict category:

- Type `1`: kappa `0.341` on small `n`
- Type `2`: kappa `0.478`
- Type `4`: kappa `0.881`
- Type `5`: kappa `0.909`

Human-vs-committee STR strict agreement by conflict category:

- Type `1`: kappa `0.766`
- Type `2`: kappa `0.658`
- Type `4`: kappa `0.859`
- Type `5`: kappa `0.752`

The low human-human STR kappa for Type `1` should not be overinterpreted in isolation because that slice is relatively small and label-skewed. Overall, STR remains robust, and committee alignment is high across all populated categories.

### FG

Human-human FG claim-level agreement by conflict category:

- Type `1`: kappa `1.000`
- Type `2`: kappa `0.879`
- Type `3`: kappa `0.755`
- Type `4`: kappa `0.650`
- Type `5`: kappa `0.917`

Human-vs-committee FG claim-level agreement by conflict category:

- Type `1`: kappa `0.862`
- Type `2`: kappa `0.784`
- Type `3`: kappa `0.738`
- Type `4`: kappa `0.517`
- Type `5`: kappa `0.860`

Human-vs-committee FG ratio agreement by conflict category:

- Type `1`: exact-match `0.925`, MAE `0.056`
- Type `2`: exact-match `0.822`, MAE `0.135`
- Type `3`: exact-match `0.761`, MAE `0.149`
- Type `4`: exact-match `0.767`, MAE `0.185`
- Type `5`: exact-match `0.933`, MAE `0.052`

FG is strong overall, but Type `4` is the weakest faithfulness slice. That makes Type `4` the best candidate for targeted qualitative error analysis in the paper, because it likely captures cases where citation grounding is harder to operationalize consistently across evaluators.

## What We Can Safely Claim in the Paper

The current evidence strongly supports the following claims:

- Human annotators show substantial agreement overall, with the strongest reliability on faithfulness and strong reliability on STR.
- The local LLM committee aligns well with humans on STR and FG, including both claim-level and answer-level grounding views.
- Committee-human agreement improves further on the unanimous-human subset, showing that the committee is most trustworthy on cases that humans themselves find unambiguous.
- The committee is a scientifically defensible large-scale evaluator for retrieval-grounding dimensions, especially where fully manual evaluation is expensive.

The current evidence supports only a more careful version of the following claim:

- The committee is useful for behavior evaluation, but behavior should be presented as the most challenging and least human-aligned axis, with explicit acknowledgment of category-specific mismatch.

The current evidence does **not** justify saying that the committee is uniformly human-equivalent across every judgment type. That would overstate the behavior results.

## Recommended Framing for ACL

The most defensible paper framing is:

1. Present **human-human agreement first** to establish that the annotation protocol is coherent.
2. Present **committee-vs-human STR and FG alignment** as the central validation result.
3. Present **behavior** as a harder semantic judgment axis where the committee captures substantial signal but still exhibits systematic disagreement pockets.
4. Use the **consensus-subset analysis** to argue that committee trustworthiness increases on unambiguous examples.
5. Use the **conflict-category breakdown** to show that evaluator failure is not homogeneous and to motivate future rubric refinement.

## Limitations of the Current Snapshot

- This snapshot reflects **partial reviewer intake**, not the final completed human study.
- `manan` is absent from the current consolidated set.
- `samyek` is intentionally accepted as a partial return.
- Human-human IAA is therefore being estimated from the currently available overlap, not from the final intended overlap design.
- Some subgroup slices have modest sample sizes, so per-category kappas should be interpreted as diagnostic rather than final headline statistics.

These limitations do not invalidate the current analysis, but they should be disclosed transparently.

## Files to Use for Follow-Up Analysis

- Main metric log: [agreement_metric_log.json](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_partial_receipts/agreement_analysis/agreement_metric_log.json)
- Coverage summary: [coverage_summary.json](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_partial_receipts/agreement_analysis/coverage_summary.json)
- Behavior disagreement audit: [behavior_review_disagreements.jsonl](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_partial_receipts/agreement_analysis/behavior_review_disagreements.jsonl)
- Behavior consensus disagreement audit: [behavior_consensus_disagreements.jsonl](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_partial_receipts/agreement_analysis/behavior_consensus_disagreements.jsonl)
- STR disagreement audit: [str_strict_review_disagreements.jsonl](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_partial_receipts/agreement_analysis/str_strict_review_disagreements.jsonl)
- FG disagreement audit: [fg_claim_review_disagreements.jsonl](/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_partial_receipts/agreement_analysis/fg_claim_review_disagreements.jsonl)

## Bottom Line

At the current July 30, 2026 snapshot, the human study already provides strong evidence that the CATS v2 local committee is a credible evaluator for **STR** and **faithfulness / grounding**, and a directionally useful but still imperfect evaluator for **behavior**. The cleanest paper claim is therefore not that the committee fully replaces humans everywhere, but that it tracks humans well on the most operationalized retrieval-grounding dimensions and remains informative, though less fully aligned, on the hardest global reasoning judgments.
