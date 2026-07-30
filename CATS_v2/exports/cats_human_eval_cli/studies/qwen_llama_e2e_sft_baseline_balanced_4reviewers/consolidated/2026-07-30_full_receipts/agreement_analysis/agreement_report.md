# Human Eval Agreement Report

## Study Construction

- Study snapshot label: `2026-07-30_full_receipts`
- Selection seed: `20260715`
- Source family: `inputs/prepped_model_eval_inputs/benchmark_set_all_modes/<model>/e2e/<prompt>/<train_type>/input.jsonl`
- Task variant used for this study: `e2e only`
- Models included: `qwen7b, llama8b`
- Prompts included: `minimal, runtime, strict`
- Train types included: `sft, baseline`
- Full selected study sample count: `350`
- Target human review slots at assignment time: `700`
- Selection excluded correct-refusal rows before balancing and assignment.
- The two 30-sample cells were: `qwen7b|minimal|baseline, llama8b|runtime|sft`
- Cell-level selected counts across the 12 study cells: `llama8b|minimal|baseline=29, llama8b|minimal|sft=29, llama8b|runtime|baseline=29, llama8b|runtime|sft=30, llama8b|strict|baseline=29, llama8b|strict|sft=29, qwen7b|minimal|baseline=30, qwen7b|minimal|sft=29, qwen7b|runtime|baseline=29, qwen7b|runtime|sft=29, qwen7b|strict|baseline=29, qwen7b|strict|sft=29`
- Selected source-row file: `/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/admin/selected_source_rows.jsonl`
- Selection audit file: `/Users/shubhammishra/Desktop/rag_reason/CATS_v2/exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/admin/assignment_audit.json`

## Sample Distribution

### Full 350 Selected Samples

- By model: `llama8b=175, qwen7b=175`
- By prompt: `minimal=117, runtime=117, strict=116`
- By train type: `baseline=175, sft=175`
- By conflict type id: `1=70, 2=70, 3=70, 4=70, 5=70`
- This 350-sample pool is exactly balanced across model, train type, and conflict category, with only a 117/117/116 prompt split due to the indivisible total of 350.

### Fully Complete 300-Sample Double-Reviewed Subset

- By model: `llama8b=153, qwen7b=147`
- By prompt: `minimal=99, runtime=101, strict=100`
- By train type: `baseline=149, sft=151`
- By conflict type id: `1=63, 2=60, 3=58, 4=60, 5=59`
- The fully complete 300-sample subset stays close to balanced, so its agreement estimates are not being driven by a single model, prompt, train type, or conflict slice.

## Conflict Type Legend

- Type `1`: No conflict
- Type `2`: Complementary information
- Type `3`: Conflicting opinions or research outcomes
- Type `4`: Conflict due to outdated information
- Type `5`: Conflict due to misinformation

## Coverage

- Submitted human reviews currently consolidated: `650`
- Double-reviewed samples available for human-human IAA: `300`
- Behavior double-review units: `300`
- STR double-review units: `215`
- FG ratio double-review units: `300`
- FG claim double-review units: `477`
- Human-vs-committee behavior units: `650`
- Human-vs-committee STR units (strict): `465`
- Human-vs-committee FG claim units: `1029`
- Behavior review-level disagreements queued for audit: `166`
- Committee-internal judge cache available locally: `True`

## Descriptive Outcomes

### What Humans Thought on the Fully Complete 300-Sample Subset

- Behavior adherence: `404/600` positive human reviews, rate `0.673`
- Behavior sample breakdown: `180` unanimous positive, `76` unanimous negative, `44` split
- STR on applicable samples: `297/430` positive human reviews, rate `0.691`
- STR sample breakdown: `138` unanimous positive, `56` unanimous negative, `21` split
- FG claim checks: `601/954` supported human claim judgments, rate `0.630`
- Human mean FG ratio across the 300-sample subset: `0.641`

### What the Committee Thought on the Same 300-Sample Subset

- Behavior adherence: `213/300` positive committee decisions, rate `0.710`
- STR strict positives: `145/215`, rate `0.674`
- STR soft positives: `147/215`, rate `0.684`
- Committee mean FG ratio across the same 300-sample subset: `0.663`

## Human-Human IAA

- Behavior: `n=300`, agreement `0.853`, Cohen's kappa `0.669`, Krippendorff alpha `0.667`
- STR: `n=215`, agreement `0.902`, Cohen's kappa `0.773`, Krippendorff alpha `0.772`
- FG claim-level: `n=477`, agreement `0.897`, Cohen's kappa `0.780`, Krippendorff alpha `0.780`
- FG sample-level ratio: `n=300`, exact-match `0.860`, MAE `0.096`, Pearson `0.807`, Spearman `0.796`

## Human vs Committee

- Behavior: `n=650`, agreement `0.745`, Cohen's kappa `0.407`
- STR primary (committee exact-match only as positive): `n=465`, agreement `0.908`, Cohen's kappa `0.790`
- STR sensitivity (committee partial-or-exact as positive): `n=465`, agreement `0.912`, Cohen's kappa `0.798`
- FG claim-level: `n=1029`, agreement `0.899`, Cohen's kappa `0.783`
- FG sample-level ratio: `n=650`, exact-match `0.865`, MAE `0.095`, Pearson `0.805`, Spearman `0.798`

## Human Consensus vs Committee

- Behavior on unanimous human subset: `n=256`, agreement `0.789`, Cohen's kappa `0.483`
- STR strict on unanimous human subset: `n=194`, agreement `0.948`, Cohen's kappa `0.876`
- STR soft sensitivity on unanimous human subset: `n=194`, agreement `0.954`, Cohen's kappa `0.888`
- FG claim-level on unanimous human subset: `n=428`, agreement `0.949`, Cohen's kappa `0.888`
- FG ratio on exact-human-match subset: `n=258`, exact-match with committee `0.938`, MAE `0.045`

## Comparison Caveat

- Each sample in the human study was reviewed by exactly `2` humans, while each sample in the local committee was judged by `3` LLM judges.
- Because of that design asymmetry, human-human and committee-internal multirater coefficients should not be treated as directly interchangeable apples-to-apples quantities.
- The fairest direct comparison is pairwise agreement on the same sample slice, especially the fully double-reviewed `300`-sample subset.
- On that `300`-sample subset, human-human kappa is `0.669` for behavior, `0.773` for STR, and `0.780` for FG claim checks.
- The corresponding mean pairwise LLM-LLM kappa values are `0.470` for behavior, `0.687` for STR, and `0.871` for FG claim checks.
- This supports a cautious paper claim: the committee is strongly reliable on STR and grounding, but only moderately stable on behavior, and humans remain more consistent than the LLM judges on behavior.

## Paper-Ready Claims And Cautions

- The strongest validation result is not on holistic behavior, but on STR and grounding. Those are the dimensions on which both human-versus-committee agreement and committee-internal agreement are strongest.
- The safest paper claim is therefore conditional: the committee is a defensible proxy for human judgment on STR and faithfulness-oriented checks, while behavior still requires more caution and supporting manual analysis.
- On the fully complete `300`-sample subset, humans labeled behaviorally aligned at rate `0.673`, while the committee labeled behaviorally aligned at rate `0.710`. That gap is not enormous, but it does show the committee is slightly more permissive than humans on behavior.
- Because the study uses `2` humans per sample and `3` LLM judges per sample, pairwise comparisons should carry the main argumentative weight when comparing human and committee reliability; multirater alpha is best used as within-family context.
- The report therefore supports a nuanced conclusion: committee-based evaluation is strongest for STR and grounding, reasonably informative but less settled for behavior, and not yet a full drop-in replacement for human behavioral judgment.

## Behavior Error Analysis Priorities

- Review-level behavior disagreements: `166`
- By conflict type: `1=18, 2=38, 3=23, 4=33, 5=54`
- By prompt: `minimal=41, runtime=51, strict=74`
- By train type: `baseline=96, sft=70`
- Direction overall: `human_neg_committee_pos=98, human_pos_committee_neg=68`
- Consensus-only disagreements after restricting to unanimous-human samples: `54`
- Consensus disagreements by conflict type: `1=7, 2=10, 3=6, 4=10, 5=21`
- Type `5` is the highest-priority slice for manual qualitative analysis. It has the largest behavior disagreement mass and remains the weakest slice even after restricting to unanimous-human samples.
- Type `2` is especially diagnostic because the disagreement is strongly asymmetric there: the committee is much more likely than humans to call the answer behaviorally aligned, which suggests over-crediting of partial reconciliation in complementary-information cases.
- Strict-prompt cases deserve focused review because they contribute the largest number of review-level behavior disagreements.
- Baseline outputs deserve somewhat more behavior-focused audit attention than SFT outputs because their disagreement mass is larger.
- The most useful files for targeted manual follow-up are `behavior_review_disagreements.jsonl` for all mismatches and `behavior_consensus_disagreements.jsonl` for the cleaner subset where the two humans already agree with each other.

## Committee Internal Agreement

### All 350 Selected Study Samples

- Behavior: `n=350`, Krippendorff alpha `0.470`
- Behavior, qwen vs mistral: agreement `0.746`, Cohen's kappa `0.443`
- Behavior, qwen vs deepseek: agreement `0.863`, Cohen's kappa `0.635`
- Behavior, mistral vs deepseek: agreement `0.729`, Cohen's kappa `0.377`
- STR strict: `n=250`, Krippendorff alpha `0.665`
- STR strict, qwen vs deepseek: agreement `0.944`, Cohen's kappa `0.875`
- STR soft matches STR strict exactly on this study slice, indicating no partial-recall boundary effect inside the cached judge outputs.
- FG claim-level: `n=552`, Krippendorff alpha `0.880`
- FG claim-level, mistral vs deepseek: agreement `0.951`, Cohen's kappa `0.892`
- FG ratio, mistral vs deepseek: `n=350`, exact-match `0.931`, MAE `0.044`, Pearson `0.913`

### Fully Double-Reviewed 300-Sample Subset

- Behavior: `n=300`, Krippendorff alpha `0.457`
- STR strict: `n=215`, Krippendorff alpha `0.678`
- STR soft again matches STR strict exactly on the fully double-reviewed subset.
- FG claim-level: `n=477`, Krippendorff alpha `0.871`
- FG ratio, mistral vs deepseek: `n=300`, exact-match `0.930`, MAE `0.045`, Pearson `0.906`

## Individual Committee Judges

- Behavior vs `local/deepseek-r1-distill-32b`: `n=650`, agreement `0.735`, Cohen's kappa `0.339`
- Behavior vs `local/mistral-small-4`: `n=650`, agreement `0.643`, Cohen's kappa `0.229`
- Behavior vs `local/qwen3.5-397b-a17b`: `n=649`, agreement `0.743`, Cohen's kappa `0.405`

## Interpretation Notes

- All four reviewer returns are now accounted for in the full snapshot. The remaining incompleteness is coverage-level only: `samyek` is accepted as a 50-submission partial return.
- Behavior and STR are treated as binary labels.
- FG is analyzed in two complementary ways: claim-level binary pass/fail and sample-level grounding-ratio agreement.
- For committee STR comparison, the primary analysis treats committee `0.5` partial matches conservatively as non-matches, with a separate sensitivity analysis where partial matches count as positive.
- The committee-internal analysis is reconstructed from cached single-judge staged outputs for the exact 12 study slices used in the human-eval package, covering both baseline and SFT runs.
- Committee-internal agreement is strongest on grounding and STR, but materially weaker on behavior. That mirrors the broader pattern that behavior is the least operationalized and most interpretation-sensitive judgment axis.
- Within behavior, qwen and deepseek are the closest pair, while mistral is the least aligned with the other two judges. This suggests the ensemble's behavior instability is driven less by pure label noise and more by one judge's rubric interpretation drift.
- For paper writing, pairwise comparisons are the primary fair bridge between the `2-human` and `3-LLM` setups; multirater alpha should be presented as within-family context rather than as a direct human-versus-committee contest.
- Committee-alignment claims for the paper should emphasize the exact coverage subset used for each metric rather than implying that every selected human-eval sample has complete double-human review.
- The disagreement slices in `behavior_review_disagreements.jsonl`, `behavior_consensus_disagreements.jsonl`, `str_strict_review_disagreements.jsonl`, and `fg_claim_review_disagreements.jsonl` are intended to support manual error analysis before drafting final paper claims.