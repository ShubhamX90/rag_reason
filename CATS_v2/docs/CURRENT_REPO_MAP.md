# Current Repo Map

This repository has been curated so that the main tree keeps the current CATS v2 evaluation pipeline, current benchmark/val data, current benchmark result artifacts, and the current human-eval package.

Anything older, superseded, or mainly archival has been moved under `legacies/` without deletion.

## Canonical Main Paths

- Core evaluator code: `rag_eval/`, `run_evaluation.py`
- Main configs:
  - `configs/default.yaml`
  - `configs/val_tier2.yaml`
  - `configs/val_tier2_mixed.yaml`
  - `configs/val_tier2_cli.yaml`
  - `configs/val_tier2_local_openai.yaml`
  - `configs/val_tier2_local_openai_2xh200_fallback.yaml`
  - `configs/benchmark_local_openai_3judge_qwen397.yaml`
  - `configs/local_staged/`
  - `configs/local_staged_gold_ceiling/`
- Canonical datasets:
  - `data/splits/92p5_7p5/`
  - `data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl`
  - `data/ceiling_pilots/val_stage3_gold_expected_as_model_output.jsonl`
- Canonical raw model outputs:
  - `final_model_outputs/`
- Canonical prepared evaluator inputs:
  - `inputs/prepped_model_eval_inputs/benchmark_set_all_modes/`
  - `inputs/prepped_model_eval_inputs/other_techniques/`
  - `inputs/prepped_model_eval_inputs/other_techniques_fixed/`
- Canonical benchmark result artifacts:
  - `outputs/benchmark_local_committee_3judge/benchmark_set_all_modes/`
  - `outputs/benchmark_local_committee_3judge/other_techniques/`
  - `outputs/benchmark_local_committee_3judge/other_techniques_fixed/`
  - `outputs/benchmark_local_committee_3judge/master_results/`
  - `outputs/benchmark_local_committee_3judge/citation_quality_posthoc/`
- Canonical val / ceiling outputs:
  - `outputs/val_gold_ceiling_local_staged/`
- Human eval package and latest study artifacts:
  - `exports/cats_human_eval_cli/`

## Documentation

- Repository entry point: `README.md`
- Authored paper-facing documentation: `docs/`
- Human-evaluation design and implementation: `docs/HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md`
- Latest human agreement report: `exports/cats_human_eval_cli/studies/qwen_llama_e2e_sft_baseline_balanced_4reviewers/consolidated/2026-07-30_full_receipts/agreement_analysis/agreement_report.md`

## Notes

- The current train/val split is `609 / 49` in `data/splits/92p5_7p5/`.
- The canonical benchmark holdout is `736` rows.
- The main local benchmark committee flow is centered on `scripts/watch_benchmark_file_pipeline.py` and the Sharanga assets in `slurm/sharanga/local_committee/`.
- The older `outputs/val_codex_deepseek_v4flash/` artifact set was moved to `legacies/` during final submission checks because its JSON and markdown payloads were internally inconsistent.
- `legacies/` is intentionally removable for clean submission packaging once you no longer need older material.
