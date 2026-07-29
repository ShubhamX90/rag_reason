# Quarantined legacy material

This directory is a reversible quarantine area created during the repository
hygiene pass. No legacy file was deleted. The material here is excluded from the
current reproduction path and is retained only for provenance, historical
comparison, or recovery.

The clean workflow is rooted at the repository top level and uses the retained
paths documented in `README.md` and `docs/conflicts_benchmark_build.md`. In
particular, use:

- `data/final_annotations/` for the current 658-example annotation outputs
- `data/splits/92p5_7p5/` for the current shared split
- `data/benchmarks/final_benchmark_2026-06-22/` for the current benchmark
- `human_reviews/training/` and `human_reviews/benchmark/` for current reviews
- `src/`, `scripts/`, `configs/`, `prompts/`, and `slurm/` for current logic

The quarantine contains superseded runners and prompts, old split and benchmark
artifacts, pilot/smoke material, duplicate export packs, unused external-source
files and nested repository history, mixed archives, generated metadata, and
other files that are not required by the latest retained workflow.
