# Current CONFLICTS-style benchmark build

> **Read the dataset card first.** The detailed current benchmark description,
> release statistics, source provenance, refusal track, selection criteria, and
> integrity notes are in [`BENCHMARK_DATASET_DESCRIPTION.md`](BENCHMARK_DATASET_DESCRIPTION.md).
> The Tavily search/fetch/windowing protocol is in
> [`TAVILY_RETRIEVAL_METHODOLOGY.md`](TAVILY_RETRIEVAL_METHODOLOGY.md). This
> document is the concise operational build guide.

This document describes the retained benchmark-build path. Older pilot, 650-query,
and smoke-test artifacts are preserved under `legacies/` and are intentionally not
part of the clean reproduction path.

The reviewer-facing released benchmark is separate from these build artifacts:
`data/releases/benchmark_dataset_v2/benchmark_final_v2_holdout_clean_736.jsonl`
(736 records). The complete current release is
`data/releases/benchmark_dataset_v2/benchmark_final_v2.jsonl` (933 records).
The files in this document remain important for reproducing how the annotation
pipeline produced and filtered the internal benchmark candidate pool.

## Current end-to-end flow

1. Build a deduplicated 2,000-query candidate pool from the pinned source data.
2. Retrieve and rank evidence, retaining the current 2,000-query retrieval artifacts.
3. Use the five-document deterministic evidence subset when the human preselection
   workflow requires the reduced display set.
4. Run first-pass human preselection and export the selected 800 non-refusal rows.
5. Run the stagewise multi-LLM committee in benchmark mode on those 800 rows.
6. Combine the retained 800 non-refusal and 200 refusal stagewise outputs to
   form the 1,000-row internal build artifact. This historical artifact is not
   the current 933-row release or its 736-row reviewer-facing holdout.

Source-dataset membership is not a gold conflict label. Conflict type is assigned
after retrieval because retrieved evidence may not preserve the source query's
intended relation.

## Current retained artifacts

- Candidate pool: `data/benchmark_build/candidates/query_pool_2000.jsonl`
- Candidate manifest: `data/benchmark_build/candidates/manifest_2000.json`
- Current retrieval outputs: `data/benchmark_build/retrieved/full2000_*`
- Human-review materials: `human_reviews/benchmark/`
- Final benchmark: `data/benchmarks/final_benchmark_2026-06-22/benchmark_final.jsonl`
- Final benchmark manifest: `data/benchmarks/final_benchmark_2026-06-22/benchmark_final_manifest.json`
- Pinned external-source map: `data/external_sources/source_manifest.json`

The source repositories are reduced to the exact files used by the current
candidate builder, plus their licenses/readmes. Unused source-repository code and
history remain recoverable under `legacies/external_sources/`.

## Candidate construction

```bash
python3 scripts/build_conflicts_benchmark_candidates.py \
  --target 2000 \
  --output data/benchmark_build/candidates/query_pool_2000.jsonl
```

The builder reads the pinned source files listed in
`data/external_sources/source_manifest.json` and excludes queries already present
in the retained CONFLICTS/training data.

## Retrieval

The retrieval script supports Google, Tavily, and DuckDuckGo providers. Google is
the closest provider to the paper; Tavily is the retained practical alternative.
The API keys are supplied through environment variables and are never stored in
the repository.

```bash
export GOOGLE_API_KEY=...
export GOOGLE_CSE_ID=...
python3 scripts/retrieve_conflicts_benchmark_docs.py \
  --input data/benchmark_build/candidates/query_pool_2000.jsonl \
  --output data/benchmark_build/retrieved/benchmark2000_retrieved.jsonl \
  --search-provider google \
  --top-k 10 \
  --window-selector tasb
```

For the retained Tavily-style route:

```bash
export TAVILY_API_KEY=...
python3 scripts/retrieve_conflicts_benchmark_docs.py \
  --input data/benchmark_build/candidates/query_pool_2000.jsonl \
  --output data/benchmark_build/retrieved/benchmark2000_retrieved.jsonl \
  --search-provider tavily \
  --top-k 10 \
  --search-max-results 20 \
  --drop-blocked \
  --min-window-words 100 \
  --window-selector tasb
```

For the deterministic reduced evidence view:

```bash
python3 scripts/build_benchmark_doc_subset.py
```

The retained subset is
`data/benchmark_build/retrieved/full2000_fresh_annotation_candidates_5docs_2top5_3bottom5_seed62002.jsonl`.

## Human review

The two review populations are deliberately separate:

- Training conflict-type review: `human_reviews/training/`
- Benchmark first- and second-pass review: `human_reviews/benchmark/`

Create or resume benchmark first-pass assignments from the repository root:

```bash
./scripts/run_benchmark_human_preselection.sh --make-assignments --reviewers 7 --seed 62002
./scripts/run_benchmark_human_preselection.sh
```

First-pass reviews are written to
`human_reviews/benchmark/first_pass/reviews/`. The current selected-800 material
is under `human_reviews/benchmark/first_pass/benchmark_selection_final/`.

Second-pass review is run with:

```bash
./scripts/run_benchmark_human_second_review.sh
```

It reads the first-pass review material and writes to
`human_reviews/benchmark/second_pass/second_reviews/`.

Training review is run with:

```bash
./scripts/run_training_conflict_type_review.sh
```

It reads `human_reviews/training/assignments/` and writes reviewer progress to
`human_reviews/training/reviews/`.

## Latest committee annotation path

The current committee is implemented in `src/` and invoked by the retained
multi-model runners under `scripts/`. The operational configurations are under
`configs/local_committee/`, with Slurm launchers under
`slurm/sharanga/local_committee/`. Prompts are under `prompts/`.

The retained benchmark inputs and outputs are documented by their run directories;
the final benchmark combines the selected non-refusal and refusal populations.
Do not substitute the files under `legacies/` when reproducing the current run.
