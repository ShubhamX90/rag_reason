# CATS Human Eval CLI

Standalone human-evaluation CLI/TUI package for CATS-style benchmark review.

This package is designed so human evaluators can act as replacements for LLM
judges at collection time while preserving the current CATS per-sample metric
framing as closely as possible:

- GR / refusal applicability remains deterministic.
- Behavior uses the current conflict-type rubric.
- FG uses the same deterministic claim extraction used by CATS.
- STR uses the same "asserts the gold answer" framing used by CATS.

What this package already covers:

- standalone study-bundle creation
- balanced reviewer assignments
- interactive per-sample human judgment capture
- draft / submitted state persistence
- raw and enriched judgment export

What is intentionally still separate:

- final multi-human aggregation into committee-equivalent scores
- downstream reconciliation of multiple reviewers into a single final metric output

For reviewer-facing deployment, reviewers should read:

- `REVIEWER_USER_MANUAL.md`

carefully before starting.

## Current shape

This first version is implemented as a Rich-based interactive CLI because the
current local environment already has `rich`, `typer`, and `pyyaml` installed.
It is intentionally standalone and does not import the parent repo's runtime.

## Package layout

- `cats_human_eval/`: package source
- `pyproject.toml`: packaging metadata

## Main commands

- `cats-human study-init`
- `cats-human build-assignments`
- `cats-human judge`
- `cats-human review`
- `cats-human export`

## Reviewer launch

For reviewer-facing bundles, the expected entrypoint is:

- `./run_reviewer.sh`

That launcher prompts once for the reviewer's first name, resolves it against
the registered reviewer list in the bundled study, and then starts the normal
judging session automatically.

If a reviewer later needs to reopen already-submitted items for inspection or
editing, the same launcher can be used in review mode:

- `./run_reviewer.sh review`

## Example

```bash
cd exports/cats_human_eval_cli
python -m cats_human_eval --help
python -m cats_human_eval study-init \
  --input-jsonl ../../inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen7b/e2e/minimal/sft/input.jsonl \
  --study-dir ./demo_study \
  --study-name "qwen7b_sft_minimal_demo"
python -m cats_human_eval build-assignments \
  --study-dir ./demo_study \
  --reviewer shubham \
  --reviewer reviewer2 \
  --sample-limit 20
python -m cats_human_eval judge --study-dir ./demo_study --reviewer shubham
```

## Study directory

`study-init` creates a standalone study bundle:

- `study.yaml`
- `data/samples.jsonl`
- `assignments/`
- `state/judgments.sqlite3`
- `state/events.jsonl`
- `exports/`

This lets the package be copied elsewhere without requiring the rest of this
repo, provided the package dependencies are available.
