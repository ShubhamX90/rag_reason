# CATS v2 Documentation Index

This folder contains the authored, paper-facing technical documentation for
the current CATS v2 repository. The root [`README.md`](../README.md) remains
the repository entry point and should be read first.

## Canonical technical documents

| Document | Purpose |
| --- | --- |
| [`CATS_METRICS_METHODOLOGY.md`](CATS_METRICS_METHODOLOGY.md) | Formula-level definitions, applicability rules, denominators, edge cases, and scientific defense for every metric in the master results workbook. |
| [`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md) | Current hierarchical CATS aggregate, gated harmonic construction, balanced and prevalence summaries, alternatives, and reviewer-facing rationale. |
| [`LOCAL_LLM_COMMITTEE_DESCRIPTION.md`](LOCAL_LLM_COMMITTEE_DESCRIPTION.md) | Scientific and logical description of the local judge committee, task-specific priorities, voting, limitations, and ACL-ready methods language. |
| [`LOCAL_COMMITTEE_GUIDE.md`](LOCAL_COMMITTEE_GUIDE.md) | Operational reproduction guide for local serving, prompts, cache staging, Slurm execution, validation, recovery, and provenance. |
| [`CURRENT_REPO_MAP.md`](CURRENT_REPO_MAP.md) | Canonical current paths and the boundary between main artifacts and archived material. |
| [`HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md`](HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md) | End-to-end human evaluation design, package implementation, receipt sanitization, consolidation, agreement metrics, current snapshot, and paper-use guidance. |
| [`EVALUATOR_IMPLEMENTATION.md`](EVALUATOR_IMPLEMENTATION.md) | Executable evaluator contract, input schema, per-sample call graph, committee/cache behavior, output schema, and active versus legacy paths. |
| [`REPRODUCTION_AND_ARTIFACT_PROVENANCE.md`](REPRODUCTION_AND_ARTIFACT_PROVENANCE.md) | End-to-end lineage from datasets and model exports through the 108-row scope, staged runs, audits, post-hoc citation analysis, and workbook export. |
| [`ENVIRONMENT_AND_SETUP.md`](ENVIRONMENT_AND_SETUP.md) | Python dependencies, secrets, NLTK data, local serving, human-review, workbook, offline, and setup verification contracts. |

## Documentation boundary

Not every Markdown file in the repository belongs in this central folder.
Several Markdown files are deliberately kept beside the artifact they
describe:

- `exports/cats_human_eval_cli/README.md` and `REVIEWER_USER_MANUAL.md` are
  package-local documents copied into reviewer bundles. Moving them would make
  the standalone bundle less reproducible.
- Markdown reports under `outputs/` and under human-evaluation
  `consolidated/` directories are generated result artifacts. Their location
  is part of the result provenance and their relative links must remain stable.
- `legacies/` contains archived documentation and generated reports. It remains
  removable as a whole for a clean submission copy, but its internal hierarchy
  is intentionally preserved.
- `prompts/README.md` and `prompts/behavior_rubric.md` stay with the active
  prompt bundle so that the prompt directory is self-describing when copied.
- Operational README files under scripts and Slurm trees stay with those
  runnable trees.

This is a structural rule, not a second set of scientific documents. The
current authored documentation is centralized here without flattening
package-local, generated, or legacy provenance.

## Recommended reading order

1. [`../README.md`](../README.md) for repository scope and the reproduction
   map.
2. [`CURRENT_REPO_MAP.md`](CURRENT_REPO_MAP.md) for canonical paths.
3. [`CATS_METRICS_METHODOLOGY.md`](CATS_METRICS_METHODOLOGY.md) for component
   metrics and denominators.
4. [`CATS_AGGREGATE_LOGIC.md`](CATS_AGGREGATE_LOGIC.md) for the secondary CATS
   summaries.
5. [`LOCAL_LLM_COMMITTEE_DESCRIPTION.md`](LOCAL_LLM_COMMITTEE_DESCRIPTION.md)
   and [`LOCAL_COMMITTEE_GUIDE.md`](LOCAL_COMMITTEE_GUIDE.md) for committee
   science and reproduction.
6. [`HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md`](HUMAN_EVAL_LOGIC_AND_IMPLEMENTATION.md)
   for the independent human validation study.
7. [`EVALUATOR_IMPLEMENTATION.md`](EVALUATOR_IMPLEMENTATION.md) and
   [`REPRODUCTION_AND_ARTIFACT_PROVENANCE.md`](REPRODUCTION_AND_ARTIFACT_PROVENANCE.md)
   for executable behavior and end-to-end artifact lineage.
8. [`ENVIRONMENT_AND_SETUP.md`](ENVIRONMENT_AND_SETUP.md) for reproducible
   environments and setup verification.
