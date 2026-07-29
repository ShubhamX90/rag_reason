This directory is the canonical output home for the 3-judge local CATS
committee run on the 736-sample benchmark prepared inputs.

Layout:

- `run_outputs/all_at_once/`
- `run_outputs/judge1_collect/`
- `run_outputs/judge2_collect/`
- `run_outputs/judge3_collect/`
- `run_outputs/final_readonly/`
- `response_cache/`

Intended use:

- A direct all-servers-available run writes into `run_outputs/all_at_once/`.
- Each staged single-judge collection run writes its evaluator artifacts into
  the matching `run_outputs/judge*_collect/` directory.
- All three staged runs share the same `response_cache/` directory so the final
  aggregation run can reuse cached judge responses.
- The final read-only aggregation run writes its outputs into
  `run_outputs/final_readonly/`.

This layout is intentionally model-agnostic until the exact three benchmark
judges are locked down in the benchmark-specific local committee configs.
