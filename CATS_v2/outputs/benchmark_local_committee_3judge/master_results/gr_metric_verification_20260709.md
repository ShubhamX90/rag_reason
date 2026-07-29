# GR Metric Verification

- Verified source result files: `108`
- Summary mismatches: `0`
- Master CSV mismatches: `0`
- Master JSON mismatches: `0`
- Structural issues: `0`
- Overall OK: `True`

This report independently recomputes all six GR-answer / GR-refusal metrics
from each run's `per_sample` records and checks them against both the
run-local `summary.gr_dataset_metrics` values and the master CSV/JSON files.

