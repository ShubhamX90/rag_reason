Benchmark local-committee sync verification

Date: 2026-07-08
Repo root: /Users/shubhammishra/Desktop/rag_reason-CATS_interactive/CATS_v2

Verified totals
- Expected benchmark inputs inspected: 102
- Final result sets found remotely and selected for sync: 94
- Final result sets missing remotely at audit time: 8
- Synced result artifacts locally:
  - detailed_results.json: 94
  - eval_report.md: 94
  - run_config.yaml: 94

Selected account split
- pabitra: 36 result sets
- kudhru: 58 result sets

Pilot-sensitive selections
- qwen7b/e2e/minimal/sft/input.jsonl
  - selected source account: pabitra
  - selected run label: pilot_qwen7b_e2e_minimal_sft_20260704_r4_behavior_pair_override_preview
- qwen7b/e2e/minimal/baseline/input.jsonl
  - selected source account: kudhru
  - selected run label: benchmark_baseline_48_20260705_r2

Local destination root
- outputs/benchmark_local_committee_3judge/benchmark_set_all_modes

Missing final result sets
- mistral7b/oracle_notes/minimal/sft/input.jsonl
- qwen32b/e2e/strict/sft/input.jsonl
- qwen32b/oracle_both/minimal/sft/input.jsonl
- qwen32b/oracle_both/runtime/sft/input.jsonl
- qwen32b/oracle_conflict/runtime/sft/input.jsonl
- qwen7b/oracle_notes/minimal/sft/input.jsonl
- qwen7b/oracle_notes/runtime/sft/input.jsonl
- qwen7b/oracle_notes/strict/sft/input.jsonl
