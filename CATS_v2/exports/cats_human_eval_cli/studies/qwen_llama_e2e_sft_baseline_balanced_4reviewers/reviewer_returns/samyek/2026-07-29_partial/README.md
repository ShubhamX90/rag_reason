## Samyek Partial Reviewer Return

- Original source archive: `qwen_llama_e2e_sft_baseline_balanced_4reviewers_package.zip`
- Source location when inspected: `/Users/shubhammishra/Downloads/qwen_llama_e2e_sft_baseline_balanced_4reviewers_package.zip`
- Ingested into repo on: July 30, 2026
- Reviewer confirmed by user on July 30, 2026: `samyek`
- Reviewer identified from `judgments.sqlite3`: `samyek`
- Last observed event timestamp in returned state: `2026-07-29T08:35:16+00:00`

### Current return state

- Canonical assigned samples for `samyek`: `100`
- Active submitted judgments in this return: `50`
- Active draft judgments in this return: `9`
- Remaining assigned samples with no active state yet: `41`
- Organizer decision confirmed by user on July 30, 2026: accept the `50` submitted reviews and make downstream use of whatever was completed

### Files

- `judgments.sqlite3`: original returned reviewer judgment state
- `events.jsonl`: original returned reviewer event log
- `active_judgments.jsonl`: organizer-side raw export generated from the returned state
- `active_judgments_enriched.jsonl`: organizer-side enriched export generated from the returned state

### Note

This is an accepted partial `samyek` return. For downstream consolidation, only the `50` active submitted judgments should count as final completed human reviews; the `9` active drafts are preserved for auditability but should not be counted as finished reviews.
