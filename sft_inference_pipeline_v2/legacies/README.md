# Legacy archive

This directory is intentionally outside the current workflow surface. It preserves 2,003 historical, duplicate, generated, or superseded files relocated during the 2026-07-29 hygiene pass. No file contents were edited during relocation.

`RELOCATION_MANIFEST_2026-07-29.jsonl` is the authoritative 2,003-entry relocation ledger: it contains each archived file's pre-move SHA-256, size, classification, original path, and destination path. Every relocated entry was checksum-verified after the move. `MANIFEST_2026-07-29.jsonl` is the complete pre-move 2,888-file inventory, including the retained current files.

The archive preserves original relative paths under `legacies/`, except for:

- `project_history/agent_chat_history.md` — working-session history.
- `generated_metadata/` — Python/OS cache and metadata files.

The answer-only SFT export that was previously staged under `export_snapshots/` was promoted to the active repository surface on 2026-07-31. It is now [`../answer_only_sft_export/`](../answer_only_sft_export/), with the portable archive at [`../answer_only_sft_export.zip`](../answer_only_sft_export.zip); neither is legacy material.

This directory can be excluded from a conference or public-code release; the root `README.md` documents the retained current reproduction workflows.
