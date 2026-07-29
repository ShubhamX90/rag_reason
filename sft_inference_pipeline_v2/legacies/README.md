# Legacy archive

This directory is intentionally outside the current workflow surface. It preserves 2,003 historical, duplicate, generated, or superseded files relocated during the 2026-07-29 hygiene pass. No file contents were edited during relocation.

`RELOCATION_MANIFEST_2026-07-29.jsonl` is the authoritative 2,003-entry relocation ledger: it contains each archived file's pre-move SHA-256, size, classification, original path, and destination path. Every relocated entry was checksum-verified after the move. `MANIFEST_2026-07-29.jsonl` is the complete pre-move 2,888-file inventory, including the retained current files.

The archive preserves original relative paths under `legacies/`, except for:

- `export_snapshots/answer_only_sft_export_20260728/` — duplicate standalone export snapshot.
- `project_history/agent_chat_history.md` — working-session history.
- `generated_metadata/` — Python/OS cache and metadata files.

This directory can be excluded from a conference or public-code release; the root `README.md` documents the retained current reproduction workflows.
