# Pipeline V3 Changes — Historical Analysis Note

This file is a point-in-time analysis memo, not the live source of truth for the current codebase.

## 1. Codebase State (v2 code living in v3)

Key findings from reading every file:

- **`llm_client.py`**: Three providers — Anthropic, OpenAI, OpenRouter. Batch API for Anthropic + OpenAI.
- **`src/parsers.py`**: Stage1/2/3/monolithic parsers. Stage2 parses `{conflict_reason, answerable_under_evidence}`.
- **`src/utils.py`**: Shared helpers.
- **Prompts (8 files)**: `system_stage1/2/3`, `user_stage1/2/3`, `system_monolithic`, `user_monolithic`.
- **Scripts**: Async + batch runners for stage1/2/3 and monolithic.
- **Normalized data (critical)**:
  - `conflicts_normalized.jsonl`: `{id, query, conflict_type, gold_answer, retrieved_docs}` — `conflict_type` is always present (gold label).
  - `refusals_normalized.jsonl`: **Same exact schema** — `{id, query, conflict_type, gold_answer, retrieved_docs}` — **`conflict_type` is ALSO always present** (200 records, all having it, with values like `'No conflict'`, `'Complementary information'`, etc.)

---

## 2. CRITICAL FINDING: The Refusal Prompt Issue

> **You are RIGHT to question Claude's new refusal prompts. They are based on an incorrect data assumption.**

### Claude's claim:
Claude proposes new files `prompts/system_stage2_refusal.txt` and `prompts/user_stage2_refusal.txt` on the grounds that refusals don't have a gold `conflict_type` — so the model needs a different prompt to *determine* `conflict_type` from scratch.

### Why this is wrong:
```
refusals WITH conflict_type : 200   (ALL records)
refusals WITHOUT conflict_type: 0
conflict_type values sample: 'No conflict', 'Conflicting opinions or research outcomes',
                              'Complementary information', 'Conflict due to outdated information',
                              'Conflict due to misinformation'
```

Every refusal record has a `conflict_type`. The refusals dataset is **structurally identical** to conflicts. Claude assumed the label was missing based on the project description, but the actual data disproves this.

As a result:
- `system_stage2.txt` + `user_stage2.txt` already work for refusals — the `{CONFLICT_TYPE}` placeholder gets filled from the record's gold label.
- The `is_refusal` detection in `run_stage2_multi_async.py` (`not bool(record.get("conflict_type","").strip())`) will **never be True** for any actual record — it's dead code.
- The new refusal prompts are motivated by a false premise.

---

## 3. Other Holes in Claude's Response

### Hole 1: `parse_stage2` issue (latent)
For the hypothetical refusal path, Claude says `parse_stage2` passes through extra fields like `conflict_type` transparently. This is partially true — `_extract_json_object` captures the whole dict — but `parse_stage2` doesn't validate `conflict_type`. `merge_stage2_votes` tries `_build_votes(model_records, "conflict_type", "")` on these records; since refusals have gold `conflict_type` in their input record (not the model output), this whole branch is moot anyway.

### Hole 2: Model slug accuracy
- `anthropic/claude-sonnet-4-5` is used, but you want Sonnet **4.6** — verify exact slug at `openrouter.ai/models`
- `qwen/qwen3-235b-a22b` — Qwen 3 availability on OpenRouter is uncertain; `qwen/qwen-2.5-72b-instruct` is the proven slug
- GPT-5 may not be on OpenRouter yet; `openai/gpt-4o` is safer
- `x-ai/grok-2-1212` replacing Mistral is a judgment call

### Hole 3: `_OPENROUTER_ALIASES` additions are redundant
The alias additions in `llm_client.py` are unnecessary — any slug containing `/` already passes through unchanged via `if "/" in model: return model`. Full slugs don't need to be in the alias table.

### Hole 4: `run_pipeline.sh` sed patches are fragile
Claude mixes sed + Python patching inconsistently. Some sed patterns are based on assumed strings that could fail silently. Python-based patching is safer and more reliable.

### Hole 5: `.env.example` and README not updated
For v3 (OpenRouter-only), `.env.example` should only show `OPENROUTER_API_KEY`. The README still says "v2" and describes Anthropic as primary.

---

## 4. What Claude Got Right

| Aspect | Verdict |
|---|---|
| Overall multi-LLM voting architecture | **Correct** |
| `src/voting.py` design | **Correct** — clean `weighted_majority_vote`, `select_winner_model`, stage-level merge helpers |
| Stage 1 multi-async (committee on `verdict`) | **Correct** |
| Stage 2 multi-async — conflicts path | **Correct** — uses existing prompts with `{CONFLICT_TYPE}` |
| Stage 3 multi-async (votes on `abstain`) | **Correct** |
| Audit trail metadata (`_vote_tally`, `_winner_model`) | **Good design** |
| OpenRouter-only in all multi scripts | **Correct** |
| Batch scripts left untouched | **Correct** |
| `parse_stage1/2/3` untouched | **Correct** — parsers need no changes |
| Resume capability preserved | **Correct** |

---

## 5. Key Design Question to Resolve Before Implementation

> **Do you want to use the existing `conflict_type` gold label for refusals, or have the committee re-annotate it independently?**

**Option A — Use existing gold label** (simpler, correct for both datasets):
- No separate refusal prompt
- Stage 2 pipeline is identical for conflicts and refusals
- Committee only votes on `answerable_under_evidence`
- `run_stage2_multi_async.py` simplified — no `is_refusal` branch

**Option B — Committee re-annotates `conflict_type` for refusals** (validation/improvement):
- You DO want `system_stage2_refusal.txt` — well-written by Claude, just wrongly motivated
- Detection should be based on **dataset source** (e.g., input file name or a flag), NOT on whether `conflict_type` is absent
- This lets you measure committee-vs-gold agreement for refusals — valuable for research

The new refusal prompts Claude wrote are genuinely well-written and useful for Option B. The only problem is the *reason* for creating them was wrong.

---

## 6. Recommended Changes to Claude's Plan

1. **If going Option A**: Delete `system_stage2_refusal.txt` and `user_stage2_refusal.txt` from the plan; remove `is_refusal` branch from `run_stage2_multi_async.py`.
2. **If going Option B**: Keep the refusal prompts but change `is_refusal` detection to be based on dataset source (e.g., `--refusal-mode` flag or input file path check), not on missing `conflict_type`.
3. Verify all OpenRouter slugs before finalizing `voting.py`.
4. Update `.env.example` to show only `OPENROUTER_API_KEY`.
5. Update `README.md` for v3.
6. Use Python-based patching for `run_pipeline.sh` throughout.

---

## 7. Summary

| Category | Rating | Note |
|---|---|---|
| Core multi-LLM voting design | ✅ Good | Right approach |
| Stage 1 | ✅ Correct | |
| Stage 2 (conflicts path) | ✅ Correct | Uses existing prompts |
| Stage 2 (refusal path) | ❌ Based on wrong premise | `conflict_type` IS present in all refusal records |
| Stage 3 | ✅ Correct | |
| `voting.py` | ✅ Correct | |
| Refusal prompts (content quality) | ✅ Well-written | Wrong motivation, but useful for Option B |
| `run_pipeline.sh` | ⚠️ Fragile | Mixed sed/Python patching |
| Model slug choices | ⚠️ Needs verification | Several may be stale/wrong |
| `.env.example` / README | ❌ Not updated | Should reflect v3 |
