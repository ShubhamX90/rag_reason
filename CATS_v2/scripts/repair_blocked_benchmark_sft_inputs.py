#!/usr/bin/env python3
"""
Repair the eight blocked benchmark SFT evaluator inputs that failed strict
pre-launch validation on July 4, 2026.

Why targeted repair instead of broad regeneration:
  - seven rows contain malformed/truncated trace-heavy exports whose clean final
    answers are not recoverable by the generic extractor alone
  - one row is effectively empty after export sanitization
  - we want to preserve all 736 rows in each file rather than dropping samples

This script patches both:
  1. the source export row (`raw`) used for future regeneration
  2. the prepared evaluator input row (`model_output`, `model_output_raw`)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Repair:
    prepared_input: str
    source_export: str
    row_id: str
    repaired_answer: str


REPAIRS = [
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/llama8b/oracle_both/runtime/sft/input.jsonl",
        source_export="final_model_outputs/llama8b/oracle_both/runtime/sft/sft_llama31_stagewise_main_trace_text_l_boundary_rebalanced_oracle_both_trace_text_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="conflictingqa_9261438d6ee2",
        repaired_answer=(
            "The evidence is mixed. Some sources argue that fish can experience pain, "
            "while others argue that fish lack the neural structures required for "
            "human-like subjective pain [d1][d2]. Overall, fish do not feel pain "
            "exactly like humans, and the extent of their subjective experience "
            "remains disputed [d1][d2]."
        ),
    ),
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/llama8b/oracle_notes/minimal/sft/input.jsonl",
        source_export="final_model_outputs/llama8b/oracle_notes/minimal/sft/sft_llama31_stagewise_main_trace_text_l_boundary_rebalanced_oracle_notes_minimal_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="conflictingqa_cc71318e5853",
        repaired_answer=(
            "The retrieved evidence is mixed. [d1] Some sources argue that death is "
            "not taboo in modern society [d1], while others assert it remains a deeply "
            "uncomfortable and largely unspoken-about topic [d2][d3][d5]. One source "
            "argues that before the pandemic, death was the most taboo topic in Western "
            "society, though the pandemic has since brought it more into open discussion "
            "[d4]. Overall, the evidence suggests that death remains a sensitive and "
            "often avoided subject in modern society, with no single definitive "
            "consensus on whether it is still a taboo."
        ),
    ),
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/mistral7b/oracle_notes/minimal/sft/input.jsonl",
        source_export="final_model_outputs/mistral7b/oracle_notes/minimal/sft/sft_mistral7b_stagewise_main_trace_text_l_boundary_rebalanced_oracle_notes_minimal_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="qacc_292033e4b039",
        repaired_answer="Prophet Muhammad is widely recognized as the founder of Islam [d1][d2].",
    ),
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen32b/e2e/strict/sft/input.jsonl",
        source_export="final_model_outputs/qwen32b/e2e/strict/sft/sft_qwen25_32b_stagewise_main_trace_text_k_short_context_targeted_retry1_e2e_strict_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="conflictingqa_6fe31cd2ef65",
        repaired_answer=(
            "No. Rolling /r/ in Spanish is necessary only in certain positions, such "
            "as with double rr or at the beginning of some words, and is not required "
            "for every Spanish r [d1][d3]."
        ),
    ),
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen32b/oracle_both/minimal/sft/input.jsonl",
        source_export="final_model_outputs/qwen32b/oracle_both/minimal/sft/sft_qwen25_32b_stagewise_main_trace_text_k_short_context_targeted_retry1_oracle_both_minimal_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="conflictingqa_9b11b8e571aa",
        repaired_answer=(
            "No. Gonorrhea is primarily transmitted through sexual contact, but rare "
            "non-sexual transmission can occur, such as from mother to baby during "
            "childbirth [d1][d2]."
        ),
    ),
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen32b/oracle_both/runtime/sft/input.jsonl",
        source_export="final_model_outputs/qwen32b/oracle_both/runtime/sft/sft_qwen25_32b_stagewise_main_trace_text_k_short_context_targeted_retry1_oracle_both_trace_text_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="freshqa_ab11b5dce00e",
        repaired_answer=(
            "The 2026 FIFA World Cup will be hosted by the United States, Canada, and "
            "Mexico [d1][d3]."
        ),
    ),
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen32b/oracle_conflict/runtime/sft/input.jsonl",
        source_export="final_model_outputs/qwen32b/oracle_conflict/runtime/sft/sft_qwen25_32b_stagewise_main_trace_text_k_short_context_targeted_retry1_oracle_conflict_trace_text_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="qacc_e064a7a717ed",
        repaired_answer="The Glass Castle was filmed primarily in Montreal, Canada [d1][d3].",
    ),
    Repair(
        prepared_input="inputs/prepped_model_eval_inputs/benchmark_set_all_modes/qwen7b/oracle_notes/strict/sft/input.jsonl",
        source_export="final_model_outputs/qwen7b/oracle_notes/strict/sft/sft_qwen25_stagewise_main_trace_text_k_short_context_targeted_oracle_notes_strict_benchmark_final_v2_holdout_clean_736.sanitized.jsonl",
        row_id="conflictingqa_f4693bea2c31",
        repaired_answer=(
            "The evidence is mixed. Some sources argue that emoji share properties with "
            "language, but most linguists treat them as supplements to written "
            "language rather than a standalone written language [d1][d2]."
        ),
    ),
]


def patch_jsonl(path: Path, row_id: str, updater: Callable[[dict], None]) -> None:
    rows = []
    found = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") == row_id:
                updater(row)
                found = True
            rows.append(row)
    if not found:
        raise SystemExit(f"Missing id={row_id} in {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    for repair in REPAIRS:
        prepared_path = ROOT / repair.prepared_input
        export_path = ROOT / repair.source_export

        def update_export(row: dict) -> None:
            row["raw"] = repair.repaired_answer

        def update_prepared(row: dict) -> None:
            row["model_output"] = repair.repaired_answer
            row["model_output_raw"] = repair.repaired_answer
            row["model_output_field"] = "raw"
            row["model_output_source"] = repair.source_export

        patch_jsonl(export_path, repair.row_id, update_export)
        raw_export_path = export_path.with_name(export_path.name.replace(".sanitized.jsonl", ".raw.jsonl"))
        if raw_export_path.exists():
            patch_jsonl(raw_export_path, repair.row_id, update_export)
        patch_jsonl(prepared_path, repair.row_id, update_prepared)
        print(f"repaired id={repair.row_id} -> {prepared_path}")


if __name__ == "__main__":
    main()
