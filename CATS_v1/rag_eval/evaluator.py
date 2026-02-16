# evaluator.py
# -*- coding: utf-8 -*-
"""
Evaluator for RAG Mixed Evaluation Toolkit
------------------------------------------

Runs both:
  1. TRUST-SCORE evaluation (grounded refusals, answer correctness, grounded citations)
  2. Conflict-aware evaluation (behavior adherence, factual grounding, single-truth recall)

Inputs: dataset records in the schema defined in data.py
Outputs: dict of results + optional Markdown report

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

import os
import copy
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
import asyncio

from .config import EvaluationConfig
from .data import (
    doc_index_from_record,
    support_doc_ids_from_notes,
    gold_answerable_from_notes,
    get_model_output,
    get_gold_answer,
)
from .metrics import (
    answered_flags,
    extract_claims_by_sentence,
    extract_bracket_citations,
    f1_gr_from_flags,
    compute_trust_score,
)
from .conflict_eval import (
    behavior_adherence,
    factual_grounding_ratio,
    single_truth_answer_recall,
    abehavior_adherence,         
    afactual_grounding_ratio,
    asingle_truth_answer_recall,
)
from .logging_config import logger
#added for local mac
import nltk
nltk.data.path.append("/Users/samyekjain/eval_pipeline/.venv/nltk_data")

class Evaluator:
    """
    Main evaluation orchestrator.
    """

    def __init__(self, config: EvaluationConfig, llm):
        """
        Args:
            config (EvaluationConfig): evaluation settings
            llm (LLM): wrapper around your model API for judge + NLI
        """
        self.config = config
        self.llm = llm
        self.result: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run evaluation over a dataset.
        Returns a dict of aggregated metrics.
        """
        cfg = self.config
        llm = self.llm

        # -------------------------------
        # TRUST-SCORE evaluation
        # -------------------------------
        trust_res = self._evaluate_trustscore(dataset, llm)
        self.result.update(trust_res)

        # # -------------------------------
        # # Conflict-aware evaluation
        # # -------------------------------
        # if cfg.conflict.enable_conflict_eval:
        #     conflict_res = self._evaluate_conflicts(dataset, llm)
        #     self.result.update(conflict_res)

        if cfg.conflict.enable_conflict_eval:
            # If using OpenAI API as judge, use async for speed
            if getattr(llm.args, "azure_openai_api", False):
                conflict_res = asyncio.run(self._aevaluate_conflicts(dataset, llm))
            else:
                # Local / HF judge – keep existing synchronous path
                conflict_res = self._evaluate_conflicts(dataset, llm)
            self.result.update(conflict_res)

        # Write report if path set
        if cfg.report_md:
            self._write_markdown_report(cfg.report_md, self.result)

            return self.result


    # ------------------------------------------------------------------
    # TRUST-SCORE
    # ------------------------------------------------------------------

    def _evaluate_trustscore(self, dataset: List[Dict[str, Any]], llm) -> Dict[str, Any]:
        """
        Placeholder: in your original repo this already exists as part of metrics.py
        and evaluator pipeline. Here we just hook compute_trust_score().
        """
        results: Dict[str, float] = {}

        # Example stub: you likely compute F1_GR, F1_AC, F1_GC in your repo already
        # results["f1_gr"] = ...
        # results["f1_ac"] = ...
        # results["f1_gc"] = ...

        # Combine into TRUST-SCORE
        results = compute_trust_score(results, self.config.trust)
        return results

    # ------------------------------------------------------------------
    # Conflict Evaluation
    # ------------------------------------------------------------------

    def _evaluate_conflicts(self, dataset: List[Dict[str, Any]], llm) -> Dict[str, Any]:
        """
        Run behavior-adherence and factual-grounding evaluation
        aligned with Dragged-into-Conflicts taxonomy.
        """
        cfg = self.config

        overall = {"n": 0, "f1_gr": [], "behavior": [], "factual_grounding": [], "single_truth_recall": []}
        per_type = {k: copy.deepcopy(overall) for k in [1, 2, 3, 4, 5]}

        for rec in tqdm(dataset, desc="Conflict eval"):
            query = rec.get("query", "")
            ctype = int(rec.get("conflict_category_id") or 1)
            notes = rec.get("per_doc_notes") or []
            doc_index = doc_index_from_record(rec)

            support_ids = support_doc_ids_from_notes(notes, accept_partial=True)
            support_docs = [doc_index[i] for i in support_ids if i in doc_index]
            gold_answerable = gold_answerable_from_notes(notes, accept_partial=True)

            answer = get_model_output(rec)
            pred_answered = answered_flags([answer])[0]
            claims = extract_claims_by_sentence(answer, cfg.conflict.max_claims_per_answer)

            # --- Metrics ---
            f1gr = f1_gr_from_flags(pred_answered, gold_answerable)
            beh = behavior_adherence(llm, query, answer, ctype)
            beh_score = 1.0 if beh["adherent"] else 0.0
            fg = factual_grounding_ratio(llm.nli_entailment, claims, support_docs)

            gold_ans = get_gold_answer(rec)
            if gold_ans and ctype in cfg.conflict.single_truth_types:
                st = single_truth_answer_recall(llm, gold_ans, answer)
            else:
                st = 0.0

            # accumulate
            def acc(bucket):
                bucket["n"] += 1
                bucket["f1_gr"].append(f1gr)
                bucket["behavior"].append(beh_score)
                bucket["factual_grounding"].append(fg)
                bucket["single_truth_recall"].append(st)

            acc(overall)
            acc(per_type[ctype])

        # finalize averages
        def finalize(bucket):
            if bucket["n"] == 0:
                return bucket
            for k in ("f1_gr", "behavior", "factual_grounding", "single_truth_recall"):
                bucket[k] = float(np.mean(bucket[k])) if bucket[k] else 0.0
            return bucket

        overall = finalize(overall)
        per_type = {k: finalize(v) for k, v in per_type.items()}

        return {
            "conflict_overall": overall,
            "conflict_per_type": per_type,
        }

    async def _a_eval_single_conflict(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async evaluation for a single record (conflict-aware metrics).
        """
        cfg = self.config
        llm = self.llm

        query = rec.get("query", "")
        ctype = int(rec.get("conflict_category_id") or 1)
        notes = rec.get("per_doc_notes") or []
        doc_index = doc_index_from_record(rec)

        support_ids = support_doc_ids_from_notes(notes, accept_partial=True)
        support_docs = [doc_index[i] for i in support_ids if i in doc_index]
        gold_answerable = gold_answerable_from_notes(notes, accept_partial=True)

        answer = get_model_output(rec)
        pred_answered = answered_flags([answer])[0]
        claims = extract_claims_by_sentence(
            answer, cfg.conflict.max_claims_per_answer
        )

        f1gr = f1_gr_from_flags(pred_answered, gold_answerable)

        # Async tasks for this record
        beh_task = asyncio.create_task(
            abehavior_adherence(llm, query, answer, ctype)
        )
        fg_task = asyncio.create_task(
            afactual_grounding_ratio(llm, claims, support_docs)
        )

        gold_ans = get_gold_answer(rec)
        if gold_ans and ctype in cfg.conflict.single_truth_types:
            st_task = asyncio.create_task(
                asingle_truth_answer_recall(llm, gold_ans, answer)
            )
        else:
            st_task = None

        tasks = [beh_task, fg_task] + ([st_task] if st_task is not None else [])
        results = await asyncio.gather(*tasks)

        beh = results[0]
        fg = results[1]
        st = results[2] if st_task is not None else 0.0

        beh_score = 1.0 if beh["adherent"] else 0.0

        return {
            "conflict_type": ctype,
            "f1_gr": f1gr,
            "behavior": beh_score,
            "factual_grounding": fg,
            "single_truth_recall": st,
        }

    async def _aevaluate_conflicts(
        self, dataset: List[Dict[str, Any]], llm
    ) -> Dict[str, Any]:
        """
        Async version of conflict evaluation.
        Uses asyncio.gather to evaluate many records in parallel.
        """
        overall = {
            "n": 0,
            "f1_gr": [],
            "behavior": [],
            "factual_grounding": [],
            "single_truth_recall": [],
        }
        per_type = {k: copy.deepcopy(overall) for k in [1, 2, 3, 4, 5]}

        tasks = [self._a_eval_single_conflict(rec) for rec in dataset]

        for res in tqdm(
            await asyncio.gather(*tasks), desc="Conflict eval (async)"
        ):
            ctype = res["conflict_type"]

            def acc(bucket):
                bucket["n"] += 1
                bucket["f1_gr"].append(res["f1_gr"])
                bucket["behavior"].append(res["behavior"])
                bucket["factual_grounding"].append(res["factual_grounding"])
                bucket["single_truth_recall"].append(res["single_truth_recall"])

            acc(overall)
            acc(per_type[ctype])

        def finalize(bucket):
            if bucket["n"] == 0:
                return bucket
            for k in ("f1_gr", "behavior", "factual_grounding", "single_truth_recall"):
                bucket[k] = float(np.mean(bucket[k])) if bucket[k] else 0.0
            return bucket

        overall = finalize(overall)
        per_type = {k: finalize(v) for k, v in per_type.items()}

        return {
            "conflict_overall": overall,
            "conflict_per_type": per_type,
        }
    
    def _write_markdown_report(self, path: str, res: Dict[str, Any]) -> None:
        """Write results to Markdown file for quick inspection."""

        # --- NEW: helper to safely format values ---
        def _safe_fmt(val: Any) -> float:
            """
            Ensure metric values are formatted as floats.
            - If it's a list, average it.
            - If it's not numeric, fall back to 0.0.
            """
            if isinstance(val, list):
                if len(val) == 0:
                    return 0.0
                return float(sum(val) / len(val))
            try:
                return float(val)
            except Exception:
                return 0.0
        # --- END of helper ---

        lines: List[str] = []
        lines.append("# Evaluation Report\n")

        # TRUST-SCORE
        if "trust_score" in res:
            lines.append("## TRUST-SCORE\n")
            lines.append(f"- TRUST-SCORE: {_safe_fmt(res['trust_score']):.3f}\n")
            if "f1_gr" in res: lines.append(f"- F1_GR: {_safe_fmt(res['f1_gr']):.3f}\n")
            if "f1_ac" in res: lines.append(f"- F1_AC: {_safe_fmt(res['f1_ac']):.3f}\n")
            if "f1_gc" in res: lines.append(f"- F1_GC: {_safe_fmt(res['f1_gc']):.3f}\n")

        # Conflict-aware
        if "conflict_overall" in res:
            o = res["conflict_overall"]
            lines.append("\n## Conflict-Aware (Dragged-into-Conflicts)\n")
            lines.append(f"- n = {o['n']}\n")
            lines.append(f"- F1_GR: {_safe_fmt(o['f1_gr']):.3f}\n")
            lines.append(f"- Behavior Adherence: {_safe_fmt(o['behavior']):.3f}\n")
            lines.append(f"- Factual Grounding: {_safe_fmt(o['factual_grounding']):.3f}\n")
            lines.append(f"- Single-Truth Recall: {_safe_fmt(o['single_truth_recall']):.3f}\n")

            lines.append("\n### Per Conflict Type\n")
            for t, b in res["conflict_per_type"].items():
                lines.append(f"- Type {t} (n={b['n']}): "
                             f"F1_GR {_safe_fmt(b['f1_gr']):.3f} | "
                             f"Behavior {_safe_fmt(b['behavior']):.3f} | "
                             f"Grounding {_safe_fmt(b['factual_grounding']):.3f} | "
                             f"Recall {_safe_fmt(b['single_truth_recall']):.3f}\n")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        logger.info(f"Markdown report written to {path}")
