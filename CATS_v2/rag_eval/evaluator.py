# rag_eval/evaluator.py
# -*- coding: utf-8 -*-
"""
Enhanced Evaluator for CATS v2.0
--------------------------------
Orchestrates multi-judge evaluation with advanced metrics.

New Features:
  • Multi-judge committee voting
  • Async parallel evaluation
  • Cost tracking and optimization
  • Detailed per-sample scoring
  • Enhanced conflict resolution metrics

Authors: Enhanced by Claude AI
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm.asyncio import tqdm as atqdm
import numpy as np

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
    f1_gr_from_flags,
)
from .conflict_eval import (
    committee_behavior_adherence,
    enhanced_factual_grounding,
    enhanced_single_truth_recall,
)
from .judge_committee import JudgeCommittee
from .logging_config import logger


class EnhancedEvaluator:
    """
    Enhanced evaluation orchestrator with multi-judge support.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.committee = None
        self.results: Dict[str, Any] = {}
        self.per_sample_results: List[Dict[str, Any]] = []
        
        # Initialize judge committee if enabled
        if config.conflict.use_judge_committee and config.conflict.committee:
            self.committee = JudgeCommittee(config.conflict.committee)
            logger.info("Initialized multi-judge committee")
        else:
            logger.warning("Multi-judge committee not enabled - using fallback")
    
    async def evaluate_async(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run async evaluation over dataset with multi-judge committee.
        """
        cfg = self.config
        
        logger.info(f"Starting evaluation on {len(dataset)} samples...")
        
        # Conflict-aware evaluation with committee
        if cfg.conflict.enable_conflict_eval and self.committee:
            conflict_res = await self._evaluate_conflicts_async(dataset)
            self.results.update(conflict_res)
        else:
            logger.warning("Conflict evaluation skipped (committee not available)")
        
        # Write reports
        if cfg.report_md:
            self._write_markdown_report(cfg.report_md, self.results)
        
        if cfg.detailed_results_json:
            self._write_detailed_results(cfg.detailed_results_json)
        
        # Cost summary
        if self.committee:
            cost_summary = self.committee.get_cost_summary()
            self.results["cost_summary"] = cost_summary
            logger.info(f"Total cost: ${cost_summary['total_cost_usd']:.4f}")
        
        return self.results
    
    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synchronous wrapper for async evaluation."""
        return asyncio.run(self.evaluate_async(dataset))
    
    async def _evaluate_conflicts_async(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Async conflict evaluation with committee voting.
        """
        cfg = self.config.conflict
        
        # Prepare tasks for all samples
        tasks = [
            self._evaluate_single_sample(rec, idx)
            for idx, rec in enumerate(dataset)
        ]
        
        # Execute with progress bar
        sample_results = []
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating samples"
        ):
            result = await coro
            sample_results.append(result)
            self.per_sample_results.append(result)
        
        # Aggregate results
        overall, per_type = self._aggregate_results(sample_results)
        
        return {
            "conflict_overall": overall,
            "conflict_per_type": per_type,
        }
    
    async def _evaluate_single_sample(self, rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """
        Evaluate a single sample with all metrics.
        """
        cfg = self.config.conflict
        
        # Extract fields
        sample_id = rec.get("id", f"sample_{idx}")
        query = rec.get("query", "")
        ctype = int(rec.get("conflict_category_id") or 1)
        notes = rec.get("per_doc_notes") or []
        doc_index = doc_index_from_record(rec)
        
        support_ids = support_doc_ids_from_notes(notes, accept_partial=True)
        support_docs = [doc_index[i] for i in support_ids if i in doc_index]
        gold_answerable = gold_answerable_from_notes(notes, accept_partial=True)
        
        answer = get_model_output(rec)
        pred_answered = answered_flags([answer])[0]
        claims = extract_claims_by_sentence(answer, cfg.max_claims_per_answer)
        
        # --- Metric 1: F1_GR ---
        f1gr = f1_gr_from_flags(pred_answered, gold_answerable)
        
        # --- Metric 2: Behavior Adherence (Committee) ---
        beh = await committee_behavior_adherence(
            self.committee, query, answer, ctype
        )
        beh_score = 1.0 if beh["adherent"] else 0.0
        
        # --- Metric 3: Enhanced Factual Grounding ---
        fg_result = await enhanced_factual_grounding(
            self.committee,
            claims,
            support_docs,
            require_cross_doc=cfg.require_cross_doc_verification
        )
        fg_score = fg_result["grounding_ratio"]
        
        # --- Metric 4: Enhanced Single-Truth Recall ---
        gold_ans = get_gold_answer(rec)
        if gold_ans and ctype in cfg.single_truth_types:
            st_result = await enhanced_single_truth_recall(
                self.committee,
                gold_ans,
                answer,
                allow_paraphrases=cfg.allow_paraphrases
            )
            st_score = st_result["recall"]
        else:
            st_result = {"recall": 0.0}
            st_score = 0.0
        
        return {
            "sample_id": sample_id,
            "conflict_type": ctype,
            "f1_gr": f1gr,
            "behavior_score": beh_score,
            "behavior_details": beh,
            "factual_grounding_score": fg_score,
            "factual_grounding_details": fg_result,
            "single_truth_recall_score": st_score,
            "single_truth_recall_details": st_result,
        }
    
    def _aggregate_results(self, sample_results: List[Dict[str, Any]]) -> tuple:
        """Aggregate sample-level results into overall and per-type metrics."""
        overall = {
            "n": 0,
            "f1_gr": [],
            "behavior": [],
            "factual_grounding": [],
            "single_truth_recall": []
        }
        
        per_type = {k: {
            "n": 0,
            "f1_gr": [],
            "behavior": [],
            "factual_grounding": [],
            "single_truth_recall": []
        } for k in [1, 2, 3, 4, 5]}
        
        for res in sample_results:
            ctype = res["conflict_type"]
            
            # Accumulate overall
            overall["n"] += 1
            overall["f1_gr"].append(res["f1_gr"])
            overall["behavior"].append(res["behavior_score"])
            overall["factual_grounding"].append(res["factual_grounding_score"])
            overall["single_truth_recall"].append(res["single_truth_recall_score"])
            
            # Accumulate per-type
            per_type[ctype]["n"] += 1
            per_type[ctype]["f1_gr"].append(res["f1_gr"])
            per_type[ctype]["behavior"].append(res["behavior_score"])
            per_type[ctype]["factual_grounding"].append(res["factual_grounding_score"])
            per_type[ctype]["single_truth_recall"].append(res["single_truth_recall_score"])
        
        # Compute averages
        def finalize(bucket):
            if bucket["n"] == 0:
                return bucket
            for k in ("f1_gr", "behavior", "factual_grounding", "single_truth_recall"):
                bucket[k] = float(np.mean(bucket[k])) if bucket[k] else 0.0
            return bucket
        
        overall = finalize(overall)
        per_type = {k: finalize(v) for k, v in per_type.items()}
        
        return overall, per_type
    
    def _write_markdown_report(self, path: str, res: Dict[str, Any]) -> None:
        """Write evaluation results to Markdown report."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        def _safe_fmt(val: Any) -> float:
            """Ensure metric values are formatted as floats."""
            if isinstance(val, list):
                if len(val) == 0:
                    return 0.0
                return float(sum(val) / len(val))
            try:
                return float(val)
            except Exception:
                return 0.0
        
        lines = []
        lines.append("# CATS v2.0 Evaluation Report\n\n")
        lines.append("=" * 80 + "\n\n")
        
        lines.append("## Overall Conflict-Aware Metrics\n\n")
        
        if "conflict_overall" in res:
            o = res["conflict_overall"]
            lines.append(f"**Total Samples**: {o['n']}\n\n")
            lines.append(f"**F1_GR**: {_safe_fmt(o['f1_gr']):.3f}\n\n")
            lines.append(f"**Behavior Adherence**: {_safe_fmt(o['behavior']):.3f}\n\n")
            lines.append(f"**Factual Grounding**: {_safe_fmt(o['factual_grounding']):.3f}\n\n")
            lines.append(f"**Single-Truth Recall**: {_safe_fmt(o['single_truth_recall']):.3f}\n\n")
            
            # CATS Score (average of all metrics)
            cats_score = np.mean([
                _safe_fmt(o['f1_gr']),
                _safe_fmt(o['behavior']),
                _safe_fmt(o['factual_grounding']),
                _safe_fmt(o['single_truth_recall'])
            ])
            lines.append("-" * 80 + "\n\n")
            lines.append(f"### CATS Score: {cats_score:.3f}\n\n")
            lines.append("-" * 80 + "\n\n")
            
            lines.append("\n## Per Conflict Type Breakdown\n\n")
            conflict_types = {
                1: "No Conflict",
                2: "Complementary Info",
                3: "Conflicting Opinions",
                4: "Outdated Info",
                5: "Misinformation"
            }
            
            for t, b in res["conflict_per_type"].items():
                lines.append(f"### Type {t}: {conflict_types.get(t, 'Unknown')}\n\n")
                lines.append(f"- **Samples**: {b['n']}\n")
                lines.append(f"- **F1_GR**: {_safe_fmt(b['f1_gr']):.3f}\n")
                lines.append(f"- **Behavior**: {_safe_fmt(b['behavior']):.3f}\n")
                lines.append(f"- **Grounding**: {_safe_fmt(b['factual_grounding']):.3f}\n")
                lines.append(f"- **Recall**: {_safe_fmt(b['single_truth_recall']):.3f}\n\n")
        
        if "cost_summary" in res:
            lines.append("\n" + "=" * 80 + "\n\n")
            lines.append("## Cost Summary\n\n")
            cost = res["cost_summary"]
            lines.append(f"**Total Cost**: ${cost['total_cost_usd']:.4f}\n\n")
            lines.append(f"**Decisions Made**: {cost['decisions_made']}\n\n")
            lines.append(f"**Average Cost per Decision**: ${cost['avg_cost_per_decision']:.6f}\n\n")
            
            # Per-model costs
            if "per_judge_costs" in cost:
                lines.append("\n### Per-Model Cost Breakdown\n\n")
                for model_id, model_cost in cost["per_judge_costs"].items():
                    lines.append(f"#### {model_id}\n\n")
                    lines.append(f"- **Total Cost**: ${model_cost['total_cost']:.4f}\n")
                    lines.append(f"- **Total Requests**: {model_cost['requests']}\n")
                    lines.append(f"- **Average Cost per Request**: ${model_cost['avg_cost']:.6f}\n\n")
        
        lines.append("\n" + "=" * 80 + "\n")
        lines.append("\n*Report generated by CATS v2.0*\n")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        
        logger.info(f"Report written to {path}")
    
    def _write_detailed_results(self, path: str) -> None:
        """Write detailed per-sample results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "summary": self.results,
            "per_sample": self.per_sample_results
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Detailed results written to {path}")
    
    async def close(self):
        """Close committee and cleanup resources."""
        if self.committee:
            await self.committee.close()
