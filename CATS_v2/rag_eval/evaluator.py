# rag_eval/evaluator.py
# -*- coding: utf-8 -*-
"""
Enhanced Evaluator for CATS v2.0
--------------------------------
Orchestrates multi-judge evaluation with multi-metric scoring.

Key v3 changes:
  • ctype=0 / out-of-range conflict types no longer silently map to Type 1
    (or KeyError after spending API budget).
  • Dedicated NLI judge (Sonnet 4.6 by default) for factual grounding —
    constructed once per evaluator, not per-sample.
  • Per-sample results sorted by sample_id before writing JSON, so two runs
    over the same input produce diff-able output.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm.asyncio import tqdm as atqdm
import numpy as np

from .config import EvaluationConfig, get_sonnet_nli_judge
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
    gr_accuracy_from_flags,
    compute_f1_gr,
)
from .conflict_eval import (
    committee_behavior_adherence,
    enhanced_factual_grounding,
    enhanced_single_truth_recall,
)
from .judge_committee import JudgeCommittee, JudgeClient
from .logging_config import logger


VALID_CONFLICT_TYPES = (1, 2, 3, 4, 5)


def _safe_ctype(raw: Any) -> int:
    """
    Coerce conflict_category_id to an int in 1..5.
    Returns the int if it's in range; otherwise returns it as-is so the
    aggregator can record an "unknown" bucket without crashing.

    The old code was `int(rec.get("conflict_category_id") or 1)`, which
    silently mapped 0 -> 1 because `0 or 1 == 1`.
    """
    if raw is None:
        return 1  # default for missing field
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"Unparseable conflict_category_id={raw!r}; defaulting to 1")
        return 1


class EnhancedEvaluator:
    """Enhanced evaluation orchestrator with multi-judge support."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.committee: JudgeCommittee = None
        self.nli_judge: JudgeClient = None
        self.results: Dict[str, Any] = {}
        self.per_sample_results: List[Dict[str, Any]] = []

        if config.conflict.use_judge_committee and config.conflict.committee:
            self.committee = JudgeCommittee(config.conflict.committee)
            logger.info("Initialized multi-judge committee")
        else:
            logger.warning("Multi-judge committee not enabled - using fallback")

        # Dedicated NLI judge (Sonnet 4.6 by default).
        nli_cfg = config.conflict.nli_judge or get_sonnet_nli_judge()
        self.nli_judge = JudgeClient(nli_cfg)
        logger.info(f"NLI judge: {nli_cfg.model_id} ({nli_cfg.provider.value})")

    async def evaluate_async(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = self.config

        logger.info(f"Starting evaluation on {len(dataset)} samples...")

        if cfg.conflict.enable_conflict_eval and self.committee:
            conflict_res = await self._evaluate_conflicts_async(dataset)
            self.results.update(conflict_res)
        else:
            logger.warning("Conflict evaluation skipped (committee not available)")

        if cfg.report_md:
            self._write_markdown_report(cfg.report_md, self.results)

        if cfg.detailed_results_json:
            self._write_detailed_results(cfg.detailed_results_json)

        if self.committee:
            cost_summary = self.committee.get_cost_summary()
            # Add NLI judge cost separately since it isn't part of the committee.
            cost_summary["nli_judge_cost"] = {
                "model_id": self.nli_judge.config.model_id,
                "total_cost": self.nli_judge.total_cost,
                "requests": self.nli_judge.request_count,
            }
            cost_summary["total_cost_usd"] = cost_summary["total_cost_usd"] + self.nli_judge.total_cost
            self.results["cost_summary"] = cost_summary
            logger.info(f"Total cost: ${cost_summary['total_cost_usd']:.4f}")

        return self.results

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synchronous wrapper. Uses asyncio.run; will raise if called inside an existing loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.evaluate_async(dataset))
        raise RuntimeError(
            "EnhancedEvaluator.evaluate() cannot be called from inside a running event loop. "
            "Use `await evaluator.evaluate_async(dataset)` instead."
        )

    async def _evaluate_conflicts_async(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        tasks = [self._evaluate_single_sample(rec, idx) for idx, rec in enumerate(dataset)]

        sample_results = []
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating samples",
        ):
            result = await coro
            sample_results.append(result)

        # Sort so two runs over the same input produce diff-able JSON.
        sample_results.sort(key=lambda r: r["sample_id"])
        self.per_sample_results = sample_results

        overall, per_type, gr_metrics = self._aggregate_results(sample_results)

        return {
            "conflict_overall": overall,
            "conflict_per_type": per_type,
            "gr_dataset_metrics": gr_metrics,
        }

    async def _evaluate_single_sample(self, rec: Dict[str, Any], idx: int) -> Dict[str, Any]:
        cfg = self.config.conflict

        sample_id = rec.get("id", f"sample_{idx:06d}")
        query = rec.get("query", "")
        ctype = _safe_ctype(rec.get("conflict_category_id"))
        notes = rec.get("per_doc_notes") or []
        doc_index = doc_index_from_record(rec)

        support_ids = support_doc_ids_from_notes(notes, accept_partial=True)
        support_docs = [doc_index[i] for i in support_ids if i in doc_index]
        gold_answerable = gold_answerable_from_notes(notes, accept_partial=True)

        answer = get_model_output(rec)
        pred_answered = answered_flags([answer])[0]
        claims = extract_claims_by_sentence(answer, cfg.max_claims_per_answer)

        # --- Metric 1: per-sample GR accuracy (binary).
        gr_acc = gr_accuracy_from_flags(pred_answered, gold_answerable)

        # --- Metric 2: behavior adherence (committee).
        beh = await committee_behavior_adherence(self.committee, query, answer, ctype)
        beh_score = 1.0 if beh["adherent"] else 0.0

        # --- Metric 3: factual grounding (dedicated NLI judge, default Sonnet 4.6).
        fg_result = await enhanced_factual_grounding(
            self.nli_judge,
            claims,
            support_docs,
            require_cross_doc=cfg.require_cross_doc_verification,
        )
        fg_score = fg_result["grounding_ratio"]

        # --- Metric 4: single-truth recall (committee).
        gold_ans = get_gold_answer(rec)
        if gold_ans and ctype in cfg.single_truth_types:
            st_result = await enhanced_single_truth_recall(
                self.committee, gold_ans, answer,
                allow_paraphrases=cfg.allow_paraphrases,
            )
            st_score = st_result["recall"]
            st_applicable = True
        else:
            st_result = {"recall": 0.0, "skipped": "no_gold_or_type_3"}
            st_score = 0.0
            st_applicable = False

        return {
            "sample_id": sample_id,
            "conflict_type": ctype,
            "pred_answered": pred_answered,
            "gold_answerable": gold_answerable,
            "gr_accuracy": gr_acc,
            "behavior_score": beh_score,
            "behavior_details": beh,
            "factual_grounding_score": fg_score,
            "factual_grounding_details": fg_result,
            "single_truth_recall_score": st_score,
            "single_truth_recall_details": st_result,
            "single_truth_applicable": st_applicable,
        }

    def _aggregate_results(self, sample_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Aggregate per-sample results.

        Returns (overall, per_type, gr_dataset_metrics).

        per_type keys are stringified so a JSON round-trip is idempotent.
        Conflict-type buckets are created on demand via setdefault, so an
        unexpected ctype no longer crashes the aggregator.

        single_truth_recall is averaged ONLY over samples where it was
        applicable (i.e., gold_answer present AND ctype in single_truth_types),
        so Type 3's structural zeros no longer drag the CATS score.
        """
        empty_bucket = {
            "n": 0,
            "gr_accuracy": [],
            "behavior": [],
            "factual_grounding": [],
            "single_truth_recall": [],
            "single_truth_recall_n": 0,
        }

        overall = {k: ([] if isinstance(v, list) else 0) for k, v in empty_bucket.items()}
        per_type: Dict[str, Dict[str, Any]] = {}

        pred_list: List[bool] = []
        gold_list: List[bool] = []

        for res in sample_results:
            ctype_key = str(res["conflict_type"])
            bucket = per_type.setdefault(
                ctype_key,
                {k: ([] if isinstance(v, list) else 0) for k, v in empty_bucket.items()},
            )

            overall["n"] += 1
            bucket["n"] += 1

            overall["gr_accuracy"].append(res["gr_accuracy"])
            bucket["gr_accuracy"].append(res["gr_accuracy"])

            overall["behavior"].append(res["behavior_score"])
            bucket["behavior"].append(res["behavior_score"])

            overall["factual_grounding"].append(res["factual_grounding_score"])
            bucket["factual_grounding"].append(res["factual_grounding_score"])

            # Only include single_truth_recall in averages when it was applicable.
            if res.get("single_truth_applicable", True):
                overall["single_truth_recall"].append(res["single_truth_recall_score"])
                bucket["single_truth_recall"].append(res["single_truth_recall_score"])
                overall["single_truth_recall_n"] += 1
                bucket["single_truth_recall_n"] += 1

            if "pred_answered" in res and "gold_answerable" in res:
                pred_list.append(bool(res["pred_answered"]))
                gold_list.append(bool(res["gold_answerable"]))

        def finalize(b: Dict[str, Any]) -> Dict[str, Any]:
            if b["n"] == 0:
                return b
            for k in ("gr_accuracy", "behavior", "factual_grounding", "single_truth_recall"):
                vals = b[k]
                b[k] = float(np.mean(vals)) if vals else 0.0
            return b

        overall = finalize(overall)
        per_type = {k: finalize(v) for k, v in per_type.items()}

        gr_dataset = compute_f1_gr(pred_list, gold_list) if pred_list else {}

        return overall, per_type, gr_dataset

    def _write_markdown_report(self, path: str, res: Dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        def _safe_fmt(val: Any) -> float:
            if isinstance(val, list):
                return float(sum(val) / len(val)) if val else 0.0
            try:
                return float(val)
            except Exception:
                return 0.0

        lines: List[str] = []
        lines.append("# CATS v2.0 Evaluation Report\n\n")
        lines.append("=" * 80 + "\n\n")

        lines.append("## Overall Conflict-Aware Metrics\n\n")

        if "conflict_overall" in res:
            o = res["conflict_overall"]
            lines.append(f"**Total Samples**: {o['n']}\n\n")
            lines.append(f"**GR Accuracy (per-sample)**: {_safe_fmt(o['gr_accuracy']):.3f}\n\n")
            lines.append(f"**Behavior Adherence**: {_safe_fmt(o['behavior']):.3f}\n\n")
            lines.append(f"**Factual Grounding**: {_safe_fmt(o['factual_grounding']):.3f}\n\n")
            lines.append(f"**Single-Truth Recall**: {_safe_fmt(o['single_truth_recall']):.3f}"
                         f" (over {o.get('single_truth_recall_n', 0)} applicable samples)\n\n")

            cats_score = float(np.mean([
                _safe_fmt(o['gr_accuracy']),
                _safe_fmt(o['behavior']),
                _safe_fmt(o['factual_grounding']),
                _safe_fmt(o['single_truth_recall']),
            ]))
            lines.append("-" * 80 + "\n\n")
            lines.append(f"### CATS Score: {cats_score:.3f}\n\n")
            lines.append("-" * 80 + "\n\n")

            if "gr_dataset_metrics" in res and res["gr_dataset_metrics"]:
                g = res["gr_dataset_metrics"]
                lines.append("\n### Dataset-level GR F1\n\n")
                lines.append(f"- **Precision**: {g['precision']:.3f}\n")
                lines.append(f"- **Recall**: {g['recall']:.3f}\n")
                lines.append(f"- **F1**: {g['f1']:.3f}\n")
                lines.append(f"- **Accuracy**: {g['accuracy']:.3f}\n")
                lines.append(f"- TP={g['tp']}, FP={g['fp']}, FN={g['fn']}, TN={g['tn']}\n\n")

            lines.append("\n## Per Conflict Type Breakdown\n\n")
            conflict_types = {
                "1": "No Conflict",
                "2": "Complementary Info",
                "3": "Conflicting Opinions",
                "4": "Outdated Info",
                "5": "Misinformation",
            }

            for t, b in sorted(res["conflict_per_type"].items()):
                lines.append(f"### Type {t}: {conflict_types.get(str(t), 'Unknown')}\n\n")
                lines.append(f"- **Samples**: {b['n']}\n")
                if b['n'] < 5:
                    lines.append(f"  - ⚠️  n<5: numbers below are noisy\n")
                lines.append(f"- **GR Accuracy**: {_safe_fmt(b['gr_accuracy']):.3f}\n")
                lines.append(f"- **Behavior**: {_safe_fmt(b['behavior']):.3f}\n")
                lines.append(f"- **Grounding**: {_safe_fmt(b['factual_grounding']):.3f}\n")
                lines.append(f"- **Recall**: {_safe_fmt(b['single_truth_recall']):.3f}"
                             f" (n={b.get('single_truth_recall_n', 0)})\n\n")

        if "cost_summary" in res:
            lines.append("\n" + "=" * 80 + "\n\n")
            lines.append("## Cost Summary\n\n")
            cost = res["cost_summary"]
            lines.append(f"**Total Cost**: ${cost['total_cost_usd']:.4f}\n\n")
            lines.append(f"**Decisions Made**: {cost['decisions_made']}\n\n")
            lines.append(f"**Average Cost per Decision**: ${cost['avg_cost_per_decision']:.6f}\n\n")

            if "per_judge_costs" in cost:
                lines.append("\n### Per-Model Cost Breakdown\n\n")
                for model_id, model_cost in cost["per_judge_costs"].items():
                    lines.append(f"#### {model_id}\n\n")
                    lines.append(f"- **Total Cost**: ${model_cost['total_cost']:.4f}\n")
                    lines.append(f"- **Total Requests**: {model_cost['requests']}\n")
                    lines.append(f"- **Average Cost per Request**: ${model_cost['avg_cost']:.6f}\n\n")

            if "nli_judge_cost" in cost:
                n = cost["nli_judge_cost"]
                lines.append(f"#### NLI Judge: {n['model_id']}\n\n")
                lines.append(f"- **Total Cost**: ${n['total_cost']:.4f}\n")
                lines.append(f"- **Total Requests**: {n['requests']}\n\n")

        lines.append("\n" + "=" * 80 + "\n")
        lines.append("\n*Report generated by CATS v2.0*\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(lines))

        logger.info(f"Report written to {path}")

    def _write_detailed_results(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "summary": self.results,
            "per_sample": self.per_sample_results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Detailed results written to {path}")

    async def close(self):
        if self.committee:
            await self.committee.close()
        if self.nli_judge:
            await self.nli_judge.close()
