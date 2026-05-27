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
    gold_answerable_from_record,
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

        if self.committee:
            cost_summary = self.committee.get_cost_summary()
            # Add NLI judge cost — it is tracked on a separate JudgeClient, not the committee.
            cost_summary["nli_judge_cost"] = {
                "model_id": self.nli_judge.config.model_id,
                "total_cost": self.nli_judge.total_cost,
                "requests": self.nli_judge.request_count,
                "avg_cost": (self.nli_judge.total_cost / self.nli_judge.request_count
                             if self.nli_judge.request_count else 0.0),
            }
            cost_summary["total_cost_usd"] = cost_summary["total_cost_usd"] + self.nli_judge.total_cost
            # Recompute avg after adding NLI so it reflects the true grand total.
            cost_summary["avg_cost_per_decision"] = (
                cost_summary["total_cost_usd"] / max(1, cost_summary["decisions_made"])
            )
            self.results["cost_summary"] = cost_summary
            logger.info(f"Total cost: ${cost_summary['total_cost_usd']:.4f}")

        # Write output files after cost_summary is populated so both files include cost data.
        if cfg.report_md:
            self._write_markdown_report(cfg.report_md, self.results)

        if cfg.detailed_results_json:
            self._write_detailed_results(cfg.detailed_results_json)

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
        # NLI-I fix: merge the annotator-verified verbatim `quote` from
        # per_doc_notes into each support doc so enhanced_factual_grounding
        # can use it as a tighter NLI premise. Falls back to snippet when no
        # quote is present.
        notes_by_id: Dict[str, Dict[str, Any]] = {
            n.get("doc_id"): n for n in notes if n.get("doc_id")
        }
        support_docs: List[Dict[str, Any]] = []
        for did in support_ids:
            if did not in doc_index:
                continue
            merged = dict(doc_index[did])  # copy to avoid mutating the input record
            quote = (notes_by_id.get(did, {}) or {}).get("quote")
            if quote and str(quote).strip():
                merged["quote"] = str(quote).strip()
            support_docs.append(merged)
        gold_answerable = gold_answerable_from_record(rec, accept_partial=True)

        answer = get_model_output(rec)
        pred_answered = answered_flags([answer])[0]

        # --- Metric 1: per-sample GR accuracy (binary, always computed).
        gr_acc = gr_accuracy_from_flags(pred_answered, gold_answerable)

        # Correct refusal: the model refused AND the question was genuinely unanswerable.
        # Behavior, factual grounding, and STR are not meaningful here — the model did
        # the right thing (GR=1.0), so we do not call the committee and exclude these
        # samples from the three sub-metric averages and from the CATS denominator.
        correct_refusal = (not gold_answerable) and (not pred_answered)

        if correct_refusal:
            beh_score = 0.0  # placeholder — not counted in average
            beh = {
                "adherent": None,
                "rationale": "N/A — correct refusal; sample excluded from behavior average",
                "skipped": "correct_refusal",
                "committee_details": None,
            }
            fg_score = 0.0  # placeholder — not counted in average
            fg_result = {
                "grounding_ratio": None,
                "supported_claims": 0,
                "total_claims": 0,
                "claim_details": [],
                "skipped": "correct_refusal",
            }
            st_score = 0.0
            st_result = {"recall": 0.0, "skipped": "correct_refusal"}
            st_applicable = False
            beh_applicable = False
            fg_applicable = False
        else:
            claims = extract_claims_by_sentence(answer, cfg.max_claims_per_answer)

            # --- Metric 2: behavior adherence (committee).
            # Pass retrieved_docs so Type 4 / Type 5 judges can see publication
            # dates / sources when judging "prioritise the up-to-date / reliable
            # source" behavior (N6 fix). For other types the docs are ignored.
            retrieved_docs = rec.get("retrieved_docs") or []
            beh = await committee_behavior_adherence(
                self.committee, query, answer, ctype,
                retrieved_docs=retrieved_docs,
            )
            beh_score = 1.0 if beh["adherent"] else 0.0
            beh_applicable = True

            # --- Metric 3: factual grounding (dedicated NLI judge, default Sonnet 4.6).
            # NLI-G fix: FG measures how well the model grounded its CLAIMS in
            # the evidence. If the model refused (pred_answered=False) there
            # are no claims to ground — FG is not applicable, regardless of
            # whether the refusal was correct or wrong. (Correct refusal is
            # already handled by the outer branch.) Previously FG ran on the
            # refusal text, produced grounding_ratio=0.0, and dragged down
            # the average for every wrong-refusal sample.
            if not pred_answered:
                fg_score = 0.0
                fg_result = {
                    "grounding_ratio": None,
                    "supported_claims": 0,
                    "total_claims": 0,
                    "claim_details": [],
                    "skipped": "model_refused",
                }
                fg_applicable = False
            else:
                fg_result = await enhanced_factual_grounding(
                    self.nli_judge,
                    claims,
                    support_docs,
                    require_cross_doc=cfg.require_cross_doc_verification,
                    min_entail_confidence=cfg.min_entail_confidence,
                    majority_support_rule=cfg.majority_support_rule,
                    conflict_type=ctype,
                    neutral_as_support=cfg.neutral_as_support,
                )
                fg_score = fg_result["grounding_ratio"]
                fg_applicable = True

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
            "correct_refusal": correct_refusal,
            "gr_accuracy": gr_acc,
            "behavior_score": beh_score,
            "behavior_applicable": beh_applicable,
            "behavior_details": beh,
            "factual_grounding_score": fg_score,
            "factual_grounding_applicable": fg_applicable,
            "factual_grounding_details": fg_result,
            "single_truth_recall_score": st_score,
            "single_truth_recall_details": st_result,
            "single_truth_applicable": st_applicable,
        }

    def _aggregate_results(self, sample_results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Aggregate per-sample results.

        Returns (overall, per_type, gr_dataset_metrics).

        Applicability rules:
        - gr_accuracy       : all samples
        - behavior          : excluded when correct_refusal=True
        - factual_grounding : excluded when correct_refusal=True
        - single_truth_recall: excluded when single_truth_applicable=False
                               (Type 3 or no gold answer)

        correct_refusal (gold_answerable=False AND pred_answered=False) means the
        model did the right thing — its GR=1.0 counts, but the other three metrics
        are structurally uninformative and must not drag the averages down.

        CATS score is the mean of whichever sub-metrics have at least one
        applicable sample, so correct refusals do not inflate a denominator they
        don't contribute to.
        """
        empty_bucket = {
            "n": 0,
            "correct_refusals": 0,
            "gr_accuracy": [],
            "behavior": [],
            "behavior_n": 0,
            "factual_grounding": [],
            "factual_grounding_n": 0,
            "single_truth_recall": [],
            "single_truth_recall_n": 0,
            # Per-bucket pred/gold lists for per-type GR F1 (N13 fix).
            "pred_answered_list": [],
            "gold_answerable_list": [],
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

            if res.get("correct_refusal", False):
                overall["correct_refusals"] += 1
                bucket["correct_refusals"] += 1

            # GR accuracy: always included.
            overall["gr_accuracy"].append(res["gr_accuracy"])
            bucket["gr_accuracy"].append(res["gr_accuracy"])

            # Behavior: excluded for correct refusals.
            if res.get("behavior_applicable", True):
                overall["behavior"].append(res["behavior_score"])
                bucket["behavior"].append(res["behavior_score"])
                overall["behavior_n"] += 1
                bucket["behavior_n"] += 1

            # Factual grounding: excluded for correct refusals.
            if res.get("factual_grounding_applicable", True):
                overall["factual_grounding"].append(res["factual_grounding_score"])
                bucket["factual_grounding"].append(res["factual_grounding_score"])
                overall["factual_grounding_n"] += 1
                bucket["factual_grounding_n"] += 1

            # Single-truth recall: excluded when not applicable (Type 3 / no gold / correct refusal).
            if res.get("single_truth_applicable", True):
                overall["single_truth_recall"].append(res["single_truth_recall_score"])
                bucket["single_truth_recall"].append(res["single_truth_recall_score"])
                overall["single_truth_recall_n"] += 1
                bucket["single_truth_recall_n"] += 1

            if "pred_answered" in res and "gold_answerable" in res:
                pred_list.append(bool(res["pred_answered"]))
                gold_list.append(bool(res["gold_answerable"]))
                # Also track per-bucket for per-type F1 (N13 fix).
                bucket["pred_answered_list"].append(bool(res["pred_answered"]))
                bucket["gold_answerable_list"].append(bool(res["gold_answerable"]))

        def finalize(b: Dict[str, Any]) -> Dict[str, Any]:
            if b["n"] == 0:
                return b
            for k in ("gr_accuracy", "behavior", "factual_grounding", "single_truth_recall"):
                vals = b[k]
                b[k] = float(np.mean(vals)) if vals else 0.0
            # N13 fix: compute per-bucket GR F1 so per-type CATS is semantically
            # consistent with overall CATS (which uses dataset-level F1, not accuracy).
            pt_pred = b.pop("pred_answered_list", [])
            pt_gold = b.pop("gold_answerable_list", [])
            if pt_pred:
                pt_gr = compute_f1_gr(pt_pred, pt_gold)
                b["gr_f1"] = pt_gr["f1"]
                gr_cats_component = pt_gr["f1"]
            else:
                gr_cats_component = b["gr_accuracy"]
            # CATS: mean of whichever sub-metrics had at least one applicable sample.
            cats_parts = [gr_cats_component]
            if b["behavior_n"] > 0:
                cats_parts.append(b["behavior"])
            if b["factual_grounding_n"] > 0:
                cats_parts.append(b["factual_grounding"])
            if b["single_truth_recall_n"] > 0:
                cats_parts.append(b["single_truth_recall"])
            b["cats_score"] = float(np.mean(cats_parts))
            return b

        overall = finalize(overall)
        per_type = {k: finalize(v) for k, v in per_type.items()}

        gr_dataset = compute_f1_gr(pred_list, gold_list) if pred_list else {}

        # Recompute overall CATS using dataset-level GR F1 instead of sample-averaged accuracy.
        # F1 penalises both false positives (answering unanswerable) and false negatives (refusing
        # answerable), while accuracy can be gamed by the class distribution.
        # Per-type cats_score retains accuracy (no per-type F1 without separate tracking).
        if gr_dataset:
            gr_f1 = gr_dataset["f1"]
            overall["gr_f1"] = gr_f1
            cats_parts = [gr_f1]
            if overall["behavior_n"] > 0:
                cats_parts.append(overall["behavior"])
            if overall["factual_grounding_n"] > 0:
                cats_parts.append(overall["factual_grounding"])
            if overall["single_truth_recall_n"] > 0:
                cats_parts.append(overall["single_truth_recall"])
            overall["cats_score"] = float(np.mean(cats_parts))

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
            n_total = o["n"]
            n_correct_refusals = o.get("correct_refusals", 0)
            lines.append(f"**Total Samples**: {n_total}\n\n")
            if n_correct_refusals:
                lines.append(
                    f"**Correct Refusals**: {n_correct_refusals} "
                    f"(GR=1.0 only; excluded from behavior/grounding/recall averages)\n\n"
                )
            lines.append(f"**GR Accuracy**: {_safe_fmt(o['gr_accuracy']):.3f}"
                         f" (over {n_total} samples)\n\n")
            if "gr_f1" in o:
                lines.append(f"**GR F1** *(used in CATS)*: {_safe_fmt(o['gr_f1']):.3f}\n\n")
            lines.append(f"**Behavior Adherence**: {_safe_fmt(o['behavior']):.3f}"
                         f" (over {o.get('behavior_n', n_total)} applicable samples)\n\n")
            lines.append(f"**Factual Grounding**: {_safe_fmt(o['factual_grounding']):.3f}"
                         f" (over {o.get('factual_grounding_n', n_total)} applicable samples)\n\n")
            lines.append(f"**Single-Truth Recall**: {_safe_fmt(o['single_truth_recall']):.3f}"
                         f" (over {o.get('single_truth_recall_n', 0)} applicable samples)\n\n")

            cats_score = _safe_fmt(o.get("cats_score", 0.0))
            lines.append("-" * 80 + "\n\n")
            lines.append(f"### CATS Score: {cats_score:.3f}\n\n")
            lines.append(
                f"*(average of {len([x for x in [True, o.get('behavior_n',0)>0, o.get('factual_grounding_n',0)>0, o.get('single_truth_recall_n',0)>0] if x])} "
                f"applicable sub-metrics)*\n\n"
            )
            lines.append("-" * 80 + "\n\n")

            if "gr_dataset_metrics" in res and res["gr_dataset_metrics"]:
                g = res["gr_dataset_metrics"]
                lines.append("\n### Dataset-level GR Metrics\n\n")
                lines.append(f"- **F1** *(CATS component)*: {g['f1']:.3f}\n")
                lines.append(f"- **Precision**: {g['precision']:.3f}\n")
                lines.append(f"- **Recall**: {g['recall']:.3f}\n")
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
                lines.append(f"- **Samples**: {b['n']}"
                             + (f" ({b['correct_refusals']} correct refusals excluded from sub-metrics)" if b.get('correct_refusals') else "")
                             + "\n")
                if b['n'] < 5:
                    lines.append(f"  - ⚠️  n<5: numbers below are noisy\n")
                lines.append(f"- **GR Accuracy**: {_safe_fmt(b['gr_accuracy']):.3f}\n")
                if "gr_f1" in b:
                    lines.append(f"- **GR F1** *(used in CATS)*: {_safe_fmt(b['gr_f1']):.3f}\n")
                lines.append(f"- **Behavior**: {_safe_fmt(b['behavior']):.3f} (n={b.get('behavior_n', b['n'])})\n")
                lines.append(f"- **Grounding**: {_safe_fmt(b['factual_grounding']):.3f} (n={b.get('factual_grounding_n', b['n'])})\n")
                lines.append(f"- **Recall**: {_safe_fmt(b['single_truth_recall']):.3f}"
                             f" (n={b.get('single_truth_recall_n', 0)})\n")
                lines.append(f"- **CATS**: {_safe_fmt(b.get('cats_score', 0.0)):.3f}\n\n")

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
