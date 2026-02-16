# config.py
# -*- coding: utf-8 -*-
"""
Configuration for RAG Mixed Evaluation Toolkit
----------------------------------------------
Defines dataclasses and constants used across the evaluation pipeline.

This config unifies:
  • TRUST-SCORE evaluation settings (grounded refusals, correctness, citations)
  • Conflict-aware evaluation settings (behavior adherence, factual grounding,
    single-truth recall) aligned to the Dragged-into-Conflicts taxonomy.

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# --------------------
# Core Model Config
# --------------------
@dataclass
class ModelConfig:
    """Base configuration for an LLM or NLI model used in evaluation."""
    name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 42


# --------------------
# TRUST-SCORE Config
# --------------------
@dataclass
class TrustScoreConfig:
    """
    Parameters for TRUST-SCORE style evaluation.
    See: Song et al., ICLR 2025.
    """
    eval_type: str = "f1" #added
    enable_rag_eval: bool = True
    compute_macro: bool = True
    compute_correctness: bool = True
    compute_citations: bool = True

    # If your dataset contains gold answerability flags, set to True
    use_gold_answerability: bool = True

    # Maximum number of claims to extract per answer
    max_claims_per_answer: int = 12


# --------------------
# Conflict-Aware Config
# --------------------
@dataclass
class ConflictEvalConfig:
    """
    Parameters for Conflict-aware evaluation aligned to
    Dragged-into-Conflicts taxonomy.
    See: Cattan et al., 2025.
    """

    enable_conflict_eval: bool = True
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    use_llm_for_nli: bool = True  # else route to a local NLI model

    # Conflict types that are treated as single-truth categories:
    # 1 = No Conflict, 4 = Outdated, 5 = Misinformation
    single_truth_types: Tuple[int, ...] = (1, 2, 4, 5)

    # If model outputs contain inline [dX] citations, enable strict checking
    require_inline_citations: bool = False

    # Maximum number of claims to check in one answer
    max_claims_per_answer: int = 12


# --------------------
# Evaluation Config
# --------------------
from dataclasses import dataclass, field #added

@dataclass
class EvaluationConfig:
    """
    Master evaluation configuration that unifies TRUST-SCORE and Conflict eval.
    """
    # Paths
    input_jsonl: str = "data/annotated.jsonl"
    outputs_dir: str = "outputs/"
    report_md: Optional[str] = "outputs/eval_report.md"

    # Core
    model: ModelConfig = field(default_factory=ModelConfig)

    # Subsystems
    trust: TrustScoreConfig = field(default_factory=TrustScoreConfig)
    conflict: ConflictEvalConfig = field(default_factory=ConflictEvalConfig)

    # Reporting
    per_type_breakdown: bool = True


# --------------------
# Global Defaults
# --------------------
# These objects can be imported directly elsewhere
model_cfg = ModelConfig()
trust_cfg = TrustScoreConfig()
conflict_cfg = ConflictEvalConfig()
eval_cfg = EvaluationConfig()
