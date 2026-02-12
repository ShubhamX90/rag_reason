# rag_eval/__init__.py
"""
CATS v2.0 - Conflict-Aware Trust Score Evaluation Pipeline
==========================================================

Enhanced RAG evaluation with multi-LLM judge committee.
"""

from .config import (
    EvaluationConfig,
    EnhancedTrustScoreConfig,
    EnhancedConflictEvalConfig,
    JudgeCommitteeConfig,
    create_default_committee,
    create_conservative_committee,
)
from .evaluator import EnhancedEvaluator
from .data import load_dataset, read_jsonl, write_jsonl
from .logging_config import logger, setup_file_logging

__version__ = "2.0.0"
__all__ = [
    "EvaluationConfig",
    "EnhancedTrustScoreConfig",
    "EnhancedConflictEvalConfig",
    "JudgeCommitteeConfig",
    "EnhancedEvaluator",
    "load_dataset",
    "read_jsonl",
    "write_jsonl",
    "logger",
    "setup_file_logging",
    "create_default_committee",
    "create_conservative_committee",
]
