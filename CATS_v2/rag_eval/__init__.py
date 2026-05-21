# rag_eval/__init__.py
"""
CATS v2.0 - Conflict-Aware Trust Score Evaluation Pipeline
==========================================================

Enhanced RAG evaluation with multi-LLM judge committee.
"""

from .config import (
    EvaluationConfig,
    EnhancedConflictEvalConfig,
    EnhancedTrustScoreConfig,
    JudgeCommitteeConfig,
    DEFAULT_JUDGE_PRIORITIES,
    create_default_committee,
    get_sonnet_judge,
    get_gpt54_judge,
    get_deepseek_v32_judge,
    get_sonnet_nli_judge,
)
from .evaluator import EnhancedEvaluator
from .data import load_dataset, read_jsonl, write_jsonl
from .logging_config import logger, setup_file_logging

__version__ = "2.0.0"
__all__ = [
    "EvaluationConfig",
    "EnhancedConflictEvalConfig",
    "EnhancedTrustScoreConfig",
    "JudgeCommitteeConfig",
    "DEFAULT_JUDGE_PRIORITIES",
    "EnhancedEvaluator",
    "load_dataset",
    "read_jsonl",
    "write_jsonl",
    "logger",
    "setup_file_logging",
    "create_default_committee",
    "get_sonnet_judge",
    "get_gpt54_judge",
    "get_deepseek_v32_judge",
    "get_sonnet_nli_judge",
]
