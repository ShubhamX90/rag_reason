# __init__.py
# -*- coding: utf-8 -*-
"""
RAG Mixed Evaluation Toolkit
----------------------------
Unified evaluation surface for Retrieval-Augmented Generation (RAG) that
combines:
  • TRUST-SCORE style evaluation (grounded refusals, answer correctness,
    grounded citations), and
  • Conflict-aware behavior evaluation aligned to the "Dragged into Conflicts"
    taxonomy (behavior adherence, factual grounding, single-truth recall).

This package expects your dataset to include (at minimum):
  - query: str
  - retrieved_docs: List[{doc_id,title,url,snippet|text,date,...}]
  - per_doc_notes: List[{doc_id, verdict in {"supports","partially supports","irrelevant"}, key_fact, quote}]
  - conflict_category_id: int in {1..5}
  - (optional) final_grounded_answer.answer or model_output: str
  - (optional) gold_answer: str (for single-truth recall on types 1/4/5)

Key Components Exposed
----------------------
Config:
  - EvaluationConfig, ConflictEvalConfig

Evaluator:
  - Evaluator  (runs TRUST-SCORE battery, then Conflict-aware battery)
  - write Markdown report if configured

Helpers:
  - behavior_adherence, factual_grounding_ratio, single_truth_answer_recall
  - answered_flags, extract_claims_by_sentence, extract_bracket_citations
  - extract_bracket_doc_ids (post-hoc citation parsing)
  - doc_index_from_record, support_doc_ids_from_notes, gold_answerable_from_notes, get_model_output

LLM Interface:
  - LLM (with judge_behavior and nli_entailment thin endpoints)

References
----------
• Conflict taxonomy & behavior evaluation:
  Cattan et al., “Dragged into Conflicts: Evaluating RAG Systems under Knowledge Conflicts” (2025).

• TRUST-SCORE framework:
  Song et al., “TRUST-SCORE / TRUST-ALIGN: Evaluating and Aligning RAG along Grounded Refusals,
  Answer Correctness, and Grounded Citations” (ICLR 2025).

"""

from __future__ import annotations

import warnings as _warnings

__version__ = "1.0.0-rc1"

# Project / build metadata for new users and institution
__metadata__ = {
    "project": "RAG Mixed Evaluation Toolkit",
    "description": "Combined TRUST-SCORE and Conflict-aware evaluation for RAG datasets.",
    "authors": [
        "Gorang Mehrishi",
        "Samyek Jain",
    ],
    "institution": "Birla Institute of Technology and Science, Pilani",
    "maintainers": [
        "Gorang Mehrishi",
        "Samyek Jain",
    ],
    "license": "Research/Academic (specify in your repo if different)",
    "url": "",
}


def get_version() -> str:
    """Return the package version string."""
    return __version__


def get_metadata() -> dict:
    """Return a shallow copy of the package metadata dictionary."""
    return dict(__metadata__)


# --- Safe, granular imports of submodules (so you can import package even if a dependency is missing) ---

__all__ = []

# Config
try:
    from .config import EvaluationConfig, ConflictEvalConfig  # type: ignore
    __all__ += ["EvaluationConfig", "ConflictEvalConfig"]
except Exception as _e:
    _warnings.warn(f"[init] config not fully available: {type(_e).__name__}: {_e}")

# Evaluator
try:
    from .evaluator import Evaluator  # type: ignore
    __all__ += ["Evaluator"]
except Exception as _e:
    _warnings.warn(f"[init] evaluator not available: {type(_e).__name__}: {_e}")

# Conflict-aware evaluation primitives
try:
    from .conflict_eval import (  # type: ignore
        behavior_adherence,
        factual_grounding_ratio,
        single_truth_answer_recall,
    )
    __all__ += [
        "behavior_adherence",
        "factual_grounding_ratio",
        "single_truth_answer_recall",
    ]
except Exception as _e:
    _warnings.warn(f"[init] conflict_eval not available: {type(_e).__name__}: {_e}")

# Judge prompts (exported for advanced users that want to customize)
try:
    from .judge_prompts import behavior_judge_prompt, nli_prompt  # type: ignore
    __all__ += ["behavior_judge_prompt", "nli_prompt"]
except Exception as _e:
    _warnings.warn(f"[init] judge_prompts not available: {type(_e).__name__}: {_e}")

# LLM interface (expects your internal provider wiring)
try:
    from .llm import LLM  # type: ignore
    __all__ += ["LLM"]
except Exception as _e:
    _warnings.warn(f"[init] llm not available: {type(_e).__name__}: {_e}")

# Metrics glue (non-breaking helpers that complement your TRUST-SCORE implementation)
try:
    from .metrics import (  # type: ignore
        answered_flags,
        extract_claims_by_sentence,
        extract_bracket_citations,
    )
    __all__ += [
        "answered_flags",
        "extract_claims_by_sentence",
        "extract_bracket_citations",
    ]
except Exception as _e:
    _warnings.warn(f"[init] metrics helpers not available: {type(_e).__name__}: {_e}")

# If your metrics module exposes a top-level compute_trust_score or similar, surface it:
try:
    from .metrics import compute_trust_score  # type: ignore
    __all__ += ["compute_trust_score"]
except Exception:
    # Optional; not all repos expose this symbol
    pass

# Post-hoc citation utilities
try:
    from .post_hoc_cite import extract_bracket_doc_ids  # type: ignore
    __all__ += ["extract_bracket_doc_ids"]
except Exception as _e:
    _warnings.warn(f"[init] post_hoc_cite helper not available: {type(_e).__name__}: {_e}")

# Dataset adapters
try:
    from .data import (  # type: ignore
        doc_index_from_record,
        support_doc_ids_from_notes,
        gold_answerable_from_notes,
        get_model_output,
    )
    __all__ += [
        "doc_index_from_record",
        "support_doc_ids_from_notes",
        "gold_answerable_from_notes",
        "get_model_output",
    ]
except Exception as _e:
    _warnings.warn(f"[init] data helpers not available: {type(_e).__name__}: {_e}")

# Expose retrieval/search modules (kept optional; evaluation typically doesn’t require them)
try:
    from .retrieval import *  # type: ignore  # noqa: F401,F403
    # not adding to __all__ to avoid wildcard pollution; users import explicitly if needed
except Exception:
    pass

try:
    from .searcher import *  # type: ignore  # noqa: F401,F403
except Exception:
    pass

# Response generator (optional for evaluation; kept available)
try:
    from .response_generator import *  # type: ignore  # noqa: F401,F403
except Exception:
    pass

# Utilities passthrough (optional)
try:
    from .utils import *  # type: ignore  # noqa: F401,F403
except Exception:
    pass


# --- Convenience banner (shows once when package top-level is imported) ---

_BANNER = (
    f"[RAG Mixed Eval] v{__version__} — BITS Pilani\n"
    f"Authors: {', '.join(__metadata__['authors'])}\n"
    f"Institution: {__metadata__['institution']}\n"
)
try:
    # Avoid noisy logs in some environments; toggle as you like
    import os as _os
    if _os.environ.get("RAG_MIXED_EVAL_SILENT_INIT", "0") != "1":
        print(_BANNER)
except Exception:
    # Never fail import because of banner
    pass
