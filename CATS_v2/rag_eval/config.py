# rag_eval/config.py
# -*- coding: utf-8 -*-
"""
Enhanced Configuration for CATS v2.0 Evaluation Pipeline
--------------------------------------------------------
All judges route exclusively through OpenRouter (OPENROUTER_API_KEY).
No Anthropic direct API key is required.

Committee composition (3 judges):
  • Claude Sonnet 4.6   (anthropic/claude-sonnet-4-6) — priority 3, anchor judge
  • GPT-5.4             (openai/gpt-5.4)              — priority 2, low-cost
  • DeepSeek V3.2       (deepseek/deepseek-v3-2)      — priority 2, low-cost reasoning

NLI judge: Claude Sonnet 4.6 via OpenRouter (separate JudgeClient instance).

Authors: Enhanced by Claude AI
Institution: Birla Institute of Technology and Science, Pilani
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum


# --------------------
# API Providers
# --------------------
class APIProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OPENAI = "openai"  # kept for backward compatibility


# --------------------
# Model Config (schema only; the evaluator does not read these fields)
# --------------------
@dataclass
class ModelConfig:
    """Configuration for the RAG model under evaluation.
    Kept for config-schema completeness; the evaluator does not read these fields.
    """
    name: str = "unknown"
    provider: str = "unknown"
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 42


# --------------------
# Judge Model Configuration
# --------------------
@dataclass
class JudgeModelConfig:
    """Configuration for a single judge model."""
    model_id: str
    provider: APIProvider
    temperature: float = 0.0
    max_tokens: int = 500
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    priority: int = 1

    # Rate limits — stored for documentation; no rate limiter is implemented.
    # Per-judge RPM gating would require a token-bucket; flagged in ISSUES.md §3.4.
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000

    api_key_env: Optional[str] = None
    base_url: Optional[str] = None


# --------------------
# Judge Committee Config
# --------------------
@dataclass
class JudgeCommitteeConfig:
    """
    Configuration for the multi-LLM judge committee.
    Implements majority voting with weighted consensus.

    Several fields (confidence_threshold, use_async, retry_attempts,
    timeout_seconds, cost_optimization, max_cost_per_sample,
    prefer_cheaper_models) are stored but not yet enforced by the
    voting logic — see ISSUES.md §3.2 / §3.4.
    """
    judges: List[JudgeModelConfig] = field(default_factory=list)

    # Voting strategy
    voting_strategy: str = "weighted_majority"  # options: majority, unanimous, weighted_majority

    # Confidence threshold — stored; not compared in any voting function
    confidence_threshold: float = 0.6

    # Async execution flag — stored; async is always used regardless
    use_async: bool = True

    # Retry / timeout — stored; not implemented in JudgeClient
    retry_attempts: int = 3
    timeout_seconds: float = 30.0

    # Cost controls — stored; no cost-gating logic exists
    cost_optimization: bool = True
    max_cost_per_sample: float = 0.05
    prefer_cheaper_models: bool = True

    # Concurrency control — stored but NOT enforced by JudgeCommittee
    # (asyncio.gather fans out all judge calls unconstrained — see ISSUES.md §3.3)
    max_concurrent_requests: int = 50


# --------------------
# Enhanced Trust Score Config (schema only; the evaluator does not read these fields)
# --------------------
@dataclass
class EnhancedTrustScoreConfig:
    """TRUST-SCORE evaluation settings.
    Kept for config-schema completeness; the evaluator does not read these fields.
    These flags advertise features that are not yet implemented — see ISSUES.md §3.2.
    """
    enable_trust_score: bool = True
    check_citation_accuracy: bool = True
    check_temporal_consistency: bool = True
    check_viewpoint_balance: bool = False
    weight_by_source_quality: bool = False
    compute_conflict_resolution_score: bool = False
    min_citation_count: int = 1
    citation_format: str = "[dX]"
    max_context_window: int = 4096
    use_nli_for_grounding: bool = True
    nli_threshold: float = 0.7
    aggregate_trust_score: bool = True


# --------------------
# Enhanced Conflict-Aware Config
# --------------------
@dataclass
class EnhancedConflictEvalConfig:
    """
    Enhanced conflict-aware evaluation with multi-judge committee.

    Several flags below (check_viewpoint_balance, check_temporal_precedence, etc.)
    advertise features that are not implemented — see ISSUES.md §3.2.
    """
    enable_conflict_eval: bool = True

    # Behavior judge committee
    use_judge_committee: bool = True
    committee: Optional[JudgeCommitteeConfig] = None

    # Dedicated NLI judge (used by enhanced_factual_grounding).
    # Defaults to Claude Sonnet 4.6 — see get_sonnet_nli_judge().
    # Set to None to fall back to committee.judges[0].
    nli_judge: Optional[JudgeModelConfig] = None

    # Conflict types and evaluation
    single_truth_types: Tuple[int, ...] = (1, 2, 4, 5)

    # Enhanced factual grounding
    require_cross_doc_verification: bool = False

    # Enhanced single-truth recall
    allow_paraphrases: bool = True

    # Citation requirements
    max_claims_per_answer: int = 5
    require_inline_citations: bool = False  # not enforced; stored only

    # Scoring enhancements
    aggregate_by_conflict_type: bool = True

    # Unimplemented feature flags — stored; no code reads these (ISSUES.md §3.2)
    check_viewpoint_balance: bool = False
    check_temporal_precedence: bool = False
    check_misinformation_rejection: bool = True
    penalize_unsupported_claims: bool = False
    weight_support_by_doc_quality: bool = False
    use_semantic_matching: bool = True
    check_partial_answers: bool = True
    compute_conflict_resolution_score: bool = False


# --------------------
# Pipeline Config
# --------------------
@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline execution."""
    use_async_evaluation: bool = True
    batch_size: int = 100
    max_workers: int = 50  # stored; evaluator uses asyncio, not thread workers
    skip_on_error: bool = False
    show_progress: bool = True
    verbose: bool = True
    enable_caching: bool = True  # stored; no cache layer is implemented
    cache_dir: str = ".cache"    # stored; no cache layer is implemented
    log_errors: bool = True      # stored; errors are always logged regardless


# --------------------
# Master Evaluation Config
# --------------------
@dataclass
class EvaluationConfig:
    """
    Master evaluation configuration for CATS v2.0.
    """
    # Paths
    input_jsonl: str = "data/input.jsonl"
    outputs_dir: str = "outputs/"
    report_md: str = "outputs/eval_report.md"
    detailed_results_json: str = "outputs/detailed_results.json"

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    trust: EnhancedTrustScoreConfig = field(default_factory=EnhancedTrustScoreConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    conflict: EnhancedConflictEvalConfig = field(default_factory=EnhancedConflictEvalConfig)

    # Output options
    per_type_breakdown: bool = True
    save_per_sample_scores: bool = True
    generate_visualizations: bool = False  # not implemented


# --------------------
# Modular judge priorities — tune the committee without editing factory code
# --------------------
DEFAULT_JUDGE_PRIORITIES: Dict[str, int] = {
    "anthropic/claude-sonnet-4-6": 3,  # anchor judge — strongest rubric adherence
    "openai/gpt-5.4":              2,  # low-cost complement
    "deepseek/deepseek-v3-2":      2,  # low-cost reasoning complement
}


def _resolve_priority(model_id: str, priority_overrides: Optional[Dict[str, int]]) -> int:
    overrides = priority_overrides or {}
    return overrides.get(model_id, DEFAULT_JUDGE_PRIORITIES.get(model_id, 1))


_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# --------------------
# Predefined Judge Configurations — all via OpenRouter
# --------------------
def get_sonnet_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """Claude Sonnet 4.6 via OpenRouter — anchor judge for the behavior committee.

    Routed through OpenRouter so only OPENROUTER_API_KEY is needed.
    max_tokens=800 to give enough room for the JSON + rationale.
    """
    return JudgeModelConfig(
        model_id="anthropic/claude-sonnet-4-6",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=800,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        priority=_resolve_priority("anthropic/claude-sonnet-4-6", priority_overrides),
        api_key_env="OPENROUTER_API_KEY",
        base_url=_OPENROUTER_BASE,
    )


# Backward-compatibility alias.
get_haiku_judge = get_sonnet_judge


def get_gpt54_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """GPT-5.4 via OpenRouter — low-cost complement judge."""
    return JudgeModelConfig(
        model_id="openai/gpt-5.4",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=800,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        priority=_resolve_priority("openai/gpt-5.4", priority_overrides),
        api_key_env="OPENROUTER_API_KEY",
        base_url=_OPENROUTER_BASE,
    )


def get_deepseek_v32_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """DeepSeek V3.2 reasoning via OpenRouter — low-cost reasoning complement.

    max_tokens=3000: DeepSeek reasoning models emit a <think> trace before the
    JSON. At 500 tokens the trace consumes the entire budget and the JSON
    never appears, causing silent parse failures. The _call_openrouter method
    strips <think>...</think> before parsing.
    """
    return JudgeModelConfig(
        model_id="deepseek/deepseek-v3-2",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=3000,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        priority=_resolve_priority("deepseek/deepseek-v3-2", priority_overrides),
        api_key_env="OPENROUTER_API_KEY",
        base_url=_OPENROUTER_BASE,
    )


def get_sonnet_nli_judge() -> JudgeModelConfig:
    """Claude Sonnet 4.6 via OpenRouter — dedicated NLI judge for factual grounding.

    Separate JudgeClient instance from the committee so NLI cost is tracked
    independently. Uses OPENROUTER_API_KEY only.
    """
    return JudgeModelConfig(
        model_id="anthropic/claude-sonnet-4-6",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        priority=1,  # irrelevant for single-judge NLI call
        api_key_env="OPENROUTER_API_KEY",
        base_url=_OPENROUTER_BASE,
    )


def create_default_committee(
    priority_overrides: Optional[Dict[str, int]] = None,
    max_concurrent_requests: int = 50,
) -> JudgeCommitteeConfig:
    """
    3-judge behavior committee — all via OpenRouter, single API key.

    Composition:
      • Claude Sonnet 4.6  (priority 3) — anchor judge
      • GPT-5.4            (priority 2) — low-cost complement
      • DeepSeek V3.2      (priority 2) — low-cost reasoning complement
    """
    return JudgeCommitteeConfig(
        judges=[
            get_sonnet_judge(priority_overrides),
            get_gpt54_judge(priority_overrides),
            get_deepseek_v32_judge(priority_overrides),
        ],
        voting_strategy="weighted_majority",
        max_concurrent_requests=max_concurrent_requests,
    )


# --------------------
# Global singletons — convenience defaults for scripts that import config directly.
# The CLI runner (run_evaluation.py) builds its own EvaluationConfig via
# setup_config(); these singletons are never used by it.
# All judges use OPENROUTER_API_KEY; no ANTHROPIC_API_KEY required.
# --------------------
model_cfg = ModelConfig()
trust_cfg = EnhancedTrustScoreConfig()
conflict_cfg = EnhancedConflictEvalConfig(
    use_judge_committee=True,
    committee=create_default_committee(),
    nli_judge=get_sonnet_nli_judge(),
)
eval_cfg = EvaluationConfig(
    model=model_cfg,
    trust=trust_cfg,
    conflict=conflict_cfg,
)
