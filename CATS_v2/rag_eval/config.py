# rag_eval/config.py
# -*- coding: utf-8 -*-
"""
Enhanced Configuration for CATS v2.0 Evaluation Pipeline
--------------------------------------------------------
Supports multi-LLM judge committee with Anthropic and OpenRouter APIs.

Key features:
  • Multi-judge committee voting system (priorities are modular and
    YAML/CLI-overridable via DEFAULT_JUDGE_PRIORITIES + priority_overrides).
  • Dedicated NLI judge (defaults to Claude Sonnet 4.6) separate from the
    behavior committee, so factual grounding decisions are made by a strong
    reasoning model.
  • Flexible API routing (Anthropic / OpenRouter).

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
    # Anthropic Sonnet 4.6 is now the default Anthropic judge for both the
    # behavior committee and as the dedicated NLI judge (replacing the
    # previously-used Claude 3.5 Haiku — Sonnet is stronger at strict NLI
    # and at distinguishing the conflict-type rubrics).
    "claude-sonnet-4-6":         3,
    "deepseek/deepseek-r1":      3,
    "qwen/qwen-2.5-7b-instruct": 1,
    "mistralai/mistral-nemo":    1,
    # Back-compat entry: kept so any pre-existing config that still names
    # Haiku gets a sensible weight (without forcing a YAML change).
    "claude-3-5-haiku-20241022": 2,
}


def _resolve_priority(model_id: str, priority_overrides: Optional[Dict[str, int]]) -> int:
    overrides = priority_overrides or {}
    return overrides.get(model_id, DEFAULT_JUDGE_PRIORITIES.get(model_id, 1))


# --------------------
# Predefined Judge Configurations
# --------------------
def get_sonnet_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """Claude Sonnet 4.6 — Anthropic judge for the behavior committee.

    Replaces the previous Claude 3.5 Haiku committee judge. Sonnet is
    strictly stronger on this evaluation: the qwen-monolithic run showed
    Haiku misapplying conflict-type rubrics on samples #0244 / #0046 /
    #0471 — Sonnet picks the correct rubric far more reliably.
    """
    return JudgeModelConfig(
        model_id="claude-sonnet-4-6",
        provider=APIProvider.ANTHROPIC,
        temperature=0.0,
        max_tokens=800,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        priority=_resolve_priority("claude-sonnet-4-6", priority_overrides),
        api_key_env="ANTHROPIC_API_KEY",
    )


# Backward-compatibility alias. Old callers that import `get_haiku_judge`
# still work; they now get the Sonnet 4.6 judge under the hood.
get_haiku_judge = get_sonnet_judge


def get_deepseek_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """DeepSeek R1 via OpenRouter - Reasoning capabilities.

    max_tokens raised to 3000 because R1 emits a long <think> trace before
    the final JSON. At 500 tokens the reasoning regularly consumes the entire
    budget and the JSON never appears, causing silent parse failures.
    """
    return JudgeModelConfig(
        model_id="deepseek/deepseek-r1",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=3000,
        cost_per_1k_input=0.00055,
        cost_per_1k_output=0.00219,
        priority=_resolve_priority("deepseek/deepseek-r1", priority_overrides),
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    )


def get_qwen_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """Qwen via OpenRouter - Balanced performance."""
    return JudgeModelConfig(
        model_id="qwen/qwen-2.5-7b-instruct",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.00006,
        cost_per_1k_output=0.00006,
        priority=_resolve_priority("qwen/qwen-2.5-7b-instruct", priority_overrides),
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    )


def get_mistral_nemo_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """Mistral Nemo via OpenRouter - Free tier option."""
    return JudgeModelConfig(
        model_id="mistralai/mistral-nemo",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        priority=_resolve_priority("mistralai/mistral-nemo", priority_overrides),
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
    )


def get_sonnet_nli_judge() -> JudgeModelConfig:
    """Claude Sonnet 4.6 — dedicated NLI judge for factual grounding.

    Used as the single NLI judge in `enhanced_factual_grounding`. Sonnet 4.6
    is also used inside the behavior committee (see `get_sonnet_judge`); the
    NLI judge is a separate `JudgeClient` instance with its own request
    counter and token budget so cost is tracked independently.
    """
    return JudgeModelConfig(
        model_id="claude-sonnet-4-6",
        provider=APIProvider.ANTHROPIC,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        priority=1,  # priority is irrelevant for a single-judge NLI call
        api_key_env="ANTHROPIC_API_KEY",
    )


def create_default_committee(
    priority_overrides: Optional[Dict[str, int]] = None,
    max_concurrent_requests: int = 50,
) -> JudgeCommitteeConfig:
    """
    Create the default 4-judge behavior committee.

    Composition: Sonnet 4.6 + DeepSeek R1 + Qwen 2.5 7B + Mistral Nemo.
    Pass priority_overrides={"deepseek/deepseek-r1": 1, ...} to retune.
    """
    return JudgeCommitteeConfig(
        judges=[
            get_sonnet_judge(priority_overrides),
            get_deepseek_judge(priority_overrides),
            get_qwen_judge(priority_overrides),
            get_mistral_nemo_judge(priority_overrides),
        ],
        voting_strategy="weighted_majority",
        max_concurrent_requests=max_concurrent_requests,
    )


def create_conservative_committee(
    priority_overrides: Optional[Dict[str, int]] = None,
    max_concurrent_requests: int = 50,
) -> JudgeCommitteeConfig:
    """
    Cost-effective committee (no DeepSeek): Sonnet 4.6 + Qwen + Mistral Nemo.
    """
    return JudgeCommitteeConfig(
        judges=[
            get_sonnet_judge(priority_overrides),
            get_qwen_judge(priority_overrides),
            get_mistral_nemo_judge(priority_overrides),
        ],
        voting_strategy="weighted_majority",
        max_concurrent_requests=max_concurrent_requests,
    )


# --------------------
# Global singletons — convenience defaults for scripts that import config directly.
# The CLI runner (run_evaluation.py) builds its own EvaluationConfig via
# setup_config(); these singletons are never used by it.
# --------------------
model_cfg = ModelConfig()
trust_cfg = EnhancedTrustScoreConfig()
conflict_cfg = EnhancedConflictEvalConfig(
    use_judge_committee=True,
    committee=create_default_committee(),
)
eval_cfg = EvaluationConfig(
    model=model_cfg,
    trust=trust_cfg,
    conflict=conflict_cfg,
)
