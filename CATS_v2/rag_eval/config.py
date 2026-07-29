# rag_eval/config.py
# -*- coding: utf-8 -*-
"""
Enhanced Configuration for CATS v2.0 Evaluation Pipeline
--------------------------------------------------------
This module supports multiple committee modes:

  • default committee: OpenRouter-backed multi-model judges
  • cli committee: local Codex CLI only
  • mixed committee: local Codex CLI + direct DeepSeek API
  • local_openai committee: OpenAI-compatible local endpoints

Factual grounding in the active evaluator path is committee-based. A standalone
`nli_judge` remains available only for legacy / alternate grounding paths.

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
    DEEPSEEK = "deepseek"
    OPENAI = "openai"  # kept for backward compatibility
    CODEX_CLI = "codex_cli"     # npx @openai/codex exec subprocess (Codex Pro subscription)
    LOCAL_OPENAI = "local_openai"  # local/vLLM/SGLang/LMDeploy OpenAI-compatible server


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
    cli_model: Optional[str] = None  # -m/--model flag for CLI providers
    cli_timeout: int = 120            # subprocess timeout in seconds
    request_timeout: float = 60.0      # HTTP timeout in seconds
    max_context_tokens: int = 4096
    context_safety_margin: int = 64
    extra_body: Dict[str, object] = field(default_factory=dict)


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

    # Retry is stored; timeout_seconds is enforced around each committee judge call.
    retry_attempts: int = 3
    timeout_seconds: float = 30.0

    # Cost controls — stored; no cost-gating logic exists
    cost_optimization: bool = True
    max_cost_per_sample: float = 0.05
    prefer_cheaper_models: bool = True

    # Concurrency control — enforced by JudgeCommittee around outbound judge calls.
    max_concurrent_requests: int = 50

    # Optional persistent per-judge response cache. Useful for local staged
    # committees when not all GPU-backed judge servers can be hosted at once.
    response_cache_dir: Optional[str] = None
    cache_mode: str = "off"  # off, read_write, read_only, write_only


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

    # Optional standalone NLI judge for legacy / alternate grounding paths.
    # The active evaluator path uses committee-based factual grounding, so this
    # is normally left as None for committee runs.
    nli_judge: Optional[JudgeModelConfig] = None

    # Conflict types and evaluation
    single_truth_types: Tuple[int, ...] = (1, 2, 4, 5)

    # Enhanced factual grounding
    require_cross_doc_verification: bool = False
    min_entail_confidence: float = 0.5  # NLI confidence floor; verdicts below this are treated as neutral
    majority_support_rule: bool = False  # if True: claim passes when entails_count > contradicts_count (majority wins)
                                         # if False (default): any contradiction blocks the claim
    neutral_as_support: bool = False     # R1: if True, claims where ALL docs are neutral (e=0, c=0) are treated
                                         # as supported — retrieval gap, not hallucination
    ignore_contradictions_types: tuple = ()  # R3: conflict types that ignore contradictions (CT3 is always included).
                                             # Add 2 and/or 4 here: CT2 docs cover different aspects (off-topic
                                             # contradictions meaningless); CT4 outdated docs should contradict
                                             # current-state claims by design.
    partial_credit_fg: bool = False      # R4: award 0.5 * (e/(e+c)) partial credit when e>0 but claim doesn't
                                         # fully pass — finer-grained than binary 0/1 per claim

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
    "anthropic/claude-sonnet-4-6":      3,  # anchor judge — strongest rubric adherence
    "anthropic/claude-haiku-4-5":       3,  # haiku anchor — fast + cheap alternative
    "openai/gpt-5.4":                   2,  # low-cost complement
    "deepseek/deepseek-v3.2":           2,  # low-cost reasoning complement
    "codex-cli/default":                2,  # codex CLI (Pro subscription)
    "local/qwen3.5-397b-a17b":          4,  # local anchor when 4x H200 is available
    "local/qwen3.5-122b":               3,  # local fallback anchor for 2x H200
    "local/deepseek-r1-distill-32b":    2,
    "local/gemma-4-31b":                1,
    "local/mistral-small-4":            1,
}


def _resolve_priority(model_id: str, priority_overrides: Optional[Dict[str, int]]) -> int:
    overrides = priority_overrides or {}
    return overrides.get(model_id, DEFAULT_JUDGE_PRIORITIES.get(model_id, 1))


_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_DEEPSEEK_BASE = "https://api.deepseek.com"


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


def get_haiku_judge(priority_overrides: Optional[Dict[str, int]] = None) -> JudgeModelConfig:
    """Claude Haiku 4.5 via OpenRouter — fast, low-cost anchor judge.

    ~10x cheaper than Sonnet on input tokens; suitable when throughput and cost
    matter more than peak reasoning quality.
    """
    return JudgeModelConfig(
        model_id="anthropic/claude-haiku-4-5",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=800,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        priority=_resolve_priority("anthropic/claude-haiku-4-5", priority_overrides),
        api_key_env="OPENROUTER_API_KEY",
        base_url=_OPENROUTER_BASE,
    )


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
        model_id="deepseek/deepseek-v3.2",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=3000,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        priority=_resolve_priority("deepseek/deepseek-v3.2", priority_overrides),
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
      • Claude Haiku 4.5   (priority 3) — anchor judge (fast, low-cost)
      • GPT-5.4            (priority 2) — low-cost complement
      • DeepSeek V3.2      (priority 2) — low-cost reasoning complement
    """
    return JudgeCommitteeConfig(
        judges=[
            get_haiku_judge(priority_overrides),
            get_gpt54_judge(priority_overrides),
            get_deepseek_v32_judge(priority_overrides),
        ],
        voting_strategy="weighted_majority",
        max_concurrent_requests=max_concurrent_requests,
    )


def create_conservative_committee(
    priority_overrides: Optional[Dict[str, int]] = None,
    max_concurrent_requests: int = 50,
) -> JudgeCommitteeConfig:
    """
    Lower-cost 2-judge behavior committee — all via OpenRouter.

    Keeps the Haiku anchor and GPT complement, omitting DeepSeek's larger
    generation budget.
    """
    return JudgeCommitteeConfig(
        judges=[
            get_haiku_judge(priority_overrides),
            get_gpt54_judge(priority_overrides),
        ],
        voting_strategy="weighted_majority",
        max_concurrent_requests=max_concurrent_requests,
    )


# --------------------
def get_codex_cli_judge(
    model_alias: Optional[str] = None,
    model_id: str = "codex-cli/default",
    priority: int = 2,
    cli_timeout: int = 120,
) -> JudgeModelConfig:
    """OpenAI Codex via `npx @openai/codex exec -` subprocess.

    Uses Codex Pro subscription auth — no OPENAI_API_KEY needed when logged in.
    model_alias is passed as -m to the codex CLI.
    Token counts are always 0; cost is $0.00.
    """
    return JudgeModelConfig(
        model_id=model_id,
        provider=APIProvider.CODEX_CLI,
        temperature=0.0,
        max_tokens=800,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        priority=priority,
        api_key_env=None,
        base_url=None,
        cli_model=model_alias,
        cli_timeout=cli_timeout,
    )


def get_deepseek_api_judge(
    model_id: str = "deepseek-v4-flash",
    priority: int = 2,
    max_tokens: int = 2500,
) -> JudgeModelConfig:
    """DeepSeek direct API judge via DeepSeek's OpenAI-compatible endpoint.

    Recommended judge candidates as of 2026-05-30:
      - deepseek-v4-flash: lowest cost, strong default for judge duties
      - deepseek-v4-pro: stronger but materially more expensive
      - deepseek-reasoner: legacy alias that now maps to v4-flash thinking mode and
        is scheduled for deprecation on 2026-07-24
    """
    pricing = {
        "deepseek-v4-flash": (0.14, 0.28),
        "deepseek-v4-pro": (0.435, 0.87),
        # Legacy compatibility alias; keep the same rough pricing as v4-flash.
        "deepseek-reasoner": (0.14, 0.28),
        "deepseek-chat": (0.14, 0.28),
    }
    input_cost, output_cost = pricing.get(model_id, pricing["deepseek-v4-flash"])
    return JudgeModelConfig(
        model_id=model_id,
        provider=APIProvider.DEEPSEEK,
        temperature=0.0,
        max_tokens=max_tokens,
        cost_per_1k_input=input_cost / 1000,
        cost_per_1k_output=output_cost / 1000,
        priority=priority,
        api_key_env="DEEPSEEK_API_KEY",
        base_url=_DEEPSEEK_BASE,
    )


def get_local_openai_judge(
    model_id: str,
    base_url: str,
    priority: int = 1,
    max_tokens: int = 1200,
    api_key_env: Optional[str] = None,
    request_timeout: float = 600.0,
    max_context_tokens: int = 4096,
    context_safety_margin: int = 64,
    extra_body: Optional[Dict[str, object]] = None,
) -> JudgeModelConfig:
    """OpenAI-compatible local judge endpoint.

    Intended for vLLM, SGLang, LMDeploy, TGI-compatible adapters, or any local
    server exposing `/v1/chat/completions`. `base_url` should include `/v1`,
    for example `http://127.0.0.1:8001/v1`.

    Local judges are reported as unmetered: token usage is still tracked when
    the server returns it, but cost remains $0.00.
    """
    return JudgeModelConfig(
        model_id=model_id,
        provider=APIProvider.LOCAL_OPENAI,
        temperature=0.0,
        max_tokens=max_tokens,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        priority=priority,
        api_key_env=api_key_env,
        base_url=base_url,
        request_timeout=request_timeout,
        max_context_tokens=max_context_tokens,
        context_safety_margin=context_safety_margin,
        extra_body=extra_body or {},
    )


def create_cli_committee(
    codex_model: Optional[str] = None,
    max_concurrent_requests: int = 4,
) -> JudgeCommitteeConfig:
    """CLI-based behavior committee using Codex CLI only.

    No OpenRouter API key is required for the committee itself.
    max_concurrent_requests is enforced via the semaphore in JudgeClient._call_codex_cli.
    """
    judges: List[JudgeModelConfig] = [
        get_codex_cli_judge(model_alias=codex_model, priority=2),
    ]

    return JudgeCommitteeConfig(
        judges=judges,
        voting_strategy="weighted_majority",
        max_concurrent_requests=max_concurrent_requests,
        timeout_seconds=150.0,
    )


def create_local_openai_committee(
    judges: List[Dict[str, object]],
    voting_strategy: str = "weighted_majority",
    max_concurrent_requests: int = 4,
    timeout_seconds: float = 600.0,
    response_cache_dir: Optional[str] = None,
    cache_mode: str = "off",
) -> JudgeCommitteeConfig:
    """Committee backed by locally hosted OpenAI-compatible endpoints."""
    judge_cfgs: List[JudgeModelConfig] = []
    for item in judges:
        model_id = str(item["model_id"])
        base_url = str(item["base_url"])
        judge_cfgs.append(
            get_local_openai_judge(
                model_id=model_id,
                base_url=base_url,
                priority=int(item.get("priority", _resolve_priority(model_id, None))),
                max_tokens=int(item.get("max_tokens", 1200)),
                api_key_env=item.get("api_key_env"),
                request_timeout=float(item.get("request_timeout", timeout_seconds)),
                max_context_tokens=int(item.get("max_context_tokens", 4096)),
                context_safety_margin=int(item.get("context_safety_margin", 64)),
                extra_body=item.get("extra_body") or {},
            )
        )

    return JudgeCommitteeConfig(
        judges=judge_cfgs,
        voting_strategy=voting_strategy,
        max_concurrent_requests=max_concurrent_requests,
        timeout_seconds=timeout_seconds,
        response_cache_dir=response_cache_dir,
        cache_mode=cache_mode,
    )


def create_mixed_committee(
    codex_model: Optional[str] = "gpt-5.4",
    codex_priority: int = 3,
    deepseek_model: str = "deepseek-v4-flash",
    deepseek_priority: int = 2,
    max_concurrent_requests: int = 4,
) -> JudgeCommitteeConfig:
    """Mixed committee: local Codex CLI judge + direct DeepSeek API judge.

    This path intentionally avoids OpenRouter. Use DEEPSEEK_API_KEY for the
    DeepSeek judge and the local Codex CLI session for the Codex judge.
    """
    judges: List[JudgeModelConfig] = [
        get_codex_cli_judge(model_alias=codex_model, priority=codex_priority),
        get_deepseek_api_judge(model_id=deepseek_model, priority=deepseek_priority),
    ]

    return JudgeCommitteeConfig(
        judges=judges,
        voting_strategy="weighted_majority",
        max_concurrent_requests=max_concurrent_requests,
        timeout_seconds=180.0,
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
    nli_judge=get_sonnet_nli_judge(),
)
eval_cfg = EvaluationConfig(
    model=model_cfg,
    trust=trust_cfg,
    conflict=conflict_cfg,
)
