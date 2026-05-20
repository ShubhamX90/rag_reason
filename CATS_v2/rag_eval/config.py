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
    """
    judges: List[JudgeModelConfig] = field(default_factory=list)

    # Voting strategy
    voting_strategy: str = "weighted_majority"  # options: majority, unanimous, weighted_majority

    # Concurrency control (enforced by JudgeCommittee via asyncio.Semaphore)
    max_concurrent_requests: int = 50


# --------------------
# Enhanced Conflict-Aware Config
# --------------------
@dataclass
class EnhancedConflictEvalConfig:
    """
    Enhanced conflict-aware evaluation with multi-judge committee.
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

    # Scoring enhancements
    aggregate_by_conflict_type: bool = True

    # Refusal carve-out:
    # When the model correctly refuses an unanswerable question
    # (pred_answered=False AND gold_answerable=False), treat all metrics
    # as 1.0 instead of letting behavior/grounding zero out the sample.
    correct_refusal_full_credit: bool = True


# --------------------
# Pipeline Config
# --------------------
@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline execution."""
    use_async_evaluation: bool = True
    batch_size: int = 100
    skip_on_error: bool = False
    show_progress: bool = True
    verbose: bool = True


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

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    conflict: EnhancedConflictEvalConfig = field(default_factory=EnhancedConflictEvalConfig)


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
