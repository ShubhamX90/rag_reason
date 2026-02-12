# rag_eval/config.py
# -*- coding: utf-8 -*-
"""
Enhanced Configuration for CATS v2.0 Evaluation Pipeline
--------------------------------------------------------
Supports multi-LLM judge committee with Anthropic and OpenRouter APIs.

Key Upgrades:
  • Multi-judge committee voting system
  • Cost-aware model selection
  • Enhanced conflict evaluation metrics
  • Flexible API routing (Anthropic/OpenRouter)

Authors: Enhanced by Claude AI
Institution: Birla Institute of Technology and Science, Pilani
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
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
    cost_per_1k_input: float = 0.0  # USD per 1k input tokens
    cost_per_1k_output: float = 0.0  # USD per 1k output tokens
    priority: int = 1  # Higher priority judges get more weight
    
    # API-specific settings
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000


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
    confidence_threshold: float = 0.6  # minimum agreement required
    
    # Performance optimizations
    use_async: bool = True
    max_concurrent_requests: int = 75
    retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Cost management
    cost_optimization: bool = True
    max_cost_per_sample: float = 0.05  # USD
    prefer_cheaper_models: bool = True


# --------------------
# Core Model Config
# --------------------
@dataclass
class ModelConfig:
    """Base configuration for evaluation models."""
    name: str = "claude-3-5-haiku-20241022"
    provider: APIProvider = APIProvider.ANTHROPIC
    temperature: float = 0.0
    max_tokens: int = 2048
    seed: int = 42


# --------------------
# Enhanced TRUST-SCORE Config
# --------------------
@dataclass
class EnhancedTrustScoreConfig:
    """
    Enhanced TRUST-SCORE evaluation with improved metrics.
    
    New Features:
      • Multi-dimensional grounding scoring
      • Citation quality assessment
      • Source reliability weighting
      • Temporal consistency checks
    """
    enable_rag_eval: bool = True
    
    # Core metrics
    compute_macro: bool = True
    compute_correctness: bool = True
    compute_citations: bool = True
    compute_consistency: bool = True  # NEW: Cross-claim consistency
    
    # Enhanced grounding
    use_hierarchical_grounding: bool = True  # NEW: Sentence → Claim → Document hierarchy
    penalize_hallucinations: bool = True
    reward_source_diversity: bool = True
    
    # Citation analysis
    check_citation_accuracy: bool = True  # NEW: Verify cited docs actually support claims
    assess_citation_necessity: bool = True  # NEW: Are citations sufficient?
    
    # Temporal & reliability
    check_temporal_consistency: bool = True  # NEW: Newer sources override old
    weight_by_source_quality: bool = True  # NEW: Trusted sources weighted higher
    
    # Scoring parameters
    use_gold_answerability: bool = True
    max_claims_per_answer: int = 12
    min_support_ratio: float = 0.7  # Minimum fraction of claims that must be grounded
    
    # NLI model for grounding
    nli_model: str = "local"  # "local" or "llm"
    nli_threshold: float = 0.5


# --------------------
# Enhanced Conflict-Aware Config
# --------------------
@dataclass
class EnhancedConflictEvalConfig:
    """
    Enhanced conflict-aware evaluation with multi-judge committee.
    
    New Features:
      • Multi-judge voting for behavior adherence
      • Enhanced factual grounding with cross-doc verification
      • Improved single-truth recall with paraphrase detection
      • Conflict resolution strategy assessment
    """
    enable_conflict_eval: bool = True
    
    # Judge committee
    use_judge_committee: bool = True
    committee: Optional[JudgeCommitteeConfig] = None
    
    # Single judge fallback
    single_judge_model: str = "claude-3-5-haiku-20241022"
    judge_provider: APIProvider = APIProvider.ANTHROPIC
    judge_temperature: float = 0.0
    
    # NLI settings
    use_llm_for_nli: bool = True
    nli_model: str = "claude-3-5-haiku-20241022"
    
    # Conflict types and evaluation
    single_truth_types: Tuple[int, ...] = (1, 2, 4, 5)
    
    # Enhanced behavior adherence
    check_viewpoint_balance: bool = True  # NEW: For type 3, check if multiple views presented
    check_temporal_precedence: bool = True  # NEW: For type 4, newer info prioritized
    check_misinformation_rejection: bool = True  # NEW: For type 5, bad sources rejected
    
    # Enhanced factual grounding
    require_cross_doc_verification: bool = False  # NEW: Claims supported by multiple docs
    penalize_unsupported_claims: bool = True
    weight_support_by_doc_quality: bool = True
    
    # Enhanced single-truth recall
    use_semantic_matching: bool = True  # NEW: Beyond string matching
    allow_paraphrases: bool = True
    check_partial_answers: bool = True  # NEW: Award partial credit
    
    # Citation requirements
    require_inline_citations: bool = False
    max_claims_per_answer: int = 5
    
    # Scoring enhancements
    aggregate_by_conflict_type: bool = True
    compute_conflict_resolution_score: bool = True  # NEW: How well model handles conflicts


# --------------------
# Pipeline Config
# --------------------
@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline execution."""
    # Parallelization
    use_async_evaluation: bool = True
    max_workers: int = 50
    batch_size: int = 100
    
    # Caching
    enable_caching: bool = True
    cache_dir: str = ".cache"
    
    # Error handling
    skip_on_error: bool = False
    log_errors: bool = True
    
    # Progress reporting
    show_progress: bool = True
    verbose: bool = True


# --------------------
# Master Evaluation Config
# --------------------
@dataclass
class EvaluationConfig:
    """
    Master evaluation configuration for CATS v2.0.
    Unifies all evaluation components with enhanced features.
    """
    # Paths
    input_jsonl: str = "data/input.jsonl"
    outputs_dir: str = "outputs/"
    report_md: str = "outputs/eval_report.md"
    detailed_results_json: str = "outputs/detailed_results.json"
    
    # Core
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Evaluation subsystems
    trust: EnhancedTrustScoreConfig = field(default_factory=EnhancedTrustScoreConfig)
    conflict: EnhancedConflictEvalConfig = field(default_factory=EnhancedConflictEvalConfig)
    
    # Reporting
    per_type_breakdown: bool = True
    save_per_sample_scores: bool = True
    generate_visualizations: bool = True


# --------------------
# Predefined Judge Configurations
# --------------------
def get_haiku_judge() -> JudgeModelConfig:
    """Anthropic Claude Haiku - Fast and economical."""
    return JudgeModelConfig(
        model_id="claude-3-5-haiku-20241022",
        provider=APIProvider.ANTHROPIC,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.001,  # $1 per MTok = $0.001 per 1k tokens
        cost_per_1k_output=0.005,  # $5 per MTok = $0.005 per 1k tokens
        priority=2,
        api_key_env="ANTHROPIC_API_KEY",
        max_requests_per_minute=100
    )


def get_deepseek_judge() -> JudgeModelConfig:
    """DeepSeek R1 via OpenRouter - Reasoning capabilities."""
    return JudgeModelConfig(
        model_id="deepseek/deepseek-r1",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.00055,  # $0.55 per MTok = $0.00055 per 1k tokens
        cost_per_1k_output=0.00219,  # $2.19 per MTok = $0.00219 per 1k tokens
        priority=3,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        max_requests_per_minute=30
    )


def get_qwen_judge() -> JudgeModelConfig:
    """Qwen via OpenRouter - Balanced performance."""
    return JudgeModelConfig(
        model_id="qwen/qwen-2.5-7b-instruct",
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.00006,  # $0.06 per MTok = $0.00006 per 1k tokens
        cost_per_1k_output=0.00006,  # $0.06 per MTok = $0.00006 per 1k tokens
        priority=1,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        max_requests_per_minute=60
    )


def get_mistral_nemo_judge() -> JudgeModelConfig:
    """Mistral Nemo via OpenRouter - Free tier option."""
    return JudgeModelConfig(
        model_id="mistralai/mistral-nemo",  # Removed :free suffix - use canonical slug
        provider=APIProvider.OPENROUTER,
        temperature=0.0,
        max_tokens=500,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        priority=1,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        max_requests_per_minute=100
    )


def create_default_committee() -> JudgeCommitteeConfig:
    """
    Create a cost-effective default judge committee.
    Uses: Haiku (Anthropic) + DeepSeek + Qwen + Mistral (Free)
    """
    return JudgeCommitteeConfig(
        judges=[
            get_haiku_judge(),
            get_deepseek_judge(),
            get_qwen_judge(),
            get_mistral_nemo_judge(),  # Added Mistral (free) to default committee
        ],
        voting_strategy="weighted_majority",
        confidence_threshold=0.6,
        use_async=True,
        max_concurrent_requests=50,  # Increased from 10 to 50 for better parallelism
        cost_optimization=True,
        max_cost_per_sample=0.02,
        prefer_cheaper_models=True
    )


def create_conservative_committee() -> JudgeCommitteeConfig:
    """
    Create a very cost-effective committee for large-scale evaluation.
    Uses: Haiku + Qwen + Mistral Nemo (free)
    """
    return JudgeCommitteeConfig(
        judges=[
            get_haiku_judge(),
            get_qwen_judge(),
            get_mistral_nemo_judge(),
        ],
        voting_strategy="weighted_majority",
        confidence_threshold=0.6,
        use_async=True,
        max_concurrent_requests=50,  # Increased from 15 to 50
        cost_optimization=True,
        max_cost_per_sample=0.01,
        prefer_cheaper_models=True
    )


# --------------------
# Global Defaults
# --------------------
model_cfg = ModelConfig()
trust_cfg = EnhancedTrustScoreConfig()
conflict_cfg = EnhancedConflictEvalConfig(
    use_judge_committee=True,
    committee=create_default_committee()
)
eval_cfg = EvaluationConfig(
    trust=trust_cfg,
    conflict=conflict_cfg
)
