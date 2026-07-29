"""
src/committee_config.py
=======================
Small config helpers for selecting the multi-LLM committee backend.

The default path remains OpenRouter through src.voting.  Local OpenAI-compatible
configs are loaded from JSON so the stage scripts can run one local judge at a
time, populate a shared response cache, and later aggregate in read-only mode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.llm_cache import normalize_cache_mode
from src.llm_client import LLMClient, Provider
from src.voting import MODEL_WEIGHTS, normalize_priorities, set_model_weights


COMMITTEE_BACKENDS = {"openrouter", "local_openai"}


@dataclass
class JudgeConfig:
    model_id: str
    base_url: str
    priority: float
    max_tokens: Optional[int] = None
    request_timeout: Optional[float] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    api_key_env: Optional[str] = None
    base_url_env: Optional[str] = None


@dataclass
class CommitteeConfig:
    provider: str
    judges: List[JudgeConfig]
    response_cache_dir: Optional[str] = None
    cache_mode: str = "off"
    max_concurrent_requests: Optional[int] = None

    @property
    def priorities(self) -> Dict[str, float]:
        return {judge.model_id: float(judge.priority) for judge in self.judges}

    @property
    def normalized_weights(self) -> Dict[str, float]:
        return normalize_priorities(self.priorities)


def _as_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path.cwd() / p


def load_committee_config(path: str | Path) -> CommitteeConfig:
    config_path = _as_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    provider = str(raw.get("provider", raw.get("type", ""))).strip().lower()
    if provider not in COMMITTEE_BACKENDS:
        raise ValueError(
            f"Unsupported committee provider {provider!r} in {config_path}; "
            f"expected one of {sorted(COMMITTEE_BACKENDS)}"
        )
    if provider != "local_openai":
        raise ValueError(
            f"{config_path} is provider={provider!r}; explicit config files are "
            "currently only needed for local_openai"
        )

    judges_raw = raw.get("judges")
    if not isinstance(judges_raw, list) or not judges_raw:
        raise ValueError(f"{config_path} must contain a non-empty 'judges' list")

    judges: List[JudgeConfig] = []
    for idx, item in enumerate(judges_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Judge #{idx} in {config_path} must be an object")
        model_id = str(item.get("model_id", "")).strip()
        base_url_env = item.get("base_url_env")
        base_url = str(item.get("base_url", "")).strip()
        if base_url_env:
            base_url = os.getenv(str(base_url_env), "").strip() or base_url
        base_url = os.path.expandvars(base_url)
        if not model_id:
            raise ValueError(f"Judge #{idx} in {config_path} is missing model_id")
        if not base_url:
            raise ValueError(f"Judge {model_id!r} in {config_path} is missing base_url")
        priority = float(item.get("priority", item.get("weight", 1)))
        if priority <= 0:
            raise ValueError(f"Judge {model_id!r} priority must be positive")
        max_tokens = item.get("max_tokens")
        request_timeout = item.get("request_timeout")
        judges.append(
            JudgeConfig(
                model_id=model_id,
                base_url=base_url,
                priority=priority,
                max_tokens=int(max_tokens) if max_tokens is not None else None,
                request_timeout=(
                    float(request_timeout) if request_timeout is not None else None
                ),
                extra_body=item.get("extra_body") or {},
                api_key_env=item.get("api_key_env"),
                base_url_env=str(base_url_env) if base_url_env else None,
            )
        )

    cache_mode = normalize_cache_mode(str(raw.get("cache_mode", "off")))
    cache_dir = raw.get("response_cache_dir")
    max_concurrent = raw.get("max_concurrent_requests")

    return CommitteeConfig(
        provider=provider,
        judges=judges,
        response_cache_dir=str(cache_dir) if cache_dir else None,
        cache_mode=cache_mode,
        max_concurrent_requests=int(max_concurrent) if max_concurrent is not None else None,
    )


def configure_committee_for_backend(
    *,
    backend: str,
    config_path: Optional[str] = None,
    cache_mode_override: Optional[str] = None,
    cache_dir_override: Optional[str] = None,
) -> Optional[CommitteeConfig]:
    """Apply backend-specific committee settings and return local config.

    For OpenRouter, this leaves src.voting's current defaults untouched.
    For local_openai, this loads the judge config and mutates MODEL_WEIGHTS /
    COMMITTEE_MODELS in-place with normalized priorities.
    """
    backend = (backend or "openrouter").strip().lower()
    if backend not in COMMITTEE_BACKENDS:
        raise ValueError(f"Unknown committee backend {backend!r}")
    if backend == "openrouter":
        if config_path:
            raise ValueError("--committee-config is only valid with --committee-backend local_openai")
        if cache_dir_override:
            os.environ["RAG_REASON_LLM_CACHE_DIR"] = str(_as_path(cache_dir_override))
        return None
    if not config_path:
        raise ValueError("--committee-config is required with --committee-backend local_openai")

    cfg = load_committee_config(config_path)
    if cache_mode_override:
        cfg.cache_mode = normalize_cache_mode(cache_mode_override)
    if cache_dir_override:
        cfg.response_cache_dir = cache_dir_override

    if cfg.response_cache_dir:
        os.environ["RAG_REASON_LLM_CACHE_DIR"] = str(_as_path(cfg.response_cache_dir))

    set_model_weights(cfg.priorities, normalize=True)
    return cfg


def build_clients_for_committee(
    *,
    backend: str,
    committee_config: Optional[CommitteeConfig],
    temperature: float,
    max_retries: int,
) -> Dict[str, LLMClient]:
    backend = (backend or "openrouter").strip().lower()
    if backend == "openrouter":
        return {
            model: LLMClient(
                provider=Provider.OPENROUTER,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
            )
            for model in MODEL_WEIGHTS
        }

    if committee_config is None:
        raise ValueError("local_openai backend requires a loaded CommitteeConfig")

    clients: Dict[str, LLMClient] = {}
    for judge in committee_config.judges:
        clients[judge.model_id] = LLMClient(
            provider=Provider.LOCAL_OPENAI,
            model=judge.model_id,
            temperature=temperature,
            max_retries=max_retries,
            base_url=judge.base_url,
            api_key_env=judge.api_key_env,
            request_timeout=judge.request_timeout,
            extra_body=judge.extra_body,
            default_max_tokens=judge.max_tokens,
        )
    return clients


def cache_mode_for_backend(
    *,
    backend: str,
    committee_config: Optional[CommitteeConfig],
    use_cache: bool,
    cache_mode_override: Optional[str] = None,
) -> str:
    if cache_mode_override:
        return normalize_cache_mode(cache_mode_override)
    if backend == "local_openai" and committee_config is not None:
        return committee_config.cache_mode
    return normalize_cache_mode(None, cache_enabled=use_cache)


def cache_dir_for_backend(committee_config: Optional[CommitteeConfig]) -> Optional[str]:
    return committee_config.response_cache_dir if committee_config else None


def max_concurrency_for_backend(
    current: int,
    committee_config: Optional[CommitteeConfig],
) -> int:
    if committee_config and committee_config.max_concurrent_requests:
        return int(committee_config.max_concurrent_requests)
    return current


def client_max_tokens(client: LLMClient, fallback: int) -> int:
    return int(client.default_max_tokens or fallback)


def describe_committee(
    *,
    backend: str,
    committee_config: Optional[CommitteeConfig],
) -> Iterable[str]:
    if backend == "local_openai" and committee_config is not None:
        for judge in committee_config.judges:
            weight = committee_config.normalized_weights[judge.model_id]
            yield (
                f"   priority={judge.priority:g} weight={weight:.1%} "
                f"{judge.model_id} @ {judge.base_url}"
            )
        return
    for model, weight in MODEL_WEIGHTS.items():
        yield f"   {weight:.0%}  {model}"
