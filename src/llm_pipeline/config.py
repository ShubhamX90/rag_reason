"""YAML config loader → typed dataclasses."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelParams:
    temperature: float = 0.0
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptsConfig:
    system: Path
    user: Path


@dataclass
class IOConfig:
    input: Path
    output: Path
    id_field: Optional[str] = "id"


@dataclass
class RetryConfig:
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 60.0
    non_retryable_status: list[int] = field(default_factory=lambda: [400, 401, 403, 404])


@dataclass
class ValidationConfig:
    mode: str = "none"          # "none" | "regex" | "json_schema"
    pattern: Optional[str] = None
    schema: Optional[dict] = None


@dataclass
class OutputConfig:
    include_raw_output: bool = True
    include_parsed: bool = True
    include_meta: bool = True


@dataclass
class PipelineConfig:
    provider: str
    model: str
    model_params: ModelParams
    prompts: PromptsConfig
    io: IOConfig
    retry: RetryConfig
    validation: ValidationConfig
    output: OutputConfig
    concurrency: int = 8
    limit: Optional[int] = None  # set at runtime via CLI


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    base_dir = path.parent

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Provider
    provider = raw.get("provider", "anthropic").lower()
    if provider not in {"openai", "anthropic"}:
        raise ValueError(f"Invalid provider: {provider!r}. Must be 'openai' or 'anthropic'.")

    # Model defaults
    defaults = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o"}
    model = raw.get("model") or defaults[provider]

    # ModelParams
    mp_raw = raw.get("model_params", {})
    known = {"temperature", "max_tokens"}
    extra = {k: v for k, v in mp_raw.items() if k not in known}
    model_params = ModelParams(
        temperature=mp_raw.get("temperature", 0.0),
        max_tokens=mp_raw.get("max_tokens", 1024),
        extra=extra,
    )

    # Prompts
    p_raw = raw.get("prompts", {})
    system_path = _resolve(base_dir, p_raw.get("system", "prompts/system.j2"))
    user_path = _resolve(base_dir, p_raw.get("user", "prompts/user.j2"))
    for p in (system_path, user_path):
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
    prompts = PromptsConfig(system=system_path, user=user_path)

    # IO
    io_raw = raw.get("io", {})
    input_path = _resolve(base_dir, io_raw.get("input", "data/input.jsonl"))
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path = _resolve(base_dir, io_raw.get("output", "data/output.jsonl"))
    id_field = io_raw.get("id_field", "id")
    io = IOConfig(input=input_path, output=output_path, id_field=id_field or None)

    # Retry
    r_raw = raw.get("retry", {})
    retry = RetryConfig(
        max_attempts=r_raw.get("max_attempts", 4),
        base_delay=r_raw.get("base_delay", 1.0),
        max_delay=r_raw.get("max_delay", 60.0),
        non_retryable_status=r_raw.get("non_retryable_status", [400, 401, 403, 404]),
    )

    # Validation
    v_raw = raw.get("validation", {})
    val_mode = v_raw.get("mode", "none")
    if val_mode not in {"none", "regex", "json_schema"}:
        raise ValueError(f"Invalid validation.mode: {val_mode!r}")
    validation = ValidationConfig(
        mode=val_mode,
        pattern=v_raw.get("pattern"),
        schema=v_raw.get("schema"),
    )

    # Output
    o_raw = raw.get("output", {})
    output_cfg = OutputConfig(
        include_raw_output=o_raw.get("include_raw_output", True),
        include_parsed=o_raw.get("include_parsed", True),
        include_meta=o_raw.get("include_meta", True),
    )

    return PipelineConfig(
        provider=provider,
        model=model,
        model_params=model_params,
        prompts=prompts,
        io=io,
        retry=retry,
        validation=validation,
        output=output_cfg,
        concurrency=raw.get("concurrency", 8),
    )


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()
