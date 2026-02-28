"""Thin async LLM wrapper — no retry logic."""
from __future__ import annotations

import os
from typing import Any, Optional


class LLMClient:
    """Async context manager wrapping Anthropic or OpenAI clients."""

    _PROVIDER_DEFAULTS = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
    }

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **extra_params: Any,
    ):
        self.provider = provider
        self.model = model or self._PROVIDER_DEFAULTS[provider]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = extra_params
        self._client = None

    async def __aenter__(self) -> "LLMClient":
        if self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable is not set."
                )
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY environment variable is not set."
                )
            import openai
            self._client = openai.AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")
        return self

    async def complete(self, system: str, user: str) -> str:
        """Single LLM call; returns raw response text."""
        if self.provider == "anthropic":
            return await self._complete_anthropic(system, user)
        return await self._complete_openai(system, user)

    async def _complete_anthropic(self, system: str, user: str) -> str:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            **self.extra_params,
        )
        return response.content[0].text

    async def _complete_openai(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.extra_params,
        )
        return response.choices[0].message.content

    async def __aexit__(self, *args):
        if self._client is not None:
            await self._client.__aexit__(*args)
            self._client = None
