# rag_eval/judge_committee.py
# -*- coding: utf-8 -*-
"""
Multi-LLM Judge Committee with Voting System
--------------------------------------------
Implements a committee of LLM judges that vote on evaluation decisions.

Features:
  • Parallel async judge execution (all judges run concurrently via asyncio.gather)
  • Weighted majority voting (weight = priority × confidence)
  • Per-judge cost tracking
  • Loud failure modes (no silent fallback to adherent=False)

Note: max_concurrent_requests is stored in JudgeCommitteeConfig but NOT enforced
here — all judge calls fan out without a semaphore. See ISSUES.md §3.3.

Authors: Enhanced by Claude AI
"""

import asyncio
import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import anthropic
import httpx

from .config import JudgeCommitteeConfig, JudgeModelConfig, APIProvider

logger = logging.getLogger(__name__)


# --------------------
# Judge Response
# --------------------
@dataclass
class JudgeResponse:
    """Response from a single judge."""
    judge_id: str
    model_id: str
    provider: str
    adherent: bool
    rationale: str
    confidence: float = 1.0
    cost: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_id": self.judge_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "adherent": self.adherent,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


# --------------------
# Committee Decision
# --------------------
@dataclass
class CommitteeDecision:
    """Aggregated decision from the judge committee."""
    adherent: bool
    confidence: float          # majority-side confidence in [0, 1]
    minority_confidence: float # minority-side confidence in [0, 1] — useful for partial-credit logic
    votes_for: int
    votes_against: int
    total_votes: int
    rationale: str
    individual_responses: List[JudgeResponse]
    total_cost: float
    total_latency_ms: float
    all_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adherent": self.adherent,
            "confidence": self.confidence,
            "minority_confidence": self.minority_confidence,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "total_votes": self.total_votes,
            "rationale": self.rationale,
            "individual_responses": [r.to_dict() for r in self.individual_responses],
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency_ms,
            "all_failed": self.all_failed,
        }


# --------------------
# Single Judge Client
# --------------------
class JudgeClient:
    """Client for a single judge model."""

    def __init__(self, config: JudgeModelConfig):
        self.config = config

        api_key_var = config.api_key_env or ""
        self.api_key = os.getenv(api_key_var)

        if not self.api_key:
            logger.warning(
                f"API key not found for {config.model_id}. Looking for env var: {api_key_var}"
            )

        if config.provider == APIProvider.ANTHROPIC:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        elif config.provider == APIProvider.OPENROUTER:
            self.base_url = config.base_url or "https://openrouter.ai/api/v1"
            self.http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/CATS-eval",
                    "X-Title": "CATS Eval Pipeline",
                },
                timeout=60.0,
            )

        self.total_cost = 0.0
        self.request_count = 0

    async def judge_behavior(self, prompt: str) -> JudgeResponse:
        """Send a behavior judgment request and return a structured response."""
        start_time = time.time()

        try:
            if self.config.provider == APIProvider.ANTHROPIC:
                response_text, input_tokens, output_tokens = await self._call_anthropic(prompt)
            elif self.config.provider == APIProvider.OPENROUTER:
                response_text, input_tokens, output_tokens = await self._call_openrouter(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")

            parsed = self._parse_judge_response(response_text)

            cost = (input_tokens * self.config.cost_per_1k_input / 1000
                    + output_tokens * self.config.cost_per_1k_output / 1000)

            self.total_cost += cost
            self.request_count += 1

            latency_ms = (time.time() - start_time) * 1000

            return JudgeResponse(
                judge_id=f"{self.config.provider.value}_{self.config.model_id}",
                model_id=self.config.model_id,
                provider=self.config.provider.value,
                adherent=parsed["adherent"],
                rationale=parsed["rationale"],
                confidence=parsed["confidence"],
                cost=cost,
                latency_ms=latency_ms,
                error=parsed.get("parse_error"),  # surfaces parse failures so weighting can ignore them
            )

        except Exception as e:
            logger.error(f"Judge {self.config.model_id} error: {e}")
            latency_ms = (time.time() - start_time) * 1000

            return JudgeResponse(
                judge_id=f"{self.config.provider.value}_{self.config.model_id}",
                model_id=self.config.model_id,
                provider=self.config.provider.value,
                adherent=False,
                rationale="Judge error",
                confidence=0.0,
                cost=0.0,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def judge_nli(self, prompt: str) -> Dict[str, Any]:
        """Send an NLI judgment request. Returns dict with relation, confidence, cost."""
        start_time = time.time()

        try:
            if self.config.provider == APIProvider.ANTHROPIC:
                response_text, input_tokens, output_tokens = await self._call_anthropic(prompt)
            elif self.config.provider == APIProvider.OPENROUTER:
                response_text, input_tokens, output_tokens = await self._call_openrouter(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")

            relation, nli_confidence = self._parse_nli_response(response_text)

            cost = (input_tokens * self.config.cost_per_1k_input / 1000
                    + output_tokens * self.config.cost_per_1k_output / 1000)

            self.total_cost += cost
            self.request_count += 1

            latency_ms = (time.time() - start_time) * 1000

            return {
                "relation": relation,
                "confidence": nli_confidence,
                "cost": cost,
                "latency_ms": latency_ms,
                "error": None,
            }

        except Exception as e:
            logger.error(f"NLI judge {self.config.model_id} error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return {
                "relation": "neutral",
                "confidence": 0.0,
                "cost": 0.0,
                "latency_ms": latency_ms,
                "error": str(e),
            }

    async def _call_anthropic(self, prompt: str) -> Tuple[str, int, int]:
        """Call Anthropic API. Returns (response_text, input_tokens, output_tokens)."""
        message = await self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text, message.usage.input_tokens, message.usage.output_tokens

    async def _call_openrouter(self, prompt: str) -> Tuple[str, int, int]:
        """Call OpenRouter API. Guards against empty/error responses."""
        response = await self.http_client.post(
            "/chat/completions",
            json={
                "model": self.config.model_id,
                "messages": [
                    {"role": "system", "content": "You are an evaluation judge. Respond ONLY with JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            err = data.get("error", "no choices in response")
            raise RuntimeError(f"OpenRouter returned no choices: {err}")

        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""

        # DeepSeek R1 wraps reasoning in <think>...</think> before the JSON.
        # Strip the trailing JSON if a think block is present.
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()

        usage = data.get("usage", {}) or {}
        input_tokens = usage.get("prompt_tokens") or (len(prompt) // 4)
        output_tokens = usage.get("completion_tokens") or (len(content) // 4)

        return content, input_tokens, output_tokens

    def _parse_judge_response(self, text: str) -> Dict[str, Any]:
        """Parse a behavior-judge JSON response. Records parse failures explicitly."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            obj = json.loads(text[start:end])

            raw_conf = obj.get("confidence", 0.5)
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            return {
                "adherent": bool(obj.get("adherent", False)),
                "rationale": str(obj.get("rationale", "")),
                "confidence": conf,
            }
        except Exception as e:
            logger.warning(f"Failed to parse judge response for {self.config.model_id}: {e}")
            return {
                "adherent": False,
                "rationale": "Parse error",
                "confidence": 0.0,
                "parse_error": str(e),
            }

    def _parse_nli_response(self, text: str) -> Tuple[str, float]:
        """Parse an NLI JSON response. Returns (relation, confidence)."""
        valid = {"entails", "contradicts", "neutral"}
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in NLI response")

            obj = json.loads(text[start:end])
            relation = str(obj.get("relation", "neutral")).strip().lower()
            if relation not in valid:
                relation = "neutral"

            raw_conf = obj.get("confidence", 0.5)
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            return relation, conf

        except Exception as e:
            logger.warning(f"Failed to parse NLI response: {e}. Text: {text[:120]}")
            return "neutral", 0.0

    async def close(self):
        if hasattr(self, "http_client"):
            await self.http_client.aclose()


# --------------------
# Judge Committee
# --------------------
class JudgeCommittee:
    """Multi-LLM judge committee with parallel async voting."""

    def __init__(self, config: JudgeCommitteeConfig):
        self.config = config
        self.judges = [JudgeClient(j) for j in config.judges]
        self.total_cost = 0.0
        self.decision_count = 0

        logger.info(f"Initialized committee with {len(self.judges)} judges:")
        for judge in config.judges:
            logger.info(f"  - {judge.model_id} ({judge.provider.value}) [priority={judge.priority}]")

    async def judge_behavior(self, prompt: str) -> CommitteeDecision:
        """Get behavior judgment from the committee. All judges run in parallel."""
        tasks = [judge.judge_behavior(prompt) for judge in self.judges]
        responses = await asyncio.gather(*tasks)

        valid_responses = [r for r in responses if r.error is None]

        if not valid_responses:
            logger.error("All judges failed!")
            return self._create_fallback_decision(responses)

        decision = self._aggregate_votes(valid_responses)
        self.total_cost += decision.total_cost
        self.decision_count += 1
        return decision

    def _aggregate_votes(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        if self.config.voting_strategy == "majority":
            return self._majority_vote(responses)
        elif self.config.voting_strategy == "unanimous":
            return self._unanimous_vote(responses)
        # default
        return self._weighted_majority_vote(responses)

    def _majority_vote(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        votes_for = sum(1 for r in responses if r.adherent)
        votes_against = len(responses) - votes_for
        adherent = votes_for > votes_against
        confidence = max(votes_for, votes_against) / len(responses)
        minority_confidence = min(votes_for, votes_against) / len(responses)

        majority_responses = [r for r in responses if r.adherent == adherent]
        rationale = majority_responses[0].rationale if majority_responses else ""

        return CommitteeDecision(
            adherent=adherent,
            confidence=confidence,
            minority_confidence=minority_confidence,
            votes_for=votes_for,
            votes_against=votes_against,
            total_votes=len(responses),
            rationale=rationale,
            individual_responses=responses,
            total_cost=sum(r.cost for r in responses),
            total_latency_ms=sum(r.latency_ms for r in responses),
        )

    def _weighted_majority_vote(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        priority_map = {j.model_id: j.priority for j in self.config.judges}

        weighted_for = 0.0
        weighted_against = 0.0
        total_weight = 0.0

        for r in responses:
            priority = priority_map.get(r.model_id, 1)
            weight = priority * max(r.confidence, 0.01)  # floor to avoid 0-weight when judge emits confidence=0
            total_weight += weight
            if r.adherent:
                weighted_for += weight
            else:
                weighted_against += weight

        adherent = weighted_for > weighted_against
        if total_weight > 0:
            confidence = max(weighted_for, weighted_against) / total_weight
            minority_confidence = min(weighted_for, weighted_against) / total_weight
        else:
            confidence = 0.0
            minority_confidence = 0.0

        winning_responses = [r for r in responses if r.adherent == adherent]
        if winning_responses:
            best = max(
                winning_responses,
                key=lambda r: priority_map.get(r.model_id, 1) * r.confidence,
            )
            rationale = best.rationale
        else:
            rationale = ""

        votes_for = sum(1 for r in responses if r.adherent)
        votes_against = len(responses) - votes_for

        return CommitteeDecision(
            adherent=adherent,
            confidence=confidence,
            minority_confidence=minority_confidence,
            votes_for=votes_for,
            votes_against=votes_against,
            total_votes=len(responses),
            rationale=rationale,
            individual_responses=responses,
            total_cost=sum(r.cost for r in responses),
            total_latency_ms=sum(r.latency_ms for r in responses),
        )

    def _unanimous_vote(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        votes_for = sum(1 for r in responses if r.adherent)
        votes_against = len(responses) - votes_for
        all_adherent = votes_for == len(responses)
        # Use majority-side fraction so confidence still has meaning when not unanimous
        confidence = max(votes_for, votes_against) / len(responses) if responses else 0.0
        minority_confidence = min(votes_for, votes_against) / len(responses) if responses else 0.0
        rationale = responses[0].rationale if responses else ""

        return CommitteeDecision(
            adherent=all_adherent,
            confidence=confidence,
            minority_confidence=minority_confidence,
            votes_for=votes_for,
            votes_against=votes_against,
            total_votes=len(responses),
            rationale=rationale,
            individual_responses=responses,
            total_cost=sum(r.cost for r in responses),
            total_latency_ms=sum(r.latency_ms for r in responses),
        )

    def _create_fallback_decision(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        """Used only when EVERY judge errored — marks all_failed so callers can skip."""
        return CommitteeDecision(
            adherent=False,
            confidence=0.0,
            minority_confidence=0.0,
            votes_for=0,
            votes_against=len(responses),
            total_votes=len(responses),
            rationale="All judges failed",
            individual_responses=responses,
            total_cost=0.0,
            total_latency_ms=sum(r.latency_ms for r in responses),
            all_failed=True,
        )

    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": self.total_cost,
            "decisions_made": self.decision_count,
            "avg_cost_per_decision": self.total_cost / max(1, self.decision_count),
            "per_judge_costs": {
                judge.config.model_id: {
                    "total_cost": judge.total_cost,
                    "requests": judge.request_count,
                    "avg_cost": judge.total_cost / max(1, judge.request_count),
                }
                for judge in self.judges
            },
        }

    async def close(self):
        for judge in self.judges:
            await judge.close()
