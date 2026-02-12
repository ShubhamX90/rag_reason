# rag_eval/judge_committee.py
# -*- coding: utf-8 -*-
"""
Multi-LLM Judge Committee with Voting System
--------------------------------------------
Implements a committee of LLM judges that vote on evaluation decisions.

Features:
  • Parallel async judge execution
  • Weighted majority voting
  • Confidence scoring
  • Cost tracking and optimization
  • Fallback mechanisms

Authors: Enhanced by Claude AI
"""

import asyncio
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
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
            "error": self.error
        }


# --------------------
# Committee Decision
# --------------------
@dataclass
class CommitteeDecision:
    """Aggregated decision from the judge committee."""
    adherent: bool
    confidence: float
    votes_for: int
    votes_against: int
    total_votes: int
    rationale: str
    individual_responses: List[JudgeResponse]
    total_cost: float
    total_latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adherent": self.adherent,
            "confidence": self.confidence,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "total_votes": self.total_votes,
            "rationale": self.rationale,
            "individual_responses": [r.to_dict() for r in self.individual_responses],
            "total_cost": self.total_cost,
            "total_latency_ms": self.total_latency_ms
        }


# --------------------
# Single Judge Client
# --------------------
class JudgeClient:
    """Client for a single judge model."""
    
    def __init__(self, config: JudgeModelConfig):
        self.config = config
        
        # Load API key from environment
        api_key_var = config.api_key_env or ""
        self.api_key = os.getenv(api_key_var)
        
        # Validate API key
        if not self.api_key:
            logger.warning(f"API key not found for {config.model_id}. Looking for env var: {api_key_var}")
            logger.warning(f"Please set {api_key_var} in your .env file")
        
        # Initialize API clients
        if config.provider == APIProvider.ANTHROPIC:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        elif config.provider == APIProvider.OPENROUTER:
            # OpenRouter uses OpenAI-compatible API
            self.base_url = config.base_url or "https://openrouter.ai/api/v1"
            self.http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/CATS-eval",
                    "X-Title": "CATS Eval Pipeline"
                },
                timeout=30.0
            )
        
        self.total_cost = 0.0
        self.request_count = 0
    
    async def judge_behavior(self, prompt: str) -> JudgeResponse:
        """
        Send a behavior judgment request to this judge.
        Returns structured JudgeResponse with voting decision.
        """
        import time
        start_time = time.time()
        
        try:
            if self.config.provider == APIProvider.ANTHROPIC:
                response_text, input_tokens, output_tokens = await self._call_anthropic(prompt)
            elif self.config.provider == APIProvider.OPENROUTER:
                response_text, input_tokens, output_tokens = await self._call_openrouter(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            # Parse JSON response
            parsed = self._parse_judge_response(response_text)
            
            # Calculate cost using actual token counts
            cost = (input_tokens * self.config.cost_per_1k_input / 1000 + 
                   output_tokens * self.config.cost_per_1k_output / 1000)
            
            self.total_cost += cost
            self.request_count += 1
            
            latency_ms = (time.time() - start_time) * 1000
            
            return JudgeResponse(
                judge_id=f"{self.config.provider.value}_{self.config.model_id}",
                model_id=self.config.model_id,
                provider=self.config.provider.value,
                adherent=parsed["adherent"],
                rationale=parsed["rationale"],
                confidence=parsed.get("confidence", 1.0),
                cost=cost,
                latency_ms=latency_ms
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
                error=str(e)
            )
    
    async def judge_nli(self, prompt: str) -> Dict[str, Any]:
        """
        Send an NLI judgment request to this judge.
        Returns dict with relation and cost info.
        """
        import time
        start_time = time.time()
        
        try:
            if self.config.provider == APIProvider.ANTHROPIC:
                response_text, input_tokens, output_tokens = await self._call_anthropic(prompt)
            elif self.config.provider == APIProvider.OPENROUTER:
                response_text, input_tokens, output_tokens = await self._call_openrouter(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            # Parse NLI response - expects {"relation": "entails"|"contradicts"|"neutral"}
            import json as json_lib
            try:
                # Find JSON in response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in NLI response")
                
                json_str = response_text[start_idx:end_idx]
                parsed = json_lib.loads(json_str)
                relation = parsed.get("relation", "neutral").lower()
            except Exception as e:
                logger.warning(f"Failed to parse NLI response, defaulting to neutral: {e}")
                relation = "neutral"
            
            # Calculate cost
            cost = (input_tokens * self.config.cost_per_1k_input / 1000 + 
                   output_tokens * self.config.cost_per_1k_output / 1000)
            
            self.total_cost += cost
            self.request_count += 1
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "relation": relation,
                "cost": cost,
                "latency_ms": latency_ms,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"NLI judge {self.config.model_id} error: {e}")
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "relation": "neutral",
                "cost": 0.0,
                "latency_ms": latency_ms,
                "error": str(e)
            }
    
    async def _call_anthropic(self, prompt: str) -> Tuple[str, int, int]:
        """Call Anthropic API and return response text + token counts."""
        message = await self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        # Get actual token counts from usage
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        return message.content[0].text, input_tokens, output_tokens
    
    async def _call_openrouter(self, prompt: str) -> Tuple[str, int, int]:
        """Call OpenRouter API (OpenAI-compatible) and return response + token counts."""
        response = await self.http_client.post(
            "/chat/completions",
            json={
                "model": self.config.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an evaluation judge. Respond ONLY with JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract token counts from usage (if available)
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", len(prompt) // 4)  # fallback to estimate
        output_tokens = usage.get("completion_tokens", len(data["choices"][0]["message"]["content"]) // 4)
        
        return data["choices"][0]["message"]["content"], input_tokens, output_tokens
    
    def _parse_judge_response(self, text: str) -> Dict[str, Any]:
        """Parse judge response JSON."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = text[start:end]
            obj = json.loads(json_str)
            
            return {
                "adherent": bool(obj.get("adherent", False)),
                "rationale": str(obj.get("rationale", "")),
                "confidence": float(obj.get("confidence", 1.0))
            }
        except Exception as e:
            logger.warning(f"Failed to parse judge response: {e}")
            return {
                "adherent": False,
                "rationale": "Parse error",
                "confidence": 0.0
            }
    
    def _parse_nli_response(self, text: str) -> str:
        """Parse NLI response to extract the relation field."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = text[start:end]
            obj = json.loads(json_str)
            
            # Get relation field
            relation = obj.get("relation", "").lower()
            
            # Validate relation value
            if relation in ["entails", "contradicts", "neutral"]:
                return relation
            else:
                # Fallback: check if any of these words appear in the text
                text_lower = text.lower()
                if "entails" in text_lower and "contradicts" not in text_lower:
                    return "entails"
                elif "contradicts" in text_lower:
                    return "contradicts"
                else:
                    return "neutral"
                    
        except Exception as e:
            logger.warning(f"Failed to parse NLI response: {e}. Text: {text[:100]}")
            # Fallback: simple text search
            text_lower = text.lower()
            if "entails" in text_lower and "contradicts" not in text_lower:
                return "entails"
            elif "contradicts" in text_lower:
                return "contradicts"
            else:
                return "neutral"
    
    async def close(self):
        """Close async clients."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()


# --------------------
# Judge Committee
# --------------------
class JudgeCommittee:
    """
    Multi-LLM judge committee with voting mechanism.
    Coordinates multiple judges and aggregates their decisions.
    """
    
    def __init__(self, config: JudgeCommitteeConfig):
        self.config = config
        self.judges = [JudgeClient(j) for j in config.judges]
        self.total_cost = 0.0
        self.decision_count = 0
        
        logger.info(f"Initialized committee with {len(self.judges)} judges:")
        for judge in config.judges:
            logger.info(f"  - {judge.model_id} ({judge.provider.value}) [priority={judge.priority}]")
    
    async def judge_behavior(self, prompt: str) -> CommitteeDecision:
        """
        Get behavior judgment from the committee.
        All judges vote in parallel, then results are aggregated.
        """
        # Execute all judges in parallel
        tasks = [judge.judge_behavior(prompt) for judge in self.judges]
        responses = await asyncio.gather(*tasks)
        
        # Filter out failed responses
        valid_responses = [r for r in responses if r.error is None]
        
        if not valid_responses:
            logger.error("All judges failed!")
            return self._create_fallback_decision(responses)
        
        # Aggregate votes
        decision = self._aggregate_votes(valid_responses)
        
        # Track costs
        self.total_cost += decision.total_cost
        self.decision_count += 1
        
        return decision
    
    def _aggregate_votes(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        """
        Aggregate judge responses using configured voting strategy.
        """
        if self.config.voting_strategy == "majority":
            return self._majority_vote(responses)
        elif self.config.voting_strategy == "weighted_majority":
            return self._weighted_majority_vote(responses)
        elif self.config.voting_strategy == "unanimous":
            return self._unanimous_vote(responses)
        else:
            return self._weighted_majority_vote(responses)
    
    def _majority_vote(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        """Simple majority voting (each judge gets 1 vote)."""
        votes_for = sum(1 for r in responses if r.adherent)
        votes_against = len(responses) - votes_for
        adherent = votes_for > votes_against
        confidence = votes_for / len(responses)
        
        # Select rationale from majority side
        majority_responses = [r for r in responses if r.adherent == adherent]
        rationale = majority_responses[0].rationale if majority_responses else ""
        
        total_cost = sum(r.cost for r in responses)
        total_latency = sum(r.latency_ms for r in responses)
        
        return CommitteeDecision(
            adherent=adherent,
            confidence=confidence,
            votes_for=votes_for,
            votes_against=votes_against,
            total_votes=len(responses),
            rationale=rationale,
            individual_responses=responses,
            total_cost=total_cost,
            total_latency_ms=total_latency
        )
    
    def _weighted_majority_vote(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        """Weighted voting based on judge priority and confidence."""
        # Get priorities from config
        priority_map = {
            j.model_id: j.priority 
            for j in self.config.judges
        }
        
        weighted_votes_for = 0.0
        weighted_votes_against = 0.0
        total_weight = 0.0
        
        for r in responses:
            priority = priority_map.get(r.model_id, 1)
            weight = priority * r.confidence
            total_weight += weight
            
            if r.adherent:
                weighted_votes_for += weight
            else:
                weighted_votes_against += weight
        
        adherent = weighted_votes_for > weighted_votes_against
        confidence = max(weighted_votes_for, weighted_votes_against) / total_weight if total_weight > 0 else 0.0
        
        # Select rationale from highest-priority judge on winning side
        winning_responses = [r for r in responses if r.adherent == adherent]
        if winning_responses:
            best_response = max(
                winning_responses,
                key=lambda r: priority_map.get(r.model_id, 1) * r.confidence
            )
            rationale = best_response.rationale
        else:
            rationale = ""
        
        votes_for = sum(1 for r in responses if r.adherent)
        votes_against = len(responses) - votes_for
        total_cost = sum(r.cost for r in responses)
        total_latency = sum(r.latency_ms for r in responses)
        
        return CommitteeDecision(
            adherent=adherent,
            confidence=confidence,
            votes_for=votes_for,
            votes_against=votes_against,
            total_votes=len(responses),
            rationale=rationale,
            individual_responses=responses,
            total_cost=total_cost,
            total_latency_ms=total_latency
        )
    
    def _unanimous_vote(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        """All judges must agree."""
        all_adherent = all(r.adherent for r in responses)
        votes_for = sum(1 for r in responses if r.adherent)
        votes_against = len(responses) - votes_for
        
        confidence = 1.0 if all_adherent else 0.0
        rationale = responses[0].rationale if responses else ""
        
        total_cost = sum(r.cost for r in responses)
        total_latency = sum(r.latency_ms for r in responses)
        
        return CommitteeDecision(
            adherent=all_adherent,
            confidence=confidence,
            votes_for=votes_for,
            votes_against=votes_against,
            total_votes=len(responses),
            rationale=rationale,
            individual_responses=responses,
            total_cost=total_cost,
            total_latency_ms=total_latency
        )
    
    def _create_fallback_decision(self, responses: List[JudgeResponse]) -> CommitteeDecision:
        """Create a fallback decision when all judges fail."""
        return CommitteeDecision(
            adherent=False,
            confidence=0.0,
            votes_for=0,
            votes_against=len(responses),
            total_votes=len(responses),
            rationale="All judges failed",
            individual_responses=responses,
            total_cost=0.0,
            total_latency_ms=sum(r.latency_ms for r in responses)
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for the committee."""
        return {
            "total_cost_usd": self.total_cost,
            "decisions_made": self.decision_count,
            "avg_cost_per_decision": self.total_cost / max(1, self.decision_count),
            "per_judge_costs": {
                judge.config.model_id: {
                    "total_cost": judge.total_cost,
                    "requests": judge.request_count,
                    "avg_cost": judge.total_cost / max(1, judge.request_count)
                }
                for judge in self.judges
            }
        }
    
    async def close(self):
        """Close all judge clients."""
        for judge in self.judges:
            await judge.close()
