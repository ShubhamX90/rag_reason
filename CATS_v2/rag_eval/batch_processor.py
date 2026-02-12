# rag_eval/batch_processor.py
# -*- coding: utf-8 -*-
"""
Anthropic Batch API Processing for Cost Optimization
----------------------------------------------------
Implements batch processing using Anthropic's Message Batches API
to reduce costs by up to 50% for large-scale evaluations.

Features:
  • Automatic batching of requests
  • Async polling for batch completion
  • Cost tracking and optimization
  • Fallback to standard API if needed

Authors: Enhanced by Claude AI
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import anthropic
from .logging_config import logger


@dataclass
class BatchRequest:
    """Single request in a batch."""
    custom_id: str
    model: str
    max_tokens: int
    messages: List[Dict[str, Any]]
    temperature: float = 0.0


@dataclass
class BatchResult:
    """Result from a batch request."""
    custom_id: str
    response_text: str
    input_tokens: int
    output_tokens: int
    error: Optional[str] = None


class AnthropicBatchProcessor:
    """
    Processor for Anthropic's Message Batches API.
    
    Enables cost-effective batch processing with up to 50% cost reduction
    for large-scale evaluations.
    """
    
    def __init__(self, api_key: str, max_batch_size: int = 10000):
        """
        Initialize batch processor.
        
        Args:
            api_key: Anthropic API key
            max_batch_size: Maximum requests per batch (API limit is 10,000)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_batch_size = min(max_batch_size, 10000)
        self.total_cost = 0.0
        
    async def process_batch(
        self,
        requests: List[BatchRequest],
        poll_interval: int = 60,
        timeout: int = 3600
    ) -> List[BatchResult]:
        """
        Process a batch of requests using Anthropic's Batch API.
        
        Args:
            requests: List of batch requests
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
            
        Returns:
            List of batch results
        """
        if not requests:
            return []
        
        logger.info(f"Processing {len(requests)} requests via Batch API")
        
        # Create batch
        batch_id = await self._create_batch(requests)
        if not batch_id:
            logger.error("Failed to create batch, falling back to standard API")
            return await self._fallback_process(requests)
        
        # Poll for completion
        results = await self._poll_batch(batch_id, poll_interval, timeout)
        
        return results
    
    async def _create_batch(self, requests: List[BatchRequest]) -> Optional[str]:
        """Create a batch and return batch ID."""
        try:
            # Format requests for Batch API
            batch_requests = []
            for req in requests:
                batch_requests.append({
                    "custom_id": req.custom_id,
                    "params": {
                        "model": req.model,
                        "max_tokens": req.max_tokens,
                        "messages": req.messages,
                        "temperature": req.temperature
                    }
                })
            
            # Create batch
            batch = self.client.messages.batches.create(
                requests=batch_requests
            )
            
            logger.info(f"Created batch {batch.id} with {len(requests)} requests")
            return batch.id
            
        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            return None
    
    async def _poll_batch(
        self,
        batch_id: str,
        poll_interval: int,
        timeout: int
    ) -> List[BatchResult]:
        """Poll batch status until complete."""
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                logger.error(f"Batch {batch_id} timed out after {timeout}s")
                return []
            
            try:
                # Get batch status
                batch = self.client.messages.batches.retrieve(batch_id)
                
                status = batch.processing_status
                logger.info(f"Batch {batch_id} status: {status}")
                
                if status == "ended":
                    # Batch complete, retrieve results
                    return await self._retrieve_results(batch_id)
                elif status in ["canceling", "canceled", "expired"]:
                    logger.error(f"Batch {batch_id} failed with status: {status}")
                    return []
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error polling batch {batch_id}: {e}")
                await asyncio.sleep(poll_interval)
    
    async def _retrieve_results(self, batch_id: str) -> List[BatchResult]:
        """Retrieve results from completed batch."""
        try:
            # Get all results
            results = []
            
            # Iterate through results
            for result in self.client.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    message = result.result.message
                    results.append(BatchResult(
                        custom_id=result.custom_id,
                        response_text=message.content[0].text,
                        input_tokens=message.usage.input_tokens,
                        output_tokens=message.usage.output_tokens
                    ))
                else:
                    # Error in this request
                    error = result.result.error
                    results.append(BatchResult(
                        custom_id=result.custom_id,
                        response_text="",
                        input_tokens=0,
                        output_tokens=0,
                        error=f"{error.type}: {error.message}"
                    ))
            
            logger.info(f"Retrieved {len(results)} results from batch {batch_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving batch results: {e}")
            return []
    
    async def _fallback_process(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Fallback to standard API if batch fails."""
        logger.warning("Using fallback standard API processing")
        results = []
        
        for req in requests:
            try:
                message = await anthropic.AsyncAnthropic(
                    api_key=self.client.api_key
                ).messages.create(
                    model=req.model,
                    max_tokens=req.max_tokens,
                    messages=req.messages,
                    temperature=req.temperature
                )
                
                results.append(BatchResult(
                    custom_id=req.custom_id,
                    response_text=message.content[0].text,
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens
                ))
                
            except Exception as e:
                results.append(BatchResult(
                    custom_id=req.custom_id,
                    response_text="",
                    input_tokens=0,
                    output_tokens=0,
                    error=str(e)
                ))
        
        return results
