"""
Anthropic Batch API helper with enhanced error handling and logging.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional

try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Install anthropic: pip install anthropic")

from exceptions import (
    APIKeyError, BatchSubmissionError, BatchProcessingTimeoutError,
    InvalidInputError, CostEstimationError, ResultParsingError, BatchProcessingError
)


class AnthropicBatchPrompter:
    """Helper for Anthropic batch API (50% discount)."""
    
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    
    # Batch pricing (50% of standard rates) - per million tokens
    PRICING = {
        "claude-haiku-4-5-20251001": {"input": 0.50, "output": 2.50},
        "claude-sonnet-4-5-20250929": {"input": 1.50, "output": 7.50},
        "claude-opus-4-20250514": {"input": 7.50, "output": 37.50},
    }
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 temperature: float = 0.0, max_tokens: int = 10, 
                 polling_interval: int = 10, max_retries: int = 3):
        import os
        self.logger = logging.getLogger(f"{__name__}.AnthropicBatchPrompter")
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise APIKeyError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            self.client = Anthropic(api_key=api_key)
        except Exception as e:
            raise APIKeyError(f"Failed to initialize Anthropic client: {e}")
        
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.polling_interval = polling_interval
        self.max_retries = max_retries
        
        self.logger.info(f"Initialized AnthropicBatchPrompter with model: {self.model}")
    
    def estimate_cost(self, summaries: List[str], system_prompt: str) -> Dict[str, float]:
        """
        Estimate cost for batch processing.
        
        Returns:
            Dict with 'input_cost', 'output_cost', 'total_cost' in USD
        """
        try:
            if not summaries:
                raise InvalidInputError("Cannot estimate cost for empty summaries list")
            
            if not system_prompt:
                raise InvalidInputError("System prompt cannot be empty")
            
            # Rough token estimation (1 token ≈ 4 chars for English)
            total_input_chars = sum(len(s) for s in summaries) + len(system_prompt) * len(summaries)
            estimated_input_tokens = total_input_chars / 4
            estimated_output_tokens = self.max_tokens * len(summaries)
            
            # Get pricing for model
            pricing = self.PRICING.get(self.model, self.PRICING[self.DEFAULT_MODEL])
            
            input_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]
            output_cost = (estimated_output_tokens / 1_000_000) * pricing["output"]
            
            cost_estimate = {
                "input_tokens": int(estimated_input_tokens),
                "output_tokens": int(estimated_output_tokens),
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            }
            
            self.logger.info(f"Cost estimate: ${cost_estimate['total_cost']:.4f} for {len(summaries)} requests")
            return cost_estimate
            
        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            raise CostEstimationError(f"Failed to estimate cost: {e}")
    
    def batch_prompt(self, summaries: List[str], ids: List[str], system_prompt: str,
                     confirm_cost: bool = True) -> Dict[str, Any]:
        """
        Submit batch to Anthropic, poll for completion, return scores.
        
        Args:
            summaries: List of text summaries
            ids: List of IDs (must match summaries)
            system_prompt: System prompt for classification
            confirm_cost: If True, show cost estimate and ask for confirmation
            
        Returns:
            Dict mapping id -> score (int or None)
        """
        try:
            # Validate inputs
            if not summaries or not ids:
                raise InvalidInputError("Summaries and IDs cannot be empty")
            
            if len(summaries) != len(ids):
                raise InvalidInputError("Number of summaries must match number of IDs")
            
            if not system_prompt:
                raise InvalidInputError("System prompt cannot be empty")
            
            self.logger.info(f"Starting batch processing for {len(summaries)} requests")
            
            # Cost estimation and confirmation
            if confirm_cost:
                cost_est = self.estimate_cost(summaries, system_prompt)
                print(f"\n{'='*70}")
                print(f"COST ESTIMATE (Anthropic Batch API)")
                print(f"{'='*70}")
                print(f"Model: {self.model}")
                print(f"Requests: {len(summaries)}")
                print(f"Estimated input tokens: {cost_est['input_tokens']:,}")
                print(f"Estimated output tokens: {cost_est['output_tokens']:,}")
                print(f"Estimated cost: ${cost_est['total_cost']:.4f}")
                print(f"  (Input: ${cost_est['input_cost']:.4f}, Output: ${cost_est['output_cost']:.4f})")
                print(f"{'='*70}")
                
                response = input("\nProceed with batch submission? [y/N]: ").strip().lower()
                if response not in ['y', 'yes']:
                    self.logger.info("Batch submission aborted by user")
                    print("Batch submission aborted.")
                    return {}
            
            print(f"\nPreparing {len(summaries)} requests for Anthropic batch...")
            
            # Build batch requests
            requests = []
            for summary_id, summary in zip(ids, summaries):
                requests.append({
                    "custom_id": str(summary_id),
                    "params": {
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "system": system_prompt,
                        "messages": [
                            {"role": "user", "content": summary}
                        ]
                    }
                })
            
            print(f"Submitting batch to Anthropic...")
            
            # Submit batch with retry logic
            batch_id = None
            for attempt in range(self.max_retries):
                try:
                    batch = self.client.beta.messages.batches.create(requests=requests)
                    batch_id = batch.id
                    self.logger.info(f"Batch submitted successfully: {batch_id}")
                    break
                except Exception as e:
                    self.logger.warning(f"Batch submission attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        raise BatchSubmissionError(f"Failed to submit batch after {self.max_retries} attempts: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            print(f"Batch submitted: {batch_id}")
            print("Polling for completion...")
            
            results = {}
            poll_count = 0
            max_polls = 360  # 1 hour max (360 * 10 seconds)
            
            # Poll for completion
            while poll_count < max_polls:
                try:
                    batch = self.client.beta.messages.batches.retrieve(batch_id)
                    poll_count += 1
                    
                    processing = batch.request_counts.processing
                    succeeded = batch.request_counts.succeeded
                    errored = batch.request_counts.errored
                    total = len(requests)
                    
                    print(f"  [{poll_count}] Status: {batch.processing_status} | {succeeded}/{total} completed | {processing} processing | {errored} errors")
                    
                    if batch.processing_status == "ended":
                        print("✓ Batch completed!")
                        self.logger.info(f"Batch {batch_id} completed successfully")
                        
                        # Process results
                        for result in self.client.beta.messages.batches.results(batch_id):
                            custom_id = result.custom_id
                            if result.result.type == "succeeded":
                                try:
                                    score = int(result.result.message.content[0].text.strip())
                                    results[custom_id] = score
                                except Exception as e:
                                    self.logger.warning(f"Parse error for {custom_id}: {e}")
                                    print(f"  Parse error for {custom_id}: {e}")
                                    results[custom_id] = None
                            else:
                                error_type = result.result.type
                                error_msg = getattr(result.result, 'error', {})
                                if hasattr(error_msg, 'message'):
                                    error_details = error_msg.message
                                else:
                                    error_details = str(error_msg)
                                self.logger.warning(f"Request failed for {custom_id}: {error_type} - {error_details}")
                                print(f"  Request failed for {custom_id}: {error_type} - {error_details}")
                                results[custom_id] = None
                        
                        break
                    
                    elif batch.processing_status in ["failed", "cancelled"]:
                        raise BatchProcessingError(f"Batch {batch_id} {batch.processing_status}")
                    
                    time.sleep(self.polling_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error during polling: {e}")
                    if poll_count == max_polls - 1:
                        raise BatchProcessingTimeoutError(f"Batch processing timed out after {max_polls} polls")
                    time.sleep(self.polling_interval)
            
            if poll_count >= max_polls:
                raise BatchProcessingTimeoutError(f"Batch processing timed out after {max_polls} polls")
            
            self.logger.info(f"Batch processing completed. {len(results)} results processed")
            return results
            
        except (APIKeyError, InvalidInputError, CostEstimationError) as e:
            self.logger.error(f"Configuration error: {e}")
            raise
        except (BatchSubmissionError, BatchProcessingError, BatchProcessingTimeoutError) as e:
            self.logger.error(f"Batch processing error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in batch_prompt: {e}")
            raise BatchProcessingError(f"Unexpected error: {e}")
