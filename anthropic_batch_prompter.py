"""
Anthropic Batch API helper.
"""

import json
import time
from typing import List, Dict, Any, Optional

try:
    from anthropic import Anthropic
except ImportError:
    raise ImportError("Install anthropic: pip install anthropic")


class AnthropicBatchPrompter:
    """Helper for Anthropic batch API (50% discount)."""
    
    DEFAULT_MODEL = "claude-haiku-4-5-20250929"
    
    # Batch pricing (50% of standard rates) - per million tokens
    PRICING = {
        "claude-haiku-4-5-20250929": {"input": 0.40, "output": 2.00},
        "claude-sonnet-4-5-20250929": {"input": 1.50, "output": 7.50},
        "claude-opus-4-20250514": {"input": 7.50, "output": 37.50},
    }
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 temperature: float = 0.0, max_tokens: int = 10):
        import os
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def estimate_cost(self, summaries: List[str], system_prompt: str) -> Dict[str, float]:
        """
        Estimate cost for batch processing.
        
        Returns:
            Dict with 'input_cost', 'output_cost', 'total_cost' in USD
        """
        # Rough token estimation (1 token ≈ 4 chars for English)
        total_input_chars = sum(len(s) for s in summaries) + len(system_prompt) * len(summaries)
        estimated_input_tokens = total_input_chars / 4
        estimated_output_tokens = self.max_tokens * len(summaries)
        
        # Get pricing for model
        pricing = self.PRICING.get(self.model, self.PRICING[self.DEFAULT_MODEL])
        
        input_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (estimated_output_tokens / 1_000_000) * pricing["output"]
        
        return {
            "input_tokens": int(estimated_input_tokens),
            "output_tokens": int(estimated_output_tokens),
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
    
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
        
        # Submit batch
        batch = self.client.beta.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"Batch submitted: {batch_id}")
        print("Polling for completion...")
        
        results = {}
        poll_count = 0
        
        # Poll for completion
        while True:
            batch = self.client.beta.messages.batches.retrieve(batch_id)
            poll_count += 1
            
            processing = batch.request_counts.processing
            succeeded = batch.request_counts.succeeded
            errored = batch.request_counts.errored
            total = len(requests)
            
            print(f"  [{poll_count}] Status: {batch.processing_status} | {succeeded}/{total} completed | {processing} processing | {errored} errors")
            
            if batch.processing_status == "ended":
                print("✓ Batch completed!")
                
                # Process results
                for result in self.client.beta.messages.batches.results(batch_id):
                    custom_id = result.custom_id
                    if result.result.type == "succeeded":
                        try:
                            score = int(result.result.message.content[0].text.strip())
                            results[custom_id] = score
                        except Exception as e:
                            print(f"  Parse error for {custom_id}: {e}")
                            results[custom_id] = None
                    else:
                        print(f"  Request failed for {custom_id}: {result.result.type}")
                        results[custom_id] = None
                
                break
            
            time.sleep(10)
        
        return results
