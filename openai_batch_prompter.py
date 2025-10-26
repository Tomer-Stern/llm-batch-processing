"""
OpenAI Batch API helper.
"""

import json
import time
import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Install openai: pip install openai")


class OpenAIBatchPrompter:
    """Helper for OpenAI batch API (50% discount)."""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # Batch pricing (50% of standard rates) - per million tokens
    PRICING = {
        "gpt-4o-mini": {"input": 0.075, "output": 0.30},
        "gpt-4o": {"input": 1.25, "output": 5.00},
        "gpt-4-turbo": {"input": 5.00, "output": 15.00},
    }
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 temperature: float = 0.0, max_tokens: int = 10):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
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
        Submit batch to OpenAI, poll for completion, return scores.
        
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
            print(f"COST ESTIMATE (OpenAI Batch API)")
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
        
        print(f"\nPreparing {len(summaries)} requests for OpenAI batch...")
        
        # Create batch requests
        requests = []
        for summary_id, summary in zip(ids, summaries):
            requests.append({
                "custom_id": str(summary_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": summary}
                    ]
                }
            })
        
        # Write batch file with unique name
        batch_file_path = f"openai_batch_{int(time.time())}.jsonl"
        
        try:
            with open(batch_file_path, "w") as f:
                for req in requests:
                    f.write(json.dumps(req) + "\n")
            
            print(f"Batch file created: {batch_file_path}")
            
            # Upload batch file
            with open(batch_file_path, "rb") as f:
                batch_file = self.client.files.create(file=f, purpose="batch")
            
            print(f"File uploaded: {batch_file.id}")
            
            # Submit batch
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            print(f"Batch submitted: {batch.id}")
            print("Polling for completion...")
            
            results = {}
            poll_count = 0
            
            # Poll until complete
            while True:
                batch = self.client.batches.retrieve(batch.id)
                poll_count += 1
                completed = batch.request_counts.completed
                total = batch.request_counts.total
                
                print(f"  [{poll_count}] Status: {batch.status} | {completed}/{total} completed")
                
                if batch.status == "completed":
                    print("✓ Batch completed!")
                    
                    # Get results
                    file_response = self.client.files.content(batch.output_file_id)
                    for line in file_response.text.strip().split("\n"):
                        if not line.strip():
                            continue
                        result = json.loads(line)
                        custom_id = result["custom_id"]
                        if result["response"]["status_code"] == 200:
                            try:
                                content = result["response"]["body"]["choices"][0]["message"]["content"]
                                score = int(content.strip())
                                results[custom_id] = score
                            except Exception as e:
                                print(f"  Parse error for {custom_id}: {e}")
                                results[custom_id] = None
                        else:
                            print(f"  Request failed for {custom_id}: {result['response']['status_code']}")
                            results[custom_id] = None
                    
                    break
                
                elif batch.status in ("failed", "expired", "cancelled"):
                    raise Exception(f"Batch {batch.status}")
                
                time.sleep(10)
            
            return results
        
        finally:
            # Clean up temporary file
            if os.path.exists(batch_file_path):
                try:
                    os.remove(batch_file_path)
                    print(f"Cleaned up temporary file: {batch_file_path}")
                except Exception as e:
                    print(f"Warning: Could not delete {batch_file_path}: {e}")
