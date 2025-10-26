"""
Simple batch processing script - just run it directly!
Uses example_input.csv by default.
"""

import csv
from anthropic_batch_prompter import AnthropicBatchPrompter
from openai_batch_prompter import OpenAIBatchPrompter

# ============================================================================
# CONFIGURATION - Edit these if needed
# ============================================================================
INPUT_FILE = "example_input.csv"
OUTPUT_FILE = "output_social_impact.csv"
RUN_COMPARISON = True  # Set to False to run only one provider

SYSTEM_PROMPT = """You are analyzing career summaries for social science research.

Rate each career's social impact orientation on a validated 5-point scale:

1 = PROFIT-MAXIMIZING: Exclusively focused on financial returns, shareholder value, 
   or commercial optimization with no mention of social considerations

2 = COMMERCIALLY-ORIENTED: Primarily profit-focused with minimal or token references 
   to social responsibility (e.g., occasional CSR initiatives)

3 = MIXED ORIENTATION: Substantial evidence of both commercial objectives and 
   meaningful social/environmental commitment (roughly balanced)

4 = SOCIAL-IMPACT FOCUSED: Career primarily oriented toward social/environmental 
   benefit, with commercial work as secondary or supporting role

5 = MISSION-DRIVEN: Exclusive or near-exclusive dedication to social, environmental, 
   or public benefit work, often with explicit rejection of commercial alternatives

Criteria for classification:
- Stated motivations and values
- Organizational types (nonprofit, B-corp, for-profit)
- Career transitions and their direction
- Trade-offs accepted (e.g., salary, prestige)

Respond with ONLY a single integer (1-5). No explanation."""


def main():
    # Read input CSV
    print(f"Loading data from {INPUT_FILE}...")
    summaries = []
    ids = []
    with open(INPUT_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            summaries.append(row["summary"])
    
    print(f"Loaded {len(summaries)} summaries\n")
    
    results = {}
    
    # Run Anthropic batch
    if RUN_COMPARISON or True:
        print("="*70)
        print("ANTHROPIC BATCH API")
        print("="*70)
        prompter = AnthropicBatchPrompter()
        anthropic_scores = prompter.batch_prompt(summaries, ids, SYSTEM_PROMPT, confirm_cost=True)
        if anthropic_scores:
            results["anthropic"] = anthropic_scores
    
    # Run OpenAI batch
    if RUN_COMPARISON:
        print("\n" + "="*70)
        print("OPENAI BATCH API")
        print("="*70)
        prompter = OpenAIBatchPrompter()
        openai_scores = prompter.batch_prompt(summaries, ids, SYSTEM_PROMPT, confirm_cost=True)
        if openai_scores:
            results["openai"] = openai_scores
    
    # Save results
    if results:
        import pandas as pd
        
        output_rows = []
        for summary_id, summary in zip(ids, summaries):
            row = {
                "id": summary_id,
                "summary": summary
            }
            if "anthropic" in results:
                row["anthropic_score"] = results["anthropic"].get(str(summary_id))
            if "openai" in results:
                row["openai_score"] = results["openai"].get(str(summary_id))
            output_rows.append(row)
        
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Results saved to {OUTPUT_FILE}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
