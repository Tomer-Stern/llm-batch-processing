"""
Rank career summaries on social impact (1-5 scale) using batch APIs.
Uses helper classes for cleaner code.

Usage:
    python rank_social_impact.py input.csv --compare
    python rank_social_impact.py input.csv --anthropic-batch
    python rank_social_impact.py input.csv --openai-batch
"""

import csv
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

from anthropic_batch_prompter import AnthropicBatchPrompter
from openai_batch_prompter import OpenAIBatchPrompter


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

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


# ============================================================================
# COMPARISON
# ============================================================================

def compare_results(anthropic_scores, openai_scores, ids, summaries):
    """Compare scores from both providers with enhanced statistics."""
    
    print("\n" + "="*70)
    print("COMPARISON: Anthropic vs OpenAI (Both Batch)")
    print("="*70)
    
    comparison_rows = []
    
    for summary_id, summary in zip(ids, summaries):
        anthropic_score = anthropic_scores.get(str(summary_id))
        openai_score = openai_scores.get(str(summary_id))
        
        agreement = "✓" if anthropic_score == openai_score else "✗"
        diff = abs(anthropic_score - openai_score) if anthropic_score and openai_score else None
        
        comparison_rows.append({
            "id": summary_id,
            "summary": summary,
            "anthropic_score": anthropic_score,
            "openai_score": openai_score,
            "agreement": agreement,
            "difference": diff
        })
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Only compute statistics for rows where both have scores
    valid_mask = (comparison_df["anthropic_score"].notna()) & (comparison_df["openai_score"].notna())
    valid_df = comparison_df[valid_mask]
    
    if len(valid_df) > 0:
        anthro_scores = valid_df["anthropic_score"].astype(int)
        openai_scores = valid_df["openai_score"].astype(int)
        
        # Core reliability metrics
        kappa_unweighted = cohen_kappa_score(anthro_scores, openai_scores)
        kappa_weighted = cohen_kappa_score(anthro_scores, openai_scores, weights='linear')
        
        # Correlation (better for ordinal scales)
        spearman_corr, spearman_p = spearmanr(anthro_scores, openai_scores)
        
        # Agreement metrics
        exact_agreement = (anthro_scores == openai_scores).sum() / len(valid_df)
        within_1 = (valid_df["difference"] <= 1).sum() / len(valid_df)
        mean_diff = valid_df["difference"].mean()
        
        # Distribution statistics
        anthro_mean = anthro_scores.mean()
        anthro_std = anthro_scores.std()
        openai_mean = openai_scores.mean()
        openai_std = openai_scores.std()
        
        # Print results
        print(f"\nAgreement Metrics:")
        print(f"  Exact agreement: {exact_agreement*100:.1f}% ({(anthro_scores == openai_scores).sum()}/{len(valid_df)})")
        print(f"  Within 1 point: {within_1*100:.1f}%")
        print(f"  Mean absolute difference: {mean_diff:.3f}")
        
        print(f"\nReliability Coefficients:")
        print(f"  Cohen's κ (unweighted): {kappa_unweighted:.3f}")
        print(f"  Cohen's κ (weighted): {kappa_weighted:.3f}", end="")
        if kappa_weighted > 0.80:
            print(" [Excellent]")
        elif kappa_weighted > 0.60:
            print(" [Substantial]")
        elif kappa_weighted > 0.40:
            print(" [Moderate]")
        else:
            print(" [Fair/Poor]")
        
        print(f"\nCorrelation Analysis:")
        print(f"  Spearman's ρ: {spearman_corr:.3f} (p={spearman_p:.4f})")
        
        print(f"\nScore Distributions:")
        print(f"  Anthropic: Mean={anthro_mean:.2f}, SD={anthro_std:.2f}")
        print(f"  OpenAI:    Mean={openai_mean:.2f}, SD={openai_std:.2f}")
        
        # Distribution by score
        print(f"\n  Score frequencies:")
        for score in range(1, 6):
            anthro_count = (anthro_scores == score).sum()
            openai_count = (openai_scores == score).sum()
            print(f"    {score}: Anthropic={anthro_count:3d}, OpenAI={openai_count:3d}")
        
        # Show disagreements
        disagreements = valid_df[valid_df["agreement"] == "✗"]
        if len(disagreements) > 0:
            print(f"\n{len(disagreements)} disagreements found:")
            for _, row in disagreements.head(10).iterrows():
                summary_preview = row['summary'][:60] + "..." if len(row['summary']) > 60 else row['summary']
                print(f"  {row['id']:12} | Anthropic: {row['anthropic_score']} | OpenAI: {row['openai_score']} | {summary_preview}")
            if len(disagreements) > 10:
                print(f"  ... and {len(disagreements) - 10} more disagreements")
    
    print("="*70)
    
    return comparison_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rank career summaries using batch APIs (both 50% discount)",
        epilog="""
Examples:
  # Validate with both providers (recommended first step)
  python rank_social_impact.py example_input.csv --compare
  
  # Production run with Anthropic only
  python rank_social_impact.py data.csv --anthropic-batch --output results.csv
  
  # Quick validation with OpenAI (faster, cheaper)
  python rank_social_impact.py sample.csv --openai-batch

For more info: see README.md
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", help="Input CSV with columns: id, summary")
    parser.add_argument("--output", default="output_social_impact.csv", help="Output CSV file")
    parser.add_argument("--compare", action="store_true", help="Run both and compare")
    parser.add_argument("--anthropic-batch", action="store_true", help="Only run Anthropic batch")
    parser.add_argument("--openai-batch", action="store_true", help="Only run OpenAI batch")
    parser.add_argument("--no-confirm", action="store_true", help="Skip cost confirmation")
    
    args = parser.parse_args()
    
    # Read input CSV
    summaries = []
    ids = []
    with open(args.input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            summaries.append(row["summary"])
    
    print(f"Loaded {len(summaries)} summaries from {args.input_file}\n")
    
    # Determine what to run
    run_anthropic = args.compare or args.anthropic_batch or (not args.openai_batch)
    run_openai = args.compare or args.openai_batch or (not args.anthropic_batch)
    
    confirm_cost = not args.no_confirm
    results = {}
    
    # Run Anthropic batch
    if run_anthropic:
        print("="*70)
        print("ANTHROPIC BATCH API (50% discount)")
        print("="*70)
        prompter = AnthropicBatchPrompter(api_key=None)  # Reads from ANTHROPIC_API_KEY env var
        anthropic_scores = prompter.batch_prompt(summaries, ids, SYSTEM_PROMPT, confirm_cost=confirm_cost)
        if anthropic_scores:  # Only store if not aborted
            results["anthropic"] = anthropic_scores
    
    # Run OpenAI batch
    if run_openai:
        print("\n" + "="*70)
        print("OPENAI BATCH API (50% discount)")
        print("="*70)
        prompter = OpenAIBatchPrompter(api_key=None)  # Reads from OPENAI_API_KEY env var
        openai_scores = prompter.batch_prompt(summaries, ids, SYSTEM_PROMPT, confirm_cost=confirm_cost)
        if openai_scores:  # Only store if not aborted
            results["openai"] = openai_scores
    
    # Check if we have results to save
    if not results:
        print("\nNo results to save (all batches were aborted or failed).")
        return
    
    # Save results
    if args.compare and len(results) == 2:
        comparison_df = compare_results(
            results["anthropic"],
            results["openai"],
            ids,
            summaries
        )
        comparison_df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to {args.output}")
    
    else:
        # Single provider output
        if "anthropic" in results:
            provider = "anthropic"
            scores = results["anthropic"]
        else:
            provider = "openai"
            scores = results["openai"]
        
        output_rows = []
        for summary_id, summary in zip(ids, summaries):
            output_rows.append({
                "id": summary_id,
                "summary": summary,
                f"{provider}_score": scores.get(str(summary_id))
            })
        
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
