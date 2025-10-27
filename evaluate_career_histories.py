"""
Simple batch processing script - just run it directly!
Uses example_input.csv by default.
"""

import csv
from anthropic_batch_prompter import AnthropicBatchPrompter
from openai_batch_prompter import OpenAIBatchPrompter

# ============================================================================
# CONFIGURATION - Edit these as needed
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
            
            # Add agreement and difference columns if both scores exist
            if "anthropic" in results and "openai" in results:
                anthro_score = results["anthropic"].get(str(summary_id))
                openai_score = results["openai"].get(str(summary_id))
                if anthro_score is not None and openai_score is not None:
                    row["agreement"] = "✓" if anthro_score == openai_score else "✗"
                    row["difference"] = abs(float(anthro_score) - float(openai_score))
                else:
                    row["agreement"] = None
                    row["difference"] = None
            
            output_rows.append(row)
        
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✓ Results saved to {OUTPUT_FILE}")
        
        # Run comparison analysis if both providers were used
        if "anthropic" in results and "openai" in results:
            compare_results(output_df)
    else:
        print("\nNo results to save.")


def compare_results(df):
    """Compare Anthropic and OpenAI scoring results."""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: ANTHROPIC vs OPENAI")
    print("="*70)
    
    # Filter to rows with both scores
    comparison_df = df.dropna(subset=["anthropic_score", "openai_score"])
    n = len(comparison_df)
    
    if n == 0:
        print("No overlapping scores to compare.")
        return
    
    anthropic_scores = comparison_df["anthropic_score"].astype(float)
    openai_scores = comparison_df["openai_score"].astype(float)
    
    # ========================================================================
    # DESCRIPTIVE STATISTICS
    # ========================================================================
    print(f"\nSample Size: {n} summaries with both scores\n")
    
    print("Descriptive Statistics:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Anthropic':<15} {'OpenAI':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {anthropic_scores.mean():<15.2f} {openai_scores.mean():<15.2f}")
    print(f"{'Median':<20} {anthropic_scores.median():<15.1f} {openai_scores.median():<15.1f}")
    print(f"{'Std Dev':<20} {anthropic_scores.std():<15.2f} {openai_scores.std():<15.2f}")
    print(f"{'Min':<20} {anthropic_scores.min():<15.0f} {openai_scores.min():<15.0f}")
    print(f"{'Max':<20} {anthropic_scores.max():<15.0f} {openai_scores.max():<15.0f}")
    
    # ========================================================================
    # CORRELATION ANALYSIS
    # ========================================================================
    print("\n\nCorrelation Analysis:")
    print("-" * 50)
    
    pearson_r, pearson_p = stats.pearsonr(anthropic_scores, openai_scores)
    spearman_r, spearman_p = stats.spearmanr(anthropic_scores, openai_scores)
    
    print(f"Pearson correlation:  r = {pearson_r:.3f} (p = {pearson_p:.4f})")
    print(f"Spearman correlation: ρ = {spearman_r:.3f} (p = {spearman_p:.4f})")
    
    # ========================================================================
    # RELIABILITY COEFFICIENTS (COHEN'S KAPPA)
    # ========================================================================
    print("\n\nReliability Coefficients:")
    print("-" * 50)
    
    kappa_unweighted = cohen_kappa_score(anthropic_scores, openai_scores, weights=None)
    kappa_weighted = cohen_kappa_score(anthropic_scores, openai_scores, weights='quadratic')
    
    # Interpret weighted kappa
    if kappa_weighted > 0.80:
        interpretation = "Excellent"
    elif kappa_weighted > 0.60:
        interpretation = "Substantial"
    elif kappa_weighted > 0.40:
        interpretation = "Moderate"
    elif kappa_weighted > 0.20:
        interpretation = "Fair"
    else:
        interpretation = "Poor"
    
    print(f"Cohen's κ (unweighted): {kappa_unweighted:.3f}")
    print(f"Cohen's κ (weighted):   {kappa_weighted:.3f} [{interpretation}]")
    
    # ========================================================================
    # AGREEMENT METRICS
    # ========================================================================
    print("\n\nAgreement Metrics:")
    print("-" * 50)
    
    differences = anthropic_scores - openai_scores
    exact_agreement = (differences == 0).sum()
    within_1 = (abs(differences) <= 1).sum()
    
    print(f"Exact agreement:        {exact_agreement:>4} ({100*exact_agreement/n:.1f}%)")
    print(f"Within 1 point:         {within_1:>4} ({100*within_1/n:.1f}%)")
    print(f"Mean absolute diff:     {abs(differences).mean():.2f}")
    print(f"Mean difference:        {differences.mean():.2f} (Anthropic - OpenAI)")
    
    # Identify ALL disagreements for manual review
    disagreements = comparison_df[differences != 0].copy()
    disagreements['difference'] = differences[differences != 0]
    disagreements['abs_difference'] = abs(differences[differences != 0])
    disagreements = disagreements.sort_values('abs_difference', ascending=False)
    
    num_disagreements = len(disagreements)
    print(f"\n{num_disagreements} disagreements found:")
    if num_disagreements > 0:
        print("-" * 70)
        for idx, row in disagreements.iterrows():
            summary_preview = row['summary'][:60] + "..." if len(row['summary']) > 60 else row['summary']
            print(f"  {row['id']:<12} | Anthropic: {int(row['anthropic_score'])} | "
                  f"OpenAI: {int(row['openai_score'])} | {summary_preview}")
        print("-" * 70)
    
    # ========================================================================
    # DISTRIBUTION COMPARISON
    # ========================================================================
    print("\n\nScore Distributions:")
    print("-" * 50)
    print(f"{'Score':<10} {'Anthropic':<15} {'OpenAI':<15}")
    print("-" * 50)
    
    for score in range(1, 6):
        anthro_count = (anthropic_scores == score).sum()
        openai_count = (openai_scores == score).sum()
        print(f"{score:<10} {anthro_count:<7} ({100*anthro_count/n:>5.1f}%)  "
              f"{openai_count:<7} ({100*openai_count/n:>5.1f}%)")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Anthropic vs OpenAI Score Comparison', fontsize=14, fontweight='bold')
    
    # 1. Side-by-side histograms
    ax1 = axes[0, 0]
    bins = np.arange(0.5, 6.5, 1)
    ax1.hist([anthropic_scores, openai_scores], bins=bins, label=['Anthropic', 'OpenAI'], 
             alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Social Impact Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Scores')
    ax1.legend()
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Scatter plot with diagonal
    ax2 = axes[0, 1]
    ax2.scatter(anthropic_scores, openai_scores, alpha=0.5, s=50)
    ax2.plot([1, 5], [1, 5], 'r--', label='Perfect agreement', linewidth=2)
    ax2.set_xlabel('Anthropic Score')
    ax2.set_ylabel('OpenAI Score')
    ax2.set_title(f'Score Correlation (r = {pearson_r:.3f})')
    ax2.set_xlim(0.5, 5.5)
    ax2.set_ylim(0.5, 5.5)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Difference distribution
    ax3 = axes[1, 0]
    ax3.hist(differences, bins=np.arange(-4.5, 5.5, 1), edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='No difference')
    ax3.set_xlabel('Score Difference (Anthropic - OpenAI)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Differences')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Agreement heatmap
    ax4 = axes[1, 1]
    confusion = np.zeros((5, 5))
    for a, o in zip(anthropic_scores, openai_scores):
        confusion[int(a)-1, int(o)-1] += 1
    
    im = ax4.imshow(confusion, cmap='YlOrRd', aspect='auto')
    ax4.set_xlabel('OpenAI Score')
    ax4.set_ylabel('Anthropic Score')
    ax4.set_title('Agreement Matrix')
    ax4.set_xticks([0, 1, 2, 3, 4])
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_xticklabels([1, 2, 3, 4, 5])
    ax4.set_yticklabels([1, 2, 3, 4, 5])
    
    # Add text annotations to heatmap
    for i in range(5):
        for j in range(5):
            text = ax4.text(j, i, f'{int(confusion[i, j])}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax4, label='Count')
    
    plt.tight_layout()
    
    # Save figure
    plot_filename = OUTPUT_FILE.replace('.csv', '_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to {plot_filename}")
    
    # Display the plot
    plt.show()
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()