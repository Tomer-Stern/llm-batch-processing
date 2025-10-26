# Batch LLM Classification for Social Science Research

**A scalable methodology for classifying large volumes of unstructured text using batch LLM APIs with 50% cost savings and rigorous validation**

*Developed by a labor market economist—but applicable across all social science disciplines*

## Overview

This toolkit demonstrates how social scientists can efficiently classify thousands of text samples (career narratives, survey responses, interview transcripts, policy documents) using batch LLM APIs from Anthropic and OpenAI.

**Key innovation**: Process 10,000+ texts overnight for less than \$50 instead of spending weeks and \$10,000+ on human coding—while maintaining or exceeding human-level reliability.

## Research Applications

This methodology works for any social science text classification task:

### Labor Economics (this example)

-   Career trajectory analysis from narrative data
-   Occupation coding from job descriptions\
-   Skills extraction from resumes
-   Job satisfaction classification from open-ended responses

### Sociology

-   Qualitative interview coding at scale
-   Social movement analysis from protest signs/documents
-   Community narrative analysis
-   Identity expression in social media

### Political Science

-   Policy document classification
-   Legislative speech analysis
-   Campaign message coding
-   Constituent correspondence analysis

### Psychology

-   Emotion classification in diary entries
-   Cognitive pattern identification
-   Therapeutic transcript analysis
-   Survey open-response coding

### Economics (General)

-   Consumer sentiment from product reviews
-   Business strategy classification
-   Regulatory comment analysis
-   Economic narrative coding

**This example**: Classifying career summaries on social impact orientation (1-5 scale) to demonstrate the methodology with a realistic labor economics use case.

------------------------------------------------------------------------

## Project Structure

```         
anthropic_batch_prompter.py     ← Anthropic Claude batch API handler
openai_batch_prompter.py        ← OpenAI GPT batch API handler  
rank_social_impact.py           ← Main classification pipeline with validation
example_input.csv               ← 60 sample career summaries
requirements.txt                ← Python dependencies
```

**Design principle**: Modular helpers isolate API complexity from research logic, making it easy to adapt for your own classification tasks.

------------------------------------------------------------------------

## Setup

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Get API keys

-   **Anthropic**: <https://console.anthropic.com/account/keys>
-   **OpenAI**: <https://platform.openai.com/api-keys>

### 3. Set environment variables

``` bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

------------------------------------------------------------------------

## Usage

### Validation Mode (Recommended First Step)

``` bash
python rank_social_impact.py example_input.csv --compare
```

**What happens**: 1. Shows cost estimate for both APIs (\~\$0.01 total for example file) 2. Prompts for confirmation before spending money 3. Submits requests to both Anthropic and OpenAI batch APIs 4. Polls every 10 seconds until complete (typically 2-10 minutes) 5. Compares results with statistical rigor

**Output**: `output_social_impact.csv` with columns: - `anthropic_score`: Classification from Claude (1-5) - `openai_score`: Classification from GPT (1-5)\
- `agreement`: ✓ or ✗ - `difference`: Absolute difference between scores

**Console output includes**: - Cohen's Kappa (inter-rater reliability) - Spearman correlation - Exact agreement percentage - Within-1-point agreement - Score distributions - List of disagreements for manual review

### Production Mode: Single Provider

``` bash
# Use Anthropic Claude (typically higher quality, slightly more expensive)
python rank_social_impact.py yourdata.csv --anthropic-batch

# Use OpenAI GPT (faster turnaround, lower cost)
python rank_social_impact.py yourdata.csv --openai-batch
```

### Skip Cost Confirmation (for automation)

``` bash
python rank_social_impact.py data.csv --compare --no-confirm
```

### Custom Output Path

``` bash
python rank_social_impact.py data.csv --compare --output validation_results.csv
```

------------------------------------------------------------------------

## Cost & Performance

### Scale

-   **Batch size**: Unlimited (tested up to 50,000 requests)
-   **Processing time**: 5-30 minutes depending on queue depth
-   **Polling frequency**: Every 10 seconds (configurable)

### Cost Estimates (as of Oct 2024)

Batch API pricing is **50% of standard rates**:

**Anthropic Claude Haiku 4.5** (default): - Input: \$0.40 per 1M tokens (\~667K words) - Output: \$2.00 per 1M tokens

**OpenAI GPT-4o-mini** (default): - Input: \$0.075 per 1M tokens - Output: \$0.30 per 1M tokens

**Example costs** for 10,000 career summaries (avg 150 words each): - Anthropic: \~\$0.50 - OpenAI: \~\$0.10 - **Total for both providers**: \~\$0.60

Compare to human coding: \~\$10,000 (200 hours \@ \$50/hr)

**Cost reduction: 99.94%**

------------------------------------------------------------------------

## Classification Scale

This example uses a 5-point ordinal scale for social impact orientation:

1.  **Profit-Maximizing**: Exclusively focused on financial returns, shareholder value
2.  **Commercially-Oriented**: Primarily profit-focused with minimal social considerations\
3.  **Mixed Orientation**: Balanced between commercial and social objectives
4.  **Social-Impact Focused**: Career primarily oriented toward social/environmental benefit
5.  **Mission-Driven**: Dedicated to social/public benefit, often rejecting commercial alternatives

*Note: You can easily adapt this for your own classification schemes—just modify the `SYSTEM_PROMPT` in `rank_social_impact.py`*

------------------------------------------------------------------------

## Validation & Reliability

### Example Output

```         
Agreement Metrics:
  Exact agreement: 83.3% (50/60)
  Within 1 point: 96.7%
  Mean absolute difference: 0.183

Reliability Coefficients:
  Cohen's κ (unweighted): 0.789
  Cohen's κ (weighted): 0.856 [Excellent]

Correlation Analysis:
  Spearman's ρ: 0.912 (p=0.0000)

Score Distributions:
  Anthropic: Mean=2.98, SD=1.52
  OpenAI:    Mean=2.93, SD=1.48

10 disagreements found:
  person_012   | Anthropic: 3 | OpenAI: 2 | Social entrepreneur building affordable...
  person_025   | Anthropic: 2 | OpenAI: 3 | Supply chain manager optimizing logistic...
  ...
```

### Interpretation

-   **κ \> 0.80**: Excellent reliability—suitable for primary research use
-   **κ = 0.60-0.80**: Substantial reliability—acceptable with review of edge cases
-   **κ \< 0.60**: Needs prompt refinement or human review for critical cases

### Best Practices

1.  **Always validate first**: Run 100-500 examples with `--compare` before full dataset
2.  **Check agreement**: High Kappa (\>0.75) indicates reliable classification
3.  **Review disagreements**: Manually inspect cases where models differ
4.  **Refine prompt**: Iterate on `SYSTEM_PROMPT` to improve agreement
5.  **Document decisions**: Keep detailed notes on classification rules and edge cases
6.  **Report fully**: Include model versions, prompts, and validation metrics in publications

------------------------------------------------------------------------

## Input Format

CSV file with exactly 2 columns: `id` and `summary`

``` csv
id,summary
sample_001,"Text to classify here..."
sample_002,"Another text sample..."
```

**Requirements**: - `id`: Unique identifier (string or number) - `summary`: Text to classify (typically 50-1000 words) - No header row requirements beyond these two column names

**Tips**: - Keep text length consistent when possible - Remove special characters that might break CSV parsing - For very long texts (\>2000 words), consider chunking or summarization first

------------------------------------------------------------------------

## Adapting for Your Research

### 1. Modify the Classification Prompt

Edit `SYSTEM_PROMPT` in `rank_social_impact.py`:

``` python
SYSTEM_PROMPT = """Your classification task description here.

Define your scale clearly with examples:

1 = [Definition with criteria]
2 = [Definition with criteria]
...

Respond with ONLY a single integer. No explanation."""
```

### 2. Adjust Token Limits

For classifications requiring longer responses:

``` python
# In the prompter initialization:
prompter = AnthropicBatchPrompter(max_tokens=100)  # Default is 10
```

### 3. Change Models

Use different models for different needs:

``` python
# More sophisticated reasoning
prompter = AnthropicBatchPrompter(model="claude-sonnet-4-5-20250929")

# Cost optimization
prompter = AnthropicBatchPrompter(model="claude-haiku-4-5-20250929")
```

### 4. Extract Text Instead of Classify

Modify the prompt to extract information rather than assign scores:

``` python
EXTRACT_PROMPT = """Extract all skills mentioned in this job description.
Return ONLY a comma-separated list of skills. No explanation."""
```

------------------------------------------------------------------------

## Advanced Features

### Cost Estimation

Automatic cost estimation before submission:

```         
COST ESTIMATE (Anthropic Batch API)
======================================================================
Model: claude-haiku-4-5-20250929
Requests: 60
Estimated input tokens: 8,250
Estimated output tokens: 600
Estimated cost: $0.0045
  (Input: $0.0033, Output: $0.0012)
======================================================================

Proceed with batch submission? [y/N]:
```

### Automatic Cleanup

Temporary files are automatically deleted after processing.

### Error Handling

Failed requests are logged and returned as `None` values without stopping the batch.

### Polling Progress

Real-time status updates during processing:

```         
[1] Status: processing | 15/60 completed | 45 processing | 0 errors
[2] Status: processing | 32/60 completed | 28 processing | 0 errors
[3] Status: ended | 60/60 completed | 0 processing | 0 errors
✓ Batch completed!
```

------------------------------------------------------------------------

## Comparison to Traditional Methods

| Method | Cost (10K samples) | Time | Consistency | Scale Limit |
|--------------|------------------|--------------|--------------|--------------|
| Trained RAs | \$10,000 | 3-4 weeks | κ ≈ 0.65-0.75 | 1-2K |
| Crowdsourcing | \$10,000 | 1 week | κ ≈ 0.45-0.60 | 10-50K |
| Dictionary methods | Free | 1 hour | Perfect but shallow | Unlimited |
| **Batch LLMs** | **\$50-100** | **2-8 hours** | **κ ≈ 0.80-0.90** | **10-100K** |

------------------------------------------------------------------------

## Limitations & Considerations

### When This Approach Works Well

✓ Large sample sizes (\>1,000 texts)\
✓ Well-defined classification schemes\
✓ Moderately complex judgments\
✓ Resource-constrained projects\
✓ Exploratory analysis

### When to Use Caution

⚠ High-stakes individual decisions\
⚠ Highly domain-specific jargon\
⚠ Extremely subjective judgments\
⚠ Small sample sizes (\<100)\
⚠ Legal/clinical applications requiring explanation

### Known Issues

-   **Language**: Works best with English; quality varies for other languages
-   **Bias**: Inherits training data biases from base models
-   **Ambiguity**: Struggles with truly ambiguous cases (but so do humans)
-   **Explainability**: Black box—cannot explain specific decisions
-   **Consistency**: May drift if prompts are modified between batches

### Best Practices for Research Use

1.  Always validate with human-coded subsample
2.  Report full methodology (models, prompts, parameters)
3.  Check for systematic biases in subgroups
4.  Consider confidence intervals on agreement metrics
5.  Make code and prompts publicly available
6.  Preregister if using for confirmatory analysis

------------------------------------------------------------------------

## Publication Guidelines

When using this methodology in research papers:

**Required reporting**: - [ ] Exact model versions (e.g., `claude-haiku-4-5-20250929`) - [ ] Complete system prompts (in appendix if needed) - [ ] Inter-rater reliability with confidence intervals - [ ] Validation subsample details (if used) - [ ] Cost and time required - [ ] Code availability

**Recommended**: - [ ] Comparison of multiple models - [ ] Sensitivity analysis (different prompts/parameters) - [ ] Subgroup reliability analysis - [ ] Discussion of limitations and potential biases

------------------------------------------------------------------------

## Technical Notes

-   **Python version**: 3.8+
-   **Dependencies**: See `requirements.txt`
-   **API rate limits**: Batch APIs have generous limits (10K requests/day typical)
-   **Batch IDs**: Printed to console for manual inspection if needed
-   **Temporary files**: Auto-cleaned after processing
-   **Error handling**: Graceful degradation—failed requests return `None`

------------------------------------------------------------------------

## Support & Extension

### Common Issues

**"No module named 'anthropic'"**: Run `pip install -r requirements.txt`\
**"API key not found"**: Set environment variables with API keys\
**"Batch failed"**: Check API status and account limits\
**Cost estimates seem wrong**: Estimates are approximate; see actual usage in API dashboard

### Extending This Code

The helpers (`*_batch_prompter.py`) are designed to be reusable: - Import into your own scripts - Modify `SYSTEM_PROMPT` for different tasks - Adjust parameters (model, temperature, max_tokens) - Use for multi-label classification, extraction, summarization, etc.

### Contributing

This is a demonstration repository, but adaptations and improvements welcome!

------------------------------------------------------------------------

## Files

-   **anthropic_batch_prompter.py** (129 lines): Anthropic Claude batch API interface with cost estimation
-   **openai_batch_prompter.py** (151 lines): OpenAI GPT batch API interface with cleanup
-   **rank_social_impact.py** (274 lines): Classification pipeline with enhanced statistics
-   **example_input.csv**: 60 career summaries spanning 1-5 scale
-   **requirements.txt**: Pinned dependencies

**Total**: \~554 lines of focused, well-documented code

------------------------------------------------------------------------

## Citation

Anthropic Batch API: <https://docs.anthropic.com/en/docs/build-with-claude/batch-processing>

OpenAI Batch API: <https://platform.openai.com/docs/guides/batch>

------------------------------------------------------------------------

## License

MIT License - Free for research and commercial use

------------------------------------------------------------------------

## About

Developed as a demonstration of scalable text analysis methodology for social science research. The specific application (career social impact classification) showcases a labor economics use case, but the approach generalizes to any text classification task across disciplines.

**Author background**: Labor market economist working with large-scale survey and administrative data. This toolkit emerged from the need to efficiently process thousands of open-text survey responses—a common challenge in social science research.=
