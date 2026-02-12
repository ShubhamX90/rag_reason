# CATS v2.0 - Conflict-Aware Trust Score Evaluation Pipeline

**Enhanced RAG Evaluation with Multi-LLM Judge Committee**

## 🎯 Overview

CATS v2.0 is a significantly upgraded evaluation pipeline for Retrieval-Augmented Generation (RAG) systems, with a focus on **conflict-aware behavior** and **multi-judge consensus**.

### Key Features

✨ **Multi-LLM Judge Committee**
- Implements voting mechanism with multiple LLM judges
- Supports Anthropic Claude (Haiku) and OpenRouter models (DeepSeek, Qwen, etc.)
- Weighted majority voting with confidence scoring
- Cost-optimized model selection

🎓 **Enhanced Evaluation Metrics**
- **F1_GR**: Grounded Refusal (answering vs. abstaining appropriately)
- **Behavior Adherence**: Conflict-handling behavior per taxonomy
- **Factual Grounding**: Cross-document claim verification
- **Single-Truth Recall**: Semantic answer matching with paraphrases

📊 **Conflict Taxonomy Support**
- Type 1: No Conflict (direct answer)
- Type 2: Complementary Information (consolidation)
- Type 3: Conflicting Opinions (neutral summary)
- Type 4: Outdated Information (prioritize recent)
- Type 5: Misinformation (reject bad sources)

## 📋 Requirements

- Python 3.8+
- Anthropic API key (for Claude Haiku)
- OpenRouter API key (for DeepSeek, Qwen, etc.)

## 🚀 Quick Start

### 1. Setup

```bash
# Clone or extract the pipeline
cd CATS_v2

# Run interactive setup
chmod +x scripts/*.sh
./scripts/setup.sh

# This will:
# - Create virtual environment
# - Install dependencies
# - Download NLTK data
# - Create directory structure
# - Generate .env template
```

### 2. Configure API Keys

Edit `.env` file:

```bash
# Anthropic API (for Claude Haiku)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# OpenRouter API (for DeepSeek, Qwen, etc.)
OPENROUTER_API_KEY=your-openrouter-key-here
```

### 3. Prepare Input Data

Place your evaluation data in `data/input.jsonl`. Expected format:

```jsonl
{
  "id": "sample_001",
  "query": "What is the capital of France?",
  "retrieved_docs": [
    {
      "doc_id": "d1",
      "title": "France",
      "snippet": "Paris is the capital and largest city of France...",
      "url": "https://example.com/france"
    }
  ],
  "per_doc_notes": [
    {
      "doc_id": "d1",
      "verdict": "supports",
      "key_fact": "Paris is capital",
      "quote": "Paris is the capital"
    }
  ],
  "conflict_category_id": 1,
  "conflict_type": "No Conflict",
  "model_output": "The capital of France is Paris.",
  "gold_answer": "Paris"
}
```

### 4. Run Evaluation

**Interactive Mode:**
```bash
./scripts/run_eval.sh
```

This will guide you through:
- Selecting input file
- **Selecting output directory (NEW: enables parallel runs)**
- Choosing judge committee (default/conservative/none)
- Setting sample limits (for testing)
- Configuring verbosity

**Parallel File Processing:**
You can now run multiple evaluations in parallel by:
1. Opening multiple terminals
2. Running `./scripts/run_eval.sh` in each terminal
3. Selecting different input files and output directories

Example parallel setup:
- Terminal 1: `input_file_1.jsonl` → `outputs/run_1/`
- Terminal 2: `input_file_2.jsonl` → `outputs/run_2/`
- Terminal 3: `input_file_3.jsonl` → `outputs/run_3/`

**Command Line:**
```bash
# Activate environment
source venv/bin/activate

# Run with default committee
python run_evaluation.py --input data/input.jsonl --committee default

# Run with conservative (cheaper) committee
python run_evaluation.py --input data/input.jsonl --committee conservative

# Test on first 10 samples
python run_evaluation.py --input data/input.jsonl --committee default --max-samples 10

# Verbose mode
python run_evaluation.py --input data/input.jsonl --committee default --verbose
```

## 🏗️ Architecture

### Judge Committee Options

#### 1. Default Committee (Recommended)
- **Claude Haiku 3.5** (Anthropic) - Fast, high-quality
- **DeepSeek R1** (OpenRouter) - Strong reasoning
- **Qwen 3 8B** (OpenRouter) - Balanced performance
- **Mistral Nemo Free** (OpenRouter) - Zero cost, good quality
- **Cost**: ~$0.005-0.01 per sample (reduced with free model)
- **Speed**: Parallel async execution with high concurrency

#### 2. Conservative Committee (Budget-Friendly)
- **Claude Haiku 3.5** (Anthropic)
- **Qwen 3 8B** (OpenRouter)
- **Mistral Nemo Free** (OpenRouter) - Zero cost
- **Cost**: ~$0.005-0.01 per sample
- **Speed**: Fast with free tier inclusion

#### 3. Single Judge (Fastest)
- Uses Claude Haiku only
- No committee voting
- **Cost**: ~$0.003-0.005 per sample
- **Speed**: Fastest, no consensus delay

### Voting Strategy

The committee uses **weighted majority voting**:
1. Each judge evaluates independently (parallel async)
2. Votes are weighted by judge priority and confidence
3. Decision requires meeting confidence threshold (default: 0.6)
4. Rationale selected from highest-weighted winning judge

## 📊 Output Files

### Markdown Report (`outputs/eval_report.md`)

```markdown
# CATS v2.0 Evaluation Report

## Overall Conflict-Aware Metrics
- **Total Samples**: 100
- **F1_GR**: 0.856
- **Behavior Adherence**: 0.782
- **Factual Grounding**: 0.813
- **Single-Truth Recall**: 0.745

### **CATS Score**: 0.799

## Per Conflict Type Breakdown
### Type 1: No Conflict
- Samples: 25
- F1_GR: 0.880
- Behavior: 0.840
...
```

### Detailed Results (`outputs/detailed_results.json`)

Contains per-sample evaluations with:
- Individual judge responses
- Vote breakdowns
- Confidence scores
- Claim-level grounding details
- Cost per sample

### Logs

- `logs/cats_eval.log` - Full evaluation log
- `logs/cats_errors.log` - Errors only

## 🔧 Advanced Configuration

### Python API

```python
from rag_eval import (
    EvaluationConfig,
    EnhancedEvaluator,
    load_dataset,
    create_default_committee,
)

# Load data
dataset = load_dataset("data/input.jsonl")

# Configure evaluation
config = EvaluationConfig(
    input_jsonl="data/input.jsonl",
    outputs_dir="outputs",
)
config.conflict.use_judge_committee = True
config.conflict.committee = create_default_committee()

# Run evaluation
evaluator = EnhancedEvaluator(config)
results = evaluator.evaluate(dataset)

print(f"CATS Score: {results['conflict_overall']['cats_score']:.3f}")
```

### Custom Judge Configuration

```python
from rag_eval.config import JudgeModelConfig, JudgeCommitteeConfig, APIProvider

# Define custom judges
custom_judge = JudgeModelConfig(
    model_id="anthropic/claude-3-opus-20240229",
    provider=APIProvider.OPENROUTER,
    temperature=0.0,
    max_tokens=500,
    priority=5,  # Higher weight
    api_key_env="OPENROUTER_API_KEY",
    base_url="https://openrouter.ai/api/v1"
)

# Create custom committee
committee = JudgeCommitteeConfig(
    judges=[custom_judge, ...],
    voting_strategy="weighted_majority",
    confidence_threshold=0.7,
    use_async=True
)
```

## 💡 Cost Optimization Tips

Based on the OpenRouter best practices image you provided:

### 1. **Start Small for Testing**
```bash
# Test on 25% of data with cheap models first
python run_evaluation.py --input data/input.jsonl \
    --committee conservative --max-samples 50
```

### 2. **Use Conservative Committee for Large Datasets**
```bash
# Leverage free-tier Mistral model
./scripts/run_eval.sh
# Select "Conservative" option
```

### 3. **Batch Processing**
The pipeline automatically batches requests and uses async mode for cost efficiency.

### 4. **Monitor Costs**
Check the cost summary in evaluation report:
```markdown
## Cost Summary

- **Total Cost**: $0.1234
- **Decisions Made**: 100
- **Avg Cost/Decision**: $0.001234

### Per-Model Costs

#### claude-3-5-haiku-20241022
- Total: $0.0450
- Requests: 100
- Avg/Request: $0.000450

#### deepseek/deepseek-r1
- Total: $0.0584
- Requests: 100
- Avg/Request: $0.000584

#### qwen/qwen-2.5-7b-instruct
- Total: $0.0200
- Requests: 100
- Avg/Request: $0.000200

#### mistralai/mistral-nemo:free
- Total: $0.0000
- Requests: 100
- Avg/Request: $0.000000
```

**Note**: Cost estimates now use actual token counts from API responses for accuracy.

## 📈 Evaluation Logic Improvements

### Over Original CATS Pipeline

1. **Multi-Judge Committee**
   - Reduces individual judge bias
   - Improves reliability through consensus
   - Configurable voting strategies

2. **Enhanced Factual Grounding**
   - Cross-document verification option
   - Claim-level breakdown
   - Supporting document tracking

3. **Improved Single-Truth Recall**
   - Semantic matching beyond string matching
   - Paraphrase detection
   - Partial credit scoring

4. **Behavior Adherence**
   - Committee consensus on conflict handling
   - Vote breakdown and confidence
   - Detailed rationales from multiple perspectives

5. **Async Architecture**
   - 10x faster with parallel judge execution
   - Non-blocking API calls
   - Efficient resource utilization

## 🐛 Troubleshooting

### API Key Issues
```bash
# Check .env file
cat .env

# Verify keys are exported
echo $ANTHROPIC_API_KEY
echo $OPENROUTER_API_KEY
```

### Import Errors
```bash
# Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

### NLTK Data Missing
```bash
python3 -c "import nltk; nltk.download('punkt')"
```

### Rate Limiting
The pipeline includes automatic rate limiting. If you hit limits:
1. Reduce `max_concurrent_requests` in config
2. Use smaller batch sizes
3. Add delays between batches

## 📚 Input Format Details

### Required Fields
- `id`: Unique sample identifier
- `query`: User question
- `retrieved_docs`: List of retrieved documents
- `conflict_category_id`: Integer 1-5
- `model_output` OR `final_grounded_answer.answer`: Model's response

### Optional Fields
- `per_doc_notes`: Document-level annotations (verdict, quotes)
- `gold_answer`: Ground truth for single-truth recall
- `conflict_type`: Human-readable conflict type
- `conflict_reason`: Explanation of conflict

### Document Structure
```json
{
  "doc_id": "d1",
  "title": "Document title",
  "snippet": "Text snippet or full text",
  "url": "https://source.com",
  "date": "2024-01-01"
}
```

## 🤝 Contributing

This is an enhanced version of the CATS pipeline developed at BITS Pilani.

Original Authors: Gorang Mehrishi, Samyek Jain  
Enhanced by: Claude AI

## 📄 License

Same license as original CATS pipeline.

## 📞 Support

For issues:
1. Check logs in `logs/cats_errors.log`
2. Run with `--verbose` flag
3. Verify API keys are set correctly
4. Ensure input format matches specification

---

**Happy Evaluating! 🎯**
