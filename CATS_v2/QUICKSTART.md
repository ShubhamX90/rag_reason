# CATS v2.0 - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Setup (2 minutes)

```bash
cd CATS_v2
./scripts/setup.sh
```

This automatically:
- Creates virtual environment
- Installs all dependencies
- Downloads NLTK data
- Creates directory structure
- Generates configuration files

### Step 2: Configure API Keys (1 minute)

Edit `.env` file with your API keys:

```bash
nano .env  # or use your favorite editor
```

Add your keys:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENROUTER_API_KEY=your-openrouter-key-here
```

**Where to get keys:**
- Anthropic: https://console.anthropic.com/
- OpenRouter: https://openrouter.ai/keys

### Step 3: Test Run (2 minutes)

```bash
# Activate environment
source venv/bin/activate

# Run on example data (3 samples)
python run_evaluation.py \
    --input data/example_input.jsonl \
    --committee default \
    --verbose
```

Expected output:
```
CATS Score: ~0.800
Total Cost: $0.01-0.02
```

Check results:
```bash
cat outputs/eval_report.md
```

## 🎯 Evaluate Your Data

### Option A: Interactive Mode (Recommended)

```bash
./scripts/run_eval.sh
```

Follow the prompts:
1. Select your input file
2. Choose judge committee (default/conservative)
3. Set sample limit (optional)
4. Confirm and run

### Option B: Command Line

```bash
# Full evaluation with default committee
python run_evaluation.py \
    --input data/your_data.jsonl \
    --committee default

# Test on first 10 samples
python run_evaluation.py \
    --input data/your_data.jsonl \
    --committee default \
    --max-samples 10

# Budget mode (conservative committee)
python run_evaluation.py \
    --input data/your_data.jsonl \
    --committee conservative
```

## 📊 Understanding Results

### Markdown Report (`outputs/eval_report.md`)

```markdown
## Overall Metrics
- F1_GR: 0.856          # Grounded refusal accuracy
- Behavior: 0.782       # Conflict handling adherence  
- Grounding: 0.813      # Claim-document support
- Recall: 0.745         # Gold answer coverage

CATS Score: 0.799       # Overall quality
```

### Per-Type Breakdown

Results split by conflict type (1-5):
- Type 1: No Conflict
- Type 2: Complementary Info
- Type 3: Conflicting Opinions
- Type 4: Outdated Info
- Type 5: Misinformation

### Cost Tracking

```markdown
Total Cost: $0.15
Avg/Decision: $0.0015
```

## 🔧 Common Tasks

### Test on Small Sample

```bash
python run_evaluation.py \
    --input data/my_data.jsonl \
    --max-samples 5 \
    --verbose
```

### Use Single Judge (Fastest)

```bash
python run_evaluation.py \
    --input data/my_data.jsonl \
    --committee none
```

### Monitor Costs

Check real-time cost in:
- `outputs/eval_report.md` (summary)
- `outputs/detailed_results.json` (per-sample)

## 📝 Input Format

Minimal required JSONL:

```jsonl
{
  "id": "sample_1",
  "query": "Your question?",
  "retrieved_docs": [
    {
      "doc_id": "d1",
      "snippet": "Document text...",
      "url": "https://..."
    }
  ],
  "conflict_category_id": 1,
  "model_output": "Model's answer...",
  "gold_answer": "Expected answer"
}
```

See `data/example_input.jsonl` for complete examples.

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "API key not found"
```bash
# Check .env file
cat .env

# Reload environment
source venv/bin/activate
export $(cat .env | grep -v '^#' | xargs)
```

### "Rate limit exceeded"
Reduce concurrent requests in config or use conservative committee.

## 💡 Tips

1. **Start Small**: Test on 10-20 samples first
2. **Use Conservative Mode**: For large datasets (>100 samples)
3. **Check Costs**: Monitor `cost_summary` in reports
4. **Batch Jobs**: Split large datasets into batches

## 📚 Next Steps

- Read full [README.md](README.md) for advanced features
- Customize judges in `rag_eval/config.py`
- Review detailed results in `outputs/detailed_results.json`
- Check logs in `logs/cats_eval.log`

---

**Need Help?** Check logs first, then review README.md troubleshooting section.
