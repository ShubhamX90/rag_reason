# Batch Processing Quick Reference

## Processing Multiple Files in Parallel

### Method 1: Enhanced run_evaluation.py (Simple)

```bash
# Basic usage - process multiple files
python run_evaluation.py --input file1.jsonl file2.jsonl file3.jsonl --committee default

# With all options
python run_evaluation.py \
    --input data/file1.jsonl data/file2.jsonl data/file3.jsonl \
    --committee default \
    --batch-size 50 \
    --max-samples 100 \
    --output-dir outputs \
    --verbose

# Process sequentially (one at a time) instead of parallel
python run_evaluation.py \
    --input file1.jsonl file2.jsonl file3.jsonl \
    --committee default \
    --process-sequentially
```

### Method 2: run_evaluation_batch.py (Advanced)

```bash
# Process all JSONL files in a directory
python run_evaluation_batch.py --inputs data/*.jsonl --committee default

# Limit concurrent processing
python run_evaluation_batch.py \
    --inputs data/*.jsonl \
    --committee default \
    --max-concurrent-files 3

# Full options
python run_evaluation_batch.py \
    --inputs data/eval1.jsonl data/eval2.jsonl data/eval3.jsonl \
    --output-prefix my_results \
    --committee default \
    --batch-size 50 \
    --max-samples 100 \
    --max-concurrent-files 5 \
    --verbose
```

## Committee Options

```bash
# Default committee (recommended)
--committee default
# Uses: Haiku + DeepSeek + Qwen + Mistral
# Cost: ~$0.05 per 3 samples

# Conservative committee (cheaper)
--committee conservative
# Uses: Haiku + Qwen + Mistral
# Cost: ~$0.03 per 3 samples

# No committee (single judge)
--committee none
# Uses: Only Haiku
# Cost: ~$0.015 per 3 samples
```

## Output Structure

When processing multiple files, each gets its own directory:

```
outputs/
├── file1_20250212_153045/
│   ├── eval_report.md
│   └── detailed_results.json
├── file2_20250212_153046/
│   ├── eval_report.md
│   └── detailed_results.json
└── file3_20250212_153047/
    ├── eval_report.md
    └── detailed_results.json
```

## Cost Estimation (Fixed!)

Expected costs with default committee:
- 3 samples: ~$0.05
- 10 samples: ~$0.15-0.20
- 100 samples: ~$1.50-2.00

Per-model breakdown:
- Haiku: ~$0.0045 per 3 samples
- DeepSeek: ~$0.0058 per 3 samples  
- Qwen: ~$0.0002 per 3 samples
- Mistral: ~$0.00 (free tier)

## Common Patterns

### Research Workflow
```bash
# Evaluate multiple model outputs on same dataset
python run_evaluation_batch.py \
    --inputs results/model_a.jsonl results/model_b.jsonl results/model_c.jsonl \
    --committee default \
    --output-prefix comparison_study
```

### Testing Workflow
```bash
# Quick test on multiple samples
python run_evaluation.py \
    --input data/*.jsonl \
    --max-samples 5 \
    --committee default
```

### Production Workflow
```bash
# Full evaluation with controlled concurrency
python run_evaluation_batch.py \
    --inputs data/*.jsonl \
    --committee default \
    --max-concurrent-files 3 \
    --output-prefix production_eval_2025
```

## Performance Tips

1. **Optimal Concurrency**: 3-5 files simultaneously
2. **Memory Usage**: ~1-2GB per file being processed
3. **Rate Limits**: Shared across all files (OpenRouter: 30-100 req/min)
4. **Batch Size**: Default 50 is optimal, increase to 100 for faster processing

## Troubleshooting

### Issue: Rate limit errors
**Solution**: Reduce `--max-concurrent-files` to 2 or 3

### Issue: Out of memory
**Solution**: Process sequentially with `--process-sequentially`

### Issue: Cost still seems high
**Solution**: Check you're using the patched version with fixed pricing

### Issue: Factual grounding still 0.000
**Solution**: 
1. Check NLI responses in detailed_results.json
2. Verify claims are being extracted from answers
3. Ensure support_docs are present in input data
