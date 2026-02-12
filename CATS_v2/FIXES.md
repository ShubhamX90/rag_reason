# CATS v2.0 - Bug Fixes and Enhancements

## Summary of Changes

This document describes all the fixes and enhancements made to the CATS v2.0 evaluation pipeline.

---

## 🐛 Bug Fixes

### 1. Cost Estimation Fixed (1000x error)

**Issue**: Cost estimates were showing $10-$21 for 3 samples when actual cost was ~$0.05

**Root Cause**: Pricing was configured per 1k tokens when it should have been per 1M tokens basis.

**Files Modified**: `rag_eval/config.py`

**Changes**:
- **Claude 3.5 Haiku**:
  - Input: $1.00 → $0.001 per 1k tokens ($1/MTok → $0.001/1k)
  - Output: $5.00 → $0.005 per 1k tokens ($5/MTok → $0.005/1k)

- **DeepSeek R1**:
  - Input: $0.55 → $0.00055 per 1k tokens
  - Output: $2.19 → $0.00219 per 1k tokens

- **Qwen 2.5**:
  - Input: $0.06 → $0.00006 per 1k tokens
  - Output: $0.06 → $0.00006 per 1k tokens

**Expected Result**: For 3 samples with default committee, cost should now show ~$0.05 instead of $10-$21.

---

### 2. Mistral Model Name Fixed

**Issue**: Mistral model was using `mistralai/mistral-nemo:free` which caused 404 errors when the `:free` variant wasn't available.

**Root Cause**: OpenRouter's canonical slug is `mistralai/mistral-nemo` without the `:free` suffix.

**Files Modified**: `rag_eval/config.py`

**Changes**:
```python
# Before
model_id="mistralai/mistral-nemo:free"

# After
model_id="mistralai/mistral-nemo"
```

**Expected Result**: Mistral model should now work correctly in the default committee without 404 errors.

---

### 3. Factual Grounding Fixed (Was showing 0.000)

**Issue**: All samples showed 0.000 for factual grounding metric.

**Root Cause**: NLI response parsing was looking at the wrong field. The code was searching for "entails" in the `rationale` text instead of properly parsing the JSON `relation` field.

**Files Modified**: `rag_eval/conflict_eval.py`

**Changes**:
The enhanced_factual_grounding function now:
1. Attempts to extract and parse JSON from the rationale
2. Looks for the `relation` field in the parsed JSON
3. Falls back to text matching if JSON parsing fails

```python
# Added proper JSON parsing
import json as json_lib
try:
    json_start = response.rationale.find("{")
    json_end = response.rationale.rfind("}") + 1
    if json_start != -1 and json_end > json_start:
        json_str = response.rationale[json_start:json_end]
        parsed_json = json_lib.loads(json_str)
        relation = parsed_json.get("relation", "").lower()
    else:
        relation = response.rationale.lower()
except Exception:
    relation = response.rationale.lower()
```

**Expected Result**: Factual grounding should now properly detect when claims are supported by documents, showing values > 0.000 for properly grounded answers.

---

## ✨ New Features

### 4. Parallel File Processing

**Issue**: Users had to open multiple terminals to process multiple files simultaneously.

**Solution**: Two new approaches for processing multiple files:

#### Option A: Enhanced Original Script

**File**: `run_evaluation.py` (modified)

**New Features**:
- Accepts multiple input files: `--input file1.jsonl file2.jsonl file3.jsonl`
- Processes files in parallel by default
- Option to process sequentially: `--process-sequentially`
- Each file gets its own timestamped output directory

**Usage Examples**:
```bash
# Process multiple files in parallel
python run_evaluation.py --input file1.jsonl file2.jsonl file3.jsonl --committee default

# Process sequentially (one at a time)
python run_evaluation.py --input file1.jsonl file2.jsonl --committee default --process-sequentially

# Single file (backwards compatible)
python run_evaluation.py --input data/input.jsonl --committee default
```

#### Option B: New Batch Runner Script

**File**: `run_evaluation_batch.py` (new)

**Features**:
- Dedicated batch processing script
- Supports glob patterns: `--inputs data/*.jsonl`
- Controlled concurrency: `--max-concurrent-files 5`
- Better logging for batch operations
- Comprehensive summary at the end

**Usage Examples**:
```bash
# Process all JSONL files in data directory
python run_evaluation_batch.py --inputs data/*.jsonl --committee default

# Process specific files with concurrency limit
python run_evaluation_batch.py --inputs file1.jsonl file2.jsonl file3.jsonl \
    --max-concurrent-files 3 --committee default

# Process with custom output prefix
python run_evaluation_batch.py --inputs data/*.jsonl --output-prefix my_results
```

**Expected Result**: Users can now process 5 files in parallel from a single terminal without any manual coordination.

---

## 📊 Enhanced Cost Reporting

**File**: `rag_eval/evaluator.py` (already had this feature)

The evaluation report now shows per-model costs separately:

```markdown
### Per-Model Costs

#### claude-3-5-haiku-20241022
- Total: $0.0045
- Requests: 9
- Avg/Request: $0.000500

#### deepseek/deepseek-r1
- Total: $0.0058
- Requests: 9
- Avg/Request: $0.000644
```

This was already implemented in the code but wasn't working correctly due to the pricing bug.

---

## 🧪 Testing the Fixes

### Test 1: Cost Estimation
```bash
# Run on 3 samples
python run_evaluation.py --input data/example_input.jsonl --max-samples 3 --committee default

# Expected cost: ~$0.05 (not $10-$21)
# Check the output report for per-model costs
```

### Test 2: Factual Grounding
```bash
# Run on example data
python run_evaluation.py --input data/example_input.jsonl --committee default

# Check eval_report.md - Factual Grounding should be > 0.000
# For the example data, expect values around 0.7-0.9
```

### Test 3: Mistral Model
```bash
# The default committee now includes Mistral
python run_evaluation.py --input data/example_input.jsonl --committee default

# Should complete without 404 errors from Mistral model
```

### Test 4: Parallel Processing
```bash
# Create multiple test files
cp data/example_input.jsonl data/test1.jsonl
cp data/example_input.jsonl data/test2.jsonl
cp data/example_input.jsonl data/test3.jsonl

# Process all in parallel
python run_evaluation.py --input data/test*.jsonl --committee default

# Or use the batch runner
python run_evaluation_batch.py --inputs data/test*.jsonl --committee default
```

---

## 📝 Migration Notes

### Backwards Compatibility

All changes are **backwards compatible**. Existing scripts and workflows will continue to work:

```bash
# This still works exactly as before
python run_evaluation.py --input data/input.jsonl --committee default
```

### Recommended Workflow for Multiple Files

**Before** (Required multiple terminals):
```bash
# Terminal 1
python run_evaluation.py --input file1.jsonl --output-dir outputs/run1

# Terminal 2
python run_evaluation.py --input file2.jsonl --output-dir outputs/run2

# Terminal 3
python run_evaluation.py --input file3.jsonl --output-dir outputs/run3
```

**After** (Single terminal):
```bash
# Option 1: Enhanced original script
python run_evaluation.py --input file1.jsonl file2.jsonl file3.jsonl --committee default

# Option 2: Batch runner (recommended for many files)
python run_evaluation_batch.py --inputs data/*.jsonl --committee default
```

---

## 🔧 Configuration Updates

No configuration file changes are required. All fixes are in the code itself.

However, if you have custom judge configurations, update pricing:
- Divide all `cost_per_1k_input` values by 1000
- Divide all `cost_per_1k_output` values by 1000

---

## 📈 Performance Notes

### Parallel Processing
- **Memory Usage**: Each file runs in its own evaluator instance. For N files, expect ~N×memory usage.
- **API Rate Limits**: All files share the same API keys, so OpenRouter/Anthropic rate limits apply globally.
- **Recommended Concurrency**: 3-5 files simultaneously for optimal performance without hitting rate limits.

### Cost Savings
With corrected pricing, you can now accurately:
- Budget for large evaluation runs
- Compare cost efficiency across different committees
- Make informed decisions about model selection

---

## 🐛 Known Limitations

1. **Factual Grounding Parsing**: While improved, the NLI parsing relies on finding JSON in the response. If a model returns malformed JSON, it falls back to text matching which may be less accurate.

2. **Mistral Model**: The model is now using the canonical slug, but pricing may vary based on your OpenRouter account and region. The config shows $0.00 cost, but check your OpenRouter dashboard for actual charges.

3. **Parallel Processing**: When processing many files (10+), consider using `--max-concurrent-files` to limit resource usage and avoid rate limit issues.

---

## 📞 Support

If you encounter issues with these fixes:

1. Check the logs in `logs/cats_eval.log` and `logs/cats_errors.log`
2. Verify API keys are set correctly in `.env`
3. For cost discrepancies, compare with OpenRouter dashboard and Anthropic console
4. For factual grounding issues, check the detailed_results.json for NLI response details

---

## ✅ Checklist

After applying these fixes, verify:

- [ ] Cost estimates are reasonable (~$0.05 for 3 samples with default committee)
- [ ] Mistral model works without 404 errors
- [ ] Factual grounding shows non-zero values for grounded answers
- [ ] Per-model costs are displayed separately in reports
- [ ] Multiple files can be processed from single terminal
- [ ] Existing single-file workflows still work

---

## Version

**CATS v2.0 - Patched**
- Original Version: v2.0
- Patch Date: 2025-02-12
- Fixed By: Claude (Anthropic AI)
