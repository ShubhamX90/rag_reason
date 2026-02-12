# CATS v2.0 - Project Structure

## 📁 Directory Layout

```
CATS_v2/
│
├── README.md                      # Main documentation
├── QUICKSTART.md                  # 5-minute getting started guide
├── CHANGELOG.md                   # Version history and improvements
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment configuration template
├── .env                          # Your API keys (create from .env.example)
│
├── run_evaluation.py             # Main evaluation script (CLI)
├── test_installation.py          # Installation validation script
│
├── rag_eval/                     # Core evaluation package
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration classes and presets
│   ├── evaluator.py              # Main evaluation orchestrator
│   ├── judge_committee.py        # Multi-LLM judge voting system
│   ├── conflict_eval.py          # Conflict-aware evaluation metrics
│   ├── metrics.py                # Utility functions for metrics
│   ├── judge_prompts.py          # LLM judge prompt templates
│   ├── data.py                   # Data loading and utilities
│   └── logging_config.py         # Logging configuration
│
├── scripts/                      # Interactive bash scripts
│   ├── setup.sh                  # One-command installation
│   └── run_eval.sh               # Interactive evaluation wizard
│
├── configs/                      # Configuration files (optional)
│   └── default.yaml              # Default configuration (generated)
│
├── data/                         # Input data directory
│   └── example_input.jsonl       # Example evaluation data
│
├── outputs/                      # Evaluation outputs
│   ├── eval_report.md            # Human-readable markdown report
│   └── detailed_results.json     # Machine-parsable detailed results
│
├── logs/                         # Log files
│   ├── cats_eval.log             # Full evaluation log
│   └── cats_errors.log           # Errors only
│
└── .cache/                       # Cache directory (optional)
```

## 📦 Core Modules

### `rag_eval/config.py`
**Purpose**: Configuration management for evaluation pipeline

**Key Classes**:
- `JudgeModelConfig`: Single judge configuration
  - Model ID, provider, costs, rate limits
  
- `JudgeCommitteeConfig`: Committee voting configuration
  - List of judges, voting strategy, thresholds
  
- `EnhancedTrustScoreConfig`: TRUST-SCORE evaluation settings
  - Grounding, citations, consistency checks
  
- `EnhancedConflictEvalConfig`: Conflict evaluation settings
  - Behavior adherence, factual grounding, recall
  
- `EvaluationConfig`: Master configuration
  - Paths, pipeline settings, subsystem configs

**Key Functions**:
- `get_haiku_judge()`: Anthropic Claude Haiku preset
- `get_deepseek_judge()`: DeepSeek R1 preset
- `get_qwen_judge()`: Qwen 3 8B preset
- `create_default_committee()`: Balanced committee
- `create_conservative_committee()`: Budget-friendly committee

### `rag_eval/judge_committee.py`
**Purpose**: Multi-LLM judge voting system

**Key Classes**:
- `JudgeResponse`: Single judge's evaluation result
  - Vote, rationale, confidence, cost, latency
  
- `CommitteeDecision`: Aggregated committee decision
  - Final vote, confidence, vote breakdown, costs
  
- `JudgeClient`: Interface to single judge model
  - API calls, response parsing, cost tracking
  
- `JudgeCommittee`: Orchestrates multiple judges
  - Parallel execution, vote aggregation, strategies

**Voting Strategies**:
- **Majority**: Simple majority (1 vote each)
- **Weighted Majority**: Priority × confidence weighting
- **Unanimous**: All must agree

### `rag_eval/evaluator.py`
**Purpose**: Main evaluation orchestration

**Key Class**:
- `EnhancedEvaluator`: Coordinates entire evaluation
  - Dataset loading
  - Async parallel evaluation
  - Committee management
  - Result aggregation
  - Report generation

**Key Methods**:
- `evaluate_async()`: Main async evaluation loop
- `evaluate()`: Sync wrapper
- `_evaluate_conflicts_async()`: Conflict-aware evaluation
- `_evaluate_single_sample()`: Per-sample evaluation
- `_aggregate_results()`: Aggregate to overall metrics

### `rag_eval/conflict_eval.py`
**Purpose**: Conflict-aware evaluation metrics

**Key Functions**:

**Behavior Adherence**:
- `committee_behavior_adherence()`: Multi-judge voting
- `behavior_adherence()`: Single judge (backward compatible)

**Factual Grounding**:
- `enhanced_factual_grounding()`: Cross-doc verification
- `factual_grounding_ratio()`: Single-doc (backward compatible)

**Single-Truth Recall**:
- `enhanced_single_truth_recall()`: Semantic matching
- `single_truth_answer_recall()`: String matching (backward compatible)

### `rag_eval/metrics.py`
**Purpose**: Utility functions for evaluation

**Key Functions**:
- `answered_flags()`: Detect answer vs refusal
- `extract_claims_by_sentence()`: Split answer into claims
- `extract_bracket_citations()`: Find [dX] citations
- `f1_gr_from_flags()`: Grounded refusal F1
- `normalize_answer()`: Text normalization
- `remove_citations()`: Clean citation markers

### `rag_eval/judge_prompts.py`
**Purpose**: Prompt templates for LLM judges

**Key Components**:
- `BEHAVIOR_RUBRIC`: Conflict type expectations
- `behavior_judge_prompt()`: Behavior adherence prompt
- `nli_prompt()`: Natural language inference prompt
- `single_truth_recall_prompt()`: Gold answer matching prompt

### `rag_eval/data.py`
**Purpose**: Data loading and record utilities

**Key Functions**:
- `load_dataset()`: Load JSONL file
- `read_jsonl()`: Stream JSONL records
- `write_jsonl()`: Write JSONL records
- `doc_index_from_record()`: Build doc ID index
- `support_doc_ids_from_notes()`: Extract supporting docs
- `gold_answerable_from_notes()`: Check answerability
- `get_model_output()`: Extract model answer
- `get_gold_answer()`: Extract gold answer

## 🔧 Scripts

### `scripts/setup.sh`
**Purpose**: Automated installation and setup

**Actions**:
1. Check Python version (≥3.8)
2. Create virtual environment
3. Install dependencies
4. Download NLTK data
5. Create directory structure
6. Generate .env template
7. Create default config

### `scripts/run_eval.sh`
**Purpose**: Interactive evaluation wizard

**Flow**:
1. Activate virtual environment
2. Load environment variables
3. List available input files
4. Select judge committee
5. Configure sample limits
6. Set verbosity
7. Confirm and run evaluation
8. Display results

### `run_evaluation.py`
**Purpose**: Main evaluation CLI

**Arguments**:
- `--input`: Input JSONL path (required)
- `--output-dir`: Output directory (default: outputs)
- `--committee`: Judge preset (default/conservative/none)
- `--config`: YAML config file (optional)
- `--batch-size`: Batch size (default: 50)
- `--max-samples`: Limit samples (for testing)
- `--verbose`: Enable verbose logging

**Example Usage**:
```bash
python run_evaluation.py \
    --input data/my_data.jsonl \
    --committee default \
    --max-samples 10 \
    --verbose
```

### `test_installation.py`
**Purpose**: Validate installation

**Tests**:
1. Python version check
2. Import core dependencies
3. Import CATS modules
4. Directory structure
5. Environment configuration
6. NLTK data
7. Example data loading

## 📊 Data Flow

```
Input JSONL
    ↓
load_dataset()
    ↓
EnhancedEvaluator
    ↓
[For each sample]
    ↓
    ├─→ Extract query, docs, conflict type
    ├─→ committee_behavior_adherence()
    │       ↓
    │   [All judges vote in parallel]
    │       ↓
    │   Aggregate votes → Decision
    │
    ├─→ enhanced_factual_grounding()
    │       ↓
    │   Extract claims
    │       ↓
    │   [Verify each claim vs docs]
    │       ↓
    │   Grounding ratio
    │
    └─→ enhanced_single_truth_recall()
            ↓
        [Committee checks gold answer]
            ↓
        Recall score
    ↓
Aggregate results
    ↓
Generate reports
    ↓
    ├─→ outputs/eval_report.md
    ├─→ outputs/detailed_results.json
    └─→ logs/cats_eval.log
```

## 🔌 Extension Points

### Adding a Custom Judge

```python
from rag_eval.config import JudgeModelConfig, APIProvider

custom_judge = JudgeModelConfig(
    model_id="your-model-id",
    provider=APIProvider.OPENROUTER,
    temperature=0.0,
    max_tokens=500,
    cost_per_1k_input=0.50,
    cost_per_1k_output=1.50,
    priority=2,
    api_key_env="YOUR_API_KEY_ENV",
    base_url="https://api.provider.com/v1"
)

# Add to committee
committee.judges.append(custom_judge)
```

### Custom Voting Strategy

Modify `JudgeCommittee._aggregate_votes()` in `judge_committee.py`:

```python
def _custom_vote(self, responses: List[JudgeResponse]) -> CommitteeDecision:
    # Your voting logic here
    pass
```

### Custom Metric

Add to `conflict_eval.py`:

```python
async def custom_metric(
    committee: JudgeCommittee,
    query: str,
    answer: str,
    docs: List[Dict]
) -> Dict[str, Any]:
    # Your evaluation logic
    pass
```

Then integrate into `EnhancedEvaluator._evaluate_single_sample()`.

## 📝 Input Format Specification

### Minimal Required Fields
```json
{
  "id": "string",
  "query": "string",
  "retrieved_docs": [
    {
      "doc_id": "string",
      "snippet": "string"
    }
  ],
  "conflict_category_id": 1-5,
  "model_output": "string"
}
```

### Full Schema
```json
{
  "id": "string",
  "query": "string",
  "retrieved_docs": [
    {
      "doc_id": "string",
      "title": "string (optional)",
      "url": "string (optional)",
      "snippet": "string",
      "text": "string (optional)",
      "date": "string (optional)"
    }
  ],
  "per_doc_notes": [
    {
      "doc_id": "string",
      "verdict": "supports|partially supports|contradicts|irrelevant",
      "key_fact": "string (optional)",
      "quote": "string (optional)"
    }
  ],
  "conflict_category_id": 1-5,
  "conflict_type": "string (optional)",
  "conflict_reason": "string (optional)",
  "model_output": "string",
  "gold_answer": "string|list (optional)"
}
```

## 🎯 Output Format

### Markdown Report Structure
```markdown
# CATS v2.0 Evaluation Report

## Overall Conflict-Aware Metrics
- Total Samples: N
- F1_GR: 0.XXX
- Behavior Adherence: 0.XXX
- Factual Grounding: 0.XXX
- Single-Truth Recall: 0.XXX
- CATS Score: 0.XXX

## Per Conflict Type Breakdown
### Type 1: No Conflict
- Samples: N
- Metrics...

## Cost Summary
- Total Cost: $X.XX
- Decisions Made: N
- Avg Cost/Decision: $X.XXXXXX
```

### JSON Results Structure
```json
{
  "summary": {
    "conflict_overall": {...},
    "conflict_per_type": {...},
    "cost_summary": {...}
  },
  "per_sample": [
    {
      "sample_id": "...",
      "conflict_type": 1,
      "f1_gr": 1.0,
      "behavior_score": 0.8,
      "behavior_details": {...},
      "factual_grounding_score": 0.9,
      "factual_grounding_details": {...},
      "single_truth_recall_score": 1.0,
      "single_truth_recall_details": {...}
    }
  ]
}
```

---

**For usage examples, see README.md and QUICKSTART.md**
