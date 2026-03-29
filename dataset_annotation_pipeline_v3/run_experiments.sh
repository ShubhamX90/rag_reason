#!/usr/bin/env bash
# =============================================================================
#  run_experiments.sh  —  Parser fix verification experiments
# =============================================================================
#  Runs 12 experiments: 2 datasets × 2 strategies × 3 models
#  - Datasets: conflicts_normalized.jsonl, refusals_normalized.jsonl
#  - Strategies: 3-stage (batch for Anthropic, async for OpenRouter), monolithic (same)
#  - Models: claude-sonnet-4-6, qwen/qwen-2.5-72b-instruct, deepseek/deepseek-v3.2
#  - Limit: 5 samples each
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
TS=$(date +"%Y%m%d_%H%M%S")
LIMIT=3
CONCURRENCY=5

# Output directory for this experiment batch
OUTDIR="data/experiments_${TS}"
mkdir -p "$OUTDIR"

echo "=================================================="
echo "  Parser Fix Verification Experiments"
echo "  Timestamp: ${TS}"
echo "  Limit: ${LIMIT} samples each"
echo "  Output dir: ${OUTDIR}"
echo "=================================================="
echo

# ──────────────────────────────────────────────────────
# Helper: run 3-stage pipeline
# ──────────────────────────────────────────────────────
run_3stage() {
    local dataset="$1" provider="$2" model="$3" tag="$4"
    local s1_out="${OUTDIR}/${tag}_stage1.jsonl"
    local s2_out="${OUTDIR}/${tag}_stage2.jsonl"
    local s3_out="${OUTDIR}/${tag}_stage3.jsonl"

    echo "──── 3-stage: ${tag} ────"
    echo "  dataset=$dataset  provider=$provider  model=$model"

    if [[ "$provider" == "anthropic" ]]; then
        # Use batch mode for Anthropic
        local batch_dir="${OUTDIR}/.batch_ids"
        mkdir -p "$batch_dir"

        echo "  → Stage 1 (batch)..."
        $PYTHON scripts/run_stage1_batch.py \
            --input "$dataset" --output "$s1_out" \
            --provider "$provider" --model "$model" \
            --batch-id-file "${batch_dir}/${tag}_s1.txt" \
            --limit $LIMIT

        echo "  → Stage 2 (batch)..."
        $PYTHON scripts/run_stage2_batch.py \
            --input "$s1_out" --output "$s2_out" \
            --provider "$provider" --model "$model" \
            --batch-id-file "${batch_dir}/${tag}_s2.txt" \
            --limit $LIMIT

        echo "  → Stage 3 (batch)..."
        $PYTHON scripts/run_stage3_batch.py \
            --input "$s2_out" --output "$s3_out" \
            --provider "$provider" --model "$model" \
            --batch-id-file "${batch_dir}/${tag}_s3.txt" \
            --limit $LIMIT
    else
        # Use async mode for OpenRouter
        echo "  → Stage 1 (async)..."
        $PYTHON scripts/run_stage1_async.py \
            --input "$dataset" --output "$s1_out" \
            --provider "$provider" --model "$model" \
            --concurrency $CONCURRENCY --limit $LIMIT

        echo "  → Stage 2 (async)..."
        $PYTHON scripts/run_stage2_async.py \
            --input "$s1_out" --output "$s2_out" \
            --provider "$provider" --model "$model" \
            --concurrency $CONCURRENCY --limit $LIMIT

        echo "  → Stage 3 (async)..."
        $PYTHON scripts/run_stage3_async.py \
            --input "$s2_out" --output "$s3_out" \
            --provider "$provider" --model "$model" \
            --concurrency $CONCURRENCY --limit $LIMIT
    fi

    echo "  ✅ 3-stage complete: $s3_out"
    echo
}

# ──────────────────────────────────────────────────────
# Helper: run monolithic pipeline
# ──────────────────────────────────────────────────────
run_monolithic() {
    local dataset="$1" provider="$2" model="$3" tag="$4"
    local mono_out="${OUTDIR}/${tag}_monolithic.jsonl"

    echo "──── Monolithic: ${tag} ────"
    echo "  dataset=$dataset  provider=$provider  model=$model"

    if [[ "$provider" == "anthropic" ]]; then
        # Use batch mode for Anthropic
        local batch_dir="${OUTDIR}/.batch_ids"
        mkdir -p "$batch_dir"

        echo "  → Monolithic (batch)..."
        $PYTHON scripts/run_monolithic_batch.py \
            --input "$dataset" --output "$mono_out" \
            --provider "$provider" --model "$model" \
            --batch-id-file "${batch_dir}/${tag}_mono.txt" \
            --limit $LIMIT
    else
        # Use async mode for OpenRouter
        echo "  → Monolithic (async)..."
        $PYTHON scripts/run_monolithic_async.py \
            --input "$dataset" --output "$mono_out" \
            --provider "$provider" --model "$model" \
            --concurrency $CONCURRENCY --limit $LIMIT
    fi

    echo "  ✅ Monolithic complete: $mono_out"
    echo
}

# ──────────────────────────────────────────────────────
#  Run all 12 experiments
# ──────────────────────────────────────────────────────

CONFLICTS="data/normalized/conflicts_normalized.jsonl"
REFUSALS="data/normalized/refusals_normalized.jsonl"

# ── Sonnet 4.6 (Anthropic batch) ──
run_3stage     "$CONFLICTS" "anthropic" "claude-sonnet-4-6" "conflicts_sonnet_3stage"
run_monolithic "$CONFLICTS" "anthropic" "claude-sonnet-4-6" "conflicts_sonnet"
run_3stage     "$REFUSALS"  "anthropic" "claude-sonnet-4-6" "refusals_sonnet_3stage"
run_monolithic "$REFUSALS"  "anthropic" "claude-sonnet-4-6" "refusals_sonnet"

# ── Qwen 72B (OpenRouter async) ──
run_3stage     "$CONFLICTS" "openrouter" "qwen/qwen-2.5-72b-instruct" "conflicts_qwen72b_3stage"
run_monolithic "$CONFLICTS" "openrouter" "qwen/qwen-2.5-72b-instruct" "conflicts_qwen72b"
run_3stage     "$REFUSALS"  "openrouter" "qwen/qwen-2.5-72b-instruct" "refusals_qwen72b_3stage"
run_monolithic "$REFUSALS"  "openrouter" "qwen/qwen-2.5-72b-instruct" "refusals_qwen72b"

# ── DeepSeek v3.2 (OpenRouter async) ──
run_3stage     "$CONFLICTS" "openrouter" "deepseek/deepseek-v3.2" "conflicts_deepseek_3stage"
run_monolithic "$CONFLICTS" "openrouter" "deepseek/deepseek-v3.2" "conflicts_deepseek"
run_3stage     "$REFUSALS"  "openrouter" "deepseek/deepseek-v3.2" "refusals_deepseek_3stage"
run_monolithic "$REFUSALS"  "openrouter" "deepseek/deepseek-v3.2" "refusals_deepseek"


# ──────────────────────────────────────────────────────
#  Summary: check malformed rates across all outputs
# ──────────────────────────────────────────────────────

echo
echo "=================================================="
echo "  RESULTS SUMMARY"
echo "=================================================="

$PYTHON - "$OUTDIR" <<'PYEOF'
import json, sys, os
from pathlib import Path

outdir = Path(sys.argv[1])
files = sorted(outdir.glob("*.jsonl"))

if not files:
    print("No output files found!")
    sys.exit(0)

print(f"{'File':<50s} {'Total':>5s} {'OK':>5s} {'Abst':>5s} {'Malf':>5s}")
print("-" * 75)

for f in files:
    total = ok = abstained = malformed = 0
    with open(f) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except:
                continue
            total += 1
            er = rec.get("expected_response", {})
            errs = rec.get("_stage3_errors", rec.get("_errors", []))
            if errs:
                malformed += 1
            elif isinstance(er, dict) and er.get("abstain"):
                abstained += 1
            else:
                ok += 1
    print(f"{f.name:<50s} {total:>5d} {ok:>5d} {abstained:>5d} {malformed:>5d}")

print()
PYEOF

echo "Done! All experiment outputs are in: ${OUTDIR}"
