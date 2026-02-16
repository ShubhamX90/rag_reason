#!/bin/bash
# Master Execution Script - Run All Experiments
# ==============================================
# Executes all experiment families in parallel or sequential mode
#
# Usage: bash run_all_experiments.sh <model_path> [--sequential]

set -e

MODEL_PATH=$1
MODE=$2
SPLIT="test"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [--sequential]"
    echo ""
    echo "Modes:"
    echo "  Default: Parallel execution (launches all in tmux)"
    echo "  --sequential: Run one after another"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/Llama-3.1-8B-Instruct"
    echo "  $0 /path/to/model --sequential"
    exit 1
fi

echo "=============================================="
echo "MASTER EXECUTION - ALL EXPERIMENTS"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Split: $SPLIT"
echo "Mode: ${MODE:-parallel}"
echo ""

# Determine execution mode
if [ "$MODE" == "--sequential" ]; then
    RUN_FLAG=""
    echo "Running sequentially (this will take ~80-100 hours total)"
else
    RUN_FLAG="--tmux"
    echo "Running in parallel (tmux sessions)"
    echo "Estimated time: ~8-10 hours (limited by slowest experiment)"
fi

echo ""
echo "Experiments to run:"
echo "  - Main Stagewise: e2e, oracle1, oracle2, oracle3 (4)"
echo "  - Monolithic: e2e, oracle1, oracle2, oracle3 (4)"
echo "  - Ablation Type1: e2e, oracle (2)"
echo "  - Ablation Type2: e2e, oracle (2)"
echo "  - Simple RAG (1)"
echo "  Total: 13 experiments"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."

# ============================================
# Main Stagewise (4 experiments)
# ============================================
echo ""
echo "=========================================="
echo "Launching Main Stagewise Experiments..."
echo "=========================================="

for level in e2e oracle1 oracle2 oracle3; do
    echo "Starting main_stagewise_${level}..."
    bash scripts/run_main_stagewise.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
    if [ "$MODE" == "--sequential" ]; then
        echo "✓ Completed main_stagewise_${level}"
    else
        sleep 2  # Brief pause between tmux sessions
    fi
done

# ============================================
# Monolithic (4 experiments)
# ============================================
echo ""
echo "=========================================="
echo "Launching Monolithic Experiments..."
echo "=========================================="

for level in e2e oracle1 oracle2 oracle3; do
    echo "Starting monolithic_${level}..."
    bash scripts/run_monolithic.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
    if [ "$MODE" == "--sequential" ]; then
        echo "✓ Completed monolithic_${level}"
    else
        sleep 2
    fi
done

# ============================================
# Ablation Type1 (2 experiments)
# ============================================
echo ""
echo "=========================================="
echo "Launching Ablation Type1 Experiments..."
echo "=========================================="

for level in e2e oracle; do
    echo "Starting ablation_type1_${level}..."
    bash scripts/run_ablation_type1.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
    if [ "$MODE" == "--sequential" ]; then
        echo "✓ Completed ablation_type1_${level}"
    else
        sleep 2
    fi
done

# ============================================
# Ablation Type2 (2 experiments)
# ============================================
echo ""
echo "=========================================="
echo "Launching Ablation Type2 Experiments..."
echo "=========================================="

for level in e2e oracle; do
    echo "Starting ablation_type2_${level}..."
    bash scripts/run_ablation_type2.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
    if [ "$MODE" == "--sequential" ]; then
        echo "✓ Completed ablation_type2_${level}"
    else
        sleep 2
    fi
done

# ============================================
# Simple RAG (1 experiment)
# ============================================
echo ""
echo "=========================================="
echo "Launching Simple RAG Baseline..."
echo "=========================================="

echo "Starting simple_rag..."
bash scripts/run_simple_rag.sh $SPLIT $MODEL_PATH $RUN_FLAG
if [ "$MODE" == "--sequential" ]; then
    echo "✓ Completed simple_rag"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "LAUNCH COMPLETE!"
echo "=============================================="

if [ "$MODE" != "--sequential" ]; then
    echo ""
    echo "All experiments launched in tmux sessions:"
    echo ""
    tmux ls 2>/dev/null || echo "(No active sessions - they may have completed)"
    echo ""
    echo "To monitor:"
    echo "  - List sessions: tmux ls"
    echo "  - Attach to session: tmux attach -t <session_name>"
    echo "  - Detach from session: Ctrl+B, then D"
    echo "  - Kill session: tmux kill-session -t <session_name>"
    echo ""
    echo "Example:"
    echo "  tmux attach -t main_stagewise_e2e_test"
    echo ""
    echo "To check status:"
    echo "  bash scripts/check_status.sh"
    echo ""
else
    echo ""
    echo "✓ All experiments completed successfully!"
    echo ""
    echo "To view results:"
    echo "  bash scripts/view_results.sh"
    echo ""
fi

echo "Outputs located in: outputs/"
echo "Reports located in: outputs/*/test/reports/"
echo "=============================================="