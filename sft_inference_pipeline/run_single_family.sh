#!/bin/bash
# Helper Script: Run a Single Experiment Family
# ==============================================
# Simplifies running just one family (e.g., only ablation_type1)
#
# Usage: bash run_single_family.sh <family_name> <model_path> [--sequential]

set -e

FAMILY=$1
MODEL_PATH=$2
MODE=$3
SPLIT="test"

# ============================================
# Validation
# ============================================

if [ -z "$FAMILY" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <family_name> <model_path> [--sequential]"
    echo ""
    echo "Available families:"
    echo "  main_stagewise   - 4 experiments (e2e, oracle1, oracle2, oracle3)"
    echo "  monolithic       - 4 experiments (e2e, oracle1, oracle2, oracle3)"
    echo "  ablation_type1   - 2 experiments (e2e, oracle)"
    echo "  ablation_type2   - 2 experiments (e2e, oracle)"
    echo "  simple_rag       - 1 experiment"
    echo ""
    echo "Modes:"
    echo "  Default: Parallel execution (launches all in tmux)"
    echo "  --sequential: Run one after another"
    echo ""
    echo "Examples:"
    echo "  $0 ablation_type1 /path/to/model              # Parallel"
    echo "  $0 main_stagewise /path/to/model --sequential # Sequential"
    echo ""
    echo "This is equivalent to manually running:"
    echo "  bash scripts/run_ablation_type1.sh e2e test /path/to/model --tmux"
    echo "  bash scripts/run_ablation_type1.sh oracle test /path/to/model --tmux"
    exit 1
fi

# Validate family name
VALID_FAMILIES=("main_stagewise" "monolithic" "ablation_type1" "ablation_type2" "simple_rag")
if [[ ! " ${VALID_FAMILIES[@]} " =~ " ${FAMILY} " ]]; then
    echo "Error: Invalid family name '$FAMILY'"
    echo "Must be one of: ${VALID_FAMILIES[@]}"
    exit 1
fi

# Determine run flag
if [ "$MODE" == "--sequential" ]; then
    RUN_FLAG=""
    echo "Mode: Sequential"
else
    RUN_FLAG="--tmux"
    echo "Mode: Parallel (tmux)"
fi

echo "=============================================="
echo "Running Family: $FAMILY"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Split: $SPLIT"
echo ""

# ============================================
# Family-specific logic
# ============================================

case $FAMILY in
    main_stagewise)
        echo "Experiments to run: e2e, oracle1, oracle2, oracle3 (4 total)"
        echo "Estimated time: ~8-10 hours (parallel) or ~30-40 hours (sequential)"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        
        for level in e2e oracle1 oracle2 oracle3; do
            echo ""
            echo "Starting main_stagewise_${level}..."
            bash scripts/run_main_stagewise.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
            if [ "$MODE" == "--sequential" ]; then
                echo "✓ Completed main_stagewise_${level}"
            else
                sleep 2
            fi
        done
        ;;
    
    monolithic)
        echo "Experiments to run: e2e, oracle1, oracle2, oracle3 (4 total)"
        echo "Estimated time: ~8-10 hours (parallel) or ~30-40 hours (sequential)"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        
        for level in e2e oracle1 oracle2 oracle3; do
            echo ""
            echo "Starting monolithic_${level}..."
            bash scripts/run_monolithic.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
            if [ "$MODE" == "--sequential" ]; then
                echo "✓ Completed monolithic_${level}"
            else
                sleep 2
            fi
        done
        ;;
    
    ablation_type1)
        echo "Experiments to run: e2e, oracle (2 total)"
        echo "Estimated time: ~5-6 hours (parallel) or ~10-12 hours (sequential)"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        
        for level in e2e oracle; do
            echo ""
            echo "Starting ablation_type1_${level}..."
            bash scripts/run_ablation_type1.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
            if [ "$MODE" == "--sequential" ]; then
                echo "✓ Completed ablation_type1_${level}"
            else
                sleep 2
            fi
        done
        ;;
    
    ablation_type2)
        echo "Experiments to run: e2e, oracle (2 total)"
        echo "Estimated time: ~5-6 hours (parallel) or ~10-12 hours (sequential)"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        
        for level in e2e oracle; do
            echo ""
            echo "Starting ablation_type2_${level}..."
            bash scripts/run_ablation_type2.sh $level $SPLIT $MODEL_PATH $RUN_FLAG
            if [ "$MODE" == "--sequential" ]; then
                echo "✓ Completed ablation_type2_${level}"
            else
                sleep 2
            fi
        done
        ;;
    
    simple_rag)
        echo "Experiment: simple_rag (1 total)"
        echo "Estimated time: ~2-3 hours"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        
        echo ""
        echo "Starting simple_rag..."
        bash scripts/run_simple_rag.sh $SPLIT $MODEL_PATH $RUN_FLAG
        if [ "$MODE" == "--sequential" ]; then
            echo "✓ Completed simple_rag"
        fi
        ;;
esac

# ============================================
# Summary
# ============================================

echo ""
echo "=============================================="
echo "LAUNCH COMPLETE!"
echo "=============================================="

if [ "$MODE" != "--sequential" ]; then
    echo ""
    echo "Launched tmux sessions:"
    echo ""
    tmux ls 2>/dev/null || echo "(No active sessions)"
    echo ""
    echo "To monitor:"
    echo "  - List sessions: tmux ls"
    echo "  - Attach: tmux attach -t <session_name>"
    echo "  - Detach: Ctrl+B, then D"
    echo ""
    echo "Example:"
    echo "  tmux attach -t ${FAMILY}_e2e_${SPLIT}"
    echo ""
else
    echo ""
    echo "✓ All experiments in '$FAMILY' completed!"
    echo ""
fi

echo "Outputs located in: outputs/${FAMILY}/"
echo "=============================================="