#!/bin/bash
# Interactive Launcher - Choose Which Family to Run
# ==================================================

set -e

MODEL_PATH=$1

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path>"
    echo ""
    echo "Example: $0 /path/to/Llama-3.1-8B-Instruct"
    echo ""
    echo "This interactive script helps you select which experiment"
    echo "family to run without needing to remember all the commands."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

SPLIT="test"

echo "=============================================="
echo "Interactive Experiment Launcher"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Split: $SPLIT"
echo ""

# ============================================
# Menu: Select Family
# ============================================

echo "Select experiment family to run:"
echo ""
echo "1) main_stagewise   (4 experiments: e2e, oracle1, oracle2, oracle3)"
echo "2) monolithic       (4 experiments: e2e, oracle1, oracle2, oracle3)"
echo "3) ablation_type1   (2 experiments: e2e, oracle)"
echo "4) ablation_type2   (2 experiments: e2e, oracle)"
echo "5) simple_rag       (1 experiment)"
echo ""
echo "6) Run ALL families (13 experiments total)"
echo "7) Custom - select specific experiments manually"
echo ""
read -p "Enter choice (1-7): " FAMILY_CHOICE

case $FAMILY_CHOICE in
    1) FAMILY="main_stagewise" ;;
    2) FAMILY="monolithic" ;;
    3) FAMILY="ablation_type1" ;;
    4) FAMILY="ablation_type2" ;;
    5) FAMILY="simple_rag" ;;
    6) 
        echo ""
        read -p "Run all experiments in parallel (p) or sequential (s)? [p/s]: " MODE_CHOICE
        if [ "$MODE_CHOICE" == "s" ]; then
            bash scripts/run_all_experiments.sh $MODEL_PATH --sequential
        else
            bash scripts/run_all_experiments.sh $MODEL_PATH
        fi
        exit 0
        ;;
    7)
        # Custom mode
        echo ""
        echo "Custom Mode - Select specific experiments"
        echo ""
        
        # Collect selections
        SELECTED_EXPERIMENTS=()
        
        echo "Main Stagewise:"
        read -p "  Run e2e? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("main_stagewise:e2e")
        read -p "  Run oracle1? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("main_stagewise:oracle1")
        read -p "  Run oracle2? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("main_stagewise:oracle2")
        read -p "  Run oracle3? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("main_stagewise:oracle3")
        
        echo ""
        echo "Monolithic:"
        read -p "  Run e2e? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("monolithic:e2e")
        read -p "  Run oracle1? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("monolithic:oracle1")
        read -p "  Run oracle2? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("monolithic:oracle2")
        read -p "  Run oracle3? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("monolithic:oracle3")
        
        echo ""
        echo "Ablation Type1:"
        read -p "  Run e2e? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("ablation_type1:e2e")
        read -p "  Run oracle? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("ablation_type1:oracle")
        
        echo ""
        echo "Ablation Type2:"
        read -p "  Run e2e? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("ablation_type2:e2e")
        read -p "  Run oracle? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("ablation_type2:oracle")
        
        echo ""
        read -p "Run Simple RAG? [y/n]: " && [ "$REPLY" == "y" ] && SELECTED_EXPERIMENTS+=("simple_rag:single")
        
        # Launch selected experiments
        echo ""
        echo "=============================================="
        echo "Launching ${#SELECTED_EXPERIMENTS[@]} experiment(s)..."
        echo "=============================================="
        
        for exp in "${SELECTED_EXPERIMENTS[@]}"; do
            IFS=':' read -r family level <<< "$exp"
            echo ""
            echo "Starting ${family}_${level}..."
            
            if [ "$family" == "simple_rag" ]; then
                bash scripts/run_simple_rag.sh $SPLIT $MODEL_PATH --tmux
            else
                bash scripts/run_${family}.sh $level $SPLIT $MODEL_PATH --tmux
            fi
            
            sleep 2
        done
        
        echo ""
        echo "✓ All selected experiments launched!"
        tmux ls 2>/dev/null || echo "(No active sessions)"
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

# ============================================
# Menu: Select Mode
# ============================================

echo ""
echo "How do you want to run '$FAMILY'?"
echo ""
echo "1) Parallel (tmux)   - All experiments at once, ~8-10 hours total"
echo "2) Sequential        - One after another, ~30-40 hours total"
echo ""
read -p "Enter choice (1-2): " MODE_CHOICE

case $MODE_CHOICE in
    1) MODE="" ;;
    2) MODE="--sequential" ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

# ============================================
# Confirmation
# ============================================

echo ""
echo "=============================================="
echo "Ready to Launch"
echo "=============================================="
echo "Family: $FAMILY"
echo "Mode: ${MODE:-parallel (tmux)}"
echo "Model: $MODEL_PATH"
echo ""

if [ -z "$MODE" ]; then
    echo "This will create tmux sessions that you can attach/detach from."
    echo "Sessions will survive SSH disconnects."
else
    echo "This will run sequentially in the foreground."
    echo "Do NOT close your terminal until complete!"
fi

echo ""
read -p "Proceed? [y/n]: " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

# ============================================
# Launch
# ============================================

echo ""
bash run_single_family.sh $FAMILY $MODEL_PATH $MODE

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="

if [ -z "$MODE" ]; then
    echo ""
    echo "Tmux sessions running. To monitor:"
    echo "  tmux ls                                  # List sessions"
    echo "  tmux attach -t <session_name>            # Attach"
    echo "  bash scripts/check_status.sh             # Check progress"
    echo ""
fi