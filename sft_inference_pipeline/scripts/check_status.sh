#!/bin/bash
# Status Checker - Monitor All Experiments
# =========================================

echo "=============================================="
echo "EXPERIMENT STATUS CHECKER"
echo "=============================================="
echo "Checking all experiment outputs..."
echo ""

check_experiment() {
    local family=$1
    local level=$2
    local split=$3
    local file_pattern=$4
    
    if [ -f "$file_pattern" ]; then
        count=$(wc -l < "$file_pattern" 2>/dev/null || echo "0")
        echo "  $family $level: $count/54"
    else
        echo "  $family $level: NOT STARTED"
    fi
}

# Main Stagewise
echo "Main Stagewise:"
for level in e2e oracle1 oracle2 oracle3; do
    check_experiment "main" "$level" "test" "outputs/main_stagewise/${level}/test/combined.sanitized.jsonl"
done

echo ""
echo "Monolithic:"
for level in e2e oracle1 oracle2 oracle3; do
    check_experiment "mono" "$level" "test" "outputs/monolithic/${level}/test/baseline_${level}_test.sanitized.jsonl"
done

echo ""
echo "Ablation Type1:"
for level in e2e oracle; do
    check_experiment "abl1" "$level" "test" "outputs/ablations/type1_${level}/test/combined.sanitized.jsonl"
done

echo ""
echo "Ablation Type2:"
for level in e2e oracle; do
    check_experiment "abl2" "$level" "test" "outputs/ablations/type2_${level}/test/combined.sanitized.jsonl"
done

echo ""
echo "Simple RAG:"
check_experiment "rag" "baseline" "test" "outputs/baselines/simple_rag_test.sanitized.jsonl"

echo ""
echo "=============================================="
echo "TMUX Sessions:"
echo "=============================================="
tmux ls 2>/dev/null || echo "No active tmux sessions"

echo ""
echo "=============================================="