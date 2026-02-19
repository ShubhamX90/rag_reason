#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh  —  Interactive RAG Annotation Pipeline Runner
# =============================================================================
#
#  Usage:  bash run_pipeline.sh
#          (Run from the project root directory)
#
#  Guides you through:
#    1. Annotation strategy  (3-stage OR monolithic)
#    2. LLM provider         (Anthropic / OpenAI / OpenRouter)
#    3. Execution mode       (async OR batch)
#    4. Data paths, concurrency, limits
#    5. Running the appropriate Python script(s)
#
# =============================================================================

set -euo pipefail

# ─── Colours ──────────────────────────────────────────────────────────────────
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}━━━  $*  ━━━${RESET}\n"; }
prompt()  { echo -en "${BOLD}${YELLOW}  ▶  $*: ${RESET}"; }
divider() { echo -e "${CYAN}──────────────────────────────────────────────────${RESET}"; }

# ─── Root detection ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "prompts/system_stage1.txt" ]]; then
    error "Must be run from the project root directory (where prompts/ lives)."
fi

# ─── Python detection ─────────────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"; break
    fi
done
[[ -z "$PYTHON" ]] && error "Python not found. Install Python 3.9+."
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Using $PYTHON (v$PY_VERSION)"

# ─── Dependency check ─────────────────────────────────────────────────────────
check_deps() {
    local missing=()
    for pkg in anthropic openai tqdm; do
        $PYTHON -c "import $pkg" 2>/dev/null || missing+=("$pkg")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        warn "Missing packages: ${missing[*]}"
        prompt "Install now? [Y/n]"
        read -r ans
        if [[ "${ans:-Y}" =~ ^[Yy]$ ]]; then
            pip install "${missing[@]}" --quiet
        else
            warn "Proceeding without installing (may fail)."
        fi
    fi
}

# ─── API Key helpers ──────────────────────────────────────────────────────────
ensure_anthropic_key() {
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && [[ ! -f "$HOME/.anthropic_key" ]]; then
        warn "ANTHROPIC_API_KEY not set and ~/.anthropic_key not found."
        prompt "Enter your Anthropic API key (or press Enter to skip)"
        read -r -s key; echo
        if [[ -n "$key" ]]; then
            export ANTHROPIC_API_KEY="$key"
            success "Key set for this session."
        else
            warn "Continuing without key — will fail if key is missing."
        fi
    else
        success "Anthropic API key found."
    fi
}

ensure_openai_key() {
    if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ ! -f "$HOME/.openai_key" ]]; then
        warn "OPENAI_API_KEY not set and ~/.openai_key not found."
        prompt "Enter your OpenAI API key (or press Enter to skip)"
        read -r -s key; echo
        if [[ -n "$key" ]]; then
            export OPENAI_API_KEY="$key"
            success "Key set for this session."
        else
            warn "Continuing without key — will fail if key is missing."
        fi
    else
        success "OpenAI API key found."
    fi
}

ensure_openrouter_key() {
    if [[ -z "${OPENROUTER_API_KEY:-}" ]] && [[ ! -f "$HOME/.openrouter_key" ]]; then
        warn "OPENROUTER_API_KEY not set and ~/.openrouter_key not found."
        prompt "Enter your OpenRouter API key (or press Enter to skip)"
        read -r -s key; echo
        if [[ -n "$key" ]]; then
            export OPENROUTER_API_KEY="$key"
            success "Key set for this session."
        fi
    else
        success "OpenRouter API key found."
    fi
}

# ─── Input helpers ────────────────────────────────────────────────────────────
choose() {
    local var="$1"; shift
    local msg="$1"; shift
    local opts=("$@")
    echo
    echo -e "${BOLD}  $msg${RESET}"
    for i in "${!opts[@]}"; do
        echo -e "    ${CYAN}$((i+1))${RESET}) ${opts[$i]}"
    done
    local sel
    while true; do
        prompt "Enter choice [1-${#opts[@]}]"
        read -r sel
        if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel <= ${#opts[@]} )); then
            eval "$var=\"${opts[$((sel-1))]}\""
            return
        fi
        warn "Invalid choice. Try again."
    done
}

ask() {
    local var="$1" msg="$2" default="$3"
    prompt "$msg [${default}]"
    read -r val
    eval "$var=\"${val:-$default}\""
}

# ─── OpenRouter model slug validation ────────────────────────────────────────
# Expands known Qwen shorthands; enforces provider/model format for everything else.
validate_openrouter_model() {
    local model_lc
    model_lc=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')
    case "$model_lc" in
        qwen|qwen2.5|qwen-72b|qwen72b|qwen2.5-72b|qwen2.5-72b-instruct|"qwen/qwen2.5-72b-instruct")
            MODEL="qwen/qwen-2.5-72b-instruct"
            info "Model expanded to canonical slug: $MODEL" ;;
        qwen-7b|qwen7b|qwen2.5-7b|qwen2.5-7b-instruct|"qwen/qwen2.5-7b-instruct")
            MODEL="qwen/qwen-2.5-7b-instruct"
            info "Model expanded to canonical slug: $MODEL" ;;
        *)
            # Any slug with '/' is fine — passes through to resolve_openrouter_model() in Python
            if [[ "$MODEL" != *"/"* ]]; then
                warn "OpenRouter model names must use the format 'provider/model-name'"
                warn "e.g.:  qwen/qwen-2.5-72b-instruct   mistralai/mistral-7b-instruct"
                warn "       meta-llama/llama-3.1-70b-instruct"
                warn "You entered: '$MODEL'"
                prompt "Enter corrected model slug (Enter = use default qwen/qwen-2.5-72b-instruct)"
                read -r fixed_model
                MODEL="${fixed_model:-qwen/qwen-2.5-72b-instruct}"
                info "Using model: $MODEL"
            fi
            ;;
    esac
}

# ─── Stage-1 error guard ─────────────────────────────────────────────────────
# Aborts the pipeline if ALL Stage-1 records failed (wrong key, bad model, etc.)
check_stage1_errors() {
    local outfile="$1"
    if [[ ! -f "$outfile" ]]; then
        warn "Stage-1 output not found: $outfile — skipping error check."
        return
    fi
    local total error_count
    total=$(wc -l < "$outfile" | tr -d ' ')
    [[ "$total" -eq 0 ]] && error "Stage-1 output is empty. Aborting pipeline."

    error_count=$($PYTHON - "$outfile" << 'PYEOF'
import json, sys
path = sys.argv[1]
total, all_err = 0, 0
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        total += 1
        try:
            rec   = json.loads(line)
            notes = rec.get("per_doc_notes", [])
            if notes and all(n.get("_error") for n in notes):
                all_err += 1
        except Exception:
            pass
print(all_err)
PYEOF
)
    if [[ "$error_count" -eq "$total" ]] && [[ "$total" -gt 0 ]]; then
        error "Stage-1 FAILED: all $total record(s) returned API errors.
  Common causes:
    • Wrong model name (e.g. 'qwen' instead of 'qwen/qwen-2.5-72b-instruct')
    • Invalid or missing API key
    • Network / quota issue
  Check _error fields in: $outfile
  Fix the issue, delete the output file, and rerun."
    elif [[ "$error_count" -gt 0 ]]; then
        warn "Stage-1: $error_count/$total record(s) had all-docs-failed errors. Proceeding — check output."
    else
        success "Stage-1 error check passed ($total records, 0 fully-failed)."
    fi
}

# ─── Command runner ──────────────────────────────────────────────────────────
run_cmd() {
    echo
    info "Running: $*"
    divider
    $PYTHON "$@"
}

# ─── Validation runner ────────────────────────────────────────────────────────
# Runs a validator script; warns on failure but never aborts the pipeline.
run_validation() {
    echo
    info "Validating: $*"
    divider
    $PYTHON "$@" || warn "Validation script exited with errors — check the report for details."
}

build_limit_flag() {
    [[ -n "${LIMIT:-}" ]] && echo "--limit $LIMIT" || echo ""
}

# =============================================================================
#  MAIN
# =============================================================================

clear
echo -e "${BOLD}${CYAN}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   RAG Dataset Annotation Pipeline                ║"
echo "  ║   Anthropic · OpenAI · OpenRouter                ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${RESET}"

check_deps

# ── Step 1: Annotation Strategy ───────────────────────────────────────────────
header "Step 1: Annotation Strategy"
echo -e "  ${BOLD}3-Stage${RESET}  : Stage-1 (per-doc) → Stage-2 (conflict) → Stage-3 (response)"
echo -e "            Best quality; full traceability; more API calls."
echo
echo -e "  ${BOLD}Monolithic${RESET}: All stages in ONE call per query"
echo -e "            Faster; fewer API calls; great for large corpora."
echo
choose STRATEGY "Choose annotation strategy:" "3-stage" "Monolithic"

# ── Step 2: Provider ──────────────────────────────────────────────────────────
header "Step 2: LLM Provider"
echo -e "  ${BOLD}Anthropic${RESET}  : claude-sonnet-4-6 (recommended — best quality)"
echo -e "             Async + Batch (50% discount)."
echo
echo -e "  ${BOLD}OpenAI${RESET}     : gpt-4o (strong quality, competitive pricing)"
echo -e "             Async + Batch (50% discount)."
echo
echo -e "  ${BOLD}OpenRouter${RESET} : any model slug, e.g. qwen/qwen-2.5-72b-instruct"
echo -e "             Async only (no batch). Great for cost/reproducibility."
echo
choose PROVIDER "Choose LLM provider:" "Anthropic" "OpenAI" "OpenRouter"

case "$PROVIDER" in
"Anthropic")
    PROVIDER_FLAG="anthropic"
    ensure_anthropic_key
    DEFAULT_MODEL="claude-sonnet-4-6"
    BATCH_AVAILABLE=true
    ;;
"OpenAI")
    PROVIDER_FLAG="openai"
    ensure_openai_key
    DEFAULT_MODEL="gpt-4o"
    BATCH_AVAILABLE=true
    ;;
"OpenRouter")
    PROVIDER_FLAG="openrouter"
    ensure_openrouter_key
    DEFAULT_MODEL="qwen/qwen-2.5-72b-instruct"
    BATCH_AVAILABLE=false
    ;;
esac

ask MODEL "Model name" "$DEFAULT_MODEL"

# Validate / expand OpenRouter model slug
if [[ "$PROVIDER_FLAG" == "openrouter" ]]; then
    validate_openrouter_model
fi

# ── Step 3: Execution Mode ────────────────────────────────────────────────────
header "Step 3: Execution Mode"
if [[ "$BATCH_AVAILABLE" == true ]]; then
    echo -e "  ${BOLD}Async${RESET}: Concurrent API calls — good for small/medium corpora (<5 000 queries)"
    echo -e "  ${BOLD}Batch${RESET}: Batch API (50% discount, async polling, crash-recoverable)"
    echo
    choose MODE "Choose execution mode:" "Async" "Batch"
else
    echo -e "  ${YELLOW}OpenRouter supports async mode only (no native batch API).${RESET}"
    MODE="Async"
    info "Mode set to: Async"
fi

# ── Step 4: Data Paths ────────────────────────────────────────────────────────
header "Step 4: Data Paths"

NORM_DEFAULT="data/normalized/conflicts_normalized.jsonl"
ask INPUT_PATH "Normalized dataset path" "$NORM_DEFAULT"

if [[ ! -f "$INPUT_PATH" ]]; then
    warn "Input file not found: $INPUT_PATH"
    warn "Run first:  python scripts/normalize_raw_dataset.py"
fi

TS=$(date +"%Y%m%d_%H%M%S")

case "$STRATEGY" in
"3-stage")
    ask STAGE1_OUT "Stage-1 output path" "data/stage1_outputs/stage1_${TS}.jsonl"
    ask STAGE2_OUT "Stage-2 output path" "data/stage2_outputs/stage2_${TS}.jsonl"
    ask STAGE3_OUT "Stage-3 output path" "data/stage3_outputs/stage3_${TS}.jsonl"
    ;;
"Monolithic")
    ask MONO_OUT "Monolithic output path" "data/monolithic_outputs/monolithic_${TS}.jsonl"
    ;;
esac

# ── Step 5: Concurrency / Limits ──────────────────────────────────────────────
header "Step 5: Concurrency & Limits"

if [[ "$MODE" == "Async" ]]; then
    case "$STRATEGY" in
    "3-stage")
        ask CONCURRENCY_S1 "Stage-1 concurrency" "10"
        ask CONCURRENCY_S2 "Stage-2 concurrency" "12"
        ask CONCURRENCY_S3 "Stage-3 concurrency" "8"
        ;;
    "Monolithic")
        ask CONCURRENCY "Concurrency (simultaneous queries)" "8"
        ;;
    esac
fi

ask LIMIT "Max records to process (0 = all)" "0"
[[ "$LIMIT" == "0" ]] && LIMIT=""

# ── Step 6: Confirm ───────────────────────────────────────────────────────────
header "Step 6: Confirm & Run"
divider
echo -e "  Strategy   : ${BOLD}$STRATEGY${RESET}"
echo -e "  Provider   : ${BOLD}$PROVIDER_FLAG${RESET}"
echo -e "  Model      : ${BOLD}$MODEL${RESET}"
echo -e "  Mode       : ${BOLD}$MODE${RESET}"
echo -e "  Input      : ${BOLD}$INPUT_PATH${RESET}"
case "$STRATEGY" in
"3-stage")
    echo -e "  Stage-1 out: ${BOLD}$STAGE1_OUT${RESET}"
    echo -e "  Stage-2 out: ${BOLD}$STAGE2_OUT${RESET}"
    echo -e "  Stage-3 out: ${BOLD}$STAGE3_OUT${RESET}"
    ;;
"Monolithic")
    echo -e "  Output     : ${BOLD}$MONO_OUT${RESET}"
    ;;
esac
[[ -n "${LIMIT:-}" ]] && echo -e "  Limit      : ${BOLD}$LIMIT${RESET}"
divider

prompt "Proceed? [Y/n]"
read -r confirm
if [[ "${confirm:-Y}" =~ ^[Nn]$ ]]; then
    warn "Aborted."
    exit 0
fi

# =============================================================================
#  EXECUTION
# =============================================================================

LIMIT_FLAG=$(build_limit_flag)

# ── 3-Stage ───────────────────────────────────────────────────────────────────
if [[ "$STRATEGY" == "3-stage" ]]; then

    if [[ "$MODE" == "Async" ]]; then
        header "Running Stage 1 (Async)"
        run_cmd scripts/run_stage1_async.py \
            --input "$INPUT_PATH" \
            --output "$STAGE1_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --concurrency "$CONCURRENCY_S1" \
            ${LIMIT_FLAG}

        check_stage1_errors "$STAGE1_OUT"

        header "Validating Stage 1"
        STAGE1_VAL_OUT="${STAGE1_OUT%.jsonl}_validation.txt"
        run_validation scripts/validate_stage1.py \
            --input  "$STAGE1_OUT" \
            --output "$STAGE1_VAL_OUT"

        header "Running Stage 2 (Async)"
        run_cmd scripts/run_stage2_async.py \
            --input "$STAGE1_OUT" \
            --output "$STAGE2_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --concurrency "$CONCURRENCY_S2" \
            ${LIMIT_FLAG}

        header "Validating Stage 2"
        STAGE2_VAL_OUT="${STAGE2_OUT%.jsonl}_validation.json"
        run_validation scripts/validate_stage2.py \
            --input  "$STAGE2_OUT" \
            --report "$STAGE2_VAL_OUT"

        header "Running Stage 3 (Async)"
        run_cmd scripts/run_stage3_async.py \
            --input "$STAGE2_OUT" \
            --output "$STAGE3_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --concurrency "$CONCURRENCY_S3" \
            ${LIMIT_FLAG}

        header "Validating Stage 3"
        run_validation scripts/validate_stage3.py \
            --input "$STAGE3_OUT"

    elif [[ "$MODE" == "Batch" ]]; then
        BATCH_ID_DIR="data/.batch_ids"
        mkdir -p "$BATCH_ID_DIR"
        BATCH_S1_FILE="$BATCH_ID_DIR/batch_stage1_${TS}.txt"
        BATCH_S2_FILE="$BATCH_ID_DIR/batch_stage2_${TS}.txt"
        BATCH_S3_FILE="$BATCH_ID_DIR/batch_stage3_${TS}.txt"

        header "Running Stage 1 (Batch)"
        run_cmd scripts/run_stage1_batch.py \
            --input "$INPUT_PATH" \
            --output "$STAGE1_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --batch-id-file "$BATCH_S1_FILE" \
            ${LIMIT_FLAG}

        check_stage1_errors "$STAGE1_OUT"

        header "Validating Stage 1"
        STAGE1_VAL_OUT="${STAGE1_OUT%.jsonl}_validation.txt"
        run_validation scripts/validate_stage1.py \
            --input  "$STAGE1_OUT" \
            --output "$STAGE1_VAL_OUT"

        header "Running Stage 2 (Batch)"
        run_cmd scripts/run_stage2_batch.py \
            --input "$STAGE1_OUT" \
            --output "$STAGE2_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --batch-id-file "$BATCH_S2_FILE" \
            ${LIMIT_FLAG}

        header "Validating Stage 2"
        STAGE2_VAL_OUT="${STAGE2_OUT%.jsonl}_validation.json"
        run_validation scripts/validate_stage2.py \
            --input  "$STAGE2_OUT" \
            --report "$STAGE2_VAL_OUT"

        header "Running Stage 3 (Batch)"
        run_cmd scripts/run_stage3_batch.py \
            --input "$STAGE2_OUT" \
            --output "$STAGE3_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --batch-id-file "$BATCH_S3_FILE" \
            ${LIMIT_FLAG}

        header "Validating Stage 3"
        run_validation scripts/validate_stage3.py \
            --input "$STAGE3_OUT"
    fi

# ── Monolithic ────────────────────────────────────────────────────────────────
elif [[ "$STRATEGY" == "Monolithic" ]]; then

    if [[ "$MODE" == "Async" ]]; then
        header "Running Monolithic Annotation (Async)"
        run_cmd scripts/run_monolithic_async.py \
            --input "$INPUT_PATH" \
            --output "$MONO_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --concurrency "$CONCURRENCY" \
            ${LIMIT_FLAG}

    elif [[ "$MODE" == "Batch" ]]; then
        BATCH_ID_DIR="data/.batch_ids"
        mkdir -p "$BATCH_ID_DIR"
        BATCH_MONO_FILE="$BATCH_ID_DIR/batch_mono_${TS}.txt"

        header "Running Monolithic Annotation (Batch)"
        run_cmd scripts/run_monolithic_batch.py \
            --input "$INPUT_PATH" \
            --output "$MONO_OUT" \
            --provider "$PROVIDER_FLAG" \
            --model "$MODEL" \
            --batch-id-file "$BATCH_MONO_FILE" \
            ${LIMIT_FLAG}
    fi
fi

# ── Final summary ─────────────────────────────────────────────────────────────
header "Pipeline Complete 🎉"
case "$STRATEGY" in
"3-stage")
    success "Stage-3 output ready: $STAGE3_OUT"
    echo
    echo -e "  Validation reports written:"
    echo -e "    Stage-1: ${CYAN}${STAGE1_VAL_OUT}${RESET}"
    echo -e "    Stage-2: ${CYAN}${STAGE2_VAL_OUT}${RESET}"
    echo -e "    Stage-3: ${CYAN}(printed above)${RESET}"
    ;;
"Monolithic")
    success "Monolithic output ready: $MONO_OUT"
    ;;
esac
echo
info "Tip: The output JSONL can be used directly for RAG evaluation or fine-tuning."
echo