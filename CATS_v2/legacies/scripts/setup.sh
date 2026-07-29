#!/bin/bash
# scripts/setup.sh
# Interactive setup script for CATS v2.0

set -e

echo "========================================"
echo "CATS v2.0 - Setup Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    echo -e "${RED}Error: Python 3.8+ required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install requirements
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Requirements installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
echo -e "${GREEN}✓ NLTK data downloaded${NC}"
echo ""

# Create directories
echo "Creating directory structure..."
mkdir -p data outputs logs configs .cache
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Setup environment file
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# CATS v2.0 Environment Configuration

# Anthropic API (for Claude Haiku)
ANTHROPIC_API_KEY=your_anthropic_key_here

# OpenRouter API (for DeepSeek, Qwen, etc.)
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional: OpenAI API (if using GPT models)
# OPENAI_API_KEY=your_openai_key_here
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}⚠  Please edit .env and add your API keys${NC}"
else
    echo -e "${YELLOW}.env file already exists. Skipping...${NC}"
fi
echo ""

# Create example config
if [ ! -f "configs/default.yaml" ]; then
    mkdir -p configs
    cat > configs/default.yaml << 'EOF'
# CATS v2.0 Default Configuration

# Evaluation settings
conflict_eval:
  enable: true
  use_judge_committee: true
  committee:
    voting_strategy: "weighted_majority"
    confidence_threshold: 0.6
    use_async: true
    max_concurrent_requests: 10

# Output settings
outputs_dir: "outputs"
report_md: "outputs/eval_report.md"
detailed_results_json: "outputs/detailed_results.json"

# Pipeline settings
pipeline:
  use_async_evaluation: true
  max_workers: 10
  batch_size: 50
  show_progress: true
  verbose: true
EOF
    echo -e "${GREEN}✓ Default config created at configs/default.yaml${NC}"
fi
echo ""

echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Place your input JSONL file in the data/ directory"
echo "3. Run: ./scripts/run_eval.sh"
echo ""
echo "For help: python run_evaluation.py --help"
echo ""
