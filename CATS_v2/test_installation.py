#!/usr/bin/env python3
# test_installation.py
"""
Quick test to validate CATS v2.0 installation.
Run this after setup to ensure everything is configured correctly.
"""

import sys
from pathlib import Path

print("=" * 60)
print("CATS v2.0 - Installation Test")
print("=" * 60)
print()

# Test 1: Python version
print("✓ Testing Python version...")
assert sys.version_info >= (3, 8), "Python 3.8+ required"
print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Import core modules
print("\n✓ Testing imports...")
try:
    import anthropic
    print("  - anthropic: OK")
except ImportError as e:
    print(f"  ✗ anthropic: FAILED ({e})")
    sys.exit(1)

try:
    import httpx
    print("  - httpx: OK")
except ImportError as e:
    print(f"  ✗ httpx: FAILED ({e})")
    sys.exit(1)

try:
    import nltk
    print("  - nltk: OK")
except ImportError as e:
    print(f"  ✗ nltk: FAILED ({e})")
    sys.exit(1)

try:
    import numpy
    print("  - numpy: OK")
except ImportError as e:
    print(f"  ✗ numpy: FAILED ({e})")
    sys.exit(1)

# Test 3: Import CATS modules
print("\n✓ Testing CATS modules...")
try:
    from rag_eval import (
        EvaluationConfig,
        EnhancedEvaluator,
        load_dataset,
        logger,
    )
    print("  - rag_eval: OK")
except ImportError as e:
    print(f"  ✗ rag_eval: FAILED ({e})")
    sys.exit(1)

# Test 4: Check directories
print("\n✓ Testing directory structure...")
required_dirs = ["data", "outputs", "logs", "configs", "scripts"]
for d in required_dirs:
    if Path(d).exists():
        print(f"  - {d}/: OK")
    else:
        print(f"  ✗ {d}/: MISSING")

# Test 5: Check .env
print("\n✓ Testing configuration...")
if Path(".env").exists():
    print("  - .env: OK")
    # Check for API keys (without displaying them)
    with open(".env") as f:
        content = f.read()
        if "ANTHROPIC_API_KEY" in content and "your-key-here" not in content:
            print("  - ANTHROPIC_API_KEY: Configured")
        else:
            print("  ⚠ ANTHROPIC_API_KEY: Not configured")
        
        if "OPENROUTER_API_KEY" in content and "your-key-here" not in content:
            print("  - OPENROUTER_API_KEY: Configured")
        else:
            print("  ⚠ OPENROUTER_API_KEY: Not configured")
else:
    print("  ✗ .env: MISSING")
    print("    Run: cp .env.example .env and configure API keys")

# Test 6: NLTK data
print("\n✓ Testing NLTK data...")
try:
    from nltk import sent_tokenize
    sent_tokenize("This is a test.")
    print("  - punkt tokenizer: OK")
except Exception as e:
    print(f"  ✗ punkt tokenizer: FAILED ({e})")
    print("    Run: python -c \"import nltk; nltk.download('punkt')\"")

# Test 7: Example data
print("\n✓ Testing example data...")
if Path("data/example_input.jsonl").exists():
    print("  - example_input.jsonl: OK")
    try:
        from rag_eval import load_dataset
        dataset = load_dataset("data/example_input.jsonl")
        print(f"  - Loaded {len(dataset)} example samples")
    except Exception as e:
        print(f"  ✗ Failed to load example data: {e}")
else:
    print("  ✗ example_input.jsonl: MISSING")

# Summary
print("\n" + "=" * 60)
print("Installation Test Complete!")
print("=" * 60)
print()
print("Next steps:")
print("1. Configure API keys in .env file")
print("2. Run: python run_evaluation.py --input data/example_input.jsonl --committee default --max-samples 3")
print("3. Check outputs/ for results")
print()
