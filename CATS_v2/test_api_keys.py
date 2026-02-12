#!/usr/bin/env python3
# test_api_keys.py
"""
Test script to verify API keys are configured correctly.
Run this before running the full evaluation.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 60)
print("CATS v2.0 - API Key Test")
print("=" * 60)
print()

# Test 1: Check .env file exists
print("1. Checking .env file...")
if os.path.exists(".env"):
    print("   ✓ .env file found")
else:
    print("   ✗ .env file not found!")
    print("   Please create .env file from .env.example")
    sys.exit(1)

print()

# Test 2: Check Anthropic API key
print("2. Checking ANTHROPIC_API_KEY...")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_key:
    if anthropic_key.startswith("sk-ant-"):
        print(f"   ✓ Key found: {anthropic_key[:15]}...")
    else:
        print("   ⚠ Key found but doesn't start with 'sk-ant-'")
        print(f"   Value: {anthropic_key[:30]}...")
else:
    print("   ✗ ANTHROPIC_API_KEY not found in environment")
    print("   Please add it to your .env file:")
    print("   ANTHROPIC_API_KEY=sk-ant-your-key-here")

print()

# Test 3: Check OpenRouter API key
print("3. Checking OPENROUTER_API_KEY...")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if openrouter_key:
    if len(openrouter_key) > 10:
        print(f"   ✓ Key found: {openrouter_key[:15]}...")
    else:
        print("   ⚠ Key found but seems too short")
        print(f"   Value: {openrouter_key}")
else:
    print("   ✗ OPENROUTER_API_KEY not found in environment")
    print("   Please add it to your .env file:")
    print("   OPENROUTER_API_KEY=your-key-here")

print()

# Test 4: Try importing dependencies
print("4. Testing imports...")
try:
    import anthropic
    print("   ✓ anthropic module imported")
except ImportError as e:
    print(f"   ✗ Failed to import anthropic: {e}")
    sys.exit(1)

try:
    import httpx
    print("   ✓ httpx module imported")
except ImportError as e:
    print(f"   ✗ Failed to import httpx: {e}")
    sys.exit(1)

print()

# Test 5: Test Anthropic API connection (if key exists)
if anthropic_key and anthropic_key.startswith("sk-ant-"):
    print("5. Testing Anthropic API connection...")
    try:
        import asyncio
        
        async def test_anthropic():
            client = anthropic.AsyncAnthropic(api_key=anthropic_key)
            response = await client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'test'"}]
            )
            return response.content[0].text
        
        result = asyncio.run(test_anthropic())
        print(f"   ✓ Anthropic API working! Response: {result}")
    except Exception as e:
        print(f"   ✗ Anthropic API error: {e}")
        print("   Please check your API key is valid")
else:
    print("5. Skipping Anthropic API test (no valid key)")

print()

# Test 6: Test OpenRouter API connection (if key exists)
if openrouter_key and len(openrouter_key) > 10:
    print("6. Testing OpenRouter API connection...")
    try:
        import asyncio
        
        async def test_openrouter():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "qwen/qwen-2.5-7b-instruct",
                        "messages": [{"role": "user", "content": "Say 'test'"}],
                        "max_tokens": 10
                    },
                    timeout=10.0
                )
                if response.status_code == 200:
                    return "Success"
                else:
                    return f"Error {response.status_code}: {response.text[:100]}"
        
        result = asyncio.run(test_openrouter())
        if "Success" in result:
            print(f"   ✓ OpenRouter API working!")
        else:
            print(f"   ✗ OpenRouter API error: {result}")
    except Exception as e:
        print(f"   ✗ OpenRouter API error: {e}")
        print("   Please check your API key is valid")
else:
    print("6. Skipping OpenRouter API test (no valid key)")

print()
print("=" * 60)

# Summary
all_good = True
if not anthropic_key or not anthropic_key.startswith("sk-ant-"):
    all_good = False
    print("⚠ ANTHROPIC_API_KEY needs to be configured")

if not openrouter_key or len(openrouter_key) < 10:
    all_good = False
    print("⚠ OPENROUTER_API_KEY needs to be configured")

if all_good:
    print("✓ All checks passed! You're ready to run evaluations.")
    print()
    print("Next step:")
    print("  python run_evaluation.py --input data/example_input.jsonl --committee default")
else:
    print("✗ Please fix the issues above before running evaluations.")
    print()
    print("To configure API keys:")
    print("  1. Edit .env file: nano .env")
    print("  2. Add your keys:")
    print("     ANTHROPIC_API_KEY=sk-ant-your-key-here")
    print("     OPENROUTER_API_KEY=your-key-here")
    print("  3. Run this test again: python test_api_keys.py")

print("=" * 60)
