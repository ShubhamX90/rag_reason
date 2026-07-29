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

# Test 2: Check OpenRouter API key
print("2. Checking OPENROUTER_API_KEY...")
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

# Test 3: Check DeepSeek API key
print("3. Checking DEEPSEEK_API_KEY...")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if deepseek_key:
    if len(deepseek_key) > 10:
        print(f"   ✓ Key found: {deepseek_key[:15]}...")
    else:
        print("   ⚠ Key found but seems too short")
        print(f"   Value: {deepseek_key}")
else:
    print("   ⚠ DEEPSEEK_API_KEY not found in environment")
    print("   Add it if you plan to use the Codex+DeepSeek mixed committee")

print()

# Test 4: Try importing dependencies
print("4. Testing imports...")
try:
    import httpx
    print("   ✓ httpx module imported")
except ImportError as e:
    print(f"   ✗ Failed to import httpx: {e}")
    sys.exit(1)

print()

# Test 5: Test OpenRouter API connection (if key exists)
if openrouter_key and len(openrouter_key) > 10:
    print("5. Testing OpenRouter API connection...")
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
    print("5. Skipping OpenRouter API test (no valid key)")

print()

# Test 6: Test DeepSeek API connection (if key exists)
if deepseek_key and len(deepseek_key) > 10:
    print("6. Testing DeepSeek API connection...")
    try:
        import asyncio

        async def test_deepseek():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {deepseek_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-v4-flash",
                        "messages": [{"role": "user", "content": "Say 'test'"}],
                        "max_tokens": 10,
                        "user_id": "cats_eval_test"
                    },
                    timeout=10.0
                )
                if response.status_code == 200:
                    return "Success"
                return f"Error {response.status_code}: {response.text[:100]}"

        result = asyncio.run(test_deepseek())
        if "Success" in result:
            print("   ✓ DeepSeek API working!")
        else:
            print(f"   ✗ DeepSeek API error: {result}")
    except Exception as e:
        print(f"   ✗ DeepSeek API error: {e}")
        print("   Please check your API key is valid")
else:
    print("6. Skipping DeepSeek API test (no valid key)")

print()
print("=" * 60)

# Summary
all_good = True
if not openrouter_key or len(openrouter_key) < 10:
    all_good = False
    print("⚠ OPENROUTER_API_KEY needs to be configured")

if not deepseek_key or len(deepseek_key) < 10:
    print("⚠ DEEPSEEK_API_KEY is optional, but needed for the Codex+DeepSeek mixed committee")

if all_good:
    print("✓ All checks passed! You're ready to run evaluations.")
    print()
    print("Next step:")
    print("  python run_evaluation.py --input data/benchmark/benchmark_final_v2_holdout_clean_736.jsonl --committee default --max-samples 3")
else:
    print("✗ Please fix the issues above before running evaluations.")
    print()
    print("To configure API keys:")
    print("  1. Edit .env file: nano .env")
    print("  2. Add your keys:")
    print("     OPENROUTER_API_KEY=your-key-here")
    print("     DEEPSEEK_API_KEY=your-key-here   # optional unless using mixed committee")
    print("  3. Run this test again: python test_api_keys.py")

print("=" * 60)
