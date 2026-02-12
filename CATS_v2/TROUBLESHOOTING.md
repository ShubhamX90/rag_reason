# TROUBLESHOOTING GUIDE - Authentication Errors

## Your Error Summary

You encountered two issues:
1. ✗ Authentication errors for both Anthropic and OpenRouter APIs
2. ✗ Format string error in report generation (now fixed)

## Solution Steps

### Step 1: Create/Update Your .env File

The API keys need to be in a `.env` file in the CATS_v2 directory.

```bash
cd CATS_v2

# Create .env file if it doesn't exist
touch .env

# Edit it
nano .env
```

Add these lines (replace with your actual keys):

```bash
# Anthropic API Key (for Claude Haiku)
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here

# OpenRouter API Key (for DeepSeek, Qwen)
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

**Important**: 
- Remove any quotes around the keys
- No spaces around the `=` sign
- Actual keys start with specific prefixes:
  - Anthropic: `sk-ant-api03-...`
  - OpenRouter: `sk-or-v1-...` or similar

### Step 2: Verify Your .env File

```bash
# Check the file exists and has content
cat .env

# You should see:
# ANTHROPIC_API_KEY=sk-ant-...
# OPENROUTER_API_KEY=sk-or-...
```

### Step 3: Test API Keys

I've created a test script to validate your keys:

```bash
python test_api_keys.py
```

This will:
- Check if .env file exists
- Verify both API keys are present
- Test actual API connectivity
- Tell you exactly what's wrong if anything fails

### Step 4: Re-run Evaluation

Once API keys are working:

```bash
python run_evaluation.py \
    --input data/example_input.jsonl \
    --committee default \
    --verbose
```

## Common Issues & Solutions

### Issue 1: "API key not found"

**Symptom**: 
```
Judge claude-haiku error: "Could not resolve authentication method"
Judge qwen error: "Client error '401 Unauthorized'"
```

**Solution**:
```bash
# Make sure .env file is in the right location
cd CATS_v2
pwd  # Should show path ending in CATS_v2

# Verify .env exists
ls -la .env

# Check environment variables are loaded
source venv/bin/activate
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Anthropic:', os.getenv('ANTHROPIC_API_KEY')[:20] if os.getenv('ANTHROPIC_API_KEY') else 'NOT FOUND')"
```

### Issue 2: Invalid API Keys

**Symptom**: Keys are present but still getting 401 errors

**Solution**:
1. Get new keys:
   - Anthropic: https://console.anthropic.com/settings/keys
   - OpenRouter: https://openrouter.ai/keys

2. Make sure to copy the ENTIRE key including prefix:
   - Anthropic keys are ~100 characters long
   - OpenRouter keys are ~40-60 characters long

3. Test each key separately using curl:

```bash
# Test Anthropic
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: YOUR_ANTHROPIC_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-3-5-20241022","max_tokens":10,"messages":[{"role":"user","content":"test"}]}'

# Test OpenRouter
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer YOUR_OPENROUTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen3-8b-instruct","messages":[{"role":"user","content":"test"}]}'
```

### Issue 3: .env File Not Being Read

**Symptom**: .env file exists with correct keys, but still not working

**Solution**:
```bash
# Export manually as a test
export ANTHROPIC_API_KEY=sk-ant-your-key-here
export OPENROUTER_API_KEY=sk-or-your-key-here

# Then try running
python run_evaluation.py --input data/example_input.jsonl --committee default

# If this works, the issue is with .env loading
# Make sure python-dotenv is installed:
pip install python-dotenv
```

### Issue 4: Format String Error (Already Fixed)

**Symptom**:
```
TypeError: unsupported format string passed to list.__format__
```

**Solution**: This is now fixed in the updated evaluator.py. The issue was trying to format list values as floats before converting them.

## Verification Checklist

Before running evaluation, verify:

- [ ] You're in the CATS_v2 directory
- [ ] Virtual environment is activated: `source venv/bin/activate`
- [ ] .env file exists: `ls .env`
- [ ] .env file has both keys: `cat .env`
- [ ] Keys don't have quotes or extra spaces
- [ ] Test script passes: `python test_api_keys.py`

## Quick Fix Script

If you want to quickly set up:

```bash
cd CATS_v2
source venv/bin/activate

# Create .env if needed
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please edit .env and add your API keys:"
    echo "  nano .env"
fi

# Test
python test_api_keys.py

# If tests pass, run evaluation
python run_evaluation.py \
    --input data/example_input.jsonl \
    --committee default \
    --verbose
```

## Still Having Issues?

If you're still getting errors:

1. **Check logs**: `cat logs/cats_errors.log`

2. **Verify Python packages**: `pip list | grep -E "anthropic|httpx|dotenv"`

3. **Try single judge mode** (only needs Anthropic key):
   ```bash
   python run_evaluation.py \
       --input data/example_input.jsonl \
       --committee none \
       --verbose
   ```

4. **Manual test**:
   ```python
   # test_manual.py
   from dotenv import load_dotenv
   import os
   import anthropic
   
   load_dotenv()
   
   key = os.getenv("ANTHROPIC_API_KEY")
   print(f"Key loaded: {key[:20]}..." if key else "Key not found")
   
   if key:
       client = anthropic.Anthropic(api_key=key)
       message = client.messages.create(
           model="claude-haiku-3-5-20241022",
           max_tokens=10,
           messages=[{"role": "user", "content": "test"}]
       )
       print("Success!", message.content[0].text)
   ```

## Need More Help?

1. Run: `python test_api_keys.py` and share the output
2. Check: `cat .env` (hide your actual key values)
3. Verify: `which python` shows path inside venv
4. Check: `pip list | grep anthropic` shows anthropic package

---

**The fixes have been applied. Follow the steps above to configure your API keys and you should be good to go!**
