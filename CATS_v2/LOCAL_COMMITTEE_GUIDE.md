# Local Judge Committee Guide

This repo can run the LLM judge committee against locally hosted
OpenAI-compatible model servers. The intended servers are vLLM, SGLang,
LMDeploy, or any adapter that exposes:

```text
POST /v1/chat/completions
```

No OpenRouter key is needed for this mode.

## CATS Configuration

Use one of the local configs:

```bash
python run_evaluation.py \
  --input data/splits/92p5_7p5/stagewise_multi/val/stage3_final.jsonl \
  --config configs/val_tier2_local_openai.yaml \
  --committee local
```

The ideal config is:

```text
configs/val_tier2_local_openai.yaml
```

It expects:

```text
local/qwen3.5-397b-a17b        http://127.0.0.1:8001/v1  priority 4
local/deepseek-r1-distill-32b  http://127.0.0.1:8002/v1  priority 2
local/gemma-4-31b              http://127.0.0.1:8003/v1  priority 1
local/mistral-small-4          http://127.0.0.1:8004/v1  priority 1
```

The 2x H200 fallback config is:

```text
configs/val_tier2_local_openai_2xh200_fallback.yaml
```

It replaces the 4x H200 Qwen anchor with `local/qwen3.5-122b`. This is a
continuity fallback, not a same-quality replacement.

## Benchmark 3-Judge Variant

For the 736-sample benchmark workflow in this repo, there is now a separate
3-judge local committee that removes Gemma and uses:

```text
local/qwen3.5-397b-a17b        priority 6
local/mistral-small-4          priority 3
local/deepseek-r1-distill-32b  priority 2
```

Primary benchmark config:

```text
configs/benchmark_local_openai_3judge_qwen397.yaml
```

Benchmark staged configs:

```text
configs/local_staged/benchmark_local_stage_qwen397_collect.yaml
configs/local_staged/benchmark_local_stage_mistral4_collect.yaml
configs/local_staged/benchmark_local_stage_deepseek32_collect.yaml
configs/local_staged/benchmark_local_stage_final_readonly.yaml
```

Benchmark prepared inputs live under:

```text
inputs/prepped_model_eval_inputs/benchmark_set_all_modes/.../input.jsonl
```

Benchmark committee outputs/cache live under:

```text
/scratch/$USER/rag-reason/cats_outputs/benchmark_local_committee_3judge/
```

Validated benchmark GPU placement at the moment:

```text
Qwen397      -> 2x H200
Mistral4     -> 2x H100
DeepSeek32   -> 1x A100
```

Important:
- the current `mistral_small4_a100_tp2_server.sbatch` path is intentionally not
  a default benchmark lane anymore.
- on a real July 4, 2026 probe run, that A100 path answered `/v1/models` but
  then crashed on `/v1/chat/completions` with
  `RuntimeError: unsupported 'a' scalar_type`.
- for clean benchmark launches, always treat a tiny successful
  `/v1/chat/completions` JSON probe as mandatory, not just server startup.

Benchmark controller job placement in the current curated repo is intentionally
simple:

```text
partition=compute
qos=cpulimit
```

That behavior is implemented by `scripts/select_controller_partition.sh`.

For dynamic benchmark orchestration, use:

```text
scripts/submit_benchmark_file_pipeline_dynamic.sh
```

This watcher-style launcher polls Slurm, submits the three collect jobs only
when prior stage dependencies are satisfied, with health-gating support for the
judge endpoints.

You can still override the controller partition manually at submit time with:

```bash
export PRIMARY_CONTROLLER_PARTITION=compute
export FORCE_CONTROLLER_PARTITION=gpu_a100_8   # optional manual override
```

## Local Server Shape

The evaluator sends the same judge prompts used by the existing committee to
each configured endpoint:

```yaml
conflict_eval:
  committee:
    type: "local_openai"
    voting_strategy: "weighted_majority"
    max_concurrent_requests: 4
    timeout_seconds: 900
    response_cache_dir: "outputs/local_committee_cache/val_tier2_4xh200"
    cache_mode: "read_write"
    judges:
      - model_id: "local/qwen3.5-397b-a17b"
        base_url: "http://127.0.0.1:8001/v1"
        priority: 4
        max_tokens: 1600
        request_timeout: 900
```

`model_id` must match the model name served by your local backend, or a name
accepted by that backend. Local calls are marked unmetered in cost summaries,
but token usage is still tracked when the server returns `usage`.

## Staged Committee Runs

If Sharanga cannot host every judge at the same time, keep
`response_cache_dir` fixed and use the response cache.

Recommended staged flow:

1. Start exactly one local judge server.
2. Run the matching single-model collection config with `cache_mode:
   "read_write"`.
3. Stop that server and repeat for the next local judge.
4. After all intended judges have populated the shared cache, run the final
   read-only aggregation config.

The staged 2x H200 fallback configs are:

```text
configs/local_staged/val_local_stage_qwen122_collect.yaml
configs/local_staged/val_local_stage_deepseek32_collect.yaml
configs/local_staged/val_local_stage_gemma31_collect.yaml
configs/local_staged/val_local_stage_mistral4_collect.yaml
configs/local_staged/val_local_stage_final_readonly.yaml
```

All five configs use the same cache:

```text
outputs/local_committee_cache/val_tier2_2xh200_fallback
```

The final score is only comparable to the intended committee when all judge
responses are present in the cache. Intermediate staged outputs are only cache
collection runs.

The 2x H200 fallback final aggregation uses weighted majority:

```text
local/qwen3.5-122b               priority 3
local/deepseek-r1-distill-32b    priority 2
local/gemma-4-31b                priority 1
local/mistral-small-4            priority 1
```

This distribution is defensible as a fallback because Qwen is the anchor but
cannot decide alone: Qwen plus any smaller judge reaches 4/7 priority mass, and
DeepSeek plus Gemma plus Mistral also reaches 4/7 and can overrule Qwen.

## Sharanga Setup Plan

Sync this repo to Sharanga under the requested project directory:

```bash
rsync -av \
  --exclude .git \
  --exclude venv \
  --exclude outputs \
  --exclude logs \
  /Users/shubhammishra/Desktop/rag_reason-CATS_interactive/CATS_v2/ \
  sharanga:~/rag-reason/CATS_v2/
```

On Sharanga, first inspect GPU availability and QoS:

```bash
sinfo -o "%P %G %D %t %m %c"
scontrol show partition
sacctmgr show qos format=Name,Priority,MaxTRESPU,MaxTRESPerUser,MaxJobsPU,MaxSubmitJobsPU
squeue -u "$USER"
```

Questions to answer before launching servers:

```text
1. Can the account get 4 concurrent H200 GPUs for Qwen3.5-397B-A17B?
2. If not, can it get 2 concurrent H200 GPUs for Qwen3.5-122B?
3. Can Gemma 4 31B and Mistral Small 4 run concurrently on H100/A100 partitions?
4. Is there a spare H200/H100/A100 allocation for DeepSeek-R1-Distill-32B Q4?
5. Can the CATS evaluator run on the same node as the servers, or does it need
   node-hostname base URLs or SSH port forwarding?
```

## Example Server Commands

Adjust paths to Sharanga's scratch model locations.

Qwen3.5-397B-A17B, ideal anchor:

```bash
vllm serve "$SCRATCH/models/Qwen3.5-397B-A17B" \
  --served-model-name local/qwen3.5-397b-a17b \
  --tensor-parallel-size 4 \
  --dtype auto \
  --quantization fp8 \
  --host 0.0.0.0 \
  --port 8001
```

Qwen3.5-122B, 2x H200 fallback anchor:

```bash
vllm serve "$SCRATCH/models/Qwen3.5-122B" \
  --served-model-name local/qwen3.5-122b \
  --tensor-parallel-size 2 \
  --dtype auto \
  --quantization fp8 \
  --host 0.0.0.0 \
  --port 8001
```

DeepSeek-R1-Distill-32B Q4:

```bash
vllm serve "$SCRATCH/models/DeepSeek-R1-Distill-32B" \
  --served-model-name local/deepseek-r1-distill-32b \
  --dtype auto \
  --quantization bitsandbytes \
  --host 0.0.0.0 \
  --port 8002
```

Gemma 4 31B:

```bash
vllm serve "$SCRATCH/models/Gemma-4-31B" \
  --served-model-name local/gemma-4-31b \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8003
```

Mistral Small 4:

```bash
vllm serve "$SCRATCH/models/Mistral-Small-4" \
  --served-model-name local/mistral-small-4 \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8004
```

These commands are templates. Confirm the actual model paths, supported
quantization flags, and vLLM/SGLang version on Sharanga before submitting long
jobs.
