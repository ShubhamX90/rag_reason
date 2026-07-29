# Sharanga Local Committee Scripts

These scripts are for locally hosted judge servers used by CATS `local_openai`
committee configs.

Current installed serving env:

```text
/scratch/pabitra/rag-reason/envs/local-judge-serving
```

Server targets:

```bash
sbatch slurm/sharanga/local_committee/qwen122_h200_tp2_server.sbatch
sbatch slurm/sharanga/local_committee/deepseek32_h100_server.sbatch
sbatch slurm/sharanga/local_committee/gemma31_h100_server.sbatch
sbatch slurm/sharanga/local_committee/mistral_small4_h100_tp2_server.sbatch
sbatch slurm/sharanga/local_committee/mistral_small4_h200_tp2_server.sbatch
sbatch slurm/sharanga/local_committee/mistral_small4_a100_tp2_server.sbatch
```

Default ports / served names:

```text
8001  local/qwen3.5-122b
8002  local/deepseek-r1-distill-32b
8003  local/gemma-4-31b
8004  local/mistral-small-4
```

The current H100 scripts reserve 8 CPUs each because the recommended workflow is
staged one-model-at-a-time cache collection. Sharanga's observed H100 QoS CPU
cap blocks multiple 8-CPU H100 jobs from the same user, so do not submit the
DeepSeek, Gemma, and Mistral H100 jobs together unless the quota changes or the
CPU requests are deliberately lowered.

When a server log prints an endpoint like:

```text
endpoint=http://gpunode7.sharanga.local:8001/v1
```

probe it from Sharanga:

```bash
source /scratch/pabitra/rag-reason/envs/local-judge-serving/bin/activate
python slurm/sharanga/local_committee/probe_openai_endpoint.py \
  --base-url http://gpunode7.sharanga.local:8001/v1 \
  --model local/qwen3.5-122b \
  --timeout 180
```

Qwen3.5-122B notes:

```text
VLLM_USE_DEEP_GEMM=0
--linear-backend cutlass
--gdn-prefill-backend triton
chat_template_kwargs.enable_thinking=false in the CATS YAML/probe
```

These are needed on Sharanga's current CUDA 12.1 module path. Without them, the
server hits CUDA >=12.3-only DeepGEMM/FlashInfer paths or returns thinking text
before the JSON judge answer.

DeepSeek32 notes:

```text
Cold start on one H100 can take around 25-30 minutes from scratch.
Once loaded, response_format={"type": "json_object"} gives clean JSON.
```

Gemma31 notes:

```text
vLLM recognizes Gemma4ForConditionalGeneration on one H100.
Cold checkpoint reads from scratch can be very slow; the first 50GB shard may
show 0/2 progress for many minutes.
Gemma tokenizer does not define a default chat template; the Slurm script passes
gemma_chat_template.jinja explicitly.
```

Mistral Small 4 notes:

```text
The downloaded model resolves as PixtralForConditionalGeneration and its
tokenizer does not define a chat template. The Slurm script passes
mistral_chat_template.jinja and forces text-only serving with:
--language-model-only --skip-mm-profiling

Without this, vLLM 0.22.1 + transformers 5.10.2 fails during multimodal dummy
profiling with MistralCommonImageProcessor.fetch_images missing.

On the current Sharanga stack, the A100 launch path is not a safe default for
benchmark work. A real probe run on July 4, 2026 reached `/v1/chat/completions`
and then died with:
RuntimeError: unsupported `a` scalar_type
inside vLLM's FP8/Marlin path. Treat H100 or H200 as the validated Mistral
placements unless the A100 route is revalidated end to end.
```

Use conservative defaults first. Increase `MAX_MODEL_LEN` only after the model
loads and answers a tiny probe.

## Benchmark 3-Judge Launch Shape

For the benchmark committee now used in this repo, the intended 3 judges are:

```text
local/qwen3.5-397b-a17b        priority 6
local/mistral-small-4          priority 3
local/deepseek-r1-distill-32b  priority 2
```

Behavior voting uses the same `weighted_majority` rule as the rest of the
binary committee decisions.

Recommended placement on Sharanga for maximum parallelism under the observed
live limits:

```text
Qwen397      -> 2x H200
Mistral4     -> 2x H100
DeepSeek32   -> 1x A100
```

Why this placement:

- `qos_gpu_h200` currently allows 2 GPUs, which Qwen397 already fully uses.
- the current A100 Mistral path is not launch-safe because it can fail on real
  chat completions inside the FP8/Marlin kernel route.
- putting Mistral4 on 2x H100 keeps it on validated FP8-capable hardware.
- DeepSeek32 is lighter and already has a dedicated 1x A100 server script, so
  moving it to A100 preserves the 3-judge parallel shape cleanly.

New benchmark helpers added in this repo:

```text
configs/benchmark_local_openai_3judge_qwen397.yaml
configs/local_staged/benchmark_local_stage_qwen397_collect.yaml
configs/local_staged/benchmark_local_stage_mistral4_collect.yaml
configs/local_staged/benchmark_local_stage_deepseek32_collect.yaml
configs/local_staged/benchmark_local_stage_final_readonly.yaml
slurm/sharanga/local_committee/benchmark_collect_eval.sbatch
slurm/sharanga/local_committee/benchmark_final_merge.sbatch
scripts/submit_benchmark_file_pipeline.sh
scripts/submit_all_benchmark_file_pipelines.sh
```

The recommended runtime pattern for 102 benchmark files is:

1. Start persistent Qwen397, Mistral4, and DeepSeek32 servers on the three GPU pools.
2. Export their live `BASE_URL`s.
3. Submit one file's three stage-collection jobs in parallel.
4. Run the read-only final merge for that file after the three stage jobs succeed.
5. Repeat file by file without restarting the servers.

This is more reliable and much faster than relaunching the giant model servers
for every individual benchmark file.
