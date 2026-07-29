# Sharanga Local Committee Servers

These Slurm scripts serve the local annotation committee for
`dataset_annotation_pipeline_v3`. They are copied into this repo so the local
committee path does not depend on the CATS repo.

Serving env:

```text
/scratch/pabitra/rag-reason/envs/local-judge-serving
```

Model roots:

```text
/scratch/pabitra/rag-reason/models/Qwen3.5-397B-A17B-NVFP4
/scratch/pabitra/rag-reason/models/DeepSeek-R1-Distill-Qwen-32B
/scratch/pabitra/rag-reason/models/gemma-4-31B
/scratch/pabitra/rag-reason/models/Mistral-Small-4-119B-2603
```

Server scripts:

```bash
sbatch slurm/sharanga/local_committee/qwen397_nvfp4_h200_tp2_server.sbatch
sbatch slurm/sharanga/local_committee/deepseek32_h100_server.sbatch
sbatch slurm/sharanga/local_committee/gemma31_h100_server.sbatch
sbatch slurm/sharanga/local_committee/mistral_small4_h200_tp2_server.sbatch
```

Default served names and ports:

```text
8001  local/qwen3.5-397b-a17b
8002  local/deepseek-r1-distill-32b
8003  local/gemma-4-31b
8004  local/mistral-small-4
```

The scripts default to `MAX_MODEL_LEN=32768` because the Stage-3 validation
prompt can require more than 20k tokens when preserving the existing full prompt
and 6000-token output budget. If a model cannot fit this, the server log should
fail loudly; do not silently truncate prompts for head-to-head comparison.

Do not submit all four servers together. The intended workflow is staged cache
collection:

1. Start one server.
2. Probe the endpoint printed in the Slurm log.
3. Run only that judge's collect config.
4. Stop that server.
5. Repeat for the next judge.
6. Run final aggregation with the read-only config after all caches exist.

When a server log prints:

```text
endpoint=http://gpunode7.sharanga.local:8001/v1
```

probe it from Sharanga:

```bash
/scratch/pabitra/rag-reason/envs/local-judge-serving/bin/python \
  scripts/probe_local_openai_endpoint.py \
  --base-url http://gpunode7.sharanga.local:8001/v1 \
  --model local/qwen3.5-397b-a17b \
  --timeout 180 \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}'
```

For collection, export the matching endpoint env var:

```bash
export LOCAL_QWEN_BASE_URL=http://gpunode7.sharanga.local:8001/v1
export LOCAL_DEEPSEEK_BASE_URL=http://gpunode6.sharanga.local:8002/v1
export LOCAL_GEMMA_BASE_URL=http://gpunode6.sharanga.local:8003/v1
export LOCAL_MISTRAL_BASE_URL=http://gpunode7.sharanga.local:8004/v1
```

Only the active server's env var must be set during cache collection. The
read-only final aggregation does not contact servers, and cache keys do not
include the transient base URL.

Model notes:

- Qwen is `Qwen3.5-397B-A17B-NVFP4` on two H200 GPUs. The script uses
  `--quantization modelopt_fp4`, `--reasoning-parser qwen3`, and
  `chat_template_kwargs.enable_thinking=false` in the request config.
- DeepSeek is `DeepSeek-R1-Distill-Qwen-32B` on one H100. DeepSeek V4 Flash is
  not the production path here.
- Gemma 31B needs `gemma_chat_template.jinja`; the script points to the copy in
  this repo.
- Mistral Small 4 is text-only and served with
  `--language-model-only --skip-mm-profiling`.
