# Local Committee Configs

These JSON configs add a `local_openai` backend beside the default OpenRouter
committee. They do not change prompts, parsers, output schemas, or weighted
voting semantics. The implementation detail and full reproducibility boundary
are documented in [`docs/ANNOTATION_PIPELINE.md`](../../docs/ANNOTATION_PIPELINE.md).

There are two intentionally separate final-committee families. Priorities are
normalized internally; do not apply the four-judge weights to the retained
three-judge benchmark artifacts.

| Final config | Model priorities | Normalized weights | Retained use |
|---|---|---|---|
| `benchmark_stage_final_readonly.json` | Qwen 4, DeepSeek 2, Gemma 1, Mistral 1 | 0.500, 0.250, 0.125, 0.125 | Four-judge local validation/collection workflow. |
| `benchmark3_stage_final_readonly.json` | Qwen 6, DeepSeek 2, Mistral 3 | 6/11, 2/11, 3/11 | Retained 800 non-refusal and 200 refusal local benchmark Stage-1/2 runs. |

The `*_collect.json` configs contain one judge and are used while that model's
server is running. Each `*_final_readonly.json` config contains its matching
full committee and is used only after cache collection; in `read_only` mode, a
missing cache entry is a hard failure.

`max_tokens` is intentionally omitted in these configs. The stage scripts keep
their original budgets:

```text
Stage 1: 512
Stage 2: 400
Stage 3: 6000
```

Each config has both a fallback `base_url` and a `base_url_env`. On Sharanga,
set the env var to the endpoint printed by the Slurm log, for example:

```bash
export LOCAL_QWEN_BASE_URL=http://gpunode7.sharanga.local:8001/v1
```

The response cache key is based on stage namespace, prompt text, model,
temperature, max_tokens, and extra request body. It does not include the
transient compute-node URL.

## Val49 Local Committee Workflow

Use the same 49 validation IDs that already have OpenRouter stagewise
annotations:

```bash
/scratch/pabitra/rag-reason/envs/local-judge-serving/bin/python \
  scripts/prepare_val49_local_committee_inputs.py \
  --output-dir outputs/local_committee_val49/inputs
```

Collect Stage 1 for one running judge:

```bash
/scratch/pabitra/rag-reason/envs/local-judge-serving/bin/python \
  scripts/run_stage1_multi_async.py \
  --input outputs/local_committee_val49/inputs/val49_all_input.jsonl \
  --output outputs/local_committee_val49/collect/stage1_qwen_collect.jsonl \
  --committee-backend local_openai \
  --committee-config configs/local_committee/benchmark_stage_qwen397_collect.json \
  --cache-dir data/.llm_cache/local_committee_val49 \
  --cache-mode read_write \
  --concurrency 1
```

Repeat Stage 1 with the DeepSeek, Gemma, and Mistral collect configs. Then
aggregate Stage 1 from cache:

```bash
/scratch/pabitra/rag-reason/envs/local-judge-serving/bin/python \
  scripts/run_stage1_multi_async.py \
  --input outputs/local_committee_val49/inputs/val49_all_input.jsonl \
  --output outputs/local_committee_val49/final/stage1_final_readonly.jsonl \
  --committee-backend local_openai \
  --committee-config configs/local_committee/benchmark_stage_final_readonly.json \
  --cache-dir data/.llm_cache/local_committee_val49 \
  --cache-mode read_only \
  --concurrency 8
```

Split Stage 1 for the correct Stage 2/3 prompt families:

```bash
/scratch/pabitra/rag-reason/envs/local-judge-serving/bin/python \
  scripts/split_val49_by_origin.py \
  --input outputs/local_committee_val49/final/stage1_final_readonly.jsonl \
  --output-dir outputs/local_committee_val49/final/stage1_split
```

Run Stage 2 and Stage 3 collection separately for:

- conflicts: no `--refusal-mode`
- refusals: add `--refusal-mode`

After collecting all four judges for a stage, rerun that stage with
`benchmark_stage_final_readonly.json` and `--cache-mode read_only`. Finally,
merge the conflict/refusal outputs:

```bash
/scratch/pabitra/rag-reason/envs/local-judge-serving/bin/python \
  scripts/merge_val49_outputs.py \
  --conflicts outputs/local_committee_val49/final/stage3_conflicts_final_readonly.jsonl \
  --refusals outputs/local_committee_val49/final/stage3_refusals_final_readonly.jsonl \
  --output outputs/local_committee_val49/final/stage3_final_combined.jsonl
```
