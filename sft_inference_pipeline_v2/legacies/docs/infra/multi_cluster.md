# Multi-Cluster Workflow

This repo is intended to stay code-synced across:

- `CSIS` as the current stable/fallback cluster
- `Sharanga` as the newer/faster cluster

Use the local Mac checkout as the source of truth for code changes. Sync outward from local to both clusters.

## What Should Stay Synced

Sync these repo paths:

- `code/`
- `prompts/`
- `scripts/`
- `slurm/`
- `env/`
- `docs/`
- `data/splits/`
- `README.md`
- `run.sh`

Do not sync generated message files by default:

- `data/messages/`

Rebuild `data/messages/` on each cluster after syncing so message files always reflect the current prompts and target-format code.

## What Should Stay Cluster-Local

Do not sync these by default:

- `models/`
- `checkpoints/`
- `outputs/`
- Hugging Face caches
- conda environments
- temporary scratch data

These are large, cluster-specific, and should live close to the compute nodes.

## Recommended Layout

### CSIS

- Repo: `/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2`
- Models: `/nfs_home/users/vsshekhawat/projects/rag-reason/models`
- Conda env: `rag-reason`

### Sharanga

- Repo: `$HOME/rag-reason/sft_inference_pipeline_v2`
- Models: `$SCRATCH/rag-reason/models`
- Conda env path: `$SCRATCH/rag-reason/envs/rag-reason`
- Caches and outputs: `$SCRATCH/rag-reason/{cache,outputs,checkpoints}`

Sharanga account aliases currently configured locally:

- `sharanga` -> `pabitra`
- `sharanga1` -> `kudhru`

Code belongs in `$HOME` on Sharanga because it is quota-limited but backed up. Heavy artifacts belong in `$SCRATCH` because they are large and performance-sensitive.

## Environment Strategy

Do not try to make the conda environments bit-identical across clusters.

Keep the Python dependency intent synced via:

- `env/common-requirements.txt`
- `env/csis-conda.yml`
- `env/sharanga-conda.yml`

Then create cluster-local environments with:

- `scripts/bootstrap_csis_env.sh`
- `scripts/bootstrap_sharanga_env.sh`

## Sync Workflow

1. Make code changes locally.
2. Sync to CSIS with `scripts/sync_csis.sh`.
3. Sync to Sharanga with `scripts/sync_sharanga.sh`.
4. Sync to the second Sharanga account with `scripts/sync_sharanga1.sh` when we want a mirrored backup account.
5. If an emergency hotfix is made remotely, pull it back to local immediately and then re-sync outward.

## Sharanga Notes

Known cluster facts discovered during exploration:

- Accessible GPU QoS: `qos_gpu_a100`, `qos_gpu_h100`, `qos_gpu_h200`
- Blackwell partition exists but requires an additional QoS not currently attached to the account
- H100 interactive access works with:
  - `--partition=gpu_h100_4`
  - `--gres=gpu:1`
  - `--cpus-per-task=4`
- H200 partition is valid but often saturated
- Login nodes should not run compute; package installs should be done inside an interactive or batch Slurm job

## First Sharanga Smoke Test

After bootstrapping the env, run:

```bash
sbatch slurm/sharanga/smoke_h100_1gpu.sh
```

That verifies:

- conda activation
- `torch` import
- CUDA visibility
- H100 GPU name
- `transformers`, `peft`, and `bitsandbytes` imports

## Current Sharanga Training Path

Use Sharanga for the next main training run and keep CSIS as fallback.

Recommended sequence after syncing:

```bash
cd ~/rag-reason/sft_inference_pipeline_v2
source slurm/sharanga/common_env.sh
python -m pip install -r env/common-requirements.txt
bash slurm/examples/rebuild_messages_trace_text_multitask.sh
bash slurm/sharanga/examples/qwen_stagewise_ddp_2h100.sh
```

The current main run is `main_trace_text_b`. It rebuilds clean trace-text multitask messages from canonical splits and trains on 2 H100 GPUs by default.

Why this is the next run:

- `main_trace_text_a` proved the readable trace-text pipeline works, but its training targets inherited annotation artifacts from older `think` fields.
- Prompt-fixed oracle runs showed that gold per-doc notes lift doc-verdict behavior sharply, so the next model should learn cleaner Stage 1 evidence notes rather than only polish conflict labels.
- The dominant remaining failure is over-abstention when retrieved support exists, so the current prompts and target builder explicitly preserve grounded answers and reject unsupported abstention.
