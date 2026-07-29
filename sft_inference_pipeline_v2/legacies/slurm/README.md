# SLURM Smoke Workflow

These scripts codify the cluster-tested smoke path for the CSIS cluster.

Expected server layout:

- Project repo: `/nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2`
- Models: `/nfs_home/users/vsshekhawat/projects/rag-reason/models`
- Conda env: `rag-reason`

Run order:

1. `sbatch slurm/smoke_2gpu_models.sh`
2. `sbatch slurm/smoke_train_1gpu.sh`
3. `sbatch slurm/smoke_generate_baseline_1gpu.sh`
4. `sbatch slurm/smoke_generate_sft_1gpu.sh`
5. `sbatch slurm/smoke_eval.sh`

Notes:

- All scripts exclude `csis.mn1`.
- `smoke_eval.sh` uses `debug` because the smoke eval is tiny and this avoids the `cpu-short` QoS group-memory bottleneck seen during testing.
- `prepare_smoke_subsets.sh` regenerates the tiny smoke JSONLs from the canonical message files.

## Real Experiment Workflow

Primary launchers:

1. `sbatch slurm/train_experiment.sh`
2. `sbatch slurm/generate_experiment.sh`
3. `sbatch slurm/evaluate_experiment.sh`

These are configured via environment variables. Important ones:

- `BASE_MODEL`
- `MODEL_NAME`
- `TRAIN_STRATEGY`
- `VAL_STRATEGY`
- `RUN_NAME`
- `DATASET_LABEL`
- `PROMPT_MODE`
- `PROMPT_PROFILE`
- `MODEL_VARIANT`

Ready-to-run first pilot:

- `bash slurm/examples/qwen_stagewise_pilot.sh`

Multi-GPU DDP pilot:

- `bash slurm/examples/qwen_stagewise_ddp_2gpu.sh`

Recommended follow-up 2-GPU recipe after the pilot systems test:

- `bash slurm/examples/qwen_stagewise_ddp_2gpu_main.sh`

Recommended next ablation after the stronger stagewise baseline:

- `bash slurm/examples/rebuild_messages_parsed_targets.sh`
- `bash slurm/examples/qwen_stagewise_ddp_2gpu_ablation.sh`

Optional bare-minimal inference-prompt rebuild:

- `bash slurm/examples/rebuild_messages_minimal_inference.sh`

This intentionally withholds trace-format details at inference time. The old strict
JSON/text-contract prompt is still available as `PROMPT_PROFILE=legacy_text_contract`.

Recommended prompt-internalization matrix after a serious SFT checkpoint:

- CSIS / Qwen 7B: `bash slurm/examples/qwen7b_stagewise_generate_eval_matrix_csis.sh`
- Sharanga / Qwen 32B H200: `bash slurm/sharanga/examples/qwen32b_stagewise_generate_eval_matrix_h200.sh`

These submit baseline `trace_text`, baseline `minimal`, SFT-B `minimal`, and SFT-C
`minimal` generations plus dependent eval jobs. SFT-B/C `trace_text` runtime outputs
are intentionally not rerun by default because they are the main runtime runs.

Why this ablation:

- keeps the strongest known 2-GPU training core
- increases the stratified dev subset so checkpoint selection is less noisy
- removes abstain-weighted selection pressure, which regressed full-val semantics in a later run
- trains on parsed-style assistant targets again for a cleaner comparison against the stronger prior baseline

Important cluster topology note:

- The cluster exposes `2 GPU instances per node`.
- A clean standard `3 GPU` DDP request is not a good fit for this topology because normal SLURM node requests are symmetric.
- In practice, the next supported test should be `2 GPUs on 1 node`.
- If later you want multi-node DDP, follow the proven DR-MVP pattern:
  - `srun bash launcher.sh`
  - `SLURM_PROCID` for `--node_rank`
  - `MASTER_ADDR` from the `192.168.1.x` inter-node network
  - never `srun torchrun` directly

Recommended first real sequence:

1. Train: `Qwen2.5-7B-Instruct` on `stagewise`
2. Generate SFT outputs on `val_stagewise` with `PROMPT_MODE=e2e`
3. Evaluate those sanitized outputs

Suggested post-train SFT generation example:

```bash
MODEL_NAME=qwen25 \
BASE_MODEL=/nfs_home/users/vsshekhawat/projects/rag-reason/models/Qwen2.5-7B-Instruct \
TRAIN_STRATEGY=stagewise \
RUN_NAME=pilot1 \
DATASET_LABEL=val_stagewise \
PROMPT_MODE=e2e \
MODEL_VARIANT=sft \
sbatch slurm/generate_experiment.sh
```

Suggested minimal-prompt SFT generation example:

```bash
MODEL_NAME=qwen25 \
BASE_MODEL=/nfs_home/users/vsshekhawat/projects/rag-reason/models/Qwen2.5-7B-Instruct \
TRAIN_STRATEGY=stagewise \
RUN_NAME=pilot1 \
DATASET_LABEL=val_stagewise \
PROMPT_MODE=e2e \
PROMPT_PROFILE=minimal \
MODEL_VARIANT=sft \
sbatch slurm/generate_experiment.sh
```

Suggested evaluation example:

```bash
MODEL_NAME=qwen25 \
TRAIN_STRATEGY=stagewise \
RUN_NAME=pilot1 \
DATASET_LABEL=val_stagewise \
PROMPT_MODE=e2e \
MODEL_VARIANT=sft \
sbatch slurm/evaluate_experiment.sh
```

## Sharanga

Sharanga-specific bootstrap and smoke helpers live under:

- `slurm/sharanga/common_env.sh`
- `slurm/sharanga/smoke_h100_1gpu.sh`

Recommended first-time setup on Sharanga:

1. Sync code to `$HOME/rag-reason/sft_inference_pipeline_v2`
2. Create the conda env on `$SCRATCH` with `bash scripts/bootstrap_sharanga_env.sh`
3. Smoke test with `sbatch slurm/sharanga/smoke_h100_1gpu.sh`

Cluster notes and sync workflow are documented in `docs/infra/multi_cluster.md`.

Sharanga 2-GPU launchers:

- generic DDP train job: `slurm/sharanga/train_experiment_ddp_2gpu.sh`
- H100 example: `bash slurm/sharanga/examples/qwen_stagewise_ddp_2h100.sh`
- H200 example: `bash slurm/sharanga/examples/qwen_stagewise_ddp_2h200.sh`

These launchers request 2 GPUs on one node, 8 CPUs total, and 192G memory to stay aligned with the observed 96G-per-GPU floor on Sharanga GPU partitions.

Current recommended Sharanga run:

```bash
bash slurm/examples/rebuild_messages_trace_text_multitask.sh
bash slurm/sharanga/examples/qwen_stagewise_ddp_2h100.sh
```

This launches `main_trace_text_b` on Qwen2.5-7B with clean trace-text multitask targets:

- full `e2e_trace`
- Stage 1 `doc_verdict`
- Stage 2 `conflict_type`
- `answer_only`

The H100 launcher runs preflight checks before `sbatch` so long jobs fail early if target artifacts, missing sentinels, or malformed trace blocks are detected.

Additional Sharanga multi-GPU smoke jobs:

- `sbatch slurm/sharanga/smoke_h100_2gpu.sh`
- `sbatch slurm/sharanga/smoke_h200_2gpu.sh`
- `sbatch slurm/sharanga/smoke_a100_4gpu.sh`

These verify that multi-GPU scheduling works and that NCCL + `torchrun` can complete a simple all-reduce on the requested GPU count.
