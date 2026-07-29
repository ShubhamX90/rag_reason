# SFT Run G: Doc-Stabilized Boundary-Guarded Strategy

Status: completed on Sharanga 32B. Training job `227209` finished successfully, and strict/runtime/minimal generations plus eval reports were synced locally.

Run G is the final SFT refinement probe after D/E/F. It is not a new broad strategy. It keeps F's successful 32B conflict/contract behavior and adds a targeted Stage-1 doc-verdict stabilizer to recover the main thing F lost versus D/E: doc-verdict accuracy.

## Why G

D:

- Best 32B doc-verdict quality.
- Weaker conflict taxonomy.

E:

- Better 32B conflict than D.
- Moderate doc-verdict degradation.
- Hurt 7B conflict, so broad conflict oversampling is risky.

F:

- Best 32B conflict/contract/citation behavior so far.
- Minimal-prompt trace behavior remains perfect.
- Main regression: doc labels, especially overuse of `partially supports`.

G tests whether we can keep F's conflict wins while bringing Stage 1 closer to D/E.

## Strategy

Keep F's backbone:

- strict/default E2E rows
- runtime multitask rows
- minimal E2E rows
- runtime conflict-boundary drill rows

Add a new derived family:

- `runtime_doc_boundary_trace_text`: copies of runtime `doc_verdict` rows with explicit `supports` / `partially supports` / `irrelevant` boundary guidance.

The doc-verdict drill says:

- Choose `supports` when the snippet directly answers the query or supplies a required fact, even if brief, low-quality, or one side of a later conflict.
- Choose `partially supports` only when the snippet is on-topic and useful but misses a necessary entity, date, scope, mechanism, or explicit answer.
- Choose `irrelevant` when the snippet is the wrong domain, only shares keywords, gives generic background, or cannot help answer.
- Do not downgrade a direct answer to partial just because other documents disagree or provide more detail.
- Do not mark wrong-domain acronyms, analogies, or tangential topics as supporting.

## Prompt Files

Same prompt families as D/E/F:

- Strict/default: `prompts/system_e2e.txt`, `prompts/user_e2e.txt`
- Runtime: `prompts/system_e2e_runtime.txt`, `prompts/user_e2e_runtime.txt`
- Minimal: `prompts/system_e2e_minimal.txt`, `prompts/user_e2e_minimal.txt`

Oracle prompts remain inference-only ablations and are not part of G training.

## Message Build

Primary builder:

```bash
bash slurm/examples/rebuild_messages_prompt_robust_g_doc_stabilized.sh
```

Output training file:

```text
data/messages/train_stagewise_prompt_robust_trace_text_g_doc_stabilized_messages.jsonl
```

Validated G mixture:

```text
strict_default:e2e_trace                         1218 rows = 609 * 2
runtime_trace_text:e2e_trace                      609 rows = 609 * 1
runtime_trace_text:doc_verdict                   1218 rows = 609 * 2
runtime_trace_text:conflict_type                 1218 rows = 609 * 2
runtime_trace_text:answer_only                    609 rows = 609 * 1
runtime_boundary_trace_text:conflict_type         609 rows = 609 * 1
runtime_doc_boundary_trace_text:doc_verdict       609 rows = 609 * 1
minimal_trace_text:e2e_trace                     2436 rows = 609 * 4
total                                            8526 rows
```

Task totals:

```text
e2e_trace      4263
conflict_type  1827
doc_verdict    1827
answer_only     609
```

Prompt-family totals:

```text
runtime_trace_text              3654
minimal_trace_text              2436
strict_default                  1218
runtime_boundary_trace_text      609
runtime_doc_boundary_trace_text  609
```

Exact weighting flags:

```bash
--strict-e2e-weight 2
--runtime-task-weight e2e_trace=1
--runtime-task-weight doc_verdict=2
--runtime-task-weight conflict_type=2
--runtime-task-weight answer_only=1
--boundary-conflict-label-weight "No conflict=1"
--boundary-conflict-label-weight "Complementary information=1"
--boundary-conflict-label-weight "Conflicting opinions or research outcomes=1"
--boundary-conflict-label-weight "Conflict due to outdated information=1"
--boundary-conflict-label-weight "Conflict due to misinformation=1"
--doc-verdict-boundary-weight 1
--minimal-e2e-weight 4
```

## Training Hyperparameters

G is F with a modest Stage-1 correction:

```text
TRAIN_STRATEGY=stagewise
VAL_STRATEGY=stagewise
TRAIN_JSONL=data/messages/train_stagewise_prompt_robust_trace_text_g_doc_stabilized_messages.jsonl
VAL_JSONL=data/messages/val_stagewise_e2e_minimal_messages.jsonl
EPOCHS=2
LR=2e-4
BSZ=1
GRAD_ACCUM=8
MAX_LEN=12288
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
NEFTUNE_ALPHA=5.0
CONFLICT_WEIGHT=3.5
CONTRACT_WEIGHT=3.0
ARRAY_WEIGHT=1.25
CITATION_WEIGHT=1.7
CLASS_BALANCE_POWER=0.55
PATIENCE=3
DEV_SUBSET=49
DEV_MAX_NEW_BASE=900
DEV_MAX_NEW_CAP=1800
DEV_DOC_VERDICT_WEIGHT=0.30
DEV_FORMAT_WEIGHT=0.35
DEV_ABSTAIN_WEIGHT=0.15
DEV_RETRY_ATTEMPTS=0
DEV_RETRY_SCALE=1.6
DEV_RETRY_CAP=2600
DDP_TIMEOUT_SEC=10800
OVERWRITE_OUTPUT_DIR=0
```

Differences from F:

- `runtime doc_verdict` task weight increases from `1` to `2`.
- Adds `runtime_doc_boundary_trace_text` at `609` rows.
- `CONFLICT_WEIGHT` softens from `3.6` to `3.5`.
- `DEV_DOC_VERDICT_WEIGHT` increases from `0.20` to `0.30`.
- Conflict-boundary drill remains unchanged.

## Sharanga 32B Launch

Training:

```bash
cd ~/rag-reason/sft_inference_pipeline_v2
source slurm/sharanga/common_env.sh
bash slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_g_doc_stabilized.sh
```

Checkpoint:

```text
/scratch/$USER/rag-reason/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_g_doc_stabilized/best_dev_f1
```

Generate/eval:

```bash
bash slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_g_doc_stabilized_h200.sh
```

Dependency submitter after training job is known:

```bash
SHARANGA_TRAIN_JOB=<job_id>
sbatch --job-name=submit-g-gen --partition=compute --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:30:00 \
  --dependency=afterok:${SHARANGA_TRAIN_JOB} \
  --output=logs/sharanga_submit_g_generate_%j.out \
  --error=logs/sharanga_submit_g_generate_%j.err \
  --wrap='cd ~/rag-reason/sft_inference_pipeline_v2 && bash slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_g_doc_stabilized_h200.sh'
```

## Local Validation

Validation completed locally:

```text
rows=8526
forbidden_hits=[]
missing_assistant=[]
missing_sentinel=[]
missing_think=[]
ok=true
```

Checks:

```bash
bash -n slurm/examples/rebuild_messages_prompt_robust_g_doc_stabilized.sh \
  slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_g_doc_stabilized.sh \
  slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_g_doc_stabilized_h200.sh

python3 -m py_compile scripts/build_prompt_robust_messages.py
python3 scripts/check_trace_text_messages.py data/messages/train_stagewise_prompt_robust_trace_text_g_doc_stabilized_messages.jsonl
```

## What Would Count as a Win

G is a win over F if:

- 32B minimal keeps `think=49/49` and `sentinel=49/49`.
- Conflict remains near F:
  - strict around or above `75`
  - runtime around or above `73`
  - minimal around or above `71`
- Doc micro recovers materially:
  - ideally minimal returns toward `83-85`
  - strict/runtime recover toward E/D without collapsing conflict
- Final abstain remains `100.0` or very close.

If G improves doc verdicts but conflict falls back near D, F remains the conflict-focused 32B candidate. If G keeps most of F conflict while recovering doc verdicts, G becomes the strongest 32B SFT candidate.

## Observed Result

G did recover doc-verdict quality, but it did not keep enough of F's conflict/contract gains to become the best overall 32B checkpoint.

| Profile | Contract adj | Doc micro | Doc macro | Conflict acc | Final abstain acc | Citation cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 77.6 | 85.17 | 0.8590 | 69.39 | 97.96 | 0.6207 |
| runtime | 75.5 | 81.59 | 0.8159 | 73.47 | 100.00 | 0.6095 |
| minimal | 71.4 | 83.63 | 0.8469 | 69.39 | 97.96 | 0.5898 |

Structural behavior stayed clean:

```text
strict:  sentinel=49/49, think=49/49
runtime: sentinel=49/49, think=49/49
minimal: sentinel=49/49, think=49/49
```

Main tradeoff versus F:

- Strict doc recovered from `80.56` to `85.17`, but conflict fell from `77.55` to `69.39` and contract-adjusted fell from `85.7` to `77.6`.
- Runtime doc recovered from `79.54` to `81.59`, while conflict fell from `75.00` to `73.47`.
- Minimal doc recovered from `81.07` to `83.63`, but conflict fell from `73.47` to `69.39` and contract-adjusted fell from `81.6` to `71.4`.

Manual inspection confirmed the evaluator pattern:

- G fixes several temporal/no-conflict rows such as `#0333`, `#0392`, and `#0531`.
- G over-smooths conflict into `No conflict` on rows such as `#0381`, `#0399`, and `#0654`.
- G over-triggers outdated conflict on `#0015`, where older season information is contextual rather than contradictory.

Final judgment:

```text
Best overall 32B SFT candidate: F boundary-guarded
Useful fallback/ablation: G doc-stabilized
Do not use G as the default large-model port strategy unless doc-verdict quality is prioritized over conflict/contract.
```

Detailed row-level notes:

- [Manual Inspection: 32B Run G Doc-Stabilized](/Users/shubhammishra/Desktop/rag_reason/sft_inference_pipeline_v2/outputs/analysis/g_32b_doc_stabilized_manual_inspection.md)

## 7B Decision

Do not automatically launch G on CSIS 7B.

Reason:

- 7B F already recovered strict/runtime conflict strongly versus E.
- 7B F minimal introduced one malformed/source-contaminated row (`#0531`), while D minimal remained structurally clean.
- G adds more runtime doc-verdict pressure, which is aimed at the 32B F weakness, not the 7B F weakness.
- Unless Sharanga G is clearly better than F without compromising minimal format, the safer 7B replication candidate remains D for minimal internalization and F for strict/runtime behavior.
