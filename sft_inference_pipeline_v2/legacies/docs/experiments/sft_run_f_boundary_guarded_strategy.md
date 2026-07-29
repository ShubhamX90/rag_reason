# SFT Run F: Boundary-Guarded Prompt-Robust Strategy

Status: completed on Sharanga 32B and CSIS 7B.

Run F is a targeted follow-up to D and E. D remains the safest reference run because it proved true minimal-prompt trace internalization. E showed that extra conflict calibration can help 32B but hurt 7B by changing the conflict-label distribution too aggressively. F therefore keeps D's prompt-robust backbone and adds only a small taxonomy-boundary drill source for conflict-type classification.

## Goal

Preserve the behavior D gave us:

- The SFT model emits `<think>...</think>` and `[[END-OF-ANSWER]]` under minimal prompting.
- Minimal, runtime, and strict prompts all remain usable evaluation settings.
- Doc verdicts and abstain behavior do not collapse.

Try to improve the main remaining weakness:

- Conflict taxonomy boundaries, especially `No conflict` vs `Complementary information`, `No conflict` vs `Conflict due to outdated information`, and `Complementary information` vs `Conflicting opinions or research outcomes`.

## Why Not Repeat E

E used broad conflict-label oversampling and a higher conflict loss weight. It was model-size sensitive:

- 7B E improved contract/final-answer cleanliness but reduced conflict accuracy versus D.
- 32B E improved conflict accuracy but reduced 32B minimal doc-verdict accuracy versus D.

F avoids E's broad pressure. Instead of heavily duplicating labels, it adds one extra boundary-drill copy of each runtime `conflict_type` row with explicit taxonomy guidance in the user message.

## Prompt Families

F keeps the same three primary prompt families as D:

- Strict/default teacher prompt: `prompts/system_e2e.txt`, `prompts/user_e2e.txt`
- Runtime guided prompt: `prompts/system_e2e_runtime.txt`, `prompts/user_e2e_runtime.txt`
- Minimal prompt: `prompts/system_e2e_minimal.txt`, `prompts/user_e2e_minimal.txt`

F adds one derived family:

- `runtime_boundary_trace_text`: copies of runtime `conflict_type` rows with a short taxonomy-boundary prefix prepended to the user message.

Oracle prompts are not part of F training. They remain inference-time ablations.

## Message Build

Primary builder:

```bash
bash slurm/examples/rebuild_messages_prompt_robust_f_boundary_guarded.sh
```

Output training file:

```text
data/messages/train_stagewise_prompt_robust_trace_text_f_boundary_guarded_messages.jsonl
```

Exact F mixture after local validation:

```text
strict_default:e2e_trace                    1218 rows = 609 * 2
runtime_trace_text:e2e_trace                 609 rows = 609 * 1
runtime_trace_text:doc_verdict               609 rows = 609 * 1
runtime_trace_text:conflict_type            1218 rows = 609 * 2
runtime_trace_text:answer_only               609 rows = 609 * 1
runtime_boundary_trace_text:conflict_type    609 rows = 609 * 1
minimal_trace_text:e2e_trace                2436 rows = 609 * 4
total                                       7308 rows
```

Task totals:

```text
e2e_trace      4263
conflict_type  1827
doc_verdict     609
answer_only     609
```

Prompt-family totals:

```text
runtime_trace_text           3045
minimal_trace_text           2436
strict_default               1218
runtime_boundary_trace_text   609
```

Exact weighting flags:

```bash
--strict-e2e-weight 2
--runtime-task-weight e2e_trace=1
--runtime-task-weight doc_verdict=1
--runtime-task-weight conflict_type=2
--runtime-task-weight answer_only=1
--boundary-conflict-label-weight "No conflict=1"
--boundary-conflict-label-weight "Complementary information=1"
--boundary-conflict-label-weight "Conflicting opinions or research outcomes=1"
--boundary-conflict-label-weight "Conflict due to outdated information=1"
--boundary-conflict-label-weight "Conflict due to misinformation=1"
--minimal-e2e-weight 4
```

Boundary-drill label counts:

```text
No conflict                              232
Complementary information                189
Conflicting opinions or research outcomes 118
Conflict due to outdated information      64
Conflict due to misinformation             6
```

The boundary prefix explicitly says:

- Use `No conflict` only when evidence aligns or is redundant/contextual.
- Use `Complementary information` when distinct valid facets must be combined.
- Use `Conflicting opinions or research outcomes` for incompatible claims on the same scope.
- Use `Conflict due to outdated information` only when older evidence competes with newer/current evidence.
- Do not collapse complementary evidence into `No conflict`.
- Do not call historical background outdated unless it competes with the current answer.

## Training Hyperparameters

F keeps D's training shape and raises conflict pressure gently:

```text
TRAIN_STRATEGY=stagewise
VAL_STRATEGY=stagewise
TRAIN_JSONL=data/messages/train_stagewise_prompt_robust_trace_text_f_boundary_guarded_messages.jsonl
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
CONFLICT_WEIGHT=3.6
CONTRACT_WEIGHT=3.0
ARRAY_WEIGHT=1.25
CITATION_WEIGHT=1.7
CLASS_BALANCE_POWER=0.55
PATIENCE=3
DEV_SUBSET=49
DEV_MAX_NEW_BASE=900
DEV_MAX_NEW_CAP=1800
DEV_DOC_VERDICT_WEIGHT=0.20
DEV_FORMAT_WEIGHT=0.35
DEV_ABSTAIN_WEIGHT=0.15
DEV_RETRY_ATTEMPTS=0
DEV_RETRY_SCALE=1.6
DEV_RETRY_CAP=2600
DDP_TIMEOUT_SEC=10800
OVERWRITE_OUTPUT_DIR=0
```

The key difference from D is `CONFLICT_WEIGHT=3.6` instead of `3.2`, plus the 609 boundary rows.

The key difference from E is that F does not use broad runtime conflict-label oversampling and does not lower the dev format/doc weights.

## Launchers

CSIS 7B training:

```bash
cd /nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2
bash slurm/examples/qwen7b_stagewise_ddp_2gpu_prompt_robust_f_boundary_guarded.sh
```

CSIS 7B checkpoint:

```text
checkpoints/qwen25_stagewise_e2e_main_trace_text_f_boundary_guarded_csis/best_dev_f1
```

Sharanga 32B training:

```bash
cd ~/rag-reason/sft_inference_pipeline_v2
source slurm/sharanga/common_env.sh
bash slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_f_boundary_guarded.sh
```

Sharanga 32B checkpoint:

```text
/scratch/$USER/rag-reason/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_f_boundary_guarded/best_dev_f1
```

## Generate and Evaluate

CSIS:

```bash
bash slurm/examples/qwen7b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_csis.sh
```

Sharanga:

```bash
bash slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_h200.sh
```

Both wrappers submit strict, runtime, and minimal generations, then submit the matching eval jobs with `afterok` dependencies.

Generation settings match D/E:

```text
strict/default: PROMPT_PROFILE=default, MESSAGE_TAG=strict, CONTRACT_MODE=trace, RETRY_ATTEMPTS=1
runtime:        PROMPT_PROFILE=runtime, MESSAGE_TAG=trace_text, CONTRACT_MODE=trace, RETRY_ATTEMPTS=1
minimal:        PROMPT_PROFILE=minimal, MESSAGE_TAG="", CONTRACT_MODE=none, RETRY_ATTEMPTS=0
```

## Optional Dependency Submitters

After the train job ID is known, the generate/eval wrapper can be queued immediately.

CSIS:

```bash
CSIS_TRAIN_JOB=<job_id>
sbatch --job-name=submit-f-gen --partition=cpu-short --dependency=afterok:${CSIS_TRAIN_JOB} \
  --output=logs/submit_f_generate_%j.out --error=logs/submit_f_generate_%j.err \
  --wrap='cd /nfs_home/users/vsshekhawat/projects/rag-reason/sft_inference_pipeline_v2 && bash slurm/examples/qwen7b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_csis.sh'
```

Sharanga:

```bash
SHARANGA_TRAIN_JOB=<job_id>
sbatch --job-name=submit-f-gen --partition=compute --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:30:00 \
  --dependency=afterok:${SHARANGA_TRAIN_JOB} \
  --output=logs/sharanga_submit_f_generate_%j.out \
  --error=logs/sharanga_submit_f_generate_%j.err \
  --wrap='cd ~/rag-reason/sft_inference_pipeline_v2 && bash slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_h200.sh'
```

## Local Validation

The local message build completed with:

```text
rows=7308
forbidden_hits=[]
missing_assistant=[]
missing_sentinel=[]
missing_think=[]
ok=true
```

Additional checks passed:

```bash
bash -n slurm/examples/rebuild_messages_prompt_robust_f_boundary_guarded.sh \
  slurm/examples/qwen7b_stagewise_ddp_2gpu_prompt_robust_f_boundary_guarded.sh \
  slurm/examples/qwen7b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_csis.sh \
  slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_f_boundary_guarded.sh \
  slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_f_boundary_guarded_h200.sh

python3 -m py_compile scripts/build_prompt_robust_messages.py
python3 scripts/check_trace_text_messages.py data/messages/train_stagewise_prompt_robust_trace_text_f_boundary_guarded_messages.jsonl
```

## Acceptance Read

F should be considered a success only if it preserves D's minimal behavior while improving or not materially hurting conflict accuracy:

- Minimal `think` and `sentinel` should remain near `49/49`.
- 7B should not repeat E's conflict collapse.
- 32B should retain most of E's conflict improvement while recovering D-like doc verdict quality.
- If F improves 32B but not 7B, preserve D as the likely 7B replication strategy and consider F as a 32B/model-capacity-dependent variant.

## Observed 32B Outcome

Sharanga 32B F completed cleanly and produced strict/runtime/minimal outputs.

Summary:

```text
F strict:
  sentinel=49/49, think=49/49
  contract_adj=85.7
  doc_micro=80.56, doc_macro=0.7909
  conflict_acc=77.55
  final_abstain_acc=100.0
  final_citation_coverage=0.6401

F runtime:
  sentinel=49/49, think=49/49
  contract_adj=77.6
  doc_micro=79.54, doc_macro=0.7865
  conflict_acc=75.00 over 48 parsed labels
  final_abstain_acc=100.0
  final_citation_coverage=0.6252

F minimal:
  sentinel=49/49, think=49/49
  contract_adj=81.6
  doc_micro=81.07, doc_macro=0.8058
  conflict_acc=73.47
  final_abstain_acc=100.0
  final_citation_coverage=0.6286
```

Compared with D/E on 32B:

```text
Strict conflict: D=63.27, E=73.47, F=77.55
Runtime conflict: D=69.39, E=73.47, F=75.00
Minimal conflict: D=63.27, E=71.43, F=73.47

Strict doc_micro: D=85.42, E=85.42, F=80.56
Runtime doc_micro: D=85.68, E=84.65, F=79.54
Minimal doc_micro: D=86.45, E=82.35, F=81.07
```

Interpretation:

- F is the strongest 32B conflict/contract/citation variant so far.
- F preserves true minimal trace internalization.
- F regresses Stage-1 doc verdicts, mainly by overusing `partially supports`.

Manual inspection artifacts:

- `outputs/analysis/f_32b_boundary_guarded_manual_inspection.md`
- `outputs/analysis/def_32b_prompt_robust_audit.csv`
- `outputs/analysis/def_32b_prompt_robust_audit.jsonl`

Key fixed rows:

```text
#0159: heated gemstones, fixed D's over-conflict error.
#0263: public transport vs driving, fixed D's over-conflict error.
#0381: world population, fixed D's missed outdated-information conflict.
#0592: Commonwealth Games gold medals, correct complementary + abstain behavior.
```

Key remaining rows:

```text
#0333: Supreme Court appointment, F misses outdated-information conflict.
#0394: Super Bowl host, F minimal correct but strict/runtime smooth temporal conflict.
#0373: Declaration signers, over-complementary vs gold No conflict.
#0416: Word of Wisdom mandatory, complementary vs gold conflicting opinions.
#0654: gravity definition, abstain correct but over-complementary and doc labels too lenient.
```

Follow-up:

- Run G should keep F's conflict-boundary idea but add a Stage-1 doc-verdict stabilizer to reduce `partially supports` overuse.

## Observed 7B Outcome

CSIS 7B F completed cleanly and produced strict/runtime/minimal outputs.

Summary:

```text
F strict:
  sentinel=49/49, think=49/49
  contract_adj=71.4
  doc_micro=79.43, doc_macro=0.7877
  conflict_acc=77.55
  final_abstain_acc=100.0
  final_citation_coverage=0.6119

F runtime:
  sentinel=49/49, think=49/49
  contract_adj=71.4
  doc_micro=77.12, doc_macro=0.7599
  conflict_acc=71.43
  final_abstain_acc=100.0
  final_citation_coverage=0.5927

F minimal:
  sentinel=49/49, think=48/49
  contract_adj=77.6
  doc_micro=76.94, doc_macro=0.7588
  conflict_acc=72.92 over 48 parsed labels
  final_abstain_acc=97.96
  final_citation_coverage=0.6255
```

Compared with D/E on 7B:

```text
Strict conflict: D=66.67, E=61.22, F=77.55
Runtime conflict: D=68.75, E=57.14, F=71.43
Minimal conflict: D=73.47, E=58.33, F=72.92

Strict doc_micro: D=74.22, E=76.98, F=79.43
Runtime doc_micro: D=75.79, E=76.98, F=77.12
Minimal doc_micro: D=76.98, E=78.52, F=76.94
```

Interpretation:

- F is a strong recovery from E on 7B and is the best 7B strict/runtime variant so far.
- F minimal is close to D minimal, but D remains cleaner for minimal-prompt internalization because F has one malformed trace row.
- The malformed row is `#0531`, where an instruction-like Chinese phrase inside a source snippet appears to contaminate the minimal output. Strict/runtime handle the same row correctly, so this looks like isolated source-instruction leakage rather than a broad F failure.
- The remaining conflict weakness is mostly complementary recall: some gold-complementary partial-evidence rows are collapsed into `No conflict`.

Manual inspection artifacts:

- `outputs/analysis/f_7b_boundary_guarded_manual_inspection.md`
- `outputs/analysis/def_7b_prompt_robust_audit.csv`
- `outputs/analysis/def_7b_prompt_robust_audit.jsonl`

Current 7B preference:

```text
Minimal internalization proof: D
Strict/runtime conflict and doc quality: F
Avoid for 7B: E
```
