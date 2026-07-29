# SFT Run D: Prompt-Robust Trace-Text Strategy

Status: reference strategy, not yet final SFT selection.

Run D is currently the strongest SFT strategy we have seen overall. It is the first run that reliably demonstrated the core research behavior we need: under a true minimal prompt, the SFT model still emits the learned public reasoning trace, final answer, citations, and sentinel without being explicitly walked through the full format.

This file records the exact strategy so it can be reproduced later for other base models such as Llama and Mistral.

## Research Intent

Run D was designed after earlier runs showed two issues:

- Guided/runtime prompting could make models follow the trace contract, but that did not prove the behavior was internalized.
- True minimal prompting initially caused SFT models to stop emitting the expected `<think>` trace, making the SFT claim weak.

Run D therefore mixed three prompt families during SFT:

- Strict/default teacher prompt: detailed, rule-heavy E2E prompt inspired by the high-quality annotation prompts.
- Runtime guided prompt: shorter practical inference prompt with explicit trace/label/format guidance.
- Minimal prompt: true internalization probe; model must learn to emit trace and answer from SFT behavior, not prompt instructions.

The target output format for all supervised E2E rows is trace-text:

```text
<think>
Stage 1 - Evidence assessment:
...
Stage 2 - Conflict assessment:
Conflict type: ...
Reason: ...
Evidence pattern: ...
Stage 3 - Answer plan:
...
</think>
...
[[END-OF-ANSWER]]
```

## Exact Prompt Files

Prompt path mapping is defined in `code/data/prepare_data.py`.

Strict/default E2E prompt:

- `prompts/system_e2e.txt`
- `prompts/user_e2e.txt`

Runtime E2E prompt:

- `prompts/system_e2e_runtime.txt`
- `prompts/user_e2e_runtime.txt`

Minimal E2E prompt:

- `prompts/system_e2e_minimal.txt`
- `prompts/user_e2e_minimal.txt`

Oracle prompts are not part of Run D training. They remain inference-time ablations.

## Message Build

Primary builder:

```bash
bash slurm/examples/rebuild_messages_prompt_robust_d.sh
```

Output training file:

```text
data/messages/train_stagewise_prompt_robust_trace_text_d_messages.jsonl
```

Run D uses stagewise data:

```text
data/splits/stagewise_multi/train/stage3_final.jsonl
data/splits/stagewise_multi/val/stage3_final.jsonl
```

The builder first creates the three component families:

- `data/messages/train_stagewise_e2e_strict_messages.jsonl`
- `data/messages/train_stagewise_multitask_trace_text_messages.jsonl`
- `data/messages/train_stagewise_e2e_minimal_messages.jsonl`

Then it combines them with `scripts/build_prompt_robust_messages.py`.

Exact D mixture:

```text
strict_default:e2e_trace              1218 rows  = 609 * 2
runtime_trace_text:e2e_trace           609 rows  = 609 * 1
runtime_trace_text:doc_verdict         609 rows  = 609 * 1
runtime_trace_text:conflict_type      1218 rows  = 609 * 2
runtime_trace_text:answer_only         609 rows  = 609 * 1
minimal_trace_text:e2e_trace          2436 rows  = 609 * 4
total                                 6699 rows
```

Task totals:

```text
e2e_trace      4263
conflict_type  1218
doc_verdict     609
answer_only     609
```

Prompt-family totals:

```text
runtime_trace_text 3045
minimal_trace_text 2436
strict_default     1218
```

Exact weighting flags:

```bash
--strict-e2e-weight 2
--runtime-task-weight e2e_trace=1
--runtime-task-weight doc_verdict=1
--runtime-task-weight conflict_type=2
--runtime-task-weight answer_only=1
--minimal-e2e-weight 4
```

Important: Run D does not use conflict-label oversampling. That was introduced in E and caused 7B conflict accuracy to regress.

## Training Hyperparameters

Shared D settings:

```text
TRAIN_STRATEGY=stagewise
VAL_STRATEGY=stagewise
TRAIN_JSONL=data/messages/train_stagewise_prompt_robust_trace_text_d_messages.jsonl
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
CONFLICT_WEIGHT=3.2
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

CSIS 7B training launcher:

```bash
bash slurm/examples/qwen7b_stagewise_ddp_2gpu_prompt_robust_d.sh
```

CSIS 7B model/checkpoint:

```text
MODEL_NAME=qwen25
BASE_MODEL=/nfs_home/users/vsshekhawat/projects/rag-reason/models/Qwen2.5-7B-Instruct
RUN_NAME=main_trace_text_d_prompt_robust_csis
OUT_DIR=checkpoints/qwen25_stagewise_e2e_main_trace_text_d_prompt_robust_csis
BEST=checkpoints/qwen25_stagewise_e2e_main_trace_text_d_prompt_robust_csis/best_dev_f1
```

Sharanga 32B training launcher:

```bash
bash slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_d.sh
```

Sharanga 32B model/checkpoint:

```text
MODEL_NAME=qwen25_32b
BASE_MODEL=/scratch/$USER/rag-reason/models/Qwen2.5-32B-Instruct
RUN_NAME=main_trace_text_d_prompt_robust
OUT_DIR=/scratch/$USER/rag-reason/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_d_prompt_robust
BEST=/scratch/$USER/rag-reason/checkpoints/qwen25_32b_stagewise_e2e_main_trace_text_d_prompt_robust/best_dev_f1
```

## Inference and Evaluation

Run D always evaluates three prompt profiles:

- Strict/default: upper-bound teacher-style setting.
- Runtime: guided practical setting.
- Minimal: internalization setting.

CSIS 7B generate/eval launcher:

```bash
bash slurm/examples/qwen7b_stagewise_generate_eval_prompt_robust_d_csis.sh
```

Sharanga 32B generate/eval launcher:

```bash
bash slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_d_h200.sh
```

Generation settings:

```text
strict/default:
  PROMPT_PROFILE=default
  MESSAGE_TAG=strict
  CONTRACT_MODE=trace
  RETRY_ATTEMPTS=1
  MAX_NEW_TOKENS_BASE=1400
  MAX_NEW_TOKENS_CAP=3200

runtime:
  PROMPT_PROFILE=runtime
  MESSAGE_TAG=trace_text
  CONTRACT_MODE=trace
  RETRY_ATTEMPTS=1
  MAX_NEW_TOKENS_BASE=1200
  MAX_NEW_TOKENS_CAP=2200

minimal:
  PROMPT_PROFILE=minimal
  MESSAGE_TAG=""
  CONTRACT_MODE=none
  RETRY_ATTEMPTS=0
  MAX_NEW_TOKENS_BASE=900
  MAX_NEW_TOKENS_CAP=1800
```

CSIS 7B output run IDs:

```text
sft_qwen25_stagewise_main_trace_text_d_prompt_robust_csis_e2e_strict_val_stagewise
sft_qwen25_stagewise_main_trace_text_d_prompt_robust_csis_e2e_trace_text_val_stagewise
sft_qwen25_stagewise_main_trace_text_d_prompt_robust_csis_e2e_minimal_val_stagewise
```

Sharanga 32B output run IDs:

```text
sft_qwen25_32b_stagewise_main_trace_text_d_prompt_robust_e2e_strict_val_stagewise
sft_qwen25_32b_stagewise_main_trace_text_d_prompt_robust_e2e_trace_text_val_stagewise
sft_qwen25_32b_stagewise_main_trace_text_d_prompt_robust_e2e_minimal_val_stagewise
```

Local summary command:

```bash
python3 scripts/summarize_eval_reports.py --format markdown \
  sft_qwen25_stagewise_main_trace_text_d_prompt_robust_csis_e2e_strict_val_stagewise \
  sft_qwen25_stagewise_main_trace_text_d_prompt_robust_csis_e2e_trace_text_val_stagewise \
  sft_qwen25_stagewise_main_trace_text_d_prompt_robust_csis_e2e_minimal_val_stagewise \
  sft_qwen25_32b_stagewise_main_trace_text_d_prompt_robust_e2e_strict_val_stagewise \
  sft_qwen25_32b_stagewise_main_trace_text_d_prompt_robust_e2e_trace_text_val_stagewise \
  sft_qwen25_32b_stagewise_main_trace_text_d_prompt_robust_e2e_minimal_val_stagewise
```

## Observed D Behavior

Current local D summary:

```text
7B strict:
  sentinel=49/49, think=49/49
  contract_adj=75.5
  doc_micro=74.22, doc_macro=0.7371
  conflict_acc=66.67 over 48 parsed labels
  final_abstain_acc=100.0

7B runtime:
  sentinel=49/49, think=48/49
  contract_adj=73.5
  doc_micro=75.79, doc_macro=0.7551
  conflict_acc=68.75 over 48 parsed labels
  final_abstain_acc=97.96

7B minimal:
  sentinel=49/49, think=49/49
  contract_adj=79.6
  doc_micro=76.98, doc_macro=0.7671
  conflict_acc=73.47 over 49 parsed labels
  final_abstain_acc=97.96
  final_citation_coverage=0.6139

32B strict:
  sentinel=49/49, think=49/49
  contract_adj=69.4
  doc_micro=85.42, doc_macro=0.8438
  conflict_acc=63.27 over 49 parsed labels
  final_abstain_acc=100.0

32B runtime:
  sentinel=49/49, think=49/49
  contract_adj=77.6
  doc_micro=85.68, doc_macro=0.8498
  conflict_acc=69.39 over 49 parsed labels
  final_abstain_acc=100.0

32B minimal:
  sentinel=49/49, think=49/49
  contract_adj=69.4
  doc_micro=86.45, doc_macro=0.8622
  conflict_acc=63.27 over 49 parsed labels
  final_abstain_acc=100.0
```

The most important D result is the 7B minimal run:

```text
sft_qwen25_stagewise_main_trace_text_d_prompt_robust_csis_e2e_minimal_val_stagewise
```

It is the cleanest proof of minimal-prompt trace internalization and the best 7B conflict accuracy so far.

## Known D Failure Modes

Run D is strong but not perfect.

Structural issues:

- 7B runtime has one malformed/misaligned think row: `#0408`.
- 7B strict has one missing conflict/stage parse: `#0263`.
- 32B minimal has one sanitizer-modified row: `#0159`; the sanitizer inserted a newline before the sentinel, and the answer contained Chinese text in the tail.

Stable conflict-taxonomy failures across D:

```text
#0127: gold Complementary information -> predicted No conflict
#0015: gold No conflict -> predicted Conflict due to outdated information
#0392: gold No conflict -> predicted Conflict due to outdated information
#0373: gold No conflict -> predicted Complementary information
#0531: gold No conflict -> predicted Complementary information
#0416: gold Conflicting opinions or research outcomes -> predicted Complementary information
#0654: gold Conflicting opinions or research outcomes -> predicted No conflict
#0334: gold Complementary information -> mixed No conflict / Conflicting opinions
```

Aggregate D conflict confusions across six D outputs:

```text
No conflict -> Complementary information: 30
Complementary information -> No conflict: 22
No conflict -> Conflict due to outdated information: 12
Complementary information -> Conflicting opinions or research outcomes: 12
Conflicting opinions or research outcomes -> Complementary information: 6
Conflicting opinions or research outcomes -> No conflict: 6
Conflict due to outdated information -> No conflict: 6
```

Implication: D solved minimal trace emergence, but conflict taxonomy boundaries remain the main SFT weakness.

## Comparison to E

E was based on D but increased conflict calibration pressure:

- More conflict-type rows.
- Conflict-label oversampling.
- Higher `CONFLICT_WEIGHT`.
- Lower format/doc/abstain checkpoint weights.

On CSIS 7B, E improved contract/citation/final-answer cleanliness but regressed conflict accuracy:

```text
D minimal conflict_acc=73.47
E minimal conflict_acc=58.33
```

On Sharanga 32B, E behaved differently and improved conflict accuracy across strict/runtime/minimal:

```text
32B D strict conflict_acc=63.27
32B E strict conflict_acc=73.47

32B D runtime conflict_acc=69.39
32B E runtime conflict_acc=73.47

32B D minimal conflict_acc=63.27
32B E minimal conflict_acc=71.43
```

However, 32B E minimal reduced doc-verdict accuracy relative to 32B D minimal:

```text
32B D minimal doc_micro=86.45
32B E minimal doc_micro=82.35
```

Interpretation: E's broad conflict-calibration pressure appears model-size sensitive. It hurt 7B conflict taxonomy by over-predicting `No conflict`, but helped 32B conflict taxonomy while slightly weakening doc-verdict accuracy. Run D remains the safest reference strategy, while E is evidence that larger models can absorb more conflict-calibration pressure than 7B.

## Replication Notes for Llama and Mistral

To replicate D for another model family:

1. Keep the D message build unchanged.
2. Keep `TRAIN_JSONL=data/messages/train_stagewise_prompt_robust_trace_text_d_messages.jsonl`.
3. Keep `VAL_JSONL=data/messages/val_stagewise_e2e_minimal_messages.jsonl`.
4. Replace only:
   - `MODEL_NAME`
   - `BASE_MODEL`
   - `RUN_NAME`
   - `OUT_DIR`
   - any model-specific loading flags if needed.
5. Preserve the D loss weights and dev-selection weights unless there is a model-specific failure.
6. Evaluate strict, runtime, and minimal profiles.
7. The acceptance gate must include minimal prompting:
   - `think ~= 49/49`
   - `sentinel ~= 49/49`
   - conflict accuracy close to or above D
   - doc verdict metrics not collapsing
   - final abstain behavior stable

Suggested naming pattern:

```text
<model>_stagewise_e2e_main_trace_text_d_prompt_robust
```

For CSIS-style local checkpoints:

```text
checkpoints/<model>_stagewise_e2e_main_trace_text_d_prompt_robust_csis/best_dev_f1
```

For Sharanga scratch checkpoints:

```text
/scratch/$USER/rag-reason/checkpoints/<model>_stagewise_e2e_main_trace_text_d_prompt_robust/best_dev_f1
```

## Files to Preserve

Strategy scripts:

- `slurm/examples/rebuild_messages_prompt_robust_d.sh`
- `slurm/examples/qwen7b_stagewise_ddp_2gpu_prompt_robust_d.sh`
- `slurm/examples/qwen7b_stagewise_generate_eval_prompt_robust_d_csis.sh`
- `slurm/sharanga/examples/qwen32b_stagewise_ddp_2h200_prompt_robust_d.sh`
- `slurm/sharanga/examples/qwen32b_stagewise_generate_eval_prompt_robust_d_h200.sh`
- `scripts/build_prompt_robust_messages.py`

Prompt files:

- `prompts/system_e2e.txt`
- `prompts/user_e2e.txt`
- `prompts/system_e2e_runtime.txt`
- `prompts/user_e2e_runtime.txt`
- `prompts/system_e2e_minimal.txt`
- `prompts/user_e2e_minimal.txt`

Data outputs needed for exact reproduction:

- `data/messages/train_stagewise_prompt_robust_trace_text_d_messages.jsonl`
- `data/messages/val_stagewise_e2e_strict_messages.jsonl`
- `data/messages/val_stagewise_e2e_trace_text_messages.jsonl`
- `data/messages/val_stagewise_e2e_minimal_messages.jsonl`
- `data/splits/val_stagewise.jsonl`

Current local audit artifacts:

- `outputs/analysis/d_prompt_robust_audit_current.csv`
- `outputs/analysis/d_prompt_robust_audit_current.jsonl`

Checkpoint directories must be preserved on clusters or synced separately if D becomes final.
