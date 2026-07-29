# Reviewer User Manual

## Important

**Please read this manual very, very carefully before you begin reviewing.**

This package records real human-evaluation judgments. If a reviewer uses the
wrong folder, restarts from a fresh copy by mistake, or submits judgments
without understanding the workflow, that can create avoidable cleanup and
reconciliation work later.

Do **not** start reviewing until you have read the sections below.

---

## What This Package Is

This reviewer bundle is a standalone human-evaluation package for a CATS-style
benchmark study.

Your job is to evaluate only the samples assigned to you. The package already
knows:

- which reviewer names are valid
- which samples are assigned to you
- which metrics are applicable for each sample
- which gold context and document notes you need to see

You do **not** need to manually choose files, sample ids, or reviewer ids.

---

## What You Will Receive

You should receive a folder or zip package containing at least:

- `run_reviewer.sh`
- `reviewer_session.py`
- `cats_human_eval/`
- `study/`
- this manual: `REVIEWER_USER_MANUAL.md`

Do not delete, rename, or move files inside this package while reviewing.

---

## Before You Start

1. Unzip the package once into a location you can keep using.
2. Open a terminal in that exact folder.
3. Use the **same unzipped folder** every time you return to your review work.

### Very important

If you start from one folder today, then later unzip a fresh copy somewhere
else and continue there, your earlier saved drafts will **not** be in that new
copy.

So:

- use one stable working copy
- keep returning to that same copy
- do not switch to a fresh unzip midway unless instructed

---

## System Requirements

The launcher expects:

- `python3`
- the Python packages `rich`, `typer`, and `PyYAML`

If the launcher fails with a missing-module error, contact the study organizer
or install the needed packages in the Python environment being used.

---

## How To Start

From inside the package folder, run:

```bash
./run_reviewer.sh
```

You will then see:

```text
Reviewer first name:
```

Type your first name only.

For this study, the valid reviewer names are:

- `manan`
- `atharv`
- `parth`
- `samyek`

The launcher recognizes names case-insensitively, so for example `Manan` and
`manan` both work.

After that, the package will automatically:

- recognize who you are
- load your assignments
- show your total assigned count
- show how many you have already submitted
- show how many drafts you currently have

You do **not** need to pass a reviewer id manually.

---

## Reopening Previously Submitted Work

If you later need to reopen your already submitted items for inspection or
editing, run:

```bash
./run_reviewer.sh review
```

Use plain:

```bash
./run_reviewer.sh
```

for normal judging mode.

---

## High-Level Workflow

For each assigned sample:

1. Read the overview screen carefully.
2. Inspect retrieved documents if needed.
3. Evaluate behavior.
4. Evaluate factual grounding (FG).
5. Evaluate single-truth recall (STR) if applicable.
6. Add notes if needed.
7. Review your current summary.
8. Submit the sample only when all applicable fields are complete.

---

## What You See On The Overview Screen

The overview is the home screen for each sample. It shows:

- sample id
- gold conflict label
- gold answerability
- whether it is a correct refusal
- whether the model answered
- autosave status
- gold conflict reason
- gold answer, if present
- query
- model think trace, if present
- model final answer
- counts for extracted claims, retrieved docs, FG-eligible docs, and STR applicability

The overview is meant to give you all the high-level context before you go into
metric-specific judgment pages.

---

## Keyboard Commands

The session command menu is shown inside the interface.

The available commands are:

- `o` : redisplay sample overview
- `d` : browse retrieved documents
- `b` : edit behavior judgment
- `f` : edit factual grounding judgments
- `t` : edit single-truth recall judgment
- `m` : edit reviewer notes
- `r` : show current review summary
- `s` : save current state
- `x` : submit sample
- `n` : move to next sample
- `p` : move to previous sample
- `j` : jump to an assigned sample number
- `q` : save current state and quit
- `h` : show command help

---

## Save Logic and Safety

This is the part you should understand especially carefully.

### Autosave behavior

The package automatically saves after:

- editing behavior
- editing FG
- editing STR
- editing reviewer notes
- moving next
- moving previous
- jumping to another sample
- quitting with save

### Manual save

You can also press:

- `s`

to explicitly save the current state at any time.

### Submit vs draft

There are two important states:

- `draft`
- `submitted`

If you are still working on a sample, it is saved as a **draft**.

When you press:

- `x`

the sample is saved as **submitted**.

### Critical caution

Do not assume that just viewing a screen means something was submitted.

Only `x` submits the sample.

If you quit before submitting, your work should still remain as a draft, but
the sample is not counted as submitted yet.

---

## If You Need To Stop Midway

If you want to stop and continue later:

1. press `s` if you want an explicit save
2. then press `q`
3. confirm save if prompted

When you come back later, reopen the **same package folder** and run:

```bash
./run_reviewer.sh
```

again.

Your saved draft state should still be there in that same working copy.

---

## Behavior Judgment Instructions

Press:

- `b`

to open the behavior review page.

You will see:

- behavior review context
- gold conflict label
- gold conflict reason
- gold answer, if present
- query
- model final answer
- detailed behavior guide
- gold per-document notes

### What behavior means here

Judge only **how** the answer behaves relative to the conflict type.

Do **not** judge factual correctness on this page.

You are judging things like:

- whether the answer is direct when it should be direct
- whether it reconciles complementary information properly
- whether it presents debate when the conflict type is a debate
- whether it prioritizes updated evidence when the conflict type is about outdated information
- whether it ignores misinformation appropriately when the conflict type is about misinformation

### Inputs on the behavior page

You will be asked:

- whether the answer is behavior-adherent: `y/n`
- your confidence category
- your rationale

### Confidence categories

When asked for confidence, use:

- `1` = low
- `2` = medium-low
- `3` = medium-high
- `4` = high

Be honest and conservative. Do not inflate confidence unless you are really
certain.

---

## Factual Grounding (FG) Instructions

Press:

- `f`

to open the FG review page.

You will see:

- FG review context
- gold conflict reason
- gold answer, if present
- query
- model final answer
- FG guide
- FG-eligible gold per-document notes

### What FG means here

Humans are reviewing the same deterministic extracted claims that the local LLM
committee would have judged.

For each extracted claim, your task is to decide:

- which eligible docs support that claim
- whether no single eligible doc supports it
- whether exactly two eligible docs together support it

### What “eligible docs” means

Eligible docs are restricted to gold per-doc verdicts:

- `supports`
- `partially supports`

Only those docs can count for FG.

### Very important

You must mark **all eligible docs that actually support the claim**, not just
the docs the model cited.

That means:

- you are judging true support
- not just citation overlap

### Input format

You may enter supporting docs as:

- `d1,d2,d3`

or equivalently:

- `1,2,3`

Blank means:

- no single eligible doc supports the claim

If no single doc supports it, the interface may ask whether exactly two docs
support it together.

### FG note

You may also add an optional FG note for each claim if needed.

---

## STR Instructions

Press:

- `t`

to open the single-truth recall page.

You will see:

- STR review context
- gold conflict reason
- gold answer
- query
- model final answer
- STR guide

### What STR means

Judge whether the model answer **asserts the gold answer as its own
conclusion**.

Count as a match when the model:

- paraphrases the gold answer
- gives a logically equivalent answer
- differs only in minor wording or formatting

Do **not** count as a match when:

- the gold answer is only quoted or attributed to a source
- the model gives a different answer
- the model only lists the gold answer as one possibility
- the model refuses

### STR applicability

STR is only applicable when the sample is marked applicable by the package.

If STR is not applicable, the package handles that automatically.

---

## Notes

Press:

- `m`

to add reviewer notes.

Use notes for things like:

- edge cases
- ambiguity you want recorded
- anything unusual you think the organizers should know

Notes are helpful, but keep them concise and relevant.

---

## Review Summary

Press:

- `r`

to see your current summary for the sample.

Use this before submitting if you want to double-check that:

- behavior is complete
- FG is complete
- STR is complete when applicable
- your notes are correct

---

## When To Submit

Submit a sample only when you are satisfied with all applicable judgments.

Press:

- `x`

to submit.

The package will block submission if required applicable fields are still
missing.

---

## Refusal Cases

For this study package, deterministic correct-refusal cases were excluded from
the selected human-review pool. You should not need to spend human-review
budget on deterministic correct-refusal examples here.

If you ever encounter something that looks inconsistent, leave a reviewer note
and notify the organizer.

---

## What Not To Do

Please do **not** do any of the following:

- do not edit `study.yaml`
- do not edit `samples.jsonl`
- do not edit `assignments.json`
- do not edit the SQLite database manually
- do not edit `events.jsonl`
- do not move or rename the `study/` folder while working
- do not switch to a fresh unzip midway unless explicitly instructed
- do not share one actively changing working copy between multiple reviewers

---

## Recommended Safe Practice

The safest workflow is:

1. keep one personal copy of the package
2. always reopen that same copy
3. save before quitting
4. submit only when ready
5. if something looks wrong, stop and ask before continuing

---

## If Something Goes Wrong

Stop and contact the organizer if:

- your name is not recognized
- your assigned count looks wrong
- the package says you have no assignments
- a sample looks clearly corrupted
- the launcher crashes
- your earlier drafts seem missing
- the interface behaves in a way that makes you unsure whether work was saved

When reporting a problem, include:

- your reviewer name
- the sample id, if applicable
- what command you were using
- the exact error text, if any

---

## Quick Start Checklist

Before your first real review session:

1. read this manual fully
2. unzip once
3. open terminal in the package folder
4. run `./run_reviewer.sh`
5. enter your first name once
6. confirm the assigned count looks reasonable
7. review carefully
8. save and quit properly if stopping midway
9. always return to the same working copy

---

## Final Reminder

**Please read carefully, review carefully, and save carefully.**

The goal is to avoid a situation where reviewers spend a lot of careful effort
and only later discover that the workflow was misunderstood.
