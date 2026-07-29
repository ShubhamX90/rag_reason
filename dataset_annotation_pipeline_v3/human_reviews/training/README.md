# Training Conflict-Type Review Pack

This folder contains the files needed for the training-dataset conflict-type
review pass.

Reviewers:

- manan -> 1
- atharv -> 2
- parth -> 3

Each training query is assigned to exactly two reviewers. Your job is simple:

- read the query and retrieved snippets
- inspect the currently assigned conflict-type label
- either accept that label or change it
- save your decision with confidence and a short note if needed

## What you need

- Python 3
- Terminal / command prompt
- No API keys
- No internet required

## How to run

Open a terminal in the repository root and run:

```bash
./scripts/run_training_conflict_type_review.sh
```

If that does not work, run:

```bash
python3 scripts/training_conflict_type_review_cli.py
```

Enter only your first name when asked. The tool will automatically map:

- `manan` -> reviewer 1
- `atharv` -> reviewer 2
- `parth` -> reviewer 3

The tool will show:

- your reviewer ID
- total assigned queries
- remaining queries
- assigned queries per conflict type
- how many of your assigned queries are paired with each other reviewer

## What to do for each query

- read the query and snippets as a set
- keep the current label if it is clearly right
- change the label only if it is materially wrong under the displayed taxonomy
- set your confidence
- add a short reason if you changed the label
- add an optional note if useful

Progress is saved automatically after each completed review.

You can stop and resume at any time.

If you go back to an already reviewed query, the tool will reopen your saved
review and let you edit it directly.

## Output file

Your review file will be saved here:

```text
human_reviews/training/reviews/reviewer_<ID>_reviews.jsonl
```

Please send that JSONL file back after finishing your assigned review job.

## Useful controls

From the decision menu:

- `f` switches full/compact snippets
- `r` redraws the current record
- `p` goes back one record and lets you edit it if it was already saved
- `s` skips the current record
- `q` saves and quits
