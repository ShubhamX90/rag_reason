# Benchmark Human Preselection Pack

This folder contains the files needed for:

- first-pass human preselection of benchmark queries
- second review for the final selected non-refusal benchmark set

The current round is the second-review pass on the final selected 800 benchmark
queries. Every selected query has been assigned to exactly one second reviewer.

## What you need

- Python 3
- Terminal / command prompt
- No API keys
- No internet required

## How to run

Open a terminal in the repository root and run:

```bash
./scripts/run_benchmark_human_preselection.sh
```

If that does not work, run:

```bash
python3 scripts/benchmark_human_preselection_cli.py
```

For second review, run:

```bash
./scripts/run_benchmark_human_second_review.sh
```

If that does not work, run:

```bash
python3 scripts/benchmark_human_second_review_cli.py
```

Enter only your first name when asked. The tool will automatically map your name
to your reviewer ID and load your second-review assignment.

Reviewer map:

- shubham -> 1
- harsh -> 2
- gorang -> 3
- shiv -> 4
- atharv -> 5
- manan -> 6
- parth -> 7

The tool will show your total assigned queries, the source-dataset mix, and the
distribution of first reviewers for your queue before the review starts.

## What to do for each query

Read the question and the retrieved document snippets. Then choose:

- accept / borderline accept / borderline reject / reject
- preliminary conflict type
- confidence, retrieval quality, evidence sufficiency
- whether a gold answer is possible
- any reject reason or note when needed

The tool saves your progress automatically after each completed review.

## What happens in second review

The second-review tool will show:

- the same query and retrieved snippets
- the full annotation entered by the first reviewer

Then the second reviewer can:

- accept the first review as-is
- edit any field and save the updated second review
- reject the query outright

This is meant to stay simple. It is mainly for second-pass checking and
inter-annotator agreement over the final benchmark selection.

## Output file

Your review file will be saved here:

```text
human_reviews/benchmark/first_pass/reviews/reviewer_<ID>_reviews.jsonl
```

Second-review output will be saved here:

```text
human_reviews/benchmark/second_pass/second_reviews/reviewer_<ID>_second_reviews.jsonl
```

Send that JSONL file back after you finish your assigned queue.

## Useful controls

From the first decision menu:

- `f` switches full/compact snippets
- `r` redraws the current record
- `p` goes back one record
- `s` skips the current record
- `q` saves and quits
