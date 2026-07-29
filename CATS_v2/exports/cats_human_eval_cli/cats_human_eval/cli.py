from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import typer

from .exporting import export_active_judgments
from .storage import (
    append_event,
    get_assignments,
    init_db,
    register_reviewers,
    replace_assignments,
)
from .study import create_study_bundle, load_samples
from .workflow import run_judge_session


app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


def _balanced_assignment(sample_ids: List[str], reviewers: List[str], min_reviewers: int, seed: int) -> dict[str, List[str]]:
    rng = random.Random(seed)
    sample_ids = list(sample_ids)
    rng.shuffle(sample_ids)
    assignment_map = {reviewer: [] for reviewer in reviewers}
    load = {reviewer: 0 for reviewer in reviewers}
    for sample_id in sample_ids:
        ordered = sorted(reviewers, key=lambda reviewer: (load[reviewer], reviewer))
        chosen = ordered[:min_reviewers]
        for reviewer in chosen:
            assignment_map[reviewer].append(sample_id)
            load[reviewer] += 1
    return assignment_map


@app.command("study-init")
def study_init(
    input_jsonl: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    study_dir: Path = typer.Option(...),
    study_name: str = typer.Option(...),
    overwrite: bool = typer.Option(False, help="Overwrite an existing study dir."),
) -> None:
    manifest = create_study_bundle(input_jsonl, study_dir, study_name, overwrite=overwrite)
    init_db(study_dir)
    append_event(study_dir, "study_init", {"study_name": study_name, "source_input_jsonl": str(input_jsonl)})
    typer.echo(json.dumps(manifest, indent=2))


@app.command("build-assignments")
def build_assignments(
    study_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    reviewer: List[str] = typer.Option(..., help="Repeat --reviewer for each evaluator."),
    sample_limit: Optional[int] = typer.Option(None, help="Assign only the first N samples after study ordering."),
    sample_ids_file: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
    min_reviewers: int = typer.Option(2, min=1),
    seed: int = typer.Option(7),
) -> None:
    samples = load_samples(study_dir)
    sample_ids = [sample["sample_id"] for sample in samples]
    if sample_ids_file is not None:
        with sample_ids_file.open("r", encoding="utf-8") as handle:
            requested = [line.strip() for line in handle if line.strip()]
        requested_set = set(requested)
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in requested_set]
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]
    if min_reviewers > len(reviewer):
        raise typer.BadParameter("min_reviewers cannot exceed number of reviewers.")
    register_reviewers(study_dir, reviewer)
    assignment_map = _balanced_assignment(sample_ids, reviewer, min_reviewers=min_reviewers, seed=seed)
    replace_assignments(study_dir, assignment_map)
    append_event(
        study_dir,
        "build_assignments",
        {
            "reviewers": reviewer,
            "sample_count": len(sample_ids),
            "min_reviewers": min_reviewers,
            "seed": seed,
        },
    )
    out_path = study_dir / "assignments" / "assignments.json"
    out_path.write_text(json.dumps(assignment_map, indent=2), encoding="utf-8")
    typer.echo(f"Wrote assignments to {out_path}")


@app.command("judge")
def judge(
    study_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    reviewer: str = typer.Option(...),
) -> None:
    sample_ids = get_assignments(study_dir, reviewer)
    run_judge_session(study_dir, reviewer, sample_ids, include_submitted=False)


@app.command("review")
def review(
    study_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    reviewer: str = typer.Option(...),
) -> None:
    sample_ids = get_assignments(study_dir, reviewer)
    run_judge_session(study_dir, reviewer, sample_ids, include_submitted=True)


@app.command("export")
def export_cmd(
    study_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    paths = export_active_judgments(study_dir)
    typer.echo(json.dumps(paths, indent=2))


def main() -> None:
    app()
