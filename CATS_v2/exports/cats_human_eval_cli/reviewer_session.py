from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cats_human_eval.storage import get_assignment_progress, get_assignments
from cats_human_eval.study import load_manifest
from cats_human_eval.workflow import run_judge_session


KNOWN_MODES = {"judge", "review"}


def discover_study_dir(root: Path) -> Path:
    direct = root / "study"
    if direct.exists():
        return direct
    studies_root = root / "studies"
    candidates = sorted([path for path in studies_root.iterdir() if path.is_dir()]) if studies_root.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No study directory found. Expected ./study or exactly one folder inside ./studies.")
    raise RuntimeError(
        "Multiple study directories found. Package should contain exactly one study bundle or a ./study symlink/copy."
    )


def load_registered_reviewers(study_dir: Path) -> list[str]:
    conn = sqlite3.connect(study_dir / "state" / "judgments.sqlite3")
    try:
        rows = conn.execute("SELECT reviewer_id FROM reviewers ORDER BY reviewer_id").fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


def normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def resolve_reviewer(raw_name: str, reviewer_ids: Iterable[str]) -> Optional[str]:
    normalized = normalize_name(raw_name)
    reviewer_ids = list(reviewer_ids)
    if not normalized:
        return None
    exact = [reviewer for reviewer in reviewer_ids if normalize_name(reviewer) == normalized]
    if len(exact) == 1:
        return exact[0]
    prefix = [reviewer for reviewer in reviewer_ids if normalize_name(reviewer).startswith(normalized)]
    if len(prefix) == 1:
        return prefix[0]
    token = [reviewer for reviewer in reviewer_ids if normalized in normalize_name(reviewer).split()]
    if len(token) == 1:
        return token[0]
    return None


def prompt_reviewer(reviewer_ids: list[str]) -> str:
    while True:
        raw_name = input("Reviewer first name: ")
        resolved = resolve_reviewer(raw_name, reviewer_ids)
        if resolved is not None:
            return resolved
        print("")
        print("Reviewer not recognized. Known reviewers are:")
        for reviewer in reviewer_ids:
            print(f"  - {reviewer}")
        print("")


def render_intro(study_dir: Path, reviewer: str, mode: str) -> None:
    manifest = load_manifest(study_dir)
    progress = get_assignment_progress(study_dir, reviewer)
    total = len(get_assignments(study_dir, reviewer))
    print("")
    print("=" * 92)
    print("CATS HUMAN EVAL LAUNCHER")
    print("=" * 92)
    print(f"Study:     {manifest['study_name']}")
    print(f"Reviewer:  {reviewer}")
    print(f"Mode:      {mode}")
    print(f"Assigned:  {total}")
    print(f"Submitted: {progress['submitted']}")
    print(f"Drafts:    {progress['drafts']}")
    print("=" * 92)
    print("")


def main(argv: list[str]) -> int:
    mode = argv[1].strip().lower() if len(argv) > 1 else "judge"
    if mode not in KNOWN_MODES:
        valid = ", ".join(sorted(KNOWN_MODES))
        print(f"Unknown mode '{mode}'. Use one of: {valid}.")
        return 2

    study_dir = discover_study_dir(ROOT)
    reviewer_ids = load_registered_reviewers(study_dir)
    if not reviewer_ids:
        print("No registered reviewers found in the study bundle.")
        return 2

    reviewer = prompt_reviewer(reviewer_ids)
    render_intro(study_dir, reviewer, mode)
    sample_ids = get_assignments(study_dir, reviewer)
    include_submitted = mode == "review"
    run_judge_session(study_dir, reviewer, sample_ids, include_submitted=include_submitted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
