#!/usr/bin/env python3
"""
Build a self-sufficient export pack for the training conflict-type review job.

The pack mirrors the benchmark preselection handoff style:
- reviewer-ready README
- only the required scripts
- training review assignments
- the training-set stage3_final input file
- empty review output directory

Run:
    python3 scripts/export_training_conflict_type_review_pack.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
PACK_DATE = datetime.now().strftime("%Y-%m-%d")
PACK_NAME = f"training_conflict_type_review_annotator_pack_{PACK_DATE}"
PACK_DIR = EXPORTS_DIR / PACK_NAME
ZIP_PATH = EXPORTS_DIR / f"{PACK_NAME}.zip"

SOURCE_INPUT = PROJECT_ROOT / "data/final_annotations/stagewise_multi/stage3_final.jsonl"
SOURCE_ASSIGNMENTS_DIR = PROJECT_ROOT / "human_reviews/training/assignments"

SCRIPT_SOURCES = [
    PROJECT_ROOT / "scripts/training_conflict_type_review_cli.py",
    PROJECT_ROOT / "scripts/benchmark_human_preselection_cli.py",
    PROJECT_ROOT / "scripts/run_training_conflict_type_review.sh",
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_readme() -> str:
    return """# Training Conflict-Type Review Pack

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

Open a terminal in this folder and run:

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
"""


def main() -> None:
    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    (PACK_DIR / "scripts").mkdir(parents=True, exist_ok=True)
    data_root = PACK_DIR / "data/final_annotations/stagewise_multi"
    assignments_dir = data_root / "conflict_type_review/assignments"
    reviews_dir = data_root / "conflict_type_review/reviews"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_INPUT, data_root / "stage3_final.jsonl")

    for script_path in SCRIPT_SOURCES:
        shutil.copy2(script_path, PACK_DIR / "scripts" / script_path.name)

    for src in sorted(SOURCE_ASSIGNMENTS_DIR.glob("*.json")):
        dst = assignments_dir / src.name
        shutil.copy2(src, dst)
        if src.name.startswith("reviewer_") and src.name.endswith("_ids.json"):
            payload = json.loads(dst.read_text(encoding="utf-8"))
            reviewer_id = int(payload.get("reviewer_id"))
            payload["input"] = "data/final_annotations/stagewise_multi/stage3_final.jsonl"
            payload["reviewer_first_name"] = {1: "manan", 2: "atharv", 3: "parth"}.get(
                reviewer_id, f"reviewer_{reviewer_id}"
            )
            dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = assignments_dir / "assignment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["input"] = "data/final_annotations/stagewise_multi/stage3_final.jsonl"
        manifest["reviewer_roster"] = {"1": "manan", "2": "atharv", "3": "parth"}
        manifest["reviewer_files"] = {
            "1": "human_reviews/training/assignments/reviewer_1_ids.json",
            "2": "human_reviews/training/assignments/reviewer_2_ids.json",
            "3": "human_reviews/training/assignments/reviewer_3_ids.json",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_text(PACK_DIR / "README.md", build_readme())

    shutil.make_archive(str(PACK_DIR), "zip", root_dir=EXPORTS_DIR, base_dir=PACK_NAME)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"manifest: {manifest_path}")
        print(f"reviewers: {manifest.get('reviewer_roster', {})}")

    print(f"wrote export pack to {PACK_DIR}")
    print(f"wrote zip to {ZIP_PATH}")


if __name__ == "__main__":
    main()
