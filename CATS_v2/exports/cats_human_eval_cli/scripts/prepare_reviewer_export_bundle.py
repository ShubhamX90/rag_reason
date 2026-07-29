from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STUDY_NAME = "qwen_llama_e2e_sft_baseline_balanced_4reviewers"
STUDY_SRC = ROOT / "studies" / STUDY_NAME
DIST_ROOT = ROOT / "dist"
PACKAGE_DIR = DIST_ROOT / f"{STUDY_NAME}_package"
ZIP_BASE = DIST_ROOT / f"{STUDY_NAME}_package"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if not STUDY_SRC.exists():
        raise FileNotFoundError(f"Missing study source directory: {STUDY_SRC}")

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    DIST_ROOT.mkdir(parents=True, exist_ok=True)

    shutil.copytree(ROOT / "cats_human_eval", PACKAGE_DIR / "cats_human_eval", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(STUDY_SRC, PACKAGE_DIR / "study", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    copy_file(ROOT / "pyproject.toml", PACKAGE_DIR / "pyproject.toml")
    copy_file(ROOT / "README.md", PACKAGE_DIR / "README.md")
    copy_file(ROOT / "REVIEWER_USER_MANUAL.md", PACKAGE_DIR / "REVIEWER_USER_MANUAL.md")
    copy_file(ROOT / "reviewer_session.py", PACKAGE_DIR / "reviewer_session.py")
    copy_file(ROOT / "run_reviewer.sh", PACKAGE_DIR / "run_reviewer.sh")
    launcher = PACKAGE_DIR / "run_reviewer.sh"
    launcher.chmod(0o755)

    zip_path = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=DIST_ROOT, base_dir=PACKAGE_DIR.name)
    print(zip_path)


if __name__ == "__main__":
    main()
