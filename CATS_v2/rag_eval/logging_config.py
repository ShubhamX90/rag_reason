# rag_eval/logging_config.py
# -*- coding: utf-8 -*-
"""
Logging configuration for CATS v2.0.

setup_file_logging() is idempotent — repeated calls won't multiply log output,
which was a bug in the previous version when the batch runner processed
multiple input files.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger("CATS_v2")
logger.setLevel(logging.INFO)
logger.propagate = False  # prevent root logger from double-printing

_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _has_handler_for(filename: str) -> bool:
    target = str(Path(filename).resolve())
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if str(Path(h.baseFilename).resolve()) == target:
                    return True
            except Exception:
                continue
    return False


# Console handler — install once at import time, guarded against double-install.
if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
           for h in logger.handlers):
    _console = logging.StreamHandler(sys.stdout)
    _console.setLevel(logging.INFO)
    _console.setFormatter(_FORMATTER)
    logger.addHandler(_console)


def setup_file_logging(log_dir: str = "logs") -> None:
    """Add file + error handlers. Safe to call multiple times — idempotent."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    eval_log = f"{log_dir}/cats_eval.log"
    if not _has_handler_for(eval_log):
        h = logging.FileHandler(eval_log)
        h.setLevel(logging.DEBUG)
        h.setFormatter(_FORMATTER)
        logger.addHandler(h)

    err_log = f"{log_dir}/cats_errors.log"
    if not _has_handler_for(err_log):
        h = logging.FileHandler(err_log)
        h.setLevel(logging.ERROR)
        h.setFormatter(_FORMATTER)
        logger.addHandler(h)


__all__ = ["logger", "setup_file_logging"]
