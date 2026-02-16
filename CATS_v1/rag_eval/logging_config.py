# logging_config.py
# -*- coding: utf-8 -*-
"""
Logging configuration for RAG Mixed Evaluation Toolkit
------------------------------------------------------

Provides a root logger with colorized console output and optional
file logging. Import `logger` anywhere in the project to log messages.

Authors: Gorang Mehrishi, Samyek Jain
Institution: Birla Institute of Technology and Science, Pilani
"""

import logging
import os
from typing import Optional
import colorlog


def setup_logger(name: Optional[str] = None,
                 log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: optional logger name; None → root logger
        log_file: if set, also write logs to this file
        level: base log level (default: INFO)

    Returns:
        logging.Logger
    """
    # Log format
    fmt_string = "%(log_color)s%(asctime)s - %(levelname)-8s - %(message)s"
    log_colors = {
        "DEBUG": "white",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    # Clear old handlers if re-initialized
    root_logger = logging.getLogger(name)
    if root_logger.handlers:
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)

    # Console handler with color
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(fmt_string, log_colors=log_colors))

    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_fmt = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)

    return root_logger


# Determine default log level from environment variable
_level_str = os.environ.get("RAG_MIXED_EVAL_LOGLEVEL", "INFO").upper()
_level = getattr(logging, _level_str, logging.INFO)

# Root logger to be imported across the project
logger = setup_logger(name=None, log_file=None, level=_level)
