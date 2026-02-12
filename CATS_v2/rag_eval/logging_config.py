# rag_eval/logging_config.py
# -*- coding: utf-8 -*-
"""
Enhanced Logging Configuration for CATS v2.0
--------------------------------------------
Provides structured logging with colors and detailed tracking.
"""

import logging
import sys
from pathlib import Path

# Create logger
logger = logging.getLogger("CATS_v2")
logger.setLevel(logging.INFO)

# Console handler with colors
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Detailed format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# File handler for errors
def setup_file_logging(log_dir: str = "logs"):
    """Setup file logging for errors and detailed traces."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(f"{log_dir}/cats_eval.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Separate error log
    error_handler = logging.FileHandler(f"{log_dir}/cats_errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

# Expose the logger
__all__ = ["logger", "setup_file_logging"]
