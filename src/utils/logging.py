"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys

from src.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a consistently-configured logger for *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
