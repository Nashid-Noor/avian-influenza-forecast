"""I/O helpers for data artefacts and model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV with sensible defaults and informative logging."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    log.info("Loading CSV: %s", path)
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Persist a DataFrame to CSV, creating parent dirs as needed."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)
    log.info("Saved CSV (%d rows): %s", len(df), path)


def save_json(data: dict, path: Path) -> None:
    """Write a JSON file."""
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    log.info("Saved JSON: %s", path)


def load_json(path: Path) -> dict:
    """Read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path) as fh:
        return json.load(fh)
