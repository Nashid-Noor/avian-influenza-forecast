"""Pluggable data ingestion for outbreak event records.

Supports three source layouts:
  1. Local generic CSV  (``events.csv``)
  2. FAO EMPRES-i export (``empres_events.csv``)
  3. Kaggle EMPRES dump  (``kaggle_empres.csv``)

Each loader normalises to a common schema:
    date (datetime64), country (str), disease (str),
    cases (float, nullable), deaths (float, nullable), species (str, nullable)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    EMPRES_CSV_NAME,
    KAGGLE_CSV_NAME,
    LOCAL_CSV_NAME,
)
from src.utils.dates import parse_dates_robust
from src.utils.io import load_csv
from src.utils.logging import get_logger

log = get_logger(__name__)

# ── Column mappings ──────────────────────────────────────────────────────────
# Maps common alternate column names → canonical names.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "date": [
        "observation_date",
        "observationdate",
        "report_date",
        "reportingdate",
        "event_date",
        "observation date",
        "report date",
        "reporting_date",
        "start_date",
        "Event.Date",
    ],
    "country": [
        "country_name",
        "Country",
        "country_code",
        "location",
        "region",
        "geo_name",
        "Admin0",
    ],
    "disease": [
        "Disease",
        "disease_name",
        "diagnosis",
        "Diagnosis",
        "disease_eng",
    ],
    "cases": [
        "Cases",
        "num_cases",
        "total_cases",
        "cases_reported",
        "outbreak_cases",
        "sumcases",
    ],
    "deaths": [
        "Deaths",
        "num_deaths",
        "total_deaths",
        "sumdeaths",
    ],
    "species": [
        "Species",
        "animal",
        "host",
        "species_name",
        "speciesdescription",
        "serotypes",
    ],
}

# Lightweight country-name normalisation
_COUNTRY_MAP: dict[str, str] = {
    "usa": "United States of America",
    "us": "United States of America",
    "united states": "United States of America",
    "uk": "United Kingdom",
    "republic of korea": "South Korea",
    "korea (rep. of)": "South Korea",
    "russian federation": "Russia",
    "viet nam": "Vietnam",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "iran (islamic republic of)": "Iran",
    "china, people's republic of": "China",
    "chinese taipei": "Taiwan",
    "hong kong sar": "Hong Kong",
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using alias table."""
    col_lower = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns={c: col_lower[c] for c in df.columns})

    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            norm = alias.strip().lower().replace(" ", "_")
            if norm in df.columns:
                rename_map[norm] = canonical
                break
    return df.rename(columns=rename_map)


def _normalise_country(name: str) -> str:
    """Best-effort country-name normalisation."""
    key = name.strip().lower()
    return _COUNTRY_MAP.get(key, name.strip().title())


def _validate_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ensure required columns exist after mapping."""
    required = {"date", "country", "disease"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{source}] Missing required columns after mapping: {missing}. "
            f"Available: {sorted(df.columns)}"
        )
    return df


def _common_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Shared post-processing: parse dates, normalise country names."""
    df["date"] = parse_dates_robust(df["date"])
    df = df.dropna(subset=["date"])
    df["country"] = df["country"].astype(str).map(_normalise_country)
    df["disease"] = df["disease"].astype(str).str.strip()

    for col in ("cases", "deaths"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")
    if "species" not in df.columns:
        df["species"] = None

    return df[["date", "country", "disease", "cases", "deaths", "species"]]


# ── Public loaders ───────────────────────────────────────────────────────────

def load_local_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Load a generic local CSV (Option 1)."""
    path = path or DATA_RAW_DIR / LOCAL_CSV_NAME
    df = load_csv(path)
    df = _resolve_columns(df)
    df = _validate_schema(df, "local_csv")
    return _common_postprocess(df)


def load_empres_export(path: Optional[Path] = None) -> pd.DataFrame:
    """Load an FAO EMPRES-i export file (Option 2).

    Expected columns (or close variants):
        observation_date | country | disease | cases | deaths | species
    """
    path = path or DATA_RAW_DIR / EMPRES_CSV_NAME
    df = load_csv(path)
    df = _resolve_columns(df)
    df = _validate_schema(df, "empres_export")
    return _common_postprocess(df)


def load_kaggle_empres(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Kaggle EMPRES dataset CSV (Option 3)."""
    path = path or DATA_RAW_DIR / KAGGLE_CSV_NAME
    df = load_csv(path)
    df = _resolve_columns(df)
    df = _validate_schema(df, "kaggle_empres")
    return _common_postprocess(df)


def auto_ingest(raw_dir: Optional[Path] = None) -> tuple[pd.DataFrame, str]:
    """Auto-detect and load the best available data source.

    Returns:
        Tuple of (DataFrame, source_name).
    """
    raw_dir = raw_dir or DATA_RAW_DIR
    loaders: list[tuple[str, Path, callable]] = [
        ("empres_export", raw_dir / EMPRES_CSV_NAME, load_empres_export),
        ("kaggle_empres", raw_dir / KAGGLE_CSV_NAME, load_kaggle_empres),
        ("local_csv", raw_dir / LOCAL_CSV_NAME, load_local_csv),
    ]
    for name, path, loader in loaders:
        if path.exists():
            log.info("Auto-detected source: %s (%s)", name, path)
            return loader(path), name

    # Search for any CSV
    csvs = sorted(raw_dir.glob("*.csv"))
    if csvs:
        log.warning("No standard file found; trying first CSV: %s", csvs[0])
        return load_local_csv(csvs[0]), "fallback_csv"

    raise FileNotFoundError(
        f"No data files found in {raw_dir}. "
        "Place events.csv, empres_events.csv, or kaggle_empres.csv there."
    )
