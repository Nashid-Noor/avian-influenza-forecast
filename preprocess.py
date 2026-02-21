"""Preprocessing: filter to Avian Influenza, aggregate to weekly outbreak counts."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config import DISEASE_ALIASES, WEEKLY_OUTBREAKS_PATH
from src.utils.dates import full_weekly_index, iso_week_start
from src.utils.io import save_csv
from src.utils.logging import get_logger

log = get_logger(__name__)


def filter_avian_influenza(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose ``disease`` column matches known AI aliases.

    Matching is case-insensitive and checks whether any alias is a
    *substring* of the disease value, so ``"Highly Pathogenic Avian
    Influenza (poultry)"`` still matches.
    """
    disease_lower = df["disease"].str.lower().str.strip()
    mask = pd.Series(False, index=df.index)
    for alias in DISEASE_ALIASES:
        mask |= disease_lower.str.contains(alias, na=False, regex=False)
    filtered = df.loc[mask].copy()
    log.info(
        "Filtered to Avian Influenza: %d / %d rows kept.",
        len(filtered),
        len(df),
    )
    return filtered


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event rows to weekly outbreak counts per country.

    Each unique event/row counts as **1 outbreak** regardless of ``cases``
    column (which may be absent or incomplete).

    Returns a DataFrame with columns: ``country``, ``week_start``, ``outbreaks``.
    """
    df = df.copy()
    df["week_start"] = df["date"].apply(lambda d: iso_week_start(d))
    weekly = (
        df.groupby(["country", "week_start"])
        .size()
        .reset_index(name="outbreaks")
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    return weekly


def fill_missing_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """Ensure every country has a contiguous weekly index, filling gaps with 0.

    Uses the global date range across all countries so that each country
    covers the same span.
    """
    if weekly.empty:
        return weekly

    global_min = weekly["week_start"].min().date()
    global_max = weekly["week_start"].max().date()
    full_idx = full_weekly_index(global_min, global_max)

    countries = weekly["country"].unique()
    frames: list[pd.DataFrame] = []
    for country in countries:
        sub = weekly.loc[weekly["country"] == country, ["week_start", "outbreaks"]]
        sub = sub.set_index("week_start")
        sub = sub.reindex(full_idx)
        sub["outbreaks"] = sub["outbreaks"].fillna(0).astype(int)
        sub["country"] = country
        sub.index.name = "week_start"
        frames.append(sub.reset_index())
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["country", "week_start"]).reset_index(drop=True)
    log.info(
        "Filled weekly index: %d countries, %d total rows.",
        len(countries),
        len(result),
    )
    return result[["country", "week_start", "outbreaks"]]


def run_preprocessing(events: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline: filter → aggregate → fill → save.

    Args:
        events: Raw event-level DataFrame (output of ingest).
        save: Whether to persist to ``WEEKLY_OUTBREAKS_PATH``.

    Returns:
        Weekly outbreak DataFrame.
    """
    filtered = filter_avian_influenza(events)
    if filtered.empty:
        log.warning("No Avian Influenza events found after filtering.")
        return pd.DataFrame(columns=["country", "week_start", "outbreaks"])
    weekly = aggregate_weekly(filtered)
    weekly = fill_missing_weeks(weekly)
    if save:
        save_csv(weekly, WEEKLY_OUTBREAKS_PATH)
    return weekly
