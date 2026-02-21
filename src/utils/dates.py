"""Date parsing and ISO-week helpers."""

from __future__ import annotations

import datetime as dt

import pandas as pd

_DATE_FORMATS: list[str] = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d %b %Y",
    "%d %B %Y",
]


def parse_dates_robust(series: pd.Series) -> pd.Series:
    """Try ``pd.to_datetime`` with *infer*, then fall back to manual formats.

    Returns a :class:`pd.Series` of ``datetime64[ns]`` with unparseable entries
    set to ``NaT``.
    """
    result = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    mask_nat = result.isna() & series.notna()
    if mask_nat.any():
        for fmt in _DATE_FORMATS:
            still_missing = result.isna() & series.notna()
            if not still_missing.any():
                break
            result = result.fillna(
                pd.to_datetime(series, format=fmt, errors="coerce")
            )
    return result


def iso_week_start(date: dt.date | pd.Timestamp) -> dt.date:
    """Return the Monday that starts the ISO week containing *date*."""
    if isinstance(date, pd.Timestamp):
        date = date.date()
    return date - dt.timedelta(days=date.weekday())


def full_weekly_index(
    start: dt.date, end: dt.date
) -> pd.DatetimeIndex:
    """Return a complete Monday-based weekly date index from *start* to *end*."""
    start = iso_week_start(start)
    end = iso_week_start(end)
    return pd.date_range(start, end, freq="W-MON")
