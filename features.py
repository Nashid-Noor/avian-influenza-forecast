"""Feature engineering for time-series outbreak forecasting.

All features for week *t* are computed using data from weeks **≤ t − 1** to
prevent data leakage during backtesting.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import LAG_WEEKS, ROLLING_STATS, ROLLING_WINDOWS, SEASONALITY_PERIOD
from src.utils.logging import get_logger

log = get_logger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    target_col: str = "outbreaks",
) -> pd.DataFrame:
    """Add lagged versions of *target_col* per country.

    ``lag_1`` means the value from 1 week prior, etc.
    """
    lags = lags or LAG_WEEKS
    df = df.sort_values(["country", "week_start"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("country")[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    stats: list[str] | None = None,
    target_col: str = "outbreaks",
) -> pd.DataFrame:
    """Add rolling statistics over *target_col*, computed from lagged data.

    The rolling window is applied to the **lag-1 shifted** series so that
    the feature for week *t* uses data up to week *t − 1*.
    """
    windows = windows or ROLLING_WINDOWS
    stats = stats or ROLLING_STATS
    df = df.sort_values(["country", "week_start"]).copy()
    shifted = df.groupby("country")[target_col].shift(1)
    for win in windows:
        rolling = shifted.groupby(df["country"]).rolling(win, min_periods=1)
        for stat in stats:
            col_name = f"roll_{stat}_{win}"
            if stat == "mean":
                df[col_name] = rolling.mean().reset_index(level=0, drop=True)
            elif stat == "std":
                vals = rolling.std().reset_index(level=0, drop=True)
                df[col_name] = vals.fillna(0.0)
            else:
                raise ValueError(f"Unsupported rolling stat: {stat}")
    return df


def add_seasonality_features(
    df: pd.DataFrame,
    period: int = SEASONALITY_PERIOD,
) -> pd.DataFrame:
    """Add sin/cos encoding of the ISO week-of-year."""
    df = df.copy()
    woy = df["week_start"].dt.isocalendar().week.astype(float)
    df["woy_sin"] = np.sin(2 * math.pi * woy / period)
    df["woy_cos"] = np.cos(2 * math.pi * woy / period)
    return df


def build_features(
    df: pd.DataFrame,
    target_col: str = "outbreaks",
) -> pd.DataFrame:
    """Full feature-engineering pipeline (lags + rolling + seasonality).

    Rows with any NaN feature are **dropped** — these are the initial rows
    where lags are undefined.

    Returns the DataFrame sorted by ``(country, week_start)`` with new
    feature columns appended.
    """
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_features(df, target_col=target_col)
    df = add_seasonality_features(df)
    feature_cols = get_feature_columns(df)
    n_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    log.info(
        "Built %d features; dropped %d rows with NaN → %d rows remain.",
        len(feature_cols),
        n_before - len(df),
        len(df),
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of model-input feature column names."""
    exclude = {"country", "week_start", "outbreaks"}
    return [c for c in df.columns if c not in exclude]
