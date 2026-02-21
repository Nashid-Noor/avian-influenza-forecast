"""Multi-step forecasting with uncertainty intervals and risk levels."""

from __future__ import annotations

import datetime as dt
import math
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    BASELINE_WINDOW,
    DEFAULT_HORIZON,
    MAX_HORIZON,
    MODEL_ARTIFACT_PATH,
    MODEL_META_PATH,
    RISK_HIGH_PCT,
    RISK_MEDIUM_PCT,
    UNCERTAINTY_Z,
)
from src.features import (
    add_lag_features,
    add_rolling_features,
    add_seasonality_features,
    get_feature_columns,
)
from src.utils.io import load_json
from src.utils.logging import get_logger

log = get_logger(__name__)


def load_model() -> tuple[lgb.Booster, dict]:
    """Load persisted LightGBM model and metadata."""
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_ARTIFACT_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_ARTIFACT_PATH))
    meta = load_json(MODEL_META_PATH)
    return booster, meta


def compute_risk_level(
    forecast_mean: float,
    recent_mean: float,
) -> tuple[str, float]:
    """Compare forecast mean to recent actuals mean and return risk label.

    Returns:
        (risk_level, pct_change) — e.g. ("High", 0.75)
    """
    if recent_mean <= 0:
        pct = float("inf") if forecast_mean > 0 else 0.0
    else:
        pct = (forecast_mean - recent_mean) / recent_mean

    if pct >= RISK_HIGH_PCT:
        return "High", round(pct, 4)
    elif pct >= RISK_MEDIUM_PCT:
        return "Medium", round(pct, 4)
    else:
        return "Low", round(pct, 4)


def _rebuild_features_for_country(
    weekly: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """Build feature matrix for a single country (for the forecast row)."""
    sub = weekly[weekly["country"] == country].copy()
    sub = sub.sort_values("week_start").reset_index(drop=True)
    sub = add_lag_features(sub)
    sub = add_rolling_features(sub)
    sub = add_seasonality_features(sub)
    return sub


def forecast_country(
    weekly: pd.DataFrame,
    country: str,
    booster: lgb.Booster,
    meta: dict,
    horizon: int = DEFAULT_HORIZON,
    residual_std: float = 1.0,
) -> pd.DataFrame:
    """Produce a multi-step forecast for *country* by iterative 1-step prediction.

    At each step the prediction is appended to the history so that subsequent
    lag/rolling features incorporate it.

    Returns a DataFrame:
        week_start, forecast, lower, upper, risk_level, pct_change
    """
    horizon = min(horizon, MAX_HORIZON)
    feature_cols = meta["feature_list"]

    sub = weekly[weekly["country"] == country].copy()
    if sub.empty:
        raise ValueError(f"No data for country: {country}")
    sub = sub.sort_values("week_start").reset_index(drop=True)

    last_date = sub["week_start"].max()
    preds: list[dict] = []

    for step in range(1, horizon + 1):
        next_week = last_date + pd.Timedelta(weeks=step)
        new_row = pd.DataFrame({
            "country": [country],
            "week_start": [next_week],
            "outbreaks": [np.nan],
        })
        extended = pd.concat([sub, new_row], ignore_index=True)

        feat = _rebuild_features_for_country(extended, country)
        row = feat.iloc[-1:]

        missing = [c for c in feature_cols if c not in row.columns]
        for c in missing:
            row[c] = 0.0

        X = row[feature_cols].values
        yhat = float(booster.predict(X)[0])
        yhat = max(yhat, 0.0)

        lower = max(yhat - UNCERTAINTY_Z * residual_std, 0.0)
        upper = yhat + UNCERTAINTY_Z * residual_std

        preds.append({
            "week_start": next_week,
            "forecast": round(yhat, 2),
            "lower": round(lower, 2),
            "upper": round(upper, 2),
        })

        # Feed back prediction for next step
        new_row["outbreaks"] = yhat
        sub = pd.concat([sub, new_row], ignore_index=True)

    result = pd.DataFrame(preds)

    # Risk level
    forecast_mean = result["forecast"].mean()
    recent = weekly[weekly["country"] == country].sort_values("week_start")
    recent_actual = recent["outbreaks"].iloc[-BASELINE_WINDOW:]
    recent_mean = float(recent_actual.mean()) if len(recent_actual) > 0 else 0.0
    risk, pct = compute_risk_level(forecast_mean, recent_mean)

    result["risk_level"] = risk
    result["pct_change"] = pct
    return result


def forecast_all_countries(
    weekly: pd.DataFrame,
    booster: lgb.Booster,
    meta: dict,
    horizon: int = DEFAULT_HORIZON,
    residual_std: float = 1.0,
) -> pd.DataFrame:
    """Run forecasts for every country in the dataset."""
    countries = sorted(weekly["country"].unique())
    frames: list[pd.DataFrame] = []
    for c in countries:
        try:
            fc = forecast_country(weekly, c, booster, meta, horizon, residual_std)
            fc.insert(0, "country", c)
            frames.append(fc)
        except Exception as exc:
            log.warning("Forecast failed for %s: %s", c, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
