"""Train the LightGBM regression model and persist artefacts."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import lightgbm as lgb
import pandas as pd

from src.config import (
    LGBM_PARAMS,
    MODEL_ARTIFACT_PATH,
    MODEL_META_PATH,
)
from src.features import build_features, get_feature_columns
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

log = get_logger(__name__)


def train_lgbm(
    weekly: pd.DataFrame,
    val_weeks: int = 12,
    params: Optional[dict] = None,
) -> tuple[lgb.LGBMRegressor, dict]:
    """Train a LightGBM model on the weekly outbreak data.

    The last *val_weeks* of each country are used as a simple
    validation split for early stopping.

    Args:
        weekly: ``(country, week_start, outbreaks)`` DataFrame.
        val_weeks: Number of trailing weeks for validation.
        params: Override default LGBM hyper-parameters.

    Returns:
        Tuple of (fitted model, metadata dict).
    """
    params = {**LGBM_PARAMS, **(params or {})}
    featured = build_features(weekly)
    feature_cols = get_feature_columns(featured)

    # Temporal split
    cutoff = featured["week_start"].max() - pd.Timedelta(weeks=val_weeks)
    train_mask = featured["week_start"] <= cutoff
    val_mask = featured["week_start"] > cutoff

    X_train = featured.loc[train_mask, feature_cols]
    y_train = featured.loc[train_mask, "outbreaks"]
    X_val = featured.loc[val_mask, feature_cols]
    y_val = featured.loc[val_mask, "outbreaks"]

    log.info(
        "Training LightGBM — train=%d, val=%d, features=%d",
        len(X_train),
        len(X_val),
        len(feature_cols),
    )

    n_estimators = params.pop("n_estimators", 300)
    early_stopping_round = params.pop("early_stopping_round", 30)

    model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)

    callbacks = [lgb.early_stopping(early_stopping_round, verbose=False)]
    if X_val.empty:
        callbacks = []
        eval_set = None
    else:
        eval_set = [(X_val, y_val)]

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        callbacks=callbacks,
    )

    # Metadata
    train_end = featured.loc[train_mask, "week_start"].max()
    best_iter = n_estimators
    try:
        if model.best_iteration_ and model.best_iteration_ > 0:
            best_iter = int(model.best_iteration_)
    except AttributeError:
        pass

    meta = {
        "model_name": "lgbm_outbreak_forecaster",
        "version": "1.0.0",
        "train_end_date": str(train_end.date()),
        "data_end_date": str(featured["week_start"].max().date()),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "feature_list": feature_cols,
        "params": {**params, "n_estimators": n_estimators},
        "best_iteration": best_iter,
        "residual_std": 1.0,  # overwritten by run.py after backtest
        "data_source": "unknown",  # overwritten by caller if known
    }
    return model, meta


def save_model(model: lgb.LGBMRegressor, meta: dict) -> None:
    """Save model artefact and metadata JSON."""
    ensure_dir(MODEL_ARTIFACT_PATH.parent)
    model.booster_.save_model(str(MODEL_ARTIFACT_PATH))
    save_json(meta, MODEL_META_PATH)
    log.info("Model saved → %s", MODEL_ARTIFACT_PATH)
