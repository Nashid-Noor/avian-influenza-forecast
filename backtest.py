"""Walk-forward backtesting and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_SUMMARY_PATH,
    BASELINE_WINDOW,
    FIGURES_DIR,
    LGBM_PARAMS,
    TOP_N_COUNTRIES,
)
from src.features import build_features, get_feature_columns
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

log = get_logger(__name__)


def _baseline_forecast(series: pd.Series, window: int = BASELINE_WINDOW) -> float:
    """Rolling-mean baseline: mean of the last *window* values."""
    tail = series.iloc[-window:]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def walk_forward_backtest(
    weekly: pd.DataFrame,
    n_splits: int = 8,
    min_train_weeks: int = 52,
) -> pd.DataFrame:
    """Country-level rolling-origin evaluation.

    For each split the model is retrained on all data up to the split point
    and tested on the next week's outbreaks.

    Returns a DataFrame of predictions:
        ``(country, week_start, actual, pred_baseline, pred_lgbm)``
    """
    featured = build_features(weekly)
    feature_cols = get_feature_columns(featured)
    all_weeks = sorted(featured["week_start"].unique())

    if len(all_weeks) < min_train_weeks + n_splits:
        n_splits = max(1, len(all_weeks) - min_train_weeks)
        log.warning("Reduced n_splits to %d due to limited data.", n_splits)

    test_weeks = all_weeks[-n_splits:]
    results: list[dict] = []

    for tw in test_weeks:
        train = featured[featured["week_start"] < tw]
        test = featured[featured["week_start"] == tw]
        if train.empty or test.empty:
            continue

        # LightGBM
        X_tr = train[feature_cols]
        y_tr = train["outbreaks"]
        X_te = test[feature_cols]

        params = {k: v for k, v in LGBM_PARAMS.items()
                  if k not in ("n_estimators", "early_stopping_round")}
        model = lgb.LGBMRegressor(n_estimators=200, **params)
        model.fit(X_tr, y_tr)
        preds_lgbm = model.predict(X_te).clip(0)

        for idx, row in test.iterrows():
            country = row["country"]
            country_train = train[train["country"] == country]["outbreaks"]
            bl = _baseline_forecast(country_train)
            results.append({
                "country": country,
                "week_start": row["week_start"],
                "actual": row["outbreaks"],
                "pred_baseline": round(bl, 4),
                "pred_lgbm": round(float(preds_lgbm[test.index.get_loc(idx)]), 4),
            })

    df = pd.DataFrame(results)
    log.info("Backtest complete: %d predictions over %d splits.", len(df), n_splits)
    return df


def compute_metrics(bt: pd.DataFrame) -> dict:
    """Compute MAE and RMSE for baseline and LightGBM, global and per-country."""
    def _metrics(actual: pd.Series, pred: pd.Series) -> dict:
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        return {"mae": round(mae, 4), "rmse": round(rmse, 4)}

    global_bl = _metrics(bt["actual"], bt["pred_baseline"])
    global_lgb = _metrics(bt["actual"], bt["pred_lgbm"])

    per_country: dict[str, dict] = {}
    for country, grp in bt.groupby("country"):
        per_country[str(country)] = {
            "baseline": _metrics(grp["actual"], grp["pred_baseline"]),
            "lgbm": _metrics(grp["actual"], grp["pred_lgbm"]),
        }

    best = "lgbm" if global_lgb["mae"] < global_bl["mae"] else "baseline"
    if global_lgb["mae"] == global_bl["mae"]:
        best = "lgbm" if global_lgb["rmse"] <= global_bl["rmse"] else "baseline"

    return {
        "global": {"baseline": global_bl, "lgbm": global_lgb},
        "per_country": per_country,
        "best_model": best,
    }


def write_backtest_report(metrics: dict, path: Optional[Path] = None) -> None:
    """Write ``backtest_summary.md``."""
    path = path or BACKTEST_SUMMARY_PATH
    ensure_dir(path.parent)
    g = metrics["global"]
    lines = [
        "# Backtest Summary",
        "",
        f"**Best model:** {metrics['best_model']}",
        "",
        "## Global Metrics",
        "",
        "| Model    | MAE    | RMSE   |",
        "|----------|--------|--------|",
        f"| Baseline | {g['baseline']['mae']:.4f} | {g['baseline']['rmse']:.4f} |",
        f"| LightGBM | {g['lgbm']['mae']:.4f} | {g['lgbm']['rmse']:.4f} |",
        "",
        "## Per-Country Metrics (top by LightGBM MAE)",
        "",
    ]
    sorted_countries = sorted(
        metrics["per_country"].items(),
        key=lambda x: x[1]["lgbm"]["mae"],
        reverse=True,
    )
    lines.append("| Country | BL MAE | LGB MAE | BL RMSE | LGB RMSE |")
    lines.append("|---------|--------|---------|---------|----------|")
    for country, m in sorted_countries[:15]:
        lines.append(
            f"| {country} | {m['baseline']['mae']:.4f} | {m['lgbm']['mae']:.4f} "
            f"| {m['baseline']['rmse']:.4f} | {m['lgbm']['rmse']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")
    log.info("Backtest report → %s", path)


def plot_top_countries(bt: pd.DataFrame, n: int = TOP_N_COUNTRIES) -> None:
    """Save actual-vs-predicted plots for the top *n* countries by volume."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots.")
        return

    ensure_dir(FIGURES_DIR)
    top = (
        bt.groupby("country")["actual"]
        .sum()
        .nlargest(n)
        .index.tolist()
    )
    for country in top:
        sub = bt[bt["country"] == country].sort_values("week_start")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sub["week_start"], sub["actual"], "ko-", label="Actual", ms=4)
        ax.plot(sub["week_start"], sub["pred_baseline"], "b--", label="Baseline", alpha=0.7)
        ax.plot(sub["week_start"], sub["pred_lgbm"], "r-", label="LightGBM", alpha=0.8)
        ax.set_title(f"Backtest — {country}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Outbreaks")
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe_name = country.replace(" ", "_").replace("/", "_")
        fig.savefig(FIGURES_DIR / f"backtest_{safe_name}.png", dpi=120)
        plt.close(fig)
    log.info("Saved backtest plots for top %d countries.", n)


def residual_std(bt: pd.DataFrame, model: str = "lgbm") -> float:
    """Global residual standard deviation for uncertainty intervals."""
    residuals = bt["actual"] - bt[f"pred_{model}"]
    return float(residuals.std()) if len(residuals) > 1 else 1.0


def run_backtest(weekly: pd.DataFrame, n_splits: int = 8) -> dict:
    """Execute full backtest pipeline and return metrics dict."""
    bt = walk_forward_backtest(weekly, n_splits=n_splits)
    if bt.empty:
        log.warning("Backtest produced no predictions.")
        return {"global": {}, "per_country": {}, "best_model": "baseline"}
    metrics = compute_metrics(bt)
    write_backtest_report(metrics)
    plot_top_countries(bt)

    # Store residual_std in metrics for use by forecast
    metrics["residual_std"] = residual_std(bt, model=metrics["best_model"])
    return metrics
