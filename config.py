"""Central configuration for the Avian Influenza Forecasting System."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DATA_RAW_DIR: Final[Path] = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Final[Path] = ROOT_DIR / "data" / "processed"
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
REPORTS_DIR: Final[Path] = ROOT_DIR / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

WEEKLY_OUTBREAKS_PATH: Final[Path] = DATA_PROCESSED_DIR / "weekly_outbreaks.csv"
MODEL_ARTIFACT_PATH: Final[Path] = MODELS_DIR / "lgbm_model.txt"
MODEL_META_PATH: Final[Path] = MODELS_DIR / "model_metadata.json"
BACKTEST_SUMMARY_PATH: Final[Path] = REPORTS_DIR / "backtest_summary.md"

# ── Disease filtering ────────────────────────────────────────────────────────
DISEASE_ALIASES: Final[list[str]] = [
    "avian influenza",
    "highly pathogenic avian influenza",
    "hpai",
    "bird flu",
    "h5n1",
    "h5n8",
    "h5n6",
    "h7n9",
    "lpai",
    "low pathogenic avian influenza",
    "influenza - avian",
]

# ── Feature engineering ──────────────────────────────────────────────────────
LAG_WEEKS: Final[list[int]] = [1, 2, 4, 8, 12]
ROLLING_WINDOWS: Final[list[int]] = [4, 8, 12]
ROLLING_STATS: Final[list[str]] = ["mean", "std"]
SEASONALITY_PERIOD: Final[int] = 52

# ── Modelling ────────────────────────────────────────────────────────────────
BASELINE_WINDOW: Final[int] = 8  # weeks for rolling-mean baseline
DEFAULT_HORIZON: Final[int] = 4
MAX_HORIZON: Final[int] = 12
UNCERTAINTY_Z: Final[float] = 1.28  # ~80 % prediction interval

# ── Risk thresholds ──────────────────────────────────────────────────────────
RISK_HIGH_PCT: Final[float] = 0.50
RISK_MEDIUM_PCT: Final[float] = 0.15

# ── LightGBM hyper-parameters (sensible defaults) ───────────────────────────
LGBM_PARAMS: Final[dict] = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "seed": 42,
    "n_estimators": 300,
    "early_stopping_round": 30,
}

# ── Source file names (conventional) ─────────────────────────────────────────
LOCAL_CSV_NAME: Final[str] = "events.csv"
EMPRES_CSV_NAME: Final[str] = "empres_events.csv"
KAGGLE_CSV_NAME: Final[str] = "kaggle_empres.csv"

# ── Top-N countries for reporting ────────────────────────────────────────────
TOP_N_COUNTRIES: Final[int] = 5

# ── Env overrides ────────────────────────────────────────────────────────────
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")
