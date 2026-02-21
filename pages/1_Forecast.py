"""Page 1: Forecast — country-level outbreak forecast with uncertainty."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path for imports
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    BASELINE_WINDOW,
    MODEL_ARTIFACT_PATH,
    MODEL_META_PATH,
    WEEKLY_OUTBREAKS_PATH,
)
from src.forecast import compute_risk_level, forecast_country, load_model

st.title("📈 Country Forecast")

# ── Load data & model ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading outbreak data…")
def _load_weekly() -> pd.DataFrame:
    if not WEEKLY_OUTBREAKS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(WEEKLY_OUTBREAKS_PATH, parse_dates=["week_start"])
    return df


@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None


weekly = _load_weekly()
booster, meta = _load_model()

if weekly.empty:
    st.error(
        "No processed data found. Run `make preprocess` first to generate "
        "`data/processed/weekly_outbreaks.csv`."
    )
    st.stop()

if booster is None:
    st.error(
        "No trained model found. Run `make train` first to produce the model artefact."
    )
    st.stop()

# ── Country selector ─────────────────────────────────────────────────────────
countries = sorted(weekly["country"].unique())
selected = st.selectbox("Select a country", countries, index=0)

# ── Residual std (from metadata or default) ──────────────────────────────────
residual_std = meta.get("residual_std", 1.0)
if isinstance(residual_std, str):
    residual_std = float(residual_std)

# ── Forecast ─────────────────────────────────────────────────────────────────
try:
    fc = forecast_country(
        weekly, selected, booster, meta,
        horizon=4, residual_std=residual_std,
    )
except Exception as exc:
    st.error(f"Forecast failed: {exc}")
    st.stop()

# ── Recent actuals ───────────────────────────────────────────────────────────
country_data = (
    weekly[weekly["country"] == selected]
    .sort_values("week_start")
    .tail(52)
    .copy()
)

# ── Risk badge ───────────────────────────────────────────────────────────────
risk_level = fc["risk_level"].iloc[0]
pct_change = fc["pct_change"].iloc[0]

badge_colours = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
badge = badge_colours.get(risk_level, "⚪")

col1, col2, col3 = st.columns(3)
col1.metric("Risk Level", f"{badge} {risk_level}")
col2.metric("Change vs last 8-week mean", f"{pct_change:+.1%}")
peak_week = fc.loc[fc["forecast"].idxmax(), "week_start"]
col3.metric("Peak forecast week", str(peak_week.date()) if hasattr(peak_week, "date") else str(peak_week)[:10])

# ── Charts ───────────────────────────────────────────────────────────────────
st.subheader("Recent Outbreaks (last 52 weeks)")
st.line_chart(country_data.set_index("week_start")["outbreaks"], height=250)

st.subheader("4-Week Forecast with Uncertainty")

import altair as alt  # noqa: E402 (conditional import kept after st calls)

fc_plot = fc.copy()
fc_plot["week_start"] = pd.to_datetime(fc_plot["week_start"])

band = (
    alt.Chart(fc_plot)
    .mark_area(opacity=0.25, color="steelblue")
    .encode(
        x=alt.X("week_start:T", title="Week"),
        y=alt.Y("lower:Q", title="Outbreaks"),
        y2="upper:Q",
    )
)
line = (
    alt.Chart(fc_plot)
    .mark_line(point=True, color="steelblue")
    .encode(
        x="week_start:T",
        y=alt.Y("forecast:Q", title="Outbreaks"),
        tooltip=["week_start:T", "forecast:Q", "lower:Q", "upper:Q"],
    )
)
st.altair_chart(band + line, use_container_width=True)

st.dataframe(
    fc[["week_start", "forecast", "lower", "upper"]].rename(
        columns={"week_start": "Week", "forecast": "Forecast", "lower": "Lower 80%", "upper": "Upper 80%"}
    ),
    use_container_width=True,
    hide_index=True,
)
