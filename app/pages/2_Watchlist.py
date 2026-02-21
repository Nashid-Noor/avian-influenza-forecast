"""Page 2: Watchlist — starred countries + top rising-risk hotspots."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    BASELINE_WINDOW,
    MODEL_ARTIFACT_PATH,
    WEEKLY_OUTBREAKS_PATH,
)
from src.forecast import forecast_country, load_model, compute_risk_level

st.title("⭐ Watchlist & Rising Risk")

# ── Load data & model ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data…")
def _load_weekly() -> pd.DataFrame:
    if not WEEKLY_OUTBREAKS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(WEEKLY_OUTBREAKS_PATH, parse_dates=["week_start"])


@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None


weekly = _load_weekly()
booster, meta = _load_model()

if weekly.empty or booster is None:
    st.error("Data or model not found. Run `make preprocess` and `make train` first.")
    st.stop()

countries = sorted(weekly["country"].unique())

# ── Session state for starred countries ──────────────────────────────────────
if "starred" not in st.session_state:
    st.session_state.starred = set()

st.subheader("Star countries to watch")

selected = st.multiselect(
    "Add countries to watchlist",
    countries,
    default=sorted(st.session_state.starred & set(countries)),
)
st.session_state.starred = set(selected)

residual_std = float(meta.get("residual_std", 1.0))

# ── Starred forecast table ───────────────────────────────────────────────────
if st.session_state.starred:
    st.markdown("### Watchlist Forecasts")
    rows: list[dict] = []
    for c in sorted(st.session_state.starred):
        try:
            fc = forecast_country(weekly, c, booster, meta, horizon=4, residual_std=residual_std)
            rows.append({
                "Country": c,
                "Risk": fc["risk_level"].iloc[0],
                "Change": f"{fc['pct_change'].iloc[0]:+.1%}",
                "Forecast (mean)": round(fc["forecast"].mean(), 1),
            })
        except Exception:
            rows.append({"Country": c, "Risk": "—", "Change": "—", "Forecast (mean)": "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No countries starred yet. Use the selector above to add some.")

# ── Top Rising Risk ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔺 Top Rising Risk Countries")
st.caption("Countries with the largest % increase in forecasted outbreaks vs. last 8-week mean.")


@st.cache_data(show_spinner="Computing risk rankings…", ttl=600)
def _compute_risk_table(_weekly_bytes: bytes, _meta_str: str) -> pd.DataFrame:
    """Compute risk for all countries. Cached by data hash."""
    _weekly = pd.read_csv(io.BytesIO(_weekly_bytes), parse_dates=["week_start"])
    _booster, _meta = load_model()
    _residual = float(_meta.get("residual_std", 1.0))
    _countries = sorted(_weekly["country"].unique())
    rows = []
    for c in _countries:
        try:
            fc = forecast_country(_weekly, c, _booster, _meta, horizon=4, residual_std=_residual)
            risk = fc["risk_level"].iloc[0]
            pct = fc["pct_change"].iloc[0]
            rows.append({"Country": c, "Risk": risk, "% Change": pct})
        except Exception:
            continue
    return pd.DataFrame(rows)


risk_df = _compute_risk_table(
    weekly.to_csv(index=False).encode(),
    str(meta),
)

if not risk_df.empty:
    rising = (
        risk_df.sort_values("% Change", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    rising["% Change"] = rising["% Change"].map(lambda x: f"{x:+.1%}")
    badge_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    rising["Risk"] = rising["Risk"].map(lambda r: f"{badge_map.get(r, '⚪')} {r}")
    st.dataframe(rising, use_container_width=True, hide_index=True)
else:
    st.info("Unable to compute risk rankings. Ensure model and data are available.")
