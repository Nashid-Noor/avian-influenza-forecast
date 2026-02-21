"""Avian Influenza Weekly Outbreak Forecasting System — Streamlit Home."""

import streamlit as st

st.set_page_config(
    page_title="Avian Influenza Forecast",
    page_icon="🐦",
    layout="wide",
)

st.title("🐦 Avian Influenza Weekly Outbreak Forecasting")

st.markdown(
    """
    Welcome to the **Avian Influenza Forecasting System** — a lightweight
    ML-powered tool that forecasts weekly outbreak counts per country over a
    4-week horizon.

    ### How it works
    1. **Ingest** outbreak event records (FAO EMPRES-i, Kaggle, or local CSV).
    2. **Preprocess** to weekly time series per country.
    3. **Train** a LightGBM model on engineered features (lags, rolling stats, seasonality).
    4. **Forecast** with uncertainty intervals and risk-level classification.

    ---

    Use the sidebar to navigate:
    - **Forecast** — Select a country and view the latest 4-week outlook.
    - **Watchlist** — Star countries and see top rising-risk hotspots.
    """
)

st.caption("Built for demonstration purposes. Data must be preprocessed and a model trained before forecasts are available.")
