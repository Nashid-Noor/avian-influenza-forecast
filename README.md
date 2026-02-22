
#  Avian Influenza Outbreak Forecasting & Risk Intelligence

**Live Project:** (https://huggingface.co/spaces/nashid16/avian-influenza-forecast)

An end-to-end Machine Learning pipeline and interactive risk dashboard designed to forecast weekly Avian Influenza outbreaks globally. This system aggregates raw epidemiological event records, engineers time-series features, and utilizes a boosted tree regression model to serve a 4-week look-ahead forecast alongside a dynamic risk-level classification.

---

##  Architecture & Methodology

1. **Robust Data Ingestion & Preprocessing**
   * Parses raw outbreak event records (e.g., from FAO EMPRES-i or Kaggle).
   * Aggregates irregular, complex event log data into consistent, country-level weekly time-series observations.

2. **Epidemiological Feature Engineering**
   * **Lag Features:** Feeds exactly prior known outbreak counts (e.g., $T-1$, $T-4$ weeks) to capture precise transmission points.
   * **Rolling Statistical Windows:** Generates rolling means and standard deviations to capture sustained momentum, volatility, and historical trends.
   * **Seasonality:** Utilizes cyclic sine/cosine encoding of the IEEE week-of-year to explicitly model predictable biological and migratory patterns.

3. **Time-Series Forecasting & Validation**
   * **Algorithm:** `LightGBM` (Gradient Boosted Trees) Regressor.
   * **Walk-Forward Backtesting:** Discards standard randomized Train/Test splits to prevent data leakage. The model is rigorously evaluated over 8 chronological splits ("walking forward" through time) to simulate true out-of-sample, real-world predictive performance.
   * **Performance:** Evaluated across 11,600+ historical records, achieving a **Global Mean Absolute Error (MAE) of 0.21**, vastly outperforming rolling-mean baseline heuristic models.

4. **Uncertainty Quantification & Risk Classification**
   * **Confidence Bounds:** Calculates the standard deviation of residuals (`residual_std`) generated during the walk-forward backtest to dynamically compute upper and lower statistical uncertainty intervals for all future predictions.
   * **Risk Intelligence Engine:** Calculates the percentage rate of change between the 4-week forecast mean and the recent historical baseline, automatically flagging countries into `High`, `Medium`, or `Low` risk tiers to assist proactive resource allocation.

---

##  Local Quick Setup

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/Nashid-Noor/avian-influenza-forecast.git
   cd avian-influenza-forecast
   pip install -r requirements.txt
   ```

2. **Run the preprocessing and training pipelines via the CLI:**
   ```bash
   # 1. Clean raw data and build time-series features
   python -m src.run preprocess
   
   # 2. Train the model and save artifacts
   python -m src.run train
   
   # 3. (Optional) Run walk-forward validation to generate metrics
   python -m src.run backtest 
   ```

3. **Spin up the Streamlit UI locally:**
   ```bash
   streamlit run app/Home.py
   ```

---

