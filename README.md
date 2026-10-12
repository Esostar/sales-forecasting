# Sales Forecasting & Revenue Intelligence Platform

An end-to-end machine learning system that forecasts daily sales at the store x product-family level, quantifies uncertainty, and exposes results through a REST API and an interactive dashboard.

**Live demo:**
- Dashboard: https://sales-forecasting-xciseueerznvnpvgp6yoje.streamlit.app/
- API docs: https://sales-forecasting-ymo1.onrender.com/docs
- API health: https://sales-forecasting-ymo1.onrender.com/health

Note: the API runs on Render's free tier and sleeps after 15 min of inactivity. First request may take ~30s to wake up.

## Results

| Metric | Model | Baseline (seasonal naive) |
|---|---|---|
| WAPE (store x family, 90-day holdout) | 11.5% | 18.3% |
| MAE | 55.4 | 87.8 |
| RMSE | 221.4 | 335.4 |
| 80% prediction interval coverage | 82.2% | n/a |

Rolling expanding-window validation across 3 folds: mean WAPE 14.9% (std 2.9%).

## Problem

Retail demand is driven by weekly cycles, annual seasonality, promotions, holidays, and payday effects. A single aggregate forecast is not actionable - inventory and staffing decisions happen at the store x product level. This project forecasts 1,782 individual store x family series with calibrated prediction intervals so planners can reason about risk, not just point estimates.

## Dataset

Favorita Store Sales (Kaggle) - 3M daily rows, 54 stores, 33 families, Jan 2013 to Aug 2017. Includes promotions, holidays, oil prices, store metadata, and transaction counts.

## Approach

### 1. EDA (notebooks/01_eda.ipynb)
- Strong weekly seasonality (weekend peaks, Wednesday dip)
- Annual December spike, February trough
- Payday effects on the 1st and 15th
- Store closures on Dec 25 and Jan 1
- 31% zero-sales rows (high intermittency)

### 2. Baselines
- Naive: 15.1% WAPE
- Seasonal naive: 10.9% WAPE (aggregate level)

### 3. Feature engineering (src/features/build_features.py)
- Calendar: year, month, day, dow, week-of-year, payday, month-boundary flags
- Fourier terms (3 harmonics) for annual seasonality
- External: oil price, holiday flags, store cluster, store type
- Lags: 7, 14, 21, 28, 364 days
- Rolling mean/std: 7, 14, 28 days (shifted to prevent leakage)

### 4. Modeling
- Point forecast: XGBoost on log1p(sales), 800 trees, depth 8
- Intervals: 3 quantile XGBoost models (alpha 0.1 / 0.5 / 0.9) - 80% interval
- Target encodings were tried and rejected: anchored to stale multi-year means, degraded WAPE from 11.56% to 12.17%

### 5. Validation (src/models/validate.py)
Three expanding-window folds, 90-day horizons. Mean WAPE 14.9%, std 2.9%.

### 6. Diagnostics (src/models/diagnostics.py)
- Learning curve: train 12.06% vs val 11.79% WAPE - no overfitting
- Residual bias: BEVERAGES under-forecast by ~150 units; store 53 over-forecast by ~27
- Feature importance: ~87% on lag_7 + roll_mean_7

### 7. Drift monitoring (src/monitoring/drift.py)
- PSI on numeric features (calendar and ID features excluded)
- Rolling weekly WAPE as concept-drift proxy
- Flagged: oil (-22% mean shift), onpromotion (+135%)
- Concept drift within horizon: none

## Architecture

Kaggle CSV -> build_features.py -> train.pkl / test.pkl
             -> train.py + train_quantile.py -> models/*.joblib
             -> FastAPI (Render) + Streamlit (Cloud)

## Project structure

sales-forecasting/
  data/
    raw/                       Kaggle CSVs (gitignored)
    processed/                 features, slim history, dashboard artifacts
  notebooks/
    01_eda.ipynb               time-series EDA + baselines
  src/
    features/build_features.py feature pipeline
    models/
      train.py                 point model
      train_quantile.py        interval models
      validate.py              expanding-window validation
      diagnostics.py           learning curve, residual bias
    monitoring/drift.py        PSI + rolling WAPE
    api/main.py                FastAPI service
    dashboard/app.py           Streamlit UI
  models/                      trained .joblib files (16 MB, committed)
  Dockerfile
  docker-compose.yml
  render.yaml

## Running locally

python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
# download Kaggle CSVs into data/raw/
python -m src.features.build_features
python -m src.models.train
python -m src.models.train_quantile
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501

## API

POST /forecast

Request:
  {"store_nbr": 1, "family_code": 0, "date": "2017-08-16", "onpromotion": 0}

Response:
  {
    "store_nbr": 1,
    "family_code": 0,
    "date": "2017-08-16",
    "point_forecast": 3.95,
    "interval_80": {"lower": 0.74, "median": 3.92, "upper": 7.03}
  }

## Stack

Python, Pandas, NumPy, XGBoost, Statsmodels, Scikit-learn, FastAPI, Streamlit, Plotly, Docker, Render, Streamlit Cloud

## Limitations

- Feature concentration: model leans on lag_7 and roll_mean_7
- Segment bias remains (BEVERAGES under-forecast by ~150 units)
- No hierarchical reconciliation: store x family forecasts do not sum exactly to family or region totals
- Rolling validation uses expanding windows, not monthly walk-forward retraining

## License

MIT
