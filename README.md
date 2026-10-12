# Sales Forecasting & Revenue Intelligence Platform

End-to-end ML system for store x family sales forecasting with prediction intervals.

## Stack
- Data: Favorita Store Sales (Kaggle)
- Models: XGBoost (point) + quantile XGBoost (80% interval)
- API: FastAPI
- Dashboard: Streamlit (Analytics + Forecast Console)
- Serving: Docker Compose

## Results
- Store x family WAPE: 11.5% (vs 18.3% seasonal naive)
- Rolling validation mean WAPE: 14.9% (std 2.9%)
- 80% prediction interval coverage: 82.2%

## Structure
- src/features/build_features.py - feature pipeline (lags, rolls, Fourier, external)
- src/models/train.py - point model
- src/models/train_quantile.py - interval models
- src/models/validate.py - expanding-window validation
- src/models/diagnostics.py - learning curve, residual bias, feature importance
- src/monitoring/drift.py - PSI + rolling WAPE
- src/api/main.py - FastAPI service
- src/dashboard/app.py - Streamlit UI

## Run
pip install -r requirements.txt
python -m src.features.build_features
python -m src.models.train
python -m src.models.train_quantile

docker compose up --build

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501
