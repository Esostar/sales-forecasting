"""FastAPI service for store x family sales forecasting with prediction intervals."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

FEATURE_COLS = [
    "store_nbr", "family_code", "cluster", "store_type_code",
    "year", "month", "day", "dow", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end", "is_payday", "is_holiday",
    "oil", "onpromotion",
    "lag_7", "lag_14", "lag_21", "lag_28", "lag_364",
    "roll_mean_7", "roll_mean_14", "roll_mean_28",
    "roll_std_7", "roll_std_14", "roll_std_28",
]

app = FastAPI(title="Sales Forecasting API", version="0.1.0")


@lru_cache(maxsize=1)
def load_models():
    return {
        "point": joblib.load(MODELS / "xgb_v1.joblib"),
        "q10": joblib.load(MODELS / "xgb_q10.joblib"),
        "q50": joblib.load(MODELS / "xgb_q50.joblib"),
        "q90": joblib.load(MODELS / "xgb_q90.joblib"),
    }


@lru_cache(maxsize=1)
def load_history():
    tr = pd.read_pickle(PROC / "train.pkl")
    return tr


class ForecastRequest(BaseModel):
    store_nbr: int = Field(..., ge=1, le=54)
    family_code: int = Field(..., ge=0)
    date: str  # ISO format YYYY-MM-DD
    onpromotion: int = 0


def build_features(req: ForecastRequest) -> pd.DataFrame:
    hist = load_history()
    d = pd.Timestamp(req.date)

    # pull last 365+ days of this series to compute lags/rolls
    series = hist[(hist.store_nbr == req.store_nbr) &
                  (hist.family_code == req.family_code)].sort_values("date")

    if series.empty:
        raise HTTPException(404, f"No history for store={req.store_nbr}, family_code={req.family_code}")

    last = series.iloc[-1]
    sales = series["sales"].values

    def lag(n):
        return float(sales[-n]) if len(sales) >= n else np.nan

    def roll_mean(w):
        return float(np.mean(sales[-w:])) if len(sales) >= 1 else np.nan

    def roll_std(w):
        return float(np.std(sales[-w:])) if len(sales) >= 2 else 0.0

    row = {
        "store_nbr": req.store_nbr,
        "family_code": req.family_code,
        "cluster": int(last["cluster"]),
        "store_type_code": int(last["store_type_code"]),
        "year": d.year, "month": d.month, "day": d.day,
        "dow": d.dayofweek, "weekofyear": int(d.isocalendar().week),
        "is_weekend": int(d.dayofweek >= 5),
        "is_month_start": int(d.is_month_start),
        "is_month_end": int(d.is_month_end),
        "is_payday": int(d.day == 15 or d.is_month_end),
        "is_holiday": 0,
        "oil": float(last.get("oil", 50.0)),
        "onpromotion": req.onpromotion,
        "lag_7": lag(7), "lag_14": lag(14), "lag_21": lag(21),
        "lag_28": lag(28), "lag_364": lag(364),
        "roll_mean_7": roll_mean(7), "roll_mean_14": roll_mean(14), "roll_mean_28": roll_mean(28),
        "roll_std_7": roll_std(7), "roll_std_14": roll_std(14), "roll_std_28": roll_std(28),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    models = load_models()
    X = build_features(req)

    point = float(np.clip(np.expm1(models["point"].predict(X))[0], 0, None))
    q10 = float(np.clip(np.expm1(models["q10"].predict(X))[0], 0, None))
    q50 = float(np.clip(np.expm1(models["q50"].predict(X))[0], 0, None))
    q90 = float(np.clip(np.expm1(models["q90"].predict(X))[0], 0, None))

    return {
        "store_nbr": req.store_nbr,
        "family_code": req.family_code,
        "date": req.date,
        "point_forecast": round(point, 2),
        "interval_80": {"lower": round(q10, 2), "median": round(q50, 2), "upper": round(q90, 2)},
    }
