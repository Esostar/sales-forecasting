"""Fast quantile models for prediction intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.5, 0.9]


def wape(y, p):
    return np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100


def coverage(y, lo, hi):
    return np.mean((y >= lo) & (y <= hi)) * 100


def main():
    tr = pd.read_pickle(PROC / "train.pkl")
    te = pd.read_pickle(PROC / "test.pkl")

    # subsample training set for quantile models (they're slow)
    tr = tr.sample(n=min(500_000, len(tr)), random_state=42).reset_index(drop=True)

    feat_cols = [c for c in tr.columns if c not in ("date", "target", "sales")]
    X_tr, y_tr = tr[feat_cols], tr["target"]
    X_te = te[feat_cols]
    actual = te["sales"].values

    preds = {}
    for q in QUANTILES:
        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        pred = np.clip(np.expm1(model.predict(X_te)), 0, None)
        preds[q] = pred
        joblib.dump(model, MODELS / f"xgb_q{int(q*100)}.joblib")
        print(f"q={q}: WAPE(if point)={wape(actual, pred):.2f}%  saved.")

    lo, mid, hi = preds[0.1], preds[0.5], preds[0.9]
    print("\n--- Interval diagnostics (target 80%) ---")
    print(f"coverage: {coverage(actual, lo, hi):.2f}%")
    print(f"mean interval width: {np.mean(hi - lo):.2f}")
    print(f"median forecast WAPE: {wape(actual, mid):.2f}%")
    print(f"below lo: {np.mean(actual < lo)*100:.2f}% | above hi: {np.mean(actual > hi)*100:.2f}%")


if __name__ == "__main__":
    main()
