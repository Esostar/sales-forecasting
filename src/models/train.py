"""Train XGBoost on store x family features, evaluate on holdout."""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main():
    tr = pd.read_pickle(PROC / "train.pkl")
    te = pd.read_pickle(PROC / "test.pkl")

    feat_cols = [c for c in tr.columns if c not in ("date", "target", "sales")]
    print(f"using {len(feat_cols)} features")

    X_tr, y_tr = tr[feat_cols], tr["target"]
    X_te, y_te = te[feat_cols], te["target"]

    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=100,
    )

    # predictions back on original scale
    pred_log = model.predict(X_te)
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 0, None)
    actual = te["sales"].values

    print("\n--- Test metrics (store x family, original scale) ---")
    print(f"MAE : {mae(actual, pred):.3f}")
    print(f"RMSE: {rmse(actual, pred):.3f}")
    print(f"WAPE: {wape(actual, pred):.2f}%")

    # naive comparison at same granularity: lag_7
    naive_pred = te["lag_7"].values
    print("\n--- Naive (lag_7) same split ---")
    print(f"MAE : {mae(actual, naive_pred):.3f}")
    print(f"RMSE: {rmse(actual, naive_pred):.3f}")
    print(f"WAPE: {wape(actual, naive_pred):.2f}%")

    joblib.dump(model, MODELS / "xgb_v1.joblib")
    print(f"\nsaved -> {MODELS / 'xgb_v1.joblib'}")


if __name__ == "__main__":
    main()
