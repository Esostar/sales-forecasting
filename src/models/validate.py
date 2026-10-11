"""Expanding-window validation across multiple 90-day folds."""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# 3 folds, each with 90-day horizon, expanding training window
FOLDS = [
    ("2017-02-16", "2017-05-17"),
    ("2017-02-16", "2017-08-15"),  # placeholder replaced below
]

# define folds explicitly: (train_end, test_start, test_end)
FOLD_DEFS = [
    (pd.Timestamp("2016-11-16"), pd.Timestamp("2016-11-17"), pd.Timestamp("2017-02-15")),
    (pd.Timestamp("2017-02-15"), pd.Timestamp("2017-02-16"), pd.Timestamp("2017-05-17")),
    (pd.Timestamp("2017-05-17"), pd.Timestamp("2017-05-18"), pd.Timestamp("2017-08-15")),
]


def wape(y, p):
    return np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100


def main():
    df = pd.read_pickle(PROC / "train.pkl")
    te_final = pd.read_pickle(PROC / "test.pkl")
    full = pd.concat([df, te_final], ignore_index=True)

    feat_cols = [c for c in df.columns if c not in ("date", "target", "sales")]
    print(f"features: {len(feat_cols)}\n")

    results = []
    for i, (tr_end, te_start, te_end) in enumerate(FOLD_DEFS, 1):
        tr = full[(full.date >= pd.Timestamp("2013-12-31")) & (full.date <= tr_end)]
        te = full[(full.date >= te_start) & (full.date <= te_end)]

        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1, random_state=42,
        )
        model.fit(tr[feat_cols], tr["target"])

        pred = np.clip(np.expm1(model.predict(te[feat_cols])), 0, None)
        actual = te["sales"].values

        w = wape(actual, pred)
        print(f"fold {i}: train<= {tr_end.date()} | test {te_start.date()}->{te_end.date()} "
              f"| rows={len(te):>6} | WAPE={w:.2f}%")
        results.append(w)

    print(f"\nmean WAPE across folds: {np.mean(results):.2f}%")
    print(f"std  WAPE across folds: {np.std(results):.2f}%")


if __name__ == "__main__":
    main()
