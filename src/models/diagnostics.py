"""Diagnostics: overfit/underfit check, residual bias, learning curve."""
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def wape(y, p):
    return np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100


def mae(y, p):
    return np.mean(np.abs(y - p))


def main():
    tr = pd.read_pickle(PROC / "train.pkl")
    te = pd.read_pickle(PROC / "test.pkl")
    feat = [c for c in tr.columns if c not in ("date", "target", "sales")]

    # ---------------- 1. Overfit / underfit: learning curve ----------------
    print("=== Learning curve (train vs val WAPE) ===")
    sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    train_wapes, val_wapes = [], []
    for s in sizes:
        sub = tr.sample(frac=s, random_state=42)
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            tree_method="hist", n_jobs=-1, random_state=42,
        )
        m.fit(sub[feat], sub["target"])
        tw = wape(sub["sales"].values,
                  np.clip(np.expm1(m.predict(sub[feat])), 0, None))
        vw = wape(te["sales"].values,
                  np.clip(np.expm1(m.predict(te[feat])), 0, None))
        train_wapes.append(tw)
        val_wapes.append(vw)
        print(f"  frac={s:.2f}  train_WAPE={tw:.2f}%  val_WAPE={vw:.2f}%")

    plt.figure(figsize=(8, 4))
    plt.plot(sizes, train_wapes, marker="o", label="Train WAPE")
    plt.plot(sizes, val_wapes, marker="o", label="Val WAPE")
    plt.xlabel("Training fraction")
    plt.ylabel("WAPE %")
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "learning_curve.png", dpi=120)
    plt.close()

    # ---------------- 2. Residual bias by segment ----------------
    print("\n=== Residual bias by segment (positive = under-forecast) ===")
    import joblib
    model = joblib.load(ROOT / "models" / "xgb_v1.joblib")
    pred = np.clip(np.expm1(model.predict(te[feat])), 0, None)
    te = te.copy()
    te["pred"] = pred
    te["resid"] = te["sales"] - te["pred"]

    for col in ["store_nbr", "family_code", "dow", "month"]:
        g = te.groupby(col).resid.mean().sort_values()
        print(f"\n  {col}: worst under-forecast (top 3)")
        print(g.tail(3).to_string())
        print(f"  {col}: worst over-forecast (bottom 3)")
        print(g.head(3).to_string())

    # visual
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    te.groupby("dow").resid.mean().plot(kind="bar", ax=axes[0], title="Mean residual by dow")
    te.groupby("month").resid.mean().plot(kind="bar", ax=axes[1], title="Mean residual by month")
    plt.tight_layout()
    plt.savefig(FIG / "residual_segments.png", dpi=120)
    plt.close()

    # ---------------- 3. Feature importance ----------------
    print("\n=== Top 15 feature importances ===")
    imp = pd.Series(model.feature_importances_, index=feat).sort_values(ascending=False)
    print(imp.head(15).to_string())

    imp.head(20).plot(kind="barh", figsize=(8, 6), title="Top 20 features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG / "feature_importance.png", dpi=120)
    plt.close()

    print(f"\nFigures saved -> {FIG}")


if __name__ == "__main__":
    main()
