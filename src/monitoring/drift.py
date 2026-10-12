"""Data drift monitoring: PSI on features, rolling WAPE on residuals."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 watch, >0.25 drift."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # quantile-based bins from expected
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0

    e_counts, _ = np.histogram(expected, bins=qs)
    a_counts, _ = np.histogram(actual, bins=qs)

    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main():
    tr = pd.read_pickle(PROC / "train.pkl")
    te = pd.read_pickle(PROC / "test.pkl")
    # full feature set for model prediction
    feat = [c for c in tr.columns if c not in ("date", "target", "sales")]

    # exclude deterministic calendar + id features from PSI (false positives on time splits)
    PSI_EXCLUDE = {"date", "target", "sales", "year", "month", "day", "dow", "weekofyear",
                   "sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3",
                   "store_nbr", "family_code", "cluster", "store_type_code",
                   "is_weekend", "is_month_start", "is_month_end", "is_payday", "is_holiday"}
    psi_feat = [c for c in feat if c not in PSI_EXCLUDE]

    print("=== Feature drift (train vs test) ===")
    rows = []
    for c in psi_feat:
        p = psi(tr[c].values, te[c].values)
        status = "OK" if p < 0.1 else ("WATCH" if p < 0.25 else "DRIFT")
        rows.append((c, round(p, 4), status))

    drift_df = pd.DataFrame(rows, columns=["feature", "psi", "status"]).sort_values("psi", ascending=False)
    print(drift_df.to_string(index=False))
    drift_df.to_csv(REPORTS / "feature_drift.csv", index=False)

    drifted = drift_df[drift_df.status == "DRIFT"]
    print(f"\n{len(drifted)} features flagged DRIFT, {len(drift_df[drift_df.status=='WATCH'])} WATCH")

    # ---- Rolling WAPE over time (concept drift proxy) ----
    print("\n=== Rolling WAPE over test window (weekly) ===")
    model = joblib.load(MODELS / "xgb_v1.joblib")
    pred = np.clip(np.expm1(model.predict(te[feat])), 0, None)
    te2 = te.copy()
    te2["pred"] = pred
    te2["abs_err"] = np.abs(te2.sales - te2.pred)
    te2["week"] = te2.date.dt.isocalendar().week

    weekly = te2.groupby("week").apply(
        lambda g: g.abs_err.sum() / max(g.sales.sum(), 1) * 100,
        include_groups=False,
    ).reset_index(name="wape")
    print(weekly.to_string(index=False))

    # shift check: how different is test vs train on key features
    print("\n=== Quick sanity: mean shift on top features ===")
    top = ["lag_7", "roll_mean_7", "onpromotion", "oil"]
    for c in top:
        print(f"  {c:15s} train={tr[c].mean():10.3f}  test={te[c].mean():10.3f}  "
              f"Δ%={((te[c].mean()-tr[c].mean())/max(abs(tr[c].mean()),1e-9)*100):6.2f}")


if __name__ == "__main__":
    main()
