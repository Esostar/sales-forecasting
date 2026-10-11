"""Build store x family feature matrix for forecasting."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2017-05-17")
LAGS = [7, 14, 21, 28, 364]
ROLLS = [7, 14, 28]


def load_raw():
    train = pd.read_csv(RAW / "train.csv", parse_dates=["date"])
    stores = pd.read_csv(RAW / "stores.csv")
    oil = pd.read_csv(RAW / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])
    return train, stores, oil, holidays


def full_grid(train):
    stores = train["store_nbr"].unique()
    families = train["family"].unique()
    dates = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product(
        [stores, families, dates], names=["store_nbr", "family", "date"]
    )
    grid = pd.DataFrame(index=idx).reset_index()
    df = grid.merge(train, on=["store_nbr", "family", "date"], how="left")
    df["sales"] = df["sales"].fillna(0)
    df["onpromotion"] = df["onpromotion"].fillna(0)
    return df


def add_calendar(df):
    d = df["date"].dt
    df["year"] = d.year
    df["month"] = d.month
    df["day"] = d.day
    df["dow"] = d.dayofweek
    df["weekofyear"] = d.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dayofweek >= 5).astype(np.int8)
    df["is_month_start"] = d.is_month_start.astype(np.int8)
    df["is_month_end"] = d.is_month_end.astype(np.int8)
    df["is_payday"] = ((df["day"] == 15) | d.is_month_end).astype(np.int8)
    return df


def add_external(df, stores, oil, holidays):
    df = df.merge(stores, on="store_nbr", how="left")

    o = oil.rename(columns={"dcoilwtico": "oil"}).set_index("date").asfreq("D")
    o["oil"] = o["oil"].ffill().bfill()
    df = df.merge(o.reset_index(), on="date", how="left")
    df["oil"] = df["oil"].ffill().bfill()

    h = holidays[holidays["transferred"] == False][["date"]].drop_duplicates()
    h["is_holiday"] = 1
    df = df.merge(h, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(np.int8)
    return df


def add_lags_rolls(df):
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    key = df["store_nbr"].astype(str) + "|" + df["family"]

    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby(key, sort=False)["sales"].shift(lag)

    df["_sh"] = df.groupby(key, sort=False)["sales"].shift(1)
    for w in ROLLS:
        df[f"roll_mean_{w}"] = (
            df.groupby(key, sort=False)["_sh"]
            .rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"roll_std_{w}"] = (
            df.groupby(key, sort=False)["_sh"]
            .rolling(w, min_periods=1).std()
            .reset_index(level=0, drop=True)
        )
    df.drop(columns=["_sh"], inplace=True)
    return df


def main():
    train, stores, oil, holidays = load_raw()
    df = full_grid(train)
    df = add_calendar(df)
    df = add_external(df, stores, oil, holidays)
    df = add_lags_rolls(df)

    df["family_code"] = df["family"].astype("category").cat.codes
    df["store_type_code"] = df["type"].astype("category").cat.codes

    df = df.dropna(subset=["lag_364"]).reset_index(drop=True)

    feature_cols = [
        "store_nbr", "family_code", "cluster", "store_type_code",
        "year", "month", "day", "dow", "weekofyear",
        "is_weekend", "is_month_start", "is_month_end", "is_payday", "is_holiday",
        "oil", "onpromotion",
        "lag_7", "lag_14", "lag_21", "lag_28", "lag_364",
        "roll_mean_7", "roll_mean_14", "roll_mean_28",
        "roll_std_7", "roll_std_14", "roll_std_28",
    ]
    df["target"] = np.log1p(df["sales"])

    tr = df[df["date"] <= CUTOFF].copy()
    te = df[df["date"] > CUTOFF].copy()

    tr[["date"] + feature_cols + ["target", "sales"]].to_pickle(OUT / "train.pkl")
    te[["date"] + feature_cols + ["target", "sales"]].to_pickle(OUT / "test.pkl")

    print(f"features: {len(feature_cols)}")
    print(f"train: {tr.shape} | test: {te.shape}")
    print(f"train dates: {tr.date.min().date()} -> {tr.date.max().date()}")
    print(f"test  dates: {te.date.min().date()} -> {te.date.max().date()}")


if __name__ == "__main__":
    main()
