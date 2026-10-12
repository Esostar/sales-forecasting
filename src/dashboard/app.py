"""Streamlit dashboard: analytics + forecast console."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC = ROOT / "data" / "processed"
import os
API_URL = os.getenv("API_URL", "http://localhost:8000/forecast")

st.set_page_config(page_title="Sales Forecasting Platform", layout="wide")


@st.cache_data
def load_all():
    import json
    train = pd.read_parquet(PROC_DIR / "dashboard_train.parquet")
    stores = pd.read_csv(PROC_DIR / "stores.csv")
    oil = pd.read_csv(PROC_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(PROC_DIR / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(PROC_DIR / "transactions.csv", parse_dates=["date"])
    with open(PROC_DIR / "families.json") as f:
        fam_names = json.load(f)
    train["family"] = pd.Categorical(train["family"], categories=fam_names)
    train = train.merge(stores, on="store_nbr", how="left")
    train["dow"] = train.date.dt.dayofweek
    train["month"] = train.date.dt.month
    train["year"] = train.date.dt.year
    train["promo_flag"] = (train.onpromotion > 0).astype(int)
    return train, stores, oil, holidays, transactions


train, stores, oil, holidays, transactions = load_all()

page = st.sidebar.radio("Page", ["📊 Analytics", "🔮 Forecast Console"])


# ==================== ANALYTICS ====================
if page == "📊 Analytics":
    st.title("📊 Sales Analytics")

    # ---- Global filters ----
    st.sidebar.subheader("Filters")
    dmin, dmax = train.date.min().date(), train.date.max().date()
    dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(dr, tuple) and len(dr) == 2:
        start, end = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
    else:
        start, end = pd.Timestamp(dmin), pd.Timestamp(dmax)

    all_families = sorted(train.family.unique())
    fam_sel = st.sidebar.multiselect("Families", all_families, default=all_families)

    df = train[(train.date >= start) & (train.date <= end) & (train.family.isin(fam_sel))]

    if df.empty:
        st.warning("No data in selected range.")
        st.stop()

    # ---- KPIs ----
    total_sales = df.sales.sum()
    avg_daily = df.groupby("date").sales.sum().mean()
    promo_share = (df.onpromotion > 0).mean() * 100

    # YoY growth
    cur_year = df.date.dt.year.max()
    prev = df[df.date.dt.year == cur_year - 1].sales.sum()
    cur = df[df.date.dt.year == cur_year].sales.sum()
    yoy = ((cur - prev) / prev * 100) if prev > 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sales (units)", f"{total_sales:,.0f}")
    c2.metric("Avg daily sales", f"{avg_daily:,.0f}")
    c3.metric(f"YoY growth ({cur_year})", f"{yoy:.1f}%" if not np.isnan(yoy) else "n/a")
    c4.metric("Promo share", f"{promo_share:.1f}%")

    st.divider()

    # ---- Total sales over time ----
    st.subheader("Total sales over time")
    daily = df.groupby("date").sales.sum().reset_index()
    daily["roll_28"] = daily.sales.rolling(28).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.date, y=daily.sales, name="Daily", opacity=0.35))
    fig.add_trace(go.Scatter(x=daily.date, y=daily.roll_28, name="28-day avg", line=dict(width=3)))
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ---- YoY growth by family ----
    st.subheader("YoY growth by family")
    fam_year = df.groupby(["family", "year"]).sales.sum().unstack().fillna(0)
    if (cur_year - 1) in fam_year.columns and cur_year in fam_year.columns:
        fam_year["yoy"] = (fam_year[cur_year] - fam_year[cur_year - 1]) / fam_year[cur_year - 1].replace(0, np.nan) * 100
        top = fam_year.dropna(subset=["yoy"]).sort_values("yoy", ascending=False)
        top = pd.concat([top.head(10), top.tail(10)]).reset_index()
        colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in top.yoy]
        fig = go.Figure(go.Bar(x=top.yoy, y=top.family, orientation="h", marker_color=colors))
        fig.update_layout(height=500, title="Top 10 growing / bottom 10 declining families")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough years in selected range for YoY.")

    st.divider()

    # ---- Store cluster × family heatmap ----
    st.subheader("Store cluster × top family heatmap")
    top_fams = df.groupby("family").sales.sum().nlargest(20).index
    piv = df[df.family.isin(top_fams)].pivot_table(
        values="sales", index="family", columns="cluster", aggfunc="mean"
    )
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Cluster", y="Family", color="Avg sales"))
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")

    st.divider()

    col1, col2 = st.columns(2)

    # ---- Promotion lift ----
    with col1:
        st.subheader("Promotion lift by family")
        fam = df.groupby(["family", "promo_flag"]).sales.mean().unstack()
        fam.columns = ["no_promo", "promo"]
        fam["lift_pct"] = (fam.promo - fam.no_promo) / fam.no_promo.replace(0, np.nan) * 100
        top = fam.dropna().sort_values("lift_pct", ascending=False).head(15).reset_index()
        st.plotly_chart(px.bar(top, x="lift_pct", y="family", orientation="h"),
                        width="stretch")

    # ---- Zero-sales % by family ----
    with col2:
        st.subheader("Zero-sales % by family (top 15 by volume)")
        top15 = df.groupby("family").sales.sum().nlargest(15).index
        z = df[df.family.isin(top15)].groupby("family").sales.apply(
            lambda s: (s == 0).mean() * 100
        ).sort_values(ascending=False).reset_index(name="zero_pct")
        st.plotly_chart(px.bar(z, x="zero_pct", y="family", orientation="h"),
                        width="stretch")

    st.divider()

    col1, col2 = st.columns(2)

    # ---- Holiday impact ----
    with col1:
        st.subheader("Holiday impact")
        h_dates = holidays[holidays.transferred == False].date.unique()
        df["is_holiday"] = df.date.isin(h_dates)
        h = df.groupby("is_holiday").sales.mean().reset_index()
        h["label"] = h.is_holiday.map({True: "Holiday", False: "Normal"})
        st.plotly_chart(px.bar(h, x="label", y="sales", title="Avg sales: Holiday vs Normal"),
                        width="stretch")

    # ---- Transactions trend ----
    with col2:
        st.subheader("Transactions trend (all stores)")
        t = transactions[(transactions.date >= start) & (transactions.date <= end)]
        t_daily = t.groupby("date").transactions.sum().reset_index()
        t_daily["roll_28"] = t_daily.transactions.rolling(28).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_daily.date, y=t_daily.transactions, opacity=0.3, name="Daily"))
        fig.add_trace(go.Scatter(x=t_daily.date, y=t_daily.roll_28, name="28d avg", line=dict(width=3)))
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # ---- Oil vs sales ----
    st.subheader("Oil price vs total sales")
    o = oil[(oil.date >= start) & (oil.date <= end)].copy()
    o["oil"] = o.dcoilwtico.ffill()
    merged = daily.merge(o[["date", "oil"]], on="date", how="left")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged.date, y=merged.roll_28, name="Sales (28d avg)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=merged.date, y=merged.oil, name="Oil price", yaxis="y2",
                             line=dict(color="orange")))
    fig.update_layout(
        height=380,
        yaxis=dict(title="Sales (28d avg)"),
        yaxis2=dict(title="Oil price", overlaying="y", side="right"),
    )
    st.plotly_chart(fig, width="stretch")


# ==================== FORECAST CONSOLE ====================
else:
    st.title("🔮 Forecast Console")

    import json
    fam_map = json.load(open(PROC_DIR / "families.json"))
    families = list(enumerate(fam_map))

    col1, col2, col3, col4 = st.columns(4)
    store = col1.number_input("Store", 1, 54, 1)
    fam_name = col2.selectbox("Family", [f[1] for f in families])
    fam_code = [f[0] for f in families if f[1] == fam_name][0]
    date = col3.date_input("Date", pd.Timestamp("2017-08-16"))
    promo = col4.selectbox("On promotion?", [0, 1])

    if st.button("Generate forecast", type="primary"):
        payload = {
            "store_nbr": int(store),
            "family_code": int(fam_code),
            "date": str(date),
            "onpromotion": int(promo),
        }
        try:
            r = requests.post(API_URL, json=payload, timeout=60)
            r.raise_for_status()
            res = r.json()

            st.success("Forecast ready")
            m1, m2, m3 = st.columns(3)
            m1.metric("Point forecast", res["point_forecast"])
            m2.metric("Lower 80%", res["interval_80"]["lower"])
            m3.metric("Upper 80%", res["interval_80"]["upper"])

            hist = pd.read_pickle(PROC / "train.pkl")
            s = hist[(hist.store_nbr == store) & (hist.family_code == fam_code)].tail(60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.date, y=s.sales, name="History"))
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(date)], y=[res["point_forecast"]],
                error_y=dict(
                    type="data", symmetric=False,
                    array=[res["interval_80"]["upper"] - res["point_forecast"]],
                    arrayminus=[res["point_forecast"] - res["interval_80"]["lower"]],
                ),
                mode="markers", marker=dict(size=12, color="red"), name="Forecast",
            ))
            fig.update_layout(title="History + forecast", height=420)
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"API error: {e}")
