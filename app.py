import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from src.train_model import forecast
from src.evaluate import evaluate_forecast

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.markdown("""
## Retail Demand Forecasting System

This application demonstrates **retail demand forecasting** using historical daily sales data
and an LSTM neural network.

The goal is to **predict future daily sales volume** so businesses can plan inventory efficiently.
""")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def plot_forecast(df, forecast_df):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Date"], df["Sales"], label="Historical Sales", color="tab:blue")
    ax.plot(
        forecast_df["Date"],
        forecast_df["Forecast"],
        linestyle="--",
        color="red",
        linewidth=2,
        label="Forecasted Demand"
    )
    ax.axvline(df["Date"].iloc[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_title("Historical Sales vs Forecasted Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold per Day")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

@st.cache_resource
def train_model_on_start(df):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    values = df["Sales"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    lookback = 14
    X, y = [], []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(lookback, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    return model, scaler, lookback

df = load_data()
model, scaler, lookback = train_model_on_start(df)

avg_daily_sales = df["Sales"].mean()

st.markdown("---")
st.subheader("Dataset Relevance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", len(df))
c2.metric("Date Range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
c3.metric("Average Daily Sales", f"{avg_daily_sales:.0f} units/day")
c4.metric("Total Units Sold", f"{df['Sales'].sum():,.0f}")

st.markdown("""
**Interpretation:**  
All forecasting errors below should be interpreted **relative to an average of
approximately `{}` units sold per day**.
""".format(int(avg_daily_sales)))

st.markdown("---")
st.subheader(" Demand Forecasting")

horizon = st.slider("Forecast Horizon (Days)", 7, 90, 30, step=7)
st.markdown("""
### How to Use This Forecast

- Select how many future days you want to forecast using the slider below
- Click **Generate Forecast**
- The **red dashed line** represents predicted future demand
- The **blue line** shows historical daily sales
""")

if st.button("Generate Forecast"):
    state = {
        "model": model,
        "scaler": scaler,
        "lookback": lookback
    }

    forecast_df = forecast(state, df, periods=horizon)

    st.pyplot(plot_forecast(df, forecast_df))
    st.dataframe(forecast_df, use_container_width=True)

    metrics = evaluate_forecast(df, forecast_df, test_size=len(forecast_df))

    rmse = metrics["RMSE"]
    mae = metrics["MAE"]
    mape = metrics["MAPE"]
    accuracy = metrics["Forecast Accuracy (%)"]

    rmse_pct = (rmse / avg_daily_sales) * 100
    mae_pct = (mae / avg_daily_sales) * 100

    st.markdown("---")
    st.subheader("📉 Model Performance Metrics (With Scale Context)")

    st.markdown("""
### Metric Explanation with Business Context

- **RMSE:** Typical prediction error magnitude  
  → **{:.0f} units**, which is **{:.1f}% of average daily sales**

- **MAE:** Average absolute error per day  
  → **{:.0f} units**, which is **{:.1f}% of daily demand**

- **MAPE:** Percentage-based error  
  → Model is off by **~{:.1f}% on average**

**Interpretation:**  
For a business selling ~{:.0f} units per day, this level of error is considered
**reasonable for medium-term retail demand forecasting**.
""".format(
        rmse, rmse_pct,
        mae, mae_pct,
        mape,
        avg_daily_sales
    ))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE (Units)", f"{rmse:.2f}")
    m2.metric("MAE (Units)", f"{mae:.2f}")
    m3.metric("MAPE (%)", f"{mape:.2f}%")
    m4.metric("Forecast Accuracy", f"{accuracy:.2f}%")

st.markdown("---")
st.subheader("Historical Sales Trend")
st.line_chart(df.set_index("Date")["Sales"])
