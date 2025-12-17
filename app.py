import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from src.train_model import forecast
from src.evaluate import evaluate_forecast

st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"

st.title("Retail Demand Forecasting Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
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
        label="Forecast"
    )
    ax.axvline(df["Date"].iloc[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
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
        X.append(scaled[i - lookback:i, 0])
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

def main():
    df = load_data()

    if df is None:
        st.error("sales_data.csv not found")
        st.stop()

    model, scaler, lookback = train_model_on_start(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric(
        "Date Range",
        f"{df['Date'].min().date()} → {df['Date'].max().date()}"
    )
    c3.metric("Average Daily Sales", f"{df['Sales'].mean():.2f}")
    c4.metric("Total Sales", f"{df['Sales'].sum():,.0f}")

    st.markdown("---")

    horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )

    if st.button("Generate Forecast"):
        state = {
            "model": model,
            "scaler": scaler,
            "lookback": lookback
        }

        forecast_df = forecast(state, df, periods=horizon)

        st.pyplot(plot_forecast(df, forecast_df))
        st.dataframe(forecast_df, use_container_width=True)

        metrics = evaluate_forecast(df, forecast_df, test_size=min(30, len(df)))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{metrics['RMSE']:.2f}")
        m2.metric("MAE", f"{metrics['MAE']:.2f}")
        m3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        m4.metric("Forecast Accuracy", f"{metrics['Forecast Accuracy (%)']:.2f}%")

    st.markdown("---")
    st.subheader("Historical Sales Trend")
    st.line_chart(df.set_index("Date")["Sales"])

if __name__ == "__main__":
    main()
