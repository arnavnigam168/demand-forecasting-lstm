import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
import joblib

from src.train_model import forecast
from src.evaluate import evaluate_forecast

st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"
MODEL_PATH = BASE_DIR / "models" / "lstm_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
META_PATH = BASE_DIR / "models" / "meta.pkl"

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

def load_model_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None, None
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH) if META_PATH.exists() else {"lookback": 14}
    return model, scaler, meta

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

def main():
    df = load_data()
    model, scaler, meta = load_model_artifacts()

    if df is None:
        st.error("sales_data.csv not found")
        st.stop()

    if model is None:
        st.error("trained model not found")
        st.stop()

    lookback = int(meta.get("lookback", 14))

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
