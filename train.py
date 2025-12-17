import pandas as pd
from pathlib import Path
import joblib
from src.data_preprocessing import prepare_data
from src.train_model import train_lstm_model
from src.evaluate import evaluate_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    data_path = BASE_DIR / "data" / "sales_data.csv"

    print("Loading and preprocessing data...")
    df = prepare_data(str(data_path))

    print("Training LSTM model...")
    state = train_lstm_model(df, lookback=14, test_size=30, epochs=30)

    print("Evaluating model...")
    metrics = evaluate_model(state["y_test"], state["y_pred"])

    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"MAE: {metrics['MAE']:.2f}")

    print("Saving model and scaler...")
    state["model"].save(MODEL_DIR / "lstm_model.keras")
    joblib.dump(state["scaler"], MODEL_DIR / "scaler.pkl")
    meta = {"lookback": state.get("lookback", 14)}
    joblib.dump(meta, MODEL_DIR / "meta.pkl")

    print("Training completed!")

if __name__ == "__main__":
    main()
