import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------
# Sequence creation
# -------------------------
def create_univariate_sequences(series, lookback):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y


# -------------------------
# Train LSTM
# -------------------------
def train_lstm_model(df, lookback=14, test_size=30, epochs=30, batch_size=32):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    sales = df['Sales'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales).reshape(-1)

    X, y = create_univariate_sequences(sales_scaled, lookback)

    split = len(X) - test_size
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer=Adam(), loss='mse')

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        validation_split=0.1 if len(X_train) > 20 else 0.0,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es] if len(X_train) > 20 else None,
        verbose=1
    )

    y_pred_scaled = model.predict(X_test).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    dates_test = df['Date'].iloc[lookback + split: lookback + split + len(y_test_inv)]

    return {
        "model": model,
        "scaler": scaler,
        "lookback": lookback,
        "y_test": y_test_inv,
        "y_pred": y_pred,
        "dates_test": dates_test
    }


# -------------------------
# Forecast future
# -------------------------
def forecast(state, df, periods=30):
    model = state["model"]
    scaler = state["scaler"]
    lookback = state["lookback"]

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    last_date = df['Date'].iloc[-1]

    sales = df['Sales'].values.reshape(-1, 1)
    sales_scaled = scaler.transform(sales).reshape(-1)

    window = sales_scaled[-lookback:].tolist()

    preds = []
    dates = []

    for _ in range(periods):
        x = np.array(window[-lookback:]).reshape(1, lookback, 1)
        yhat_scaled = model.predict(x, verbose=0)[0][0]
        yhat = scaler.inverse_transform([[yhat_scaled]])[0][0]

        preds.append(float(yhat))
        last_date += pd.Timedelta(days=1)
        dates.append(last_date)

        window.append(yhat_scaled)

    return pd.DataFrame({
        "Date": dates,
        "Forecast": preds
    })
