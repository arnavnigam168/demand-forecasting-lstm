import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_accuracy_from_mape(mape):
    if np.isnan(mape):
        return np.nan
    return max(0.0, 100.0 - mape)


def evaluate_model(y_true, y_pred):
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    accuracy = calculate_accuracy_from_mape(mape)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Forecast Accuracy (%)": accuracy
    }


def evaluate_forecast(df_historical, forecast_df, test_size=30):

    df_test = df_historical.tail(test_size).copy()
    forecast_test = forecast_df.head(test_size).copy()

    y_true = df_test["Sales"].values
    y_pred = forecast_test["Forecast"].values

    return evaluate_model(y_true, y_pred)
