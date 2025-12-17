import pandas as pd
import numpy as np

def create_time_features(df):
    df = df.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week'] = df['Date'].dt.isocalendar().week
    df['quarter'] = df['Date'].dt.quarter
    return df

def create_lag_features(df, lags=[1, 7, 30]):
    df = df.copy()
    for lag in lags:
        df[f'sales_lag_{lag}'] = df['Sales'].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 30]):
    df = df.copy()
    for window in windows:
        df[f'sales_rolling_mean_{window}'] = df['Sales'].rolling(window=window).mean()
        df[f'sales_rolling_std_{window}'] = df['Sales'].rolling(window=window).std()
    return df

def create_seasonal_features(df):
    df = df.copy()
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)
    
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def engineer_features(df):
    df = create_time_features(df)
    df = create_seasonal_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = df.dropna().reset_index(drop=True)
    return df

