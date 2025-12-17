import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df = df.copy()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'date': 'Date'})
    
    if 'Sales' in df.columns:
        pass
    elif 'sales' in df.columns:
        df = df.rename(columns={'sales': 'Sales'})
    elif 'Demand' in df.columns:
        df = df.rename(columns={'Demand': 'Sales'})
    elif 'demand' in df.columns:
        df = df.rename(columns={'demand': 'Sales'})
    
    df = df[['Date', 'Sales']].copy()
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    df = df.dropna(subset=['Sales'])
    
    df = df[df['Sales'] >= 0]
    
    return df

def aggregate_daily(df):
    df_agg = df.groupby('Date')['Sales'].sum().reset_index()
    df_agg = df_agg.sort_values('Date').reset_index(drop=True)
    return df_agg

def fill_missing_dates(df):
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
    df_complete = pd.DataFrame({'Date': date_range})
    df_complete = df_complete.merge(df, on='Date', how='left')
    df_complete['Sales'] = df_complete['Sales'].ffill().bfill().fillna(0)
    return df_complete

def prepare_data(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df = aggregate_daily(df)
    df = fill_missing_dates(df)
    return df

