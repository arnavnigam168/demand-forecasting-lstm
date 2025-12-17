import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(start_date='2020-01-01', end_date='2024-12-15', output_path='data/sales_data.csv'):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    
    base_sales = 1000
    
    trend = np.linspace(0, 500, len(dates))
    
    yearly_seasonality = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    
    monthly_seasonality = 100 * np.sin(2 * np.pi * pd.to_datetime(dates).month / 12)
    
    weekly_seasonality = 50 * np.sin(2 * np.pi * pd.to_datetime(dates).dayofweek / 7)
    
    noise = np.random.normal(0, 50, len(dates))
    
    sales = base_sales + trend + yearly_seasonality + monthly_seasonality + weekly_seasonality + noise
    
    sales = np.maximum(sales, 100)
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sample data generated: {len(df)} records")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Saved to: {output_path}")
    
    return df

if __name__ == '__main__':
    generate_sample_data()

