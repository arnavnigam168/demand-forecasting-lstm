# Retail Demand Forecasting ML Project

A complete, production-ready demand forecasting system for retail products using time series analysis and machine learning.

## Problem Statement

This project addresses the challenge of predicting future retail product demand using historical sales data. Accurate demand forecasting helps businesses optimize inventory management, reduce costs, and improve customer satisfaction by ensuring products are available when needed.

## Dataset Description

The project expects a CSV file (`data/sales_data.csv`) with the following columns:
- **Date**: Date of the sale (format: YYYY-MM-DD)
- **Sales**: Sales amount or quantity sold

The dataset should contain daily sales data for at least 1-2 years to capture seasonal patterns effectively.

## Approach

1. **Data Preprocessing**: Load, clean, and aggregate sales data by date
2. **Feature Engineering**: Create time-based features (year, month, day, seasonality indicators)
3. **Model Training**: Train a Prophet forecasting model on historical data
4. **Evaluation**: Assess model performance using RMSE, MAPE, and MAE metrics
5. **Deployment**: Interactive Streamlit dashboard for visualization and forecasting

## Model Used

**Facebook Prophet**: A robust time series forecasting tool that handles:
- Trend detection
- Yearly, weekly, and daily seasonality
- Holiday effects
- Missing data and outliers

Prophet is particularly well-suited for business time series with strong seasonal patterns.

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors
- **MAPE (Mean Absolute Percentage Error)**: Expresses accuracy as a percentage
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values

## Project Structure

```
Demand Forecasting ML/
├── data/
│   └── sales_data.csv          # Input sales data
├── notebooks/                   # Jupyter notebooks for exploration
├── src/
│   ├── data_preprocessing.py    # Data loading and cleaning
│   ├── feature_engineering.py   # Feature creation
│   ├── train_model.py           # Model training utilities
│   └── evaluate.py              # Evaluation metrics
├── models/
│   └── prophet_model.pkl        # Trained model (generated)
├── app.py                       # Streamlit dashboard
├── train.py                     # Main training script
├── generate_sample_data.py      # Sample data generator
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: Prophet requires additional system dependencies. On Windows, you may need Visual C++ Build Tools. On Linux/Mac:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

## How to Run Locally

### Step 1: Generate Sample Data (if you don't have your own)

```bash
python generate_sample_data.py
```

This creates `data/sales_data.csv` with synthetic sales data from 2020-2024.

### Step 2: Train the Model

```bash
python train.py
```

This will:
- Load and preprocess the data
- Engineer features
- Train the Prophet model
- Evaluate performance
- Save the model to `models/prophet_model.pkl`

### Step 3: Launch the Streamlit App

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Using Your Own Data

1. Place your CSV file in the `data/` folder as `sales_data.csv`
2. Ensure it has `Date` and `Sales` columns (case-insensitive)
3. Run `train.py` to train on your data
4. Launch the Streamlit app

## Streamlit Dashboard Features

- **Overview Metrics**: Total records, date range, average sales
- **Interactive Forecast**: Adjustable forecast period (7-90 days)
- **Visualizations**: 
  - Historical vs forecasted sales
  - Monthly seasonality patterns
  - Yearly trends
- **Performance Metrics**: RMSE, MAPE, MAE
- **Forecast Details**: Table with predictions and confidence intervals

## Deployment on Streamlit Cloud

1. **Push your code to GitHub**

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

3. **Sign in and click "New app"**

4. **Select your repository and branch**

5. **Set the main file path**: `app.py`

6. **Click "Deploy"**

**Important**: Ensure your `requirements.txt` includes all dependencies. The trained model (`models/prophet_model.pkl`) should be committed to the repository, or you can add a GitHub Action to train it automatically.

### Alternative: Include Model Training in App

If you prefer not to commit the model, modify `app.py` to train automatically if the model doesn't exist:

```python
if model is None:
    with st.spinner("Training model... This may take a few minutes."):
        from src.train_model import train_prophet_model, save_model
        model = train_prophet_model(df_historical)
        save_model(model, 'models/prophet_model.pkl')
```

## Model Performance

Typical performance on retail sales data:
- **RMSE**: Varies based on data scale (aim for <10% of average sales)
- **MAPE**: Typically 5-15% for good models
- **MAE**: Similar scale to RMSE

## Future Enhancements

- Support for multiple products/categories
- External regressors (promotions, holidays, weather)
- Model comparison (Prophet vs ARIMA vs LSTM)
- Automated hyperparameter tuning
- Real-time data integration

## Troubleshooting

**Prophet installation issues**: 
- Ensure you have the required system dependencies
- Try: `pip install prophet --no-cache-dir`

**Model not found error**: 
- Run `python train.py` first to generate the model

**Data format errors**: 
- Ensure Date column is in YYYY-MM-DD format
- Check for missing values in Sales column

## License

This project is open source and available for educational and commercial use.

## Author

Built as a production-ready ML project demonstrating end-to-end time series forecasting capabilities.

