import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import requests

# Fetch Bitcoin data from Binance API
def fetch_btc_data():
    # Calculate dates for data since 2018
    start_date = '2018-01-01'
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Format dates for Binance API
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'startTime': start_timestamp,
        'endTime': end_timestamp,
        'limit': 10000  # Max allowed
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert timestamp to datetime and select close prices
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df['close'] = df['close'].astype(float)
    
    return df[['close']]

# Calculate 21-day Simple Moving Average
def calculate_sma21(data):
    return data['close'].rolling(window=21).mean()

# Grid search for optimal ARIMA parameters
def grid_search_arima(train_data, max_p=3, max_d=2, max_q=3):
    best_aic = float('inf')
    best_params = None
    
    p_values = range(0, max_p + 1)
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    
    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = ARIMA(train_data, order=(p, d, q))
            fitted_model = model.fit()
            
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_params = (p, d, q)
        except:
            continue
    
    return best_params, best_aic

# Main function to prepare data and train model
def run_arima_analysis():
    print("Fetching Bitcoin data...")
    btc_data = fetch_btc_data()
    
    print("Calculating 21-day SMA...")
    sma21 = calculate_sma21(btc_data)
    sma21_clean = sma21.dropna()
    
    # Split data into train and test sets (use last 30 days as test)
    split_idx = len(sma21_clean) - 30
    train_data = sma21_clean[:split_idx]
    test_data = sma21_clean[split_idx:]
    
    print("Performing grid search for optimal parameters...")
    best_params, best_aic = grid_search_arima(train_data)
    print(f"Best parameters: {best_params}, AIC: {best_aic}")
    
    # Fit model with best parameters
    model = ARIMA(train_data, order=best_params)
    fitted_model = model.fit()
    
    # Forecast next 7 days
    forecast_steps = 7
    forecast = fitted_model.forecast(steps=forecast_steps)
    forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
    
    # Prepare plotting data
    plt.figure(figsize=(15, 8))
    plt.plot(sma21_clean.index, sma21_clean.values, label='Actual 21-day SMA', color='blue')
    plt.plot(test_data.index, test_data.values, label='Test Data', color='green')
    
    # Plot forecast
    forecast_dates = pd.date_range(start=sma21_clean.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
    
    # Confidence intervals
    plt.fill_between(forecast_dates, 
                     forecast_ci.iloc[:, 0], 
                     forecast_ci.iloc[:, 1], 
                     color='pink', alpha=0.3, label='Confidence Interval')
    
    plt.title('Bitcoin 21-Day SMA Forecast using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plot_url = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plot_url, best_params, best_aic, forecast, forecast_dates

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    plot_url, best_params, best_aic, forecast, forecast_dates = run_arima_analysis()
    
    forecast_text = "<ul>"
    for i, date in enumerate(forecast_dates):
        forecast_text += f"<li>{date.strftime('%Y-%m-%d')}: ${forecast[i]:.2f}</li>"
    forecast_text += "</ul>"
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin 21-Day SMA ARIMA Forecast</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .info-box {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .plot-container {{ text-align: center; }}
            h1 {{ color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bitcoin 21-Day SMA ARIMA Forecast</h1>
            
            <div class="info-box">
                <h2>Model Information</h2>
                <p><strong>Optimal Parameters:</strong> ARIMA{best_params}</p>
                <p><strong>Akaike Information Criterion (AIC):</strong> {best_aic:.2f}</p>
                <p><strong>Forecast Period:</strong> Next 7 days</p>
            </div>
            
            <div class="info-box">
                <h2>Forecasted Values</h2>
                {forecast_text}
            </div>
            
            <div class="plot-container">
                <h2>Historical vs Forecasted 21-Day SMA</h2>
                <img src="data:image/png;base64,{plot_url}" width="100%" alt="ARIMA Forecast Plot">
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template)

if __name__ == '__main__':
    print("Starting server... Please wait while data is processed.")
    app.run(debug=True, host='0.0.0.0', port=5000)