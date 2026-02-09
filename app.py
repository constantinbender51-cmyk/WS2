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
import threading
import time

# Global variables to store model state
model_state = {
    'model_trained': False,
    'forecast_plot': None,
    'best_params': None,
    'aic_value': None,
    'forecast_values': None,
    'forecast_dates': None,
    'last_update': None
}

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

# Grid search for optimal ARIMA parameters with improved handling
def grid_search_arima(train_data, max_p=3, max_d=2, max_q=3):
    best_aic = float('inf')
    best_params = None
    
    p_values = range(0, min(max_p + 1, len(train_data)//10))  # Limit p based on data size
    d_values = range(0, max_d + 1)
    q_values = range(0, max_q + 1)
    
    for p, d, q in product(p_values, d_values, q_values):
        try:
            # Set enforce_stationarity and invertibility to False to handle convergence issues
            model = ARIMA(train_data, order=(p, d, q))
            fitted_model = model.fit(method_kwargs={"warn_convergence": False})
            
            if fitted_model.aic < best_aic:
                best_aic = fitted_model.aic
                best_params = (p, d, q)
        except Exception as e:
            continue
    
    return best_params, best_aic

# Train the model in a separate thread
def train_model():
    global model_state
    
    print("Starting model training...")
    print("Fetching Bitcoin data...")
    btc_data = fetch_btc_data()
    
    print("Calculating 21-day SMA...")
    sma21 = calculate_sma21(btc_data)
    sma21_clean = sma21.dropna()
    
    # Use only recent data to reduce computational complexity
    recent_data = sma21_clean.tail(500)  # Use last 500 days
    
    # Split data into train and test sets (use last 30 days as test)
    split_idx = len(recent_data) - 30
    train_data = recent_data[:split_idx]
    test_data = recent_data[split_idx:]
    
    print("Performing grid search for optimal parameters...")
    best_params, best_aic = grid_search_arima(train_data)
    print(f"Best parameters: {best_params}, AIC: {best_aic}")
    
    # If no good parameters found, use default
    if best_params is None:
        best_params = (1, 1, 1)
        print(f"No optimal parameters found. Using default: {best_params}")
    
    # Fit model with best parameters
    try:
        model = ARIMA(train_data, order=best_params)
        fitted_model = model.fit(method_kwargs={"warn_convergence": False})
    except:
        # If fitting fails with best params, use default
        model = ARIMA(train_data, order=(1, 1, 1))
        fitted_model = model.fit(method_kwargs={"warn_convergence": False})
    
    # Forecast next 7 days
    forecast_steps = 7
    forecast = fitted_model.forecast(steps=forecast_steps)
    forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
    
    # Prepare plotting data
    plt.figure(figsize=(15, 8))
    plt.plot(recent_data.index, recent_data.values, label='Actual 21-day SMA', color='blue')
    plt.plot(test_data.index, test_data.values, label='Test Data', color='green')
    
    # Plot forecast
    forecast_dates = pd.date_range(start=recent_data.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
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
    
    # Convert forecast to list to avoid indexing issues
    forecast_list = forecast.tolist()
    
    # Update model state
    model_state['model_trained'] = True
    model_state['forecast_plot'] = plot_url
    model_state['best_params'] = best_params
    model_state['aic_value'] = best_aic if best_params else float('inf')
    model_state['forecast_values'] = forecast_list
    model_state['forecast_dates'] = forecast_dates
    model_state['last_update'] = datetime.now()
    
    print("Model training completed!")

# Start model training in background
def start_training():
    training_thread = threading.Thread(target=train_model)
    training_thread.daemon = True
    training_thread.start()
    return training_thread

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    global model_state
    
    # Check if model is trained
    if not model_state['model_trained']:
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bitcoin 21-Day SMA ARIMA Forecast</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; text-align: center; }
                .container { max-width: 800px; margin: 0 auto; }
                .loading { background-color: #f5f5f5; padding: 20px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Bitcoin 21-Day SMA ARIMA Forecast</h1>
                <div class="loading">
                    <h2>Training model...</h2>
                    <p>Please wait while the ARIMA model is being trained. This may take a few minutes.</p>
                    <p>The page will automatically refresh once training is complete.</p>
                    <script>
                        setTimeout(function() {
                            window.location.reload();
                        }, 10000); // Refresh every 10 seconds
                    </script>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template
    
    # If model is trained, show results
    plot_url = model_state['forecast_plot']
    best_params = model_state['best_params']
    aic_value = model_state['aic_value']
    forecast = model_state['forecast_values']
    forecast_dates = model_state['forecast_dates']
    
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
                <p><strong>Akaike Information Criterion (AIC):</strong> {aic_value:.2f}</p>
                <p><strong>Forecast Period:</strong> Next 7 days</p>
                <p><strong>Last Updated:</strong> {model_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}</p>
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
    print("Starting server...")
    print("Model training will begin in the background...")
    
    # Start model training in background
    training_thread = start_training()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)