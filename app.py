import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, request
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
    'last_update': None,
    'periods': []  # Store available periods for pagination
}

# Fetch Bitcoin data from Binance API with pagination
def fetch_btc_data_paginated():
    # Start from 2018-01-01
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now() + timedelta(days=1)
    
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        # Set end date for this chunk (max 1000 days per request)
        chunk_end = min(current_start + timedelta(days=999), end_date)
        
        # Format dates for Binance API
        start_timestamp = int(current_start.timestamp() * 1000)
        end_timestamp = int(chunk_end.timestamp() * 1000)
        
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1d',
            'startTime': start_timestamp,
            'endTime': end_timestamp,
            'limit': 1000
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data:
            all_data.extend(data)
        
        # Move to next chunk
        current_start = chunk_end + timedelta(seconds=1)
        
        # Be respectful to the API
        time.sleep(0.1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
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

# Get available 6-month periods for pagination
def get_available_periods():
    periods = []
    start_date = datetime(2018, 1, 1)
    current_date = datetime.now()
    
    # Create 6-month periods from 2018 to 2026
    year = 2018
    while year <= 2026:
        # First half of the year
        period1_start = datetime(year, 1, 1)
        if period1_start <= current_date:
            periods.append({
                'start': period1_start,
                'end': datetime(year, 6, 30),
                'label': f"{year} Jan-Jun",
                'id': f"{year}_H1"
            })
        
        # Second half of the year
        period2_start = datetime(year, 7, 1)
        if period2_start <= current_date:
            periods.append({
                'start': period2_start,
                'end': datetime(year, 12, 31),
                'label': f"{year} Jul-Dec",
                'id': f"{year}_H2"
            })
        
        year += 1
    
    return periods

# Train the model in a separate thread
def train_model():
    global model_state
    
    print("Starting model training...")
    print("Fetching Bitcoin data with pagination...")
    btc_data = fetch_btc_data_paginated()
    
    print("Calculating 21-day SMA...")
    sma21 = calculate_sma21(btc_data)
    sma21_clean = sma21.dropna()
    
    # Get available periods
    periods = get_available_periods()
    model_state['periods'] = periods
    
    # Use the most recent period for initial model
    if periods:
        current_period = periods[-1]  # Most recent period
    else:
        # Default to recent 6 months if no periods found
        current_period = {
            'start': datetime.now() - timedelta(days=180),
            'end': datetime.now(),
            'label': "Recent 6 months",
            'id': "recent"
        }
    
    # Filter data for the selected period
    period_data = sma21_clean[
        (sma21_clean.index >= current_period['start']) & 
        (sma21_clean.index <= current_period['end'])
    ]
    
    # Add next month for testing
    test_start = current_period['end'] + timedelta(days=1)
    test_end = test_start + timedelta(days=30)  # 1 month for testing
    
    test_data_full = sma21_clean[
        (sma21_clean.index >= test_start) & 
        (sma21_clean.index <= test_end)
    ]
    
    # Combine train and test data for visualization
    combined_data = pd.concat([period_data, test_data_full])
    
    # Define train/test split
    if len(test_data_full) > 0:
        train_data = period_data
        test_data = test_data_full
    else:
        # If no test data available, use last 30 days of period as test
        train_data = period_data.iloc[:-30] if len(period_data) > 30 else period_data
        test_data = period_data.iloc[-30:] if len(period_data) > 30 else pd.Series(dtype=float)
    
    print(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")
    
    if len(train_data) < 50:
        print("Not enough training data, using last 100 available points")
        train_data = sma21_clean.tail(100)
        test_data = pd.Series(dtype=float)
    
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
    
    # Plot all available data for context
    plt.plot(sma21_clean.index, sma21_clean.values, label='All Historical 21-day SMA', color='lightgray', alpha=0.5)
    
    # Highlight the selected period
    plt.plot(period_data.index, period_data.values, label='Selected 6-Month Period', color='blue')
    
    # Plot test data if available
    if len(test_data) > 0:
        plt.plot(test_data.index, test_data.values, label='Test Data (Next Month)', color='green')
    
    # Plot forecast
    forecast_dates = pd.date_range(start=sma21_clean.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--', linewidth=2)
    
    # Confidence intervals
    plt.fill_between(forecast_dates, 
                     forecast_ci.iloc[:, 0], 
                     forecast_ci.iloc[:, 1], 
                     color='red', alpha=0.2, label='Confidence Interval')
    
    plt.title('Bitcoin 21-Day SMA Forecast using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    
    # Get period ID from query parameter, default to latest
    period_id = request.args.get('period', 'latest')
    
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
    
    # Generate pagination links
    periods_html = ""
    if model_state['periods']:
        periods_html = "<div class='pagination'><h3>Select Period:</h3><ul style='list-style-type: none; padding: 0;'>"
        for period in model_state['periods'][-10:]:  # Show last 10 periods
            periods_html += f"<li style='display: inline-block; margin-right: 10px;'><a href='/?period={period['id']}' style='padding: 5px 10px; background-color: #007bff; color: white; text-decoration: none; border-radius: 3px;'>{period['label']}</a></li>"
        periods_html += "</ul></div>"
    
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
            .pagination a:visited {{ color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bitcoin 21-Day SMA ARIMA Forecast</h1>
            
            {periods_html}
            
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