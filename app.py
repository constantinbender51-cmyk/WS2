import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flask import Flask, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Function to fetch OHLCV data from Binance
# Note: Binance API has limits; this fetches daily data for BTCUSDT
# Starting from 2018-01-01 to present
def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1d', start_date='2018-01-01'):
    """Fetch OHLCV data from Binance public API."""
    base_url = 'https://api.binance.com/api/v3/klines'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert dates to timestamps in milliseconds
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    limit = 1000  # Binance max per request
    
    while current_start < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': limit
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break
            
        data = response.json()
        if not data:
            break
            
        all_data.extend(data)
        
        # Update start time for next batch
        current_start = data[-1][0] + 1
        
        # Break if we've reached the end
        if len(data) < limit:
            break
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert timestamp to datetime and numeric columns to float
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Keep only relevant columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    return df

# Function to calculate log returns
def calculate_log_returns(df, column='close'):
    """Calculate log returns for a given column."""
    return np.log(df[column] / df[column].shift(1))

# Function to prepare features and target for linear regression
def prepare_regression_data(log_returns, poly_degree=1):
    """Prepare polynomial features from log returns for regression."""
    # Remove NaN values
    log_returns_clean = log_returns.dropna().values.reshape(-1, 1)
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X = poly.fit_transform(log_returns_clean)
    
    # Target is the next period's log return
    y = log_returns_clean[1:].ravel()
    X = X[:-1]  # Remove last row since we don't have target for it
    
    return X, y, poly

# Function to project future data using linear regression
def project_future_data(model, poly, last_log_return, periods=16*365):
    """Project future log returns using trained linear regression model."""
    # Start with the last known log return
    current_features = np.array([[last_log_return]])
    current_features_poly = poly.transform(current_features)
    
    projected_returns = []
    
    for _ in range(periods):
        # Predict next log return
        next_return = model.predict(current_features_poly)[0]
        projected_returns.append(next_return)
        
        # Update features for next prediction
        current_features = np.array([[next_return]])
        current_features_poly = poly.transform(current_features)
    
    return np.array(projected_returns)

# Function to convert log returns back to price
def log_returns_to_price(initial_price, log_returns):
    """Convert log returns back to price series."""
    cumulative_returns = np.cumsum(log_returns)
    prices = initial_price * np.exp(cumulative_returns)
    return prices

@app.route('/')
def index():
    """Main endpoint that displays the data and projections."""
    try:
        # Fetch data from Binance
        print("Fetching data from Binance...")
        df = fetch_binance_ohlcv()
        
        if df.empty:
            return "Error: No data fetched from Binance."
        
        print(f"Fetched {len(df)} data points from {df.index[0].date()} to {df.index[-1].date()}")
        
        # Calculate log returns
        log_returns = calculate_log_returns(df)
        
        # Prepare data for regression with degree 1 (straight line)
        X, y, poly = prepare_regression_data(log_returns, poly_degree=1)
        
        # Train linear regression model for degree 1
        model = LinearRegression()
        model.fit(X, y)
        
        # Get last log return for projection
        last_log_return = log_returns.dropna().iloc[-1]
        last_price = df['close'].iloc[-1]
        
        # Project future data (16 years = 16 * 365 days)
        periods = 16 * 365
        
        # Project using degree 1 model
        projected_returns = project_future_data(model, poly, last_log_return, periods)
        projected_prices = log_returns_to_price(last_price, projected_returns)
        
        # Create future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Create visualization - single plot with both datasets
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot historical price
        ax.plot(df.index, df['close'], label='Historical Price (2018-Present)', color='blue', linewidth=1.5)
        
        # Plot projected continuation price
        ax.plot(future_dates, projected_prices, label='Projected Continuation (Next 16 Years)', color='orange', linewidth=1.5, linestyle='--')
        
        # Add vertical line at the transition point
        ax.axvline(x=last_date, color='red', linestyle=':', alpha=0.7, label=f'Transition: {last_date.date()}')
        
        # Add horizontal line at last price
        ax.axhline(y=last_price, color='green', linestyle='--', alpha=0.5, label=f'Last Price: ${last_price:.2f}')
        
        ax.set_title('BTC/USDT Price with Linear Regression Projection (Degree 1)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USDT)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates for better readability
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Convert plot to base64 for HTML display
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Prepare data summary
        data_summary = {
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'data_points': len(df),
            'last_price': f"${last_price:.2f}",
            'last_log_return': f"{last_log_return:.6f}",
            'projection_start': future_dates[0].strftime('%Y-%m-%d'),
            'projection_end': future_dates[-1].strftime('%Y-%m-%d'),
            'projection_periods': periods,
            'model_score': f"{model.score(X, y):.4f}"
        }
        
        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BTC/USDT Price Analysis and Projection</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .summary h2 { margin-top: 0; color: #4CAF50; }
                .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }
                .summary-item { padding: 10px; background-color: white; border: 1px solid #ddd; border-radius: 5px; }
                .plot-container { text-align: center; margin: 20px 0; }
                .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                .disclaimer { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin-top: 20px; }
                .footer { margin-top: 30px; text-align: center; color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BTC/USDT Price Analysis and Projection</h1>
                
                <div class="summary">
                    <h2>Data Summary</h2>
                    <div class="summary-grid">
                        <div class="summary-item"><strong>Historical Period:</strong> {{ start_date }} to {{ end_date }}</div>
                        <div class="summary-item"><strong>Data Points:</strong> {{ data_points }}</div>
                        <div class="summary-item"><strong>Last Price:</strong> {{ last_price }}</div>
                        <div class="summary-item"><strong>Last Log Return:</strong> {{ last_log_return }}</div>
                        <div class="summary-item"><strong>Projection Period:</strong> {{ projection_start }} to {{ projection_end }}</div>
                        <div class="summary-item"><strong>Projection Periods:</strong> {{ projection_periods }} days (16 years)</div>
                        <div class="summary-item"><strong>Model R² Score (Degree 1):</strong> {{ model_score }}</div>
                    </div>
                </div>
                
                <div class="plot-container">
                    <h2>Price and Continuation Dataset</h2>
                    <p>Historical price (blue solid line) and projected continuation (orange dashed line) using linear regression with degree 1 polynomial features.</p>
                    <img src="data:image/png;base64,{{ plot_url }}" alt="Price and Continuation Plot">
                </div>
                
                <div class="disclaimer">
                    <h3>Important Disclaimer</h3>
                    <p>This is a technical demonstration only. The linear regression model with polynomial features is a simple statistical approach and should not be used for financial decision-making. Financial markets are complex and influenced by many factors not captured in this model. Past performance does not guarantee future results.</p>
                    <p>The projection of 16 years into the future is purely mathematical and based on historical patterns. It does not account for market changes, economic events, or other real-world factors.</p>
                </div>
                
                <div class="footer">
                    <p>Data Source: Binance Public API | Generated on {{ current_time }}</p>
                    <p>Note: This tool fetches real-time data from Binance. The projection is updated with each page refresh.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Render template with data
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        html = render_template_string(html_template, 
                                     plot_url=plot_url,
                                     current_time=current_time,
                                     **data_summary)
        
        return html
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print("Starting web server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)