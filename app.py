import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from datetime import datetime
import requests
import json

app = Flask(__name__)

# Function to fetch BTC OHLCV data from Binance
# Binance API endpoint for klines (OHLCV)
def fetch_btc_ohlcv(start_date='2018-01-01', end_date=None):
    """
    Fetch BTC/USDT OHLCV data from Binance.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format, defaults to current date
    
    Returns:
    pandas.DataFrame: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert dates to milliseconds for Binance API
    start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    # Binance API parameters
    symbol = 'BTCUSDT'
    interval = '1d'  # Daily data
    limit = 1000  # Max per request
    
    all_data = []
    current_start = start_timestamp
    
    while current_start < end_timestamp:
        # Calculate current end (min of end_timestamp or current_start + limit days)
        current_end = min(current_start + (limit * 24 * 60 * 60 * 1000), end_timestamp)
        
        url = f'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': current_end,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                all_data.extend(data)
                # Update start time for next batch
                current_start = int(data[-1][0]) + 1
            else:
                break
        else:
            print(f"Error fetching data: {response.status_code}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(all_data, columns=columns)
    
    # Convert types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    # Keep only necessary columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    return df

# Function to calculate Rogers Satchell estimator
# Rogers Satchell estimator formula: sqrt(1/n * sum((ln(high/close) * ln(high/open) + ln(low/close) * ln(low/open))))
def calculate_rogers_satchell(df):
    """
    Calculate Rogers Satchell volatility estimator.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns
    
    Returns:
    pandas.Series: Rogers Satchell estimator values
    """
    # Calculate log returns
    log_high_close = np.log(df['high'] / df['close'])
    log_high_open = np.log(df['high'] / df['open'])
    log_low_close = np.log(df['low'] / df['close'])
    log_low_open = np.log(df['low'] / df['open'])
    
    # Calculate Rogers Satchell estimator for each period
    rs_estimator = np.sqrt(log_high_close * log_high_open + log_low_close * log_low_open)
    
    # Calculate rolling Rogers Satchell estimator (30-day window)
    window = 30
    rolling_rs = rs_estimator.rolling(window=window).mean() * np.sqrt(252)  # Annualized
    
    return rolling_rs

# Function to calculate SMAs
def calculate_smas(df):
    """
    Calculate 120-day and 365-day simple moving averages.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'close' column
    
    Returns:
    tuple: (sma_120, sma_365)
    """
    sma_120 = df['close'].rolling(window=120).mean()
    sma_365 = df['close'].rolling(window=365).mean()
    return sma_120, sma_365


# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(compounded_returns):
    """
    Calculate Sharpe ratio from compounded returns.
    
    Parameters:
    compounded_returns (pandas.Series): Compounded returns series
    
    Returns:
    float: Annualized Sharpe ratio
    """
    if compounded_returns.empty or len(compounded_returns) < 2:
        return None
    
    # Calculate daily returns from compounded returns
    daily_returns = compounded_returns.pct_change().dropna()
    
    if len(daily_returns) == 0:
        return None
    
    # Calculate mean and std of daily returns
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    # Avoid division by zero
    if std_daily_return == 0:
        return None
    
    # Calculate Sharpe ratio (annualized)
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
    
    return sharpe_ratio# Function to calculate compounded returns with leverage
def calculate_compounded_returns(df, rs_estimator, factor=3):
    """
    Calculate compounded returns based on conditions:
    - When price > 365 SMA and > 120 SMA: use positive returns
    - When price < 365 SMA and < 120 SMA: use negative returns
    - Otherwise: 0
    Multiply returns by leverage = rs_estimator * factor
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'close' column
    rs_estimator (pandas.Series): Rogers Satchell estimator values
    factor (float): Factor to multiply with rs_estimator for leverage (default 3)
    
    Returns:
    pandas.Series: Compounded returns series
    """
    # Calculate SMAs
    sma_120, sma_365 = calculate_smas(df)
    
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    
    # Initialize leverage-adjusted returns with zeros
    adjusted_returns = pd.Series(0.0, index=df.index)
    
    # Create conditions
    above_both = (df['close'] > sma_120) & (df['close'] > sma_365)
    below_both = (df['close'] < sma_120) & (df['close'] < sma_365)
    
    # Apply conditions
    # When above both SMAs: use positive returns
    adjusted_returns[above_both] = daily_returns[above_both]
    # When below both SMAs: use negative returns (inverse position)
    adjusted_returns[below_both] = -daily_returns[below_both]
    # Otherwise: already 0
    
    # Calculate leverage (using fixed value 4)
    leverage = 4
    
    # Apply leverage
    leveraged_returns = adjusted_returns * leverage.shift(1)  # Use previous day's leverage
    
    # Calculate compounded returns
    compounded_returns = (1 + leveraged_returns).cumprod()
    
    # Normalize to start at 1
    if not compounded_returns.empty:
        first_valid = compounded_returns.first_valid_index()
        if first_valid is not None:
            start_value = compounded_returns.loc[first_valid]
            if start_value != 0:
                compounded_returns = compounded_returns / start_value
    
    return compounded_returns

# Function to create plot
def create_plot(df, rs_estimator, compounded_returns):
    """
    Create a plot with BTC price, Rogers Satchell estimator, and compounded returns.
    
    Parameters:
    df (pandas.DataFrame): BTC price data
    rs_estimator (pandas.Series): Rogers Satchell estimator values
    compounded_returns (pandas.Series): Compounded returns series
    
    Returns:
    str: Base64 encoded image of the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # Plot BTC price with SMAs
    ax1.plot(df.index, df['close'], label='BTC Close Price', color='blue', linewidth=1)
    
    # Calculate and plot SMAs
    sma_120, sma_365 = calculate_smas(df)
    ax1.plot(sma_120.index, sma_120, label='120-day SMA', color='orange', linewidth=1, alpha=0.7)
    ax1.plot(sma_365.index, sma_365, label='365-day SMA', color='green', linewidth=1, alpha=0.7)
    
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.set_title('BTC/USDT Price, Rogers Satchell Estimator, and Compounded Returns (2018-Present)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot Rogers Satchell estimator
    ax2.plot(rs_estimator.index, rs_estimator, label='Rogers Satchell Estimator (30-day, Annualized)', 
             color='red', linewidth=1)
    ax2.set_ylabel('Volatility Estimator', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Plot compounded returns
    ax3.plot(compounded_returns.index, compounded_returns, 
             label='Compounded Returns (Leverage = 4)', color='purple', linewidth=2)
    ax3.set_ylabel('Compounded Returns (Normalized)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to bytes
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plt.close(fig)
    
    # Encode to base64
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return plot_url

# Route for the main page
@app.route('/')
def index():
    # Fetch data
    print("Fetching BTC data from Binance...")
    df = fetch_btc_ohlcv(start_date='2018-01-01')
    
    if df.empty:
        return "Error: Could not fetch data from Binance. Please try again later."
    
    # Calculate Rogers Satchell estimator
    print("Calculating Rogers Satchell estimator...")
    rs_estimator = calculate_rogers_satchell(df)
    
    # Calculate compounded returns
    print("Calculating compounded returns...")
    compounded_returns = calculate_compounded_returns(df, rs_estimator, factor=1)
    
    # Create plot
    print("Creating plot...")
    plot_url = create_plot(df, rs_estimator, compounded_returns)
    
    # Get latest values
    latest_price = df['close'].iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    latest_rs = rs_estimator.iloc[-1] if not rs_estimator.isna().all() else None
    latest_compounded = compounded_returns.iloc[-1] if not compounded_returns.empty else None
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(compounded_returns)
    
    # Create HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTC Price and Rogers Satchell Estimator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .stat-box {
                text-align: center;
                padding: 10px;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }
            .stat-label {
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 5px;
            }
            .plot-container {
                text-align: center;
                margin-top: 20px;
            }
            .plot-img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .info {
                margin-top: 20px;
                padding: 15px;
                background-color: #e8f4f8;
                border-radius: 5px;
                font-size: 14px;
                color: #2c3e50;
            }
            .footer {
                margin-top: 20px;
                text-align: center;
                font-size: 12px;
                color: #95a5a6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BTC/USDT Price and Rogers Satchell Volatility Estimator</h1>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{{ latest_price }}</div>
                    <div class="stat-label">Latest BTC Price (USDT)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ latest_date }}</div>
                    <div class="stat-label">Latest Data Date</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ latest_rs }}</div>
                    <div class="stat-label">Latest Rogers Satchell Estimator</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ latest_compounded }}</div>
                    <div class="stat-label">Latest Compounded Returns</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ sharpe_ratio }}</div>
                    <div class="stat-label">Sharpe Ratio (Annualized)</div>
                </div>
            </div>
            
            <div class="plot-container">
                <img src="data:image/png;base64,{{ plot_url }}" class="plot-img" alt="BTC Price and Rogers Satchell Estimator Plot">
            </div>
            
            <div class="info">
                <h3>About This Visualization</h3>
                <p><strong>Data Source:</strong> Binance API (BTC/USDT daily OHLCV from 2018-01-01 to present)</p>
                <p><strong>Rogers Satchell Estimator:</strong> A volatility estimator that accounts for opening jumps. Calculated as: √(1/n * Σ(ln(high/close) * ln(high/open) + ln(low/close) * ln(low/open)))</p>
                <p><strong>Compounded Returns:</strong> Calculated based on conditions: positive returns when price > 365-day and 120-day SMAs, negative returns when price < both SMAs, otherwise 0. Returns are multiplied by leverage = Rogers Satchell estimator × 3.</p>
                <p><strong>Plot Details:</strong> Top chart shows BTC closing price with 120-day and 365-day SMAs. Middle chart shows the 30-day rolling Rogers Satchell estimator (annualized). Bottom chart shows compounded returns (normalized to start at 1).</p>
                <p><strong>Note:</strong> This is a technical implementation for educational purposes only.</p>
            </div>
            
            <div class="footer">
                <p>Data fetched from Binance | Rogers Satchell estimator calculated on daily data | Server running on port 8080</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                 latest_price=f"${latest_price:,.2f}",
                                 latest_date=latest_date,
                                 latest_rs=f"{latest_rs:.4f}" if latest_rs else "N/A",
                                 latest_compounded=f"{latest_compounded:.4f}" if latest_compounded else "N/A",
                                 sharpe_ratio=f"{sharpe_ratio:.4f}" if sharpe_ratio is not None else "N/A",
                                 plot_url=plot_url)

if __name__ == '__main__':
    print("Starting web server on port 8080...")
    print("Open http://localhost:8080 in your browser to view the plot.")
    app.run(host='0.0.0.0', port=8080, debug=False)