import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from flask import Flask, render_template_string

app = Flask(__name__)

# HTML template for the web page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Binance OHLCV with Noisy SMA</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        p { color: #666; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .container { max-width: 1200px; margin: 0 auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Binance BTC/USDT OHLCV with Noisy 120-period SMA (Shifted 60 Left)</h1>
        <p>Data fetched from Binance starting from 2018-01-01. The plot shows the closing prices and the noisy SMA shifted 60 periods left.</p>
        <img src="data:image/png;base64,{{ plot_data }}" alt="OHLCV with Noisy SMA Plot">
        <p>Generated at: {{ timestamp }}</p>
    </div>
</body>
</html>
"""

def fetch_binance_ohlcv(symbol='BTCUSDT', interval='1d', start_date='2018-01-01'):
    """
    Fetch OHLCV data from Binance API.
    
    Args:
        symbol: Trading pair symbol (default: BTCUSDT)
        interval: Kline interval (default: 1d)
        start_date: Start date in YYYY-MM-DD format (default: 2018-01-01)
    
    Returns:
        pandas.DataFrame with columns: timestamp, open, high, low, close, volume
    """
    base_url = 'https://api.binance.com/api/v3/klines'
    
    # Convert start_date to timestamp in milliseconds
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    
    all_data = []
    limit = 1000  # Binance API limit per request
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_timestamp,
            'limit': limit
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        
        # Update start_timestamp for next batch
        start_timestamp = data[-1][0] + 1
        
        # Break if we've fetched all data up to now
        if len(data) < limit:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert columns to appropriate types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df[['open', 'high', 'low', 'close', 'volume']]


def compute_sma_with_noise(df, window=120, noise_level=0.1):
    """
    Compute SMA and add noise to it.
    
    Args:
        df: DataFrame with 'close' column
        window: SMA window period (default: 120)
        noise_level: Standard deviation of Gaussian noise as fraction of SMA (default: 0.1 = 10%)
    
    Returns:
        DataFrame with added 'sma', 'noisy_sma', and 'noisy_sma_shifted' columns
    """
    df = df.copy()
    
    # Compute SMA
    df['sma'] = df['close'].rolling(window=window).mean()
    
    # Add noise to SMA
    # Only add noise where SMA is not NaN
    sma_values = df['sma'].dropna()
    if len(sma_values) > 0:
        # Calculate derivative of SMA using gradient
        sma_derivative = np.gradient(sma_values.values)
        
        # Calculate absolute distance traveled using a 30-day rolling window
        # Create a rolling window of 30 periods for cumulative sum of absolute derivatives
        abs_derivatives = np.abs(sma_derivative)
        
        # Initialize absolute_distance_traveled array
        absolute_distance_traveled = np.zeros_like(sma_derivative)
        
        # Calculate rolling cumulative sum for 30-day window
        for i in range(len(abs_derivatives)):
            start_idx = max(0, i - 29)  # 30-day window (inclusive of current point)
            absolute_distance_traveled[i] = np.sum(abs_derivatives[start_idx:i+1])
        
        # Use the SMA value as the base for noise magnitude
        # Higher SMA values result in more noise
        noise_magnitude = sma_values.values * noise_level
        
        # Create noise array based on the SMA value
        noise = np.random.normal(0, noise_magnitude, len(sma_values))
        
        # Add noise to SMA (no longer dependent on distance traveled)
        df.loc[sma_values.index, 'noisy_sma'] = sma_values + noise
        
        # Shift noisy SMA 60 periods to the left
        df['noisy_sma_shifted'] = df['noisy_sma'].shift(-60)
    else:
        df['noisy_sma'] = np.nan
        df['noisy_sma_shifted'] = np.nan
    
    return df


def create_plot(df):
    """
    Create a plot of closing prices and noisy SMA.
    
    Args:
        df: DataFrame with 'close', 'noisy_sma', and 'noisy_sma_shifted' columns
    
    Returns:
        Base64 encoded PNG image string
    """
    plt.figure(figsize=(12, 6))
    
    # Plot closing price
    plt.plot(df.index, df['close'], label='Close Price', alpha=0.7, linewidth=1)
    
    # Plot shifted noisy SMA if available
    if 'noisy_sma_shifted' in df.columns and not df['noisy_sma_shifted'].isna().all():
        plt.plot(df.index, df['noisy_sma_shifted'], label='Noisy SMA (shifted 60 left)', 
                color='green', linewidth=2, alpha=0.8)
    
    plt.title('Binance BTC/USDT - Close Price with Noisy 120-period SMA (Shifted 60 Left)')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    # Encode to base64
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    return plot_data


@app.route('/')
def index():
    """Main endpoint that displays the plot."""
    try:
        # Fetch data from Binance
        print("Fetching data from Binance...")
        df = fetch_binance_ohlcv()
        
        # Compute SMA with noise
        print("Computing SMA with noise...")
        df = compute_sma_with_noise(df, window=120, noise_level=0.1)
        
        # Create plot
        print("Creating plot...")
        plot_data = create_plot(df)
        
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Render HTML with plot
        return render_template_string(HTML_TEMPLATE, 
                                    plot_data=plot_data, 
                                    timestamp=timestamp)
    
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error</h1>
            <p>An error occurred: {str(e)}</p>
            <p>Please check the server logs for details.</p>
        </body>
        </html>
        """
        return error_html


if __name__ == '__main__':
    print("Starting web server on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)