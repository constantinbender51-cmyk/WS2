import ccxt
import pandas as pd
import numpy as np
import datetime as dt
from flask import Flask, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import time

# Configuration
SYMBOL = 'BTC/USDT'
START_DATE_STR = '2018-01-01'
ROLLING_WINDOW_DAYS = 60
PORT = 8080

app = Flask(__name__)

# Generate sample data for demonstration
def generate_sample_data():
    print("Generating sample data for demonstration...")
    dates = pd.date_range(start=START_DATE_STR, end=pd.Timestamp.now(), freq='D')
    n = len(dates)
    
    # Generate realistic BTC price data
    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.0005, 0.04, n)  # Daily returns
    price = base_price * np.exp(np.cumsum(returns))
    
    # Add some volatility clusters
    for i in range(5):
        start = np.random.randint(0, n-100)
        length = np.random.randint(20, 60)
        price[start:start+length] *= (1 + np.random.normal(0, 0.2, length))
    
    df = pd.DataFrame({
        'open': price * 0.99,
        'high': price * 1.02,
        'low': price * 0.98,
        'close': price,
        'volume': np.random.lognormal(10, 1, n) * 1000
    }, index=dates)
    
    print(f"Generated {len(df)} sample daily candles for {SYMBOL}.")
    return df

# Data fetching function
def fetch_ohlcv(symbol, since_date_str):
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        since_ms = exchange.parse8601(since_date_str + 'T00:00:00Z')
        all_ohlcv = []
        
        print(f"Attempting to fetch {symbol} OHLCV data from {since_date_str}...")
        
        # Try to fetch data
        exchange.load_markets()
        if symbol not in exchange.symbols:
            print(f"Symbol {symbol} not found on Binance. Using sample data.")
            return generate_sample_data()
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                while True:
                    ohlcv = exchange.fetch_ohlcv(symbol, '1d', since_ms, limit=1000)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    since_ms = ohlcv[-1][0] + (24 * 60 * 60 * 1000)
                    
                    if since_ms > exchange.milliseconds():
                        break
                    
                    print(f"Fetched {len(all_ohlcv)} entries, continuing...")
                    time.sleep(1)  # Rate limiting
                
                if all_ohlcv:
                    break
                
            except (ccxt.DDoSProtection, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
                print(f"Exchange error (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
                    continue
                else:
                    print("Max attempts reached. Using sample data.")
                    return generate_sample_data()
            except Exception as e:
                print(f"Unexpected error (attempt {attempt+1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
                    continue
                else:
                    print("Max attempts reached. Using sample data.")
                    return generate_sample_data()
        
        if not all_ohlcv:
            print("No OHLCV data fetched. Using sample data.")
            return generate_sample_data()
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"Successfully fetched {len(df)} daily candles for {symbol}.")
        return df
        
    except Exception as e:
        print(f"Critical error in fetch_ohlcv: {e}")
        print("Falling back to sample data...")
        return generate_sample_data()

# Calculate inefficiency index
def calculate_inefficiency_index(df, window_days):
    if df is None or len(df) < window_days:
        return pd.Series([], dtype=float)
    
    # Calculate log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate rolling sums using numpy functions
    rolling_sum_abs_log_returns = log_returns.rolling(window=window_days).apply(
        lambda x: np.nansum(np.abs(x)) if not np.all(np.isnan(x)) else np.nan, raw=True
    )
    rolling_sum_log_returns = log_returns.rolling(window=window_days).sum()
    
    # Calculate inefficiency index
    inefficiency_index = rolling_sum_abs_log_returns / np.abs(rolling_sum_log_returns)
    
    # Handle edge cases
    inefficiency_index = inefficiency_index.replace([np.inf, -np.inf], np.nan)
    
    # Handle near-zero denominators
    denominator_mask = np.abs(rolling_sum_log_returns) < 1e-9
    inefficiency_index[denominator_mask] = np.nan
    
    # Clean the data for plotting
    # Remove NaN values and cap extreme values for better visualization
    inefficiency_index_clean = inefficiency_index.dropna()
    
    # Cap extreme values at 100 for visualization (still shows high inefficiency)
    if not inefficiency_index_clean.empty:
        inefficiency_index_clean = inefficiency_index_clean.clip(upper=100)
    
    return inefficiency_index_clean

# Web server routes
@app.route('/')
def index():
    df = fetch_ohlcv(SYMBOL, START_DATE_STR)
    
    if df is None or df.empty:
        print("DataFrame is None or empty. Using sample data.")
        df = generate_sample_data()
    
    # Debug: Print data info
    print(f"DataFrame info:")
    print(f"  Shape: {df.shape}")
    print(f"  Index range: {df.index[0]} to {df.index[-1]}")
    print(f"  Close price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    
    inefficiency_series = calculate_inefficiency_index(df, ROLLING_WINDOW_DAYS)    
    # Debug: Print some statistics about the inefficiency index
    print(f"Data length: {len(df)}")
    print(f"Inefficiency series length: {len(inefficiency_series)}")
    if not inefficiency_series.empty:
        print(f"Inefficiency index stats - min: {inefficiency_series.min():.2f}, max: {inefficiency_series.max():.2f}, mean: {inefficiency_series.mean():.2f}")
        print(f"First 5 values: {inefficiency_series.head().tolist()}")
        print(f"Last 5 values: {inefficiency_series.tail().tolist()}")
    
    # Create price chart with Matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], color='blue', linewidth=1.5)
    plt.title(f'{SYMBOL} Price (Daily Close)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot to base64 string
    img_price = io.BytesIO()
    plt.savefig(img_price, format='png', dpi=100)
    plt.close()
    img_price.seek(0)
    price_chart_url = base64.b64encode(img_price.getvalue()).decode('utf8')
    
    # Create inefficiency index chart
    if inefficiency_series.empty:
        # Create an empty figure if no data
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, 'No inefficiency index data available\n(check window size or data)', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title(f'{SYMBOL} Inefficiency Index ({ROLLING_WINDOW_DAYS}-day Rolling) - No Data', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Inefficiency Index', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(inefficiency_series.index, inefficiency_series.values, color='red', linewidth=1.5)
        plt.title(f'{SYMBOL} Inefficiency Index ({ROLLING_WINDOW_DAYS}-day Rolling)', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Inefficiency Index', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Set y-axis range if needed
        if inefficiency_series.max() > 10:
            plt.ylim(0, min(100, inefficiency_series.max() * 1.1))
        
        plt.tight_layout()
    
    # Save plot to base64 string
    img_inefficiency = io.BytesIO()
    plt.savefig(img_inefficiency, format='png', dpi=100)
    plt.close()
    img_inefficiency.seek(0)
    inefficiency_chart_url = base64.b64encode(img_inefficiency.getvalue()).decode('utf8')
    
    return render_template('index.html',
                           price_chart_url=price_chart_url,
                           inefficiency_chart_url=inefficiency_chart_url,
                           symbol=SYMBOL,
                           window=ROLLING_WINDOW_DAYS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)