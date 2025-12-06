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
ROLLING_WINDOW_DAYS = 30
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
        return pd.Series([], dtype=float), pd.Series([], dtype=float)
    
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
    
    # Compute inverse inefficiency index (1/x)
    inverse_inefficiency_index = 1 / inefficiency_index
    
    # Handle division by zero or near-zero values in the original index
    inverse_inefficiency_index = inverse_inefficiency_index.replace([np.inf, -np.inf], np.nan)
    
    # Clean the data for plotting
    # Remove NaN values and cap extreme values for better visualization
    inverse_inefficiency_index_clean = inverse_inefficiency_index.dropna()
    
    # Cap extreme values at 100 for visualization
    if not inverse_inefficiency_index_clean.empty:
        inverse_inefficiency_index_clean = inverse_inefficiency_index_clean.clip(upper=100)
    
    # Compute 14-day SMA for smoothing the inverse inefficiency index
    sma_window = 14
    inverse_inefficiency_index_smoothed = inverse_inefficiency_index_clean.rolling(window=sma_window, min_periods=1).mean()
    
    return inverse_inefficiency_index_clean, inverse_inefficiency_index_smoothed

# Web server routes
@app.route('/')
def index():
    df = fetch_ohlcv(SYMBOL, START_DATE_STR)
    
    if df is None or df.empty:
        print("DataFrame is None or empty. Using sample data.")
        df = generate_sample_data()

    # Calculate simple returns for the entire DataFrame
    df['returns'] = df['close'].pct_change()
    
    # Debug: Print data info
    print(f"DataFrame info:")
    print(f"  Shape: {df.shape}")
    print(f"  Index range: {df.index[0]} to {df.index[-1]}")
    print(f"  Close price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    
    inefficiency_series, inefficiency_smoothed = calculate_inefficiency_index(df, ROLLING_WINDOW_DAYS)    
    
    # Calculate the new metric: yesterday's smoothed inefficiency * today's log return
    # Create a temporary DataFrame to align series by index
    temp_df = pd.DataFrame(index=df.index)
    temp_df['returns'] = df['returns']
    temp_df['iii_sma'] = inefficiency_smoothed # This aligns by index, filling with NaNs where no match

    temp_df['iii_sma_yesterday'] = temp_df['iii_sma'].shift(1)
    
    # Calculate the daily factor for compounding
    daily_compounding_factor = 1 + (temp_df['iii_sma_yesterday'] * temp_df['returns'])
    
    # Compute the cumulative product for compounding
    cumulative_compounded_series = daily_compounding_factor.cumprod()
    
    iii_sma_x_returns = cumulative_compounded_series.dropna()

    # Debug: Print some statistics about the inefficiency index
    print(f"Data length: {len(df)}")
    print(f"Inefficiency series length: {len(inefficiency_series)}")
    if not inefficiency_series.empty:
        print(f"Inefficiency index stats - min: {inefficiency_series.min():.2f}, max: {inefficiency_series.max():.2f}, mean: {inefficiency_series.mean():.2f}")
        print(f"First 5 values: {inefficiency_series.head().tolist()}")
        print(f"Last 5 values: {inefficiency_series.tail().tolist()}")
    if not inefficiency_smoothed.empty:
        print(f"Smoothed inefficiency index stats - min: {inefficiency_smoothed.min():.2f}, max: {inefficiency_smoothed.max():.2f}, mean: {inefficiency_smoothed.mean():.2f}")
    
    # Create combined price and inverse inefficiency index chart with second y-axis
    plt.figure(figsize=(12, 6))
    
    # Plot price on primary y-axis (left)
    ax1 = plt.gca()
    ax1.plot(df.index, df['close'], color='blue', linewidth=1.5, label='Price')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price (USDT)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # Plot smoothed inverse inefficiency index (14-day SMA) on secondary y-axis (right) if data exists
    if not inefficiency_smoothed.empty:
        ax2 = ax1.twinx()
        ax2.plot(inefficiency_smoothed.index, inefficiency_smoothed.values, color='darkred', linewidth=2, label='Inverse Inefficiency Index (14-day SMA)')
        ax2.set_ylabel('Inverse Inefficiency Index (1/x)', fontsize=12, color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        # Set y-axis range for inverse inefficiency index if needed
        if inefficiency_smoothed.max() > 10:
            ax2.set_ylim(0, min(100, inefficiency_smoothed.max() * 1.1))
        # Add legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        # Add legend for price only if no inefficiency data
        ax1.legend(loc='upper left')
    
    plt.title(f'{SYMBOL} Price with Inverse Inefficiency Index ({ROLLING_WINDOW_DAYS}-day Rolling, 14-day SMA Only)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot to base64 string
    img_combined = io.BytesIO()
    plt.savefig(img_combined, format='png', dpi=100)
    plt.close()
    img_combined.seek(0)
    combined_chart_url = base64.b64encode(img_combined.getvalue()).decode('utf8')

    # Create the second plot for III SMA * Log Returns
    iii_x_returns_chart_url = None # Initialize to None
    if not iii_sma_x_returns.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(iii_sma_x_returns.index, iii_sma_x_returns.values, color='purple', linewidth=1.5)
        plt.title(f'{SYMBOL} Cumulative Product of (1 + Yesterday\'s III SMA * Today\'s Return) ({ROLLING_WINDOW_DAYS}-day Rolling, 14-day SMA)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Compounded Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        img_iii_x_returns = io.BytesIO()
        plt.savefig(img_iii_x_returns, format='png', dpi=100)
        plt.close()
        img_iii_x_returns.seek(0)
        iii_x_returns_chart_url = base64.b64encode(img_iii_x_returns.getvalue()).decode('utf8')
    
    return render_template('index.html',
                           combined_chart_url=combined_chart_url,
                           iii_x_returns_chart_url=iii_x_returns_chart_url,
                           symbol=SYMBOL,
                           window=ROLLING_WINDOW_DAYS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)