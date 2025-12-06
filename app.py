import ccxt
import pandas as pd
import numpy as np
import datetime as dt
from flask import Flask, render_template
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
    
    # Create price chart
    # Convert datetime index to string for proper JSON serialization
    date_strings = df.index.strftime('%Y-%m-%d').tolist()
    close_prices = df['close'].tolist()  # Convert Series to list for proper JSON serialization
    
    # Debug: Check data being passed to chart
    print(f"Price chart data - Dates count: {len(date_strings)}, Prices count: {len(close_prices)}")
    if len(date_strings) > 0 and len(close_prices) > 0:
        print(f"  First date: {date_strings[0]}, First price: {close_prices[0]:.2f}")
        print(f"  Last date: {date_strings[-1]}, Last price: {close_prices[-1]:.2f}")
    
    fig_price = px.line(x=date_strings, y=close_prices, 
                        title=f'{SYMBOL} Price (Daily Close)',
                        labels={'y': 'Price (USDT)', 'x': 'Date'})
    fig_price.update_layout(
        hovermode="x unified", 
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        height=500
    )
    
    # Create inefficiency index chart
    if inefficiency_series.empty:
        # Create an empty figure if no data
        fig_inefficiency = go.Figure()
        fig_inefficiency.update_layout(
            title=f'{SYMBOL} Inefficiency Index ({ROLLING_WINDOW_DAYS}-day Rolling) - No Data',
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Inefficiency Index",
            height=500,
            annotations=[dict(
                text="No inefficiency index data available (check window size or data)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )]
        )
    else:
        # Convert datetime index to string for proper JSON serialization
        inefficiency_dates = inefficiency_series.index.strftime('%Y-%m-%d').tolist()
        inefficiency_values = inefficiency_series.tolist()  # Convert Series to list for proper JSON serialization
        
        # Debug: Check inefficiency data being passed to chart
        print(f"Inefficiency chart data - Dates count: {len(inefficiency_dates)}, Values count: {len(inefficiency_values)}")
        if len(inefficiency_dates) > 0 and len(inefficiency_values) > 0:
            print(f"  First date: {inefficiency_dates[0]}, First value: {inefficiency_values[0]:.2f}")
            print(f"  Last date: {inefficiency_dates[-1]}, Last value: {inefficiency_values[-1]:.2f}")
        
        fig_inefficiency = px.line(x=inefficiency_dates, y=inefficiency_values,
                                   title=f'{SYMBOL} Inefficiency Index ({ROLLING_WINDOW_DAYS}-day Rolling)',
                                   labels={'y': 'Inefficiency Index', 'x': 'Date'})
        fig_inefficiency.update_layout(
            hovermode="x unified", 
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Inefficiency Index",
            height=500
        )
        
        # Add better y-axis range for visualization
        if inefficiency_series.max() > 10:
            fig_inefficiency.update_yaxes(range=[0, min(100, inefficiency_series.max() * 1.1)])
    
    # Convert to JSON for template
    graphJSON_price = pio.to_json(fig_price)
    graphJSON_inefficiency = pio.to_json(fig_inefficiency)
    
    return render_template('index.html',
                           graphJSON_price=graphJSON_price,
                           graphJSON_inefficiency=graphJSON_inefficiency,
                           symbol=SYMBOL,
                           window=ROLLING_WINDOW_DAYS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)