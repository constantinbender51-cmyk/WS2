import ccxt
import pandas as pd
import numpy as np
import datetime as dt
from flask import Flask, render_template
import plotly.express as px
import plotly.graph_objects as go
import json

# Configuration
SYMBOL = 'BTC/USDT'
START_DATE_STR = '2018-01-01'
ROLLING_WINDOW_DAYS = 30
PORT = 8080

app = Flask(__name__)

# Data fetching function
def fetch_ohlcv(symbol, since_date_str):
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    since_ms = exchange.parse8601(since_date_str + 'T00:00:00Z')
    all_ohlcv = []
    
    try:
        exchange.load_markets()
        if symbol not in exchange.symbols:
            print(f"Symbol {symbol} not found on Binance.")
            return None
    except Exception as e:
        print(f"Error loading markets: {e}")
        return None
    
    print(f"Fetching {symbol} OHLCV data from {since_date_str}...")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since_ms, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since_ms = ohlcv[-1][0] + (24 * 60 * 60 * 1000)
            
            if since_ms > exchange.milliseconds():
                break
            
            print(f"Fetched {len(all_ohlcv)} entries, continuing...")
            
        except ccxt.DDoSProtection as e:
            print(f"DDoS Protection: {e}")
            exchange.sleep(exchange.rateLimit / 1000)
        except ccxt.RequestTimeout as e:
            print(f"Request Timeout: {e}")
            exchange.sleep(exchange.timeout / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    if not all_ohlcv:
        print("No OHLCV data fetched.")
        return None
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Successfully fetched {len(df)} daily candles for {symbol}.")
    return df

# Calculate inefficiency index
def calculate_inefficiency_index(df, window_days):
    if df is None or len(df) < window_days:
        return pd.Series([], dtype=float)
    
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    rolling_sum_abs_log_returns = log_returns.rolling(window=window_days).apply(
        lambda x: np.sum(np.abs(x)) if not x.isnull().all() else np.nan, raw=True
    )
    rolling_sum_log_returns = log_returns.rolling(window=window_days).sum()
    
    inefficiency_index = rolling_sum_abs_log_returns / np.abs(rolling_sum_log_returns)
    inefficiency_index = inefficiency_index.replace([np.inf, -np.inf], np.nan)
    inefficiency_index[np.abs(rolling_sum_log_returns) < 1e-9] = np.nan
    
    return inefficiency_index

# Web server routes
@app.route('/')
def index():
    df = fetch_ohlcv(SYMBOL, START_DATE_STR)
    
    if df is None or df.empty:
        return render_template('index.html', 
                               error_message="Could not fetch data. Please check logs.")
    
    inefficiency_series = calculate_inefficiency_index(df, ROLLING_WINDOW_DAYS)
    
    # Create price chart
    fig_price = px.line(df, x=df.index, y='close', 
                        title=f'{SYMBOL} Price (Daily Close)',
                        labels={'close': 'Price (USDT)', 'timestamp': 'Date'})
    fig_price.update_layout(hovermode="x unified", template="plotly_dark")
    
    # Create inefficiency index chart
    fig_inefficiency = px.line(x=inefficiency_series.index, y=inefficiency_series,
                               title=f'{SYMBOL} Inefficiency Index ({ROLLING_WINDOW_DAYS}-day Rolling)',
                               labels={'y': 'Inefficiency Index', 'x': 'Date'})
    fig_inefficiency.update_layout(hovermode="x unified", template="plotly_dark")
    
    # Convert to JSON for template
    graphJSON_price = json.dumps(fig_price, cls=go.Figure.json_encoder)
    graphJSON_inefficiency = json.dumps(fig_inefficiency, cls=go.Figure.json_encoder)
    
    return render_template('index.html',
                           graphJSON_price=graphJSON_price,
                           graphJSON_inefficiency=graphJSON_inefficiency,
                           symbol=SYMBOL,
                           window=ROLLING_WINDOW_DAYS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)