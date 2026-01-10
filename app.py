import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuration ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
START_DATE = '2020-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00'
ROUNDING_MULTIPLIER = 10  # 'x'
LOOKBACK_CANDLES = 5      # 'b'

# --- Storage Configuration ---
DATA_DIR = '/app/data/'
FILE_NAME = 'ethusdt_15m_2020_2026.csv'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

def slow_print(text, delay=0.1):
    """Prints text with a slight delay."""
    print(text)
    time.sleep(delay)

def get_tick_size(symbol):
    """Fetches tick size from Binance. Always requires a lightweight API call."""
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Symbol {symbol} not found on Binance.")
    return markets[symbol]['precision']['price']

def get_ohlcv_data(symbol, timeframe, start_str, end_str):
    """
    Checks for local file. If present, loads it. 
    If not, fetches from Binance and saves it.
    """
    # 1. Check if file exists
    if os.path.exists(FILE_PATH):
        slow_print(f"[CACHE] Found data at {FILE_PATH}. Loading...")
        df = pd.read_csv(FILE_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        slow_print(f"[CACHE] Loaded {len(df)} candles from file.")
        return df

    # 2. If not, Fetch from Binance
    slow_print(f"[API] File not found. Connecting to Binance...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    data = []
    current_ts = start_ts
    
    slow_print(f"[API] Starting download for {symbol}. This may take a while...")
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            
            current_ts = ohlcv[-1][0] + 1
            data += ohlcv
            time.sleep(exchange.rateLimit / 1000) # Respect rate limits
            
            if current_ts >= exchange.milliseconds():
                break
                
        except Exception as e:
            slow_print(f"[ERROR] {str(e)}")
            break

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Filter strictly within requested range
    mask = (df.index >= start_str) & (df.index <= end_str)
    df = df.loc[mask]
    
    # 3. Save to Disk
    slow_print(f"[IO] Saving data to {FILE_PATH}...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR) # Create directory if it doesn't exist
    
    df.to_csv(FILE_PATH)
    slow_print(f"[IO] Data saved successfully.")
    
    return df

def prepare_data(df, tick_size, x, b):
    """Preprocesses data: Rounding, Logs, Feature Engineering."""
    
    slow_print("[PROCESSING] Rounding prices...")
    step_size = x * tick_size
    df['close_rounded'] = np.round(df['close'] / step_size) * step_size
    
    # Calculate the integer "step" value
    df['price_step_index'] = np.round(df['close'] / step_size)

    slow_print("[PROCESSING] Computing Log Returns...")
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    slow_print("[PROCESSING] Creating targets and features...")
    # Target: Next candle derivative (change in steps)
    df['target_derivative'] = df['price_step_index'].shift(-1) - df['price_step_index']
    
    # Features: Lagged log returns
    feature_cols = []
    for i in range(1, b + 1):
        col_name = f'lag_{i}'
        df[col_name] = df['log_returns'].shift(i)
        feature_cols.append(col_name)
    
    df.dropna(inplace=True)
    return df, feature_cols

def main():
    slow_print("--- INITIALIZING SCRIPT ---")
    
    # Get Tick Size (Always need this metadata)
    try:
        tick_size = get_tick_size(SYMBOL)
        slow_print(f"[INFO] Tick Size for {SYMBOL}: {tick_size}")
    except Exception as e:
        slow_print(f"[CRITICAL ERROR] Could not fetch tick size: {e}")
        return

    # Get Data (Load or Fetch)
    try:
        df = get_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    except Exception as e:
        slow_print(f"[CRITICAL ERROR] Could not get data: {e}")
        return

    # Prepare Data
    df_processed, feature_cols = prepare_data(df, tick_size, ROUNDING_MULTIPLIER, LOOKBACK_CANDLES)
    
    X = df_processed[feature_cols]
    y = df_processed['target_derivative']
    
    # Split Data 60/20/20
    slow_print("[SPLIT] Splitting data 60/20/20...")
    n = len(df_processed)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    slow_print(f"[STATS] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train Linear Regression
    slow_print("--- TRAINING LINEAR REGRESSION ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    slow_print("[INFO] Linear Regression Training Complete.")

    # Train Random Forest
    slow_print("--- TRAINING RANDOM FOREST ---")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    slow_print("[INFO] Random Forest Training Complete.")

    # Analyze Performance
    slow_print("--- PERFORMANCE METRICS (TEST SET) ---")
    
    # LR Metrics
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    slow_print(f"Linear Regression MSE: {lr_mse:.5f}")
    slow_print(f"Linear Regression MAE: {lr_mae:.5f}")
    slow_print(f"Linear Regression R2 : {lr_r2:.5f}")
    
    # RF Metrics
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    slow_print(f"Random Forest MSE    : {rf_mse:.5f}")
    slow_print(f"Random Forest MAE    : {rf_mae:.5f}")
    slow_print(f"Random Forest R2     : {rf_r2:.5f}")
    
    slow_print("--- ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    main()
