import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuration ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
START_DATE = '2020-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00' # Future date; API will fetch up to "now"
ROUNDING_MULTIPLIER = 10  # 'x' in your prompt
LOOKBACK_CANDLES = 5      # 'b' candles for features

def slow_print(text, delay=0.1):
    """Prints text with a slight delay to satisfy output requirements."""
    print(text)
    time.sleep(delay)

def fetch_binance_data(symbol, timeframe, start_str, end_str):
    """Fetches historical OHLCV data from Binance using CCXT with pagination."""
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    # Get Tick Size
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Symbol {symbol} not found on Binance.")
    
    tick_size = markets[symbol]['precision']['price']
    slow_print(f"[INFO] Fetched Tick Size for {symbol}: {tick_size}")
    
    data = []
    current_ts = start_ts
    
    slow_print(f"[INFO] Starting data download for {symbol}. This may take a moment...")
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            
            current_ts = ohlcv[-1][0] + 1 # Move to next timestamp
            data += ohlcv
            
            # Rate limit protection
            time.sleep(exchange.rateLimit / 1000)
            
            # Break if we've reached current time (for future dates)
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
    
    slow_print(f"[INFO] Total candles fetched: {len(df)}")
    return df, tick_size

def prepare_data(df, tick_size, x, b):
    """Preprocesses data: Rounding, Logs, Feature Engineering."""
    
    slow_print("[PROCESSING] Rounding prices...")
    # 4. Round price to x * tick size
    step_size = x * tick_size
    df['close_rounded'] = np.round(df['close'] / step_size) * step_size
    
    # Calculate the integer "step" value (for derivative calculation)
    df['price_step_index'] = np.round(df['close'] / step_size)

    slow_print("[PROCESSING] Computing Log Returns...")
    # 5 & 6. Computes returns -> Logarithm
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    slow_print("[PROCESSING] creating targets and features...")
    # 7. Target: Next candle derivative (change in steps)
    # Target = Step(t+1) - Step(t)
    df['target_derivative'] = df['price_step_index'].shift(-1) - df['price_step_index']
    
    # Create Features: Lagged log returns for 'b' candles
    feature_cols = []
    for i in range(1, b + 1):
        col_name = f'lag_{i}'
        df[col_name] = df['log_returns'].shift(i)
        feature_cols.append(col_name)
    
    df.dropna(inplace=True)
    return df, feature_cols

def main():
    slow_print("--- INITIALIZING SCRIPT ---")
    
    # 1 & 3. Fetch Data & Tick Size
    try:
        df, tick_size = fetch_binance_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    except Exception as e:
        slow_print(f"[CRITICAL ERROR] Could not fetch data: {e}")
        return

    # 4, 5, 6, 7. Prepare Data
    df_processed, feature_cols = prepare_data(df, tick_size, ROUNDING_MULTIPLIER, LOOKBACK_CANDLES)
    
    X = df_processed[feature_cols]
    y = df_processed['target_derivative']
    
    # 2. Split Data 60/20/20
    slow_print("[SPLIT] Splitting data 60/20/20...")
    n = len(df_processed)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    slow_print(f"[STATS] Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # 7. Train Linear Regression
    slow_print("--- TRAINING LINEAR REGRESSION ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    slow_print("[INFO] Linear Regression Training Complete.")

    # 8. Train Random Forest
    slow_print("--- TRAINING RANDOM FOREST ---")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    slow_print("[INFO] Random Forest Training Complete.")

    # 9. Analyze Performance Metrics
    slow_print("--- PERFORMANCE METRICS (TEST SET) ---")
    
    # Linear Regression Metrics
    lr_mse = mean_squared_error(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    
    slow_print(f"Linear Regression MSE: {lr_mse:.5f}")
    slow_print(f"Linear Regression MAE: {lr_mae:.5f}")
    slow_print(f"Linear Regression R2 : {lr_r2:.5f}")
    
    # Random Forest Metrics
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    slow_print(f"Random Forest MSE    : {rf_mse:.5f}")
    slow_print(f"Random Forest MAE    : {rf_mae:.5f}")
    slow_print(f"Random Forest R2     : {rf_r2:.5f}")
    
    slow_print("--- ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    main()
