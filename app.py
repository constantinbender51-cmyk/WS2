import ccxt
import pandas as pd
import numpy as np
import time
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuration ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
START_DATE = '2020-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00'
ROUNDING_MULTIPLIER = 10 
LOOKBACK_CANDLES = 5      

# --- Storage Configuration ---
DATA_DIR = '/app/data/'
FILE_NAME = 'ethusdt_15m_2020_2026.csv'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

def slow_print(text, delay=0.1):
    print(text)
    time.sleep(delay)

def get_tick_size(symbol):
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Symbol {symbol} not found on Binance.")
    return markets[symbol]['precision']['price']

def get_ohlcv_data(symbol, timeframe, start_str, end_str):
    if os.path.exists(FILE_PATH):
        slow_print(f"[CACHE] Found data at {FILE_PATH}. Loading...")
        df = pd.read_csv(FILE_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    slow_print(f"[API] File not found. Connecting to Binance...")
    exchange = ccxt.binance()
    start_ts = exchange.parse8601(start_str)
    end_ts = exchange.parse8601(end_str)
    
    data = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=1000)
            if not ohlcv:
                break
            current_ts = ohlcv[-1][0] + 1
            data += ohlcv
            time.sleep(exchange.rateLimit / 1000)
            if current_ts >= exchange.milliseconds():
                break
        except Exception as e:
            slow_print(f"[ERROR] {str(e)}")
            break

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    mask = (df.index >= start_str) & (df.index <= end_str)
    df = df.loc[mask]
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    df.to_csv(FILE_PATH)
    slow_print(f"[IO] Data saved to {FILE_PATH}.")
    return df

def prepare_data(df, tick_size, x, b):
    slow_print("[PROCESSING] Rounding prices...")
    step_size = x * tick_size
    df['close_rounded'] = np.round(df['close'] / step_size) * step_size
    df['price_step_index'] = np.round(df['close'] / step_size)

    slow_print("[PROCESSING] Computing Log Returns...")
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    slow_print("[PROCESSING] Creating targets and features...")
    df['target_derivative'] = df['price_step_index'].shift(-1) - df['price_step_index']
    
    feature_cols = []
    for i in range(1, b + 1):
        col_name = f'lag_{i}'
        df[col_name] = df['log_returns'].shift(i)
        feature_cols.append(col_name)
    
    df.dropna(inplace=True)
    return df, feature_cols

def calculate_custom_accuracy(y_true, y_pred_raw):
    """
    Computes custom accuracy metrics based on user requirements.
    y_true: Actual step changes (integers)
    y_pred_raw: Raw model predictions (floats)
    """
    # Round predictions to nearest integer (step)
    y_pred = np.round(y_pred_raw)
    
    # Create DataFrame for easier filtering
    results = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    
    # 1. Total Directional Predictions (Model predicted != 0)
    directional_preds = results[results['pred'] != 0]
    total_directional = len(directional_preds)
    
    if total_directional == 0:
        return 0.0, 0.0

    # 2. "Flat Outcome" Metric
    # Count where model predicted move, but actual was 0
    flat_outcomes = directional_preds[directional_preds['actual'] == 0]
    flat_outcome_ratio = len(flat_outcomes) / total_directional

    # 3. Accuracy Metric
    # "If non flat prediction results in flat do not count it"
    # Filter: Pred != 0 AND Actual != 0
    valid_scoring_moves = directional_preds[directional_preds['actual'] != 0]
    
    if len(valid_scoring_moves) == 0:
        accuracy = 0.0
    else:
        # Check if signs match (Direction correct)
        # We use np.sign so +2 and +1 are considered "same direction"
        correct_direction = np.sign(valid_scoring_moves['pred']) == np.sign(valid_scoring_moves['actual'])
        accuracy = correct_direction.sum() / len(valid_scoring_moves)

    return accuracy, flat_outcome_ratio

def main():
    slow_print("--- INITIALIZING SCRIPT ---")
    
    try:
        tick_size = get_tick_size(SYMBOL)
        slow_print(f"[INFO] Tick Size: {tick_size}")
    except Exception as e:
        slow_print(f"[CRITICAL] {e}")
        return

    try:
        df = get_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    except Exception as e:
        slow_print(f"[CRITICAL] {e}")
        return

    df_processed, feature_cols = prepare_data(df, tick_size, ROUNDING_MULTIPLIER, LOOKBACK_CANDLES)
    
    X = df_processed[feature_cols]
    y = df_processed['target_derivative']
    
    n = len(df_processed)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    slow_print(f"[STATS] Train: {len(X_train)}, Test: {len(X_test)}")

    # --- Linear Regression ---
    slow_print("--- TRAINING LINEAR REGRESSION ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    lr_acc, lr_flat_ratio = calculate_custom_accuracy(y_test, lr_pred)
    
    # --- Random Forest ---
    slow_print("--- TRAINING RANDOM FOREST ---")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_acc, rf_flat_ratio = calculate_custom_accuracy(y_test, rf_pred)

    # --- REPORTING ---
    slow_print("--- FINAL PERFORMANCE METRICS ---")
    
    slow_print(f"[LINEAR REGRESSION]")
    slow_print(f"MSE: {mean_squared_error(y_test, lr_pred):.4f}")
    slow_print(f"Non-Flat Accuracy: {lr_acc:.2%}")
    slow_print(f"Flat Outcome Ratio:  {lr_flat_ratio:.2%} (Ghost Moves)")
    
    slow_print(f"-----------------------------")
    
    slow_print(f"[RANDOM FOREST]")
    slow_print(f"MSE: {mean_squared_error(y_test, rf_pred):.4f}")
    slow_print(f"Non-Flat Accuracy: {rf_acc:.2%}")
    slow_print(f"Flat Outcome Ratio:  {rf_flat_ratio:.2%} (Ghost Moves)")
    
    slow_print("--- ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    main()
