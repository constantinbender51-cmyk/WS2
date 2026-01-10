import ccxt
import pandas as pd
import numpy as np
import time
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =========================================
# 1. GLOBAL CONFIGURATION
# =========================================

# --- Exchange & Data Settings ---
SYMBOL = 'ETH/USDT'            # Pair to fetch
TIMEFRAME = '15m'              # Candle timeframe
START_DATE = '2020-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00'

# --- Feature Engineering Parameters ---
ROUNDING_MULTIPLIER = 1000       # 'x': Used to calculate step size (x * tick_size)
LOOKBACK_CANDLES = 5           # 'b': Number of previous candles to use as features

# --- Data Storage ---
DATA_DIR = '/app/data/'
FILE_NAME = 'ethusdt_15m_2020_2026.csv'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# --- Model Hyperparameters ---
RF_ESTIMATORS = 50             # Random Forest number of trees
RF_MAX_DEPTH = 10          # Random Forest max depth
RANDOM_STATE = 42              # Seed for reproducibility

# --- Output Settings ---
PRINT_DELAY = 0.1              # Delay in seconds for text output

# =========================================
# 2. UTILITY FUNCTIONS
# =========================================

def slow_print(text, delay=PRINT_DELAY):
    print(text)
    time.sleep(delay)

def get_tick_size(symbol):
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Symbol {symbol} not found on Binance.")
    return markets[symbol]['precision']['price']

def get_ohlcv_data(symbol, timeframe, start_str, end_str):
    # Check cache
    if os.path.exists(FILE_PATH):
        slow_print(f"[CACHE] Found data at {FILE_PATH}. Loading...")
        df = pd.read_csv(FILE_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    # Fetch from API
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
    
    # Calculate the integer "step" index (e.g., price 2000 with step 10 -> index 200)
    df['price_step_index'] = np.round(df['close'] / step_size)

    slow_print("[PROCESSING] Computing Log Returns...")
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    slow_print("[PROCESSING] Creating targets and features...")
    # Target: The change in steps (derivative)
    # e.g., Index 200 -> Index 202 = Target +2
    df['target_derivative'] = df['price_step_index'].shift(-1) - df['price_step_index']
    
    feature_cols = []
    for i in range(1, b + 1):
        col_name = f'lag_{i}'
        df[col_name] = df['log_returns'].shift(i)
        feature_cols.append(col_name)
    
    df.dropna(inplace=True)
    return df, feature_cols

def calculate_metrics(y_true, y_pred_raw):
    """
    y_true: Actual integer step changes
    y_pred_raw: Raw float predictions from model
    """
    # 1. Round predictions to nearest integer
    y_pred = np.round(y_pred_raw)
    
    results = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    
    # 2. Filter: Only look at where Model Predicted a Move (Non-Flat)
    directional_preds = results[results['pred'] != 0]
    total_predictions = len(directional_preds)
    
    if total_predictions == 0:
        return 0.0, 0.0

    # 3. Metric: Flat Outcome Ratio (Ghost Moves)
    # Model said move, Market stayed flat (Actual == 0)
    flat_outcomes = directional_preds[directional_preds['actual'] == 0]
    flat_outcome_ratio = len(flat_outcomes) / total_predictions

    # 4. Metric: Non-Flat Accuracy
    # Filter: Model Predicted Move AND Market Moved (Actual != 0)
    valid_scoring_moves = directional_preds[directional_preds['actual'] != 0]
    
    if len(valid_scoring_moves) == 0:
        accuracy = 0.0
    else:
        # Check if signs match (Direction correct)
        correct_direction = np.sign(valid_scoring_moves['pred']) == np.sign(valid_scoring_moves['actual'])
        accuracy = correct_direction.sum() / len(valid_scoring_moves)

    return accuracy, flat_outcome_ratio

# =========================================
# 3. MAIN EXECUTION
# =========================================

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
    y = df_processed['target_derivative'] # Integers (-1, 0, 1, 2...)
    
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
    
    lr_acc, lr_flat_ratio = calculate_metrics(y_test, lr_pred)
    
    # --- Random Forest ---
    slow_print("--- TRAINING RANDOM FOREST ---")
    rf_model = RandomForestRegressor(n_estimators=RF_ESTIMATORS, 
                                     max_depth=RF_MAX_DEPTH, 
                                     n_jobs=-1, 
                                     random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_acc, rf_flat_ratio = calculate_metrics(y_test, rf_pred)

    # --- REPORTING ---
    slow_print("--- FINAL PERFORMANCE METRICS ---")
    slow_print("Metric Definitions:")
    slow_print("1. Non-Flat Accuracy: When model predicts a move AND market moves, did we get the direction right?")
    slow_print("2. Flat Outcome Ratio: When model predicts a move, how often did the market actually stay flat?")
    slow_print("-" * 30)

    slow_print(f"[LINEAR REGRESSION]")
    slow_print(f"Non-Flat Accuracy:  {lr_acc:.2%}")
    slow_print(f"Flat Outcome Ratio:   {lr_flat_ratio:.2%}")
    
    slow_print(f"-" * 15)
    
    slow_print(f"[RANDOM FOREST]")
    slow_print(f"Non-Flat Accuracy:  {rf_acc:.2%}")
    slow_print(f"Flat Outcome Ratio:   {rf_flat_ratio:.2%}")
    
    slow_print("--- ANALYSIS COMPLETE ---")

if __name__ == "__main__":
    main()
