import ccxt
import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =========================================
# 1. CONFIGURATION
# =========================================

# --- Exchange & Data Settings ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
START_DATE = '2020-01-01 00:00:00'
END_DATE = '2026-01-01 00:00:00'

# --- Grid Search Ranges ---
X_START = 50
X_END = 400
X_STEP = 50

B_START = 2
B_END = 10

# --- Data Storage ---
DATA_DIR = '/app/data/'
FILE_NAME = 'ethusdt_15m_2020_2026.csv'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# --- Model Settings ---
RF_ESTIMATORS = 30 
RF_MAX_DEPTH = 8
RANDOM_STATE = 42

# --- Output Settings ---
PRINT_DELAY = 0.01

# =========================================
# 2. HELPER FUNCTIONS
# =========================================

def slow_print(text, delay=PRINT_DELAY):
    print(text)
    sys.stdout.flush()
    time.sleep(delay)

def get_tick_size(symbol):
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Symbol {symbol} not found on Binance.")
    return markets[symbol]['precision']['price']

def get_ohlcv_data(symbol, timeframe, start_str, end_str):
    if os.path.exists(FILE_PATH):
        slow_print(f"[CACHE] Loading data from {FILE_PATH}...")
        df = pd.read_csv(FILE_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

    slow_print(f"[API] Fetching new data from Binance...")
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
    return df

def prepare_data(df_input, tick_size, x, b):
    df = df_input.copy()
    step_size = x * tick_size
    
    # 1. Round Close price
    df['close_rounded'] = np.round(df['close'] / step_size) * step_size
    df['price_step_index'] = np.round(df['close'] / step_size)

    # 2. Log Returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # 3. Target: Derivative of steps
    df['target_derivative'] = df['price_step_index'].shift(-1) - df['price_step_index']
    
    # 4. Features
    feature_cols = []
    for i in range(1, b + 1):
        col_name = f'lag_{i}'
        df[col_name] = df['log_returns'].shift(i)
        feature_cols.append(col_name)
    
    df.dropna(inplace=True)
    return df, feature_cols

def calculate_accuracy_only(y_true, y_pred_raw):
    """
    Calculates pure directional accuracy for non-flat predictions.
    """
    # Round predictions
    y_pred = np.round(y_pred_raw)
    
    results = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    
    # Filter for directional predictions (Model != 0)
    directional_preds = results[results['pred'] != 0]
    total_trades = len(directional_preds)
    
    if total_trades == 0:
        return 0.0, 0

    # Accuracy: (Model != 0) AND (Model Sign == Actual Sign)
    # We filter out cases where actual was 0 (ghost moves) from the *numerator* of accuracy usually,
    # but strictly speaking "Accuracy" is Correct / Total.
    # If model says +1 and market is 0, that is WRONG (False Positive).
    
    # Check strict correctness:
    # If pred > 0, actual must be > 0. If pred < 0, actual must be < 0.
    correct_direction = np.sign(directional_preds['pred']) == np.sign(directional_preds['actual'])
    
    # Note: np.sign(0) is 0. So if actual is 0, sign won't match pred (+1/-1), which is correct (it's a miss).
    accuracy = correct_direction.sum() / total_trades

    return accuracy, total_trades

# =========================================
# 3. MAIN EXECUTION
# =========================================

def main():
    slow_print("--- STARTING GRID SEARCH (ACCURACY ONLY) ---")
    
    try:
        tick_size = get_tick_size(SYMBOL)
    except Exception as e:
        slow_print(f"[CRITICAL] {e}")
        return

    try:
        df_raw = get_ohlcv_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    except Exception as e:
        slow_print(f"[CRITICAL] {e}")
        return

    # Range is inclusive of start, exclusive of end in Python
    x_values = list(range(X_START, X_END + 1, X_STEP))
    b_values = list(range(B_START, B_END + 1))
    
    total_combinations = len(x_values) * len(b_values)
    slow_print(f"[INFO] Grid: X({len(x_values)}) * B({len(b_values)}) = {total_combinations} combinations.")
    slow_print(f"[INFO] Optimizing: VALIDATION ACCURACY")
    slow_print("-" * 40)

    best_acc = -1.0
    best_params = {}
    best_trades = 0
    
    counter = 0
    
    for x in x_values:
        for b in b_values:
            counter += 1
            
            # Prepare Data
            df_processed, feature_cols = prepare_data(df_raw, tick_size, x, b)
            
            X_feat = df_processed[feature_cols]
            y = df_processed['target_derivative']
            
            # Split (60/20 Train/Val)
            n = len(df_processed)
            train_end = int(n * 0.60)
            val_end = int(n * 0.80)
            
            X_train = X_feat.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_val = X_feat.iloc[train_end:val_end]
            y_val = y.iloc[train_end:val_end]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, 
                                       max_depth=RF_MAX_DEPTH, 
                                       n_jobs=-1, 
                                       random_state=RANDOM_STATE)
            rf.fit(X_train, y_train)
            
            # Validate
            preds = rf.predict(X_val)
            acc, trades = calculate_accuracy_only(y_val, preds)
            
            # Output
            if counter % 10 == 0 or acc > best_acc:
                print(f"[{counter}/{total_combinations}] X={x}, B={b} | Val Acc: {acc:.2%} ({trades} trades)")

            if acc > best_acc:
                best_acc = acc
                best_params = {'x': x, 'b': b}
                best_trades = trades

    slow_print("-" * 40)
    slow_print("--- GRID SEARCH COMPLETE ---")
    slow_print(f"BEST VALIDATION PARAMETERS:")
    slow_print(f"X (Multiplier) : {best_params['x']}")
    slow_print(f"B (Candles)    : {best_params['b']}")
    slow_print(f"Validation Acc : {best_acc:.2%}")
    slow_print(f"Total Trades   : {best_trades}")
    slow_print("-" * 40)
    slow_print("[INFO] Test set results hidden as requested.")

if __name__ == "__main__":
    main()
