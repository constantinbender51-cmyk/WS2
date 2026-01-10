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
X_START = 200
X_END = 2000
X_STEP = 100

B_START = 2
B_END = 10

# --- Data Storage ---
DATA_DIR = '/app/data/'
FILE_NAME = 'ethusdt_15m_2020_2026.csv'
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

# --- Model Settings for Grid Search ---
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

def calculate_metrics_and_score(y_true, y_pred_raw):
    # Round predictions
    y_pred = np.round(y_pred_raw)
    
    results = pd.DataFrame({'actual': y_true, 'pred': y_pred})
    
    # Filter for directional predictions
    directional_preds = results[results['pred'] != 0]
    total_trades = len(directional_preds)
    
    if total_trades == 0:
        return 0.0, 0.0, 0, 0.0

    # 1. Flat Outcome Ratio
    flat_outcomes = directional_preds[directional_preds['actual'] == 0]
    flat_ratio = len(flat_outcomes) / total_trades

    # 2. Non-Flat Accuracy
    valid_scoring_moves = directional_preds[directional_preds['actual'] != 0]
    if len(valid_scoring_moves) == 0:
        accuracy = 0.0
    else:
        correct_direction = np.sign(valid_scoring_moves['pred']) == np.sign(valid_scoring_moves['actual'])
        accuracy = correct_direction.sum() / len(valid_scoring_moves)

    # 3. NEW CUSTOM SCORE
    # Score = (Accuracy - 0.5) * Trades * (1 - Flat_Ratio)
    # This sets 50% accuracy as the baseline (score 0). 
    # Below 50% becomes negative score.
    score = (accuracy - 0.50) * total_trades * (1 - flat_ratio)

    return accuracy, flat_ratio, total_trades, score

# =========================================
# 3. MAIN EXECUTION
# =========================================

def main():
    slow_print("--- STARTING GRID SEARCH ---")
    
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

    x_values = list(range(X_START, X_END + 1, X_STEP))
    b_values = list(range(B_START, B_END + 1))
    
    total_combinations = len(x_values) * len(b_values)
    slow_print(f"[INFO] Grid: X({len(x_values)}) * B({len(b_values)}) = {total_combinations} combinations.")
    slow_print(f"[INFO] Optimizing Score = (Accuracy - 0.5) * Trades * (1 - Flat_Ratio)")
    slow_print("-" * 40)

    best_score = -float('inf') # Initialize to negative infinity since scores can be negative
    best_params = {}
    best_metrics = {}
    
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
            
            # Train
            rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, 
                                       max_depth=RF_MAX_DEPTH, 
                                       n_jobs=-1, 
                                       random_state=RANDOM_STATE)
            rf.fit(X_train, y_train)
            
            # Validate
            preds = rf.predict(X_val)
            acc, flat_ratio, trades, score = calculate_metrics_and_score(y_val, preds)
            
            # Output
            if counter % 10 == 0 or score > best_score:
                print(f"[{counter}/{total_combinations}] X={x}, B={b} | Score: {score:.2f} | Acc: {acc:.1%} | Trades: {trades} | Flat: {flat_ratio:.1%}")

            if score > best_score:
                best_score = score
                best_params = {'x': x, 'b': b}
                best_metrics = {'acc': acc, 'trades': trades, 'flat': flat_ratio}

    slow_print("-" * 40)
    slow_print("--- GRID SEARCH COMPLETE ---")
    slow_print(f"BEST PARAMETERS FOUND:")
    slow_print(f"X (Multiplier) : {best_params['x']}")
    slow_print(f"B (Candles)    : {best_params['b']}")
    slow_print(f"Validation Score : {best_score:.4f}")
    slow_print(f" > Accuracy      : {best_metrics['acc']:.2%}")
    slow_print(f" > Trades        : {best_metrics['trades']}")
    slow_print(f" > Flat Ratio    : {best_metrics['flat']:.2%}")
    
    slow_print("-" * 40)
    slow_print("--- RUNNING FINAL TEST ON BEST PARAMS ---")
    
    x_best = best_params['x']
    b_best = best_params['b']
    
    df_final, feat_cols_final = prepare_data(df_raw, tick_size, x_best, b_best)
    
    X_f = df_final[feat_cols_final]
    y_f = df_final['target_derivative']
    
    n = len(df_final)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    X_train_final = X_f.iloc[:train_end]
    y_train_final = y_f.iloc[:train_end]
    X_test_final = X_f.iloc[val_end:] 
    y_test_final = y_f.iloc[val_end:]
    
    rf_final = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_final.fit(X_train_final, y_train_final)
    
    test_preds = rf_final.predict(X_test_final)
    t_acc, t_flat, t_trades, t_score = calculate_metrics_and_score(y_test_final, test_preds)
    
    slow_print(f"[TEST SET RESULTS]")
    slow_print(f"Non-Flat Accuracy : {t_acc:.2%}")
    slow_print(f"Total Trades      : {t_trades}")
    slow_print(f"Flat Outcome Ratio: {t_flat:.2%}")
    slow_print(f"Final Score       : {t_score:.4f}")
    slow_print("--- DONE ---")

if __name__ == "__main__":
    main()
